"""LangGraph orchestration (hub-and-spoke: the graph is the orchestrator, agents are spokes).

  ingest -> diagnostics -> [act?] -> gate -> context -> decision -> validation -+-> finalize
                                                          ^                      |
                                                          +---- retry (<= max) --+  (Red only)

Routing, retries, gates and the dry-run/audit trail live here - not inside the agents."""
from __future__ import annotations

import time

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from .agents.context import ContextAgent
from .agents.decision import DecisionAgent
from .agents.diagnostics import DiagnosticsAgent
from .agents.validation import ValidationAgent
from .config import Settings
from .datastore import DataStore
from .guardrails import sanitize_text
from .llm import LLMClient
from .observability import make_log
from .rag import VectorStore
from .rules import customer_gate
from .schemas import (ContextBundle, CustomerRecord, DecisionOutput, DiagnosticsOutput, GraphState, Recommendation,
                      RTSSignal, ValidationReport)

OFFER_LEVEL_RULES = {"R-08", "R-09", "R-10", "R-11", "R-12", "R-13", "R-14"}


class Pipeline:
    def __init__(self, settings: Settings, ds: DataStore, store: VectorStore, llm: LLMClient, checkpointer=None):
        self.s, self.ds = settings, ds
        self.diag = DiagnosticsAgent(settings, llm)
        self.ctx = ContextAgent(settings, ds, store)
        self.dec = DecisionAgent(settings, ds, llm)
        self.val = ValidationAgent(settings, ds, llm)
        self.graph = self._build(checkpointer or MemorySaver())

    # ------------------------------------------------------------------ public
    def run_one(self, customer: dict, signal: dict | None, thread_id: str) -> GraphState:
        return self.graph.invoke({"customer": customer, "signal": signal},
                                 config={"configurable": {"thread_id": thread_id}, "recursion_limit": 40})

    # ------------------------------------------------------------------ nodes
    def _build(self, checkpointer):
        s = self.s

        def ingest(state: GraphState):
            t0 = time.perf_counter()
            flags: list[str] = []
            try:
                cust = CustomerRecord(**state["customer"]).model_dump()
            except ValidationError as e:
                log = make_log("guardrails", (time.perf_counter() - t0) * 1000, status="blocked", flags=["invalid_customer_record"], note=str(e)[:120])
                return {"blocked": True, "block_reasons": ["INVALID_INPUT: customer record failed schema validation"],
                        "flags": ["invalid_customer_record"], "audit_log": [log]}
            sig = None
            if state.get("signal"):
                try:
                    sig = RTSSignal(**state["signal"]).model_dump()
                    sig["call_snippet"], f = sanitize_text(sig["call_snippet"])  # untrusted free text
                    flags += f
                except ValidationError:
                    flags.append("invalid_signal_ignored")
            log = make_log("guardrails", (time.perf_counter() - t0) * 1000, flags=flags, note="input validated + sanitised")
            return {"customer": cust, "signal": sig, "flags": flags, "blocked": False, "retries": 0, "retry": False,
                    "excluded_offers": [], "feedback": "", "audit_log": [log]}

        def diagnostics(state: GraphState):
            out, log = self.diag.run(state["customer"], state.get("signal"))
            return {"diagnostics": out.model_dump(), "audit_log": [log], "flags": log["flags"]}

        def gate(state: GraphState):
            t0 = time.perf_counter()
            v = customer_gate(state["customer"], s)
            log = make_log("guardrails", (time.perf_counter() - t0) * 1000, status="blocked" if v else "ok",
                           flags=[x.rule_id for x in v], note="eligibility gate")
            if v:
                return {"blocked": True, "block_reasons": [f"{x.rule_id}: {x.message}" for x in v], "audit_log": [log]}
            return {"blocked": False, "audit_log": [log]}

        def context(state: GraphState):
            diag = DiagnosticsOutput(**state["diagnostics"])
            snippet = (state.get("signal") or {}).get("call_snippet", "")
            ctx, log = self.ctx.run(state["customer"], diag, snippet, state.get("excluded_offers", []))
            return {"context": ctx.model_dump(), "audit_log": [log], "flags": log["flags"]}

        def decision(state: GraphState):
            diag = DiagnosticsOutput(**state["diagnostics"])
            ctx = ContextBundle(**state["context"])
            snippet = (state.get("signal") or {}).get("call_snippet", "")
            out, log = self.dec.run(state["customer"], diag, ctx, snippet, state.get("feedback", ""),
                                    state.get("excluded_offers", []), state.get("retries", 0))
            return {"decision": out.model_dump(), "audit_log": [log], "flags": log["flags"]}

        def validation(state: GraphState):
            diag = DiagnosticsOutput(**state["diagnostics"])
            ctx = ContextBundle(**state["context"])
            dec = DecisionOutput(**state["decision"])
            snippet = (state.get("signal") or {}).get("call_snippet", "")
            r = state.get("retries", 0)
            rep, log = self.val.run(state["customer"], diag, ctx, dec, snippet, r)
            log["rule_pass"], log["badge"] = rep.rule_pass, rep.badge
            update = {"validation": rep.model_dump(), "audit_log": [log], "retry": False}
            if rep.badge == "red" and r < s.max_retries:
                excl = list(state.get("excluded_offers", []))
                if dec.offer_id and (any(v.rule_id in OFFER_LEVEL_RULES for v in rep.violations) or rep.fit < 3):
                    excl.append(dec.offer_id)
                update.update(retry=True, retries=r + 1, feedback=rep.feedback, excluded_offers=excl)
            return update

        def finalize(state: GraphState):
            return {"recommendation": self._finalize(state).model_dump()}

        g = StateGraph(GraphState)
        for name, fn in [("ingest", ingest), ("diagnostics", diagnostics), ("gate", gate), ("context", context),
                         ("decision", decision), ("validation", validation), ("finalize", finalize)]:
            g.add_node(name, fn)
        g.set_entry_point("ingest")
        g.add_conditional_edges("ingest", lambda st: "finalize" if st.get("blocked") else "diagnostics")
        g.add_conditional_edges("diagnostics", lambda st: "gate" if st["diagnostics"]["act"] else "finalize")
        g.add_conditional_edges("gate", lambda st: "finalize" if st.get("blocked") else "context")
        g.add_edge("context", "decision")
        g.add_conditional_edges("decision", lambda st: "validation" if st["decision"]["action"] == "offer" else "finalize")
        g.add_conditional_edges("validation", lambda st: "decision" if st.get("retry") else "finalize")
        g.add_edge("finalize", END)
        return g.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------ result assembly
    def _finalize(self, st: GraphState) -> Recommendation:
        s = self.s
        cid = st["customer"].get("customer_id", "UNKNOWN")
        diag = st.get("diagnostics")
        base = dict(customer_id=cid, risk_score=diag["risk_score"] if diag else 0.0, tier=diag["tier"] if diag else "low",
                    reasons=[r for r in diag["reasons"]] if diag else [], retries=st.get("retries", 0))
        if st.get("blocked"):
            return Recommendation(status="blocked", skip_reason="; ".join(st.get("block_reasons", [])), **base,
                                  scores={"flags": st.get("flags", [])})
        if diag and not diag["act"]:
            status = "monitor" if diag["tier"] == "medium" else "no_action"
            return Recommendation(status=status, skip_reason=diag.get("note", "") or f"risk {diag['risk_score']:.2f} below action threshold",
                                  **base, scores={"flags": st.get("flags", [])})
        dec, val = st.get("decision"), st.get("validation")
        if not dec or dec["action"] != "offer" or (val and val["badge"] == "red"):
            why = (val or {}).get("feedback") or (dec or {}).get("rationale", "no viable offer")
            return Recommendation(status="escalated", skip_reason=f"sent to human review: {why}"[:400], badge=(val or {}).get("badge"),
                                  overall_score=(val or {}).get("overall"), **base,
                                  scores=dict(validation=val, flags=st.get("flags", [])))
        offer = self.ds.offer(dec["offer_id"])
        needs_approval = val["requires_approval"] or val["badge"] == "amber"
        status = "pending_approval" if needs_approval else "ready"
        return Recommendation(
            status=status, offer_id=offer["offer_id"], offer_name=offer["name"], offer_cost=offer["cost_usd"],
            fallback_offer_id=dec.get("fallback_offer_id"), badge=val["badge"], overall_score=val["overall"],
            talking_points=dec["talking_points"], rationale=dec["rationale"], citations=dec["citations"],
            skip_reason=("amber: needs human review" if val["badge"] == "amber" else "") + (" cost above approval limit" if val["requires_approval"] else ""),
            scores=dict(validation=val, method=dec["method"], confidence=dec["confidence"], flags=st.get("flags", [])), **base)
