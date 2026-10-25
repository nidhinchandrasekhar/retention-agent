"""Validation agent (the critic / quality gate). It does NOT generate offers; it grades them.

Three layers, cheapest first:
  1. Rules (code)     - hard compliance; any hard failure => Red
  2. Groundedness     - citations exist in retrieved context + every number is traceable (code)
  3. Quality judge    - fit & actionability; heuristic always, LLM-as-judge (small model) when available
Badge: Green (auto-eligible) / Amber (needs human review) / Red (rejected; loop back, max retries then escalate)."""
from __future__ import annotations

import json

from ..config import Settings
from ..datastore import DataStore
from ..guardrails import extract_numbers, untraced_numbers, allowed_numbers
from ..llm import LLMClient
from ..observability import make_log
from ..rules import hard, offer_violations, output_violations
from ..schemas import ContextBundle, DecisionOutput, DiagnosticsOutput, ValidationReport
from ..utils import Timer

import re

JUDGE_SYSTEM = (
    "You grade a retention recommendation for a prepaid telecom care agent. Score 1-5 (integers). "
    "fit: how well the chosen offer addresses the customer's diagnosed reasons, cost and history. "
    "actionability: could a care agent use the talking points as-is on a call (clear, short, empathetic, has a next step). "
    'Return JSON: {"fit": int, "actionability": int, "comment": str}.'
)
FIT_MIN, ACT_MIN, GROUND_MIN = 4.0, 4.0, 0.75


class ValidationAgent:
    name = "validation"

    def __init__(self, settings: Settings, ds: DataStore, llm: LLMClient):
        self.s, self.ds, self.llm = settings, ds, llm

    def run(self, c: dict, diag: DiagnosticsOutput, ctx: ContextBundle, dec: DecisionOutput, snippet: str, attempt: int):
        with Timer() as t:
            rep, usage = self._validate(c, diag, ctx, dec, snippet)
        flags = [f"violation:{v.rule_id}" for v in rep.violations]
        return rep, make_log(self.name, t.ms, usage, flags=flags, attempt=attempt,
                             note=f"{rep.badge} overall={rep.overall:.2f} fit={rep.fit:.1f} act={rep.actionability:.1f} ground={rep.groundedness:.2f}")

    # ------------------------------------------------------------------
    def _validate(self, c, diag, ctx, dec, snippet):
        s = self.s
        offer = self.ds.offer(dec.offer_id) if dec.offer_id else None
        reasons = [r.code for r in diag.reasons]
        evidence = [r.evidence for r in diag.reasons]
        snippets = [snippet] + [x.snippet for x in ctx.similar_calls + ctx.policies + ctx.offer_citations]
        extra = [diag.risk_score] + [n for v in ctx.acceptance_counts.values() for n in v]
        refs = ctx.all_refs()
        d = dec.model_dump()

        viols = offer_violations(offer, c, reasons, s) + output_violations(d, offer, refs, c, evidence, snippets, extra, s)
        hv = hard(viols)
        requires_approval = any(v.severity == "gate" for v in viols)

        # groundedness
        cites = dec.citations
        cite_frac = (sum(1 for x in cites if x in refs) / len(cites)) if cites else 0.0
        texts = [dec.rationale] + dec.talking_points
        all_nums = set()
        for tx in texts:
            all_nums |= extract_numbers(re.sub(r"\[[a-z]+:[^\]]+\]", "", tx))
        allowed = allowed_numbers(offer, c, evidence, snippets, extra)
        bad = set(untraced_numbers(texts, allowed))
        num_frac = 1.0 if not all_nums else (len(all_nums) - len(bad)) / len(all_nums)
        ground = round(0.5 * cite_frac + 0.5 * num_frac, 3)

        # quality
        fit_h = self._fit_heuristic(c, diag, ctx, offer)
        act_h = self._actionability(dec.talking_points, offer)
        fit, act, method, fit_llm, usage = fit_h, act_h, "heuristic", None, None
        if self.llm.enabled and not hv:
            data, usage = self.llm.chat_json(JUDGE_SYSTEM, json.dumps(dict(
                customer=dict(plan=c["plan_id"], tenure=c["tenure_months"], arpu=c["arpu_30d"]),
                reasons=[r.model_dump() for r in diag.reasons], offer=offer, rationale=dec.rationale,
                talking_points=dec.talking_points), ensure_ascii=False), tier="small", max_tokens=150)
            if data and isinstance(data.get("fit"), (int, float)) and isinstance(data.get("actionability"), (int, float)):
                fit_llm = float(max(1, min(5, data["fit"])))
                fit = fit_llm
                act = round((act_h + float(max(1, min(5, data["actionability"])))) / 2, 2)
                method = "llm+heuristic"

        overall = round(0.4 * fit + 0.3 * act + 0.3 * (5 * ground), 2)
        below = [n for n, ok in (("fit", fit >= FIT_MIN), ("actionability", act >= ACT_MIN), ("groundedness", ground >= GROUND_MIN)) if not ok]
        if hv or overall < 3.0 or len(below) >= 2:
            badge = "red"
        elif overall >= 4.0 and not below:
            badge = "green"
        else:
            badge = "amber"

        fb = [f"{v.rule_id}: {v.message}" for v in hv]
        if below:
            fb.append("weak dimensions: " + ", ".join(below))
        if bad:
            fb.append(f"untraceable numbers {sorted(bad)}")
        rep = ValidationReport(rule_pass=not hv, violations=viols, requires_approval=requires_approval, groundedness=ground,
                               fit=round(fit, 2), actionability=round(act, 2), overall=overall, badge=badge,
                               feedback=" | ".join(fb), judge_method=method, fit_heuristic=round(fit_h, 2), fit_llm=fit_llm)
        return rep, usage

    def _fit_heuristic(self, c, diag, ctx, offer) -> float:
        if not offer:
            return 1.0
        align = 2.0 if diag.primary in offer["targets"] else 1.0 if diag.secondary in offer["targets"] else 0.0
        ratio = offer["cost_usd"] / max(c["arpu_30d"], 1)
        cost_pts = 1.0 if ratio <= 1 else 0.5 if ratio <= 2 else 0.0
        hist = ctx.acceptance_by_offer.get(offer["offer_id"], 0.5)
        return round(min(5.0, 1 + align + cost_pts + hist), 2)

    @staticmethod
    def _actionability(tps: list[str], offer) -> float:
        a = 5.0
        if not 2 <= len(tps) <= 4:
            a -= 1
        if any(len(t.split()) > 25 for t in tps):
            a -= 1
        joined = " ".join(tps).lower()
        hours = str(offer["valid_hours"]) if offer else "24"
        if hours not in joined:
            a -= 1
        if not any(t.strip().endswith("?") for t in tps):
            a -= 1
        if offer and offer["name"].lower() not in joined:
            a -= 1
        return max(1.0, a)
