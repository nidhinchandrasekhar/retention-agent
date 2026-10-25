"""Exercises every LLM code path with a stub client (no network, no key needed)."""
import json

from retention_agent.graph import Pipeline
from retention_agent.llm import LLMClient, Usage


class StubLLM(LLMClient):
    def __init__(self, s, mode="good"):
        super().__init__(s)
        self.enabled = True
        self.mode = mode
        self.calls = []

    def chat_json(self, system, user, tier="small", max_tokens=700):
        u = Usage(tokens_in=100, tokens_out=50, cost=0.001, latency_ms=5, model="stub")
        if "Decision agent" in system:
            self.calls.append(("decision", tier))
            facts = json.loads(user.split("\nPREVIOUS")[0].split("\nREVIEWER")[0])
            c = facts["CANDIDATES"][0]
            oid = c["offer_id"]
            if self.mode == "hallucinate":
                tps = ["I am sorry about that.", f"Today I can offer {c['name']}: get $99 off and unlimited data forever.", "This offer is valid for 24 hours. Shall I apply it?"]
            else:
                tps = ["I am sorry about the trouble you have had.", f"Today I can offer {c['name']}: {c['benefit']}.", "This offer is valid for 24 hours. Would you like me to apply it now?"]
            return {"offer_id": oid, "rationale": f"{c['name']} fits the diagnosed reason [offer:{oid}].", "talking_points": tps,
                    "citations": [f"offer:{oid}"], "confidence": 0.8}, u
        if "strict reviewer" in system:
            self.calls.append(("reflection", tier))
            return {"ok": True, "issues": []}, u
        if "grade a retention" in system:
            self.calls.append(("judge", tier))
            return {"fit": 4, "actionability": 5, "comment": "ok"}, u
        if "classify why" in system:
            self.calls.append(("tiebreak", tier))
            return {"reason": "disengaged"}, u
        return None, u


def _pipe(world, mode):
    llm = StubLLM(world["s"], mode)
    return Pipeline(world["s"], world["ds"], world["store"], llm), llm


def test_llm_path_end_to_end_with_model_routing(world):
    pipe, llm = _pipe(world, "good")
    ds = world["ds"]
    st = pipe.run_one(ds.customer("C9008"), ds.signal("C9008"), "llm-good")
    rec = st["recommendation"]
    assert rec["status"] in ("ready", "pending_approval") and rec["scores"]["method"] == "llm"
    assert rec["scores"]["validation"]["judge_method"] == "llm+heuristic"
    tiers = dict(llm.calls)
    assert tiers["decision"] == "large" and tiers["reflection"] == "small" and tiers["judge"] == "small"   # model routing
    assert sum(e["cost_usd"] for e in st["audit_log"]) > 0


def test_hallucinating_llm_never_ships(world):
    pipe, llm = _pipe(world, "hallucinate")
    ds = world["ds"]
    st = pipe.run_one(ds.customer("C9008"), ds.signal("C9008"), "llm-bad")
    rec = st["recommendation"]
    assert rec["status"] == "escalated" and rec["offer_id"] is None      # fails closed to a human
    assert st["retries"] == world["s"].max_retries                       # critic loop hit its cap
    assert "R-19" in rec["skip_reason"] or "R-16" in rec["skip_reason"]


def test_llm_failure_degrades_to_deterministic(world):
    class Broken(StubLLM):
        def chat_json(self, *a, **k):
            return None, Usage(error="boom")
    llm = Broken(world["s"])
    pipe = Pipeline(world["s"], world["ds"], world["store"], llm)
    ds = world["ds"]
    st = pipe.run_one(ds.customer("C9008"), ds.signal("C9008"), "llm-broken")
    assert st["recommendation"]["status"] in ("ready", "pending_approval")
    assert st["recommendation"]["scores"]["method"] == "deterministic"
