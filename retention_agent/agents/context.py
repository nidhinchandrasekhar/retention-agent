"""Context agent: RAG. Retrieves *grounding* for the Decision agent from three Chroma collections.

  offers       -> semantic ranking of catalog offers for this customer's problem (hard eligibility is a structured pre-filter)
  transcripts  -> similar past calls (metadata-filtered by diagnosed reason) and what was accepted
  policies     -> playbook guidance for the reason + the most relevant rules
Every retrieved item becomes a Citation the Decision agent must reference as [source:ID]."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from ..config import Settings
from ..datastore import DataStore
from ..observability import make_log
from ..rag import VectorStore
from ..rules import offer_violations
from ..schemas import Candidate, Citation, ContextBundle, DiagnosticsOutput
from ..utils import Timer

QUERY_TEXT = {
    "price_sensitivity": "customer finds prices too high and wants a cheaper deal or price protection",
    "bill_shock": "customer was charged unexpected deductions and wants a refund or credit",
    "network_quality": "customer has repeated network problems such as dropped calls and slow data",
    "competitor_offer": "customer is considering porting out to a competitor with a better deal",
    "disengaged": "customer recharges and uses the service less and needs a reason to stay engaged",
    "plan_mismatch": "customer's plan does not fit their data usage",
}


class ContextAgent:
    name = "context"

    def __init__(self, settings: Settings, ds: DataStore, store: VectorStore):
        self.s, self.ds, self.store = settings, ds, store

    def run(self, c: dict, diag: DiagnosticsOutput, snippet: str, excluded: list[str] | None = None):
        with Timer() as t:
            ctx = self._retrieve(c, diag, snippet, excluded or [])
        flags = (["no_hits"] if ctx.no_hits else []) + ctx.stale_flags
        return ctx, make_log(self.name, t.ms, flags=flags,
                             note=f"{len(ctx.candidates)} candidates, {len(ctx.similar_calls)} calls, {len(ctx.policies)} policies")

    def _retrieve(self, c, diag, snippet, excluded):
        s = self.s
        reasons = [r.code for r in diag.reasons]
        query = " ".join(QUERY_TEXT[r] for r in reasons[:2]) + (f" Customer said: {snippet}" if snippet else "")

        # 1) offers: structured eligibility pre-filter, then semantic ranking
        if s.prefilter_eligibility:
            pool = [o["offer_id"] for o in self.ds.offers if not offer_violations(o, c, reasons, s)]
        else:
            pool = [o["offer_id"] for o in self.ds.offers if o["active"]]
        pool = [oid for oid in pool if oid not in excluded]
        hits = self.store.query("offers", query, k=8, where={"offer_id": {"$in": pool}} if pool else None,
                                max_distance=None) if pool else []
        no_hits = False
        hits = [h for h in hits if h.distance <= s.max_distance] if hits else []
        if not hits and pool:  # RAG returned nothing usable -> structured fallback, flagged
            no_hits = True
            hits = []
        cands, offer_cites = [], []
        for rank, h in enumerate(hits, 1):
            o = self.ds.offer(h.id)
            cands.append(Candidate(offer_id=o["offer_id"], name=o["name"], cost_usd=o["cost_usd"],
                                   semantic_score=h.score, rank=rank, citation=f"offer:{o['offer_id']}"))
            offer_cites.append(Citation(ref=f"offer:{o['offer_id']}", source="offer", id=o["offer_id"], score=h.score,
                                        snippet=o["benefit_text"], meta=dict(cost_usd=o["cost_usd"], targets=o["targets"])))
        if no_hits:
            for rank, oid in enumerate([x for x in pool if reasons[0] in self.ds.offer(x)["targets"]][:3], 1):
                o = self.ds.offer(oid)
                cands.append(Candidate(offer_id=oid, name=o["name"], cost_usd=o["cost_usd"], semantic_score=0.0, rank=rank,
                                       citation=f"offer:{oid}"))
                offer_cites.append(Citation(ref=f"offer:{oid}", source="offer", id=oid, score=0.0, snippet=o["benefit_text"],
                                            meta=dict(cost_usd=o["cost_usd"], targets=o["targets"])))

        # 2) similar past calls (filtered by reason when enabled)
        where = {"reason": reasons[0]} if s.use_reason_filter else None
        call_hits = self.store.query("transcripts", query, k=s.retrieval_top_k * 2, where=where, max_distance=s.max_distance)
        seen, calls = set(), []
        for h in call_hits:
            tid = h.meta["transcript_id"]
            if tid in seen:
                continue
            seen.add(tid)
            calls.append(Citation(ref=f"call:{tid}", source="call", id=tid, score=h.score, snippet=h.doc[:240],
                                  meta=dict(outcome=h.meta["outcome"], offer_id=h.meta["offer_id"], reason=h.meta["reason"])))
            if len(calls) >= s.retrieval_top_k:
                break
        counts = defaultdict(lambda: [0, 0])
        for cl in calls:
            counts[cl.meta["offer_id"]][1] += 1
            counts[cl.meta["offer_id"]][0] += cl.meta["outcome"] == "accepted"
        accept = {k: round(v[0] / v[1], 3) for k, v in counts.items()}

        # 3) policies: playbooks for the reasons, plus the most relevant rules
        pb = self.store.query("policies", query, k=2, max_distance=s.max_distance,
                              where={"$and": [{"kind": "playbook"}, {"reason": {"$in": reasons[:2]}}]})
        rl = self.store.query("policies", query + " offer eligibility approval cost limit", k=2, max_distance=s.max_distance,
                              where={"kind": "rule"})
        pols = [Citation(ref=f"{h.meta['kind']}:{h.meta['ref_id']}", source=h.meta["kind"], id=h.meta["ref_id"], score=h.score,
                         snippet=h.doc[:240]) for h in pb + rl]

        stale = []
        now = datetime.now(timezone.utc)
        for cd in cands:
            o = self.ds.offer(cd.offer_id)
            try:
                age = (now - datetime.fromisoformat(o["last_reviewed"]).replace(tzinfo=timezone.utc)).days
                if age > 180:
                    stale.append(f"stale_offer:{cd.offer_id}")
            except Exception:
                pass
        return ContextBundle(candidates=cands, similar_calls=calls, policies=pols, offer_citations=offer_cites,
                             acceptance_by_offer=accept, acceptance_counts={k: list(v) for k, v in counts.items()},
                             query=query, no_hits=no_hits or not cands, stale_flags=stale)
