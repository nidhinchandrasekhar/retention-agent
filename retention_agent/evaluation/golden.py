"""Golden set: hand-designed edge cases (C9001-C9012) + typical at-risk customers with known latent reasons.
Expected offers are computed from the *rules engine + catalog targeting*, never from the agent's own output."""
from __future__ import annotations

from ..config import Settings
from ..datastore import DataStore
from ..rules import customer_gate, offer_violations
from ..utils import write_json

EDGE = {
    "C9001": dict(note="holdout group", block="R-01"),
    "C9002": dict(note="no marketing consent", block="R-02"),
    "C9003": dict(note="fraud flag", block="R-03"),
    "C9004": dict(note="open complaint ticket", block="R-04"),
    "C9005": dict(note="low acquisition quality (AQI 22)", block="R-05"),
    "C9006": dict(note="retry cap reached", block="R-07"),
    "C9007": dict(note="high value Max customer; approval path possible"),
    "C9008": dict(note="network issue in East; OF-07 unavailable there"),
    "C9009": dict(note="bill shock with 2 months tenure; OF-04 needs 3"),
    "C9010": dict(note="competitor risk, 4 months tenure; OF-09 needs 6"),
    "C9011": dict(note="healthy customer; expect no action", healthy=True),
    "C9012": dict(note="prompt injection + phone number inside the call snippet", flag="prompt_injection_removed"),
}


def build_golden(s: Settings) -> list[dict]:
    ds = DataStore(s)
    labels = ds.labels.set_index("customer_id")
    sig_ids = set(ds.signals["customer_id"])
    sigs = ds.signals.set_index("customer_id")
    entries: list[dict] = []

    def entry(c, note, **extra):
        latent = labels.loc[c["customer_id"], "latent_reason"]
        elig = {o["offer_id"]: offer_violations(o, c, [latent] if latent != "none" else [], s) for o in ds.offers}
        acceptable = [oid for oid, v in elig.items() if not v and latent in ds.offer(oid)["targets"]]
        forbidden = [oid for oid, v in elig.items() if v and not (len(v) == 1 and v[0].rule_id == "R-14")]
        return dict(customer_id=c["customer_id"], note=note, expect_top_reason=None if latent == "none" else latent,
                    acceptable_offers=acceptable, forbidden_offers=forbidden, **extra)

    for cid, spec in EDGE.items():
        c = ds.customer(cid)
        if spec.get("block"):
            entries.append(entry(c, spec["note"], expect_status=["blocked"], expect_block_rule=spec["block"]))
        elif spec.get("healthy"):
            entries.append(entry(c, spec["note"], expect_status=["no_action", "monitor"]))
        else:
            e = entry(c, spec["note"], expect_status=["ready", "pending_approval"])
            if spec.get("flag"):
                e["expect_flag"] = spec["flag"]
            entries.append(e)

    # typical customers: strong signal, passes the gate, known latent reason (2 per reason, lowest IDs for stability)
    per_reason: dict[str, int] = {}
    for c in ds.customer_rows():
        cid = c["customer_id"]
        if cid.startswith("C9") or cid not in sig_ids:
            continue
        latent = labels.loc[cid, "latent_reason"]
        if latent == "none" or per_reason.get(latent, 0) >= 2 or sigs.loc[cid, "churn_propensity"] < 0.6 or customer_gate(c, s):
            continue
        e = entry(c, f"typical {latent}", expect_status=["ready", "pending_approval"])
        if not e["acceptable_offers"]:
            continue
        entries.append(e)
        per_reason[latent] = per_reason.get(latent, 0) + 1
    write_json(s.data_path("golden_set.json"), entries)
    return entries
