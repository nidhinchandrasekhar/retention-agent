import pytest

from retention_agent import db
from retention_agent.run_daily import run_daily


def run(world, cid):
    ds = world["ds"]
    return world["pipe"].run_one(ds.customer(cid), ds.signal(cid), f"t-{cid}")


@pytest.mark.parametrize("cid,rule", [("C9001", "R-01"), ("C9002", "R-02"), ("C9003", "R-03"), ("C9004", "R-04"), ("C9005", "R-05"), ("C9006", "R-07")])
def test_blocked_edge_cases(world, cid, rule):
    rec = run(world, cid)["recommendation"]
    assert rec["status"] == "blocked" and rule in rec["skip_reason"]
    assert rec["offer_id"] is None


def test_healthy_customer_no_action(world):
    assert run(world, "C9011")["recommendation"]["status"] in ("no_action", "monitor")


def test_injection_is_sanitised_and_not_obeyed(world):
    st = run(world, "C9012")
    assert "prompt_injection_removed" in st["flags"]
    assert "9876543210" not in (st["signal"] or {}).get("call_snippet", "")
    assert st["recommendation"]["offer_id"] != "OF-14"


def test_ineligible_offers_never_recommended(world):
    assert run(world, "C9008")["recommendation"]["offer_id"] != "OF-07"   # wrong region
    assert run(world, "C9009")["recommendation"]["offer_id"] != "OF-04"   # tenure too short
    assert run(world, "C9010")["recommendation"]["offer_id"] != "OF-09"   # tenure too short


def test_recommendation_is_cited_and_traceable(world):
    st = run(world, "C9008")
    rec, val = st["recommendation"], st["recommendation"]["scores"]["validation"]
    assert rec["citations"] and val["rule_pass"] and val["groundedness"] >= 0.75
    assert "?" in rec["talking_points"][-1] and "24 hours" in " ".join(rec["talking_points"])


def test_critic_loop_recovers_when_prefilter_off(world):
    """Baseline config lets the Decision agent pick ineligible offers; the Validation agent must catch and loop."""
    from retention_agent.graph import Pipeline
    s = world["s"].with_overrides({"prefilter_eligibility": "false", "use_reason_filter": "false"})
    pipe = Pipeline(s, world["ds"], world["store"], world["llm"])
    ds = world["ds"]
    retried = 0
    for g in ds.golden:
        st = pipe.run_one(ds.customer(g["customer_id"]), ds.signal(g["customer_id"]), f"b-{g['customer_id']}")
        rec = st["recommendation"]
        if rec["offer_id"]:   # whatever finally ships must be compliant
            assert not [v for v in rec["scores"]["validation"]["violations"] if v["severity"] == "hard"]
        retried += st.get("retries", 0) > 0
        assert st.get("retries", 0) <= s.max_retries
    assert retried >= 1


def test_daily_run_pega_lifecycle(world):
    s = world["s"]
    r1 = run_daily(s, as_of="2026-10-01", label="d1", limit=150, progress=False)
    conn = db.connect(s)
    n_open = conn.execute("SELECT COUNT(*) FROM recommendations WHERE run_id=? AND status IN ('ready','pending_approval')", (r1["run_id"],)).fetchone()[0]
    assert n_open > 0
    # approvals only move pending items
    pend = conn.execute("SELECT rec_id FROM recommendations WHERE run_id=? AND status='pending_approval'", (r1["run_id"],)).fetchone()
    if pend:
        from datetime import datetime, timezone
        db.decide(conn, pend[0], "approve", "tester", datetime.now(timezone.utc))
        assert conn.execute("SELECT status FROM recommendations WHERE rec_id=?", (pend[0],)).fetchone()[0] == "approved"
    # next day: older open items expire (24h) or are superseded by the new run
    r2 = run_daily(s, as_of="2026-10-02", label="d2", limit=150, progress=False)
    assert conn.execute("SELECT COUNT(*) FROM recommendations WHERE run_id=? AND status IN ('expired','superseded')", (r1["run_id"],)).fetchone()[0] >= n_open
    logs = conn.execute("SELECT COUNT(*) FROM agent_logs WHERE run_id=?", (r2["run_id"],)).fetchone()[0]
    assert logs > 0
    assert conn.execute("SELECT COUNT(*) FROM agent_logs WHERE customer_hash IN (SELECT customer_id FROM recommendations)").fetchone()[0] == 0  # ids are hashed
    conn.close()
