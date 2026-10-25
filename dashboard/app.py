"""Streamlit dashboard for the retention agent.
Run:  streamlit run dashboard/app.py     (or: python -m retention_agent dashboard)"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from retention_agent import db
from retention_agent.config import get_settings
from retention_agent.datastore import DataStore

st.set_page_config(page_title="XYZ Telecom - Retention Agent", page_icon=":telephone_receiver:", layout="wide")
S = get_settings()
BADGE = {"green": "🟢 Green", "amber": "🟠 Amber", "red": "🔴 Red"}
STATUS_HELP = {
    "ready": "Green, within limits: delivered to Pega for care agents", "pending_approval": "needs human approval (Amber or cost above limit)",
    "approved": "approved by a human", "rejected": "rejected by a human", "escalated": "agent could not produce a compliant offer: human review",
    "blocked": "excluded by an eligibility rule", "monitor": "medium risk: watch only", "no_action": "low risk",
    "expired": "24h validity passed", "superseded": "replaced by a newer run",
}


@st.cache_resource
def get_ds():
    return DataStore(S)


@st.cache_resource
def get_pipeline():
    from retention_agent.graph import Pipeline
    from retention_agent.run_daily import prepare
    ds, store, llm = prepare(S)
    return Pipeline(S, ds, store, llm), store


def conn():
    return db.connect(S)


def j(x, default=None):
    try:
        return json.loads(x) if isinstance(x, str) and x else default
    except Exception:
        return default


# --------------------------------------------------------------------------- empty state
ds = get_ds()
c = conn()
runs = db.df(c, "SELECT * FROM runs WHERE status='done' ORDER BY started_at DESC")

st.title("📞 Retention Agent — XYZ Telecom (Value Prepaid)")
st.caption(f"Mode: **{S.mode}** · embeddings: **{S.embedding_backend}** · synthetic data only · daily 5 AM run · offers valid {S.offer_ttl_hours}h")

if runs.empty or not ds.exists():
    st.info("No runs yet. Generate synthetic data and run the daily job (takes under a minute in mock mode).")
    if st.button("Generate data + run daily job", type="primary"):
        from retention_agent.data_gen import generate
        from retention_agent.run_daily import run_daily
        with st.spinner("Generating data, indexing Chroma, running the pipeline..."):
            generate(S)
            run_daily(S, progress=False)
        st.rerun()
    st.stop()

with st.sidebar:
    st.header("Run")
    run_id = st.selectbox("Daily run", runs["run_id"], format_func=lambda r: f"{r}  ({runs.set_index('run_id').loc[r, 'label']})")
    run = runs.set_index("run_id").loc[run_id]
    st.write(f"As of **{run['as_of']}**")
    st.write(f"Mode **{run['mode']}** · LLM cost **${run['cost_usd']:.4f}**")
    with st.expander("Settings used"):
        st.json(j(run["settings_json"], {}))
    if st.button("Expire stale recommendations now"):
        n = db.expire_stale(c, datetime.now(timezone.utc))
        st.toast(f"Expired {n}")

recs = db.df(c, "SELECT * FROM recommendations WHERE run_id=?", (run_id,))
recs["primary_reason"] = recs["reasons"].map(lambda x: (j(x, []) or [{}])[0].get("code", "") if j(x, []) else "")
logs = db.df(c, "SELECT * FROM agent_logs WHERE run_id=?", (run_id,))

tabs = st.tabs(["Overview", "Recommendations", "Approvals", "Agent observability", "Evaluation", "RAG explorer", "Customer lab"])

# --------------------------------------------------------------------------- overview
with tabs[0]:
    k = st.columns(6)
    k[0].metric("Customers scored", int(run["n_customers"]))
    k[1].metric("High risk", int(run["n_high"]))
    k[2].metric("Ready for care agents", int(run["n_ready"]))
    k[3].metric("Pending approval", int(run["n_pending"]))
    k[4].metric("Blocked by rules", int(run["n_blocked"]))
    k[5].metric("Escalated to humans", int(run["n_escalated"]))
    a, b = st.columns(2)
    with a:
        st.subheader("Funnel")
        funnel = pd.DataFrame({"stage": ["1 scored", "2 high risk", "3 passed eligibility", "4 offer produced"],
                               "customers": [run["n_customers"], run["n_high"], run["n_high"] - run["n_blocked"],
                                             run["n_ready"] + run["n_pending"]]}).set_index("stage")
        st.bar_chart(funnel)
    with b:
        st.subheader("Quality badges")
        st.bar_chart(pd.DataFrame({"count": [run["n_green"], run["n_amber"], run["n_red"]]}, index=["green", "amber", "red"]))
    a, b = st.columns(2)
    with a:
        st.subheader("Why customers are at risk (high tier)")
        hi = recs[recs["tier"] == "high"]
        st.bar_chart(hi["primary_reason"].value_counts())
    with b:
        st.subheader("Status of every customer")
        st.bar_chart(recs["status"].value_counts())
    with st.expander("What do the statuses mean?"):
        st.table(pd.DataFrame({"status": STATUS_HELP.keys(), "meaning": STATUS_HELP.values()}))

# --------------------------------------------------------------------------- recommendations
def resolve_citation(ref: str):
    src, _, rid = ref.partition(":")
    if src == "offer":
        o = ds.offer(rid)
        return f"**{o['name']}** (${o['cost_usd']:.0f}) — {o['benefit_text']}" if o else ref
    if src == "call":
        t = next((t for t in ds.transcripts if t["transcript_id"] == rid), None)
        return f"Past call, reason *{t['reason']}*, outcome **{t['outcome']}** with {t['offer_id']}:\n\n" + "\n\n".join(t["turns"]) if t else ref
    if src == "playbook":
        p = next((p for p in ds.playbooks if p["playbook_id"] == rid), None)
        return f"**{p['title']}** — {p['text']}" if p else ref
    if src == "rule":
        r = next((r for r in ds.rules if r["rule_id"] == rid), None)
        return f"**{rid}** — {r['text']}" if r else ref
    return ref


def show_recommendation(r):
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### Customer {r['customer_id']} · {r['status'].replace('_', ' ')}")
        st.write(f"Risk **{r['risk_score']:.2f}** ({r['tier']}) · retries **{r['retries']}**")
        for rs in j(r["reasons"], []):
            st.write(f"- **{rs['code'].replace('_', ' ')}** (weight {rs['weight']:.2f}): {rs['evidence']}")
        if r["offer_id"]:
            st.success(f"**{r['offer_name']}** ({r['offer_id']}) · cost ${r['offer_cost']:.0f}" + (f" · fallback: {r['fallback_offer_id']}" if r["fallback_offer_id"] else ""))
            st.markdown("**Talking points for the care agent**")
            for tp in j(r["talking_points"], []):
                st.markdown(f"> {tp}")
            st.markdown("**Rationale**")
            st.write(r["rationale"])
        if r["skip_reason"]:
            st.warning(r["skip_reason"])
    with right:
        sc = j(r["scores"], {}) or {}
        v = sc.get("validation")
        if v:
            st.markdown(f"**Validation: {BADGE.get(v['badge'], v['badge'])}** — overall {v['overall']:.2f}")
            st.progress(min(v["groundedness"], 1.0), text=f"groundedness {v['groundedness']:.2f}")
            st.progress(min(v["fit"] / 5, 1.0), text=f"fit {v['fit']:.1f}/5")
            st.progress(min(v["actionability"] / 5, 1.0), text=f"actionability {v['actionability']:.1f}/5")
            st.caption(f"judge: {v['judge_method']} · decision method: {sc.get('method')}")
            for x in v["violations"]:
                st.write(f"{'⛔' if x['severity'] == 'hard' else '🔒'} {x['rule_id']}: {x['message']}")
        if sc.get("flags"):
            st.caption("flags: " + ", ".join(sorted(set(sc["flags"]))))
    cites = j(r["citations"], [])
    if cites:
        st.markdown("**Sources (RAG citations)**")
        for ref in cites:
            with st.expander(ref):
                st.markdown(resolve_citation(ref))


with tabs[1]:
    f = st.columns(4)
    status_f = f[0].multiselect("Status", sorted(recs["status"].unique()), default=[s_ for s_ in ["ready", "pending_approval", "escalated", "approved", "rejected"] if s_ in set(recs["status"])])
    badge_f = f[1].multiselect("Badge", ["green", "amber", "red"])
    reason_f = f[2].multiselect("Primary reason", sorted(x for x in recs["primary_reason"].unique() if x))
    text_f = f[3].text_input("Customer id contains")
    view = recs[recs["status"].isin(status_f)] if status_f else recs
    if badge_f:
        view = view[view["badge"].isin(badge_f)]
    if reason_f:
        view = view[view["primary_reason"].isin(reason_f)]
    if text_f:
        view = view[view["customer_id"].str.contains(text_f, case=False)]
    st.dataframe(view[["customer_id", "status", "badge", "risk_score", "primary_reason", "offer_id", "offer_name", "offer_cost", "overall_score", "retries", "skip_reason"]],
                 use_container_width=True, hide_index=True, height=300)
    if len(view):
        pick = st.selectbox("Open recommendation", view["customer_id"].tolist())
        show_recommendation(view[view["customer_id"] == pick].iloc[0])

# --------------------------------------------------------------------------- approvals (HITL)
with tabs[2]:
    st.subheader("Approval queue (human-in-the-loop)")
    st.caption("Amber badges and offers above the cost limit wait here. Approved items flow to Pega; unattended items expire after 24h.")
    pend = db.df(c, "SELECT * FROM recommendations WHERE status='pending_approval' ORDER BY risk_score DESC")
    approver = st.text_input("Approver", value="retention.lead")
    if pend.empty:
        st.success("Nothing waiting for approval.")
    for _, r in pend.iterrows():
        with st.container(border=True):
            h = st.columns([4, 1, 1])
            h[0].markdown(f"**{r['customer_id']}** · {r['offer_name']} · ${r['offer_cost']:.0f} · {BADGE.get(r['badge'], r['badge'])} · risk {r['risk_score']:.2f} · expires {r['expires_at']}")
            if h[1].button("Approve", key=f"a{r['rec_id']}"):
                db.decide(c, int(r["rec_id"]), "approve", approver, datetime.now(timezone.utc))
                st.rerun()
            if h[2].button("Reject", key=f"r{r['rec_id']}"):
                db.decide(c, int(r["rec_id"]), "reject", approver, datetime.now(timezone.utc))
                st.rerun()
            if r["fallback_offer_id"]:
                st.caption(f"Fallback if rejected (under ${S.approval_limit_usd:.0f}): {r['fallback_offer_id']}")
            with st.expander("Details"):
                show_recommendation(r)
    st.subheader("Decided")
    st.dataframe(db.df(c, "SELECT rec_id, customer_id, status, offer_name, approver, decided_at FROM recommendations WHERE status IN ('approved','rejected') ORDER BY decided_at DESC LIMIT 50"),
                 use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- observability
with tabs[3]:
    st.subheader("Per-agent performance (this run)")
    if logs.empty:
        st.info("No logs.")
    else:
        agg = logs.groupby("agent").agg(calls=("id", "count"), mean_ms=("latency_ms", "mean"), p95_ms=("latency_ms", lambda x: x.quantile(0.95)),
                                        tokens_in=("tokens_in", "sum"), tokens_out=("tokens_out", "sum"), cost_usd=("cost_usd", "sum"),
                                        errors=("status", lambda x: (x == "error").sum())).round(3)
        st.dataframe(agg, use_container_width=True)
        a, b = st.columns(2)
        a.markdown("**Mean latency per agent (ms)**")
        a.bar_chart(agg["mean_ms"])
        b.markdown("**Guardrail & validation flags**")
        flags = logs["flags"].str.split(",").explode()
        flags = flags[flags.fillna("") != ""]
        b.bar_chart(flags.value_counts().head(15))
        st.markdown("**Critic loop**")
        val = logs[logs["agent"] == "validation"]
        st.write(f"Validation passes: **{len(val)}** · customers needing a retry: **{int((val['attempt'] > 0).sum())}** · max retries allowed: **{S.max_retries}**")
    st.subheader("Run history")
    hist = db.df(c, "SELECT run_id, label, as_of, mode, n_customers, n_high, n_ready, n_pending, n_blocked, n_escalated, cost_usd FROM runs WHERE status='done' ORDER BY started_at")
    st.dataframe(hist, use_container_width=True, hide_index=True)
    if len(hist) > 1:
        st.line_chart(hist.set_index("as_of")[["n_high", "n_ready", "n_pending"]])
    st.caption("Customer ids are hashed in agent logs; raw inputs are never stored there.")

# --------------------------------------------------------------------------- evaluation
with tabs[4]:
    ev = db.df(c, "SELECT eval_id, created_at, label, mode, metrics_json, cases_json, settings_json FROM eval_runs ORDER BY created_at DESC")
    if ev.empty:
        st.info("No evaluations yet. Run `python -m retention_agent eval --label baseline` (optionally with --set overrides) and again for the tuned config.")
    else:
        ev["name"] = ev["label"] + " · " + ev["created_at"]
        cols = st.columns(2)
        A = cols[0].selectbox("Baseline", ev["name"], index=min(1, len(ev) - 1))
        B = cols[1].selectbox("Candidate", ev["name"], index=0)
        ma = j(ev.set_index("name").loc[A, "metrics_json"], {})
        mb = j(ev.set_index("name").loc[B, "metrics_json"], {})
        rows = []
        for k_ in mb:
            x, y = ma.get(k_), mb.get(k_)
            d = (y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
            rows.append({"metric": k_, "baseline": x, "candidate": y, "delta": None if d is None else round(d, 3)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Methods: golden set · rules-as-oracle · heuristic / LLM judge · guardrail tests · AUC vs churn labels · operational metrics.")
        payload = j(ev.set_index("name").loc[B, "cases_json"], {})
        st.markdown("**Golden cases (candidate)**")
        gdf = pd.DataFrame(payload.get("golden", []))
        if not gdf.empty:
            st.dataframe(gdf, use_container_width=True, hide_index=True)
        st.markdown("**Guardrail tests (candidate)**")
        st.dataframe(pd.DataFrame(payload.get("guardrails", [])), use_container_width=True, hide_index=True)
        with st.expander("Settings of the candidate"):
            st.json(j(ev.set_index("name").loc[B, "settings_json"], {}))

# --------------------------------------------------------------------------- RAG explorer
with tabs[5]:
    st.subheader("Query the vector database directly")
    q = st.text_input("Query", "customer keeps getting dropped calls at home")
    cc = st.columns(3)
    coll = cc[0].selectbox("Collection", ["transcripts", "offers", "policies"])
    kk = cc[1].slider("Top k", 1, 10, 5)
    reason = cc[2].selectbox("Metadata filter: reason", ["(none)"] + ["price_sensitivity", "bill_shock", "network_quality", "competitor_offer", "disengaged", "plan_mismatch"])
    if st.button("Search"):
        try:
            _, store = get_pipeline()
            where = {"reason": reason} if reason != "(none)" and coll in ("transcripts",) else None
            hits = store.query(coll, q, kk, where=where)
            for h in hits:
                st.markdown(f"**{h.id}** · similarity **{h.score:.3f}** · `{h.meta}`")
                st.text(h.doc)
            if not hits:
                st.info("No hits.")
            st.caption(f"Collections: {store.counts()}")
        except Exception as e:
            st.error(f"Search failed: {e}")

# --------------------------------------------------------------------------- customer lab
with tabs[6]:
    st.subheader("Run the agents on one customer, live")
    cid = st.selectbox("Customer", ds.customers["customer_id"].tolist(), index=int(ds.customers.index[ds.customers["customer_id"] == "C9007"][0]) if "C9007" in set(ds.customers["customer_id"]) else 0)
    cust, sig = ds.customer(cid), ds.signal(cid)
    left, right = st.columns(2)
    left.markdown("**Customer features**")
    left.json(cust, expanded=False)
    right.markdown("**Latest RTS signal**")
    right.json(sig or {"info": "no call in the last 7 days"}, expanded=False)
    lab = ds.labels[ds.labels["customer_id"] == cid]
    if not lab.empty:
        st.caption(f"Ground truth (hidden from agents): latent reason = {lab.iloc[0]['latent_reason']}, churned in 30d = {lab.iloc[0]['churned_30d']}")
    if st.button("Run pipeline (dry run, nothing written)", type="primary"):
        try:
            pipe, _ = get_pipeline()
            stt = pipe.run_one(cust, sig, f"lab-{cid}-{datetime.now().timestamp()}")
            steps = pd.DataFrame(stt["audit_log"])[["agent", "latency_ms", "status", "tokens_in", "tokens_out", "cost_usd", "note", "flags"]]
            st.dataframe(steps, use_container_width=True, hide_index=True)
            rec = stt["recommendation"]
            show_recommendation({**rec, "reasons": json.dumps(rec["reasons"]), "talking_points": json.dumps(rec["talking_points"]),
                                 "citations": json.dumps(rec["citations"]), "scores": json.dumps(rec["scores"], default=str),
                                 "offer_cost": rec["offer_cost"], "fallback_offer_id": rec["fallback_offer_id"]})
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
