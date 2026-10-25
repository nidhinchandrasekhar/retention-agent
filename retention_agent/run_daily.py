"""The daily 5 AM job: score every customer, run the agent pipeline for high-risk ones, write the Pega outbox.
Schedule it with cron / Airflow / GitHub Actions: `python -m retention_agent run`."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from . import db
from .config import Settings
from .datastore import DataStore
from .graph import Pipeline
from .llm import LLMClient
from .rag import VectorStore
from .utils import iso, now_utc, parse_ts, sha


def prepare(s: Settings, auto_generate: bool = True):
    ds = DataStore(s)
    if not ds.exists():
        if not auto_generate:
            raise FileNotFoundError("No data found. Run: python -m retention_agent generate")
        from .data_gen import generate
        generate(s)
        ds = DataStore(s)
    store = VectorStore(s)
    store.ensure(ds)
    return ds, store, LLMClient(s)


def process(pipe: Pipeline, ds: DataStore, customers: list[dict], tag: str, progress: bool = False) -> list[dict]:
    sigs = {r["customer_id"]: r for r in (ds.signals.to_dict(orient="records"))}
    for r in sigs.values():
        r["intents"] = [i for i in str(r.get("intents", "")).split("|") if i]
    out = []
    for i, c in enumerate(customers, 1):
        st = pipe.run_one(c, sigs.get(c["customer_id"]), f"{tag}-{c['customer_id']}")
        out.append(st)
        if progress and i % 100 == 0:
            print(f"  processed {i}/{len(customers)}")
    return out


def run_daily(s: Settings, as_of: str | None = None, label: str = "daily", limit: int | None = None,
              dry_run: bool = False, progress: bool = True, customer_ids: list[str] | None = None) -> dict:
    ds, store, llm = prepare(s)
    pipe = Pipeline(s, ds, store, llm)
    when = parse_ts(as_of) if as_of else now_utc().replace(hour=5, minute=0, second=0, microsecond=0)
    rows = ds.customer_rows()
    if customer_ids:
        rows = [r for r in rows if r["customer_id"] in set(customer_ids)]
    if limit:
        rows = rows[:limit]
    run_id = f"run-{when:%Y%m%d}-{uuid.uuid4().hex[:6]}"
    conn = None if dry_run else db.connect(s)
    if conn:
        n_exp = db.expire_stale(conn, when)
        db.start_run(conn, run_id, s, when, label, iso(now_utc()))
        if progress and n_exp:
            print(f"  expired {n_exp} stale recommendations from earlier runs")
    if progress:
        print(f"Run {run_id} | mode={s.mode} | embeddings={s.embedding_backend} | customers={len(rows)} | as_of={iso(when)}")
    states = process(pipe, ds, rows, run_id, progress)

    stats = dict(n_customers=len(states), n_high=0, n_ready=0, n_pending=0, n_escalated=0, n_blocked=0, n_monitor=0,
                 n_no_action=0, n_green=0, n_amber=0, n_red=0, cost_usd=0.0)
    key = {"ready": "n_ready", "pending_approval": "n_pending", "escalated": "n_escalated", "blocked": "n_blocked",
           "monitor": "n_monitor", "no_action": "n_no_action"}
    for st in states:
        rec = st["recommendation"]
        stats[key[rec["status"]]] += 1
        stats["n_high"] += rec["tier"] == "high"
        if rec.get("badge"):
            stats[f"n_{rec['badge']}"] += 1
        stats["cost_usd"] += sum(e["cost_usd"] for e in st["audit_log"])
        if conn:
            db.log_agents(conn, run_id, sha(rec["customer_id"]), st["audit_log"])
            db.save_recommendation(conn, run_id, rec, when, s.offer_ttl_hours)
    stats["cost_usd"] = round(stats["cost_usd"], 5)
    if conn:
        conn.commit()
        db.finish_run(conn, run_id, stats, iso(now_utc()))
        conn.close()
    if progress:
        print("  " + ", ".join(f"{k[2:]}={v}" for k, v in stats.items() if k.startswith("n_") and v))
        print(f"  llm cost ${stats['cost_usd']}")
    return dict(run_id=run_id, stats=stats, states=states)
