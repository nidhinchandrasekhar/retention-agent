"""SQLite persistence: run history, agent audit logs, the mocked Pega outbox and evaluation results."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from .config import Settings
from .utils import iso, parse_ts

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, as_of TEXT, mode TEXT, label TEXT,
  settings_json TEXT, n_customers INT, n_high INT, n_ready INT, n_pending INT, n_escalated INT,
  n_blocked INT, n_monitor INT, n_no_action INT, n_green INT, n_amber INT, n_red INT,
  cost_usd REAL, status TEXT);
CREATE TABLE IF NOT EXISTS agent_logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, customer_hash TEXT, agent TEXT, latency_ms INT,
  tokens_in INT, tokens_out INT, cost_usd REAL, model TEXT, status TEXT, flags TEXT, note TEXT, attempt INT);
CREATE INDEX IF NOT EXISTS ix_logs_run ON agent_logs(run_id);
CREATE TABLE IF NOT EXISTS recommendations(
  rec_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, customer_id TEXT, created_at TEXT, expires_at TEXT,
  status TEXT, offer_id TEXT, offer_name TEXT, offer_cost REAL, fallback_offer_id TEXT, badge TEXT,
  overall_score REAL, risk_score REAL, tier TEXT, reasons TEXT, talking_points TEXT, rationale TEXT,
  citations TEXT, scores TEXT, skip_reason TEXT, retries INT, approver TEXT, decided_at TEXT);
CREATE INDEX IF NOT EXISTS ix_rec_run ON recommendations(run_id);
CREATE INDEX IF NOT EXISTS ix_rec_cust ON recommendations(customer_id);
CREATE TABLE IF NOT EXISTS eval_runs(
  eval_id TEXT PRIMARY KEY, created_at TEXT, label TEXT, mode TEXT, settings_json TEXT, metrics_json TEXT, cases_json TEXT);
"""
OPEN_STATUSES = ("ready", "pending_approval", "approved")


def connect(s: Settings) -> sqlite3.Connection:
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(s.db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def start_run(conn, run_id: str, s: Settings, as_of: datetime, label: str, started_at: str) -> None:
    conn.execute("INSERT INTO runs(run_id,started_at,as_of,mode,label,settings_json,status) VALUES(?,?,?,?,?,?,?)",
                 (run_id, started_at, iso(as_of), s.mode, label, json.dumps(s.public_dict()), "running"))
    conn.commit()


def finish_run(conn, run_id: str, stats: dict, finished_at: str) -> None:
    cols = ",".join(f"{k}=?" for k in stats)
    conn.execute(f"UPDATE runs SET finished_at=?, status='done', {cols} WHERE run_id=?", (finished_at, *stats.values(), run_id))
    conn.commit()


def log_agents(conn, run_id: str, customer_hash: str, entries: list[dict]) -> None:
    conn.executemany(
        "INSERT INTO agent_logs(run_id,customer_hash,agent,latency_ms,tokens_in,tokens_out,cost_usd,model,status,flags,note,attempt) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        [(run_id, customer_hash, e["agent"], e["latency_ms"], e["tokens_in"], e["tokens_out"], e["cost_usd"], e.get("model", ""),
          e["status"], ",".join(e["flags"]), e.get("note", "")[:300], e.get("attempt", 0)) for e in entries])


def expire_stale(conn, now: datetime) -> int:
    cur = conn.execute(
        f"UPDATE recommendations SET status='expired' WHERE status IN {OPEN_STATUSES} AND expires_at < ?", (iso(now),))
    conn.commit()
    return cur.rowcount


def save_recommendation(conn, run_id: str, rec: dict, created: datetime, ttl_hours: int) -> int:
    # Pega behaviour: the next run supersedes any still-open recommendation for this customer.
    conn.execute(f"UPDATE recommendations SET status='superseded' WHERE customer_id=? AND status IN {OPEN_STATUSES}",
                 (rec["customer_id"],))
    cur = conn.execute(
        "INSERT INTO recommendations(run_id,customer_id,created_at,expires_at,status,offer_id,offer_name,offer_cost,fallback_offer_id,"
        "badge,overall_score,risk_score,tier,reasons,talking_points,rationale,citations,scores,skip_reason,retries) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, rec["customer_id"], iso(created), iso(created + timedelta(hours=ttl_hours)), rec["status"], rec.get("offer_id"),
         rec.get("offer_name", ""), rec.get("offer_cost", 0.0), rec.get("fallback_offer_id"), rec.get("badge"), rec.get("overall_score"),
         rec.get("risk_score", 0.0), rec.get("tier", "low"), json.dumps(rec.get("reasons", [])), json.dumps(rec.get("talking_points", [])),
         rec.get("rationale", ""), json.dumps(rec.get("citations", [])), json.dumps(rec.get("scores", {}), default=str),
         rec.get("skip_reason", ""), rec.get("retries", 0)))
    return cur.lastrowid


def decide(conn, rec_id: int, decision: str, approver: str, when: datetime) -> None:
    """Human-in-the-loop: approve/reject a pending recommendation (dashboard buttons call this)."""
    status = {"approve": "approved", "reject": "rejected"}[decision]
    conn.execute("UPDATE recommendations SET status=?, approver=?, decided_at=? WHERE rec_id=? AND status='pending_approval'",
                 (status, approver, iso(when), rec_id))
    conn.commit()


def df(conn, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def latest_run_id(conn) -> str | None:
    r = conn.execute("SELECT run_id FROM runs WHERE status='done' ORDER BY started_at DESC LIMIT 1").fetchone()
    return r[0] if r else None


def save_eval(conn, eval_id: str, created: str, label: str, mode: str, settings: dict, metrics: dict, cases: list) -> None:
    conn.execute("INSERT OR REPLACE INTO eval_runs VALUES(?,?,?,?,?,?,?)",
                 (eval_id, created, label, mode, json.dumps(settings), json.dumps(metrics), json.dumps(cases, default=str)))
    conn.commit()
