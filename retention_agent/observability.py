"""Uniform audit-log entry every agent returns. Customer ids are hashed; raw inputs are never logged."""
from __future__ import annotations

from .llm import Usage


def make_log(agent: str, latency_ms: int, usage: Usage | None = None, status: str = "ok",
             flags: list[str] | None = None, note: str = "", **extra) -> dict:
    u = usage or Usage()
    return dict(agent=agent, latency_ms=int(latency_ms), tokens_in=u.tokens_in, tokens_out=u.tokens_out,
                cost_usd=round(u.cost, 6), model=u.model, status=status, flags=flags or [], note=note, **extra)


def merge_usage(*usages: Usage) -> Usage:
    out = Usage()
    for u in usages:
        if u is None:
            continue
        out.tokens_in += u.tokens_in
        out.tokens_out += u.tokens_out
        out.cost += u.cost
        out.latency_ms += u.latency_ms
        out.model = u.model or out.model
        out.error = u.error or out.error
    return out
