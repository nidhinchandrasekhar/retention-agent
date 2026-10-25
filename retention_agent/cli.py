"""Command line: python -m retention_agent <command>"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import get_settings


def _overrides(items) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects KEY=VALUE, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _print_metrics(m: dict) -> None:
    groups = {
        "Golden set": ["golden_pass_rate", "detection_rate", "top_reason_accuracy", "acceptable_offer_rate", "forbidden_offer_rate",
                       "block_accuracy", "healthy_no_action", "injection_handled"],
        "Rules / critic": ["first_pass_rule_compliance", "final_rule_compliance", "retry_rate", "escalation_rate", "avg_overall_score",
                           "avg_groundedness", "green_share", "judge_fit_gap", "guardrail_pass_rate"],
        "Diagnostics": ["diagnostics_auc", "high_tier_precision", "high_tier_recall", "reason_accuracy_atrisk"],
        "Operations": ["latency_p50_ms", "latency_p95_ms", "avg_cost_usd"],
    }
    for g, keys in groups.items():
        print(f"\n{g}")
        for k in keys:
            print(f"  {k:<28} {m.get(k)}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="retention_agent", description="Retention agent: synthetic data, RAG, multi-agent pipeline, dashboard")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, helptext in [("generate", "generate synthetic data"), ("index", "(re)build the Chroma vector index"),
                           ("run", "run the daily job"), ("eval", "run the evaluation harness"),
                           ("demo", "generate + index + run + eval (baseline vs tuned) in one go"),
                           ("search", "ad-hoc RAG query for debugging"), ("dashboard", "launch the Streamlit dashboard"),
                           ("explain", "run the pipeline for one customer and print every step")]:
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override any setting, e.g. --set EMBEDDING_BACKEND=hashing")
        if name == "generate":
            sp.add_argument("--seed", type=int, default=42)
            sp.add_argument("--customers", type=int, default=500)
        if name in ("run",):
            sp.add_argument("--as-of", help="simulate the run date (YYYY-MM-DD, 05:00 UTC)")
            sp.add_argument("--limit", type=int)
            sp.add_argument("--label", default="daily")
        if name == "eval":
            sp.add_argument("--label", default="eval")
        if name == "search":
            sp.add_argument("query")
            sp.add_argument("--collection", default="transcripts", choices=["offers", "transcripts", "policies"])
            sp.add_argument("-k", type=int, default=5)
        if name == "explain":
            sp.add_argument("customer_id")
    a = p.parse_args(argv)
    s = get_settings(_overrides(getattr(a, "set", None)))

    if a.cmd == "generate":
        from .data_gen import generate
        print("Generated:", generate(s, a.seed, a.customers))
    elif a.cmd == "index":
        from .datastore import DataStore
        from .rag import VectorStore
        print("Indexed:", VectorStore(s).build(DataStore(s)), f"(embeddings={s.embedding_backend})")
    elif a.cmd == "run":
        from .run_daily import run_daily
        run_daily(s, as_of=a.as_of, label=a.label, limit=a.limit)
    elif a.cmd == "eval":
        from .evaluation.evaluate import run_eval
        r = run_eval(s, a.label)
        _print_metrics(r["metrics"])
        fails = [c for c in r["cases"] if not c["passed"]]
        print(f"\nGolden cases failed: {len(fails)}/{len(r['cases'])}")
        for c in fails:
            print(f"  {c['customer_id']} {c['note']}: status={c['status']} offer={c['offer']} {c['detail']}")
    elif a.cmd == "demo":
        from .data_gen import generate
        from .evaluation.evaluate import run_eval
        from .run_daily import run_daily
        print("1/4 generating synthetic data ...", generate(s))
        print("2/4 daily run (day 1) ...")
        run_daily(s, as_of=None, label="demo-day1")
        print("3/4 baseline evaluation (RAG reason filter OFF, eligibility pre-filter OFF) ...")
        base = run_eval(s.with_overrides({"use_reason_filter": "false", "prefilter_eligibility": "false"}), "baseline")
        print("4/4 tuned evaluation (defaults) ...")
        tuned = run_eval(s, "tuned")
        print("\nBaseline -> Tuned")
        for k, v in tuned["metrics"].items():
            print(f"  {k:<28} {base['metrics'].get(k)!s:>8} -> {v}")
        print("\nNow run:  python -m retention_agent dashboard")
    elif a.cmd == "search":
        from .datastore import DataStore
        from .rag import VectorStore
        store = VectorStore(s)
        store.ensure(DataStore(s))
        for h in store.query(a.collection, a.query, a.k):
            print(f"{h.score:.3f}  {h.id:<10} {h.meta}\n       {h.doc[:140]!r}")
    elif a.cmd == "explain":
        import json
        from .graph import Pipeline
        from .run_daily import prepare
        ds, store, llm = prepare(s)
        pipe = Pipeline(s, ds, store, llm)
        c, sig = ds.customer(a.customer_id), ds.signal(a.customer_id)
        if not c:
            raise SystemExit("unknown customer")
        st = pipe.run_one(c, sig, f"explain-{a.customer_id}")
        for e in st["audit_log"]:
            print(f"[{e['agent']:<11}] {e['latency_ms']:>4} ms  {e['status']:<8} {e['note']}  {e['flags']}")
        print(json.dumps(st["recommendation"], indent=2, default=str))
    elif a.cmd == "dashboard":
        app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))


if __name__ == "__main__":
    main()
