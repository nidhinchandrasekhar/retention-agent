"""Business rules engine. Rules are *data* (data/rules.json, also indexed in the vector DB so answers can
cite them) and the checks are *code* here. LLMs never decide compliance - code does."""
from __future__ import annotations

import re

from .config import Settings
from .guardrails import allowed_numbers, text_flags, untraced_numbers
from .schemas import Violation


def retry_cap(customer: dict, s: Settings) -> int:
    return s.max_offers_90d if customer["aqi"] >= 50 else 1


def customer_gate(c: dict, s: Settings) -> list[Violation]:
    v = []
    if c["holdout"]:
        v.append(Violation(rule_id="R-01", message="customer is in the holdout (control) group"))
    if not c["consent_marketing"]:
        v.append(Violation(rule_id="R-02", message="no marketing consent"))
    if c["fraud_flag"]:
        v.append(Violation(rule_id="R-03", message="fraud flag set"))
    if c["open_ticket"]:
        v.append(Violation(rule_id="R-04", message="open complaint ticket - route to care"))
    if c["aqi"] < s.min_aqi:
        v.append(Violation(rule_id="R-05", message=f"AQI {c['aqi']} below minimum {s.min_aqi}"))
    if c["segment"] != "Value Prepaid":
        v.append(Violation(rule_id="R-06", message=f"segment '{c['segment']}' out of scope"))
    cap = retry_cap(c, s)
    if c["offers_last_90d"] >= cap:
        v.append(Violation(rule_id="R-07", message=f"retry cap reached ({c['offers_last_90d']}/{cap} offers in 90 days)"))
    return v


def offer_violations(offer: dict | None, c: dict, reasons: list[str], s: Settings) -> list[Violation]:
    if offer is None or not offer.get("active", False):
        return [Violation(rule_id="R-08", message="offer not in catalog or inactive")]
    v = []
    oid = offer["offer_id"]
    if c["plan_id"] not in offer["eligible_plans"]:
        v.append(Violation(rule_id="R-09", message=f"{oid} not available on plan {c['plan_id']}"))
    if c["tenure_months"] < offer["min_tenure_months"]:
        v.append(Violation(rule_id="R-10", message=f"{oid} needs tenure >= {offer['min_tenure_months']} months (customer has {c['tenure_months']})"))
    if "ALL" not in offer["regions"] and c["region"] not in offer["regions"]:
        v.append(Violation(rule_id="R-11", message=f"{oid} not available in region {c['region']}"))
    if offer["cost_usd"] > s.max_cost_to_arpu * c["arpu_30d"]:
        v.append(Violation(rule_id="R-12", message=f"{oid} cost ${offer['cost_usd']:.0f} exceeds {s.max_cost_to_arpu}x ARPU (${c['arpu_30d']:.2f})"))
    if c["last_offer_id"] == oid and c["last_offer_accepted"] and c["last_offer_days_ago"] <= 90:
        v.append(Violation(rule_id="R-13", message=f"{oid} was already accepted {c['last_offer_days_ago']} days ago"))
    if reasons and not (set(offer["targets"]) & set(reasons)):
        v.append(Violation(rule_id="R-14", message=f"{oid} targets {offer['targets']}, not the diagnosed reasons {reasons}"))
    return v


def output_violations(decision: dict, offer: dict | None, known_refs: set[str], c: dict, evidence: list[str],
                      snippets: list[str], extra_numbers, s: Settings) -> list[Violation]:
    v = []
    texts = [decision.get("rationale", "")] + list(decision.get("talking_points", []))
    flags = text_flags(texts)
    if "pii_in_output" in flags:
        v.append(Violation(rule_id="R-15", message="personal data found in output"))
    if "competitor_disparagement" in flags or "unauthorised_promise" in flags:
        v.append(Violation(rule_id="R-16", message=f"disallowed language: {', '.join(f for f in flags if f != 'pii_in_output')}"))
    tps = decision.get("talking_points", [])
    if not (2 <= len(tps) <= 4) or any(len(t.split()) > 30 for t in tps):
        v.append(Violation(rule_id="R-17", message="need 2-4 talking points of at most 30 words"))
    cites = decision.get("citations", [])
    if not cites or any(ct not in known_refs for ct in cites):
        bad = [ct for ct in cites if ct not in known_refs]
        v.append(Violation(rule_id="R-18", message="missing citations" if not cites else f"citations not in retrieved context: {bad}"))
    allowed = allowed_numbers(offer, c, evidence, snippets, extra_numbers)
    bad_nums = untraced_numbers(texts, allowed)
    if bad_nums:
        v.append(Violation(rule_id="R-19", message=f"numbers not traceable to sources: {bad_nums}"))
    if offer and offer["valid_hours"] > s.offer_ttl_hours:
        v.append(Violation(rule_id="R-20", message=f"offer validity {offer['valid_hours']}h exceeds {s.offer_ttl_hours}h"))
    if offer and offer["cost_usd"] > s.approval_limit_usd:
        v.append(Violation(rule_id="R-21", message=f"cost ${offer['cost_usd']:.0f} above ${s.approval_limit_usd:.0f} limit - human approval required", severity="gate"))
    return v


def hard(violations: list[Violation]) -> list[Violation]:
    return [x for x in violations if x.severity == "hard"]
