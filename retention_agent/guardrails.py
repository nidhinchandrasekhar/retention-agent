"""Deterministic guardrails: input sanitising, PII masking, prompt-injection stripping,
number tracing (hallucination check) and scope checks. No LLM involved - these must be cheap and reliable."""
from __future__ import annotations

import re

PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){10,13}(?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
INJECTION_RE = re.compile(
    r"(ignore (all |any )?(previous|prior|above) (instructions|rules)|disregard (the )?(rules|instructions)|"
    r"system prompt|you are now|act as|reveal your|jailbreak|override (the )?polic)", re.I)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

DISPARAGE = ["scam", "rip-off", "rip off", "useless", "terrible network", "worst operator", "they cheat", "their network is bad"]
UNAUTHORISED = ["guarantee", "guaranteed", "for life", "forever", "permanent", "unlimited", "no questions asked", "lifetime"]


def has_pii(text: str) -> bool:
    return bool(PHONE_RE.search(text) or EMAIL_RE.search(text))


def mask_pii(text: str) -> tuple[str, bool]:
    new = EMAIL_RE.sub("[email]", PHONE_RE.sub("[phone]", text))
    return new, new != text


def sanitize_text(text: str) -> tuple[str, list[str]]:
    """Strip prompt-injection sentences and mask PII in free text coming from upstream systems."""
    flags: list[str] = []
    if not text:
        return "", flags
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        if INJECTION_RE.search(s):
            flags.append("prompt_injection_removed")
        else:
            kept.append(s)
    clean, masked = mask_pii(" ".join(kept))
    if masked:
        flags.append("pii_masked")
    return clean.strip(), flags


def extract_numbers(text: str) -> set[float]:
    return {float(x) for x in NUM_RE.findall(text or "")}


def _add_value(bag: set[float], v) -> None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return
    bag.update({f, float(round(f)), round(f, 1), round(f, 2)})


def allowed_numbers(offer: dict | None, customer: dict, evidence: list[str], snippets: list[str], extra=()) -> set[float]:
    """Every number the model may legitimately mention."""
    bag: set[float] = set()
    if offer:
        for k in ("cost_usd", "valid_hours", "min_tenure_months"):
            _add_value(bag, offer.get(k))
        bag |= extract_numbers(offer.get("benefit_text", "")) | extract_numbers(offer.get("name", ""))
    for v in customer.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            _add_value(bag, v)
    for t in list(evidence) + list(snippets):
        bag |= extract_numbers(t)
    for v in extra:
        _add_value(bag, v)
    return bag


def untraced_numbers(texts: list[str], allowed: set[float]) -> list[float]:
    found: set[float] = set()
    for t in texts:
        # ignore citation tokens such as [offer:OF-04] / [call:T-012]
        found |= extract_numbers(re.sub(r"\[[a-z]+:[^\]]+\]", "", t))
    return sorted(n for n in found if n not in allowed)


def text_flags(texts: list[str]) -> list[str]:
    joined = " ".join(texts).lower()
    flags = []
    if has_pii(joined):
        flags.append("pii_in_output")
    if any(p in joined for p in DISPARAGE):
        flags.append("competitor_disparagement")
    if any(p in joined for p in UNAUTHORISED):
        flags.append("unauthorised_promise")
    return flags
