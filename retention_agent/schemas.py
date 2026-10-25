"""Pydantic contracts = the validated handoffs between agents (one role, one job, one output contract)."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

Reason = Literal[
    "price_sensitivity", "bill_shock", "network_quality",
    "competitor_offer", "disengaged", "plan_mismatch",
]
REASONS: list[str] = list(get_args(Reason))
INTENTS = ["competitor_mention", "bill_shock", "network", "price", "cancel_intent", "none"]
Tier = Literal["high", "medium", "low"]
Badge = Literal["green", "amber", "red"]


# ------------------------------------------------------------------ inputs
class CustomerRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    customer_id: str = Field(pattern=r"^C\d{4,6}$")
    segment: str
    plan_id: str
    tenure_months: int = Field(ge=0, le=600)
    region: str
    arpu_30d: float = Field(ge=0, le=1000)
    arpu_prev_90d_avg: float = Field(ge=0, le=1000)
    recharge_count_30d: int = Field(ge=0, le=60)
    recharge_count_prev_avg: float = Field(ge=0, le=60)
    days_since_last_recharge: int = Field(ge=0, le=3650)
    data_gb_30d: float = Field(ge=0)
    data_gb_prev_avg: float = Field(ge=0)
    plan_data_gb: float = Field(ge=0)
    network_complaints_30d: int = Field(ge=0)
    dropped_call_rate: float = Field(ge=0, le=1)
    unexpected_deductions_30d: int = Field(ge=0)
    downgrade_attempts_90d: int = Field(ge=0)
    port_out_inquiry: bool
    offers_last_90d: int = Field(ge=0)
    last_offer_id: str = ""
    last_offer_accepted: bool = False
    last_offer_days_ago: int = 999
    consent_marketing: bool
    holdout: bool
    fraud_flag: bool
    open_ticket: bool
    aqi: int = Field(ge=0, le=100)


class RTSSignal(BaseModel):
    """Output of the (upstream) real-time speech model for the customer's latest call."""
    model_config = ConfigDict(extra="ignore")
    customer_id: str
    call_id: str
    ts: str
    churn_propensity: float = Field(ge=0, le=1)
    intents: list[str] = Field(default_factory=list)
    sentiment: float = Field(default=0.0, ge=-1, le=1)
    call_snippet: str = Field(default="", max_length=600)

    @field_validator("intents")
    @classmethod
    def _known(cls, v):
        return [i for i in v if i in INTENTS]


# ------------------------------------------------------------------ agent outputs
class ReasonScore(BaseModel):
    code: Reason
    weight: float = Field(ge=0, le=1)
    evidence: str


class DiagnosticsOutput(BaseModel):
    customer_id: str
    risk_score: float = Field(ge=0, le=1)
    tier: Tier
    reasons: list[ReasonScore] = Field(default_factory=list)
    rts_used: bool = False
    act: bool = False
    note: str = ""

    @property
    def primary(self) -> Optional[str]:
        return self.reasons[0].code if self.reasons else None

    @property
    def secondary(self) -> Optional[str]:
        return self.reasons[1].code if len(self.reasons) > 1 else None


class Citation(BaseModel):
    ref: str            # e.g. offer:OF-04, call:T-012, rule:R-12, playbook:PB-03
    source: str         # offer | call | rule | playbook
    id: str
    score: float = 0.0  # similarity (1 - cosine distance); 1.0 for exact lookups
    snippet: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    offer_id: str
    name: str
    cost_usd: float
    semantic_score: float
    rank: int
    citation: str


class ContextBundle(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list)
    similar_calls: list[Citation] = Field(default_factory=list)
    policies: list[Citation] = Field(default_factory=list)
    offer_citations: list[Citation] = Field(default_factory=list)
    acceptance_by_offer: dict[str, float] = Field(default_factory=dict)
    acceptance_counts: dict[str, list[int]] = Field(default_factory=dict)  # offer_id -> [accepted, total]
    query: str = ""
    no_hits: bool = False
    stale_flags: list[str] = Field(default_factory=list)

    def all_refs(self) -> set[str]:
        refs = {c.ref for c in self.similar_calls + self.policies + self.offer_citations}
        refs |= {c.citation for c in self.candidates}
        return refs


class DecisionOutput(BaseModel):
    action: Literal["offer", "escalate", "no_action"]
    offer_id: Optional[str] = None
    offer_name: str = ""
    rationale: str = ""
    talking_points: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    fallback_offer_id: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    method: str = "deterministic"   # deterministic | llm


class Violation(BaseModel):
    rule_id: str
    message: str
    severity: str = "hard"


class ValidationReport(BaseModel):
    rule_pass: bool
    violations: list[Violation] = Field(default_factory=list)
    requires_approval: bool = False
    groundedness: float = 0.0
    fit: float = 0.0
    actionability: float = 0.0
    overall: float = 0.0
    badge: Badge = "red"
    feedback: str = ""
    judge_method: str = "heuristic"
    fit_heuristic: float = 0.0
    fit_llm: Optional[float] = None


class Recommendation(BaseModel):
    customer_id: str
    status: str
    offer_id: Optional[str] = None
    offer_name: str = ""
    offer_cost: float = 0.0
    fallback_offer_id: Optional[str] = None
    badge: Optional[str] = None
    overall_score: Optional[float] = None
    risk_score: float = 0.0
    tier: str = "low"
    reasons: list[dict] = Field(default_factory=list)
    talking_points: list[str] = Field(default_factory=list)
    rationale: str = ""
    citations: list[str] = Field(default_factory=list)
    scores: dict[str, Any] = Field(default_factory=dict)
    skip_reason: str = ""
    retries: int = 0


# ------------------------------------------------------------------ graph state
class GraphState(TypedDict, total=False):
    customer: dict
    signal: Optional[dict]
    diagnostics: dict
    context: dict
    decision: dict
    validation: dict
    feedback: str
    excluded_offers: list[str]
    retries: int
    retry: bool
    blocked: bool
    block_reasons: list[str]
    recommendation: dict
    flags: Annotated[list[str], operator.add]
    audit_log: Annotated[list[dict], operator.add]
