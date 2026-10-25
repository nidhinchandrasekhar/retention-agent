"""Central configuration. Every field can be set via environment variable (UPPERCASE name),
a .env file, or a CLI override (--set KEY=VALUE) so experiments are reproducible."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _cast(value: str, like):
    if isinstance(like, bool):
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, int):
        return int(float(value))
    if isinstance(like, float):
        return float(value)
    return str(value)


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    llm_provider: str = "auto"            # auto | openai | mock
    openai_api_key: str = ""
    openai_model_small: str = "gpt-4o-mini"   # routing / reflection / judge
    openai_model_large: str = "gpt-4o"        # decision reasoning
    # --- embeddings / vector DB ---
    embedding_backend: str = "onnx"       # onnx | hashing | sentence-transformers
    retrieval_top_k: int = 5
    max_distance: float = 0.85            # cosine distance above this = "no hit"
    use_reason_filter: bool = True        # metadata-filter RAG by diagnosed reason
    prefilter_eligibility: bool = True    # only offer eligible candidates to Decision
    # --- business rules ---
    risk_threshold: float = 0.6           # >= this -> full pipeline (high tier)
    watch_threshold: float = 0.35         # >= this -> monitor (medium tier)
    approval_limit_usd: float = 25.0      # offers costing more need human approval
    offer_ttl_hours: int = 24
    max_cost_to_arpu: float = 2.0         # offer cost <= this x 30d ARPU
    max_offers_90d: int = 2               # retry cap (1 if AQI < 50)
    min_aqi: int = 30                     # below = low-quality acquisition, no retention spend
    # --- orchestration ---
    max_retries: int = 2                  # critic loop cap
    enable_reflection: bool = True
    # --- paths ---
    data_dir: str = "data"
    output_dir: str = "outputs"

    # ---- derived ----
    @property
    def use_llm(self) -> bool:
        if self.llm_provider == "mock":
            return False
        return bool(self.openai_api_key)

    @property
    def mode(self) -> str:
        return "openai" if self.use_llm else "mock"

    @property
    def db_path(self) -> Path:
        return Path(self.output_dir) / "retention.db"

    @property
    def chroma_dir(self) -> Path:
        return Path(self.data_dir) / "chroma"

    def data_path(self, name: str) -> Path:
        return Path(self.data_dir) / name

    # ---- builders ----
    @classmethod
    def from_env(cls) -> "Settings":
        base = cls()
        kwargs = {}
        for f in fields(cls):
            raw = os.getenv(f.name.upper())
            if raw not in (None, ""):
                kwargs[f.name] = _cast(raw, getattr(base, f.name))
        return replace(base, **kwargs)

    def with_overrides(self, overrides: dict | None) -> "Settings":
        if not overrides:
            return self
        names = {f.name for f in fields(self)}
        kwargs = {}
        for k, v in overrides.items():
            k = k.lower()
            if k not in names:
                raise ValueError(f"Unknown setting '{k}'. Valid: {sorted(names)}")
            kwargs[k] = _cast(v, getattr(self, k)) if isinstance(v, str) else v
        return replace(self, **kwargs)

    def public_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["openai_api_key"] = "***" if self.openai_api_key else ""
        d["mode"] = self.mode
        return d


def get_settings(overrides: dict | None = None) -> Settings:
    return Settings.from_env().with_overrides(overrides)
