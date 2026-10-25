"""Thin OpenAI wrapper with model routing, cost tracking and graceful degradation.

If no key is configured (or a call fails) callers get (None, usage) and fall back to the
deterministic path - so the whole system runs offline in "mock" mode, which is what CI uses."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .config import Settings

# USD per 1M tokens (input, output) - approximate list prices; edit if they change.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    model: str = ""
    error: str = ""


class LLMClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.enabled = settings.use_llm
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.s.openai_api_key)
        return self._client

    def model_for(self, tier: str) -> str:
        return self.s.openai_model_large if tier == "large" else self.s.openai_model_small

    def chat_json(self, system: str, user: str, tier: str = "small", max_tokens: int = 700):
        """Returns (parsed_json | None, Usage). Never raises."""
        if not self.enabled:
            return None, Usage(error="llm_disabled")
        model = self.model_for(tier)
        t0 = time.perf_counter()
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=max_tokens,
        )
        try:
            try:
                r = self.client.chat.completions.create(**kwargs)
            except Exception as e:  # some models reject temperature
                if "temperature" in str(e):
                    kwargs.pop("temperature")
                    r = self.client.chat.completions.create(**kwargs)
                else:
                    raise
            data = json.loads(r.choices[0].message.content)
            ti, to = r.usage.prompt_tokens, r.usage.completion_tokens
            pi, po = PRICES.get(model, (1.0, 3.0))
            return data, Usage(ti, to, (ti * pi + to * po) / 1e6, int((time.perf_counter() - t0) * 1000), model)
        except Exception as e:  # network, auth, bad JSON...
            return None, Usage(latency_ms=int((time.perf_counter() - t0) * 1000), model=model, error=str(e)[:200])
