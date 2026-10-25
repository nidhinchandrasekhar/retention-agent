"""Loads the synthetic 'source systems' from disk (CRM/billing features, RTS signals, catalog, rules...)."""
from __future__ import annotations

import json
from functools import cached_property

import pandas as pd

from .config import Settings
from .utils import read_json


class DataStore:
    def __init__(self, settings: Settings):
        self.s = settings

    def exists(self) -> bool:
        return self.s.data_path("customers.csv").exists()

    def _csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.s.data_path(name), keep_default_na=False)

    @cached_property
    def customers(self) -> pd.DataFrame:
        return self._csv("customers.csv")

    @cached_property
    def signals(self) -> pd.DataFrame:
        df = self._csv("signals.csv")
        return df

    @cached_property
    def labels(self) -> pd.DataFrame:
        return self._csv("labels.csv")

    @cached_property
    def offers(self) -> list[dict]:
        return read_json(self.s.data_path("offers.json"))

    @cached_property
    def rules(self) -> list[dict]:
        return read_json(self.s.data_path("rules.json"))

    @cached_property
    def playbooks(self) -> list[dict]:
        return read_json(self.s.data_path("playbooks.json"))

    @cached_property
    def transcripts(self) -> list[dict]:
        out = []
        with open(self.s.data_path("transcripts.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out

    @cached_property
    def golden(self) -> list[dict]:
        return read_json(self.s.data_path("golden_set.json"))

    def offer(self, offer_id: str) -> dict | None:
        return next((o for o in self.offers if o["offer_id"] == offer_id), None)

    def customer(self, customer_id: str) -> dict | None:
        df = self.customers
        row = df[df["customer_id"] == customer_id]
        return None if row.empty else _clean(row.iloc[0].to_dict())

    def signal(self, customer_id: str) -> dict | None:
        df = self.signals
        row = df[df["customer_id"] == customer_id]
        if row.empty:
            return None
        d = _clean(row.iloc[0].to_dict())
        d["intents"] = [i for i in str(d.get("intents", "")).split("|") if i]
        return d

    def customer_rows(self) -> list[dict]:
        return [_clean(r) for r in self.customers.to_dict(orient="records")]


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            v = v.item()
        out[k] = v
    return out
