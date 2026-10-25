"""RAG layer: Chroma (local, free) + pluggable embeddings.

Three collections, each with its own chunking and metadata so retrieval can be filtered:
  offers      one doc per catalog offer          (meta: offer_id, type, cost_usd, targets)
  transcripts two chunks per past call           (meta: transcript_id, reason, offer_id, outcome, chunk)
  policies    one doc per rule / playbook        (meta: kind, ref_id, reason)
We compute embeddings ourselves and pass them to Chroma, so the embedding backend is swappable and the
persisted index records which backend built it (a marker file triggers a rebuild if you switch)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .datastore import DataStore

_STOP = set("a an the and or of to in on for with is are was were be i my me you your it this that at as by from have has had not no do does did can will would so but if we our".split())


class Embedder:
    def __init__(self, backend: str):
        self.backend = backend
        self._ef = None
        self._st = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "hashing":
            return [self._hash(t) for t in texts]
        if self.backend == "sentence-transformers":
            if self._st is None:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer("all-MiniLM-L6-v2")
            return self._st.encode(texts, normalize_embeddings=True).tolist()
        if self._ef is None:
            try:
                from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
                self._ef = ONNXMiniLM_L6_V2()
                self._ef(["warm up"])  # downloads the model on first use
            except Exception as e:  # pragma: no cover - environment specific
                raise RuntimeError(
                    "Could not load the local ONNX MiniLM embedding model (needs internet once to download ~80MB). "
                    "Set EMBEDDING_BACKEND=hashing for an offline fallback. Original error: " + str(e)) from e
        return [np.asarray(v, dtype=float).tolist() for v in self._ef(texts)]

    @staticmethod
    def _hash(text: str, dim: int = 384) -> list[float]:
        toks = [t for t in re.findall(r"[a-z0-9$%]+", text.lower()) if t not in _STOP]
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        v = np.zeros(dim)
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()


@dataclass
class Hit:
    id: str
    doc: str
    meta: dict
    distance: float

    @property
    def score(self) -> float:
        return round(1.0 - self.distance, 4)


class VectorStore:
    COLLECTIONS = ("offers", "transcripts", "policies")

    def __init__(self, settings: Settings):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self.s = settings
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False))
        self.embedder = Embedder(settings.embedding_backend)

    # ---- collections
    def _make(self, name: str):
        try:
            return self.client.get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)
        except Exception:  # older chroma API
            return self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def _marker(self):
        return self.s.chroma_dir / "backend.txt"

    def is_ready(self) -> bool:
        try:
            ok = self._marker().exists() and self._marker().read_text().strip() == self.s.embedding_backend
            return ok and all(self._make(c).count() > 0 for c in self.COLLECTIONS)
        except Exception:
            return False

    def ensure(self, ds: DataStore) -> None:
        if not self.is_ready():
            self.build(ds)

    def counts(self) -> dict:
        return {c: self._make(c).count() for c in self.COLLECTIONS}

    # ---- indexing
    def build(self, ds: DataStore) -> dict:
        for c in self.COLLECTIONS:
            try:
                self.client.delete_collection(c)
            except Exception:
                pass
        self._add("offers", [
            (o["offer_id"], f"{o['name']}. {o['description']} Helps with: {', '.join(o['targets']).replace('_', ' ')}.",
             dict(offer_id=o["offer_id"], type=o["type"], cost_usd=o["cost_usd"], targets="|".join(o["targets"])))
            for o in ds.offers])
        items = []
        for t in ds.transcripts:
            for ci, turns in enumerate((t["turns"][:3], t["turns"][3:])):  # problem chunk / resolution chunk
                items.append((f"{t['transcript_id']}#{ci}", "\n".join(turns),
                              dict(transcript_id=t["transcript_id"], reason=t["reason"], offer_id=t["offer_id"],
                                   outcome=t["outcome"], chunk=ci, date=t["date"])))
        self._add("transcripts", items)
        pol = [(r["rule_id"], r["text"], dict(kind="rule", ref_id=r["rule_id"], reason="all")) for r in ds.rules]
        pol += [(p["playbook_id"], f"{p['title']}. {p['text']}", dict(kind="playbook", ref_id=p["playbook_id"], reason=p["reason"]))
                for p in ds.playbooks]
        self._add("policies", pol)
        self._marker().write_text(self.s.embedding_backend)
        return self.counts()

    def _add(self, name: str, items: list[tuple[str, str, dict]]) -> None:
        col = self._make(name)
        ids, docs, metas = zip(*items)
        embs = self.embedder.embed(list(docs))
        col.add(ids=list(ids), documents=list(docs), metadatas=list(metas), embeddings=embs)

    # ---- retrieval
    def query(self, name: str, text: str, k: int = 5, where: dict | None = None, max_distance: float | None = None) -> list[Hit]:
        col = self._make(name)
        n = min(k, col.count())
        if n == 0:
            return []
        emb = self.embedder.embed([text])
        res = col.query(query_embeddings=emb, n_results=n, where=where or None)
        hits = [Hit(i, d, m, dist) for i, d, m, dist in
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]
        if max_distance is not None:
            hits = [h for h in hits if h.distance <= max_distance]
        return hits
