"""Hybrid retrieval for the AML entry: dense + TF-IDF with RRF fusion.

Why hybrid: TF-IDF is lexical (misses paraphrases, cross-lingual matches).
Dense embeddings catch semantics ("CEO" ~ "首席执行官", "Acme" ~ "这家公司").
RRF (Reciprocal Rank Fusion) combines both rankings without score
normalization headaches.

The dense model is local and free (fastembed ONNX, no API costs).
If the model can't load, falls back to TF-IDF alone — never crashes.

Kept separate from retrieval.py (pure TF-IDF) so the model-free
story stays intact; the AML service selects via RETRIEVAL_MODE env.
"""

from __future__ import annotations

import math
import os

from veritas.store import BeliefStore

try:
    from .retrieval import BeliefRetriever
except ImportError:
    from retrieval import BeliefRetriever

RETRIEVAL_MODE = os.environ.get("VERITAS_RETRIEVAL", "hybrid")
_DENSE_MODEL = os.environ.get(
    "VERITAS_DENSE_MODEL", "jinaai/jina-embeddings-v2-small-en")


class HybridRetriever:
    """Dense + TF-IDF hybrid over active Veritas beliefs."""

    def __init__(self, store: BeliefStore) -> None:
        self.store = store
        self.tfidf = BeliefRetriever(store)
        self._dense = None
        self._dense_ids: list[str] = []
        self._dense_vecs = None
        self._dense_gen: int = -1
        if RETRIEVAL_MODE == "hybrid":
            self._init_dense()

    def _init_dense(self) -> None:
        try:
            from fastembed import TextEmbedding
            self._dense = TextEmbedding(_DENSE_MODEL)
        except Exception as e:
            # Model unavailable: TF-IDF alone. Log, don't crash.
            print(f"[hybrid] dense model unavailable ({e}); TF-IDF only")
            self._dense = None

    @property
    def dense_active(self) -> bool:
        return self._dense is not None

    def _rebuild_dense(self) -> None:
        import numpy as np

        beliefs = self.store.active_beliefs()
        self._dense_ids = [b.id for b in beliefs]
        texts = [b.proposition for b in beliefs]
        if texts and self._dense is not None:
            vecs = list(self._dense.embed(texts))
            mat = np.array(vecs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._dense_vecs = mat / norms
        else:
            self._dense_vecs = None
        self._dense_gen = self.store.generation

    def _maybe_rebuild_dense(self) -> None:
        if self._dense is None:
            return
        if self.store.generation != self._dense_gen:
            self._rebuild_dense()

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return [(belief_id, cosine)] ranked."""
        import numpy as np

        self._maybe_rebuild_dense()
        if self._dense is None or self._dense_vecs is None:
            return []
        if not self._dense_ids:
            return []
        qv = np.array(
            list(self._dense.embed([query]))[0], dtype=np.float32)
        n = np.linalg.norm(qv)
        if n == 0:
            return []
        qv = qv / n
        sims = self._dense_vecs @ qv
        order = np.argsort(-sims)[:top_k]
        return [(self._dense_ids[i], float(sims[i])) for i in order]

    def search(
        self,
        query: str,
        top_k: int = 5,
        as_of: float | None = None,
    ) -> list[dict]:
        """Hybrid search with RRF fusion.

        as_of: temporal filter applied after fusion.
        Returns evidence dicts (same shape as BeliefRetriever.search).
        """
        # TF-IDF ranking (over-fetch for fusion).
        tfidf_hits = self.tfidf.search(query, top_k=top_k * 3, as_of=None)
        tfidf_rank = {h["text"]: i for i, h in enumerate(tfidf_hits)}

        # Dense ranking.
        dense_hits = self._dense_search(query, top_k * 3)
        by_id = {b.id: b for b in self.store.active_beliefs()}
        dense_rank = {}
        for i, (did, _sim) in enumerate(dense_hits):
            b = by_id.get(did)
            if b is not None:
                dense_rank[b.proposition] = i

        # RRF fusion: score = sum(1 / (k + rank)).
        k = 60.0
        fused: dict[str, float] = {}
        for text, rank in tfidf_rank.items():
            fused[text] = fused.get(text, 0.0) + 1.0 / (k + rank + 1)
        for text, rank in dense_rank.items():
            fused[text] = fused.get(text, 0.0) + 1.0 / (k + rank + 1)

        # Resolve back to belief dicts, apply temporal filter.
        text_to_hit = {h["text"]: h for h in tfidf_hits}
        for did, _ in dense_hits:
            b = by_id.get(did)
            if b is not None and b.proposition not in text_to_hit:
                text_to_hit[b.proposition] = {
                    "text": b.proposition,
                    "score": 0.0,
                    "entrenchment": round(self.store.entrenchment_of(b), 4),
                    "timestamp": b.timestamp,
                    "source": b.source,
                    "confidence": b.confidence,
                }
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for text, fscore in ranked[:top_k]:
            h = text_to_hit.get(text)
            if h is None:
                continue
            if as_of is not None:
                b = next(
                    (x for x in by_id.values() if x.proposition == text),
                    None,
                )
                if b is not None:
                    vf = (b.valid_from if b.valid_from is not None
                          else float("-inf"))
                    vu = (b.valid_until if b.valid_until is not None
                          else float("inf"))
                    if not (vf <= as_of <= vu):
                        continue
            h = dict(h)
            h["score"] = round(fscore, 4)
            h["fusion"] = "rrf"
            out.append(h)
        return out
