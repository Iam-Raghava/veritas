"""Hybrid retrieval for the AML entry: dense + TF-IDF with RRF fusion.

Why hybrid: TF-IDF is lexical (misses paraphrases, cross-lingual matches).
Dense embeddings catch semantics ("CEO" ~ "首席执行官", "Acme" ~ "这家公司").
RRF (Reciprocal Rank Fusion) combines both rankings without score
normalization headaches.

The dense model is local ONNX (aml/dense.py) — no API costs, no torch.
Set VERITAS_DENSE_DIR to the directory with model.onnx + tokenizer.json.
If unset or unloadable, falls back to TF-IDF alone — never crashes.

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


class HybridRetriever:
    """Dense + TF-IDF hybrid over active Veritas beliefs."""

    def __init__(self, store: BeliefStore) -> None:
        self.store = store
        self.tfidf = BeliefRetriever(store)
        self._dense = None
        self._dense_ids: list[str] = []
        self._dense_vecs = None
        self._dense_gen: int = -1
        self._reranker = None
        if RETRIEVAL_MODE == "hybrid":
            self._init_dense()
            self._init_reranker()

    def _init_reranker(self) -> None:
        try:
            from rerank import Reranker
        except ImportError:
            from aml.rerank import Reranker
        self._reranker = Reranker.load()

    @property
    def rerank_active(self) -> bool:
        return self._reranker is not None

    def _init_dense(self) -> None:
        try:
            from dense import DenseEmbedder
        except ImportError:
            from aml.dense import DenseEmbedder
        self._dense = DenseEmbedder.load()
        if self._dense is None:
            print("[hybrid] no dense model (set VERITAS_DENSE_DIR); "
                  "TF-IDF only")

    @property
    def dense_active(self) -> bool:
        return self._dense is not None

    def _rebuild_dense(self) -> None:
        import numpy as np

        beliefs = self.store.active_beliefs()
        active_ids = {b.id for b in beliefs}
        # Drop vectors for retracted beliefs.
        keep = [i for i, did in enumerate(self._dense_ids)
                if did in active_ids]
        if keep and self._dense_vecs is not None:
            self._dense_ids = [self._dense_ids[i] for i in keep]
            self._dense_vecs = self._dense_vecs[keep]
        else:
            self._dense_ids = []
            self._dense_vecs = None
        # Embed only NEW beliefs (incremental, not O(N) every time).
        known = set(self._dense_ids)
        new_beliefs = [b for b in beliefs if b.id not in known]
        if new_beliefs and self._dense is not None:
            mat = self._dense.embed(
                [b.proposition for b in new_beliefs]).astype(np.float32)
            self._dense_ids.extend(b.id for b in new_beliefs)
            self._dense_vecs = (mat if self._dense_vecs is None
                                else np.vstack([self._dense_vecs, mat]))
        self._dense_gen = self.store.generation

    def _maybe_rebuild_dense(self) -> None:
        if self._dense is None:
            return
        if self.store.generation != self._dense_gen:
            self._rebuild_dense()

    def _expand_query(self, query: str, top_docs: int = 3,
                      max_terms: int = 5) -> str:
        """Pseudo-relevance feedback: expand query with distinctive terms
        from top TF-IDF hits. Helps when the query uses different words
        than the documents ("CEO" vs "chief executive")."""
        from collections import Counter

        try:
            from retrieval import _tokens as _tok
        except ImportError:
            from aml.retrieval import _tokens as _tok

        hits = self.tfidf.search(query, top_k=top_docs)
        if not hits:
            return query
        query_toks = set(_tok(query))
        term_scores: Counter = Counter()
        for h in hits:
            for tok in _tok(h["text"]):
                if tok not in query_toks and not tok.startswith("cjk:"):
                    # Weight by TF-IDF score of the doc it came from.
                    term_scores[tok] += h["score"]
        # Take top distinctive terms.
        extra = [t for t, _ in term_scores.most_common(max_terms)
                 if len(t) > 2]
        if extra:
            return query + " " + " ".join(extra)
        return query

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return [(belief_id, cosine)] ranked."""
        import numpy as np

        self._maybe_rebuild_dense()
        if self._dense is None or self._dense_vecs is None:
            return []
        if not self._dense_ids:
            return []
        qv = self._dense.embed([query])[0].astype(np.float32)
        sims = self._dense_vecs @ qv
        order = np.argsort(-sims)[:top_k]
        return [(self._dense_ids[i], float(sims[i])) for i in order]

    def search(
        self,
        query: str,
        top_k: int = 5,
        as_of: float | None = None,
    ) -> list[dict]:
        """Hybrid search with RRF fusion + rerank.

        as_of: temporal filter applied after fusion.
        Returns evidence dicts (same shape as BeliefRetriever.search).
        """
        # Query expansion via pseudo-relevance feedback (helps vocabulary
        # mismatch). Disabled via VERITAS_EXPAND=0.
        search_query = query
        if os.environ.get("VERITAS_EXPAND", "1") == "1":
            search_query = self._expand_query(query)

        # TF-IDF ranking (over-fetch for fusion).
        tfidf_hits = self.tfidf.search(search_query, top_k=top_k * 3,
                                       as_of=None)
        tfidf_rank = {h["text"]: i for i, h in enumerate(tfidf_hits)}

        # Dense ranking.
        dense_hits = self._dense_search(search_query, top_k * 3)
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

        # Cross-encoder rerank: rescore top candidates for precision.
        fusion = "rrf"
        if self._reranker is not None and ranked:
            try:
                from rerank import RERANK_TOPK
            except ImportError:
                from aml.rerank import RERANK_TOPK
            cands = [t for t, _ in ranked[:RERANK_TOPK]]
            try:
                scores = self._reranker.score(query, cands)
                rescored = sorted(zip(cands, scores),
                                  key=lambda kv: -kv[1])
                ranked = [(t, float(s)) for t, s in rescored] + ranked[RERANK_TOPK:]
                fusion = "rrf+rerank"
            except Exception as e:
                print(f"[hybrid] rerank failed ({e}); RRF ranking kept")

        out = []
        seen_texts: set[str] = set()
        for text, fscore in ranked[:top_k * 2]:  # over-fetch for dedup
            h = text_to_hit.get(text)
            if h is None:
                continue
            # Deduplicate near-identical texts.
            norm = text.lower().strip()
            if norm in seen_texts:
                continue
            seen_texts.add(norm)
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
            h["fusion"] = fusion
            out.append(h)
            if len(out) >= top_k:
                break
        return out
