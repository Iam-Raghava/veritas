"""Vector pre-filter for expensive contradiction detectors.

Running NLI or LLM-as-a-judge across thousands of beliefs is O(N) per
assertion. This module uses dense embeddings (BGE-M3 via ONNX, from the
AML stack) to retrieve the top-K semantically similar candidates, then
runs the expensive detector only on those.

This reduces contradiction search from O(N) to O(K) where K << N.

Usage:
    from veritas.detect_vector import VectorPrefilterDetector
    from veritas.detect_nli import NLIDetector

    base = NLIDetector()  # expensive
    detector = VectorPrefilterDetector(base, top_k=20)
    store = BeliefStore(detector=detector)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.belief import Belief
    from veritas.detect import Detector


class VectorPrefilterDetector:
    """Wraps an expensive detector with vector pre-filtering.

    For each incoming belief, embeds the proposition, finds top-K most
    similar active beliefs via cosine similarity, and only runs the
    wrapped detector on those candidates.

    Falls back to brute force if the embedder is unavailable.
    """

    name = "vector_prefilter"

    def __init__(
        self,
        base_detector: Detector,
        top_k: int = 20,
        dense_dir: str | None = None,
        similarity_threshold: float = 0.0,
    ) -> None:
        self.base = base_detector
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self._embedder = None
        self._dense_dir = (
            dense_dir
            or os.environ.get("VERITAS_DENSE_DIR")
            or os.path.expanduser("~/workspace/models/bge-m3")
        )
        # Cache: belief_id -> embedding. Invalidated on store mutations
        # via the generation counter (set by attach_store).
        self._emb_cache: dict[str, object] = {}
        self._cache_gen: int = -1
        self._store = None

    def attach_store(self, store) -> None:
        """Attach to a BeliefStore for candidate retrieval."""
        self._store = store

    def _get_embedder(self):
        """Lazy-load the ONNX embedder."""
        if self._embedder is None:
            try:
                import sys
                sys.path.insert(0, os.path.join(
                    os.path.dirname(__file__), "..", "aml"))
                from dense import DenseEmbedder
                self._embedder = DenseEmbedder(self._dense_dir)
            except Exception as e:
                # No embedder available; fall back to brute force.
                print(f"[vector_prefilter] embedder unavailable ({e}); "
                      f"using brute force")
                self._embedder = False
        return self._embedder if self._embedder is not False else None

    def _candidates(self, belief: Belief) -> list[Belief]:
        """Top-K semantically similar active beliefs."""
        if self._store is None:
            return []
        embedder = self._get_embedder()
        if embedder is None:
            # Fallback: all active beliefs (brute force).
            return [b for b in self._store.active_beliefs()
                    if b.id != belief.id]
        import numpy as np
        # Check cache generation.
        gen = self._store._ent_gen
        if gen != self._cache_gen:
            self._emb_cache.clear()
            self._cache_gen = gen
        # Embed query.
        q_emb = embedder.embed([belief.proposition])[0]
        # Embed candidates (cached).
        candidates = [b for b in self._store.active_beliefs()
                      if b.id != belief.id]
        if not candidates:
            return []
        texts = []
        to_embed_idx = []
        for i, b in enumerate(candidates):
            if b.id in self._emb_cache:
                continue
            texts.append(b.proposition)
            to_embed_idx.append(i)
        if texts:
            embs = embedder.embed(texts)
            for idx, emb in zip(to_embed_idx, embs):
                self._emb_cache[candidates[idx].id] = emb
        # Cosine similarity.
        q_norm = np.linalg.norm(q_emb)
        scored = []
        for b in candidates:
            emb = self._emb_cache.get(b.id)
            if emb is None:
                continue
            sim = float(np.dot(q_emb, emb) /
                        (q_norm * np.linalg.norm(emb) + 1e-9))
            if sim >= self.similarity_threshold:
                scored.append((sim, b))
        scored.sort(key=lambda x: -x[0])
        return [b for _, b in scored[:self.top_k]]

    def contradicts(self, a: Belief, b: Belief) -> bool:
        """Delegate to base detector (for pairwise checks)."""
        return self.base.contradicts(a, b)

    def find_contradictors(
        self, belief: Belief, candidates: list[Belief] | None = None
    ) -> list[Belief]:
        """Find contradictors using vector pre-filter."""
        if candidates is None:
            candidates = self._candidates(belief)
        return [c for c in candidates
                if self.base.contradicts(belief, c)]
