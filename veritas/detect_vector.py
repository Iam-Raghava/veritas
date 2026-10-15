"""Vector pre-filter for expensive contradiction detectors.

Running NLI or LLM-as-a-judge across thousands of beliefs is O(N) per
assertion. This module uses dense embeddings (BGE-M3 via ONNX, from the
veritas.dense embedder) to retrieve the top-K semantically similar candidates, then
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

import logging
import os
from typing import TYPE_CHECKING

_logger = logging.getLogger(__name__)

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
            # No hardcoded fallback: user must specify dense_dir or
            # VERITAS_DENSE_DIR. Fails clearly if not provided.
        )
        if not self._dense_dir:
            raise ValueError(
                "VectorPrefilterDetector requires dense_dir or "
                "VERITAS_DENSE_DIR environment variable. "
                "No default model path is assumed."
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
                from veritas.dense import DenseEmbedder
                self._embedder = DenseEmbedder(self._dense_dir)
            except Exception as e:
                # No embedder available; fall back to brute force.
                _logger.warning(
                    "vector_prefilter embedder unavailable (%s); using brute force",
                    e,
                )
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
        # Evict retracted beliefs from cache (embeddings are immutable
        # per belief ID — proposition text never changes — so we only
        # need to drop IDs that are no longer active, not wipe the
        # entire cache on every write).
        if self._store is not None:
            active_ids = {b.id for b in self._store.active_beliefs()}
            # Also keep the query belief's own ID out (it's excluded below).
            stale = [bid for bid in self._emb_cache if bid not in active_ids]
            for bid in stale:
                del self._emb_cache[bid]
        self._cache_gen = self._store._ent_gen if self._store else -1
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
        # Temporal filter for consistency with find_contradictors.
        if not a.temporally_overlaps(b):
            return False
        return self.base.contradicts(a, b)

    def find_contradictors(
        self, belief: Belief, candidates: list[Belief] | None = None
    ) -> list[Belief]:
        """Find contradictors using vector pre-filter."""
        if candidates is None:
            candidates = self._candidates(belief)
        # Temporal filter: the base detector does not check validity
        # intervals; without this, non-overlapping claims (e.g., "was CEO
        # in 2020" vs "is CEO in 2026") would be flagged as contradictions.
        return [c for c in candidates
                if belief.temporally_overlaps(c)
                and self.base.contradicts(belief, c)]
