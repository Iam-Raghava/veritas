"""Model-free TF-IDF retrieval over active Veritas beliefs.

Deliberately no embeddings and no LLM: the AML entry runs the heuristic
(model-free) detector, so the retrieval layer stays model-free too.
Only ACTIVE beliefs are indexed — retracted (outdated/contradicted)
beliefs never surface. That is the governance story: the platform gets
current valid evidence, not stale context.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from veritas.store import BeliefStore

_WORD = re.compile(r"[a-z0-9]+")

# CJK character ranges: tokenize as character bigrams (no word
# boundaries in Chinese/Japanese/Korean running text).
_CJK = re.compile(
    "["
    "\u4e00-\u9fff"      # CJK Unified Ideographs
    "\u3400-\u4dbf"      # CJK Extension A
    "\u3040-\u30ff"      # Hiragana + Katakana
    "\uac00-\ud7af"      # Hangul syllables
    "]"
)


def _tokens(text: str) -> list[str]:
    """Tokenize for TF-IDF: alphanumeric words + CJK character bigrams.

    Chinese/Japanese/Korean have no spaces; character bigrams give
    lexical overlap where word segmentation is unavailable.
    """
    lowered = text.lower()
    toks = _WORD.findall(lowered)
    # CJK bigrams: sliding window over consecutive CJK chars.
    cjk_chars = _CJK.findall(lowered)
    toks.extend(
        "cjk:" + cjk_chars[i] + cjk_chars[i + 1]
        for i in range(len(cjk_chars) - 1)
    )
    return toks


class BeliefRetriever:
    """TF-IDF search over a BeliefStore's active beliefs."""

    def __init__(self, store: BeliefStore) -> None:
        self.store = store
        self._doc_ids: list[str] = []
        self._tf: list[Counter] = []
        self._idf: dict[str, float] = {}
        self._built_gen: int = -1
        self.rebuild()

    def rebuild(self) -> None:
        beliefs = self.store.active_beliefs()
        self._doc_ids = [b.id for b in beliefs]
        self._tf = [Counter(_tokens(b.proposition)) for b in beliefs]
        n = max(len(beliefs), 1)
        df: Counter = Counter()
        for tf in self._tf:
            for tok in tf:
                df[tok] += 1
        self._idf = {
            tok: math.log((1 + n) / (1 + c)) + 1.0 for tok, c in df.items()
        }
        self._built_gen = self.store.generation

    def _maybe_rebuild(self) -> None:
        if self.store.generation != self._built_gen:
            self.rebuild()

    def search(
        self,
        query: str,
        top_k: int = 5,
        as_of: float | None = None,
    ) -> list[dict]:
        """Return top-k evidence dicts for query.

        as_of: only beliefs temporally valid at that timestamp.
        Results ranked by TF-IDF cosine; ties break by entrenchment,
        then recency (newest first).
        """
        self._maybe_rebuild()
        q_tf = Counter(_tokens(query))
        if not q_tf or not self._doc_ids:
            return []
        q_norm = math.sqrt(
            sum((c * self._idf.get(t, 0.0)) ** 2 for t, c in q_tf.items())
        )
        scored: list[tuple[float, float, float, str]] = []
        by_id = {b.id: b for b in self.store.active_beliefs()}
        for did, tf in zip(self._doc_ids, self._tf):
            b = by_id.get(did)
            if b is None:
                continue
            if as_of is not None:
                vf = b.valid_from if b.valid_from is not None else float("-inf")
                vu = b.valid_until if b.valid_until is not None else float("inf")
                if not (vf <= as_of <= vu):
                    continue
            dot = sum(
                c * tf.get(t, 0) * self._idf.get(t, 0.0) ** 2
                for t, c in q_tf.items()
            )
            d_norm = math.sqrt(
                sum((c * self._idf.get(t, 0.0)) ** 2 for t, c in tf.items())
            )
            cos = dot / (q_norm * d_norm) if q_norm and d_norm else 0.0
            ent = self.store.entrenchment_of(b)
            scored.append((cos, ent, b.timestamp, did))
        scored.sort(key=lambda s: (s[0], s[1], s[2]), reverse=True)
        out = []
        for cos, ent, _ts, did in scored[:top_k]:
            if cos <= 0.0:
                continue  # no lexical overlap: not evidence, don't return noise
            b = by_id[did]
            out.append(
                {
                    "text": b.proposition,
                    "score": round(cos, 4),
                    "entrenchment": round(ent, 4),
                    "timestamp": b.timestamp,
                    "source": b.source,
                    "confidence": b.confidence,
                }
            )
        return out
