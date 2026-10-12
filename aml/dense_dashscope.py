"""DashScope text-embedding-v4 embedder (OpenAI-compatible API) for the AML entry.

Why this exists: the challenge's academic (open-methods) board mandates
text-embedding-v4 for embeddings (official FAQ Q05). This module is the
API-backed drop-in for the local ONNX DenseEmbedder — same interface,
same graceful-fallback contract.

Selection (never on by default — the local stack stays the default):
    VERITAS_DENSE_BACKEND=dashscope   # default: local
Auth:
    DASHSCOPE_API_KEY=<key>           # or VERITAS_DASHSCOPE_KEY
Tuning:
    VERITAS_DASHSCOPE_BATCH=100       # texts per API call

No new dependencies: stdlib urllib only. Dimensions are detected from
the first response (text-embedding-v4 defaults to 1024), so a future
dimension change needs no code edit.

Falls back gracefully: load() returns None when the key is missing or
the probe call fails, and the hybrid retriever keeps TF-IDF-only.
"""

from __future__ import annotations

import json
import os
import urllib.request

ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
MODEL = "text-embedding-v4"


def _api_key() -> str:
    return os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get(
        "VERITAS_DASHSCOPE_KEY", ""
    )


class DashScopeEmbedder:
    """Embeddings via DashScope's OpenAI-compatible embeddings endpoint."""

    def __init__(self, api_key: str) -> None:
        import numpy as np

        self._np = np
        self._key = api_key
        self._batch = max(1, int(os.environ.get("VERITAS_DASHSCOPE_BATCH", "100")))
        # Probe once: detects output dims, fails fast on a bad key.
        probe = self.embed(["probe"])
        self.dims = probe.shape[1]

    def _call(self, texts: list[str]):
        np = self._np
        payload = json.dumps(
            {"model": MODEL, "input": texts, "encoding_format": "float"}
        ).encode()
        req = urllib.request.Request(
            ENDPOINT,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
        except Exception as e:
            raise RuntimeError(f"DashScope embeddings call failed: {e}") from e
        items = sorted(body.get("data", []), key=lambda d: d["index"])
        vecs = [d["embedding"] for d in items]
        if len(vecs) != len(texts):
            raise RuntimeError(
                f"DashScope returned {len(vecs)} vectors for {len(texts)} texts"
            )
        return np.asarray(vecs, dtype=np.float32)

    def embed(self, texts: list[str]):
        np = self._np
        if not texts:
            dims = getattr(self, "dims", 1024)
            return np.zeros((0, dims), dtype=np.float32)
        vecs = [self._call(texts[i : i + self._batch])
                for i in range(0, len(texts), self._batch)]
        mat = np.concatenate(vecs, axis=0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.clip(norms, 1e-9, None)

    @classmethod
    def load(cls) -> "DashScopeEmbedder | None":
        """Load with DASHSCOPE_API_KEY, or None if missing/unreachable."""
        key = _api_key()
        if not key:
            return None
        try:
            return cls(key)
        except Exception as e:
            print(f"[dense-dashscope] unavailable ({e}); TF-IDF only")
            return None
