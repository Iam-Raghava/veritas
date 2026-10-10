"""Lightweight ONNX dense embedder for the AML hybrid retriever.

Uses onnxruntime + tokenizers directly (no torch, no fastembed).
Model files (model.onnx, tokenizer.json) come from VERITAS_DENSE_DIR;
downloaded at Docker build time, not committed to git.

Falls back gracefully: if the directory or files are missing,
DenseEmbedder.load returns None and retrieval stays TF-IDF-only.
"""

from __future__ import annotations

import os

DENSE_DIR = os.environ.get("VERITAS_DENSE_DIR", "")


class DenseEmbedder:
    """Mean-pooled, L2-normalized embeddings from an ONNX model."""

    def __init__(self, model_dir: str) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tok = tok
        self._sess = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        # Warm up + report dims.
        probe = self.embed(["probe"])
        self.dims = probe.shape[1]

    def embed(self, texts: list[str]):
        np = self._np
        enc = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        tids = np.zeros_like(ids)
        out = self._sess.run(
            None,
            {"input_ids": ids, "attention_mask": mask,
             "token_type_ids": tids},
        )[0]
        m = mask[:, :, None].astype(np.float32)
        emb = (out * m).sum(axis=1) / m.sum(axis=1).clip(min=1e-9)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    @classmethod
    def load(cls) -> "DenseEmbedder | None":
        """Load from VERITAS_DENSE_DIR, or None if unavailable."""
        d = DENSE_DIR
        if not d:
            return None
        try:
            if not (os.path.exists(os.path.join(d, "model.onnx"))
                    and os.path.exists(os.path.join(d, "tokenizer.json"))):
                return None
            return cls(d)
        except Exception as e:
            print(f"[dense] model load failed ({e}); TF-IDF only")
            return None
