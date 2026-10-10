"""Cross-encoder reranker for the AML hybrid retriever.

Pipeline: hybrid (dense + TF-IDF) retrieves top-K candidates fast,
then the cross-encoder scores each (query, document) pair precisely.
This is the standard retrieve-then-rerank architecture that wins
retrieval competitions: bi-encoders for recall, cross-encoders
for precision.

Model: bge-reranker-base (ONNX, ~280MB, EN/ZH bilingual).
Set VERITAS_RERANK_DIR to the directory with model.onnx + tokenizer.json.
Set VERITAS_RERANK_TOPK (default 20) for candidates to rerank.
Set VERITAS_RERANK=0 to disable.

Falls back gracefully: if the model is missing, hybrid ranking stands.
"""

from __future__ import annotations

import os

RERANK_DIR = os.environ.get("VERITAS_RERANK_DIR", "")
RERANK_TOPK = int(os.environ.get("VERITAS_RERANK_TOPK", "20"))
RERANK_ENABLED = os.environ.get("VERITAS_RERANK", "1") == "1"


class Reranker:
    """Cross-encoder: scores (query, document) pairs."""

    def __init__(self, model_dir: str) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        tok.enable_padding(pad_id=0, pad_token="[PAD]")
        tok.enable_truncation(max_length=512)
        self._tok = tok
        self._sess = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Return relevance scores, one per doc, higher = more relevant."""
        np = self._np
        if not docs:
            return []
        pairs = [[query, d] for d in docs]
        enc = self._tok.encode_batch(pairs)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        out = self._sess.run(
            None,
            {"input_ids": ids, "attention_mask": mask},
        )[0]
        # bge-reranker outputs a single logit per pair.
        return out[:, 0].astype(float).tolist()

    @classmethod
    def load(cls) -> "Reranker | None":
        if not RERANK_ENABLED or not RERANK_DIR:
            return None
        try:
            if not (os.path.exists(os.path.join(RERANK_DIR, "model.onnx"))
                    and os.path.exists(
                        os.path.join(RERANK_DIR, "tokenizer.json"))):
                return None
            r = cls(RERANK_DIR)
            print(f"[rerank] loaded from {RERANK_DIR}")
            return r
        except Exception as e:
            print(f"[rerank] load failed ({e}); hybrid ranking only")
            return None
