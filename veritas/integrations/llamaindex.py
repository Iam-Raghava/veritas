"""LlamaIndex integration for Veritas.

Wraps document insertion so updated document chunks automatically
retract outdated ones based on document metadata timestamps.

NOTE: This is a standalone helper that accepts plain dicts
({"doc_id", "text", "timestamp"}). It is NOT a subclass of
llama_index BaseDocumentStore and does not accept Document/BaseNode
objects. Use it directly for truth-maintained document state
alongside LlamaIndex.

When a document with the same doc_id is re-inserted with a newer
timestamp, the old chunks are retracted (superseded) rather than
duplicated.

Usage:
    from veritas.integrations.llamaindex import VeritasDocumentStore

    store = VeritasDocumentStore()
    store.add_documents([
        {"doc_id": "doc1", "text": "Acme's CEO is Jane",
         "timestamp": 1000.0},
    ])
    # Later, updated version:
    store.add_documents([
        {"doc_id": "doc1", "text": "Acme's CEO is John",
         "timestamp": 2000.0},
    ])
    # Old "Jane" belief is retracted; "John" is active.
"""

from __future__ import annotations

import time
from typing import Any

from ..store import BeliefStore


class VeritasDocumentStore:
    """Veritas-backed document store for LlamaIndex.

    Each document chunk becomes a belief. Re-inserting a doc_id with
    a newer timestamp retracts the old chunks (temporal supersession).
    Contradictions between different doc_ids are adjudicated by
    entrenchment.
    """

    def __init__(
        self,
        store: BeliefStore | None = None,
        source_reliability: float = 0.8,
    ) -> None:
        self.store = store or BeliefStore()
        self.source_reliability = source_reliability
        # doc_id -> list of belief_ids (for retraction on update).
        self._doc_index: dict[str, list[str]] = {}
        # doc_id -> latest timestamp.
        self._doc_ts: dict[str, float] = {}

    def add_documents(self, documents: list[dict[str, Any]]) -> list[str]:
        """Add document chunks with truth maintenance.

        Each document dict needs: doc_id, text. Optional: timestamp,
        metadata.

        Returns list of belief IDs created.
        """
        belief_ids = []
        for doc in documents:
            doc_id = doc["doc_id"]
            text = doc["text"]
            ts = doc.get("timestamp", time.time())
            # Validate BEFORE retracting old chunks: a failed update must
            # not destroy the previous version (transactional).
            if not text or not text.strip():
                raise ValueError("proposition must be non-empty")
            # If this doc_id was seen with an older timestamp, retract
            # the old chunks (supersession).
            if doc_id in self._doc_ts and ts > self._doc_ts[doc_id]:
                for bid in self._doc_index.get(doc_id, []):
                    self.store.retract_belief(
                        bid,
                        reason=f"superseded by newer version of '{doc_id}'",
                    )
                self._doc_index[doc_id] = []
            # Skip if older than current (stale update).
            if doc_id in self._doc_ts and ts < self._doc_ts[doc_id]:
                continue
            # Chunk text if needed (simple sentence split).
            from ..extractor import HeuristicExtractor
            extractor = HeuristicExtractor()
            claims = extractor.extract_claims(text)
            texts = [c.proposition for c in claims] if claims else [text]
            for chunk_text in texts:
                belief = self.store.assert_belief(
                    chunk_text,
                    source=f"llamaindex:{doc_id}",
                    source_reliability=doc.get(
                        "reliability", self.source_reliability),
                    ground=True,
                    timestamp=ts,
                    valid_from=ts,
                    metadata={
                        "doc_id": doc_id,
                        "chunk": True,
                        **doc.get("metadata", {}),
                    },
                )
                if belief is not None:
                    belief_ids.append(belief.id)
                    self._doc_index.setdefault(doc_id, []).append(belief.id)
            self._doc_ts[doc_id] = ts
        return belief_ids

    def get_document(
        self, doc_id: str,
    ) -> list[dict[str, Any]]:
        """Get active chunks for a doc_id."""
        result = []
        for bid in self._doc_index.get(doc_id, []):
            b = self.store.get(bid)
            if b and b.is_active:
                result.append({
                    "proposition": b.proposition,
                    "belief_id": b.id,
                    "timestamp": b.timestamp,
                })
        return result

    def delete_document(self, doc_id: str) -> None:
        """Retract all chunks for a doc_id."""
        for bid in self._doc_index.pop(doc_id, []):
            self.store.retract_belief(
                bid, reason=f"document '{doc_id}' deleted")
        self._doc_ts.pop(doc_id, None)
