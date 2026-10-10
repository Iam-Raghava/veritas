"""Veritas AML service: Add/Search HTTP API for the Agent Memory Challenge.

Veritas is the consistency layer: Add asserts incoming texts as beliefs
(contraction + dedup + conflict resolution happen automatically);
Search returns only ACTIVE beliefs as evidence, so the evaluation
platform never sees retracted/stale context. That is the "memory
governance" the Textual Memory track scores.

Run:
    pip install -r requirements.txt
    VERITAS_DB=/data/veritas.db uvicorn aml.app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /add    {items: [{text, timestamp?, source?, confidence?}]} -> counts
    POST /search {query, top_k?, as_of?} -> {evidence: [...]}
    GET  /health -> {status, active_beliefs, consistent}
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from veritas import BeliefStore
from veritas import persist

from .retrieval import BeliefRetriever

DB_PATH = os.environ.get("VERITAS_DB", os.path.join(os.path.dirname(__file__), "veritas_aml.db"))

app = FastAPI(title="Veritas AML Add/Search", version="0.1.0")

if os.path.exists(DB_PATH):
    store: BeliefStore = persist.load(DB_PATH)
else:
    store = BeliefStore()
retriever = BeliefRetriever(store)


class AddItem(BaseModel):
    text: str = Field(min_length=1)
    timestamp: float | None = None
    source: str = "aml"
    confidence: float = 0.8
    source_reliability: float = 0.6


class AddRequest(BaseModel):
    items: list[AddItem]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    as_of: float | None = None


@app.post("/add")
def add(req: AddRequest) -> dict:
    from veritas.belief import Belief

    added = rejected = superseded = 0
    active_before = len(store.active_beliefs())
    for item in req.items:
        ts = item.timestamp if item.timestamp is not None else time.time()
        # Same-source update: a source correcting its own earlier report
        # is a supersession, not a contradiction to adjudicate. Retract
        # the older same-source contradictors first (audited), then assert.
        probe = Belief(
            proposition=item.text,
            source=item.source,
            timestamp=ts,
            ground=True,
            valid_from=ts,
        )
        for c in store.contradictors_of(probe):
            if c.source == item.source and c.timestamp < ts:
                store.retract_belief(
                    c.id, reason="superseded by newer report from same source"
                )
                superseded += 1
        res = store.assert_belief(
            item.text,
            confidence=item.confidence,
            source=item.source,
            source_reliability=item.source_reliability,
            timestamp=ts,
            ground=True,  # Add events are direct observations
            valid_from=ts,
        )
        if res is None:
            rejected += 1
        else:
            added += 1
    active_after = len(store.active_beliefs())
    # Contraction retractions = beliefs displaced by stronger newcomers or
    # lost to cascades, excluding supersessions (counted separately).
    retracted = max(0, active_before + added - active_after - superseded)
    persist.save(store, DB_PATH)
    return {
        "added": added,
        "rejected": rejected,
        "superseded": superseded,
        "retracted_by_contraction": retracted,
        "active_beliefs": active_after,
        "consistent": store.is_consistent(),
    }


@app.post("/search")
def search(req: SearchRequest) -> dict:
    evidence = retriever.search(req.query, top_k=req.top_k, as_of=req.as_of)
    return {"evidence": evidence, "count": len(evidence)}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "active_beliefs": len(store.active_beliefs()),
        "consistent": store.is_consistent(),
        "version": "0.1.0",
    }
