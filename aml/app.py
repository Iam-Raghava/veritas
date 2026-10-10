"""Veritas AML service: Add/Search HTTP API for the Agent Memory Challenge.

Veritas is the consistency layer: Add asserts incoming texts as beliefs
(contraction + dedup + conflict resolution happen automatically);
Search returns only ACTIVE beliefs as evidence, so the evaluation
platform never sees retracted/stale context. That is the "memory
governance" the Textual Memory track scores.

Production features: structured logging with request IDs, Prometheus
metrics, liveness/readiness probes, input limits, graceful degradation.

Run:
    pip install -r requirements.txt
    VERITAS_DB=/data/veritas.db uvicorn aml.app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /add     {items: [...]} -> counts
    POST /search  {query, top_k?, as_of?} -> {evidence: [...]}
    GET  /health  -> liveness
    GET  /ready   -> readiness (deps checked)
    GET  /metrics -> Prometheus-style stats
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from veritas import BeliefStore
from veritas import persist

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("veritas-aml")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = os.environ.get(
    "VERITAS_DB", os.path.join(os.path.dirname(__file__), "veritas_aml.db"))
DETECTOR_NAME = os.environ.get("VERITAS_DETECTOR", "heuristic")
RETRIEVAL_MODE = os.environ.get("VERITAS_RETRIEVAL", "hybrid")
MAX_BATCH = int(os.environ.get("VERITAS_MAX_BATCH", "1000"))
MAX_TEXT_LEN = int(os.environ.get("VERITAS_MAX_TEXT_LEN", "100000"))

# ---------------------------------------------------------------------------
# Metrics (in-memory counters)
# ---------------------------------------------------------------------------
_metrics = {
    "add_requests": 0,
    "add_items": 0,
    "add_errors": 0,
    "search_requests": 0,
    "search_errors": 0,
    "add_latency_ms": [],
    "search_latency_ms": [],
    "start_time": time.time(),
}


def _pct(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    return s[min(int(len(s) * p), len(s) - 1)]


# ---------------------------------------------------------------------------
# Store + retriever
# ---------------------------------------------------------------------------
def _make_store() -> BeliefStore:
    if os.path.exists(DB_PATH):
        log.info("loading store from %s", DB_PATH)
        return persist.load(DB_PATH)
    log.info("new store (detector=%s)", DETECTOR_NAME)
    if DETECTOR_NAME == "nli":
        from veritas.detect_nli import HuggingFaceNLIDetector
        return BeliefStore(detector=HuggingFaceNLIDetector())
    if DETECTOR_NAME == "llm":
        from veritas.detect_llm import LLMJudgeDetector
        return BeliefStore(detector=LLMJudgeDetector(
            api_key=os.environ.get("VERITAS_LLM_API_KEY"),
            model=os.environ.get("VERITAS_LLM_MODEL", "gpt-4o-mini"),
            api_base=os.environ.get(
                "VERITAS_LLM_API_BASE", "https://api.openai.com/v1"),
        ))
    return BeliefStore()  # heuristic (default)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup: detector=%s retrieval=%s db=%s",
             DETECTOR_NAME, RETRIEVAL_MODE, DB_PATH)
    yield
    log.info("shutdown: persisting store")
    try:
        persist.save(store, DB_PATH, detector_name=store.detector_name)
        log.info("shutdown: store persisted")
    except Exception as e:
        log.error("shutdown: persist failed: %s", e)


app = FastAPI(title="Veritas AML Add/Search", version="0.2.0",
              lifespan=lifespan)

store: BeliefStore = _make_store()

if RETRIEVAL_MODE == "hybrid":
    from .retrieval_hybrid import HybridRetriever
    retriever = HybridRetriever(store)
    log.info("retriever: hybrid (dense=%s)",
             getattr(retriever, "dense_active", False))
else:
    from .retrieval import BeliefRetriever
    retriever = BeliefRetriever(store)
    log.info("retriever: tfidf")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.rid = rid
    t0 = time.time()
    try:
        resp = await call_next(request)
    except Exception as e:
        log.error("[%s] unhandled: %s %s: %s",
                  rid, request.method, request.url.path, e)
        return JSONResponse({"error": "internal error", "request_id": rid},
                            status_code=500)
    dt_ms = (time.time() - t0) * 1000
    resp.headers["X-Request-ID"] = rid
    log.info("[%s] %s %s -> %d (%.1fms)",
             rid, request.method, request.url.path,
             resp.status_code, dt_ms)
    return resp


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AddItem(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LEN)
    timestamp: float | None = None
    source: str = Field(default="aml", max_length=500)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_reliability: float = Field(default=0.6, ge=0.0, le=1.0)


class AddRequest(BaseModel):
    items: list[AddItem] = Field(min_length=1, max_length=MAX_BATCH)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_TEXT_LEN)
    top_k: int = Field(default=5, ge=1, le=50)
    as_of: float | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/add")
def add(req: AddRequest, request: Request) -> dict:
    from veritas.belief import Belief

    rid = request.state.rid
    t0 = time.time()
    _metrics["add_requests"] += 1
    try:
        added = rejected = superseded = 0
        active_before = len(store.active_beliefs())
        for item in req.items:
            ts = item.timestamp if item.timestamp is not None else time.time()
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
                        c.id,
                        reason="superseded by newer report from same source",
                    )
                    superseded += 1
            res = store.assert_belief(
                item.text,
                confidence=item.confidence,
                source=item.source,
                source_reliability=item.source_reliability,
                timestamp=ts,
                ground=True,
                valid_from=ts,
            )
            if res is None:
                rejected += 1
            else:
                added += 1
        active_after = len(store.active_beliefs())
        retracted = max(0, active_before + added - active_after - superseded)
        persist.save(store, DB_PATH, detector_name=store.detector_name)
        _metrics["add_items"] += len(req.items)
        dt_ms = (time.time() - t0) * 1000
        _metrics["add_latency_ms"].append(dt_ms)
        if len(_metrics["add_latency_ms"]) > 1000:
            _metrics["add_latency_ms"] = _metrics["add_latency_ms"][-1000:]
        log.info("[%s] add: %d items -> +%d/-%d/sup=%d (%.0fms)",
                 rid, len(req.items), added, rejected, superseded, dt_ms)
        return {
            "added": added,
            "rejected": rejected,
            "superseded": superseded,
            "retracted_by_contraction": retracted,
            "active_beliefs": active_after,
            "consistent": store.is_consistent(),
            "request_id": rid,
        }
    except Exception as e:
        _metrics["add_errors"] += 1
        log.error("[%s] add failed: %s", rid, e)
        raise


@app.post("/search")
def search(req: SearchRequest, request: Request) -> dict:
    rid = request.state.rid
    t0 = time.time()
    _metrics["search_requests"] += 1
    try:
        evidence = retriever.search(req.query, top_k=req.top_k,
                                    as_of=req.as_of)
        dt_ms = (time.time() - t0) * 1000
        _metrics["search_latency_ms"].append(dt_ms)
        if len(_metrics["search_latency_ms"]) > 1000:
            _metrics["search_latency_ms"] = _metrics["search_latency_ms"][-1000:]
        return {"evidence": evidence, "count": len(evidence),
                "request_id": rid}
    except Exception as e:
        _metrics["search_errors"] += 1
        log.error("[%s] search failed: %s", rid, e)
        raise


@app.get("/health")
def health() -> dict:
    """Liveness: is the process alive?"""
    return {"status": "ok", "version": "0.2.0"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: can we serve traffic? Checks dependencies."""
    checks = {}
    # DB writable?
    try:
        d = os.path.dirname(os.path.abspath(DB_PATH)) or "."
        checks["db_writable"] = os.access(d, os.W_OK)
    except Exception:
        checks["db_writable"] = False
    # Retriever functional?
    try:
        checks["retriever_ok"] = retriever is not None
        checks["dense_active"] = bool(
            getattr(retriever, "dense_active", False))
    except Exception:
        checks["retriever_ok"] = False
        checks["dense_active"] = False
    # Store consistent?
    try:
        checks["store_consistent"] = store.is_consistent()
        checks["active_beliefs"] = len(store.active_beliefs())
    except Exception:
        checks["store_consistent"] = False
        checks["active_beliefs"] = -1
    ok = all([checks["db_writable"], checks["retriever_ok"]])
    return {"ready": ok, **checks}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Prometheus-style metrics."""
    up = time.time() - _metrics["start_time"]
    lines = [
        "# HELP veritas_uptime_seconds Service uptime",
        "# TYPE veritas_uptime_seconds gauge",
        f"veritas_uptime_seconds {up:.0f}",
        "# HELP veritas_add_requests_total Total /add requests",
        "# TYPE veritas_add_requests_total counter",
        f"veritas_add_requests_total {_metrics['add_requests']}",
        "# HELP veritas_add_items_total Total items ingested",
        "# TYPE veritas_add_items_total counter",
        f"veritas_add_items_total {_metrics['add_items']}",
        "# HELP veritas_search_requests_total Total /search requests",
        "# TYPE veritas_search_requests_total counter",
        f"veritas_search_requests_total {_metrics['search_requests']}",
        "# HELP veritas_errors_total Errors by endpoint",
        "# TYPE veritas_errors_total counter",
        f"veritas_errors_total{{endpoint=\"add\"}} {_metrics['add_errors']}",
        f"veritas_errors_total{{endpoint=\"search\"}} {_metrics['search_errors']}",
        "# HELP veritas_latency_ms_p50 p50 latency",
        "# TYPE veritas_latency_ms_p50 gauge",
        f"veritas_latency_ms_p50{{op=\"add\"}} {_pct(_metrics['add_latency_ms'], 0.5):.1f}",
        f"veritas_latency_ms_p50{{op=\"search\"}} {_pct(_metrics['search_latency_ms'], 0.5):.1f}",
        "# HELP veritas_latency_ms_p99 p99 latency",
        "# TYPE veritas_latency_ms_p99 gauge",
        f"veritas_latency_ms_p99{{op=\"add\"}} {_pct(_metrics['add_latency_ms'], 0.99):.1f}",
        f"veritas_latency_ms_p99{{op=\"search\"}} {_pct(_metrics['search_latency_ms'], 0.99):.1f}",
        "# HELP veritas_active_beliefs Active beliefs",
        "# TYPE veritas_active_beliefs gauge",
        f"veritas_active_beliefs {len(store.active_beliefs())}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")
