# Veritas AML service

Add/Search HTTP wrapper over the Veritas truth-maintenance store, built for
the **Agent Memory Challenge 2026 — Cycle 2** (Textual Memory track,
open-source division).

Veritas is the **consistency layer**: `Add` asserts incoming texts as beliefs
(contradiction detection, entrenchment-ordered contraction, and dependency
cascades happen automatically); `Search` returns only **active** beliefs as
evidence, so the evaluation platform never sees retracted or stale context.
That is the "memory governance" the track scores.

**Retrieval is hybrid** (dense + TF-IDF with RRF fusion): local ONNX
multilingual embeddings (jina-embeddings-v2-small-en, 512-dim, no torch,
no API costs) catch paraphrases and cross-lingual matches ("CEO" ~
"首席执行官"); CJK-aware TF-IDF (character bigrams) handles exact terms.
Set `VERITAS_DENSE_DIR` to the model directory (Docker image bundles it);
without it, retrieval falls back to TF-IDF cleanly.
**Detection is selectable**: heuristic (default, transparent),
NLI (`VERITAS_DETECTOR=nli`, roberta-large-mnli), or LLM judge
(`VERITAS_DETECTOR=llm`). The governance engine — entrenchment-ordered
contraction, cascades, audit — is identical regardless.

## Configuration

| Env var | Default | Options |
|---|---|---|
| `VERITAS_DB` | `./aml/veritas_aml.db` | persistent path |
| `VERITAS_RETRIEVAL` | `hybrid` | `hybrid` (dense+TF-IDF) or `tfidf` |
| `VERITAS_DETECTOR` | `heuristic` | `heuristic`, `nli`, `llm` |
| `VERITAS_DENSE_DIR` | — | dir with `model.onnx` + `tokenizer.json` |
| `VERITAS_LLM_API_KEY` / `_MODEL` / `_API_BASE` | — | for `llm` detector |

## Run locally

```bash
pip install -r requirements.txt
pip install -e ..            # the veritas package
uvicorn aml.app:app --port 8000
```

## API

`POST /add` — ingest memory items:

```json
{"items": [
  {"text": "The CEO is Alice.", "timestamp": 1728518400, "source": "meeting_notes"},
  {"text": "The CEO is Bob.", "source": "rumor", "source_reliability": 0.2}
]}
```

→ `{"added": 2, "rejected": 0, "retracted_by_contraction": 1, ...}`
(Bob's rumor contradicts Alice's grounded fact; the weaker side loses.)

`POST /search` — retrieve current evidence:

```json
{"query": "who is the CEO?", "top_k": 5}
```

→ `{"evidence": [{"text": "The CEO is Alice.", "score": 0.71, "entrenchment": 0.83, ...}], "count": 1}`

Retracted beliefs never appear in `Search` results.

`GET /health` — liveness probe (is the process alive?).

`GET /ready` — readiness probe (can we serve? checks DB writability,
retriever status, dense model, store consistency).

`GET /metrics` — Prometheus-style metrics (request counts, p50/p99
latencies, active beliefs, uptime).

Production notes: structured logs with `X-Request-ID` tracing, input
limits (`VERITAS_MAX_BATCH=1000`, `VERITAS_MAX_TEXT_LEN=100000`),
graceful shutdown persists the store, unhandled errors return
`{"error": ..., "request_id": ...}` (never a bare 500).

## Deploy

Any host with a public HTTPS endpoint works (fly.io, Railway, cheap VPS).
Set `VERITAS_DB` to a persistent path; the store saves after every `/add`
batch and reloads on startup.

```bash
# bare metal
VERITAS_DB=/data/veritas.db uvicorn aml.app:app --host 0.0.0.0 --port 8000

# docker
docker build -f aml/Dockerfile -t veritas-aml .
docker run -p 8000:8000 -v veritas-data:/data veritas-aml

# compose
docker compose -f aml/docker-compose.yml up -d
```

## Test

```bash
python -m pytest tests/test_aml_service.py -q
```
