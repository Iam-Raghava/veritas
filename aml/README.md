# Veritas AML service

Add/Search HTTP wrapper over the Veritas truth-maintenance store, built for
the **Agent Memory Challenge 2026 — Cycle 2** (Textual Memory track,
open-source division).

Veritas is the **consistency layer**: `Add` asserts incoming texts as beliefs
(contradiction detection, entrenchment-ordered contraction, and dependency
cascades happen automatically); `Search` returns only **active** beliefs as
evidence, so the evaluation platform never sees retracted or stale context.
That is the "memory governance" the track scores.

Model-free by design: the heuristic contradiction detector and TF-IDF
retrieval need no embeddings and no LLM calls.

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

`GET /health` — liveness + store stats.

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
