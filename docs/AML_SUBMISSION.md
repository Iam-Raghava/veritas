# Agent Memory Challenge 2026 — Cycle 2: Submission Materials (DRAFT)

Track: **Textual Memory** · Division: **Open-source Methods** · Detector: heuristic (model-free)

## Method name

**Veritas** — truth maintenance for agent memory: a deterministic epistemic
engine that retracts contradicted beliefs by entrenchment, propagates
retractions through belief dependencies, and audits everything.

Repo: https://github.com/Iam-Raghava/veritas · License: MIT

## What it is

Most memory systems are append-only: they accumulate facts and retrieve by
similarity. When facts change — a CEO is replaced, a deadline moves, a
preference updates — the old facts stay in the index and pollute retrieval.
Veritas is the governance layer that prevents this:

1. **Add** asserts each incoming text as a belief (a direct observation,
   `ground=True`, timestamped, source-attributed).
2. Contradiction detection runs automatically. On conflict, Veritas computes
   each side's *entrenchment* (source reliability, corroboration, confidence,
   recency — Hansson's non-prioritized belief-base revision, EE2/EE3) and
   retracts weakest-first. A weak rumor can never displace a grounded fact.
3. Retractions **cascade**: beliefs justified only by retracted premises fall
   too (JTMS-style), unless independently grounded.
4. A **same-source update** (a source correcting its own earlier report) is
   treated as supersession, not adjudication.
5. **Search** retrieves from *active beliefs only*, ranked by TF-IDF with
   temporal `as_of` scoping. Retracted beliefs never surface as evidence.

Everything is appended to an immutable audit log with reasons.

## Why it matters for this benchmark

The Textual Memory track scores memory governance, freshness, and temporal
understanding — not just recall. A pure retrieval system can return a fact
that was true in March and false since June; Veritas returns the latest
valid state, because the outdated belief was retracted when the update
arrived. The innovation claim: **principled, deterministic, fully audited
belief revision as a memory layer** — no LLM in the retraction decision.

## Architecture

```
POST /add  → BeliefStore.assert_belief (contraction + cascades + audit)
             → SQLite persist
POST /search → Hybrid retrieval (dense + TF-IDF, RRF fusion)
               over active beliefs → evidence (text, score,
               entrenchment, timestamp, source)
```

**Retrieval**: hybrid dense + TF-IDF. Local ONNX multilingual embeddings
(jina-embeddings-v2-small-en via fastembed — no API costs) catch
paraphrases and cross-lingual matches; CJK-aware TF-IDF (character
bigrams) handles exact terms. RRF fusion combines rankings. Only
ACTIVE beliefs are indexed — retracted beliefs never surface.

**Detection** (selectable via `VERITAS_DETECTOR`): heuristic (default,
transparent), NLI (roberta-large-mnli), or LLM judge. The governance
engine — entrenchment-ordered contraction, JTMS cascades, audit —
is identical regardless.

## Why this wins

Most entries are pure retrieval: they return whatever matches, including
stale and contradicted facts. Veritas is the only entry with principled
memory governance — when "CEO is John" arrives against "CEO is Jane",
the weaker claim is retracted (audited), its dependents cascade, and
search never surfaces the outdated fact. On the Adapt (memory governance)
dimension this is the decisive edge; on Recall, hybrid dense+TF-IDF
keeps us competitive.

## Verification (all reproducible in-repo)

- 40/40 core unit tests, 12/12 + 22/22 formal-rigor suites
- 14/14 adversarial cases (corrupt DBs, hostile detectors, clock skew…)
- Differential testing: 20 trials × 300 random ops vs an independent naive
  reference implementation — every accept/reject decision identical
- 7/7 targeted mutants killed; 30/30 end-to-end gate (`tests/verify_e2e.sh`)

## Fixed commit

TODO: pin the evaluated commit hash here after the final pre-submission run.
