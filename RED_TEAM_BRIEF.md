# Veritas — Technical Brief for Red-Teaming

> **Historical snapshot.** This brief was written on 2026-10-10 for Veritas
> v0.11.0. It is preserved as a record of that review. For the current
> state, see [README.md](README.md) (v0.14.1 as of 2026-10-15).

**Author:** RAGS · **Repo:** https://github.com/Iam-Raghava/veritas · **Version:** 0.11.0 · **License:** MIT
**Date:** 2026-10-10 · **Purpose:** Complete technical disclosure for independent review by frontier LLMs. Includes known weaknesses — attack them.

---

## 1. What Veritas Is

Veritas is a **truth maintenance system (TMS) for AI agent memory**. The core problem: LLM agents accumulate facts in append-only memory (conversation history, vector stores, RAG indexes). When facts change, contradict, or expire, nothing retracts them — the agent keeps retrieving stale context and hallucinating on it.

Veritas is the **garbage collector for beliefs**: a principled layer that tracks what the agent believes, why it believes it (justifications), how strongly (entrenchment), and automatically retracts beliefs that are contradicted, superseded, or temporally expired — with full audit trails.

**Formal positioning:** Hansson-style non-prioritized belief-base revision (not classical AGM — AGM operates on closed theories; Veritas operates on finite belief bases, which is the computationally tractable variant). Entrenchment-ordered contraction + JTMS-style dependency propagation.

**What it is NOT:** It is not a new mathematical result. The defensible novelty is the standalone integration and production engineering of belief revision + dependency propagation + temporal validity + auditability + retrieval — as one coherent system for agent memory. Prior art exists for each piece (AGM, JTMS, EPISTEMICS, Mem0, hirn); the gap we verified was that no one had built automated principled belief retraction as infrastructure for LLM agents.

---

## 2. Core Architecture (veritas/ package)

### 2.1 Belief model (`veritas/belief.py`)
A belief has: `proposition` (text), `source`, `source_reliability` (0-1), `confidence` (0-1), `timestamp`, `valid_from`/`valid_until` (temporal bounds), `justifications` (list of belief IDs this depends on — Horn-clause AND-sets, OR across sets), `ground` (bool — direct observation vs derived).

Key design decisions:
- **Snapshot semantics:** `snapshot()` returns deep copies. `TimeTravel` and `Explainer` operate on snapshots, not live objects (fixed a bug where live objects leaked).
- **Timestamp None-semantics:** explicit `timestamp=0.0` is a valid timestamp, not "missing" (fixed a bug where `if not self.timestamp` treated 0.0 as absent).

### 2.2 Entrenchment (`veritas/entrenchment.py`)
Entrenchment score = f(source_reliability, corroboration count, confidence, recency). Higher entrenchment = harder to dislodge. Used to order contraction: when beliefs contradict, the least entrenched is retracted first.

- **EE2/EE3 caps:** entrenchment is bounded to prevent runaway scores from breaking the ordering (added in v0.3.0 after adversarial review).
- **Borrowed entrenchment:** derived beliefs inherit a discounted fraction of their justifiers' entrenchment.

### 2.3 Contraction policies (`veritas/policies.py`)
- `entrenchment` (default): retract least-entrenched first.
- `conservative`: prefer keeping existing beliefs over new ones.
- `maxichoice` was documented but unimplemented — **removed** (phantom feature caught in audit).

### 2.4 Contradiction detection (`veritas/detect.py`, `detect_nli.py`, `detect_llm.py`, `detect_semantic.py`)
Pluggable `Detector` protocol with `contradicts(a, b) -> bool`:
- **Heuristic** (default): transparent keyword/negation patterns. Fast, explainable, misses paraphrases.
- **NLI** (`detect_nli.py`): HuggingFace roberta-large-mnli. Verified against real weights (4/4 pairs correct). Heavy (~1.4GB).
- **LLM judge** (`detect_llm.py`): any OpenAI-compatible API (Claude 5.5, GPT-6, Gemini 4). Fail-open by design (documented).
- **Semantic** (`detect_semantic.py`): embedding-based.
- **Commutativity harness:** detectors are tested for `contradicts(a,b) == contradicts(b,a)`.
- **Constraint exceptions** are audited as `CONSTRAINT_ERROR`, not silently treated as satisfied (fixed in audit).

### 2.5 Dependency propagation (`veritas/store.py`)
JTMS-style: when a belief is retracted, all beliefs justified by it are cascade-retracted. Supports:
- **Horn-clause justifications:** AND-sets (all must hold), OR across sets.
- **Well-foundedness check:** kills circular A↔B justification bypass (v0.4.0).
- **N-ary constraints:** joint inconsistency across >2 beliefs (v0.5.0).
- **Doomed simulation:** before asserting, simulates whether the new belief would doom existing justifications.
- **Invalidation bus:** cache invalidation on state changes.
- **Ground-observation zombie prevention:** retracted ground facts can't be resurrected by derived beliefs.

### 2.6 Audit (`veritas/audit.py`)
Append-only log of every state change: assertions, retractions, cascade events, contradictions, constraint checks. Every entry is causally linked. `test_audit_proof.py` verifies completeness — every state change has a corresponding audit entry.

### 2.7 Time travel (`veritas/timetravel.py`)
Reconstructs belief state at any timestamp via audit replay. `when_retracted(belief_id)`, `history_of(belief_id)`. Returns snapshots, not live objects (fixed in audit). Handles cascade retraction events (fixed in audit).

### 2.8 Explanation (`veritas/explain.py`)
`why(belief_id)` — why is this believed? `why_retracted(belief_id)` — why was it retracted? Handles cascade events (fixed in audit). Does not touch private store state (fixed in audit).

### 2.9 Persistence (`veritas/persist.py`)
SQLite backend. Round-trips: beliefs, audit log, config, **policy** (fixed — conservative stores were silently reloading as entrenchment), **detector name** (fixed). Temporal bounds (`valid_from`/`valid_until`) preserved (fixed — `snapshot()`, `to_dict()`, `from_dict()` were all dropping them).

### 2.10 Other modules
- `veritas/calibration.py`: REMOVED in the 0.16.x cleanup — CalibrationTracker was 100% unwired (never called by the store); the manual-wiring pattern is documented in RED_TEAM_BRIEF history.
- `veritas/persist_backend.py`: REMOVED in the 0.16.x cleanup — a 50-line wrapper around `persist.save`/`persist.load` with a single implementation; use `veritas.persist` directly.
- `veritas/experimental/`: `distributed_gossip.py` (GossipEventLog) and `distributed_bft.py` (ByzantineTolerantLog) — functional but quarantined out of the core story; re-exported for compatibility (`veritas.GossipEventLog`, lazy `veritas.ByzantineTolerantLog`).
- `veritas/export.py`: `to_json`, `to_dot` (Graphviz), `to_graphml` (Gephi/Cytoscape).
- `veritas/async_store.py`: AsyncBeliefStore. Has `stats()` wrapper (added in audit).
- `veritas/cli.py`: Full CLI (init/assert/retract/list/contradictions/audit/stats).
- `veritas/observability.py`: Metrics — assertions, rejections, retractions, cascades, contradictions, constraint violations, cache hits/misses. **Mostly unwired before audit; now instrumented.**
- `veritas/integrations/`: standalone memory helpers (autogen, crewai); experimental langgraph/llamaindex helpers. (LangChain removed.)

---

## 3. Competition service — REMOVED 2026-10-12

The Agent Memory Challenge entry (`aml/` FastAPI service, Docker/Fly config,
`docs/AML_CHALLENGE_ENTRY_PLAN.md`, `docs/AML_SUBMISSION.md`) was deleted in
commit 33b7bb0: the ~$25 hosting cost broke the zero-investment constraint and
the project pivoted to direct frontier-lab outreach. The ONNX dense embedder
survives as `veritas/dense.py` (core vector prefilter, local BGE-M3, no API
costs); everything else competition-specific is gone.

---

## 4. Testing Regime

**E2E gate** (`tests/verify_e2e.sh`): **38/38 green.** Covers package integrity, all unit suites, CLI, persistence, deployment, publish-clean.

**Test suites:**
- Core: 40/40. Adversarial: 14/14. Differential (vs reference implementation): clean. Properties, determinism (frozen-time replay byte-identical), README doctest (3/3), README sync (68/68 — version/exports/test-files/changelog match code).
- `test_aml_service.py`: governance + temporal as_of behavior.

**Stress testing (to failure):**
- 200,000 beliefs: 10.6s, 324MB, no OOM. Linear scaling (~1.6KB/belief).
- 5,000-deep cascade: built 63s, retracted all in 5s. 10,000 fan-out: 18s cascade. Correct.
- 500-contradiction storm: exactly 1 winner (highest reliability), consistent.
- 16-thread concurrent writes: 0 errors, consistent.
- Dense: 1MB text OK (tokenizer truncates), 5,000-batch OK. **Found:** empty list crashed ONNX → fixed.
- **Found:** dense rebuild re-embedded all beliefs per search (7.7s/query) → incremental indexing (0.6s/query).
- SQLite 50k: save/load 1.1s each, 16.8MB.

**Audit history:** Two rounds of adversarial review by frontier models (v0.3.0–v0.5.0) found real bugs, all fixed. Full line-by-line audit (2026-10-10) found 9 real bugs, all fixed.

---

## 5. Known Limitations and Weaknesses (Attack These)

1. **Deep-chain builds are O(N²)** (~12ms/assert at depth 5000). Acceptable for flat fact workloads, but a genuine algorithmic limit. No indexing on justification chains.
2. **Heuristic detector misses paraphrases.** The default detector is keyword-based. NLI/LLM detectors exist but aren't the default (cost/latency).
3. **CalibrationTracker removed (0.16.x cleanup).** It was 100% unwired; detector confidence scores remain uncalibrated by default.
4. **No chunking for long Add items.** Long documents get truncated by the tokenizer. Each Add item is one belief.
5. **Query expansion is TF-IDF-only.** Doesn't use embeddings for expansion terms. Could miss semantic expansions.
6. **Reranker is EN/ZH bilingual**, not truly multilingual like BGE-M3. Other languages get weaker reranking.
7. **TF-IDF index rebuilds are O(N)** on generation change. Fine at 200k, unknown beyond.
8. **No distributed operation.** Single-process, SQLite. No sharding, no replication.
9. **Ent
...[truncated 2388 chars]