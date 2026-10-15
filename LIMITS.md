# Veritas: Honest Limits

This document states what Veritas does NOT do. Read this before
evaluating Veritas for production use.

## Positioning: Epistemic Middleware, Not a Database

Veritas is an **in-process, deterministic conflict arbiter** that sits
between your agent reasoning engine and your storage backend (vector
store, graph database, etc.).

Veritas is NOT:
- A database replacement (not an alternative to Zep, Mem0, or Postgres)
- A distributed system (single-process only)
- A vector store (it governs beliefs; pair it with your retriever)

## Contradiction Detection

**The heuristic detector (default) is pattern-based.** It catches:
- Functional properties: "X's CEO is A" vs "X's CEO is B"
- Explicit negations: "S is O" vs "S is not O"
- Relations in the FUNCTIONAL_RELATIONS whitelist

**It does NOT catch:**
- Paraphrases: "Acme's chief executive is Jane" vs "Acme's CEO is John"
  (different words for the same relation)
- Complex semantics requiring inference
- Contradictions across 3+ beliefs (multi-hop)

**For production:** Use `HybridDetector` (heuristic + NLI fallback) or
`HuggingFaceNLIDetector` or `LLMJudgeDetector`. The heuristic is for
tests and transparent demos, not for adversarial production data.

**LLMJudgeDetector warning:** Makes synchronous HTTP calls with 30s
timeout INSIDE the store lock. Under load, this blocks all operations.
Fail-open on exception (returns False = "no contradiction"). Use only
with careful timeout tuning and circuit breakers, or not at all.

## Scale

**Tested:**
- 200k beliefs, 324MB memory, no OOM
- 5000-deep cascade retracts in 5s
- 16-thread concurrent writes, 0 errors

**NOT tested / NOT supported:**
- 1M+ beliefs (likely works, not verified)
- Distributed deployment (single-machine only)
- Deep chains are O(N²) to build (~12ms/assert at depth 5000)

## Persistence

**SQLite:** Single-process only. WAL mode + busy timeout handle
brief contention, but multiple writers will conflict.

**PostgreSQL:** Removed in v0.11.0. The backend was never tested against
a real server and used destructive multi-writer behavior. PostgreSQL
URLs are now rejected with a clear error.

## Audit Log

Unbounded by default. `AuditLog(max_events=N)` bounds it (drops oldest).
For long-running agents, set a bound or periodically serialize and clear.
Without bounds, memory grows with total operations.

## Entrenchment Weights

The default weights (0.40 source, 0.25 corroboration, 0.20 confidence,
0.15 recency) and 30-day recency half-life are heuristics, not derived
from axioms. They work well for general agent memory but are NOT
theoretically grounded. Tune them for your domain:
- Financial/medical: shorter half-life, higher source weight
- Personal biography: longer half-life, higher corroboration weight

## What Veritas Is Not

- **Not a vector store.** Veritas governs beliefs; it does not do
  similarity search. Pair it with your retriever.
- **Not a framework.** It's a library. You integrate it; it doesn't
  run your agent.
- **Not multi-language.** Python only.
- **Not real-time.** Assertion latency is microseconds for heuristic,
  milliseconds for NLI. Fine for agent loops, not for HFT.

## The "Build vs Buy" Question

A competent team can reimplement core TMS in 2-3 months. Veritas's
value is not the algorithm (1980s math) but:
1. 27+ bugs already found and fixed via red-teaming
2. Differential testing against a reference implementation
3. Property-based tests, cascade verification, temporal correctness
4. Framework helpers (CrewAI, AutoGen as standalone helpers; LangGraph /
   LlamaIndex integrations are experimental and not drop-in adapters)

You're buying 6 months of bug-hunting, not a secret algorithm.
