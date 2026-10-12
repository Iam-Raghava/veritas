# Veritas: Honest Limits

This document states what Veritas does NOT do. Read this before
evaluating Veritas for production use.

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

## Scale

**Tested:**
- 200k beliefs, 324MB memory, no OOM
- 5000-deep cascade retracts in 5s
- 16-thread concurrent writes, 0 errors

**NOT tested / NOT supported:**
- 1M+ beliefs (likely works, not verified)
- Distributed deployment (single-machine only)
- Deep chains are O(N²) to build (~12ms/assert at depth 5000)

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
4. Production integrations (LangGraph, LlamaIndex)

You're buying 6 months of bug-hunting, not a secret algorithm.
