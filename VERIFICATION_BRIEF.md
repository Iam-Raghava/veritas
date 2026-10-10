# Veritas — Verification Brief

**For independent verification by frontier AI models.**
Date: 2026-10-09. Version: v0.7.0. Location: `~/workspace/veritas`.

---

## 1. What it is

**Veritas** is a truth maintenance system (TMS) for AI agent memory. One-sentence
pitch: **garbage collection for beliefs**.

Long-lived AI agents accumulate beliefs from tool outputs, web searches, user
statements, and their own inferences. Over time these go stale or contradict
each other. Current approaches are: append everything (contradictions fester),
naive vector-memory retrieval (retrieves contradictions without resolving
them), or per-case LLM deletion judgments (unprincipled, unaudited).

Veritas provides a principled, automatic, fully-audited layer:

1. **Contradiction detection** — pluggable detectors (transparent heuristic
   default, optional HuggingFace NLI, custom via a `Detector` protocol).
2. **Entrenchment-based contraction** — when a new belief contradicts stored
   ones, compute each side's *entrenchment* (how well-supported it is) from
   source reliability, corroboration, confidence, and recency. Weaker side
   loses automatically, weakest-first (Hansson's non-prioritized base
   revision; entrenchment respects Gärdenfors–Makinson EE2/EE3).
3. **Dependency cascades** — beliefs record which other beliefs justify them
   (JTMS-style). Retracting a belief re-evaluates its dependents: those that
   keep other justifications survive; orphaned beliefs survive ONLY if
   grounded in direct observation (`ground=True`); ungrounded orphans
   cascade-retract (zombie-belief prevention).
4. **Complete audit** — every assertion, retraction, cascade, and rejection
   is appended to an immutable log with reasons and entrenchment values.
5. **Invalidation bus** — `store.subscribe(belief_id, callback)` fires on
   retraction, so long-running agents can invalidate stale snapshots.
6. **N-ary constraints** — `store.add_constraint(name, check_fn)` for joint
   inconsistencies that pairwise detectors miss (e.g. budget violations).
   Weakest involved beliefs retracted in entrenchment order.
7. **Semantic detection** — `SemanticDetector` uses embeddings + antonym
   analysis to catch contradictions beyond pattern matching.
8. **Temporal reasoning** — beliefs carry `valid_from`/`valid_until`;
   non-overlapping intervals don't contradict.
9. **Async API** — `AsyncBeliefStore` for asyncio-based agent frameworks.
10. **Framework adapters** — drop-in memory for LangChain, CrewAI, AutoGen.

---

## 2. Core algorithms (verify these)

### Entrenchment (`veritas/entrenchment.py`, capped in `store.entrenchment_of`)
```
raw(b) = w_source * source_reliability
       + w_corroboration * (1 - 0.5^n_justifications)
       + w_confidence * confidence
       + w_recency * 0.5^(age_days / 30)
entrenchment(b) = min(raw(b), min(raw(j) for j in transitive premises))
```
Clamped to [0,1]. Default weights: source 0.35, corroboration 0.25,
confidence 0.25, recency 0.15. The EE2 cap (derived <= weakest premise)
is enforced by `store.entrenchment_of` via cycle-safe BFS. Sorting uses
`(entrenchment, belief_id)` as the key — the ID tiebreak makes ordering
deterministic across Python hash seeds (verified by test).

### Contraction (`veritas/store.py::assert_belief`)
1. Prune justifications pointing at nonexistent/retracted beliefs.
2. Simulate which justifications are "doomed" (would die if contradictors
   are retracted, via fixed-point `_doomed_justifications`) and prune them
   *before* the decision — a newcomer cannot borrow corroboration from
   beliefs its own assertion will kill.
3. Find contradictors via indexed search + detector.
4. If any contradictor has `entrenchment >= newcomer_entrenchment` → **reject**
   the newcomer (audited, nothing changes). Ties go to the incumbent.
5. Else retract all contradictors weakest-first, cascading.

### Cascade (`veritas/store.py::_retract_single`)
Iterative (explicit stack, no recursion — verified to 5,000-deep chains and
10,000-wide fan-out). For each dependent that lost a justification:
- Still has justifications AND a well-founded path to ground → survives
  (audited). Well-foundedness is checked via cycle-safe BFS: the belief
  must be able to reach a `ground=True` belief through active justifications.
  Circular self-justifying loops (A↔B with no ground) are detected and
  retract.
- Orphaned (no justifications) → survives ONLY if `ground=True`
  (zombie-belief prevention). Ungrounded orphans retract.
- Else retracts and its own dependents are queued.

### Detection (`veritas/detect.py`)
Heuristic detector handles: functional properties ("X's CEO is Y" vs "X's CEO
is Z"), `is`/`is not` negation, `will`/`will not`, and explicit
`metadata["contradicts"]` pairs. Indexed by (subject, verb) and
(subject, object) for sub-quadratic lookup; custom detectors fall back to
exact scan (correctness over speed).

---

## 3. Novelty claim (qualified — challenge this)

**We do NOT claim the idea is new.** Belief revision (AGM, 1985; Hansson's
non-prioritized base revision) and truth maintenance (Doyle's TMS /
de Kleer's JTMS, late 1970s–80s) are classical.

**Claim:** Veritas is the first *standalone, general-purpose, fully audited
belief-base revision engine* with automated entrenchment-based contraction
and dependency cascades for Python agent runtimes — contradiction detection
→ entrenchment-ordered automatic contraction → dependency cascades →
complete audit — as a tested, pip-installable library.

**Prior art checked** (25 candidates surveyed 2026-10-08; closest below):
- *Universal LLM Gateway* AGM compliance report: entrenchment ordering
  present, but automated contraction explicitly deferred, no cascades.
- *EPISTEMICS/LiBrainian* issue #390: contraction work disconnected/frozen.
- *hirn*: confidence adjustment and reconciliation, not automatic cascading
  retraction.
- *Mem0*: per-case LLM deletion judgments — unprincipled, no entrenchment,
  no dependency propagation, no audit guarantee.
- *yigraf* (PyPI): tracks justifications and has grounding tiers, but
  conflicts are resolved *manually* ("never silently picks a winner") —
  no automated entrenchment-ordered contraction, no cascades, and it is
  code-repo-anchored rather than general-purpose.

**Falsification welcome:** if you know of a shipped system that does all
four (detection → entrenchment-ordered auto-contraction → cascades →
complete audit) for agent memory, that narrows the claim to "best-tested"
rather than "first."

---

## 4. Implementation

```
veritas/
  belief.py        Belief dataclass (proposition, confidence, source,
                   source_reliability, justifications, timestamp, status,
                   metadata). IDs/timestamps via __post_init__ (patchable).
                   snapshot() returns an isolated copy.
  store.py         BeliefStore: assert/retract/query, indexes, cascades,
                   RLock thread safety, __deepcopy__ support.
                   Public queries return snapshots (callers cannot corrupt
                   the index by mutating returned objects).
  entrenchment.py  Entrenchment scoring, ordering, independent_entrenchment.
  detect.py        Heuristic detector, Detector protocol, as_function adapter.
  detect_nli.py    Optional HuggingFace NLI detector.
  audit.py         Append-only AuditLog with typed events.
  persist.py       SQLite save/load, schema versioning (v1), repr-fallback
                   for unserializable metadata.
  cli.py           Full CLI: init/assert/retract/list/contradictions/
                   audit/stats.
  ~1,800 lines total. MIT. pip-installable (pyproject.toml).
```

---

## 5. Verification (challenge any of these)

| Suite | What it proves | Result |
|---|---|---|
| `tests/run.py` | 40 core behavior tests | 40/40 |
| `tests/test_v02.py` | 23 persistence/CLI/detector tests | 23/23 |
| `tests/test_edge.py` | 11 adversarial: cycles, diamonds, 5k-deep chains, hostile detectors, contradiction storms | 11/11 |
| `tests/test_ordering.py` | 6 entrenchment-ordering proofs + 200 randomized decision-rule trials | 6/6 |
| `tests/test_robust.py` | 10 API-misuse / detector-contract / CLI-fuzz cases | 10/10 |
| `tests/test_readme.py` | Every README code block executes | 2/2 |
| `tests/test_determinism.py` | Frozen-time replay byte-identical in-process AND across PYTHONHASHSEED 0/1/42 AND a constructed 3-way exact-tie retraction | pass |
| `tests/test_audit_proof.py` | Audit completeness: every active belief has exactly one birth event; every retracted belief has a full lifecycle; every cascade names its lost justification; seq dense | pass |
| `tests/test_differential.py` | 20 trials × 300 random ops vs an independent naive reference implementation: every accept/reject decision and final state identical | clean |
| `tests/test_mutation.py` | 7 targeted mutants (tie rule, retraction order, pruning, cascade, rejection) | 7/7 killed |
| `tests/test_adversarial.py` | 14 cases: corrupt DBs, future/legacy schemas, 10k fan-out, self-contradicting detector, negative weights, clock skew, concurrent CLI, reentrant detector, snapshot isolation, deepcopy | 14/14 |
| `tests/test_v03.py` | 12 tests: EE2/EE3 dominance, zombie prevention, ground survival, zombie-bridge, Hansson rejection, subscriptions | 12/12 |
| `tests/test_v04.py` | 22 checks: circular bypass, well-foundedness, Horn AND/OR, EE2 best-proof, commutativity, 4 constraint tests | 22/22 |
| `tests/test_properties.py` | 7 mathematical invariants: EE2, monotonicity, cache consistency, temporal | 7/7 |
| `tests/test_benchmarks.py` | 3 head-to-head vs naive baselines | 3/3 |
| `tests/verify_e2e.sh` | End-to-end: package, demo, dogfood, CLI, persistence, live deployment, publish-clean | 30/30 |

**Bugs found BY testing (not by inspection):** recursive cascade stack
overflow → iterative; concurrent dict mutation → RLock; dead justifications
accepted → pruned; audit event missing `lost_justification`; hash-dependent
retraction order → ID tiebreak; live-reference index corruption → snapshots;
`save()` crash on unserializable metadata → repr fallback; raw
`OperationalError` on non-Veritas DBs → clean `ValueError`; stale patterns
in the mutation suite itself.

---

## 6. Live deployment

Veritas truth-maintains the developer's own agent journal
(`deploy/journal_sync.py`, run via cron): parses the decision log and memory
file, detects contradictions with a journal-domain detector, treats
`RESOLVED` decisions as high-entrenchment evidence. Has found and retracted
3 real contradictions. This is production dogfooding, not a demo.

---

## 7. Honest limitations

- The heuristic detector is English-pattern-based; subtle semantic
  contradictions need the NLI detector (which needs a 1.4GB model).
- Entrenchment weights are hand-tuned defaults, not learned.
- No multi-process (only multi-thread) story; no distributed version.
- **Joint inconsistency**: v0.5.0 adds an n-ary constraint system
  (`add_constraint`) for user-defined joint constraints (e.g. budget).
  However, *automatic discovery* of joint inconsistencies still requires
  a pairwise detector to flag them; the constraint system handles
  known constraint types, not arbitrary k-way logical contradictions.
- **Generation gap**: the invalidation bus fires on retraction, but it
  cannot reach into an already-running LLM generation on remote GPUs.
  An agent that has already sent a prompt cannot have its context
  surgically updated mid-generation. The bus prevents *future* actions
  on stale beliefs, not in-flight ones.
- The novelty is in integration and rigor, not in any single algorithm.
- Commercial value today is as proof-of-ability, not as a product: no
  users, no revenue, no moat on the idea itself.

---

## 8. Suggested verification prompts for other models

(v0.3.0 addressed the first adversarial review: Hansson framing,
EE2/EE3 caps, ground-observation zombie prevention, invalidation bus.
v0.4.0 addressed the second: well-foundedness, Horn clauses, detector
commutativity. v0.5.0 added n-ary constraints and production hardening.
v0.6.0 added async API, entrenchment caching, temporal reasoning.
v0.7.0 added semantic detector, framework adapters, observability.)

1. "Is the entrenchment formula sound? Does weakest-first contraction
   correctly implement Hansson's non-prioritized base revision?"
2. "Is the EE2 cap (derived <= weakest premise) correctly implemented?
   Can a derived belief still exceed its premises via any path?"
3. "Is the zombie-prevention rule (orphans retract unless grounded)
   sound? Can you construct a case where an ungrounded belief survives
   premise loss, or where the zombie-bridge attack still works?"
4. "Is the novelty claim accurate? What prior art am I missing?
   (Known: yigraf does manual conflict resolution, not automated.)"
5. "What would break if two agents shared one store with conflicting
   detectors?"
6. "Is snapshot isolation + the subscription bus sufficient for
   long-running agents, or is there a deeper temporal-consistency
   problem?"
