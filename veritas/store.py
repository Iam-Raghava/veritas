"""BeliefStore: a Truth Maintenance System for agent memory.

This is the heart of Veritas. It maintains a set of beliefs that is kept
*consistent* automatically:

  ASSERTION  When a new belief contradicts existing ones, the store does
             not do "latest wins". It compares epistemic entrenchment and
             retracts the weakest-held beliefs first (Hansson-style
             contraction). If the new belief is weaker than *any*
             contradictor, it is rejected — the store refuses to corrupt
             itself with weak claims.

  PROPAGATION  Every belief records its justifications (JTMS-style). When a
             belief is retracted, all beliefs justified by it are re-evaluated:
             a derived belief that loses all justification is cascade-retracted
             unless it can stand on its own source. Changes ripple through the
             dependency graph automatically.

  AUDIT      Every assertion, retraction, cascade, and rejection is recorded
             with its reason. Nothing disappears silently.

The store is deliberately *not* an LLM wrapper. It is a deterministic
epistemic engine: detectors are pluggable, but the contraction and
propagation logic is exact.

Contradiction search is indexed, not quadratic: functional-property claims
are indexed by (subject, relation), "S is O" claims by (subject, object),
and explicit metadata pairs by direct lookup. Assertion cost is
proportional to the number of *candidate* contradictors, not store size.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import threading
from typing import Any, Callable

from .audit import (
    ASSERTED,
    CASCADE_RETRACTED,
    CASCADE_SURVIVED,
    CONSTRAINT_ERROR,
    REJECTED,
    RETRACTED,
    AuditLog,
)
from .belief import RETRACTED as _RETRACTED_STATUS, Belief
from .detect import (
    ContradictionFn,
    Detector,
    as_function,
    functional_parts,
    heuristic_contradiction,
    subject_object_parts,
)

# Deprecated: orphan survival is now decided by the belief's `ground` flag,
# not by a scalar threshold (zombie-belief prevention). Kept for
# backwards compatibility.
DEFAULT_SURVIVAL_THRESHOLD = 0.5


def _locked(fn):
    """Serialize a BeliefStore method on the instance RLock."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    return wrapper


class BeliefStore:
    """A truth-maintained collection of beliefs.

    Implements Hansson's non-prioritized belief base revision: incoming
    beliefs are screened against incumbents by epistemic entrenchment,
    and may be rejected. Contraction is weakest-first with JTMS-style
    dependency cascades. Every state change is audited.
    """

    def __init__(
        self,
        contradiction_fn: ContradictionFn | None = None,
        detector: Detector | None = None,
        entrenchment_weights: dict[str, float] | None = None,
        survival_threshold: float | None = None,
        metrics: Any | None = None,
        policy: str = "entrenchment",
    ) -> None:
        from .policies import POLICIES
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self._beliefs: dict[str, Belief] = {}
        self._audit = AuditLog()
        # Unique store identity: used by persist.save() to detect when a
        # DIFFERENT store is saved to an existing database path. Without
        # this, the append-only audit table (keyed by seq) would silently
        # drop the new store's audit events due to seq collisions.
        import uuid
        self._store_id: str = str(uuid.uuid4())
        # Observability: optional Metrics collector (veritas.observability).
        self.metrics = metrics
        # Contraction policy: entrenchment (default) or conservative.
        self.policy = policy
        if detector is not None:
            base_fn = as_function(detector)
            self.detector_name = getattr(detector, "name", "custom")
            # Keep the detector object for pre-filtering (e.g.,
            # VectorPrefilterDetector.find_contradictors).
            self.detector: Detector | None = detector
            # Wire vector pre-filter: detectors with attach_store need
            # the store reference to access active beliefs.
            if hasattr(detector, "attach_store"):
                try:
                    detector.attach_store(self)
                except Exception:
                    pass  # fail-open: pre-filter just won't run
        else:
            raw_fn = contradiction_fn or heuristic_contradiction
            # Always wrap for commutativity (symmetric contradiction).
            base_fn = as_function(raw_fn)
            self.detector = None
            self.detector_name = (
                "heuristic"
                if raw_fn is heuristic_contradiction
                else "custom"
            )
        # Temporal filter: beliefs only contradict if their validity
        # intervals overlap. Prevents "was X" vs "is not X" false positives.
        def _temporal_contradiction(a: Belief, b: Belief) -> bool:
            if not a.temporally_overlaps(b):
                return False
            return base_fn(a, b)
        self.contradiction_fn = _temporal_contradiction
        # The index only covers the heuristic detector's patterns
        # (functional properties, is/is-not, explicit pairs). Custom
        # detectors fall back to exact scan — slower but always correct.
        self._indexed = self.detector_name == "heuristic"
        self.entrenchment_weights = entrenchment_weights
        # survival_threshold is deprecated: orphan survival is now decided
        # by the belief's `ground` flag (zombie-belief prevention). Kept
        # for backwards compatibility; ignored.
        self.survival_threshold = survival_threshold
        # Indexes over ACTIVE beliefs only.
        self._by_functional: dict[tuple[str, str], set[str]] = {}
        self._by_subj_obj: dict[tuple[str, str], set[str]] = {}
        self._contradicts_rev: dict[str, set[str]] = {}
        # Reverse dependency index: premise_id -> {dependent_ids}.
        # Makes cascade lookup O(1) per node instead of O(N) scan.
        self._dependents_by_premise: dict[str, set[str]] = {}
        # Bulk-load flag: set by persist.load() to suppress per-belief
        # re-derivation during restore (declared here for mypy).
        self._bulk_loading: bool = False
        # Invalidation bus: belief_id -> [callbacks]. Fired on retraction
        # so agents holding snapshots can invalidate stale working memory.
        self._subscribers: dict[str, list[Callable]] = {}
        # N-ary constraints: name -> check function. Each check takes the
        # list of active belief snapshots and returns a list of belief IDs
        # participating in a violation (empty list = satisfied). Checked
        # after every assertion; violations trigger entrenchment-ordered
        # contraction of the weakest involved beliefs. This catches
        # joint inconsistencies that pairwise detectors miss (e.g. budget
        # + costs that sum over the budget).
        self._constraints: dict[str, Callable[[list[Belief]], list[str]]] = {}
        # Entrenchment cache: belief_id -> (generation, value). The
        # generation bumps on any mutation (assert/retract); cache hits
        # avoid the O(subgraph) traversal. Correct because entrenchment
        # is a pure function of the current graph state.
        self._ent_cache: dict[str, tuple[int, float, float]] = {}
        self._ent_gen: int = 0
        # Reentrant lock: the store is safe for concurrent use from
        # multiple threads. RLock because public methods call each other
        # (assert_belief -> contradictors_of -> ...) on the same thread.
        self._lock = threading.RLock()

    def _bump_metric(self, name: str, amount: int = 1) -> None:
        """Increment a Metrics counter if a collector is attached."""
        if self.metrics is not None:
            # Use the thread-safe incr() method, not direct setattr.
            incr = getattr(self.metrics, "incr", None)
            if callable(incr):
                incr(name, amount)
            else:
                setattr(self.metrics, name,
                        getattr(self.metrics, name, 0) + amount)

    def _ent_sort_key(self, belief: Belief) -> tuple:
        """Deterministic sort key for entrenchment ordering.

        Ties break by (-timestamp, sha256(proposition), id):
        - Older beliefs sort first (incumbent wins ties in weakest-first
          order, i.e., the newcomer is retracted first on ties).
        - SHA-256 (not Python's hash()) for cross-process determinism.
        - ID as final tie-breaker (only if timestamps AND propositions
          are identical, which is vanishingly rare).
        Using raw UUIDs for ties would make retraction order depend on
        cryptographically random IDs — different processes running the
        same assertions would retract different beliefs!
        """
        ts = belief.timestamp if belief.timestamp is not None else 0.0
        digest = hashlib.sha256(belief.proposition.encode("utf-8")).digest()
        ph = int.from_bytes(digest[:8], "big")
        return (self.entrenchment_of(belief), -ts, ph, belief.id)

    @property
    def generation(self) -> int:
        """Monotonic counter bumped on every graph mutation.

        External indexes (e.g. the vector prefilter) can compare
        this against their build generation to know when to rebuild.
        """
        with self._lock:
            return self._ent_gen

    def __deepcopy__(self, memo):
        """Deep copy without the lock (RLock can't be copied)."""
        import copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        with self._lock:
            new._beliefs = copy.deepcopy(self._beliefs, memo)
            new._audit = copy.deepcopy(self._audit, memo)
            new.contradiction_fn = self.contradiction_fn
            new.detector_name = self.detector_name
            # Detector: share the reference (detectors are stateless or
            # read-only after construction; deepcopying ML models is
            # wasteful and may fail). The copy's contradiction_fn is
            # already set above.
            new.detector = self.detector
            new._indexed = self._indexed
            new.entrenchment_weights = copy.deepcopy(
                self.entrenchment_weights, memo)
            new.survival_threshold = self.survival_threshold
            new._by_functional = copy.deepcopy(self._by_functional, memo)
            new._by_subj_obj = copy.deepcopy(self._by_subj_obj, memo)
            new._contradicts_rev = copy.deepcopy(self._contradicts_rev, memo)
            new._dependents_by_premise = copy.deepcopy(
                self._dependents_by_premise, memo)
            # Subscribers and constraints are callables bound to the
            # original owner's context; do not copy them into the new store.
            new._subscribers = {}
            new._constraints = {}
            # Entrenchment cache: start fresh (generation 0, empty).
            new._ent_cache = {}
            new._ent_gen = 0
            # Newer attributes (v0.7+): copy policy, share metrics.
            new.policy = self.policy
            new.metrics = self.metrics
            # Store identity: a deep copy is the same logical store.
            new._store_id = self._store_id
            # Bulk-loading flag: must be copied (defaults to False in
            # __init__, but __deepcopy__ bypasses __init__).
            new._bulk_loading = self._bulk_loading
        new._lock = threading.RLock()
        return new

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, belief_id: str) -> Belief | None:
        """Snapshot of the belief, or None. Mutating it is safe."""
        with self._lock:
            b = self._beliefs.get(belief_id)
            return b.snapshot() if b is not None else None

    def active_beliefs(self) -> list[Belief]:
        """Snapshots of active beliefs."""
        with self._lock:
            return [b.snapshot() for b in self._beliefs.values()
                    if b.is_active]

    def all_beliefs(self) -> list[Belief]:
        """Snapshots of all beliefs."""
        with self._lock:
            return [b.snapshot() for b in self._beliefs.values()]

    @property
    def audit(self) -> AuditLog:
        return self._audit

    @_locked
    def entrenchment_of(self, belief: Belief) -> float:
        """Entrenchment with Horn-clause EE2 cap.

        Justifications are Horn clauses: a list of AND-sets (OR across
        sets). Corroboration accumulates across independent derivations
        (number of sets), not raw premise counts. The EE2 cap is:
        ent(b) <= max(min(capped premise ents in set) for each set) —
        a belief is warranted by its BEST proof, not crippled by a weak
        alternative. Within a set, the weakest premise bounds (AND).
        Transitive via iterative post-order (no recursion limit).

        Results are cached per graph generation; the cache is
        invalidated on any mutation (assert/retract).
        """
        # Cache hit? Only for beliefs already in the store — newcomers
        # not yet stored may have their justifications mutated (doomed
        # pruning) without bumping the generation, which would make
        # the cache return stale values.
        # Cache tuple: (generation, computed_at_wallclock, value).
        # Recency decays with wall-clock time, so we invalidate entries
        # older than RECENCY_CACHE_TTL even if the generation hasn't changed.
        import time
        cached = None
        if belief.id in self._beliefs:
            cached = self._ent_cache.get(belief.id)
        if cached is not None and cached[0] == self._ent_gen:
            age = time.time() - cached[1]
            if age < 3600:  # RECENCY_CACHE_TTL: 1 hour
                if self.metrics is not None:
                    self.metrics.incr("cache_hits")
                return cached[2]
        if self.metrics is not None:
            self.metrics.incr("cache_misses")

        from .entrenchment import entrenchment

        # Iterative post-order traversal: collect all reachable beliefs,
        # then compute capped entrenchments bottom-up.
        # Phase 1: DFS to get post-order (cycle-safe).
        order = []
        visited = set()
        # Stack of (belief_id, iterator_state). Use two-phase approach:
        # push, then process children, then add to order.
        stack = [(belief.id, iter(belief.all_premise_ids))]
        # We need the belief objects; for the root, use the passed belief.
        # For others, look up in store.
        belief_map = {belief.id: belief}
        visited.add(belief.id)
        while stack:
            bid, children_iter = stack[-1]
            advanced = False
            for child_id in children_iter:
                if child_id not in visited:
                    visited.add(child_id)
                    child = self._beliefs.get(child_id)
                    if child is not None and child.is_active:
                        belief_map[child_id] = child
                        stack.append((child_id, iter(child.all_premise_ids)))
                        advanced = True
                        break
            if not advanced:
                # All children processed (or no children): add to order.
                order.append(bid)
                stack.pop()
        # Phase 2: compute capped entrenchments in post-order
        # (children before parents).
        capped: dict[str, float] = {}
        for bid in order:
            b = belief_map.get(bid)
            if b is None:
                continue
            # Filter to VALID AND-sets first: an AND-set is valid only if
            # EVERY premise is active. Dead sets (with retracted premises)
            # must not count toward corroboration nor contribute to the
            # EE2 cap. (Non-destructive cascades leave dead premises in
            # justifications; scoring must ignore them.)
            valid_sets = [
                jset for jset in b.justifications
                if jset and all(
                    (p := self._beliefs.get(pid)) is not None and p.is_active
                    for pid in jset
                )
            ]
            n_sets = len(valid_sets)
            score = entrenchment(
                b, self.entrenchment_weights, n_justifications=n_sets)
            if valid_sets:
                best_set_strength = float("-inf")
                from .detect import _norm as _canon_prop
                b_canon = _canon_prop(b.proposition)
                for jset in valid_sets:
                    # Corroboration exemption: a singleton set pointing to
                    # a belief with the SAME proposition is independent
                    # confirmation, not a derivation. It boosts n_sets
                    # (counted above) but is NOT subject to the EE2 cap —
                    # otherwise corroboration could never raise the score.
                    if (len(jset) == 1):
                        prem = self._beliefs.get(jset[0])
                        if (prem is not None
                                and _canon_prop(prem.proposition) == b_canon):
                            continue
                    set_min = float("inf")
                    for pid in jset:
                        if pid in capped:
                            set_min = min(set_min, capped[pid])
                        else:
                            # Premise active but not in order (should not
                            # happen for valid sets): use raw as fallback.
                            p = self._beliefs.get(pid)
                            if p is not None and p.is_active:
                                set_min = min(set_min, entrenchment(
                                    p, self.entrenchment_weights))
                    if set_min != float("inf"):
                        best_set_strength = max(best_set_strength, set_min)
                if best_set_strength != float("-inf"):
                    score = min(score, best_set_strength)
            capped[bid] = score
        result = capped.get(belief.id, entrenchment(
            belief, self.entrenchment_weights,
            n_justifications=len(belief.justifications)))
        # Cache for this generation (only if in store — see above).
        # Tuple: (generation, wallclock_computed_at, value).
        if belief.id in self._beliefs:
            import time
            self._ent_cache[belief.id] = (self._ent_gen, time.time(), result)
        return result

    def contradictors_of(self, belief: Belief) -> list[Belief]:
        """Snapshots of active beliefs contradicting `belief`."""
        # Fast path: detectors with a batched implementation run OUTSIDE
        # the global lock (they may do GPU inference or API calls).
        # We snapshot candidates under lock, release, then adjudicate.
        det = getattr(self, "detector", None)
        batch_fn = getattr(det, "contradicts_batch", None)
        # Only use the batch path if the detector OVERRIDES the default
        # (which just loops over contradicts — no benefit outside lock).
        # getattr with default None handles detectors that don't define
        # contradicts_batch at all (e.g., HeuristicDetector).
        _default_batch = getattr(Detector, "contradicts_batch", None)
        _det_batch = getattr(type(det), "contradicts_batch", None) if det else None
        if (det is not None and batch_fn is not None
                and _det_batch is not None
                and _det_batch is not _default_batch):
            with self._lock:
                candidates = self._candidate_snapshots_locked(belief)
            hits = batch_fn(belief, candidates)
            # Filter to still-active (may have changed during batch).
            with self._lock:
                active_ids = {b.id for b in self._beliefs.values()
                              if b.is_active}
                return [b.snapshot() for b in hits
                        if b.id in active_ids and b.id != belief.id]
        with self._lock:
            return [b.snapshot()
                    for b in self._contradictors_of_locked(belief)]

    def _candidate_snapshots_locked(self, belief: Belief) -> list[Belief]:
        """Candidate contradictors (snapshots) for batch adjudication.

        Uses the same index as _contradictors_of_locked but returns
        snapshots without running the detector. Caller must hold the lock.
        """
        if not self._indexed:
            return [b.snapshot() for b in self.active_beliefs()
                    if b.id != belief.id]
        candidate_ids: set[str] = set()
        fp = functional_parts(belief.proposition)
        if fp:
            candidate_ids.update(self._by_functional.get((fp[0], fp[1]), set()))
        sp = subject_object_parts(belief.proposition)
        if sp:
            candidate_ids.update(self._by_subj_obj.get(sp, set()))
        for cid in belief.metadata.get("contradicts", []) or []:
            candidate_ids.add(cid)
        candidate_ids.update(self._contradicts_rev.get(belief.id, set()))
        candidate_ids.discard(belief.id)
        result = []
        for cid in sorted(candidate_ids):
            other = self._beliefs.get(cid)
            if other is not None and other.is_active:
                result.append(other.snapshot())
        return result

    def _vector_candidates(self, belief: Belief) -> list[Belief] | None:
        """Top-K semantically similar candidates via vector pre-filter.

        Returns None if vector filtering is disabled or unavailable
        (caller falls back to brute force). Enabled via
        VERITAS_VECTOR_PREFILTER=1 and requires the ONNX embedder.
        """
        import os
        if os.environ.get("VERITAS_VECTOR_PREFILTER", "0") != "1":
            return None
        # Lazy embedder (cached on the store).
        if not hasattr(self, "_vector_embedder"):
            try:
                from veritas.dense import DenseEmbedder
                dense_dir = (
                    os.environ.get("VERITAS_DENSE_DIR")
                    or os.path.expanduser("~/workspace/models/bge-m3")
                )
                self._vector_embedder: DenseEmbedder | None = DenseEmbedder(
                    dense_dir)
                self._vector_cache: dict[str, object] = {}
                self._vector_cache_gen = -1
            except Exception:
                self._vector_embedder = None
        if self._vector_embedder is None:
            return None
        import numpy as np
        embedder = self._vector_embedder
        # Evict retracted beliefs (embeddings are immutable per ID;
        # no need to wipe the entire cache on every write).
        active_ids = {b.id for b in self.active_beliefs()}
        stale = [bid for bid in self._vector_cache if bid not in active_ids]
        for bid in stale:
            del self._vector_cache[bid]
        self._vector_cache_gen = self._ent_gen
        # Embed query.
        try:
            q_emb = embedder.embed([belief.proposition])[0]
        except Exception:
            return None
        candidates = [b for b in self.active_beliefs()
                      if b.id != belief.id]
        if not candidates:
            return []
        # Embed uncached candidates.
        to_embed = [b for b in candidates if b.id not in self._vector_cache]
        if to_embed:
            try:
                embs = embedder.embed([b.proposition for b in to_embed])
                for b, emb in zip(to_embed, embs):
                    self._vector_cache[b.id] = emb
            except Exception:
                return None
        # Top-K by cosine similarity.
        top_k = int(os.environ.get("VERITAS_VECTOR_TOPK", "15"))
        q_norm = np.linalg.norm(q_emb)
        scored = []
        for b in candidates:
            emb = self._vector_cache.get(b.id)
            if emb is None:
                continue
            sim = float(np.dot(q_emb, emb) /
                        (q_norm * np.linalg.norm(emb) + 1e-9))
            scored.append((sim, b))
        scored.sort(key=lambda x: -x[0])
        return [b for _, b in scored[:top_k]]

    def _contradictors_of_locked(self, belief: Belief) -> list[Belief]:
        """Active beliefs contradicting `belief`, via index + detector.

        Result is sorted by belief ID: deterministic across runs,
        independent of hash randomization.
        """
        if not self._indexed:
            # If the detector has its own candidate pre-filter (e.g.,
            # VectorPrefilterDetector.find_contradictors), use it directly.
            # This avoids O(N) scans for expensive detectors.
            _det = getattr(self, "detector", None)
            find_fn = getattr(_det, "find_contradictors", None) if _det else None
            if callable(find_fn):
                try:
                    return find_fn(belief)
                except Exception:
                    pass  # fall through to brute force
            # Custom detector: use vector pre-filter if available,
            # else exact scan (index covers heuristic patterns only).
            candidates = self._vector_candidates(belief)
            if candidates is not None:
                return [
                    b for b in candidates
                    if b.id != belief.id and self.contradiction_fn(belief, b)
                ]
            return [
                b for b in self.active_beliefs()
                if b.id != belief.id and self.contradiction_fn(belief, b)
            ]
        candidate_ids: set[str] = set()
        fp = functional_parts(belief.proposition)
        if fp:
            candidate_ids.update(self._by_functional.get((fp[0], fp[1]), set()))
        sp = subject_object_parts(belief.proposition)
        if sp:
            candidate_ids.update(self._by_subj_obj.get(sp, set()))
        for cid in belief.metadata.get("contradicts", []) or []:
            candidate_ids.add(cid)
        candidate_ids.update(self._contradicts_rev.get(belief.id, set()))
        candidate_ids.discard(belief.id)
        result = []
        # Sorted for deterministic iteration across hash seeds.
        for cid in sorted(candidate_ids):
            other = self._beliefs.get(cid)
            if other is not None and other.is_active and self.contradiction_fn(
                belief, other
            ):
                result.append(other)
        result.sort(key=lambda b: b.id)
        return result

    def _doomed_justifications(
        self, contradictor_ids: set[str],
        justifications: list[list[str]]
    ) -> set[str]:
        """Premise IDs (across all sets) that would die if `contradictor_ids`
        were retracted.

        Pure simulation of the cascade in _retract_single: no mutation,
        no audit. A belief is doomed if it is a contradictor, or if ALL
        its Horn-clause sets would die (each set dies if any premise is
        doomed) and it is not a ground observation. Mirrors the cascade's
        survival rule exactly, so the all-or-nothing decision can't borrow
        entrenchment from beliefs the assertion is about to kill
        (zombie-bridge prevention).

        Computed top-down with memoization (cycle-safe): cost is
        proportional to the justification subgraph reachable from the
        queried justifications, not to store size — this runs on the
        hot assertion path.
        """
        doomed = set(contradictor_ids)
        memo: dict[str, bool] = {}
        in_progress: set[str] = set()  # gray set: cycle guard

        def is_doomed(bid: str) -> bool:
            if bid in doomed:
                return True
            if bid in memo:
                return memo[bid]
            if bid in in_progress:
                # Cycle with no contradictor grounding on this path:
                # not doomed via this path. Matches the fixed-point
                # semantics, where an ungrounded cycle never becomes doomed.
                return False
            b = self._beliefs.get(bid)
            if b is None or not b.is_active or not b.justifications:
                memo[bid] = False
                return False
            in_progress.add(bid)
            try:
                # Doomed iff EVERY set has a doomed premise AND not grounded.
                # A ground observation survives orphaning, so it is never
                # doomed via cascade.
                result = not b.ground and all(
                    any(is_doomed(pid) for pid in jset)
                    for jset in b.justifications
                )
            finally:
                in_progress.discard(bid)
            memo[bid] = result
            if result:
                doomed.add(bid)
            return result

        # Return flat set of doomed premise IDs (across all sets).
        doomed_premises = set()
        for jset in justifications:
            for pid in jset:
                if is_doomed(pid):
                    doomed_premises.add(pid)
        return doomed_premises

    def find_contradictions(self) -> list[tuple[Belief, Belief]]:
        """All contradicting pairs among active beliefs. Empty = consistent.

        Pairs are sorted for deterministic output. Beliefs are snapshots.
        """
        with self._lock:
            pairs: list[tuple[Belief, Belief]] = []
            seen: set[tuple[str, str]] = set()
            live = [b for b in self._beliefs.values() if b.is_active]
            for b in live:
                for c in self._contradictors_of_locked(b):
                    a_id, z_id = sorted((b.id, c.id))
                    key: tuple[str, str] = (a_id, z_id)
                    if key not in seen:
                        seen.add(key)
                        pairs.append((b.snapshot(), c.snapshot()))
            pairs.sort(key=lambda p: (p[0].id, p[1].id))
            return pairs

    def is_consistent(self) -> bool:
        with self._lock:
            return not self.find_contradictions()

    def _dependents_of_locked(self, belief_id: str) -> list[Belief]:
        """LIVE dependents. Internal use only (cascade mutates them)."""
        # O(1) via reverse dependency index (vs O(N) full scan).
        # Sorted for deterministic order across hash seeds.
        dep_ids = self._dependents_by_premise.get(belief_id, set())
        return [
            self._beliefs[did] for did in sorted(dep_ids)
            if did in self._beliefs and self._beliefs[did].is_active
        ]

    def dependents_of(self, belief_id: str) -> list[Belief]:
        """Snapshots of active beliefs listing belief_id as justification."""
        with self._lock:
            return [b.snapshot()
                    for b in self._dependents_of_locked(belief_id)]

    # ------------------------------------------------------------------
    # Invalidation bus
    # ------------------------------------------------------------------

    def subscribe(
        self, belief_id: str, callback: Callable[[str, str], None]
    ) -> None:
        """Register `callback(belief_id, event_type)` for retraction events.

        Fired synchronously (under the store lock) when the belief is
        retracted or cascade-retracted. Lets long-running agents
        invalidate snapshots held in working memory instead of acting
        on stale beliefs. Callbacks must be fast and non-blocking;
        they may call back into the store (the lock is reentrant).
        """
        with self._lock:
            self._subscribers.setdefault(belief_id, []).append(callback)

    def unsubscribe(
        self, belief_id: str, callback: Callable[[str, str], None]
    ) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            cbs = self._subscribers.get(belief_id, [])
            if callback in cbs:
                cbs.remove(callback)
            if not cbs:
                self._subscribers.pop(belief_id, None)

    def _fire_subscribers(self, belief_id: str, event_type: str) -> None:
        # Pop (not get): retracted beliefs cannot be reactivated via the
        # normal API, so their callbacks would leak indefinitely.
        # (Re-derivation uses _try_rederive_dependents which re-subscribes
        # if needed.)
        import asyncio
        import inspect
        # Main event loop (set by AsyncBeliefStore): used to schedule
        # async callbacks from worker threads without creating a second
        # loop (which breaks callbacks bound to the main loop).
        main_loop = getattr(self, "_async_main_loop", None)
        for cb in self._subscribers.pop(belief_id, []):
            try:
                if inspect.iscoroutinefunction(cb):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop is not None:
                        # We're in the event loop thread: schedule directly.
                        loop.create_task(cb(belief_id, event_type))
                    elif (main_loop is not None
                            and main_loop.is_running()):
                        # Worker thread: schedule on the main loop.
                        asyncio.run_coroutine_threadsafe(
                            cb(belief_id, event_type), main_loop)
                    else:
                        # No loop available: run to completion in a new
                        # loop (sync context, no AsyncBeliefStore).
                        asyncio.run(cb(belief_id, event_type))
                else:
                    cb(belief_id, event_type)
            except Exception:
                pass  # a broken subscriber must not break retraction

    # ------------------------------------------------------------------
    # N-ary constraints (joint inconsistency)
    # ------------------------------------------------------------------

    def add_constraint(
        self,
        name: str,
        check: Callable[[list[Belief]], list[str]],
    ) -> None:
        """Register an n-ary constraint checked after every assertion.

        `check` receives snapshots of all active beliefs and returns the
        IDs of beliefs participating in a violation (empty = satisfied).
        When a violation is found, the weakest involved beliefs are
        retracted in entrenchment order until the constraint is satisfied
        (or the newcomer is rejected if it is among the weakest and the
        constraint cannot be satisfied without removing it).

        Failure policy: if `check` raises, the error is audit-logged as a
        CONSTRAINT_ERROR event and that constraint is skipped for the run
        (not treated as satisfied). Other constraints still enforce.

        Example — budget constraint:
            store.add_constraint("budget", lambda beliefs: (
                [b.id for b in beliefs if b.metadata.get("cost", 0) > 0]
                if sum(b.metadata.get("cost", 0) for b in beliefs)
                   > next((b.metadata.get("budget", float("inf"))
                           for b in beliefs if "budget" in b.metadata),
                          float("inf"))
                else []
            ))
        """
        with self._lock:
            self._constraints[name] = check

    def remove_constraint(self, name: str) -> None:
        """Remove a previously registered constraint."""
        with self._lock:
            self._constraints.pop(name, None)

    def check_constraints(self) -> dict[str, list[str]]:
        """Run all constraints; returns {name: [violating belief IDs]}.

        Failure policy: if a constraint's check function raises, the
        error is recorded in the audit log as a CONSTRAINT_ERROR event
        and that constraint is skipped (treated as unknown, not as
        satisfied). A broken constraint never blocks enforcement of
        the others, and never silently passes.
        """
        with self._lock:
            snapshots = [b.snapshot() for b in self._beliefs.values()
                         if b.is_active]
            result = {}
            for name, check in self._constraints.items():
                try:
                    violated = check(snapshots)
                except Exception as exc:
                    # Audited, not silent: a broken constraint is skipped,
                    # not treated as satisfied.
                    self._audit.record(
                        CONSTRAINT_ERROR,
                        "",
                        "",
                        f"constraint '{name}' raised {type(exc).__name__}: {exc}",
                        {"constraint": name,
                         "error": f"{type(exc).__name__}: {exc}"},
                    )
                    continue
                if violated:
                    # Only report IDs that are actually active.
                    active_ids = {b.id for b in snapshots}
                    result[name] = [vid for vid in violated
                                    if vid in active_ids]
            return result

    def _simulate_with_contra_retracted(
        self, newcomer: Belief, contra_ids: set[str], newcomer_ent: float
    ) -> set[str] | None:
        """Simulate: add newcomer, retract contradictors, check constraints.

        Returns the set of IDs that would be retracted by constraints, or
        None if the newcomer itself would be retracted (meaning: reject the
        newcomer without touching contradictors).

        This is used BEFORE the actual contradiction retraction to ensure
        transactional semantics across the contradiction+constraint pipeline.

        newcomer_ent MUST be the EE2-capped entrenchment_of(newcomer) as
        computed by the caller — the same valuation the post-retraction
        enforcement (_simulate_constraint_contraction, via
        _enforce_constraints_locked) will use. Using the raw uncapped
        entrenchment() here once let the simulation predict "newcomer
        survives" (raw > incumbents) while the real run rejected it
        (capped < incumbents) — after the contradictors were already
        retracted. A rejected assertion must leave existing information
        untouched.
        """
        # Hypothetical active set: current active - contradictors + newcomer.
        # (Newcomer not yet in _beliefs, so we handle it specially.)
        sim_active = {
            bid for bid, b in self._beliefs.items()
            if b.is_active and bid not in contra_ids
        }
        # For entrenchment ordering, we need values for all in sim_active
        # plus the newcomer. Incumbents use current (unmutated) capped
        # values; the newcomer uses the caller-supplied capped value so
        # the simulation agrees with the real enforcement pass.
        ent: dict[str, float] = {}
        for bid in sim_active:
            ent[bid] = self.entrenchment_of(self._beliefs[bid])
        ent[newcomer.id] = newcomer_ent
        # Simulate constraint check on hypothetical state.
        # We need snapshots; for newcomer, use its snapshot.
        def _check_on(hypo_active: set[str], include_newcomer: bool):
            snaps = []
            for bid in hypo_active:
                if bid == newcomer.id and include_newcomer:
                    snaps.append(newcomer.snapshot())
                elif bid in self._beliefs:
                    snaps.append(self._beliefs[bid].snapshot())
            result: dict[str, list[str]] = {}
            for name, check in self._constraints.items():
                try:
                    violated = check(snaps)
                except Exception as exc:
                    # Audited, not silent (matches check_constraints policy).
                    self._audit.record(
                        CONSTRAINT_ERROR,
                        "",
                        "",
                        f"constraint '{name}' raised {type(exc).__name__}: {exc}",
                        {"constraint": name,
                         "error": f"{type(exc).__name__}: {exc}"},
                    )
                    continue
                if violated:
                    # Filter to hypo_active
                    result[name] = [v for v in violated
                                    if v in hypo_active or
                                    (v == newcomer.id and include_newcomer)]
            return result

        # Include newcomer in the hypothetical active set.
        hypo = set(sim_active)
        hypo.add(newcomer.id)
        
        to_retract: set[str] = set()
        for _ in range(len(hypo) + 1):
            violations = _check_on(hypo, True)
            viol_ids: set[str] = set()
            for vids in violations.values():
                viol_ids.update(vids)
            if not viol_ids:
                break
            # Find weakest by entrenchment. The newcomer is valued with the
            # caller-supplied EE2-capped entrenchment (same valuation the
            # real post-retraction enforcement uses).
            def _ent_for(bid: str) -> float:
                return ent.get(bid, float('inf'))
            weakest = min(viol_ids, key=lambda bid: (_ent_for(bid), bid))
            to_retract.add(weakest)
            hypo.discard(weakest)
            if weakest == newcomer.id:
                # Newcomer must go; abort.
                return None
            # Simulate cascade for the retracted belief.
            changed = True
            while changed:
                changed = False
                for bid in list(hypo):
                    if bid == newcomer.id:
                        continue
                    b = self._beliefs.get(bid)
                    if b is None or b.ground or not b.justifications:
                        continue
                    has_valid = any(
                        all(pid in hypo for pid in jset)
                        for jset in b.justifications
                    )
                    if not has_valid:
                        to_retract.add(bid)
                        hypo.discard(bid)
                        changed = True
                        if bid == newcomer.id:
                            return None
        return to_retract

    def _simulate_constraint_contraction(
        self, newcomer_id: str | None
    ) -> set[str] | None:
        """Simulate constraint resolution WITHOUT mutating state.

        Returns the set of belief IDs that would be retracted to satisfy
        all constraints, or an empty set if no violations. Returns None
        only if violations exist but cannot be resolved (should not happen
        in practice; the real loop breaks in that case too).

        This enables transactional semantics: if the newcomer is in the
        retraction set, the caller rejects it WITHOUT touching incumbents.
        """
        active = {bid for bid, b in self._beliefs.items() if b.is_active}
        # Entrenchment on current (unmutated) state; cache is valid.
        ent = {bid: self.entrenchment_of(self._beliefs[bid]) for bid in active}

        def _check_on(sim_active: set[str]) -> dict[str, list[str]]:
            snapshots = [self._beliefs[bid].snapshot()
                         for bid in sim_active]
            result: dict[str, list[str]] = {}
            for name, check in self._constraints.items():
                try:
                    violated = check(snapshots)
                except Exception as exc:
                    # Audited, not silent (matches check_constraints policy).
                    self._audit.record(
                        CONSTRAINT_ERROR,
                        "",
                        "",
                        f"constraint '{name}' raised {type(exc).__name__}: {exc}",
                        {"constraint": name,
                         "error": f"{type(exc).__name__}: {exc}"},
                    )
                    continue  # broken constraint: skip, don't block
                if violated:
                    result[name] = [v for v in violated if v in sim_active]
            return result

        to_retract: set[str] = set()
        sim_active = set(active)
        for _ in range(len(sim_active) + 1):
            violations = _check_on(sim_active)
            viol_ids: set[str] = set()
            for vids in violations.values():
                viol_ids.update(vids)
            if not viol_ids:
                break
            weakest = min(viol_ids, key=lambda bid: (ent[bid], bid))
            to_retract.add(weakest)
            sim_active.discard(weakest)
            # Simulate JTMS cascade: iteratively drop dependents whose
            # every AND-set lost at least one premise (and who aren't
            # ground observations).
            changed = True
            while changed:
                changed = False
                for bid in list(sim_active):
                    b = self._beliefs[bid]
                    if b.ground or not b.justifications:
                        continue
                    has_valid = any(
                        all(pid in sim_active for pid in jset)
                        for jset in b.justifications
                    )
                    if not has_valid:
                        to_retract.add(bid)
                        sim_active.discard(bid)
                        changed = True
        return to_retract

    def _enforce_constraints_locked(
        self, newcomer_id: str | None
    ) -> list[Belief] | None:
        """Contract weakest violators until all constraints satisfied.

        TRANSACTIONAL: first simulates the full retraction set without
        mutating. If the newcomer must be retracted to satisfy constraints,
        returns None and the caller rejects the newcomer WITHOUT touching
        any incumbent — a rejected input leaves the belief base invariant
        (Hansson non-prioritized revision postulate).

        Returns snapshots of retracted beliefs on success.
        """
        # Fast path: no constraints registered.
        if not self._constraints:
            return []

        # Phase 1: simulate. No mutations.
        to_retract = self._simulate_constraint_contraction(newcomer_id)
        if not to_retract:
            return []
        if newcomer_id is not None and newcomer_id in to_retract:
            self._audit.record(
                REJECTED,
                newcomer_id,
                self._beliefs[newcomer_id].proposition
                if newcomer_id in self._beliefs else "",
                "rejected: constraint satisfaction requires retracting "
                "the newcomer itself; incumbents untouched",
                {"constraint_retracted": []},
            )
            return None

        # Phase 2: perform retractions for real (newcomer survives).
        # Retract in deterministic weakest-first order for audit clarity.
        # Use the newcomer's timestamp for historical consistency.
        newcomer_ts = None
        if newcomer_id is not None and newcomer_id in self._beliefs:
            newcomer_ts = self._beliefs[newcomer_id].timestamp
        ordered = sorted(
            to_retract,
            key=lambda bid: self._ent_sort_key(self._beliefs[bid]),
        )
        retracted: list[Belief] = []
        for bid in ordered:
            if bid in self._beliefs and self._beliefs[bid].is_active:
                retracted.extend(self._retract_single(
                    bid,
                    "constraint violation",
                    cascade=True,
                    timestamp=newcomer_ts,
                ))
        return retracted

    # ------------------------------------------------------------------
    # Assertion (with automatic contraction)
    # ------------------------------------------------------------------

    @_locked
    def assert_belief(
        self,
        proposition: str,
        confidence: float = 0.8,
        source: str = "unknown",
        source_reliability: float = 0.5,
        justifications: list[str] | list[list[str]] | None = None,
        metadata: dict | None = None,
        timestamp: float | None = None,
        ground: bool = False,
        valid_from: float | None = None,
        valid_until: float | None = None,
        _belief_id: str | None = None,
    ) -> Belief | None:
        """Assert a new belief, contracting contradictors if necessary.

        Returns a snapshot of the stored Belief on success, or None if the
        new belief was *rejected* (a contradictor outranks it in
        entrenchment). Rejections are audited — the store never silently
        drops input. timestamp overrides the creation time (for importing
        historical beliefs); defaults to now.

        justifications: Horn clauses — list of AND-sets (OR across sets).
        A flat list of IDs is treated as a single AND-set.

        ground=True marks a direct observation (tool output, user
        statement, sensor reading) as opposed to a derived inference.
        A belief that loses all justifications survives ONLY if grounded;
        derived beliefs cannot outlive their premises (zombie prevention).

        valid_from/valid_until: temporal validity interval. Beliefs only
        contradict if their intervals overlap. None means unbounded.
        """
        from .belief import _normalize_justifications

        # Normalize to Horn clauses, then prune dead premises.
        # A belief cannot be justified by a retracted or nonexistent belief.
        normed = _normalize_justifications(justifications)
        valid_justs: list[list[str]] = []
        pruned_justs: list[str] = []
        for jset in normed:
            # Horn-clause AND semantics: if ANY premise in a set is dead,
            # the ENTIRE set is invalid (all premises are required for the
            # derivation). We drop the whole set, not just the dead premise
            # — pruning individual premises would corrupt P1∧P2⇒C into
            # P1⇒C, an unearned derivation.
            if all(
                (tgt := self._beliefs.get(j)) is not None and tgt.is_active
                for j in jset
            ):
                valid_justs.append(jset)
            else:
                pruned_justs.extend(jset)

        # Corroboration: if an ACTIVE belief with the same normalized
        # proposition already exists, link to it. Independent sources
        # asserting the same claim boost each other's corroboration
        # (via n_justifications) instead of being isolated beliefs that
        # a single slightly-stronger contradictor can wipe out one by one.
        # (Same proposition cannot contradict itself; skip if the caller
        # already justified via this belief.)
        from .detect import _norm as _canon
        canon_prop = _canon(proposition)
        # Sort by ID for deterministic selection across hash seeds.
        for existing in sorted(self._beliefs.values(), key=lambda b: b.id):
            if (existing.is_active and _canon(existing.proposition) == canon_prop
                    and existing.id not in valid_justs
                    and all(existing.id not in js for js in valid_justs)):
                # Add as a separate AND-set (OR across sets): the new
                # belief is corroborated by the existing one, but doesn't
                # require it (independent source).
                valid_justs.append([existing.id])
                break  # One link suffices; transitivity handles the rest.

        new = Belief(
            proposition=proposition,
            confidence=confidence,
            source=source,
            source_reliability=source_reliability,
            justifications=valid_justs,
            metadata=copy.deepcopy(metadata) if metadata else {},
            timestamp=timestamp,  # None -> Belief defaults to now; 0.0 stays 0.0
            ground=ground,
            valid_from=valid_from,
            valid_until=valid_until,
            **({"id": _belief_id} if _belief_id else {}),
        )
        # _belief_id is for distributed replay; it must not collide with
        # an existing belief (that would silently overwrite history).
        if _belief_id and _belief_id in self._beliefs:
            raise ValueError(
                f"belief id {_belief_id} already exists; _belief_id "
                f"must be unique"
            )
        new_ent = self.entrenchment_of(new)

        contradictors = self.contradictors_of(new)

        # Conservative policy: reject newcomer if any contradiction.
        if self.policy == "conservative" and contradictors:
            self._audit.record(
                REJECTED,
                new.id,
                new.proposition,
                f"rejected by conservative policy: "
                f"{len(contradictors)} contradictor(s)",
                {"entrenchment": new_ent,
                 "contradictor_ids": [b.id for b in contradictors]},
            )
            self._bump_metric("rejections")
            self._bump_metric("contradictions_found", len(contradictors))
            return None

        if not contradictors:
            # Zombie-birth prevention (no-contradictor path): a non-ground
            # belief whose justifications were all pruned (dead premises)
            # is an ungrounded orphan at birth. Reject before storing —
            # the contradictor-path check below only runs when
            # contradictors exist, so this path needs its own guard.
            if not new.ground and not new.justifications and normed:
                self._audit.record(
                    REJECTED,
                    new.id,
                    new.proposition,
                    "rejected: all justifications pruned (dead premises) "
                    "and belief is not grounded; no incumbents touched",
                    {"entrenchment": new_ent,
                     "pruned_justifications": pruned_justs},
                )
                self._bump_metric("rejections")
                return None
            self._store(new)
            # Enforce n-ary constraints (joint inconsistency).
            # Transactional: returns None if the newcomer must be rejected;
            # incumbents are untouched in that case.
            c_retracted = self._enforce_constraints_locked(new.id)
            if c_retracted is None:
                # Newcomer rejected by constraint simulation. Remove it
                # (it was staged) and leave the belief base invariant.
                self._beliefs.pop(new.id, None)
                self._unindex(new)
                self._ent_gen += 1
                self._audit.record(
                    REJECTED,
                    new.id,
                    new.proposition,
                    "rejected: constraint violation could only be resolved "
                    "by retracting the newcomer; incumbents untouched",
                    {"entrenchment": new_ent},
                )
                self._bump_metric("rejections")
                return None
            if not self._beliefs.get(new.id, new).is_active:
                # Newcomer was retracted to satisfy a constraint.
                self._audit.record(
                    REJECTED,
                    new.id,
                    new.proposition,
                    "rejected: constraint violation could only be resolved "
                    "by retracting the newcomer",
                    {"entrenchment": new_ent},
                )
                self._bump_metric("rejections")
                return None
            self._audit.record(
                ASSERTED,
                new.id,
                new.proposition,
                f"no contradictions; entrenchment={new_ent:.3f}",
                {"entrenchment": new_ent,
                 "pruned_justifications": pruned_justs,
                 "constraint_retracted": [b.id for b in c_retracted],
                 # Snapshot for TimeTravel reconstruction (immutable).
                 "belief_snapshot": new.to_dict()},
                timestamp=new.timestamp,
            )
            self._bump_metric("assertions")
            return new.snapshot()

        # A belief cannot borrow entrenchment from beliefs its own
        # assertion will kill. Any justification set containing a
        # contradictor (or a belief doomed to cascade-retract with one)
        # is dead on arrival: drop it BEFORE the all-or-nothing comparison,
        # so the decision uses the entrenchment the stored belief will
        # actually have. (Without this, a newcomer justified by its own
        # contradictor could win on borrowed corroboration, then keep the
        # win after the justification was pruned post-contraction.)
        doomed_justs: list[str] = []
        if valid_justs:
            # No justifications -> nothing to borrow; skip the simulation
            # entirely (it runs on the hot assertion path).
            doomed_justs = sorted(
                self._doomed_justifications(
                    {b.id for b in contradictors}, valid_justs
                )
            )
        if doomed_justs:
            doomed_set = set(doomed_justs)
            # Drop any AND-set containing a doomed premise (AND requires all).
            new_justs = []
            for jset in valid_justs:
                if not any(pid in doomed_set for pid in jset):
                    new_justs.append(jset)
            new.justifications = new_justs
            pruned_justs.extend(doomed_justs)
            new_ent = self.entrenchment_of(new)

        # Zombie-birth prevention (BEFORE any mutations): a non-ground
        # belief whose justifications were entirely doomed is an ungrounded
        # orphan — UNLESS its independent entrenchment (after pruning)
        # still beats the contradictors. A genuinely strong newcomer
        # wins on its own merits; the doomed justification is pruned.
        if not new.ground and not new.justifications and normed:
            # Check if new_ent (independent, post-pruning) beats weakest
            # contradictor. If yes, allow it (strong newcomer wins).
            weakest_contra_ent = min(
                (self.entrenchment_of(c) for c in contradictors),
                default=float('inf'),
            )
            if new_ent < weakest_contra_ent:
                self._audit.record(
                    REJECTED,
                    new.id,
                    new.proposition,
                    "rejected: all justifications doomed and belief is not "
                    "grounded; no incumbents touched",
                    {"entrenchment": new_ent,
                     "pruned_justifications": pruned_justs},
                )
                self._bump_metric("rejections")
                self._bump_metric("contradictions_found", len(contradictors))
                return None
            # Strong newcomer: proceed to entrenchment comparison below
            # (will win, justifications already pruned).

        # All-or-nothing: the newcomer must outrank EVERY contradictor.
        # (Hansson: a screened revision rejects input that cannot beat
        # the incumbent entrenchment.)
        # Sort by CAPPED entrenchment (EE2 post-order cap via
        # self.entrenchment_of), not raw scores — a derived belief with
        # high raw confidence but a weak premise must sort as weak.
        weakest_first = sorted(
            contradictors,
            key=self._ent_sort_key,
        )
        unbeatable = [
            b for b in weakest_first if self.entrenchment_of(b) >= new_ent
        ]
        if unbeatable:
            strongest = max(unbeatable, key=self.entrenchment_of)
            self._audit.record(
                REJECTED,
                new.id,
                new.proposition,
                f"outranked by '{strongest.proposition}' "
                f"(entrenchment {self.entrenchment_of(strongest):.3f} >= {new_ent:.3f})",
                {
                    "new_entrenchment": new_ent,
                    "blocker_id": strongest.id,
                    "blocker_entrenchment": self.entrenchment_of(strongest),
                    "n_contradictors": len(contradictors),
                    "pruned_justifications": pruned_justs,
                },
            )
            self._bump_metric("rejections")
            self._bump_metric("contradictions_found", len(contradictors))
            return None

        self._bump_metric("contradictions_found", len(contradictors))
        # TRANSACTIONAL: simulate constraint satisfaction BEFORE retracting
        # contradictors. If the newcomer would be rejected by constraints
        # (after contradictor retraction), abort now without mutating.
        # This preserves Hansson's postulate: rejected input leaves the
        # belief base invariant.
        contra_ids = {b.id for b in contradictors}
        # new_ent is the EE2-capped entrenchment_of(new) after all
        # justification pruning — the same valuation the real
        # post-retraction enforcement will use for the newcomer.
        sim_result = self._simulate_with_contra_retracted(
            new, contra_ids, new_ent)
        if sim_result is None:
            # Newcomer would not survive constraints; reject without
            # touching contradictors.
            self._audit.record(
                REJECTED,
                new.id,
                new.proposition,
                "rejected: constraint simulation shows newcomer cannot "
                "survive; contradictors untouched",
                {"entrenchment": new_ent,
                 "n_contradictors": len(contradictors)},
            )
            self._bump_metric("rejections")
            return None

        # Contract weakest-first, with cascading propagation.
        retracted: list[Belief] = []
        for old in weakest_first:
            retracted.extend(
                self._retract_single(
                    old.id,
                    f"contradicted by '{new.proposition}' "
                    f"(new entrenchment {new_ent:.3f} > "
                    f"{self.entrenchment_of(old):.3f})",
                    cascade=True,
                    timestamp=new.timestamp,
                )
            )

        # Prune justifications that died in the contraction.
        # Remove dead premises from all sets; drop empty sets.
        dead = {b.id for b in retracted}
        # Drop any AND-set containing a dead premise.
        surviving_justs: list[list[str]] = []
        for jset in new.justifications:
            if not any(pid in dead for pid in jset):
                surviving_justs.append(jset)
        new.justifications = surviving_justs

        # Note: zombie-birth was already checked BEFORE retractions (above).
        # If we reach here, the newcomer has valid justifications or is
        # ground. The post-contraction pruning above cannot create a zombie
        # from a non-zombie (it only removes sets, and we checked the
        # doomed case earlier).

        self._store(new)
        # Enforce n-ary constraints (joint inconsistency).
        # Transactional: returns None if the newcomer must be rejected;
        # incumbents are untouched in that case.
        c_retracted = self._enforce_constraints_locked(new.id)
        if c_retracted is None:
            # Newcomer rejected by constraint simulation. Remove it
            # (it was staged) and leave the belief base invariant.
            self._beliefs.pop(new.id, None)
            self._unindex(new)
            self._ent_gen += 1
            self._audit.record(
                REJECTED,
                new.id,
                new.proposition,
                "rejected: constraint violation could only be resolved "
                "by retracting the newcomer; incumbents untouched",
                {"entrenchment": new_ent},
            )
            self._bump_metric("rejections")
            return None
        if not self._beliefs.get(new.id, new).is_active:
            self._audit.record(
                REJECTED,
                new.id,
                new.proposition,
                "rejected: constraint violation could only be resolved "
                "by retracting the newcomer",
                {"entrenchment": new_ent},
            )
            self._bump_metric("rejections")
            return None
        self._audit.record(
            ASSERTED,
            new.id,
            new.proposition,
            f"asserted after contracting {len(retracted)} belief(s); "
            f"entrenchment={new_ent:.3f}",
            {"entrenchment": new_ent,
             "retracted_ids": [b.id for b in retracted],
             "pruned_justifications": pruned_justs,
             "constraint_retracted": [b.id for b in c_retracted],
             # Snapshot for TimeTravel reconstruction (immutable).
             "belief_snapshot": new.to_dict()},
            timestamp=new.timestamp,
        )
        self._bump_metric("assertions")
        return new.snapshot()

    @_locked
    def assert_batch(
        self, items: list[dict],
    ) -> list[Belief | None]:
        """Assert multiple beliefs in a single locked pass.

        Sorts items topologically by justification dependencies (premises
        before dependents) using Kahn's algorithm, then asserts each via
        assert_belief. Holds the lock once for the entire batch.

        Each item is a dict of kwargs for assert_belief (proposition,
        justifications, etc.). Items may reference each other via a
        temporary '_id' key in justifications; these are resolved to
        real belief IDs after assertion.

        Returns a list of snapshots (or None for rejected) in input order.
        """
        # Topological sort: premises before dependents.
        # Build dependency graph from justification references.
        # For simplicity, we sort by: ground beliefs first, then those
        # whose justifications reference earlier items.
        n = len(items)
        # Map from index to set of indices it depends on.
        # We detect dependencies via '_ref' markers in justifications.
        # If no explicit refs, preserve input order (stable).
        order = list(range(n))
        # True topological sort via Kahn's algorithm.
        # Build adjacency: temp _id -> index, and dependencies.
        id_to_idx: dict[str, int] = {}
        for i, item in enumerate(items):
            tid = item.get("_id")
            if tid:
                id_to_idx[tid] = i
        # For each item, find which other batch items it depends on.
        deps: list[set[int]] = [set() for _ in range(n)]
        for i, item in enumerate(items):
            justs = item.get("justifications") or []
            for j in justs:
                # j is a premise ref (str) or a list of refs (AND-set).
                if isinstance(j, list):
                    refs = j
                else:
                    refs = [j]
                for ref in refs:
                    if ref in id_to_idx and id_to_idx[ref] != i:
                        deps[i].add(id_to_idx[ref])
        # Kahn's: in-degree = number of dependencies.
        in_deg = [len(d) for d in deps]
        # Reverse adjacency: who depends on me.
        dependents: list[list[int]] = [[] for _ in range(n)]
        for i, ds in enumerate(deps):
            for d in ds:
                dependents[d].append(i)
        # Start with zero in-degree, in input order (stable).
        from collections import deque
        queue: deque[int] = deque(
            i for i in range(n) if in_deg[i] == 0)
        order = []
        while queue:
            i = queue.popleft()
            order.append(i)
            for dep_idx in dependents[i]:
                in_deg[dep_idx] -= 1
                if in_deg[dep_idx] == 0:
                    queue.append(dep_idx)
        # Cycle fallback: detect circular justifications. Items in a cycle
        # cannot be topologically ordered; if they're non-ground with no
        # external ground anchor, they'd be rejected as zombies anyway.
        # Log a warning and append in input order (best effort).
        if len(order) < n:
            remaining = [i for i in range(n) if i not in order]
            # Check if cycle members have external (non-cycle) justifications.
            cycle_set = set(remaining)
            for i in remaining:
                item = items[i]
                justs = item.get("justifications") or []
                # If non-ground and all justifications are within the cycle,
                # this will fail as zombie — warn now.
                if not item.get("ground") and justs:
                    all_internal = True
                    for j in justs:
                        refs = j if isinstance(j, list) else [j]
                        for ref in refs:
                            if ref in id_to_idx and id_to_idx[ref] not in cycle_set:
                                all_internal = False
                                break
                    if all_internal:
                        import warnings
                        warnings.warn(
                            f"assert_batch: circular justification detected "
                            f"for item {i} (proposition: "
                            f"{item.get('proposition', '?')[:50]}); "
                            f"may be rejected as ungrounded"
                        )
            order.extend(remaining)
        # Assert in topological order. The @_locked decorator holds
        # the lock once for the entire batch (RLock is reentrant, so
        # the nested assert_belief calls are safe).
        results: list[Belief | None] = [None] * n
        # Resolve _id references: allow items to reference each other.
        id_map: dict[str, str] = {}  # temp _id -> real belief id
        for idx in order:
            kwargs = dict(items[idx])
            # Resolve justification references.
            justs = kwargs.get("justifications")
            if justs:
                resolved = []
                for jset in justs:
                    if isinstance(jset, list):
                        resolved.append([id_map.get(p, p) for p in jset])
                    else:
                        resolved.append(id_map.get(jset, jset))
                kwargs["justifications"] = resolved
            temp_id = kwargs.pop("_id", None)
            result = self.assert_belief(**kwargs)
            results[idx] = result
            if temp_id and result is not None:
                id_map[temp_id] = result.id
        return results

    # ------------------------------------------------------------------
    # Retraction (manual + automatic propagation)
    # ------------------------------------------------------------------

    @_locked
    def retract_belief(
        self, belief_id: str, reason: str = "manual retraction"
    ) -> list[Belief]:
        """Retract a belief and propagate through its dependents.

        Returns every belief retracted (the target plus all cascades).
        """
        belief = self._beliefs.get(belief_id)
        if belief is None or not belief.is_active:
            return []
        return self._retract_single(belief_id, reason, cascade=True)

    def _has_ground_path_locked(self, belief_id: str) -> bool:
        """Check if belief has a valid Horn-clause proof from ground observations.

        Well-foundedness (Doyle 1979) for AND-OR hypergraphs: a belief is
        valid iff it is ground itself, or there EXISTS an AND-set in its
        justifications such that EVERY premise in that set is well-founded
        (recursive). This is AND-OR semantics, not OR-reachability:
        P1∧P2⇒C requires BOTH P1 and P2 to be grounded, not just one.

        Computed bottom-up via standard Horn-clause marking (fixed-point
        iteration). This is cycle-safe by construction: a belief is marked
        well-founded only when all premises of one of its AND-sets are
        already marked. Cycle-dependent memoization bugs are impossible.
        """
        # Note: we do NOT require the root belief to be active — this
        # allows evaluating whether a RETRACTED belief would be
        # well-founded if re-activated (for TMS re-derivation).
        # Premises must still be active.
        b0 = self._beliefs.get(belief_id)
        if b0 is None:
            return False
        if b0.ground:
            return True
        if not b0.justifications:
            return False

        # Bottom-up marking: start with active ground beliefs, iteratively
        # add any belief with a fully-marked AND-set, until fixed point.
        marked: set[str] = {
            bid for bid, b in self._beliefs.items()
            if b.is_active and b.ground
        }
        # The root is exempt from the active check (see above), but its
        # premises must be active and marked.
        changed = True
        while changed:
            changed = False
            for bid, b in self._beliefs.items():
                if bid in marked:
                    continue
                # Root exempt from active check; others must be active.
                if bid != belief_id and not b.is_active:
                    continue
                if not b.justifications:
                    continue
                for jset in b.justifications:
                    if jset and all(pid in marked for pid in jset):
                        marked.add(bid)
                        changed = True
                        break
        return belief_id in marked


    def _retract_single(
        self, belief_id: str, reason: str, cascade: bool,
        timestamp: float | None = None,
    ) -> list[Belief]:
        belief = self._beliefs.get(belief_id)
        if belief is None or not belief.is_active:
            return []
        belief.status = _RETRACTED_STATUS
        self._unindex(belief)
        # Graph mutated: bump entrenchment cache generation.
        self._ent_gen += 1
        retracted = [belief]
        self._audit.record(
            RETRACTED,
            belief.id,
            belief.proposition,
            reason,
            {"entrenchment": self.entrenchment_of(belief)},
            timestamp=timestamp,
        )
        self._fire_subscribers(belief.id, RETRACTED)
        if cascade:
            # Iterative cascade (explicit stack, not recursion): a deep
            # justification chain must not blow the Python stack.
            # Each item is (dependent_belief, id_of_lost_justification).
            stack: list[tuple[Belief, str]] = [
                (dep, belief_id) for dep in self._dependents_of_locked(belief_id)
            ]
            while stack:
                dependent, lost_id = stack.pop()
                if not dependent.is_active:
                    continue
                # NON-DESTRUCTIVE cascade (Doyle TMS semantics):
                # We do NOT mutate dependent.justifications. The justification
                # structure is permanent; only status toggles IN/OUT.
                # The well-foundedness check naturally accounts for the lost
                # premise because it is now inactive — any AND-set containing
                # it will fail the check, but the structure is preserved for
                # potential future re-derivation if the premise is re-asserted.
                if dependent.justifications:
                    # Well-foundedness check (Doyle 1979): a belief with
                    # justifications survives ONLY if it has a valid
                    # Horn-clause proof from ground observations, where
                    # every premise in at least one AND-set is active and
                    # well-founded. This prevents circular self-justifying
                    # loops (A<->B with no ground) from sustaining each other.
                    if self._has_ground_path_locked(dependent.id):
                        self._audit.record(
                            CASCADE_SURVIVED,
                            dependent.id,
                            dependent.proposition,
                            f"lost justification '{lost_id}' but retains "
                            f"valid proof via remaining justifications; "
                            f"survives (structure preserved)",
                            {"lost_justification": lost_id},
                            timestamp=timestamp,
                        )
                        continue
                    # Has justifications but no valid proof: circular or
                    # ungrounded. Must retract (fall through to retraction).
                    # Justifications are PRESERVED for potential re-derivation.

                # Orphaned: zombie-belief prevention. A derived belief
                # cannot outlive its premises. It survives ONLY if
                # grounded in direct observation (tool output, user
                # statement); otherwise it must retract.
                if dependent.ground:
                    self._audit.record(
                        CASCADE_SURVIVED,
                        dependent.id,
                        dependent.proposition,
                        "lost all justifications but survives as a "
                        "ground observation",
                        {"lost_justification": lost_id,
                         "ground": True},
                        timestamp=timestamp,
                    )
                    continue

                dependent.status = _RETRACTED_STATUS
                self._unindex(dependent)
                retracted.append(dependent)
                if dependent.justifications:
                    reason_msg = (
                        f"cascade-retracted: retains justifications but "
                        f"has no path to ground via '{lost_id}' "
                        f"(circular/ungrounded)"
                    )
                else:
                    reason_msg = (
                        f"cascade-retracted: lost all justifications via "
                        f"'{lost_id}' and is not a ground observation"
                    )
                self._audit.record(
                    CASCADE_RETRACTED,
                    dependent.id,
                    dependent.proposition,
                    reason_msg,
                    {
                        "survived": False,
                        "lost_justification": lost_id,
                    },
                    timestamp=timestamp,
                )
                self._fire_subscribers(dependent.id, CASCADE_RETRACTED)
                for sub in self._dependents_of_locked(dependent.id):
                    stack.append((sub, dependent.id))
        # Metrics: every retracted belief counts; a cascade is a
        # retraction event that took dependents with it.
        self._bump_metric("retractions", len(retracted))
        if cascade and len(retracted) > 1:
            self._bump_metric("cascades")
            self._bump_metric("cascade_size_total", len(retracted) - 1)
        return [b.snapshot() for b in retracted]

    # ------------------------------------------------------------------
    # Index maintenance
    # ------------------------------------------------------------------

    def _index(self, belief: Belief) -> None:
        fp = functional_parts(belief.proposition)
        if fp:
            self._by_functional.setdefault((fp[0], fp[1]), set()).add(belief.id)
        sp = subject_object_parts(belief.proposition)
        if sp:
            self._by_subj_obj.setdefault(sp, set()).add(belief.id)
        for cid in belief.metadata.get("contradicts", []) or []:
            self._contradicts_rev.setdefault(cid, set()).add(belief.id)
        # Reverse dependency index for O(1) cascade lookup.
        for pid in belief.all_premise_ids:
            self._dependents_by_premise.setdefault(pid, set()).add(belief.id)

    def _unindex(self, belief: Belief) -> None:
        fp = functional_parts(belief.proposition)
        if fp:
            self._by_functional.get((fp[0], fp[1]), set()).discard(belief.id)
        sp = subject_object_parts(belief.proposition)
        if sp:
            self._by_subj_obj.get(sp, set()).discard(belief.id)
        for cid in belief.metadata.get("contradicts", []) or []:
            self._contradicts_rev.get(cid, set()).discard(belief.id)
        for pid in belief.all_premise_ids:
            self._dependents_by_premise.get(pid, set()).discard(belief.id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _store(self, belief: Belief) -> None:
        self._beliefs[belief.id] = belief
        if belief.is_active:
            self._index(belief)
        # Graph mutated: bump entrenchment cache generation.
        self._ent_gen += 1
        # TMS re-derivation: if this belief was retracted and is now
        # re-asserted (or a new belief with the same ID pattern), check
        # retracted dependents — their preserved justifications may now
        # be satisfiable again.
        # Skip during bulk load (persist.load sets _bulk_loading=True):
        # re-derivation on every row is O(N^2). The loaded state is
        # already consistent; no re-derivation needed.
        if belief.is_active and not self._bulk_loading:
            self._try_rederive_dependents(belief.id)

    def _try_rederive_dependents(self, belief_id: str) -> None:
        """Re-evaluate retracted dependents after a premise is (re-)asserted.

        Classical TMS behavior: when a premise comes back IN, dependents
        whose justifications are now satisfiable return to IN automatically.
        This is only possible because cascades are non-destructive.

        CRITICAL: re-derivation must check for contradictors. A belief that
        was retracted due to contradiction (not just cascade) must NOT be
        blindly revived — it must win an entrenchment adjudication against
        current active contradictors, or stay retracted.
        """
        # Find retracted beliefs that depend on this premise.
        # Sorted by ID for deterministic re-derivation order across
        # hash seeds.
        for b in sorted(self._beliefs.values(), key=lambda x: x.id):
            if b.is_active:
                continue
            if not b.justifications:
                continue
            # Does this belief depend on the (re-)asserted premise?
            depends = any(
                belief_id in jset for jset in b.justifications
            )
            if not depends:
                continue
            # Is it now well-founded? (All premises in some AND-set active.)
            if not self._has_ground_path_locked(b.id):
                continue
            # CONTRADICTION CHECK: if active contradictors exist, the
            # re-derived belief must outrank them via entrenchment.
            # Otherwise we'd revive a contradicted belief and break
            # consistency.
            contradictors = self._contradictors_of_locked(b)
            if contradictors:
                b_ent = self.entrenchment_of(b)
                # Must beat EVERY contradictor (screened revision).
                if any(self.entrenchment_of(c) >= b_ent
                       for c in contradictors):
                    # Outranked: stay retracted.
                    self._audit.record(
                        "rederived_blocked",
                        b.id,
                        b.proposition,
                        "re-derivation blocked: contradicted by stronger "
                        "active belief",
                        {"premise": belief_id,
                         "n_contradictors": len(contradictors)},
                    )
                    continue
                # DOOMED CHECK: if B's justifications depend on beliefs
                # that would be doomed by retracting the contradictors,
                # B is borrowing warrant — block re-derivation.
                # (Mirrors _doomed_justifications in assert_belief.)
                contra_ids = {c.id for c in contradictors}
                doomed = self._doomed_justifications(
                    contra_ids, b.justifications)
                if doomed:
                    self._audit.record(
                        "rederived_blocked",
                        b.id,
                        b.proposition,
                        "re-derivation blocked: justifications doomed "
                        "if contradictors retracted",
                        {"premise": belief_id,
                         "doomed": list(doomed)},
                    )
                    continue
                # Outranks all and not doomed: check N-ary constraints
                # HYPOTHETICALLY before retracting contradictors.
                # Simulate: active - contradictors + B.
                if self._constraints:
                    contra_ids_sim = {c.id for c in contradictors}
                    sim_beliefs = [
                        bb.snapshot() for bid, bb in self._beliefs.items()
                        if (bb.is_active and bid not in contra_ids_sim)
                        or bid == b.id
                    ]
                    violated_ids: set[str] = set()
                    for _name, _check in self._constraints.items():
                        try:
                            vids = _check(sim_beliefs)
                            violated_ids.update(vids)
                        except Exception:
                            continue
                    if b.id in violated_ids:
                        # B would violate constraints; block WITHOUT
                        # touching contradictors (transactional).
                        self._audit.record(
                            "rederived_blocked",
                            b.id,
                            b.proposition,
                            "re-derivation blocked: N-ary constraint "
                            "violation (contradictors untouched)",
                            {"premise": belief_id},
                        )
                        continue
                # Constraints pass: retract the contradictors.
                for c in contradictors:
                    self._retract_single(
                        c.id,
                        f"contradicted by re-derived '{b.proposition}'",
                        cascade=True,
                    )
            # Safe to re-derive.
            b.status = "active"
            self._index(b)
            self._ent_gen += 1
            self._audit.record(
                "rederived",
                b.id,
                b.proposition,
                f"re-derived: premise '{belief_id}' re-asserted, "
                f"justification now satisfiable",
                {"premise": belief_id,
                 "belief_snapshot": b.to_dict()},
            )
            # Recursively re-derive ITS dependents.
            self._try_rederive_dependents(b.id)

    @_locked
    def stats(self) -> dict:
        active = self.active_beliefs()
        return {
            "total": len(self._beliefs),
            "active": len(active),
            "retracted": len(self._beliefs) - len(active),
            "consistent": self.is_consistent(),
            "audit_events": len(self._audit),
            "detector": self.detector_name,
        }
