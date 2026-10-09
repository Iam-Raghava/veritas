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

import functools
import threading
from typing import Any, Callable

from .audit import (
    ASSERTED,
    CASCADE_RETRACTED,
    REJECTED,
    RETRACTED,
    AuditLog,
)
from .belief import ACTIVE, RETRACTED as _RETRACTED_STATUS, Belief
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
        # Observability: optional Metrics collector (veritas.observability).
        self.metrics = metrics
        # Contraction policy: entrenchment (default), maxichoice, conservative.
        self.policy = policy
        if detector is not None:
            base_fn = as_function(detector)
            self.detector_name = getattr(detector, "name", "custom")
        else:
            raw_fn = contradiction_fn or heuristic_contradiction
            # Always wrap for commutativity (symmetric contradiction).
            base_fn = as_function(raw_fn)
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
        self._ent_cache: dict[str, tuple[int, float]] = {}
        self._ent_gen: int = 0
        # Reentrant lock: the store is safe for concurrent use from
        # multiple threads. RLock because public methods call each other
        # (assert_belief -> contradictors_of -> ...) on the same thread.
        self._lock = threading.RLock()

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
            new._indexed = self._indexed
            new.entrenchment_weights = copy.deepcopy(
                self.entrenchment_weights, memo)
            new.survival_threshold = self.survival_threshold
            new._by_functional = copy.deepcopy(self._by_functional, memo)
            new._by_subj_obj = copy.deepcopy(self._by_subj_obj, memo)
            new._contradicts_rev = copy.deepcopy(self._contradicts_rev, memo)
            # Subscribers and constraints are callables bound to the
            # original owner's context; do not copy them into the new store.
            new._subscribers = {}
            new._constraints = {}
            # Entrenchment cache: start fresh (generation 0, empty).
            new._ent_cache = {}
            new._ent_gen = 0
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
        # Cache hit?
        cached = self._ent_cache.get(belief.id)
        if cached is not None and cached[0] == self._ent_gen:
            if self.metrics is not None:
                self.metrics.cache_hits += 1
            return cached[1]
        if self.metrics is not None:
            self.metrics.cache_misses += 1

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
            n_sets = len(b.justifications)
            score = entrenchment(
                b, self.entrenchment_weights, n_justifications=n_sets)
            if b.justifications:
                best_set_strength = float("-inf")
                for jset in b.justifications:
                    set_min = float("inf")
                    for pid in jset:
                        if pid in capped:
                            set_min = min(set_min, capped[pid])
                        else:
                            # Premise not in order (inactive/missing):
                            # use raw as fallback.
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
        # Cache for this generation.
        self._ent_cache[belief.id] = (self._ent_gen, result)
        return result

    def contradictors_of(self, belief: Belief) -> list[Belief]:
        """Snapshots of active beliefs contradicting `belief`."""
        with self._lock:
            return [b.snapshot()
                    for b in self._contradictors_of_locked(belief)]

    def _contradictors_of_locked(self, belief: Belief) -> list[Belief]:
        """Active beliefs contradicting `belief`, via index + detector.

        Result is sorted by belief ID: deterministic across runs,
        independent of hash randomization.
        """
        if not self._indexed:
            # Custom detector: exact scan (index covers heuristic patterns only).
            return [
                b for b in self.active_beliefs()
                if b.id != belief.id and self.contradiction_fn(belief, b)
            ]
        candidates: set[str] = set()
        fp = functional_parts(belief.proposition)
        if fp:
            candidates.update(self._by_functional.get((fp[0], fp[1]), set()))
        sp = subject_object_parts(belief.proposition)
        if sp:
            candidates.update(self._by_subj_obj.get(sp, set()))
        for cid in belief.metadata.get("contradicts", []) or []:
            candidates.add(cid)
        candidates.update(self._contradicts_rev.get(belief.id, set()))
        candidates.discard(belief.id)
        result = []
        for cid in candidates:
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
        return [
            b for b in self._beliefs.values()
            if b.is_active and belief_id in b.all_premise_ids
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
        for cb in list(self._subscribers.get(belief_id, [])):
            try:
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
        """Run all constraints; returns {name: [violating belief IDs]}."""
        with self._lock:
            snapshots = [b.snapshot() for b in self._beliefs.values()
                         if b.is_active]
            result = {}
            for name, check in self._constraints.items():
                try:
                    violated = check(snapshots)
                except Exception:
                    # A broken constraint must not break the store.
                    violated = []
                if violated:
                    # Only report IDs that are actually active.
                    active_ids = {b.id for b in snapshots}
                    result[name] = [vid for vid in violated
                                    if vid in active_ids]
            return result

    def _enforce_constraints_locked(
        self, newcomer_id: str | None
    ) -> list[Belief]:
        """Contract weakest violators until all constraints satisfied.

        Returns snapshots of retracted beliefs. If the newcomer itself
        must go to satisfy a constraint, it is retracted (caller treats
        the assertion as rejected).
        """
        from .entrenchment import order_by_entrenchment

        # Fast path: no constraints registered.
        if not self._constraints:
            return []

        retracted: list[Belief] = []
        # Bound iterations: each pass retracts at least one belief.
        for _ in range(len(self._beliefs) + 1):
            violations = self.check_constraints()
            if not violations:
                break
            # Collect all violating IDs across constraints.
            viol_ids: set[str] = set()
            for vids in violations.values():
                viol_ids.update(vids)
            viol_beliefs = [
                self._beliefs[vid] for vid in viol_ids
                if vid in self._beliefs and self._beliefs[vid].is_active
            ]
            if not viol_beliefs:
                break
            ordered = order_by_entrenchment(
                viol_beliefs, self.entrenchment_weights, ascending=True)
            weakest = ordered[0]
            # _retract_single audits and cascades; returns snapshots.
            retracted.extend(self._retract_single(
                weakest.id,
                f"constraint violation {list(violations.keys())}",
                cascade=True,
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
        from .entrenchment import order_by_entrenchment

        # Normalize to Horn clauses, then prune dead premises.
        # A belief cannot be justified by a retracted or nonexistent belief.
        normed = _normalize_justifications(justifications)
        valid_justs: list[list[str]] = []
        pruned_justs: list[str] = []
        for jset in normed:
            valid_set = []
            for j in jset:
                target = self._beliefs.get(j)
                if target is not None and target.is_active:
                    valid_set.append(j)
                else:
                    pruned_justs.append(j)
            # Drop empty sets (all premises dead); keep non-empty.
            if valid_set:
                valid_justs.append(valid_set)

        new = Belief(
            proposition=proposition,
            confidence=confidence,
            source=source,
            source_reliability=source_reliability,
            justifications=valid_justs,
            metadata=dict(metadata or {}),
            timestamp=timestamp or 0.0,
            ground=ground,
            valid_from=valid_from,
            valid_until=valid_until,
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
            return None

        if not contradictors:
            self._store(new)
            # Enforce n-ary constraints (joint inconsistency).
            c_retracted = self._enforce_constraints_locked(new.id)
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
                return None
            self._audit.record(
                ASSERTED,
                new.id,
                new.proposition,
                f"no contradictions; entrenchment={new_ent:.3f}",
                {"entrenchment": new_ent,
                 "pruned_justifications": pruned_justs,
                 "constraint_retracted": [b.id for b in c_retracted]},
            )
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

        # All-or-nothing: the newcomer must outrank EVERY contradictor.
        # (Hansson: a screened revision rejects input that cannot beat
        # the incumbent entrenchment.)
        weakest_first = order_by_entrenchment(
            contradictors, self.entrenchment_weights, ascending=True
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

        self._store(new)
        # Enforce n-ary constraints (joint inconsistency).
        c_retracted = self._enforce_constraints_locked(new.id)
        if not self._beliefs.get(new.id, new).is_active:
            self._audit.record(
                REJECTED,
                new.id,
                new.proposition,
                "rejected: constraint violation could only be resolved "
                "by retracting the newcomer",
                {"entrenchment": new_ent},
            )
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
             "constraint_retracted": [b.id for b in c_retracted]},
        )
        return new.snapshot()

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
        """Check if belief has a directed path to a ground observation.

        Well-foundedness (Doyle 1979): a belief is valid iff it is
        ground itself or can reach a ground belief via active
        justifications (across all Horn-clause sets). Cycle-safe via
        visited set. Used by the cascade to detect circular
        self-justifying loops.
        """
        seen: set[str] = set()
        stack = [belief_id]
        while stack:
            bid = stack.pop()
            if bid in seen:
                continue
            seen.add(bid)
            b = self._beliefs.get(bid)
            if b is None or not b.is_active:
                continue
            if b.ground:
                return True
            stack.extend(b.all_premise_ids)
        return False

    def _retract_single(
        self, belief_id: str, reason: str, cascade: bool
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
                # Remove lost_id from all AND-sets. For AND semantics,
                # if ANY premise in a set is lost, the ENTIRE set dies
                # (all premises are required). Drop dead sets.
                new_justs = []
                for jset in dependent.justifications:
                    if lost_id not in jset:
                        new_justs.append(jset)
                dependent.justifications = new_justs

                if dependent.justifications:
                    # Well-foundedness check (Doyle 1979): a belief with
                    # justifications survives ONLY if it has a directed
                    # path to a ground observation. This prevents circular
                    # self-justifying loops (A<->B with no ground) from
                    # sustaining each other indefinitely.
                    if self._has_ground_path_locked(dependent.id):
                        self._audit.record(
                            CASCADE_RETRACTED,
                            dependent.id,
                            dependent.proposition,
                            f"lost justification '{lost_id}' but retains "
                            f"{len(dependent.justifications)} set(s) with "
                            f"ground path; survives",
                            {"survived": True,
                             "lost_justification": lost_id},
                        )
                        continue
                    # Has justifications but no ground path: circular or
                    # ungrounded. Must retract (fall through to retraction).

                # Orphaned: zombie-belief prevention. A derived belief
                # cannot outlive its premises. It survives ONLY if
                # grounded in direct observation (tool output, user
                # statement); otherwise it must retract.
                if dependent.ground:
                    self._audit.record(
                        CASCADE_RETRACTED,
                        dependent.id,
                        dependent.proposition,
                        f"lost all justifications but survives as a "
                        f"ground observation",
                        {"survived": True,
                         "lost_justification": lost_id,
                         "ground": True},
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
                )
                self._fire_subscribers(dependent.id, CASCADE_RETRACTED)
                for sub in self._dependents_of_locked(dependent.id):
                    stack.append((sub, dependent.id))
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

    def _unindex(self, belief: Belief) -> None:
        fp = functional_parts(belief.proposition)
        if fp:
            self._by_functional.get((fp[0], fp[1]), set()).discard(belief.id)
        sp = subject_object_parts(belief.proposition)
        if sp:
            self._by_subj_obj.get(sp, set()).discard(belief.id)
        for cid in belief.metadata.get("contradicts", []) or []:
            self._contradicts_rev.get(cid, set()).discard(belief.id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _store(self, belief: Belief) -> None:
        self._beliefs[belief.id] = belief
        if belief.is_active:
            self._index(belief)
        # Graph mutated: bump entrenchment cache generation.
        self._ent_gen += 1

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
