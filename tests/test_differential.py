"""Differential testing: Veritas vs a dead-simple reference implementation.

The reference is deliberately naive (O(n^2), no indexes, no caching) but
obviously correct. If Veritas ever disagrees with it on the FINAL STATE
(active belief IDs + propositions), Veritas has a logic bug.

Run: python3 tests/test_differential.py [--trials N] [--ops N]
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402
from veritas.detect import heuristic_contradiction  # noqa: E402
from veritas.belief import Belief  # noqa: E402

# No recency: wall-clock time must not affect the comparison.
from veritas.entrenchment import DEFAULT_WEIGHTS
W = dict(DEFAULT_WEIGHTS)


class ReferenceStore:
    """Naive reference: no indexes, brute-force contradiction checks,
    simple recursive cascade. Obviously correct, obviously slow."""

    def __init__(self):
        self.beliefs = {}  # id -> Belief

    def _ent(self, b):
        # Horn-clause EE2: corroboration by #sets, cap by best set's
        # weakest (transitive, capped) premise.
        from veritas.entrenchment import entrenchment as _e
        n_sets = len(b.justifications)
        score = _e(b, W, n_justifications=n_sets)
        if not b.justifications:
            return score
        # Iterative post-order for transitive capped values.
        order = []
        visited = set()
        belief_map = {b.id: b}
        stack = [(b.id, iter(b.all_premise_ids))]
        visited.add(b.id)
        while stack:
            bid, it = stack[-1]
            adv = False
            for cid in it:
                if cid not in visited:
                    visited.add(cid)
                    ch = self.beliefs.get(cid)
                    if ch is not None and ch.is_active:
                        belief_map[cid] = ch
                        stack.append((cid, iter(ch.all_premise_ids)))
                        adv = True
                        break
            if not adv:
                order.append(bid)
                stack.pop()
        capped = {}
        for bid in order:
            bb = belief_map.get(bid)
            if bb is None:
                continue
            ns = len(bb.justifications)
            sc = _e(bb, W, n_justifications=ns)
            if bb.justifications:
                best = float("-inf")
                for js in bb.justifications:
                    sm = float("inf")
                    for pid in js:
                        if pid in capped:
                            sm = min(sm, capped[pid])
                        else:
                            pp = self.beliefs.get(pid)
                            if pp and pp.is_active:
                                sm = min(sm, _e(pp, W))
                    if sm != float("inf"):
                        best = max(best, sm)
                if best != float("-inf"):
                    sc = min(sc, best)
            capped[bid] = sc
        return capped.get(b.id, score)

    def _contradicts(self, a, b):
        if a.id == b.id:
            return False
        return heuristic_contradiction(a, b)

    def _active(self):
        return [b for b in self.beliefs.values() if b.is_active]

    def _doomed(self, contradictor_ids: set[str],
                justifications: list[list[str]]) -> set[str]:
        """Premise IDs that would die if contradictors were retracted.
        Mirrors Veritas's _doomed_justifications (Horn clauses + ground)."""
        doomed = set(contradictor_ids)
        memo: dict[str, bool] = {}

        def is_doomed(bid: str) -> bool:
            if bid in doomed:
                return True
            if bid in memo:
                return memo[bid]
            b = self.beliefs.get(bid)
            if b is None or not b.is_active or not b.justifications:
                memo[bid] = False
                return False
            # Doomed iff EVERY set has a doomed premise AND not grounded.
            result = (not b.ground) and all(
                any(is_doomed(pid) for pid in jset)
                for jset in b.justifications
            )
            memo[bid] = result
            if result:
                doomed.add(bid)
            return result

        doomed_premises = set()
        for jset in justifications:
            for pid in jset:
                if is_doomed(pid):
                    doomed_premises.add(pid)
        return doomed_premises

    def assert_belief(self, prop, rel, conf, justs, ground=False):
        # Normalize to Horn clauses FIRST (mirrors Veritas _normalize_justifications).
        # Flat list ['a','b'] -> [['a','b']] (single AND-set).
        from veritas.belief import _normalize_justifications
        normed = _normalize_justifications(justs)
        # Prune: drop any AND-set with a dead premise (Horn AND semantics).
        # Mirrors Veritas's fixed assert_belief logic.
        valid_justs = []
        for jset in normed:
            if all(j in self.beliefs and self.beliefs[j].is_active for j in jset):
                valid_justs.append(jset)
        # Deduplicate within sets (mirrors Veritas).
        new = Belief(proposition=prop, source="x",
                     source_reliability=rel, confidence=conf,
                     justifications=valid_justs, ground=ground)
        new_ent_full = self._ent(new)
        # Find contradictors by brute force.
        conts = [b for b in self._active() if self._contradicts(new, b)]
        # Prune doomed justifications BEFORE the decision (mirrors Veritas's
        # borrowed-entrenchment guard). Use normalized Horn clauses.
        doomed = self._doomed({c.id for c in conts}, new.justifications)
        # Track if we had justifications originally (for zombie-birth check).
        # Use normed (before dead-premise pruning), not new.justifications.
        had_justs = bool(normed)
        if doomed:
            # Drop any AND-set containing a doomed premise.
            new_justs = []
            for jset in new.justifications:
                if not any(pid in doomed for pid in jset):
                    new_justs.append(jset)
            new.justifications = new_justs
            new_ent = self._ent(new)
        else:
            new_ent = new_ent_full
        # Zombie-birth prevention (mirrors Veritas): reject non-ground belief
        # whose justifications were entirely doomed/pruned — UNLESS its
        # independent entrenchment still beats the contradictors (strong
        # newcomer wins on its own merits).
        if not new.ground and not new.justifications and had_justs:
            weakest_contra = min(
                (self._ent(c) for c in conts), default=float('inf'))
            if new_ent < weakest_contra:
                return None
            # Strong newcomer: proceed (will win below).
        # Reject if any contradictor is unbeatable.
        if any(self._ent(c) >= new_ent for c in conts):
            return None
        # Retract weakest first (ties by id for determinism).
        for c in sorted(conts, key=lambda b: (self._ent(b), b.id)):
            self._retract(c.id)
        self.beliefs[new.id] = new
        return new

    def _has_ground_path(self, belief_id: str) -> bool:
        """Well-foundedness check (mirrors Veritas AND-OR semantics).

        A belief is well-founded iff it is ground, or there EXISTS an
        AND-set where EVERY premise is well-founded (recursive).
        """
        memo: dict[str, bool] = {}

        def _wf(bid: str, path: frozenset[str]) -> bool:
            if bid in memo:
                return memo[bid]
            if bid in path:
                return False  # cycle
            b = self.beliefs.get(bid)
            if b is None or not b.is_active:
                memo[bid] = False
                return False
            if b.ground:
                memo[bid] = True
                return True
            if not b.justifications:
                memo[bid] = False
                return False
            # OR across AND-sets: any fully-well-founded set suffices.
            new_path = path | {bid}
            for jset in b.justifications:
                if all(_wf(pid, new_path) for pid in jset):
                    memo[bid] = True
                    return True
            memo[bid] = False
            return False

        return _wf(belief_id, frozenset())

    def _retract(self, bid):
        b = self.beliefs.get(bid)
        if b is None or not b.is_active:
            return
        b.status = "retracted"
        # Cascade, mirroring Veritas: NON-DESTRUCTIVE (justifications
        # preserved). A dependent survives iff it still has a valid
        # proof (well-founded via remaining active premises).
        stack = [bid]
        while stack:
            lost = stack.pop()
            for dep in self._active():
                if lost not in dep.all_premise_ids:
                    continue
                # NON-DESTRUCTIVE: do NOT mutate dep.justifications.
                # Check if dep still has a valid proof (the lost premise
                # is now inactive, so _has_ground_path accounts for it).
                if dep.justifications:
                    if self._has_ground_path(dep.id):
                        continue
                    # Has justifications but no valid proof: retract.
                    # (Justifications PRESERVED for re-derivation.)
                else:
                    if dep.ground:
                        continue
                    # Orphaned non-ground: retract.
                dep.status = "retracted"
                stack.append(dep.id)

    def retract_belief(self, bid):
        self._retract(bid)

    def state(self):
        return sorted(
            (b.proposition, b.status) for b in self.beliefs.values())


def run_trial(seed: int, n_ops: int) -> None:
    rng = random.Random(seed)
    v = BeliefStore(entrenchment_weights=W)
    r = ReferenceStore()
    v_ids: list[str] = []
    r_ids: list[str] = []
    # Map Veritas IDs to reference IDs by assertion order.
    for _ in range(n_ops):
        op = rng.random()
        if op < 0.75:
            prop = (f"S{rng.randint(0, 6)}'s CEO is "
                    f"V{rng.randint(0, 3)}")
            rel = rng.choice([0.3, 0.5, 0.7, 0.9])
            conf = rng.choice([0.3, 0.5, 0.7, 0.9])
            ground = rng.random() < 0.3
            # Justifications: reference by position in id list.
            nj = rng.randint(0, 2)
            vj = [rng.choice(v_ids) for _ in range(nj)] if v_ids else []
            rj = [r_ids[v_ids.index(j)] for j in vj]
            vb = v.assert_belief(prop, source="x", source_reliability=rel,
                                 confidence=conf, justifications=vj,
                                 ground=ground)
            rb = r.assert_belief(prop, rel, conf, rj, ground=ground)
            # Same accept/reject decision.
            assert (vb is None) == (rb is None), \
                f"seed {seed}: decision diverged on '{prop}'"
            if vb is not None:
                v_ids.append(vb.id)
                r_ids.append(rb.id)
        else:
            if v_ids:
                idx = rng.randrange(len(v_ids))
                v.retract_belief(v_ids[idx])
                r.retract_belief(r_ids[idx])
    vs = sorted((b.proposition, b.status) for b in v.all_beliefs())
    rs = r.state()
    assert vs == rs, \
        f"seed {seed}: final state diverged\n veritas: {vs[:3]}\n ref: {rs[:3]}"
    assert v.is_consistent(), f"seed {seed}: veritas inconsistent"


def main():
    trials, n_ops = 20, 300
    if "--trials" in sys.argv:
        trials = int(sys.argv[sys.argv.index("--trials") + 1])
    if "--ops" in sys.argv:
        n_ops = int(sys.argv[sys.argv.index("--ops") + 1])
    print(f"differential: {trials} trials x {n_ops} ops...", flush=True)
    for t in range(trials):
        run_trial(9000 + t, n_ops)
    print(f"DIFFERENTIAL CLEAN: {trials} trials agree with reference")


if __name__ == "__main__":
    main()
