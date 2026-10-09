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
from veritas.entrenchment import (  # noqa: E402
    entrenchment, independent_entrenchment)
from veritas.belief import Belief  # noqa: E402
from veritas.store import DEFAULT_SURVIVAL_THRESHOLD  # noqa: E402

# No recency: wall-clock time must not affect the comparison.
W = {"source": 0.4, "corroboration": 0.2, "confidence": 0.4, "recency": 0.0}


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
        # Prune justifications to active beliefs (mirrors Veritas).
        justs = [j for j in justs
                 if j in self.beliefs and self.beliefs[j].is_active]
        # Deduplicate (mirrors Veritas).
        justs = list(dict.fromkeys(justs))
        new = Belief(proposition=prop, source="x",
                     source_reliability=rel, confidence=conf,
                     justifications=list(justs), ground=ground)
        new_ent_full = self._ent(new)
        # Find contradictors by brute force.
        conts = [b for b in self._active() if self._contradicts(new, b)]
        # Prune doomed justifications BEFORE the decision (mirrors Veritas's
        # borrowed-entrenchment guard). Use normalized Horn clauses.
        doomed = self._doomed({c.id for c in conts}, new.justifications)
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
        # Reject if any contradictor is unbeatable.
        if any(self._ent(c) >= new_ent for c in conts):
            return None
        # Retract weakest first (ties by id for determinism).
        for c in sorted(conts, key=lambda b: (self._ent(b), b.id)):
            self._retract(c.id)
        self.beliefs[new.id] = new
        return new

    def _has_ground_path(self, belief_id: str) -> bool:
        """Well-foundedness check (mirrors Veritas)."""
        seen = set()
        stack = [belief_id]
        while stack:
            bid = stack.pop()
            if bid in seen:
                continue
            seen.add(bid)
            b = self.beliefs.get(bid)
            if b is None or not b.is_active:
                continue
            if b.ground:
                return True
            stack.extend(b.all_premise_ids)
        return False

    def _retract(self, bid):
        b = self.beliefs.get(bid)
        if b is None or not b.is_active:
            return
        b.status = "retracted"
        # Cascade, mirroring Veritas v0.4: Horn clauses + well-foundedness.
        stack = [bid]
        while stack:
            lost = stack.pop()
            for dep in self._active():
                if lost not in dep.all_premise_ids:
                    continue
                # Drop any AND-set containing the lost premise.
                new_justs = []
                for jset in dep.justifications:
                    if lost not in jset:
                        new_justs.append(jset)
                dep.justifications = new_justs
                # Survive iff: (has sets AND ground path) OR
                # (orphaned AND ground).
                if dep.justifications:
                    if self._has_ground_path(dep.id):
                        continue
                    # Has sets but no ground path: retract.
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
