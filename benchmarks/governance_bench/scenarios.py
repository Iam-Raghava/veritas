"""Adversarial scenarios. Each returns {system_name: score}.

Every scenario is seeded (random.Random) for determinism. Scores are
percentages unless noted. A baseline that lacks a capability reports
None (rendered as "n/a") — we do not invent scores for missing features.
"""

from __future__ import annotations

import random

from .baselines import AppendOnly, LatestWins, VeritasSys

SYSTEMS = (LatestWins, AppendOnly, VeritasSys)


def _systems():
    return [cls() for cls in SYSTEMS]


# ---------------------------------------------------------------------------
# 1. Contradiction storm
# ---------------------------------------------------------------------------
def contradiction_storm(seed, n_facts=50):
    """50 facts; each gets 4 assertions in RANDOM order: one from a
    high-reliability source (0.9) and three from weaker ones (0.3/0.6/0.3).

    Models the core agent-memory attack: established facts bombarded by
    low-quality sources. Ground truth = the 0.9 source's value.

    Veritas should keep the 0.9 value regardless of arrival order
    (entrenchment beats recency). Latest-wins keeps whoever wrote last.
    """
    rng = random.Random(seed)
    # Generate the workload ONCE, then replay identically for each system.
    # (Using one shared rng across systems gives each system different
    # random values — an unfair comparison.)
    workload = []
    for i in range(n_facts):
        key = f"fact{i:02d}"
        vals = [f"Person{rng.randint(0, 99999):05d}" for _ in range(4)]
        truth_val = vals[0]
        writes = [(vals[0], 0.9), (vals[1], 0.3),
                  (vals[2], 0.6), (vals[3], 0.3)]
        rng.shuffle(writes)
        workload.append((key, truth_val, writes))
    results = {}
    for sys in _systems():
        correct = 0
        ts = [0.0]
        for key, truth_val, writes in workload:
            i = int(key[4:])
            for v, rel in writes:
                ts[0] += 1.0
                prop = f"Company{i:02d}'s CEO is {v}"
                sys.assert_fact(key, prop, reliability=rel,
                                source=f"src_{rel}", timestamp=ts[0])
            truth_prop = f"Company{i:02d}'s CEO is {truth_val}"
            if sys.query(key) == truth_prop:
                correct += 1
        results[sys.name] = 100.0 * correct / n_facts
    return results


# ---------------------------------------------------------------------------
# 2. Cascade integrity
# ---------------------------------------------------------------------------
def cascade_integrity(seed, n_chains=20, depth=50):
    """20 derivation chains of depth 50. Retract all roots.

    Metric: orphan rate = derived beliefs still present after their root
    (and therefore their whole justification chain) was retracted.

    Veritas cascades via TMS. The baselines record a parent pointer but
    have no propagation mechanism — retraction is a no-op for dependents.
    We measure that honestly instead of pretending they cascade.
    """
    results = {}
    for sys in _systems():
        ts = [0.0]
        roots = []
        for c in range(n_chains):
            ts[0] += 1.0
            rkey = f"chain{c:02d}_d00"
            sys.assert_fact(rkey, f"Chain{c:02d} step 00 holds",
                            reliability=0.9, source="root", timestamp=ts[0])
            roots.append(rkey)
            parent = rkey
            for d in range(1, depth):
                ts[0] += 1.0
                k = f"chain{c:02d}_d{d:02d}"
                sys.assert_fact(k, f"Chain{c:02d} step {d:02d} holds",
                                reliability=0.5, source="derived",
                                timestamp=ts[0], ground=False, parent=parent)
                parent = k
        for r in roots:
            sys.retract_key(r)
        total_derived = n_chains * (depth - 1)
        orphans = sys.orphan_count()
        results[sys.name] = 100.0 * orphans / total_derived
    return results  # LOWER is better (0 = clean cascade)


# ---------------------------------------------------------------------------
# 3. Temporal accuracy
# ---------------------------------------------------------------------------
def temporal_accuracy(seed, n_facts=30):
    """30 facts, each with two validity windows:
    owner=A on [0,100), owner=B on [100,200). Query 10 timestamps.

    Veritas: TimeTravel replay + validity-window filter.
    Baselines: no temporal concept -> None (n/a).
    """
    rng = random.Random(seed + 1)
    query_ts = sorted(rng.uniform(0, 200) for _ in range(10))
    results = {}
    for sys in _systems():
        if sys.name != "veritas":
            results[sys.name] = None
            continue
        ts = [0.0]
        for i in range(n_facts):
            key = f"tsite{i:02d}"
            a = f"OwnerA{rng.randint(0, 999):03d}"
            b = f"OwnerB{rng.randint(0, 999):03d}"
            # Assertion (recording) times are all BEFORE the query range:
            # these are backfilled historical facts; the validity windows
            # [0,100)/[100,200) carry the temporal semantics, not the
            # moment the agent recorded them.
            ts[0] += 0.001
            sys.assert_fact(key, f"Site{i:02d}'s owner is {a}",
                            reliability=0.9, source="reg",
                            timestamp=ts[0], valid_from=0.0, valid_until=100.0)
            ts[0] += 0.001
            sys.assert_fact(key, f"Site{i:02d}'s owner is {b}",
                            reliability=0.9, source="reg",
                            timestamp=ts[0], valid_from=100.0, valid_until=200.0)
        correct = total = 0
        # re-derive expected owners per fact (deterministic from seed)
        rng2 = random.Random(seed + 1)
        _ = sorted(rng2.uniform(0, 200) for _ in range(10))  # replay query_ts
        for i in range(n_facts):
            a = f"OwnerA{rng2.randint(0, 999):03d}"
            b = f"OwnerB{rng2.randint(0, 999):03d}"
            for T in query_ts:
                total += 1
                got = sys.query_at(f"tsite{i:02d}", T)
                want = (f"Site{i:02d}'s owner is {a}" if T < 100
                        else f"Site{i:02d}'s owner is {b}")
                if got == want:
                    correct += 1
        results[sys.name] = 100.0 * correct / total
    return results


# ---------------------------------------------------------------------------
# 4. Deletion compliance
# ---------------------------------------------------------------------------
def deletion_compliance(seed, n=100):
    """100 beliefs from 3 sources; 20 derived from src_b beliefs.
    Then: forget all from src_b + forget one src_a proposition.

    Metrics: recall (% of targets actually gone) and precision
    (% of non-targets preserved). For Veritas, src_b's derived beliefs
    are also targets (cascade). For baselines they are measurable
    orphans — deletion has no propagation.
    """
    rng = random.Random(seed + 2)
    results = {}
    for sys in _systems():
        ts = [0.0]
        src_b_keys, derived_keys = [], []
        all_keys = []
        for i in range(n):
            ts[0] += 1.0
            src = ["src_a", "src_b", "src_c"][i % 3]
            k = f"del{i:03d}"
            sys.assert_fact(k, f"Fact {i:03d} holds", reliability=0.7,
                            source=src, timestamp=ts[0])
            all_keys.append(k)
            if src == "src_b":
                src_b_keys.append(k)
        # 20 derived beliefs, justified by src_b beliefs (source src_a)
        for j, pb in enumerate(rng.sample(src_b_keys, 20)):
            ts[0] += 1.0
            k = f"del_derived{j:02d}"
            sys.assert_fact(k, f"Derived {j:02d} holds", reliability=0.5,
                            source="src_a", timestamp=ts[0],
                            ground=False, parent=pb)
            derived_keys.append(k)
            all_keys.append(k)
        # One specific src_a proposition to forget: the first src_a key.
        # (Deterministic: keys are created round-robin src_a/src_b/src_c,
        # so del000 is always src_a.)
        target_a = "del000"
        targets = set(src_b_keys) | set(derived_keys) | {target_a}
        nontargets = set(all_keys) - targets

        sys.forget_source("src_b")
        sys.retract_key(target_a)

        def gone(k):
            if sys.name == "veritas":
                ids = sys._ids.get(k, [])
                live = {b.id for b in sys.store.active_beliefs()}
                return not any(i in live for i in ids)
            if sys.name == "append-only":
                return k in sys._deleted
            return k not in sys._facts

        def present(k):
            return not gone(k)

        recall = 100.0 * sum(1 for k in targets if gone(k)) / len(targets)
        precision = (100.0 * sum(1 for k in nontargets if present(k))
                     / len(nontargets))
        results[sys.name] = {"recall": recall, "precision": precision,
                             "orphans": sys.orphan_count()}
    return results


# ---------------------------------------------------------------------------
# 5. Zombie resistance
# ---------------------------------------------------------------------------
def zombie_resistance(seed, n_pairs=10):
    """P (rel 0.9) asserted, killed by Q (rel 0.95), then P re-asserted 5x.

    Metric: % of pairs where P stays dead. Veritas rejects the
    re-assertions (Q outranks). Naive stores resurrect P on rewrite.
    """
    results = {}
    for sys in _systems():
        ts = [0.0]
        stayed_dead = 0
        for i in range(n_pairs):
            key = f"z{i:02d}"
            p = f"Service{i:02d} is online"
            q = f"Service{i:02d} is not online"
            ts[0] += 1.0
            sys.assert_fact(key, p, reliability=0.9, source="mon",
                            timestamp=ts[0])
            ts[0] += 1.0
            sys.assert_fact(key, q, reliability=0.95, source="ops",
                            timestamp=ts[0])
            for _ in range(5):
                ts[0] += 1.0
                sys.assert_fact(key, p, reliability=0.9, source="mon",
                                timestamp=ts[0])
            if sys.query(key) != p:
                stayed_dead += 1
        results[sys.name] = 100.0 * stayed_dead / n_pairs
    return results


# ---------------------------------------------------------------------------
# 6. Audit completeness
# ---------------------------------------------------------------------------
def audit_completeness(seed):
    """Mixed workload, then two questions:
    Q1: why was belief X retracted?  Q2: what is the state now
        (are the retracted beliefs actually gone)?

    Binary per question. Veritas: explainer + time-travel. Baselines:
    no audit log -> cannot answer (honest fail, not a fake zero).

    Note: retraction audit events carry wall-clock time, so Q2 queries
    the state as of now; the check is that retracted beliefs (and their
    cascaded dependents) are absent.
    """
    import time as _time
    results = {}
    for sys in _systems():
        ts = [0.0]
        # workload: A(0.9) killed by B(0.95); C derived from B; retract B
        ts[0] += 1.0
        sys.assert_fact("ak", "Alpha holds", reliability=0.9,
                        source="s1", timestamp=ts[0])
        ts[0] += 1.0
        sys.assert_fact("ak", "Alpha fails", reliability=0.95,
                        source="s2", timestamp=ts[0])
        ts[0] += 1.0
        sys.assert_fact("ck", "Gamma holds", reliability=0.5, source="s1",
                        timestamp=ts[0], ground=False, parent="ak")
        sys.retract_key("ak")

        q1 = sys.explain_retraction("ak")
        q2_ok = False
        q2 = sys.state_at(_time.time())
        if q2 is not None:
            if sys.name == "veritas":
                props = {b.proposition for b in q2}
                # A killed by B, B retracted, C cascade-retracted:
                # none should be present.
                q2_ok = not ({"Alpha holds", "Alpha fails", "Gamma holds"}
                             & props)
            else:
                q2_ok = True  # baseline returned *something*
        results[sys.name] = 100.0 * sum([bool(q1), q2_ok]) / 2
    return results


SCENARIOS = [
    ("contradiction_storm",
     "Contradiction storm: 50 facts x 4 writes, truth = 0.9-src value",
     contradiction_storm, "higher", "% facts correct"),
    ("cascade_integrity",
     "Cascade integrity: 20 chains x depth 50, retract roots",
     cascade_integrity, "lower", "% orphans remaining"),
    ("temporal_accuracy",
     "Temporal accuracy: 30 facts x 2 windows, 10 time queries",
     temporal_accuracy, "higher", "% queries correct"),
    ("deletion_compliance",
     "Deletion compliance: forget src_b + 1 proposition (recall/precision)",
     deletion_compliance, "higher", "recall / precision %"),
    ("zombie_resistance",
     "Zombie resistance: P killed by Q, P re-asserted 5x",
     zombie_resistance, "higher", "% pairs P stayed dead"),
    ("audit_completeness",
     "Audit completeness: why-retracted? state-at-T?",
     audit_completeness, "higher", "% questions answerable"),
]
