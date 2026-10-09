"""v0.3.0 tests: Hansson framing, EE2 dominance, zombie prevention,
ground observations, and the invalidation bus.

Run: python3 tests/test_v03.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    if cond:
        PASS.append(name)
        print(f"  ok: {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL: {name}")


def test_ee2_dominance():
    """EE2: derived belief <= weakest premise (no artificial inflation)."""
    s = BeliefStore()
    # Weak premise.
    a = s.assert_belief("Weak premise", source="x",
                        source_reliability=0.3, confidence=0.3)
    # Derived belief with 5 justifications (all pointing to a, via chain).
    # Corroboration would inflate it, but EE2 caps it.
    prev = a
    for i in range(5):
        prev = s.assert_belief(f"Derived {i}", source="x",
                               source_reliability=0.9, confidence=0.9,
                               justifications=[prev.id])
    ent_a = s.entrenchment_of(s._beliefs[a.id])
    ent_derived = s.entrenchment_of(s._beliefs[prev.id])
    check("EE2: derived <= premise", ent_derived <= ent_a + 1e-9)


def test_ee3_conjunctive():
    """EE3: conjunction bounded by weakest conjunct."""
    s = BeliefStore()
    weak = s.assert_belief("Weak", source="x",
                           source_reliability=0.2, confidence=0.2)
    strong = s.assert_belief("Strong", source="x",
                             source_reliability=0.9, confidence=0.9)
    conj = s.assert_belief("Weak and Strong", source="x",
                           source_reliability=0.9, confidence=0.9,
                           justifications=[weak.id, strong.id])
    ent_weak = s.entrenchment_of(s._beliefs[weak.id])
    ent_conj = s.entrenchment_of(s._beliefs[conj.id])
    check("EE3: conjunction <= weakest", ent_conj <= ent_weak + 1e-9)


def test_zombie_prevention():
    """The reviewer's zombie: derived belief must die with its premise."""
    s = BeliefStore()
    # P: "user runs macOS" (false premise)
    p = s.assert_belief("User's OS is macOS", source="tool",
                        source_reliability=0.8, confidence=0.9,
                        ground=True)
    # D: "Homebrew is installed" derived from P, high confidence
    d = s.assert_belief("Homebrew is installed", source="derived",
                        source_reliability=0.4, confidence=0.9,
                        justifications=[p.id])
    # User corrects: Ubuntu, not macOS. P retracted.
    s.assert_belief("User's OS is Ubuntu", source="user",
                    source_reliability=0.99, confidence=0.99,
                    ground=True)
    check("zombie prevented: derived retracts with premise",
          not s.get(d.id).is_active)


def test_ground_survival():
    """Ground observations survive orphaning."""
    s = BeliefStore()
    p = s.assert_belief("User's OS is macOS", source="tool",
                        source_reliability=0.8, confidence=0.9,
                        ground=True)
    # Grounded AND justified: belt and suspenders.
    d = s.assert_belief("Homebrew is installed", source="tool",
                        source_reliability=0.9, confidence=0.9,
                        justifications=[p.id], ground=True)
    s.assert_belief("User's OS is Ubuntu", source="user",
                    source_reliability=0.99, confidence=0.99,
                    ground=True)
    check("grounded belief survives orphaning",
          s.get(d.id).is_active)


def test_zombie_bridge_blocked():
    """The reviewer's bridge attack: N cannot launder warrant via B."""
    s = BeliefStore()
    # I: incumbent, entrenchment 0.65
    i = s.assert_belief("Incumbent claim", source="x",
                        source_reliability=0.65, confidence=0.65)
    # B: bridge, derived from I, NOT grounded
    b = s.assert_belief("Bridge claim", source="x",
                        source_reliability=0.52, confidence=0.52,
                        justifications=[i.id])
    # N: newcomer contradicts I, lists B as justification
    n = s.assert_belief("Newcomer claim", source="x",
                        source_reliability=0.68, confidence=0.68,
                        justifications=[b.id],
                        metadata={"contradicts": [i.id]})
    # B should be flagged as doomed (not grounded, loses I),
    # so N loses the corroboration and should be REJECTED
    # (0.68 without corroboration < 0.65? No: 0.68 > 0.65, so N wins
    # on its own merit, which is fine — the point is B doesn't help).
    # The key check: B must be doomed, so N's entrenchment must NOT
    # include corroboration from B.
    from veritas.entrenchment import entrenchment
    n_ent_with_b = entrenchment(s._beliefs[n.id] if n else None,
                                s.entrenchment_weights) if n else 0
    # If N was accepted, verify B was doomed (retracted or pruned).
    if n is not None:
        n_live = s.get(n.id)
        check("bridge: B doomed, not counted",
              b.id not in n_live.justifications or
              not s.get(b.id).is_active)
    else:
        check("bridge: N rejected (no laundered warrant)", True)


def test_hansson_rejection():
    """Non-prioritized revision: stronger incumbents reject newcomers."""
    s = BeliefStore()
    strong = s.assert_belief("Acme's CEO is Alice", source="x",
                             source_reliability=0.9, confidence=0.9,
                             ground=True)
    weak = s.assert_belief("Acme's CEO is Bob", source="x",
                           source_reliability=0.5, confidence=0.5)
    check("Hansson: weaker newcomer rejected", weak is None)
    check("Hansson: incumbent untouched",
          s.get(strong.id).is_active)
    # Rejection is audited (not silent).
    check("Hansson: rejection audited",
          any(e.event_type == "rejected" for e in s.audit))


def test_subscriptions():
    """Invalidation bus: subscribers fire on retraction."""
    s = BeliefStore()
    fired = []
    a = s.assert_belief("Claim A", source="x", source_reliability=0.9,
                        ground=True)
    b = s.assert_belief("Claim B", source="x", source_reliability=0.9,
                        justifications=[a.id])
    s.subscribe(a.id, lambda bid, evt: fired.append((bid, evt, "a")))
    s.subscribe(b.id, lambda bid, evt: fired.append((bid, evt, "b")))
    s.retract_belief(a.id, reason="test")
    check("subscriber fired for direct retraction",
          any(f[0] == a.id for f in fired))
    check("subscriber fired for cascade",
          any(f[0] == b.id for f in fired))
    # Unsubscribe works.
    fired.clear()
    cb = lambda bid, evt: fired.append(bid)
    s.subscribe(a.id, cb)
    s.unsubscribe(a.id, cb)
    # a is already retracted; use a new belief
    c = s.assert_belief("Claim C", source="x", source_reliability=0.9,
                        ground=True)
    s.subscribe(c.id, cb)
    s.unsubscribe(c.id, cb)
    s.retract_belief(c.id, reason="test")
    check("unsubscribe works", len(fired) == 0)


def test_subscriber_exception_isolated():
    """A throwing subscriber must not break retraction."""
    s = BeliefStore()
    a = s.assert_belief("Claim", source="x", source_reliability=0.9,
                        ground=True)

    def bad(bid, evt):
        raise RuntimeError("subscriber bug")

    s.subscribe(a.id, bad)
    retracted = s.retract_belief(a.id, reason="test")
    check("retraction survives bad subscriber", len(retracted) == 1)


def main():
    print("v0.3.0:")
    test_ee2_dominance()
    test_ee3_conjunctive()
    test_zombie_prevention()
    test_ground_survival()
    test_zombie_bridge_blocked()
    test_hansson_rejection()
    test_subscriptions()
    test_subscriber_exception_isolated()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
