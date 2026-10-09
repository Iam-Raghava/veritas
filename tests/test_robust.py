"""Robustness: API misuse, detector contracts, and CLI fuzz.

A big-company reviewer will call the API wrong on purpose. The store
must never corrupt, hang, or crash — it raises TypeError/ValueError
on bad input, or handles it gracefully.

Run: python3 tests/test_robust.py
"""
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import Belief, BeliefStore  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok: {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, e))
        print(f"  FAIL: {name}: {type(e).__name__}: {e}")


def test_none_proposition_rejected():
    s = BeliefStore()
    for bad in (None, 123, b"bytes", ["list"], "", "   "):
        try:
            s.assert_belief(bad)  # type: ignore[arg-type]
            raise AssertionError(f"accepted {type(bad).__name__}")
        except (TypeError, AttributeError, ValueError):
            pass
    assert s.is_consistent() and len(s.all_beliefs()) == 0


def test_extreme_confidence_values():
    s = BeliefStore()
    for v in (-1.0, 2.0, float("inf"), float("-inf")):
        try:
            b = s.assert_belief(f"Extreme {v}", confidence=v,
                                source_reliability=0.5)
            e = s.entrenchment_of(b)
            assert 0.0 <= e <= 1.0, f"entrenchment {e} out of [0,1] for {v}"
        except (ValueError, OverflowError):
            pass  # rejecting is also fine
    assert s.is_consistent()


def test_nan_confidence():
    import math
    s = BeliefStore()
    try:
        b = s.assert_belief("NaN belief", confidence=float("nan"),
                            source_reliability=0.5)
        e = s.entrenchment_of(b)
        # NaN must not poison ordering: entrenchment must be a real number.
        assert not math.isnan(e), "NaN entrenchment poisons comparisons"
    except ValueError:
        pass
    assert s.is_consistent()


def test_detector_nonboolean():
    """Detector returning truthy/falsy non-bools must work."""
    s = BeliefStore(
        contradiction_fn=lambda a, b: 1 if "X" in a.proposition and
        "X" in b.proposition and a.id != b.id else 0)
    s.assert_belief("X is up", source="x", source_reliability=0.9)
    r = s.assert_belief("X is down", source="x", source_reliability=0.1)
    assert r is None  # 0.1 < 0.9: rejected
    assert s.is_consistent()


def test_detector_asymmetric():
    """Asymmetric detector: store must still end consistent."""
    s = BeliefStore(
        contradiction_fn=lambda a, b: a.proposition.startswith("~")
        and b.proposition == a.proposition[1:])
    s.assert_belief("sky is blue", source="x", source_reliability=0.9)
    s.assert_belief("~sky is blue", source="x", source_reliability=0.95)
    assert s.is_consistent()


def test_detector_none_return():
    s = BeliefStore(contradiction_fn=lambda a, b: None)
    for i in range(10):
        s.assert_belief(f"Claim {i} here", source="x", source_reliability=0.5)
    assert len(s.active_beliefs()) == 10
    assert s.is_consistent()


def test_slow_detector():
    """A slow detector must not break correctness (just slower)."""
    import time
    calls = []

    def slow(a, b):
        calls.append(1)
        time.sleep(0.001)
        return False

    s = BeliefStore(contradiction_fn=slow)
    for i in range(5):
        s.assert_belief(f"Slow {i}", source="x", source_reliability=0.5)
    assert len(s.active_beliefs()) == 5


def test_belief_direct_construction():
    b = Belief("test", confidence=0.5)
    assert b.is_active and b.id
    # Mutating status directly is allowed; store reads it live.
    s = BeliefStore()
    stored = s.assert_belief("Direct", source="x", source_reliability=0.9)
    assert stored.id in [x.id for x in s.active_beliefs()]


def test_cli_fuzz():
    """Random CLI args: never crash with traceback, exit codes sane."""
    rng = random.Random(99)
    cmds = ["init", "assert", "retract", "list", "contradictions", "audit",
            "stats", "bogus-cmd", ""]
    bad_args = [["--db"], ["--reliability", "abc"], ["--reliability", "99"],
                ["assert"], ["retract"], ["--db", "/nonexistent/x.db", "list"]]
    crashes = 0
    for _ in range(40):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
            db = f.name
        cmd = [sys.executable, "-m", "veritas.cli", "--db", db,
               rng.choice(cmds)]
        if rng.random() < 0.5:
            cmd += rng.choice(bad_args)
        if rng.random() < 0.3:
            cmd += ["weird proposition !@#$%^&*()"]
        env = dict(os.environ, PYTHONPATH=os.path.expanduser("~/workspace/veritas"))
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=15,
                           env=env)
        out = (p.stdout + p.stderr).lower()
        if "traceback (most recent call last)" in out:
            crashes += 1
            print(f"    CLI crash: {' '.join(cmd[4:])} -> "
                  f"{out.splitlines()[-1][:100]}")
    assert crashes == 0, f"{crashes} CLI crashes"


def test_empty_store_queries():
    s = BeliefStore()
    assert s.is_consistent()
    assert s.find_contradictions() == []
    assert s.active_beliefs() == []
    assert s.get("nope") is None
    assert s.dependents_of("nope") == []
    assert s.stats()["active"] == 0


def main():
    print("robustness:")
    check("None/bad proposition rejected", test_none_proposition_rejected)
    check("extreme confidence values", test_extreme_confidence_values)
    check("NaN confidence", test_nan_confidence)
    check("non-boolean detector", test_detector_nonboolean)
    check("asymmetric detector", test_detector_asymmetric)
    check("None-returning detector", test_detector_none_return)
    check("slow detector", test_slow_detector)
    check("direct Belief construction", test_belief_direct_construction)
    check("CLI fuzz (40 cases)", test_cli_fuzz)
    check("empty store queries", test_empty_store_queries)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
