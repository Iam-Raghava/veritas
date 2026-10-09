"""Adversarial persistence + structural edges for Veritas.

  - SQLite: corrupt files, wrong files, future schema versions.
  - Wide fan-out: one belief with 10,000 dependents cascades cleanly.
  - Self-contradicting detector: must not suicide the store.
  - Weight/clock edges: negative/zero weights, clock skew.

Run: python3 tests/test_adversarial.py
"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore, load, save  # noqa: E402
from veritas.persist import SCHEMA_VERSION  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok: {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, e))
        print(f"  FAIL: {name}: {type(e).__name__}: {e}")


def tmp_db() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


def test_corrupt_db():
    path = tmp_db()
    try:
        with open(path, "wb") as f:
            f.write(os.urandom(500))
        try:
            load(path)
            raise AssertionError("corrupt DB loaded without error")
        except Exception as e:  # noqa: BLE001
            assert not isinstance(e, AssertionError)
            assert "Traceback" not in str(e)
    finally:
        os.unlink(path)


def test_valid_sqlite_not_veritas():
    path = tmp_db()
    try:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE foo (x INT)")
        conn.commit()
        conn.close()
        try:
            load(path)
            raise AssertionError("non-Veritas DB loaded")
        except ValueError as e:
            assert "not a Veritas store" in str(e)
    finally:
        os.unlink(path)


def test_future_schema_rejected():
    import json
    s = BeliefStore()
    s.assert_belief("Hello", source="x", source_reliability=0.5)
    path = tmp_db()
    try:
        save(s, path)
        conn = sqlite3.connect(path)
        row = conn.execute(
            "SELECT value FROM meta WHERE key='config'").fetchone()
        cfg = json.loads(row[0])
        cfg["schema_version"] = SCHEMA_VERSION + 99
        conn.execute("UPDATE meta SET value=? WHERE key='config'",
                     (json.dumps(cfg),))
        conn.commit()
        conn.close()
        try:
            load(path)
            raise AssertionError("future schema loaded")
        except ValueError as e:
            assert "schema v" in str(e)
    finally:
        os.unlink(path)


def test_old_schema_no_version_loads():
    """DBs written before versioning (no schema_version key) still load."""
    import json
    s = BeliefStore()
    s.assert_belief("Legacy", source="x", source_reliability=0.5)
    path = tmp_db()
    try:
        save(s, path)
        conn = sqlite3.connect(path)
        row = conn.execute(
            "SELECT value FROM meta WHERE key='config'").fetchone()
        cfg = json.loads(row[0])
        del cfg["schema_version"]
        conn.execute("UPDATE meta SET value=? WHERE key='config'",
                     (json.dumps(cfg),))
        conn.commit()
        conn.close()
        s2 = load(path)  # must not raise
        assert s2.is_consistent()
    finally:
        os.unlink(path)


def test_wide_fanout_cascade():
    """One belief with 10,000 dependents: iterative cascade, all fall."""
    s = BeliefStore()
    root = s.assert_belief("Root belief", source="x", source_reliability=0.9)
    for i in range(10000):
        s.assert_belief(f"Dependent {i}", source="x", source_reliability=0.1,
                        justifications=[root.id])
    assert len(s.active_beliefs()) == 10001
    retracted = s.retract_belief(root.id, reason="test")
    assert len(retracted) == 10001, f"got {len(retracted)}"
    assert len(s.active_beliefs()) == 0
    assert s.is_consistent()


def test_self_contradicting_detector():
    """A detector claiming self-contradiction must not nuke the store."""
    s = BeliefStore(contradiction_fn=lambda a, b: True)
    b = s.assert_belief("Anything", source="x", source_reliability=0.9)
    # contradictors_of filters self via `b.id != belief.id` (exact scan).
    assert b is not None and b.is_active
    assert s.is_consistent()


def test_negative_zero_weights():
    s = BeliefStore(entrenchment_weights={
        "source": -1.0, "corroboration": 0.0, "confidence": 0.0,
        "recency": 0.0})
    b = s.assert_belief("Weird weights", source="x", source_reliability=0.9)
    e = s.entrenchment_of(b)
    assert 0.0 <= e <= 1.0, f"entrenchment {e} escaped [0,1]"
    assert s.is_consistent()


def test_clock_skew():
    """System clock jumps backwards: recency must not explode."""
    from unittest import mock
    from veritas import Belief
    with mock.patch("time.time", return_value=1_000.0):
        b = Belief("Old belief", source="x", source_reliability=0.5)
    s = BeliefStore()
    # Pretend now is BEFORE the belief was created (clock skew).
    with mock.patch("time.time", return_value=100.0):
        e = s.entrenchment_of(b)
    assert 0.0 <= e <= 1.0, f"clock skew broke entrenchment: {e}"


def test_duplicate_proposition():
    """Same text twice: both live (corroboration), no contradiction."""
    s = BeliefStore()
    a = s.assert_belief("Acme's CEO is Alice", source="x", source_reliability=0.5)
    b = s.assert_belief("Acme's CEO is Alice", source="y", source_reliability=0.5)
    assert a is not None and b is not None
    assert a.id != b.id
    assert s.is_consistent()
    assert len(s.active_beliefs()) == 2


def test_concurrent_cli_same_db():
    """Two CLI processes on one DB: SQLite locking, no corruption."""
    import subprocess
    path = tmp_db()
    os.unlink(path)  # init refuses to overwrite an existing file
    try:
        env = dict(os.environ,
                   PYTHONPATH=os.path.expanduser("~/workspace/veritas"))
        subprocess.run(
            [sys.executable, "-m", "veritas.cli", "--db", path, "init"],
            check=True, capture_output=True, env=env, timeout=30)
        procs = [
            subprocess.Popen(
                [sys.executable, "-m", "veritas.cli", "--db", path,
                 "assert", f"CLI claim {i} is value",
                 "--source", "x", "--reliability", "0.5"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            for i in range(6)
        ]
        for p in procs:
            p.wait(timeout=30)
        # DB must still load and be consistent (some asserts may have
        # failed on locking; none may corrupt).
        s = load(path)
        assert s.is_consistent()
        for p in procs:
            err = (p.stderr.read() or b"").decode().lower()
            assert "traceback" not in err, f"CLI traceback: {err[:150]}"
    finally:
        os.unlink(path)


def test_unserializable_metadata():
    """save() must not crash on metadata json can't encode."""
    s = BeliefStore()
    s.assert_belief("Weird meta", source="x", source_reliability=0.5,
                    metadata={"obj": object(), "set": {1, 2}})
    path = tmp_db()
    try:
        save(s, path)  # must not raise
        s2 = load(path)
        assert s2.is_consistent()
    finally:
        os.unlink(path)


def test_reentrant_detector():
    """A detector that calls back into the store must not deadlock."""

    def nosy(a, b):
        # Sneaky: queries the store from inside the detector.
        _ = store.active_beliefs()
        return False

    store = BeliefStore(contradiction_fn=nosy)
    import threading

    done = threading.Event()

    def worker():
        for i in range(20):
            store.assert_belief(f"Reentrant {i}", source="x",
                                source_reliability=0.5)
        done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert done.wait(timeout=30), "deadlock: reentrant detector hung"
    t.join(timeout=5)
    assert store.is_consistent()


def test_snapshot_isolation():
    """Mutating returned beliefs must not corrupt the store."""
    s = BeliefStore()
    a = s.assert_belief("Acme's CEO is Alice", source="x", source_reliability=0.9)
    b = s.assert_belief("Alice founded Acme", source="x", source_reliability=0.5,
                        justifications=[a.id])
    # Attack every public query method.
    s.get(a.id).proposition = "HACKED"
    s.get(a.id).justifications.append("evil")
    s.get(a.id).status = "retracted"
    for bl in s.active_beliefs():
        bl.proposition = "HACKED"
    for bl in s.all_beliefs():
        bl.justifications.clear()
    for bl in s.contradictors_of(a):
        bl.proposition = "HACKED"
    for x, y in s.find_contradictions():
        x.proposition = y.proposition = "HACKED"
    for bl in s.dependents_of(a.id):
        bl.justifications.clear()
    r = s.assert_belief("Acme's CEO is Bob", source="y", source_reliability=0.99)
    assert r is not None
    r.proposition = "HACKED"
    for bl in s.retract_belief(a.id, reason="test"):
        bl.proposition = "HACKED"
    # Store must be pristine.
    assert s.get(a.id).proposition == "Acme's CEO is Alice"
    assert s.is_consistent()
    assert len(s.audit) > 0  # audit still recorded everything


def test_deepcopy_store():
    """Stores must survive deepcopy (lock excluded, rebuilt)."""
    import copy
    s = BeliefStore()
    a = s.assert_belief("Acme's CEO is Alice", source="x", source_reliability=0.9)
    s2 = copy.deepcopy(s)
    assert len(s2.all_beliefs()) == 1
    assert s2.is_consistent()
    # Independent: mutating the copy doesn't touch the original.
    s2.assert_belief("Acme's CEO is Bob", source="y", source_reliability=0.1)
    assert len(s.all_beliefs()) == 1
    assert len(s2.all_beliefs()) == 1
    # Copy is fully functional (lock works).
    assert s2.get(a.id).is_active


def main():
    print("adversarial:")
    check("corrupt DB rejected cleanly", test_corrupt_db)
    check("valid SQLite, not Veritas", test_valid_sqlite_not_veritas)
    check("future schema rejected", test_future_schema_rejected)
    check("legacy schema (no version) loads", test_old_schema_no_version_loads)
    check("wide fan-out: 10k dependents cascade", test_wide_fanout_cascade)
    check("self-contradicting detector", test_self_contradicting_detector)
    check("negative weights clamped", test_negative_zero_weights)
    check("clock skew", test_clock_skew)
    check("duplicate proposition", test_duplicate_proposition)
    check("concurrent CLI on one DB", test_concurrent_cli_same_db)
    check("unserializable metadata degrades", test_unserializable_metadata)
    check("reentrant detector: no deadlock", test_reentrant_detector)
    check("snapshot isolation: mutation can't corrupt", test_snapshot_isolation)
    check("deepcopy store", test_deepcopy_store)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
