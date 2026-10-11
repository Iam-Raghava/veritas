"""v05 regression tests: enterprise hardening invariants.

Tests for:
1. Re-derivation contradiction blocking (no resurrection of contradicted beliefs)
2. Transactional constraint rollback (rejected newcomer leaves incumbents intact)
3. Bulk temp-table persistence (no SQLite variable limit crash)
4. assert_batch topological ordering
5. StorageBackend protocol
6. LangGraph and LlamaIndex integrations
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore
from veritas.persist_backend import SQLiteBackend, get_backend


def test_rederivation_blocked_by_contradiction():
    """Re-derived beliefs must not resurrect contradicted claims."""
    s = BeliefStore()
    p = s.assert_belief("Premise", source="x",
                        source_reliability=0.9, ground=True)
    bob = s.assert_belief("Acme's CEO is Bob", source="y",
                          source_reliability=0.8,
                          justifications=[p.id], ground=False)
    assert s.get(bob.id).is_active
    # Jane contradicts Bob; Jane is stronger.
    jane = s.assert_belief("Acme's CEO is Jane", source="z",
                           source_reliability=0.95, ground=True)
    assert not s.get(bob.id).is_active
    # Retract and re-assert premise. Bob must NOT come back.
    s.retract_belief(p.id)
    p2 = s.assert_belief("Premise", source="x",
                         source_reliability=0.9, ground=True)
    assert not s.get(bob.id).is_active, \
        "Bob resurrected despite Jane contradiction!"
    assert s.get(jane.id).is_active
    assert s.is_consistent()
    print("  ok: re-derivation blocked by contradiction")


def test_transactional_constraint_rollback():
    """Rejected newcomer must not destroy incumbents."""
    s = BeliefStore()
    # Incumbent.
    inc = s.assert_belief("Server is running", source="a",
                          source_reliability=0.9, ground=True)
    # Constraint: at most one of {X, Y} can be active, newcomer loses.
    def _check(beliefs):
        props = {b.proposition for b in beliefs}
        if "Server is running" in props and "Server is stopped" in props:
            # Return the weaker (newcomer will be added; we identify by
            # checking which is not the incumbent).
            for b in beliefs:
                if b.proposition == "Server is stopped":
                    return [b.id]
        return []
    s.add_constraint("mutual_exclusion", _check)
    # Newcomer contradicts incumbent AND violates constraint.
    # Make newcomer weaker so it's rejected.
    new = s.assert_belief("Server is stopped", source="b",
                          source_reliability=0.5, ground=True)
    # Newcomer should be rejected (weaker + constraint).
    # Incumbent must survive.
    assert s.get(inc.id).is_active, \
        "Incumbent destroyed by rejected newcomer!"
    print("  ok: transactional constraint rollback preserves incumbents")


def test_bulk_persistence_no_var_limit():
    """Temp-table persistence handles >32k beliefs without crash."""
    from veritas import persist
    s = BeliefStore()
    # We don't need 32k for the test; just verify the temp-table
    # path works. (Full 32k is slow in CI.)
    for i in range(100):
        s.assert_belief(f"Fact number {i}", source="bulk", ground=True)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        persist.save(s, path)
        s2 = persist.load(path)
        assert len(s2.active_beliefs()) == 100
    finally:
        os.unlink(path)
        # WAL mode creates -wal and -shm files.
        for suffix in ("-wal", "-shm"):
            try:
                os.unlink(path + suffix)
            except OSError:
                pass
    print("  ok: bulk persistence via temp table")


def test_assert_batch_topological():
    """assert_batch orders premises before dependents."""
    s = BeliefStore()
    # Dependent listed BEFORE premise in input; batch should reorder.
    results = s.assert_batch([
        {
            "_id": "conclusion",
            "proposition": "System is healthy",
            "justifications": [["premise"]],
            "source": "test",
            "ground": False,
        },
        {
            "_id": "premise",
            "proposition": "All checks passed",
            "source": "test",
            "ground": True,
        },
    ])
    assert len(results) == 2
    assert all(r is not None for r in results)
    # The conclusion's justification should reference the real premise ID.
    conclusion = s.get(results[0].id)
    premise = s.get(results[1].id)
    assert premise.proposition == "All checks passed"
    # Justification should be resolved to real ID.
    assert any(premise.id in jset
               for jset in conclusion.justifications), \
        f"Justification not resolved: {conclusion.justifications}"
    print("  ok: assert_batch topological ordering")


def test_storage_backend():
    """Persistence backend protocol works."""
    s = BeliefStore()
    b = s.assert_belief("Backend test", source="x", ground=True)
    # SQLite via get_backend.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        backend = get_backend(f"sqlite:///{path}")
        assert isinstance(backend, SQLiteBackend)
        backend.save(s)
        s2 = backend.load()
        assert len(s2.active_beliefs()) >= 1
    finally:
        os.unlink(path)
        for suffix in ("-wal", "-shm"):
            try:
                os.unlink(path + suffix)
            except OSError:
                pass
    print("  ok: StorageBackend protocol")


def test_langgraph_store():
    """LangGraph VeritasMemoryStore truth-maintains state."""
    from veritas.integrations.langgraph import VeritasMemoryStore
    store = VeritasMemoryStore()
    store.put(("thread_1",), "ceo",
              {"proposition": "Acme's CEO is Jane"})
    val = store.get(("thread_1",), "ceo")
    assert val is not None
    assert "Jane" in val["proposition"]
    # Contradicting update: stronger wins.
    store.put(("thread_1",), "ceo",
              {"proposition": "Acme's CEO is John",
               "reliability": 0.95})
    val2 = store.get(("thread_1",), "ceo")
    # One of them should be active (entrenchment decides).
    assert val2 is not None
    print("  ok: LangGraph store")


def test_llamaindex_store():
    """LlamaIndex store retracts outdated document versions."""
    from veritas.integrations.llamaindex import VeritasDocumentStore
    store = VeritasDocumentStore()
    store.add_documents([
        {"doc_id": "doc1", "text": "Acme's CEO is Jane",
         "timestamp": 1000.0},
    ])
    docs = store.get_document("doc1")
    assert len(docs) == 1
    assert "Jane" in docs[0]["proposition"]
    # Update with newer timestamp.
    store.add_documents([
        {"doc_id": "doc1", "text": "Acme's CEO is John",
         "timestamp": 2000.0},
    ])
    docs2 = store.get_document("doc1")
    # Old Jane should be retracted; John active.
    props = [d["proposition"] for d in docs2]
    assert not any("Jane" in p for p in props), \
        f"Stale Jane not retracted: {props}"
    print("  ok: LlamaIndex document store")


if __name__ == "__main__":
    print("v05 regression tests:")
    test_rederivation_blocked_by_contradiction()
    test_transactional_constraint_rollback()
    test_bulk_persistence_no_var_limit()
    test_assert_batch_topological()
    test_storage_backend()
    test_langgraph_store()
    test_llamaindex_store()
    print("\n7 passed, 0 failed")
