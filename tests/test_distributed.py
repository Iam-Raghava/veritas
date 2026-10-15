"""Tests for veritas.distributed: multi-node shared belief state."""

import os
import tempfile

import pytest

from veritas import DistributedBeliefStore, SQLiteEventLog


@pytest.fixture
def log_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # SQLiteEventLog creates it
    yield path
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except OSError:
            pass


def test_nodes_converge(log_path):
    log = SQLiteEventLog(log_path)
    a = DistributedBeliefStore("a", log)
    b = DistributedBeliefStore("b", log)
    a.assert_belief("Sky is blue", source_reliability=0.9, ground=True)
    b.sync()
    assert a.state_hash() == b.state_hash()
    assert len(b.active_beliefs()) == 1


def test_cross_node_contradiction_resolution(log_path):
    log = SQLiteEventLog(log_path)
    a = DistributedBeliefStore("a", log)
    b = DistributedBeliefStore("b", log)
    a.assert_belief("Acme's CEO is Alice", source_reliability=0.7, ground=True)
    # B's stronger claim wins on both nodes after sync.
    b.assert_belief("Acme's CEO is Bob", source_reliability=0.9, ground=True)
    a.sync()
    assert a.state_hash() == b.state_hash()
    ceos = [x.proposition for x in a.active_beliefs() if "CEO is" in x.proposition]
    assert ceos == ["Acme's CEO is Bob"]


def test_cross_node_justification_and_cascade(log_path):
    log = SQLiteEventLog(log_path)
    a = DistributedBeliefStore("a", log)
    b = DistributedBeliefStore("b", log)
    premise = a.assert_belief("Acme's CEO is Bob", source_reliability=0.9, ground=True)
    b.sync()
    # B derives from A's belief (cross-node justification).
    b.assert_belief("Bob approved the budget", source_reliability=0.7,
                    justifications=[premise.id])
    a.sync()
    assert any("budget" in x.proposition for x in a.active_beliefs())
    # Retract cluster-wide: cascade fires on both nodes.
    a.retract_belief(premise.id, reason="correction")
    b.sync()
    assert a.state_hash() == b.state_hash()
    assert not any("budget" in x.proposition for x in a.active_beliefs())
    assert not any("budget" in x.proposition for x in b.active_beliefs())


def test_rejected_belief_returns_none(log_path):
    log = SQLiteEventLog(log_path)
    a = DistributedBeliefStore("a", log)
    a.assert_belief("Acme's CEO is Alice", source_reliability=0.95, ground=True)
    result = a.assert_belief("Acme's CEO is Bob", source_reliability=0.2, ground=True)
    assert result is None


def test_new_node_catches_up_on_history(log_path):
    log = SQLiteEventLog(log_path)
    a = DistributedBeliefStore("a", log)
    a.assert_belief("Fact one", source_reliability=0.9, ground=True)
    a.assert_belief("Fact two", source_reliability=0.9, ground=True)
    # Late joiner replays full history on init.
    c = DistributedBeliefStore("c", log)
    assert a.state_hash() == c.state_hash()
    assert len(c.active_beliefs()) == 2


def test_concurrent_writes_converge(log_path):
    """Simultaneous writes from two nodes: log order decides, both converge."""
    import threading
    log = SQLiteEventLog(log_path)
    a = DistributedBeliefStore("a", log)
    b = DistributedBeliefStore("b", log)
    t1 = threading.Thread(
        target=lambda: a.assert_belief("Acme's CEO is Alice",
                                       source_reliability=0.8, ground=True))
    t2 = threading.Thread(
        target=lambda: b.assert_belief("Acme's CEO is Bob",
                                       source_reliability=0.8, ground=True))
    t1.start(); t2.start(); t1.join(); t2.join()
    a.sync(); b.sync()
    # Both nodes agree (whichever won, they agree).
    assert a.state_hash() == b.state_hash()
    assert a.is_consistent()
