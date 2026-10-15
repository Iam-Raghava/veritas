"""Tests for veritas.distributed_gossip: P2P replication without SPOF."""

import os
import tempfile

import pytest

from veritas import DistributedBeliefStore, GossipEventLog


@pytest.fixture
def two_nodes():
    pa = tempfile.mktemp(suffix=".db")
    pb = tempfile.mktemp(suffix=".db")
    log_a = GossipEventLog(pa, node_id="a")
    log_b = GossipEventLog(pb, node_id="b")
    yield log_a, log_b, pa, pb
    for p in (pa, pb):
        try:
            os.unlink(p)
        except OSError:
            pass


def test_gossip_converges(two_nodes):
    log_a, log_b, _, _ = two_nodes
    a = DistributedBeliefStore("agent-a", log_a)
    b = DistributedBeliefStore("agent-b", log_b)
    a.assert_belief("Sky is blue", source_reliability=0.9, ground=True)
    log_a.exchange(log_b)
    b.sync()
    assert a.state_hash() == b.state_hash()


def test_partition_heal_deterministic_merge(two_nodes):
    log_a, log_b, _, _ = two_nodes
    a = DistributedBeliefStore("agent-a", log_a)
    b = DistributedBeliefStore("agent-b", log_b)
    # Partition: independent writes, no communication.
    a.assert_belief("Acme's CEO is Alice", source_reliability=0.8, ground=True)
    b.assert_belief("Acme's CEO is Bob", source_reliability=0.9, ground=True)
    assert a.state_hash() != b.state_hash()
    # Heal: exchange and converge.
    log_a.exchange(log_b)
    a.sync()
    b.sync()
    assert a.state_hash() == b.state_hash()
    ceos = [x.proposition for x in a.active_beliefs() if "CEO is" in x.proposition]
    assert ceos == ["Acme's CEO is Bob"]  # higher reliability wins


def test_no_single_point_of_failure(two_nodes):
    """Every node holds a full copy; losing a node loses nothing."""
    log_a, log_b, pa, pb = two_nodes
    a = DistributedBeliefStore("agent-a", log_a)
    a.assert_belief("Fact one", source_reliability=0.9, ground=True)
    a.assert_belief("Fact two", source_reliability=0.9, ground=True)
    log_a.exchange(log_b)
    # Simulate node A dying: delete its database entirely.
    del a, log_a
    os.unlink(pa)
    # Node B still has everything.
    b = DistributedBeliefStore("agent-b", GossipEventLog(pb, node_id="b"))
    assert len(b.active_beliefs()) == 2


def test_three_nodes_converge():
    paths = [tempfile.mktemp(suffix=".db") for _ in range(3)]
    logs = [GossipEventLog(p, node_id=n) for p, n in zip(paths, "abc")]
    try:
        stores = [DistributedBeliefStore(f"agent-{n}", log)
                  for n, log in zip("abc", logs)]
        stores[0].assert_belief("Shared fact", source_reliability=0.9, ground=True)
        # Full mesh exchange.
        for i in range(3):
            for j in range(i + 1, 3):
                logs[i].exchange(logs[j])
        for s in stores:
            s.sync()
        hashes = [s.state_hash() for s in stores]
        assert hashes[0] == hashes[1] == hashes[2]
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
