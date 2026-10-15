"""Tests for veritas.distributed_bft: Byzantine-tolerant gossip."""

import os
import tempfile

import pytest

from veritas import DistributedBeliefStore
from veritas.distributed_bft import ByzantineTolerantLog, _canonical

try:
    import cryptography
    CRYPTO = cryptography.__name__ == "cryptography"
except ImportError:
    CRYPTO = False
    CRYPTO = False

pytestmark = pytest.mark.skipif(not CRYPTO, reason="cryptography not installed")


@pytest.fixture
def cluster():
    paths = [tempfile.mktemp(suffix=".db") for _ in range(3)]
    logs = [ByzantineTolerantLog(p, node_id=f"node-{i}") for i, p in enumerate(paths)]
    for i, la in enumerate(logs):
        for j, lb in enumerate(logs):
            if i != j:
                la.register_peer_key(f"node-{j}", lb.get_public_key_bytes())
    yield logs, paths
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _sync_all(logs, stores):
    for i in range(len(logs)):
        for j in range(i + 1, len(logs)):
            logs[i].exchange_signed(logs[j])
    for s in stores:
        s.sync()


def test_honest_nodes_converge(cluster):
    logs, _ = cluster
    stores = [DistributedBeliefStore(f"a-{i}", log) for i, log in enumerate(logs)]
    stores[0].assert_belief("Honest fact", source_reliability=0.9, ground=True)
    _sync_all(logs, stores)
    hashes = [s.state_hash() for s in stores]
    assert hashes[0] == hashes[1] == hashes[2]


def test_forgery_rejected(cluster):
    """A node cannot forge events as another node (bad signature)."""
    logs, _ = cluster
    payload = {"belief_id": "x", "proposition": "Forged!", "ground": True}
    # Sign with node-0's key but claim to be node-1.
    msg = _canonical("assert", payload, 1, "node-1")
    sig = logs[0]._private_key.sign(msg).hex()
    ok, reason = logs[1].receive_signed(1, "node-1", "assert", payload, sig)
    assert not ok
    assert reason == "bad signature"


def test_equivocation_blacklists(cluster):
    """Signing two different events at same (lamport, node) = blacklist."""
    logs, _ = cluster
    evil = logs[2]
    lamport = 999
    for target_idx, prop in [(0, "Evil A"), (1, "Evil B")]:
        payload = {"belief_id": f"e{target_idx}", "proposition": prop,
                   "confidence": 0.9, "source": "x", "source_reliability": 0.9,
                   "ground": True, "timestamp": 0}
        msg = _canonical("assert", payload, lamport, "node-2")
        sig = evil._private_key.sign(msg).hex()
        logs[target_idx].receive_signed(lamport, "node-2", "assert", payload, sig)
    # Exchange triggers detection.
    logs[0].exchange_signed(logs[1])
    assert logs[0].is_blacklisted("node-2")
    assert logs[1].is_blacklisted("node-2")


def test_honest_converge_despite_byzantine(cluster):
    logs, _ = cluster
    stores = [DistributedBeliefStore(f"a-{i}", log) for i, log in enumerate(logs)]
    stores[0].assert_belief("Honest fact", source_reliability=0.9, ground=True)
    # Byzantine node equivocates.
    evil = logs[2]
    for target_idx, prop in [(0, "Evil A"), (1, "Evil B")]:
        payload = {"belief_id": f"e{target_idx}", "proposition": prop,
                   "confidence": 0.9, "source": "x", "source_reliability": 0.9,
                   "ground": True, "timestamp": 0}
        msg = _canonical("assert", payload, 999, "node-2")
        sig = evil._private_key.sign(msg).hex()
        logs[target_idx].receive_signed(999, "node-2", "assert", payload, sig)
    _sync_all(logs, stores)
    # Honest nodes converge; evil beliefs excluded.
    assert stores[0].state_hash() == stores[1].state_hash()
    for s in stores[:2]:
        props = [b.proposition for b in s.active_beliefs()]
        assert not any("Evil" in p for p in props)
        assert "Honest fact" in props


def test_unauthorized_retract_ignored(cluster):
    """Only the original author can explicitly retract a belief."""
    logs, _ = cluster
    stores = [DistributedBeliefStore(f"a-{i}", log) for i, log in enumerate(logs)]
    belief = stores[0].assert_belief("Victim belief", source_reliability=0.9,
                                     ground=True)
    _sync_all(logs, stores)
    # Node-1 (not the author) tries to retract node-0's belief.
    # Craft a signed retract event from node-1.
    payload = {"belief_id": belief.id, "reason": "malicious"}
    msg = _canonical("retract", payload, 500, "node-1")
    sig = logs[1]._private_key.sign(msg).hex()
    ok, _ = logs[0].receive_signed(500, "node-1", "retract", payload, sig)
    assert ok  # Signature valid, event accepted into log...
    _sync_all(logs, stores)
    # ...but the retract is ignored (not authorized).
    assert "Victim belief" in [b.proposition for b in stores[0].active_beliefs()]
    # Author CAN retract.
    stores[0].retract_belief(belief.id)
    _sync_all(logs, stores)
    assert "Victim belief" not in [b.proposition for b in stores[0].active_beliefs()]
