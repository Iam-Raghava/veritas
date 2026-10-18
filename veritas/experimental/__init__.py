"""Veritas experimental modules.

These modules are functional but live outside the core truth-maintenance
story. They are kept importable for the users who need them, but they are
not part of the reviewed core and are not exported from ``veritas``'s
primary namespace surface (except where re-exported for compatibility).

Contents:
- ``distributed_gossip``: GossipEventLog — P2P replication of the event
  log (partition tolerance, no single point of failure).
- ``distributed_bft``: ByzantineTolerantLog — Ed25519-signed event log
  with equivocation detection (requires the ``cryptography`` package).
"""
