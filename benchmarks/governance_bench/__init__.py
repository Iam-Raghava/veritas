"""Governance benchmark: Veritas vs naive agent-memory baselines.

What this measures (and why it matters for frontier agent systems):

    Agent memory is not a key-value store. In production, facts get
    contradicted by sources of varying reliability, derived conclusions
    outlive their premises, temporal validity windows expire, users
    demand deletion, and every retraction needs an audit trail. Naive
    stores (dict overwrite, append-only logs) fail these silently.

    This suite runs six adversarial scenarios against three systems and
    reports honest, reproducible numbers. Baselines are NOT strawmen:
    LatestWins and AppendOnly are what most agent frameworks actually do.
    Where a baseline lacks a capability entirely (temporal queries, audit),
    we report n/a rather than inventing a score.

Determinism: every scenario takes a seeded random.Random. Same seed ->
same numbers, on any machine. No network, no API calls, pure local.
"""

SEED = 20261012
