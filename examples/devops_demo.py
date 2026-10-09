#!/usr/bin/env python3
"""Veritas demo: DevOps agent debugging a production incident.

Shows why truth maintenance matters: an agent gathering evidence from
multiple tools accumulates contradictions. Without Veritas, it acts on
stale beliefs. With Veritas, contradictions resolve automatically by
entrenchment, and cascades clean up derived conclusions.

Run: python3 examples/devops_demo.py
"""

import sys
import time
sys.path.insert(0, ".")

from veritas import BeliefStore
from veritas.detect_semantic import SemanticDetector
from veritas.observability import Metrics


def main():
    print("=" * 70)
    print("Veritas Demo: Production Incident Debugging")
    print("=" * 70)

    metrics = Metrics()
    store = BeliefStore(detector=SemanticDetector(), metrics=metrics)
    now = time.time()

    print("\n[00:00] Agent starts investigating latency spike...\n")

    # Phase 1: Initial observations (ground truths from tools)
    store.assert_belief(
        "API latency p99 is 2.5s (normal: 200ms)",
        source="datadog", source_reliability=0.95,
        ground=True, valid_from=now - 300,
    )
    print("  + Datadog: latency spike confirmed (ground)")

    store.assert_belief(
        "Database CPU is at 95%",
        source="cloudwatch", source_reliability=0.95,
        ground=True, valid_from=now - 280,
    )
    print("  + CloudWatch: DB CPU critical (ground)")

    # Phase 2: Agent derives a hypothesis
    db_cpu = store.active_beliefs()[1]
    store.assert_belief(
        "Root cause is database overload",
        source="agent", source_reliability=0.6,
        justifications=[[db_cpu.id]],
    )
    print("  + Agent infers: DB overload is root cause (derived)")

    # Phase 3: Contradictory evidence arrives
    print("\n[00:05] New evidence contradicts the hypothesis...\n")
    # The DB CPU reading was wrong — sensor glitch. Retract it.
    # This should cascade to the "DB overload" conclusion.
    old_cpu = [b for b in store.active_beliefs()
               if "Database CPU is at 95%" in b.proposition][0]
    store.retract_belief(old_cpu.id, reason="sensor recalibrated: reading was faulty")
    print("  - Retracted: 'Database CPU is at 95%' (sensor faulty)")
    print("  - Cascade: 'Root cause is database overload' should fall...")

    store.assert_belief(
        "Database CPU is at 15% (normal)",
        source="cloudwatch", source_reliability=0.95,
        ground=True, valid_from=now - 60,
    )
    print("  + CloudWatch: DB CPU normal (ground, recent)")

    # The old "DB CPU 95%" belief is temporally scoped to the past,
    # so it doesn't contradict. But the derived "root cause" belief
    # loses its justification...
    store.assert_belief(
        "API latency is still 2.5s",
        source="datadog", source_reliability=0.95,
        ground=True, valid_from=now - 30,
    )
    print("  + Datadog: latency still high (ground, recent)")

    # Phase 4: Agent finds the real cause
    print("\n[00:10] Agent investigates further...\n")
    store.assert_belief(
        "Redis cache hit rate dropped to 12%",
        source="redis-cli", source_reliability=0.9,
        ground=True, valid_from=now - 20,
    )
    print("  + Redis: cache hit rate collapsed (ground)")

    redis = [b for b in store.active_beliefs()
             if "Redis" in b.proposition][0]
    latency = [b for b in store.active_beliefs()
               if "latency is still" in b.proposition][0]
    store.assert_belief(
        "Root cause is cache failure causing DB thundering herd",
        source="agent", source_reliability=0.75,
        justifications=[[redis.id, latency.id]],
    )
    print("  + Agent infers: cache failure is root cause (derived)")

    # Results
    print("\n" + "=" * 70)
    print("FINAL BELIEF STATE")
    print("=" * 70)
    for b in sorted(store.active_beliefs(), key=lambda b: b.timestamp):
        status = "GROUND" if b.ground else "derived"
        print(f"  [{status:7s}] {b.proposition}")
        print(f"             entrenchment={store.entrenchment_of(b):.3f}")

    print(f"\nMetrics: {metrics.summary()}")
    print(f"\nAudit trail: {len(store.audit)} events")
    print("\nWithout Veritas, the agent would still believe 'DB overload'")
    print("was the root cause. With Veritas, contradictions resolved")
    print("automatically by entrenchment ordering.")


if __name__ == "__main__":
    main()
