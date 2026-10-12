"""Realistic agent trajectory test: not synthetic CEO facts.

Simulates a customer-support agent's memory over 30 days:
- Subscription changes (upgrades, downgrades)
- Support tickets (issues reported, resolved)
- Contradictory information (stale cache vs live API)
- Deletion request (GDPR)

This is the kind of workload a frontier lab would actually test.
"""
import sys
sys.path.insert(0, ".")

from veritas import BeliefStore


def test_support_agent_trajectory():
    """30-day support agent memory: must stay coherent."""
    s = BeliefStore()

    # Day 1: Customer signs up for Pro
    s.assert_belief("Customer X subscription is Pro",
                    source="billing-api", source_reliability=0.95,
                    confidence=0.95, ground=True, timestamp=1.0)

    # Day 5: Support ticket - "can't access feature"
    t1 = s.assert_belief("Customer X reported issue with Feature A",
                         source="support-ticket", source_reliability=0.8,
                         confidence=0.9, ground=True, timestamp=5.0)

    # Day 6: Derived: "Feature A is broken for Pro users"
    # (justified by the ticket + subscription)
    sub = [b for b in s.active_beliefs()
           if "subscription is Pro" in b.proposition][0]
    s.assert_belief("Feature A has outage affecting Pro tier",
                    justifications=[t1.id, sub.id],
                    source="support-agent", source_reliability=0.6,
                    confidence=0.7, timestamp=6.0)

    # Day 10: Customer downgrades to Free (billing API, high reliability)
    # Same source, newer timestamp: explicit supersession (retract old, assert new)
    for b in list(s.active_beliefs()):
        if "Customer X subscription" in b.proposition:
            s.retract_belief(b.id, reason="superseded by newer billing API report")
    s.assert_belief("Customer X subscription is Free",
                    source="billing-api", source_reliability=0.95,
                    confidence=0.95, ground=True, timestamp=10.0)

    # Day 11: Stale cache says Pro (low reliability, should be rejected)
    r = s.assert_belief("Customer X subscription is Pro",
                        source="stale-cache", source_reliability=0.2,
                        confidence=0.5, ground=True, timestamp=11.0)
    assert r is None, "Stale cache should not overwrite billing API"

    # Verify: subscription is Free, not Pro
    active = [b.proposition for b in s.active_beliefs()]
    assert any("subscription is Free" in p for p in active)
    assert not any("subscription is Pro" in p and "Free" not in p
                   for p in active if "Customer X subscription" in p), \
        f"Pro should be gone: {active}"

    # Day 15: Issue resolved
    s.assert_belief("Feature A outage resolved",
                    source="engineering", source_reliability=0.9,
                    confidence=0.95, ground=True, timestamp=15.0)

    # Day 20: GDPR deletion request - forget all Customer X data
    for b in list(s.active_beliefs()):
        if "Customer X" in b.proposition:
            s.retract_belief(b.id, reason="GDPR deletion request")

    # Verify: no Customer X beliefs remain, no orphans
    remaining = [b for b in s.active_beliefs() if "Customer X" in b.proposition]
    assert len(remaining) == 0, f"GDPR failed: {remaining}"

    # The derived "outage affecting Pro tier" should be cascade-retracted
    # (its justifications are gone)
    orphans = [b for b in s.active_beliefs()
               if "outage affecting" in b.proposition]
    assert len(orphans) == 0, f"Orphaned derivation: {orphans}"

    # Audit: can we explain what happened?
    # (At least verify the audit log has entries)
    assert len(s._audit._events) > 0

    print("PASS: realistic trajectory (subscription, outage, stale cache, GDPR)")


def test_contradiction_chain():
    """A chain of contradictory updates: only the strongest survives."""
    s = BeliefStore()

    # Three sources, different reliabilities, contradictory claims
    # (Using possessive form for heuristic detector)
    s.assert_belief("Deployment's target is us-east-1",
                    source="docs", source_reliability=0.5,
                    confidence=0.7, ground=True, timestamp=1.0)
    s.assert_belief("Deployment's target is us-west-2",
                    source="slack-rumor", source_reliability=0.3,
                    confidence=0.5, ground=True, timestamp=2.0)
    s.assert_belief("Deployment's target is eu-west-1",
                    source="infra-api", source_reliability=0.95,
                    confidence=0.95, ground=True, timestamp=3.0)

    active = [b.proposition for b in s.active_beliefs()
              if "Deployment's target" in b.proposition]
    assert len(active) == 1, f"Should have exactly 1 winner: {active}"
    assert "eu-west-1" in active[0], f"Strongest should win: {active}"

    print("PASS: contradiction chain (strongest source wins)")


if __name__ == "__main__":
    test_support_agent_trajectory()
    test_contradiction_chain()
    print("\nAll realistic trajectory tests pass")
