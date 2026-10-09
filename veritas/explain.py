"""Explanation generator for Veritas retractions.

When a belief is retracted, developers need to know *why*. This module
reconstructs human-readable explanations from the audit trail:

- What contradicted it?
- Why did it lose (entrenchment comparison)?
- What cascaded from it?
- What survives and why?

This is critical for agent debugging: "Why did my agent forget X?"
"""

from __future__ import annotations

from veritas.store import BeliefStore


class Explainer:
    """Generate natural-language explanations for belief changes."""

    def __init__(self, store: BeliefStore) -> None:
        self.store = store

    def why_retracted(self, belief_id: str) -> str:
        """Explain why a belief was retracted."""
        # Find retraction event in audit
        events = [e for e in self.store.audit if e.belief_id == belief_id]
        if not events:
            return f"No record of belief {belief_id}."
        # Get the belief (may be retracted)
        belief = self.store._beliefs.get(belief_id)
        prop = belief.proposition if belief else "(unknown)"
        # Find retraction event
        ret = next(
            (e for e in events if e.event_type == "retracted"), None
        )
        if not ret:
            return f"Belief '{prop}' was not retracted."
        lines = [f"Belief '{prop}' was retracted."]
        lines.append(f"Reason: {ret.reason}")
        # Find what contradicted it (assertion events around the same time)
        # Look for contraction events mentioning this belief
        for e in self.store.audit:
            if e.event_type == "asserted" and belief_id in str(
                e.details.get("retracted_ids", [])
            ):
                lines.append(
                    f"Contradicted by: '{e.proposition}' "
                    f"(entrenchment {e.details.get('entrenchment', '?')})"
                )
        # Check cascade
        cascaded = [
            e for e in self.store.audit
            if e.event_type == "retracted"
            and "cascade" in e.reason.lower()
            and e.belief_id != belief_id
        ]
        if cascaded:
            lines.append(
                f"Cascade: {len(cascaded)} dependent belief(s) also retracted."
            )
        return "\n".join(lines)

    def why_rejected(self, proposition: str) -> str:
        """Explain why a proposition was rejected at assertion time."""
        events = [
            e for e in self.store.audit
            if e.event_type == "rejected" and e.proposition == proposition
        ]
        if not events:
            return f"No rejection record for '{proposition}'."
        e = events[-1]
        return (
            f"Proposition '{proposition}' was rejected.\n"
            f"Reason: {e.reason}\n"
            f"Entrenchment: {e.details.get('entrenchment', 'unknown')}"
        )

    def belief_lineage(self, belief_id: str) -> str:
        """Show the justification chain for a belief."""
        belief = self.store._beliefs.get(belief_id)
        if not belief:
            return f"Belief {belief_id} not found."
        lines = [f"Lineage for '{belief.proposition}':"]
        def _walk(bid: str, depth: int, seen: set) -> None:
            if bid in seen or depth > 5:
                return
            seen.add(bid)
            b = self.store._beliefs.get(bid)
            if not b:
                return
            indent = "  " * depth
            status = "active" if b.is_active else "retracted"
            lines.append(f"{indent}- {b.proposition} [{status}]")
            for jset in b.justifications:
                for pid in jset:
                    _walk(pid, depth + 1, seen)
        _walk(belief_id, 0, set())
        return "\n".join(lines)

    def summary(self) -> str:
        """Overall store health summary."""
        active = self.store.active_beliefs()
        total = len(self.store._beliefs)
        retracted = total - len(active)
        events = list(self.store.audit)
        return (
            f"Store summary:\n"
            f"  Active beliefs: {len(active)}\n"
            f"  Retracted: {retracted}\n"
            f"  Total events: {len(events)}\n"
            f"  Consistent: {self.store.is_consistent()}"
        )
