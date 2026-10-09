"""CrewAI integration for Veritas.

Wraps CrewAI agent memory with Veritas truth maintenance.
Every tool result and agent output is asserted as a belief;
contradictions trigger automatic entrenchment-ordered contraction.

Usage:
    from veritas.integrations.crewai import VeritasCrewMemory
    memory = VeritasCrewMemory()
    # Pass to your CrewAI agent config
"""

from __future__ import annotations

from typing import Any

from ..store import BeliefStore


class VeritasCrewMemory:
    """Veritas-backed memory for CrewAI agents.

    Intercepts agent observations and outputs, maintaining a
    consistent belief base instead of an append-only log.
    """

    def __init__(self, store: BeliefStore | None = None) -> None:
        self.store = store or BeliefStore()

    def remember_observation(
        self, observation: str, source: str = "tool",
        reliability: float = 0.8,
    ) -> None:
        """Assert a tool observation as a ground belief."""
        self.store.assert_belief(
            observation,
            source=source,
            source_reliability=reliability,
            ground=True,
        )

    def remember_conclusion(
        self, conclusion: str, reliability: float = 0.6,
    ) -> None:
        """Assert an agent conclusion as a derived belief."""
        self.store.assert_belief(
            conclusion,
            source="agent",
            source_reliability=reliability,
        )

    def get_context(self, max_beliefs: int = 20) -> str:
        """Format active beliefs as context for the agent."""
        beliefs = sorted(
            self.store.active_beliefs(),
            key=lambda b: b.timestamp,
        )[-max_beliefs:]
        if not beliefs:
            return "(no beliefs)"
        return "\n".join(f"- {b.proposition}" for b in beliefs)

    def check_consistency(self) -> bool:
        return self.store.is_consistent()
