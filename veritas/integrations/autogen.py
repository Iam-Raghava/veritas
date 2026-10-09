"""AutoGen integration for Veritas.

Provides Veritas truth maintenance for AutoGen multi-agent conversations.
When agents disagree, Veritas resolves by entrenchment rather than
letting contradictions accumulate in the chat history.

Usage:
    from veritas.integrations.autogen import VeritasChatMemory
    memory = VeritasChatMemory()
    # Hook into your AutoGen conversation
"""

from __future__ import annotations

from ..store import BeliefStore


class VeritasChatMemory:
    """Veritas-backed memory for AutoGen conversations.

    Tracks claims made by agents across a conversation, automatically
    retracting weaker claims when stronger evidence contradicts them.
    """

    def __init__(self, store: BeliefStore | None = None) -> None:
        self.store = store or BeliefStore()

    def record_message(
        self, agent: str, message: str, reliability: float = 0.7,
    ) -> None:
        """Record an agent's claim as a belief."""
        # Ground if from a tool/function, derived if from LLM reasoning.
        is_ground = agent.endswith("_tool") or "function" in agent.lower()
        self.store.assert_belief(
            f"{agent}: {message}",
            source=agent,
            source_reliability=reliability,
            ground=is_ground,
        )

    def get_consistent_view(self) -> list[str]:
        """Return active (non-retracted) claims."""
        return [
            b.proposition
            for b in sorted(
                self.store.active_beliefs(), key=lambda b: b.timestamp
            )
        ]

    def contradictions_resolved(self) -> int:
        """Count of retractions (contradictions auto-resolved)."""
        return sum(
            1 for e in self.store.audit
            if e.event_type == "retracted"
        )
