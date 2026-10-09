"""LangChain integration for Veritas.

Provides a Veritas-backed memory that automatically truth-maintains
an agent's beliefs: contradictions trigger entrenchment-ordered
contraction, not silent overwrites.

Usage:
    from veritas.integrations.langchain import VeritasMemory
    from langchain.agents import create_react_agent

    memory = VeritasMemory()
    # ... use with your agent ...

Requires langchain-core (pip install langchain-core).
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.memory import BaseMemory
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseMemory = object  # type: ignore[assignment,misc]

from ..store import BeliefStore


class VeritasMemory(BaseMemory):  # type: ignore[valid-type,misc]
    """LangChain memory backed by Veritas truth maintenance.

    Every observation is asserted as a belief. When new information
    contradicts stored beliefs, Veritas automatically retracts the
    weaker side (entrenchment-ordered) and cascades to dependents —
    instead of LangChain's default latest-wins behavior.
    """

    def __init__(self, store: BeliefStore | None = None, **kwargs: Any) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required: pip install langchain-core"
            )
        super().__init__(**kwargs)
        self.store = store or BeliefStore()

    @property
    def memory_variables(self) -> list[str]:
        return ["veritas_beliefs"]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        beliefs = self.store.active_beliefs()
        # Format as context for the agent.
        lines = [
            f"- {b.proposition} (confidence={b.confidence:.2f}, "
            f"source={b.source})"
            for b in sorted(beliefs, key=lambda b: b.timestamp)
        ]
        return {"veritas_beliefs": "\n".join(lines) if lines else "(no beliefs)"}

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> None:
        # Extract human input and AI output as beliefs.
        human = inputs.get("input", "")
        ai_out = outputs.get("output", "")
        if human:
            self.store.assert_belief(
                f"User said: {human}",
                source="user",
                source_reliability=0.9,
                ground=True,
            )
        if ai_out:
            # AI outputs are derived, not ground — they can be retracted
            # if contradicted by stronger evidence.
            self.store.assert_belief(
                f"Agent concluded: {ai_out}",
                source="agent",
                source_reliability=0.6,
            )

    def clear(self) -> None:
        # Retract all (audited, not wiped).
        for b in self.store.active_beliefs():
            self.store.retract_belief(b.id, reason="memory cleared")
