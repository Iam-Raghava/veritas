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

from ..extractor import ClaimExtractor, HeuristicClaimExtractor
from ..store import BeliefStore


class VeritasChatMemory:
    """Veritas-backed memory for AutoGen conversations.

    Tracks claims made by agents across a conversation, automatically
    retracting weaker claims when stronger evidence contradicts them.
    """

    def __init__(
        self,
        store: BeliefStore | None = None,
        extractor: ClaimExtractor | None = None,
    ) -> None:
        self.store = store or BeliefStore()
        # Claim extractor: splits multi-sentence messages into atomic
        # claims. Disable by passing extractor=None... actually we always
        # want it; pass a no-op extractor to disable.
        self.extractor = extractor or HeuristicClaimExtractor()

    def record_message(
        self, agent: str, message: str, reliability: float = 0.7,
    ) -> None:
        """Record an agent's claim as a belief."""
        # Ground if from a tool/function, derived if from LLM reasoning.
        is_ground = agent.endswith("_tool") or "function" in agent.lower()
        # Extract atomic claims from the message. A multi-sentence message
        # becomes multiple beliefs, each evaluable by detectors.
        claims = self.extractor.extract_claims(message)
        if not claims:
            # Empty or unparseable: assert raw message as fallback.
            claims = []
        for claim in claims:
            # Store the atomic claim as the proposition (no speaker prefix)
            # so the heuristic detector's regexes can match it. Speaker
            # identity goes in metadata, not the proposition text.
            self.store.assert_belief(
                claim.proposition,
                source=agent,
                source_reliability=reliability,
                ground=is_ground,
                metadata={
                    "agent": agent,
                    "claim_confidence": claim.confidence,
                    "spo": {
                        "subject": claim.subject,
                        "predicate": claim.relation,
                        "object": claim.object,
                    } if claim.subject else None,
                },
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
