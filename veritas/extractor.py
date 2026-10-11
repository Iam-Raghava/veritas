"""Semantic-to-symbolic ingestion layer for Veritas.

LLMs emit stream-of-consciousness paragraphs, not atomic propositions.
This module decomposes unstructured narrative text into atomic claims
with full Veritas schema (justifications, temporal bounds, grounding).

Pipeline:
    Raw Agent Text ──► [ClaimExtractor] ──► List[ExtractedClaim] ──► [Veritas Store]

Schema:
    - proposition: atomic claim text
    - subject, relation, object: SPO triplet (if extractable)
    - premises_referred: substrings/IDs of premises this claim depends on
    - confidence: extractor confidence
    - source: origin identifier
    - valid_from, valid_until: temporal bounds (if detectable)
    - ground: True if direct observation, False if derived inference

Implementations:
- HeuristicExtractor: rule-based, no model. (Re-exports from extract.py
  with the full schema.)
- StructuredLLMExtractor: uses an OpenAI-compatible API with JSON schema
  to extract claims. Supports OpenAI, Anthropic (via proxy), Ollama, vLLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ExtractedClaim:
    """An atomic claim with full Veritas ingestion schema."""
    proposition: str
    confidence: float = 0.8
    source: str = "agent"
    # Substrings or belief IDs this claim depends on (for justifications).
    premises_referred: list[str] = field(default_factory=list)
    # SPO triplet if extractable.
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    # Temporal bounds (Unix timestamps, if detectable from text).
    valid_from: float | None = None
    valid_until: float | None = None
    # True if direct observation (tool output, user statement);
    # False if derived inference (LLM reasoning).
    ground: bool = False
    # Character offsets in source text (provenance).
    start: int = 0
    end: int = 0


class ClaimExtractor(Protocol):
    """Protocol for claim extractors."""

    def extract_claims(
        self, text: str, source: str = "agent",
    ) -> list[ExtractedClaim]:
        """Decompose text into atomic claims."""
        ...


# Re-export the heuristic implementation with the new schema.
from veritas.extract import HeuristicClaimExtractor as _HeuristicBase


class HeuristicExtractor:
    """Rule-based extractor with full ExtractedClaim schema.

    Wraps HeuristicClaimExtractor to emit ExtractedClaim objects.
    """

    def __init__(self, min_length: int = 3) -> None:
        self._base = _HeuristicBase(min_length=min_length)

    def extract_claims(
        self, text: str, source: str = "agent",
    ) -> list[ExtractedClaim]:
        base_claims = self._base.extract_claims(text)
        return [
            ExtractedClaim(
                proposition=c.text,
                confidence=c.confidence,
                source=source,
                subject=c.subject,
                relation=c.predicate,
                object=c.object,
                ground=False,  # Heuristic can't determine; default False.
                start=c.start,
                end=c.end,
            )
            for c in base_claims
        ]

    def extract_propositions(self, text: str) -> list[str]:
        """Convenience: return just the claim texts."""
        return self._base.extract_propositions(text)


class StructuredLLMExtractor:
    """LLM-based extractor using structured JSON output.

    Requires an OpenAI-compatible API (OpenAI, Anthropic via proxy,
    Ollama, vLLM). The model must emit a JSON array of claims with
    the ExtractedClaim schema.

    The prompt instructs the model to:
    - Break text into single-relation atomic claims
    - Mark ground=True for direct observations, False for inferences
    - Extract temporal bounds when mentioned
    - List premises each claim depends on
    """

    SYSTEM_PROMPT = """You are a claim extraction system. Decompose the input text into atomic propositions.

Rules:
1. Each claim must express a SINGLE factual relation (subject-predicate-object).
2. Split conjunctions: "A and B are X" becomes two claims.
3. Mark ground=true for direct observations (tool outputs, user statements, measurements).
   Mark ground=false for inferences, conclusions, or reasoning.
4. Extract temporal bounds if the text mentions when the claim is/was valid.
5. List premises_referred: substrings from the text that this claim depends on for its justification.
6. Output ONLY a JSON array. No markdown fences, no explanation.

Example input: "Acme's CEO is Jane. She was appointed in January 2024. Revenue grew because sales increased."

Example output:
[
  {"proposition": "Acme's CEO is Jane", "confidence": 0.95, "subject": "Acme", "relation": "CEO", "object": "Jane", "ground": true, "premises_referred": [], "valid_from": null, "valid_until": null},
  {"proposition": "Jane was appointed in January 2024", "confidence": 0.9, "subject": "Jane", "relation": "appointed", "object": "January 2024", "ground": true, "premises_referred": [], "valid_from": 1704067200.0, "valid_until": null},
  {"proposition": "Revenue grew", "confidence": 0.85, "subject": "Revenue", "relation": "grew", "object": null, "ground": true, "premises_referred": [], "valid_from": null, "valid_until": null},
  {"proposition": "Sales increased", "confidence": 0.85, "subject": "Sales", "relation": "increased", "object": null, "ground": true, "premises_referred": [], "valid_from": null, "valid_until": null}
]"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def extract_claims(
        self, text: str, source: str = "agent",
    ) -> list[ExtractedClaim]:
        """Extract claims via LLM structured output."""
        import urllib.request
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"]
                # Strip markdown fences if present.
                clean = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "", content.strip(), flags=re.MULTILINE,
                ).strip()
                data = json.loads(clean)
                # Handle both {"claims": [...]} and [...] formats.
                items = data.get("claims", data) if isinstance(data, dict) else data
                if not isinstance(items, list):
                    return []
                return [
                    ExtractedClaim(
                        proposition=item.get("proposition", ""),
                        confidence=float(item.get("confidence", 0.8)),
                        source=source,
                        premises_referred=item.get("premises_referred", []),
                        subject=item.get("subject"),
                        relation=item.get("relation"),
                        object=item.get("object"),
                        valid_from=item.get("valid_from"),
                        valid_until=item.get("valid_until"),
                        ground=bool(item.get("ground", False)),
                    )
                    for item in items
                    if item.get("proposition")
                ]
        except Exception as e:
            # Fail open: fall back to heuristic.
            print(f"[StructuredLLMExtractor] LLM failed ({e}); "
                  f"falling back to heuristic")
            return HeuristicExtractor().extract_claims(text, source=source)
