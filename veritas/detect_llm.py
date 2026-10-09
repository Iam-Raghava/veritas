"""LLM-as-judge contradiction detector.

Uses a frontier LLM (Claude 5.5, GPT-6, Gemini 4, etc.) to judge whether
two propositions contradict. This is the most accurate detector — it
leverages the model's reasoning to catch subtle contradictions that
patterns and embeddings miss.

The LLM is used ONLY as a judge, not for the retraction decision.
Veritas still makes the deterministic entrenchment-ordered contraction.

Supports any OpenAI-compatible API (OpenAI, Anthropic via proxy,
together.ai, etc.) or custom callables.

Usage:
    from veritas.detect_llm import LLMJudgeDetector
    detector = LLMJudgeDetector(
        api_key="...",
        model="claude-sonnet-5-5",  # or gpt-6-sol, gemini-4-argon, etc.
        api_base="https://api.anthropic.com/v1",  # optional
    )
    store = BeliefStore(detector=detector)
"""

from __future__ import annotations

import json
from typing import Callable

from veritas.belief import Belief


JUDGE_PROMPT = """Do these two statements contradict each other?

Statement A: {a}
Statement B: {b}

Consider:
- Direct logical contradiction (A says X, B says not X)
- Temporal scope (if about different times, no contradiction)
- Semantic opposition (antonyms, mutually exclusive states)

Reply with ONLY a JSON object: {{"contradicts": true/false, "reason": "..."}}"""


class LLMJudgeDetector:
    """Contradiction detector using an LLM as judge.

    Args:
        judge_fn: Callable taking (prop_a, prop_b) -> bool.
            Use this for custom LLM integrations.
        api_key: API key for OpenAI-compatible endpoint.
        model: Model ID (e.g. "claude-sonnet-5-5", "gpt-6-sol").
        api_base: Base URL (default: OpenAI).
    """

    name = "llm_judge"

    def __init__(
        self,
        judge_fn: Callable[[str, str], bool] | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
    ) -> None:
        if judge_fn is not None:
            self._judge = judge_fn
        elif api_key is not None:
            self._judge = self._make_api_judge(api_key, model, api_base)
        else:
            raise ValueError("Provide judge_fn or api_key")
        self.model = model

    def _make_api_judge(
        self, api_key: str, model: str, api_base: str
    ) -> Callable[[str, str], bool]:
        def judge(a: str, b: str) -> bool:
            import urllib.request
            prompt = JUDGE_PROMPT.format(a=a, b=b)
            data = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 200,
            }).encode()
            req = urllib.request.Request(
                f"{api_base}/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    text = result["choices"][0]["message"]["content"]
                    # Parse JSON from response
                    parsed = json.loads(text.strip())
                    return bool(parsed.get("contradicts", False))
            except Exception:
                return False
        return judge

    def __call__(self, a: Belief | str, b: Belief | str) -> bool:
        pa = a.proposition if isinstance(a, Belief) else a
        pb = b.proposition if isinstance(b, Belief) else b
        if pa == pb:
            return False
        return self._judge(pa, pb)
