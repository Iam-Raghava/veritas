# Agent Memory Challenge 2026 — Cycle 2: Entry Plan (DRAFT)

Verified against agentmemories.ai on 2026-10-08. Organizer: 中国图像图形学会 (CSIG).

## Why enter

- The Textual Memory track's capability taxonomy explicitly measures **Memory
  governance** (updates, conflict resolution, deletion, forgetting) and
  **Temporal & event understanding** (distinguish order, updates, latest valid
  state) — these are Veritas's home turf. Most memory systems do recall;
  Veritas retracts outdated beliefs on principle. This is the independent
  validation vector the project is missing: self-benchmarks are honest, but
  an external leaderboard is what a lab hiring manager or acquirer would trust.
- Plausible prize target: **Best Technical Innovation Award (¥5,000)** rather
  than first place — recall quality at 6,000+ instances / ~300M tokens is a
  retrieval game where industrial systems lead. Enter for validation and the
  innovation angle, not to sweep the recall leaderboard.
- >50 teams already registered. Results publish mid-November.

## Hard facts

| Item | Value |
|---|---|
| Deadline (materials + evaluation applications) | **2026-10-31 23:59 UTC+8** (23 days) |
| Evaluation queue closes | 2026-11-04 23:59 UTC+8 |
| Tracks | Textual / Coding / Multimodal (frozen independently) |
| Division | Open-source Methods (public repo + fixed commit required) |
| Flow | Apply for AML Key → deploy Add/Search API → smoke test → full eval → review |
| Full runs per AML key | Max 2 per track; second only 30 days after first completes |
| Key issuance | Next day, before 19:00 Beijing time |

## What entering requires (in order)

1. **Public GitHub repo + fixed commit** — needs RAG: repo creation under his
   account, license pick (MIT recommended), `veritas-tms` name consistent.
   PyPI name `veritas-tms` verified available 2026-10-08 (`veritas` itself is taken
   by an unrelated package; pyproject already uses `veritas-tms`).
2. **AML Key application** — needs RAG: submit evaluation access request at
   agentmemories.ai/evaluation; key issued next day by 19:00 Beijing time.
3. **Add/Search HTTP service** over the Veritas store:
   - `Add`: receive conversation/event/document text → assert as beliefs with
     source entrenchment, run contraction (dedup + conflict resolution).
   - `Search`: return relevant evidence for a query + scope; the platform
     controls answer generation and scoring, so Search returns evidence only.
   - Small FastAPI/Flask wrapper (~150 lines); Veritas's hot path is ~80µs per
     belief and 58k inserts/sec, so scale is not the bottleneck.
4. **Public deployment** — a reachable HTTPS endpoint RAG controls (fly.io /
   Railway / cheap VPS all fine; participant funds their own infra, the
   platform runs answer + eval + orchestration).
5. **Smoke test → Full evaluation → publication review**, monitored on the AML
   Evaluation page.

## Model-constraint nuance (read this before choosing a division)

The academic leaderboard requires embeddings = `text-embedding-v4` and LLM
components = `gpt-4o-mini` (reranker unrestricted). Systems using other
open-weight models are steered to the **industry leaderboard**, which does NOT
require commercialization — non-commercial open-source projects may enter.

Consequences for Veritas:
- With the **heuristic (default) detector**: no external model at all — eligible
  anywhere; strongest "principled, model-free" story.
- With the **HuggingFace NLI detector** (roberta-large-mnli): enter the industry
  leaderboard; richer contradiction detection but heavier to deploy.

Recommendation: enter Textual Memory, industry leaderboard, heuristic detector —
fastest path, cleanest story, no model-dependency audit trail. The innovation
angle (principled retraction + audit) survives regardless of detector.

## Working-back timeline (deadline Oct 31)

| Week | Action |
|---|---|
| Now – Oct 12 | Public repo + license; build Add/Search wrapper (assistant) |
| Oct 13 – 16 | Deploy endpoint; apply for AML key; smoke test |
| Oct 17 – 24 | Full evaluation run #1; review private results |
| Oct 25 – 31 | Buffer for re-run/fixes; final materials submitted |

## What still needs RAG (cannot be done without him)

- Public GitHub repo under his account + license decision
- AML key application (tied to his identity)
- Hosting account for the Add/Search endpoint
- Track/division choice confirmation

Everything else — wrapper service, submission materials, README framing —
can be built and prepared in advance.
