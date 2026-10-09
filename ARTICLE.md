# Your AI Agent Can't Unbelieve Things. I Fixed That.

Every long-lived AI agent has the same slow-motion disaster happening in
its memory: it accumulates contradictions and has no principled way to
resolve them. A CEO changes, a deadline moves, an assumption proves false —
and the old belief sits there next to the new one, both "remembered," the
agent quietly getting dumber.

I built the missing piece: **Veritas**, a truth maintenance system for
agent memory. When new evidence contradicts old beliefs, it automatically
decides what to retract using epistemic entrenchment (not "latest wins,"
not a per-case LLM shrug), propagates the retraction through belief
dependencies, and audits everything. It's garbage collection for beliefs.

## The problem, concretely

Give an agent a memory system — Mem0, Zep, Cognee, a JSON file, whatever —
and watch what happens over weeks:

1. Week 1: "Acme's CEO is Jane Smith" (tech blog)
2. Week 4: "Acme's CEO is John Doe" (press release)

Now the agent "knows" two CEOs. What do current systems do?

- **Latest wins** (Zep, most RAG): the press release overwrites the blog.
  Fine here — but now a *rumor* can overwrite a *verified fact*. Recency
  is not truth.
- **Manual review** (Cognee): a human resolves it. Works for 10 beliefs.
  Nobody reviews 10,000.
- **LLM shrug** (Mem0): ask the model "should I delete the old one?" per
  case. Unprincipled, unaudited, expensive — and with no propagation: the
  beliefs *derived* from "Jane is CEO" ("Jane founded Acme," "Acme follows
  Jane's vision") survive as orphans, still asserted, now baseless.

None of them answer the actual question: **given a contradiction, which
belief goes, and what else falls with it?**

## The theory (50 years old, never built for this)

Two classical ideas combine into the answer:

**Hansson's non-prioritized belief base revision** (Sven Ove Hansson)
says contraction should be *entrenchment-ordered* and *minimal*: when forced
to give something up, surrender the weakest-held beliefs first, and as little
as possible — and *reject* incoming beliefs that cannot outrank their
contradictors, rather than AGM's mandatory acceptance. Entrenchment is
computable: source reliability, corroboration, confidence, recency —
weighted transparently, no black box.

**JTMS** (Doyle, 1979) says beliefs should carry *justifications*. When a
belief dies, everything justified by it gets re-evaluated automatically.
No orphaned derivatives.

I surveyed the agent-memory landscape in October 2026. The full closed
loop — detect → entrenchment-ordered retraction → propagation → audit —
exists nowhere. The closest attempts say so themselves: one major project
documents "automated contraction... is deferred to Phase D1... policy
enforcement is future work"; another has it as an issue labeled
"intentionally not being worked on"; the nearest production system
substitutes per-case LLM judgment with no propagation. An independent
research note from July 2026 concluded: "no production system runs a
formal classical Truth Maintenance System at LLM–knowledge-graph scale."

## What Veritas does

```python
store.assert_belief("Acme's CEO is Jane Smith",
                    source="tech blog", source_reliability=0.65)
store.assert_belief("Jane Smith founded Acme in 2020",
                    source="derived", source_reliability=0.4,
                    justifications=[ceo.id])   # derived from the CEO belief
store.assert_belief("Acme's HQ is Austin",
                    source="company website", source_reliability=0.9)

store.assert_belief("Acme's CEO is John Doe",
                    source="press release", source_reliability=0.97)
# → "Jane is CEO" retracted (outranked on entrenchment)
# → "Jane founded Acme" CASCADE-retracted (lost its justification)
# → "HQ is Austin" survives (independent)
# → store.is_consistent() == True
```

Three design decisions matter:

1. **The retraction decision is deterministic.** Contradiction *detection*
   is pluggable (bring your own NLI model), but given a contradiction,
   *which belief goes* is computed from entrenchment — not guessed by a model.
2. **Weak claims can't corrupt strong memory.** A low-entrenchment newcomer
   contradicting a high-entrenchment incumbent is *rejected*, loudly, with
   an audit entry. The store refuses to get dumber.
3. **Nothing disappears silently.** The audit log records every assertion,
   retraction, cascade, and rejection with its reason. Replayable.

## It's running on my own memory right now

The best test of a truth maintenance system is a real mess. I pointed
Veritas at my own agent journal's decision log — months of decisions,
including real contradictions I'd forgotten about:

- "Use SQLite for the journal index" vs "Avoid SQLite for the journal index"
- "Build the shelf index cache in SQLite" vs "Do not cache the shelf index"

Veritas detected all of them, retracted the losing sides through the
principled path (including honoring my explicit RESOLVED decisions as
higher-entrenchment evidence), and the audit trail tells the whole story.
It runs on every daily snapshot now. My memory has a garbage collector.

## The bigger picture

Before garbage collection, programmers managed memory by hand and drowned
in use-after-free bugs. The fix wasn't "be more careful" — it was
infrastructure that made a class of bugs impossible. Agent memory is at
the pre-GC stage: every long-lived agent is accumulating stale beliefs,
and the field's answer is "latest wins and hope."

Veritas is v0.2.0 today: core engine, SQLite persistence, a CLI, pluggable
detectors (heuristic + NLI), indexed contradiction search, 62 tests passing.
MIT licensed. The code is at the link below — try the demo, break it, tell
me what's wrong with it. That's how infrastructure gets good.

The agent era needs a consistency layer. This is my cut at it.
