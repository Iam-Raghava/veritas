# Prior-art survey: automated belief retraction for agent memory

**Verdict (verified October 2026): the full closed loop — contradiction
detection → automated entrenchment-ordered retraction → dependency
propagation → audit — has not been built by any AI agent memory system.**

What exists are fragments. Every system below was checked against the
complete loop; each is missing the automated *principled* retraction
decision, the propagation, or both.

## The four closest attempts (in their own words)

**krunch3r76/universal-llm-gateway** — closest on ordering, explicitly
defers the decision:
> "Automated contraction — where the system autonomously decides which
> beliefs to contract based on entrenchment — is deferred to Phase D1...
> The ordering mechanism is proven; policy enforcement is future work."
> "supersede operates on exactly one assertion ID. No cascade, no side effects."

**EPISTEMICS (nateschmiedehaus/librainian, issue #390)** — open, labeled
"intentionally not being worked on":
> "No code path calls contract() or revise() with the persistent graph
> as input"; "there is no unified retraction policy."

**hupe1980/hirn** — reflection adjusts confidence; it does not retract:
> "halving a belief's credence and writing a Contradicts edge are not
> reversible"; conflict repair is "proposals + required approval".

**yigraf** (PyPI) — justification tracking with grounding tiers, but
explicitly manual resolution:
> "yigraf never silently picks a winner" — conflicts are settled by the
> human ("you decide which wins"). Code-repo-anchored, not general-purpose.
> Has "grounding tier" ("we measured this" vs "we assumed this") but no
> automated entrenchment-ordered contraction and no dependency cascades.

## Near misses

| System | What it does | What's missing |
|---|---|---|
| Mem0 | LLM picks DELETE for contradicted memories | Per-case LLM judgment — no principled policy, no propagation, no entrenchment |
| Zep | Invalidates older facts at write time | Model-decided recency = latest-wins |
| Cognee | Manual invalidation marks | Manual, off by default |
| nasiko_labs | Explicit refusal: "deciding which of two contradictory beliefs is really true requires judgment the memory layer should not be making on its own" | The decision itself |

## Classical foundations (not agent-memory implementations)

- **AGM belief revision** (Alchourrón, Gärdenfors, Makinson 1985) — the
  entrenchment-ordered contraction theory. Pure logic; no agent system.
  Veritas actually implements Hansson's *non-prioritized* base revision:
  unlike AGM, it can reject incoming beliefs that fail to outrank
  incumbents — the correct model for noisy agent inputs.
- **JTMS** (Doyle 1979) / **ATMS** (de Kleer 1986) — justification
  tracking and propagation. Built for symbolic KBs, never adapted to
  LLM agent memory at production scale.
- An independent research note (July 2026) concluded: "no production
  system runs a formal classical Truth Maintenance System at
  LLM–knowledge-graph scale."

## What Veritas adds

The combination — automated + principled (entrenchment-ordered) +
propagating (JTMS-style) + audited — implemented as a deterministic
engine for agent memory. That combination exists nowhere else as of
this writing.
