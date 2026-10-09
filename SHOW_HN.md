# Show HN draft

**Title:** Show HN: Veritas – Truth maintenance for AI agents (automated belief retraction)

**Body:**

Every long-lived agent accumulates contradictions and has no principled way
to resolve them. Current memory systems do latest-wins (a rumor overwrites a
fact), manual review (doesn't scale), or per-case LLM judgment (unprincipled,
no propagation — derived beliefs survive as orphans).

Veritas is the missing piece: when new evidence contradicts old beliefs, it
automatically decides what to retract using epistemic entrenchment
(Hansson-style: weakest first, minimal change; rejects weaker newcomers),
propagates through justification dependencies (JTMS-style: ungrounded
orphans cascade), and audits everything.

Design decisions:
- The retraction decision is deterministic — computed, not guessed.
- Weak claims can't corrupt strong memory: a low-entrenchment newcomer
  contradicting a high-entrenchment incumbent is rejected, loudly.
- Nothing disappears silently: append-only audit log with reasons.

It's live on my own agent journal: it found real contradictions I'd forgotten
("Use SQLite for the journal index" vs "Avoid SQLite..."), retracted the
losing sides, honored my explicit RESOLVED decisions as higher-entrenchment
evidence. Runs on every snapshot now.

v0.2.0: core engine, SQLite persistence, CLI, pluggable detectors
(heuristic + HuggingFace NLI), indexed contradiction search, 62 tests, MIT.

Prior-art survey included in docs/ — I checked the landscape carefully in
Oct 2026 and the full closed loop (detect → entrenchment-ordered retract →
propagate → audit) exists nowhere in agent memory systems.

Try the demo: `python3 examples/demo.py`. Tell me what's wrong with it.

https://github.com/Iam-Raghava/veritas
