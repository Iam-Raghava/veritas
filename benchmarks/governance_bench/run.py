#!/usr/bin/env python3
"""Governance benchmark runner.

Runs all six scenarios against latest-wins, append-only, and Veritas.
Deterministic (fixed seed). No network. Writes results.json + RESULTS.md
next to this file and prints the markdown table to stdout.

    python3 benchmarks/governance_bench/run.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, ".")

from benchmarks.governance_bench import SEED  # noqa: E402
from benchmarks.governance_bench.scenarios import SCENARIOS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["veritas", "latest-wins", "append-only"]


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, dict):
        return (f"R:{v['recall']:.1f}% / P:{v['precision']:.1f}% "
                f"(orphans:{v['orphans']})")
    return f"{v:.1f}%"


def main():
    all_results = {}
    timings = {}
    total_start = time.time()
    for sid, desc, fn, direction, unit in SCENARIOS:
        start = time.time()
        res = fn(SEED)
        elapsed = time.time() - start
        timings[sid] = elapsed
        all_results[sid] = {
            "description": desc,
            "direction": direction,
            "unit": unit,
            "scores": res,
            "seconds": round(elapsed, 2),
        }
        print(f"[{elapsed:5.1f}s] {sid}: " +
              ", ".join(f"{k}={fmt(v)}" for k, v in res.items()),
              flush=True)
    total = time.time() - total_start

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    lines = [
        "# Governance Benchmark: Veritas vs Naive Baselines",
        "",
        f"_Seed {SEED}. Deterministic — rerun reproduces these numbers. "
        f"Total runtime {total:.1f}s. No network, no API calls._",
        "",
        "## Results",
        "",
        "| Scenario | Veritas | Latest-wins | Append-only | Unit |",
        "|---|---|---|---|---|",
    ]
    for sid, desc, fn, direction, unit in SCENARIOS:
        scores = all_results[sid]["scores"]
        lines.append(
            f"| {desc} | {fmt(scores.get('veritas'))} | "
            f"{fmt(scores.get('latest-wins'))} | "
            f"{fmt(scores.get('append-only'))} | {unit} "
            f"({'higher' if direction == 'higher' else 'lower'} better) |"
        )
    lines += [
        "",
        "## Reading the table",
        "",
        "- **Contradiction storm**: high-reliability facts under attack by "
        "weaker sources. Veritas adjudicates by entrenchment; naive stores "
        "keep whoever wrote last.",
        "- **Cascade integrity**: retracting a root must retract its "
        "derivations (TMS). Naive stores have no derivation tracking, so "
        "dependents are orphaned — measured honestly, not assumed.",
        "- **Temporal accuracy**: naive stores have no validity-window "
        "concept, hence n/a (not a fake zero).",
        "- **Deletion compliance**: recall = targets actually gone, "
        "precision = non-targets preserved. Veritas cascades deletion to "
        "derived beliefs.",
        "- **Zombie resistance**: a killed belief must stay dead under "
        "re-assertion pressure.",
        "- **Audit completeness**: can the system answer _why_ something "
        "was retracted and _what_ the state was at time T?",
        "",
        "## Reproduce",
        "",
        "```",
        "python3 benchmarks/governance_bench/run.py",
        "```",
    ]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"Wrote results.json and RESULTS.md ({total:.1f}s total)")


if __name__ == "__main__":
    main()
