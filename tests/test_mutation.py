"""Targeted mutation testing for Veritas: does the test suite actually guard
the decision rule?

Instead of mutating every line (noise), each mutant breaks one specific
guarantee a reviewer cares about. A mutant is KILLED if the fast test
battery fails against it; it SURVIVES if the suite stays green — which
means the suite has a blind spot for that guarantee.

Run: python3 tests/test_mutation.py  (takes ~1-2 min)
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

BATTERY = [
    "tests/run.py",
    "tests/test_edge.py",
    "tests/test_ordering.py",
    "tests/test_v02.py",
    "tests/test_borrowed_entrenchment.py",
    "tests/test_doomed_simulation.py",
]

# Each mutant: (name, file, old, new, why_it_must_die)
MUTANTS = [
    (
        "tie-goes-to-newcomer",
        "veritas/store.py",
        "b for b in weakest_first if self.entrenchment_of(b) >= new_ent",
        "b for b in weakest_first if self.entrenchment_of(b) > new_ent",
        "ties must lose; test_entrenchment_tie_goes_to_incumbent guards this",
    ),
    (
        "retract-strongest-first",
        "veritas/store.py",
        "weakest_first = order_by_entrenchment(\n"
        "            contradictors, self.entrenchment_weights, ascending=True\n"
        "        )",
        "weakest_first = order_by_entrenchment(\n"
        "            contradictors, self.entrenchment_weights, ascending=False\n"
        "        )",
        "AGM weakest-first contraction is the core guarantee",
    ),
    (
        "drop-borrowed-prune",
        "veritas/store.py",
        "            doomed_justs = sorted(\n"
        "                self._doomed_justifications(\n"
        "                    {b.id for b in contradictors}, deduped\n"
        "                )\n"
        "            )",
        "            doomed_justs = []",
        "borrowed-entrenchment fix must be guarded",
    ),
    (
        "no-corroboration",
        "veritas/entrenchment.py",
        '    "corroboration": 0.25,  # how many independent beliefs support it',
        '    "corroboration": 0.0,  # how many independent beliefs support it',
        "corroboration component must matter",
    ),
    (
        "cascade-never-spares",
        "veritas/store.py",
        "                if indie >= self.survival_threshold:",
        "                if False:",
        "orphans that can stand alone must survive the cascade",
    ),
    (
        "cascade-always-spares",
        "veritas/store.py",
        "                if indie >= self.survival_threshold:",
        "                if True:",
        "orphans that cannot stand alone must be cascade-retracted",
    ),
    (
        "invert-rejection",
        "veritas/store.py",
        "        if unbeatable:\n            strongest = max(unbeatable, key=self.entrenchment_of)",
        "        if not unbeatable:\n            strongest = max(weakest_first, key=self.entrenchment_of)",
        "rejection must fire exactly when a contradictor is unbeatable",
    ),
]


def run_battery(root):
    """Run the fast battery; return True if ALL pass (mutant survived)."""
    env = dict(os.environ)
    for t in BATTERY:
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(root, t)],
                capture_output=True,
                timeout=300,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return True  # hung = survived (bad)
        if r.returncode != 0:
            return False  # battery failed = mutant killed
    return True


def main():
    print("targeted mutation testing: 8 decision-rule mutants")
    killed, survived, errors = [], [], []
    for name, relpath, old, new, why in MUTANTS:
        tmp = tempfile.mkdtemp(prefix="veritas-mut-")
        try:
            for d in ("veritas", "tests"):
                shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d))
            target = os.path.join(tmp, relpath)
            src = open(target).read()
            if old not in src:
                errors.append((name, "patch did not apply"))
                print(f"  ERROR: {name}: patch did not apply")
                continue
            open(target, "w").write(src.replace(old, new, 1))
            if run_battery(tmp):
                survived.append((name, why))
                print(f"  SURVIVED: {name} -- suite blind spot! ({why})")
            else:
                killed.append(name)
                print(f"  killed: {name}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    total = len(killed) + len(survived)
    rate = 100.0 * len(killed) / total if total else 0
    print(f"\nkill rate: {len(killed)}/{total} ({rate:.0f}%)")
    if survived:
        print("SURVIVORS (suite cannot detect these breakages):")
        for name, why in survived:
            print(f"  - {name}: {why}")
    if errors:
        print("ERRORS:")
        for name, e in errors:
            print(f"  - {name}: {e}")
    if survived or errors:
        sys.exit(1)
    print("all decision-rule mutants killed: suite guards the guarantees")


if __name__ == "__main__":
    main()
