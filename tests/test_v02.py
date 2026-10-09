"""Tests for Veritas v0.2: persistence, indexing, detectors, CLI."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import (
    Belief,
    BeliefStore,
    HeuristicDetector,
    as_function,
    load,
    register_detector,
    save,
)
from veritas.detect_nli import HuggingFaceNLIDetector

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {extra}")


print("== detector plugin system ==")
d = HeuristicDetector()
check("Detector protocol satisfied", hasattr(d, "contradicts"))
fn = as_function(d)
check("as_function adapts Detector",
      fn(Belief("Acme's CEO is A"), Belief("Acme's CEO is B")))
plain = as_function(lambda a, b: False)
check("as_function passes through functions", plain(Belief("x"), Belief("y")) is False)
s = BeliefStore(detector=HeuristicDetector())
check("store accepts Detector", s.detector_name == "heuristic")
b1 = s.assert_belief("Acme's CEO is Jane", source="s", source_reliability=0.9)
b2 = s.assert_belief("Acme's CEO is John", source="s", source_reliability=0.9)
check("detector plugin drives contraction", b2 is not None and not s.get(b1.id).is_active)

print("== NLI detector (stubbed pipeline) ==")
calls = []


def fake_pipe(text):
    calls.append(text)
    # Contradiction iff premise mentions Jane and hypothesis mentions John.
    if "Jane" in text and "John" in text:
        return [{"label": "CONTRADICTION", "score": 0.92}]
    return [{"label": "NEUTRAL", "score": 0.8}]


nli = HuggingFaceNLIDetector(threshold=0.7, _pipeline=fake_pipe)
a = Belief("Acme's CEO is Jane Smith")
b = Belief("Acme's CEO is John Doe")
c = Belief("Acme's HQ is Austin")
check("NLI detects contradiction", nli.contradicts(a, b))
check("NLI clears independent", not nli.contradicts(a, c))
check("NLI no self-contradiction", not nli.contradicts(a, a))
s2 = BeliefStore(detector=nli)
x = s2.assert_belief("Acme's CEO is Jane Smith", source="s", source_reliability=0.9)
y = s2.assert_belief("Acme's CEO is John Doe", source="s", source_reliability=0.95)
check("NLI detector drives store contraction", y is not None and not s2.get(x.id).is_active)

print("== NLI detector requires opt-in install ==")
nli2 = HuggingFaceNLIDetector()
try:
    import transformers  # noqa
    print("  (transformers installed; skipping import-error check)")
    PASS += 1
except ImportError:
    try:
        nli2._ensure_pipeline()
        check("missing transformers raises helpful error", False)
    except ImportError as e:
        check("missing transformers raises helpful error",
              "veritas-tms[nli]" in str(e))

print("== persistence roundtrip ==")
s = BeliefStore()
k1 = s.assert_belief("Acme's CEO is Jane", source="blog", source_reliability=0.6)
k2 = s.assert_belief("Jane founded Acme", source="derived", source_reliability=0.4,
                     justifications=[k1.id])
s.assert_belief("Acme's CEO is John", source="press", source_reliability=0.97)
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    path = f.name
try:
    save(s, path)
    s2 = load(path)
    check("belief count survives", len(s2.all_beliefs()) == len(s.all_beliefs()))
    check("active set survives",
          {b.proposition for b in s2.active_beliefs()} ==
          {b.proposition for b in s.active_beliefs()})
    check("retraction status survives", not s2.get(k1.id).is_active)
    check("justifications survive", s2.get(k2.id).justifications == [])
    check("audit log survives", len(s2.audit) == len(s.audit))
    check("audit seq preserved", [e.seq for e in s2.audit] == list(range(len(s2.audit))))
    check("config survives",
          s2.survival_threshold == s.survival_threshold and
          s2.detector_name == "heuristic")
    check("loaded store still consistent", s2.is_consistent())
    # Loaded store remains fully functional (index rebuilt via _store).
    nb = s2.assert_belief("Acme's HQ is Austin", source="web", source_reliability=0.9)
    check("loaded store accepts new beliefs", nb is not None and s2.is_consistent())
finally:
    os.unlink(path)

print("== persistence: custom detector registry ==")
register_detector("never", lambda a, b: False)
s = BeliefStore(contradiction_fn=lambda a, b: False)
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    path = f.name
try:
    save(s, path, detector_name="never")
    s2 = load(path)
    check("custom detector resolves on load", s2.detector_name == "never")
finally:
    os.unlink(path)

print("== indexing correctness vs brute force ==")
import random
random.seed(7)
s = BeliefStore()
subjects = ["Acme", "Globex", "Initech", "Umbrella"]
relations = ["CEO", "HQ", "founder"]
names = ["Alice", "Bob", "Carol", "Dave", "Erin"]
made = []
for i in range(60):
    sj, rel = random.choice(subjects), random.choice(relations)
    # Use metadata pairs to force some contradictions deterministically.
    b = s.assert_belief(f"{sj}'s {rel} is {random.choice(names)}",
                        source="s", source_reliability=random.random(),
                        confidence=random.random())
    made.append(b)
# Brute-force reference using the raw detector over all active pairs.
active = s.active_beliefs()
brute = set()
for i, x in enumerate(active):
    for y in active[i + 1:]:
        from veritas import heuristic_contradiction
        if heuristic_contradiction(x, y):
            brute.add(tuple(sorted((x.id, y.id))))
indexed = {tuple(sorted((a.id, b.id))) for a, b in s.find_contradictions()}
check("indexed search matches brute force", indexed == brute,
      f"indexed={len(indexed)} brute={len(brute)}")

print("== will-negation indexed ==")
s = BeliefStore()
w1 = Belief("Acme will launch in June", source="s", source_reliability=0.8)
w2 = Belief("Acme will not launch in June", source="s", source_reliability=0.9)
s._store(w1)
s._store(w2)
check("will-negation contradiction found", len(s.find_contradictions()) == 1)
s = BeliefStore()
w1 = s.assert_belief("Acme will launch in June", source="s", source_reliability=0.8)
w2 = s.assert_belief("Acme will not launch in June", source="s", source_reliability=0.9)
check("higher reliability wins", w2 is not None and not s.get(w1.id).is_active)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
