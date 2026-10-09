"""Tests for Veritas. Run with: python -m pytest tests/ -q  (or python tests/run.py)"""
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import (
    Belief,
    BeliefStore,
    entrenchment,
    heuristic_contradiction,
    order_by_entrenchment,
)

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


def fresh_store(**kw):
    return BeliefStore(**kw)


print("== entrenchment ==")
b_high = Belief("X is Y", confidence=0.95, source="press release",
                source_reliability=0.97, timestamp=time.time())
b_low = Belief("X is Z", confidence=0.5, source="rumor",
               source_reliability=0.3,
               timestamp=time.time() - 60 * 86400)
check("higher reliability+recency => higher entrenchment",
      entrenchment(b_high) > entrenchment(b_low),
      f"{entrenchment(b_high):.3f} vs {entrenchment(b_low):.3f}")
check("entrenchment in [0,1]",
      0.0 <= entrenchment(b_low) <= 1.0 and 0.0 <= entrenchment(b_high) <= 1.0)
b_corr = Belief("X is Y", justifications=["a", "b", "c"])
b_nocorr = Belief("X is Y")
check("corroboration raises entrenchment",
      entrenchment(b_corr) > entrenchment(b_nocorr))
ordered = order_by_entrenchment([b_high, b_low], ascending=True)
check("ascending order puts weakest first", ordered[0] is b_low)

print("== detection ==")
d1 = Belief("Acme's CEO is Jane Smith")
d2 = Belief("Acme's CEO is John Doe")
d3 = Belief("Acme's headquarters is Austin")
check("functional property conflict detected",
      heuristic_contradiction(d1, d2))
check("independent claims not conflicting",
      not heuristic_contradiction(d1, d3))
d4 = Belief("The sky is blue")
d5 = Belief("The sky is not blue")
check("negation detected", heuristic_contradiction(d4, d5))
check("no self-contradiction", not heuristic_contradiction(d1, d1))
e1 = Belief("P", metadata={"contradicts": ["xyz"]})
e2 = Belief("Q")
e2.id = "xyz"
check("explicit metadata pairs detected", heuristic_contradiction(e1, e2))

print("== assertion without conflict ==")
s = fresh_store()
b = s.assert_belief("Acme's HQ is Austin", source="website", source_reliability=0.9)
check("assert returns belief", b is not None)
check("belief is active", b.is_active)
check("store consistent", s.is_consistent())
check("audit recorded assertion", len(s.audit.of_type("asserted")) == 1)

print("== contraction: stronger newcomer wins ==")
s = fresh_store()
import time as _time
old = s.assert_belief("Acme's CEO is Jane Smith",
                      source="blog", source_reliability=0.6,
                      confidence=0.7,
                      metadata={},
                      # age 60 days so recency favors the newcomer
                      timestamp=_time.time() - 60 * 86400)
new = s.assert_belief("Acme's CEO is John Doe",
                      source="press release", source_reliability=0.97,
                      confidence=0.98)
check("newcomer accepted", new is not None)
check("old belief retracted", not s.get(old.id).is_active)
check("only newcomer active",
      [x.proposition for x in s.active_beliefs()] == ["Acme's CEO is John Doe"])
check("store consistent after contraction", s.is_consistent())
check("retraction audited", len(s.audit.of_type("retracted")) == 1)

print("== rejection: weaker newcomer loses ==")
s = fresh_store()
strong = s.assert_belief("Acme's CEO is Jane Smith",
                         source="SEC filing", source_reliability=0.99,
                         confidence=0.99)
weak = s.assert_belief("Acme's CEO is John Doe",
                       source="rumor", source_reliability=0.2,
                       confidence=0.4)
check("weak newcomer rejected", weak is None)
check("strong belief still active", s.get(strong.id).is_active)
check("rejection audited", len(s.audit.of_type("rejected")) == 1)
check("store consistent", s.is_consistent())

print("== propagation: cascade through dependencies ==")
s = fresh_store()
ceo = s.assert_belief("Acme's CEO is Jane Smith",
                      source="article", source_reliability=0.7, confidence=0.8,
                      timestamp=_time.time() - 60 * 86400)
founder = s.assert_belief("Jane Smith founded Acme in 2020",
                          source="derived", source_reliability=0.4,
                          confidence=0.7, justifications=[ceo.id])
hq = s.assert_belief("Acme's HQ is Austin",
                     source="website", source_reliability=0.9, confidence=0.95)
new_ceo = s.assert_belief("Acme's CEO is John Doe",
                          source="press release", source_reliability=0.97,
                          confidence=0.98)
check("newcomer accepted", new_ceo is not None)
check("contradicted CEO belief retracted", not s.get(ceo.id).is_active)
check("dependent founder belief cascade-retracted",
      not s.get(founder.id).is_active)
check("independent HQ belief survives", s.get(hq.id).is_active)
check("cascade audited",
      len(s.audit.of_type("cascade_retracted")) >= 1)
check("store consistent", s.is_consistent())

print("== propagation: multi-level cascade ==")
s = fresh_store()
a = s.assert_belief("Acme's CEO is Jane Smith",
                    source="article", source_reliability=0.7, confidence=0.8,
                    timestamp=_time.time() - 60 * 86400)
b = s.assert_belief("Jane leads Acme", source="derived",
                    source_reliability=0.4, confidence=0.6,
                    justifications=[a.id])
c = s.assert_belief("Acme strategy follows Jane's vision", source="derived",
                    source_reliability=0.4, confidence=0.6,
                    justifications=[b.id])
s.assert_belief("Acme's CEO is John Doe",
                source="press release", source_reliability=0.97, confidence=0.98)
check("level-1 dependent retracted", not s.get(b.id).is_active)
check("level-2 dependent retracted", not s.get(c.id).is_active)

print("== propagation: ground observation survives orphaning ==")
s = fresh_store()
base = s.assert_belief("Acme's CEO is Jane Smith",
                       source="article", source_reliability=0.7, confidence=0.8,
                       timestamp=_time.time() - 60 * 86400)
# Ground observation: survives even when its justification dies.
dep = s.assert_belief("Jane Smith runs Acme day-to-day",
                      source="interview", source_reliability=0.95,
                      confidence=0.95, justifications=[base.id],
                      ground=True)
s.assert_belief("Acme's CEO is John Doe",
                source="press release", source_reliability=0.97, confidence=0.98)
check("grounded dependent survives orphaning",
      s.get(dep.id).is_active)
check("orphan lost the dead justification",
      base.id not in s.get(dep.id).justifications)

print("== propagation: ungrounded orphan retracts (zombie prevention) ==")
s = fresh_store()
base2 = s.assert_belief("Acme's CEO is Jane Smith",
                        source="article", source_reliability=0.7,
                        confidence=0.8,
                        timestamp=_time.time() - 60 * 86400)
# Derived (not ground): high confidence alone must NOT save it.
zombie = s.assert_belief("Jane Smith runs Acme day-to-day",
                         source="derived", source_reliability=0.4,
                         confidence=0.99, justifications=[base2.id])
s.assert_belief("Acme's CEO is John Doe",
                source="press release", source_reliability=0.97,
                confidence=0.98)
check("ungrounded orphan retracts (no zombie)",
      not s.get(zombie.id).is_active)

print("== manual retraction propagates ==")
s = fresh_store()
x = s.assert_belief("X is true", source="s", source_reliability=0.8)
y = s.assert_belief("Y follows from X", source="derived",
                    source_reliability=0.4, confidence=0.6,
                    justifications=[x.id])
retracted = s.retract_belief(x.id, "operator decision")
check("manual retraction returns cascade",
      {r.id for r in retracted} == {x.id, y.id})
check("retracting unknown id is safe", s.retract_belief("nope") == [])

print("== all-or-nothing: cannot beat every contradictor ==")
s = fresh_store()
s1 = s.assert_belief("Acme's CEO is Jane Smith",
                     source="SEC filing", source_reliability=0.99, confidence=0.99)
mid = s.assert_belief("Acme's CEO is John Doe",
                      source="press release", source_reliability=0.97,
                      confidence=0.98)
# mid should have been rejected (SEC filing outranks press release here)
check("mid-strength newcomer rejected by stronger incumbent", mid is None)
check("incumbent untouched", s.get(s1.id).is_active)

print("== find_contradictions ==")
s = fresh_store()
p1 = Belief("Acme's CEO is Jane Smith")
p2 = Belief("Acme's CEO is John Doe")
s._store(p1)
s._store(p2)
pairs = s.find_contradictions()
check("detects inconsistent pair", len(pairs) == 1)

print("== audit completeness ==")
s = fresh_store()
k = s.assert_belief("K", source="s", source_reliability=0.9)
evts = s.audit.events_for(k.id)
check("every belief has audit trail", len(evts) >= 1)
check("seq numbers ordered",
      [e.seq for e in s.audit] == list(range(len(s.audit))))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
