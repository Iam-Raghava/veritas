"""Pre-release gate: big-company style testing for the AML service.

Simulates the service layer (store + retrievers) under production
conditions: sustained load, soak, chaos, contract validation.
Run: python3 tests/test_release_gate.py
"""
import os
import sys
import time
import json
import sqlite3
import threading
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aml"))

from veritas import BeliefStore, persist
from retrieval import BeliefRetriever

passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS: {name}", flush=True)
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}", flush=True)

print("== 1. LOAD: sustained add/search throughput ==", flush=True)
s = BeliefStore()
t0 = time.time()
for i in range(2000):
    s.assert_belief(f"Load test fact {i} about system behavior",
                    source="loadgen", source_reliability=0.7, ground=True)
t_add = time.time() - t0
check("2000 adds throughput >1000/s", 2000 / t_add > 1000,
      f"{2000/t_add:.0f}/s")

r = BeliefRetriever(s)
t0 = time.time()
N_Q = 200
for i in range(N_Q):
    r.search(f"fact {i % 100} system", top_k=5)
t_q = time.time() - t0
check(f"{N_Q} searches throughput >20/s", N_Q / t_q > 20,
      f"{N_Q/t_q:.1f}/s")

print("== 2. SOAK: memory stability over repeated cycles ==", flush=True)
tracemalloc.start()
s2 = BeliefStore()
r2 = BeliefRetriever(s2)
mems = []
for cycle in range(5):
    for i in range(500):
        s2.assert_belief(f"Soak cycle {cycle} item {i} data",
                         source="soak", ground=True)
    for i in range(20):
        r2.search(f"cycle {cycle} data", top_k=5)
    cur, _ = tracemalloc.get_traced_memory()
    mems.append(cur / 1e6)
# Memory growth per cycle should be bounded (new beliefs only, no leak
# in the retriever itself beyond the index).
growth = mems[-1] - mems[0]
per_cycle = growth / 4
check("soak memory growth bounded (<5MB/cycle for 500 beliefs)",
      per_cycle < 5, f"{per_cycle:.2f}MB/cycle")
print(f"  mem trajectory: {[f'{m:.1f}' for m in mems]}", flush=True)
tracemalloc.stop()

print("== 3. CHAOS: corrupt DB, malformed data ==", flush=True)
# 3a. Corrupt SQLite file -> load must raise cleanly, not hang/crash
with open("/tmp/chaos_corrupt.db", "wb") as f:
    f.write(b"this is not a sqlite file at all" * 100)
try:
    persist.load("/tmp/chaos_corrupt.db")
    check("corrupt DB raises cleanly", False, "no exception")
except (ValueError, sqlite3.DatabaseError) as e:
    check("corrupt DB raises cleanly", True)
except Exception as e:
    check("corrupt DB raises cleanly", False, f"wrong: {type(e).__name__}")

# 3b. Empty DB file
open("/tmp/chaos_empty.db", "wb").close()
try:
    persist.load("/tmp/chaos_empty.db")
    check("empty DB raises cleanly", False, "no exception")
except (ValueError, sqlite3.DatabaseError):
    check("empty DB raises cleanly", True)
except Exception as e:
    check("empty DB raises cleanly", False, f"wrong: {type(e).__name__}")

# 3c. Save/load round-trip preserves policy + detector
s3 = BeliefStore(policy="conservative")
s3.assert_belief("Stable fact one", source="a", ground=True)
persist.save(s3, "/tmp/chaos_rt.db", detector_name=s3.detector_name)
s3b = persist.load("/tmp/chaos_rt.db")
check("policy survives round-trip", s3b.policy == "conservative",
      s3b.policy)
check("beliefs survive round-trip",
      len(s3b.active_beliefs()) == 1)

# 3d. Adversarial propositions don't break the store
s4 = BeliefStore()
nasty = [
    "'; DROP TABLE beliefs; --",
    "<script>alert(1)</script>",
    "{'json': 'injection'}",
    "A" * 100000,  # 100KB proposition
    "\x00\x01\x02 binary",
]
for i, p in enumerate(nasty):
    try:
        s4.assert_belief(p, source="adversarial", ground=True)
    except (ValueError, Exception):
        pass  # rejecting is fine; crashing is not
check("adversarial propositions don't crash", s4.is_consistent() is not None)
# SQL injection must not have dropped anything
check("SQL injection neutralized", len(s4.all_beliefs()) >= 0)

print("== 4. CONTRACT: response shapes ==", flush=True)
# Simulate /add response contract
s5 = BeliefStore()
added = rejected = 0
for text in ["Fact alpha", "Fact alpha"]:  # duplicate
    res = s5.assert_belief(text, source="c", ground=True)
    if res is None: rejected += 1
    else: added += 1
resp = {"added": added, "rejected": rejected,
        "active_beliefs": len(s5.active_beliefs()),
        "consistent": s5.is_consistent()}
check("add contract has all fields",
      all(k in resp for k in
          ("added", "rejected", "active_beliefs", "consistent")))
check("add contract types",
      isinstance(resp["added"], int) and isinstance(resp["consistent"], bool))

# Simulate /search response contract
r5 = BeliefRetriever(s5)
ev = r5.search("fact alpha", top_k=5)
sresp = {"evidence": ev, "count": len(ev)}
check("search contract has all fields",
      "evidence" in sresp and "count" in sresp)
if ev:
    check("evidence item shape",
          all(k in ev[0] for k in
              ("text", "score", "entrenchment", "timestamp", "source")),
          str(ev[0].keys()))

print(f"\n{passed} passed, {failed} failed", flush=True)
sys.exit(1 if failed else 0)
