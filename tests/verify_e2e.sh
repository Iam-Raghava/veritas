#!/bin/bash
# verify_e2e.sh — full end-to-end verification of Veritas.
# Checks: package integrity, both test suites, demo, dogfood, CLI cycle,
# persistence roundtrip, journal sync, snapshot hook, publish-cleanliness,
# docs presence. Exits non-zero on ANY failure.
set -euo pipefail

V=~/workspace/veritas
cd "$V"
PASS=0; FAIL=0

ok()   { PASS=$((PASS+1)); echo "  ok: $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL: $1"; }

echo "== 1. package integrity =="
if python3 -m py_compile veritas/*.py tests/*.py examples/*.py deploy/*.py; then
  ok "all python files compile"
else bad "compile errors"; fi
if python3 -c "import sys; sys.path.insert(0,'.'); import veritas; assert veritas.__version__=='0.11.0'"; then
  ok "package imports, version 0.11.0"
else bad "import/version"; fi
for f in README.md ARTICLE.md SHOW_HN.md docs/prior-art.md pyproject.toml .gitignore; do
  [ -f "$f" ] && ok "doc exists: $f" || bad "missing: $f"
done

echo "== 2. test suites =="
if python3 tests/run.py > /tmp/vt1.log 2>&1 && grep -q "40 passed, 0 failed" /tmp/vt1.log; then
  ok "core suite 40/40"
else bad "core suite"; tail -3 /tmp/vt1.log; fi
if python3 tests/test_v02.py > /tmp/vt2.log 2>&1 && grep -q "23 passed, 0 failed" /tmp/vt2.log; then
  ok "v0.2 suite 23/23"
else bad "v0.2 suite"; tail -3 /tmp/vt2.log; fi
if python3 tests/test_edge.py > /tmp/vt3.log 2>&1 && grep -q "11 passed, 0 failed" /tmp/vt3.log; then
  ok "edge suite 11/11"
else bad "edge suite"; tail -3 /tmp/vt3.log; fi
if python3 tests/test_ordering.py > /tmp/vt4.log 2>&1 && grep -q "6 passed, 0 failed" /tmp/vt4.log; then
  ok "ordering proof 6/6"
else bad "ordering proof"; tail -3 /tmp/vt4.log; fi
if python3 tests/test_robust.py > /tmp/vt5.log 2>&1 && grep -q "10 passed, 0 failed" /tmp/vt5.log; then
  ok "robustness 10/10"
else bad "robustness"; tail -3 /tmp/vt5.log; fi
if python3 tests/test_readme.py > /tmp/vt6.log 2>&1 && grep -q "3/3 README blocks run" /tmp/vt6.log; then
  ok "README doctest 3/3"
else bad "README doctest"; tail -3 /tmp/vt6.log; fi
if python3 tests/test_determinism.py > /tmp/vt7.log 2>&1 && grep -q "DETERMINISTIC" /tmp/vt7.log; then
  ok "determinism"
else bad "determinism"; tail -3 /tmp/vt7.log; fi
if timeout 90 python3 tests/fuzz.py --ops 3000 --seed 11 > /tmp/vt8.log 2>&1 && grep -q "ALL INVARIANTS HELD" /tmp/vt8.log; then
  ok "fuzz 3000 ops"
else bad "fuzz"; tail -3 /tmp/vt8.log; fi
if timeout 90 python3 tests/test_audit_proof.py --ops 5000 > /tmp/vt9.log 2>&1 && grep -q "AUDIT COMPLETE" /tmp/vt9.log; then
  ok "audit proof"
else bad "audit proof"; tail -3 /tmp/vt9.log; fi
if timeout 90 python3 tests/test_persist_fuzz.py --trials 10 > /tmp/vt10.log 2>&1 && grep -q "ROUND-TRIPS IDENTICAL" /tmp/vt10.log; then
  ok "persistence fuzz"
else bad "persistence fuzz"; tail -3 /tmp/vt10.log; fi
if timeout 150 python3 tests/test_adversarial.py > /tmp/vt11.log 2>&1 && grep -q "14 passed, 0 failed" /tmp/vt11.log; then
  ok "adversarial 14/14"
else bad "adversarial"; tail -3 /tmp/vt11.log; fi
if timeout 100 python3 tests/test_deploy_fuzz.py > /tmp/vt12.log 2>&1 && grep -q "5 passed, 0 failed" /tmp/vt12.log; then
  ok "deployment fuzz 5/5"
else bad "deployment fuzz"; tail -3 /tmp/vt12.log; fi
if timeout 120 python3 tests/test_differential.py --trials 15 --ops 200 > /tmp/vt13.log 2>&1 && grep -q "DIFFERENTIAL CLEAN" /tmp/vt13.log; then
  ok "differential vs reference"
else bad "differential"; tail -3 /tmp/vt13.log; fi
if python3 tests/test_v03.py > /tmp/vt14.log 2>&1 && grep -q "12 passed, 0 failed" /tmp/vt14.log; then
  ok "v0.3 suite 12/12"
else bad "v0.3 suite"; tail -3 /tmp/vt14.log; fi
if python3 tests/test_v04.py > /tmp/vt15.log 2>&1 && grep -q "22 passed, 0 failed" /tmp/vt15.log; then
  ok "v0.5 suite 22/22"
else bad "v0.5 suite"; tail -3 /tmp/vt15.log; fi

echo "== 3. demo =="
if python3 examples/demo.py > /tmp/vtdemo.log 2>&1 && grep -q "All demo assertions passed" /tmp/vtdemo.log; then
  ok "demo runs, self-checks pass"
else bad "demo"; tail -5 /tmp/vtdemo.log; fi

echo "== 4. dogfood =="
if python3 examples/dogfood.py > /tmp/vtdog.log 2>&1 && grep -q "dogfood complete" /tmp/vtdog.log; then
  ok "dogfood runs, memory audited"
else bad "dogfood"; tail -5 /tmp/vtdog.log; fi

echo "== 5. CLI full cycle =="
export PYTHONPATH="$V"
TDB=$(mktemp /tmp/vtcli-XXXX.db); rm -f "$TDB"
cli() { python3 -m veritas.cli --db "$TDB" "$@"; }
if cli init > /dev/null 2>&1 \
  && ID1=$(cli assert "Acme's CEO is Jane Smith" --source blog --reliability 0.6 2>/dev/null | grep -o '\[[0-9a-f]*\]' | tr -d '[]') \
  && [ -n "$ID1" ] \
  && cli assert "Acme's CEO is John Doe" --source "press release" --reliability 0.97 2>/dev/null | grep -q "contracted 1" \
  && cli list 2>/dev/null | grep -q "John Doe" \
  && cli contradictions 2>/dev/null | grep -q "consistent" \
  && cli stats 2>/dev/null | grep -q "consistent.*True" \
  && cli audit 2>/dev/null | grep -q "retracted"; then
  # Retracting the already-contracted belief must report "no active belief".
  # (pipefail is on: neutralize retract's expected exit-1 before grepping.)
  retract_out=$(cli retract "$ID1" --reason "test" 2>&1 || true)
  if echo "$retract_out" | grep -q "no active belief"; then
    ok "CLI: init/assert/contract/list/contradictions/stats/audit (+correct retract-noop)"
  else
    bad "CLI retract-noop behavior"
  fi
else bad "CLI cycle"; fi
rm -f "$TDB"

echo "== 6. persistence roundtrip =="
if python3 - > /tmp/vtpersist.log 2>&1 <<'EOF'
import sys, tempfile, os
sys.path.insert(0, os.path.expanduser("~/workspace/veritas"))
from veritas import BeliefStore, save, load
s = BeliefStore()
a = s.assert_belief("Acme's CEO is Jane", source="blog", source_reliability=0.6)
b = s.assert_belief("Jane founded Acme", source="derived", source_reliability=0.4, justifications=[a.id])
s.assert_belief("Acme's CEO is John", source="press", source_reliability=0.97)
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f: path = f.name
save(s, path)
s2 = load(path)
assert len(s2.all_beliefs()) == len(s.all_beliefs()), "belief count"
assert {x.proposition for x in s2.active_beliefs()} == {x.proposition for x in s.active_beliefs()}, "active set"
assert not s2.get(a.id).is_active, "retraction status"
assert len(s2.audit) == len(s.audit), "audit"
assert s2.is_consistent(), "consistency"
os.unlink(path)
print("PERSIST OK")
EOF
grep -q "PERSIST OK" /tmp/vtpersist.log; then
  ok "persistence: save/load preserves beliefs, audit, consistency"
else bad "persistence"; tail -5 /tmp/vtpersist.log; fi

echo "== 7. journal sync (live deployment) =="
rm -f "$V/deploy/journal.db"
if python3 "$V/deploy/journal_sync.py" > /tmp/vtsync.log 2>&1 \
  && [ -f "$V/deploy/journal.db" ] \
  && [ -f "$V/deploy/report.md" ] \
  && grep -q "retracted" "$V/deploy/report.md"; then
  ok "journal sync: DB + report created, real retractions found"
else bad "journal sync"; tail -5 /tmp/vtsync.log; fi
if python3 -c "
import sys; sys.path.insert(0, '$V')
from veritas import load, register_detector
sys.path.insert(0, '$V/deploy')
import journal_sync
s = load('$V/deploy/journal.db')
assert s.is_consistent(), 'live store inconsistent'
print('live store consistent:', s.stats())
" > /tmp/vtlive.log 2>&1; then
  ok "live store loads and is consistent"
else bad "live store load"; tail -5 /tmp/vtlive.log; fi

echo "== 8. snapshot.sh hook =="
if bash -n ~/workspace/agent-journal/scripts/snapshot.sh \
  && grep -q "journal_sync.py" ~/workspace/agent-journal/scripts/snapshot.sh; then
  ok "snapshot.sh valid bash + veritas hook present"
else bad "snapshot hook"; fi

echo "== 9. publish-clean =="
if ! grep -ri "password\s*=\s*[\"'][^\"']*[\"']\|api_key\s*=\s*[\"'][^\"'...][^\"']*[\"']\|secret_key\s*=\s*[\"'][^\"']*[\"']" --include="*.py" veritas/ | grep -v "^Binary" | head -1 | grep -q .; then
  ok "no credentials in code"
else bad "possible credential in code"; fi
if ! grep -r "RAG's\|RAG " --include="*.py" veritas/ examples/ | grep -v "most RAG" | head -1 | grep -q .; then
  ok "no personal name in code"
else bad "personal name in code"; fi
if grep -q "deploy/\*.db" .gitignore && grep -q "deploy/report.md" .gitignore; then
  ok "deploy artifacts gitignored"
else bad ".gitignore"; fi

echo ""
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
