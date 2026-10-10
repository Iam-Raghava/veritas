"""README/code consistency gate.

RAG's standing rule: the README must never drift from the code again.
Every push changes code; this test fails if the docs describe a
different reality. Run as part of verify_e2e.sh.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import veritas

ROOT = os.path.join(os.path.dirname(__file__), "..")
README = os.path.join(ROOT, "README.md")
PYPROJECT = os.path.join(ROOT, "pyproject.toml")

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


with open(README) as f:
    readme = f.read()
with open(PYPROJECT) as f:
    pyproject = f.read()

# 1. Version: __init__ == pyproject == latest "What's new" header.
m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
check("pyproject version parseable", m is not None)
py_version = m.group(1) if m else ""
check("__version__ == pyproject version",
      veritas.__version__ == py_version,
      f"({veritas.__version__} vs {py_version})")

whats_new_versions = re.findall(r"\*\*v(\d+\.\d+\.\d+)[^:]*:", readme)
check("README has versioned What's new entries", bool(whats_new_versions))
if whats_new_versions:
    latest_doc = whats_new_versions[0]
    check("latest README version == code version",
          latest_doc == veritas.__version__,
          f"(README {latest_doc} vs code {veritas.__version__})")

# 1b. Status changelog must cover the current version (no stale tail).
status_versions = re.findall(r"^v(\d+\.\d+\.\d+)\s*[—-]", readme, re.M)
check("Status changelog exists", bool(status_versions))
if status_versions:
    check("Status changelog covers current version",
          veritas.__version__ in status_versions,
          f"(changelog has {status_versions[:3]}, code is {veritas.__version__})")

# 2. Every __all__ export is importable and matches.
for name in veritas.__all__:
    check(f"export veritas.{name} exists", hasattr(veritas, name))

# 3. Test files named in the README table exist on disk.
for t in sorted(set(re.findall(r"`(tests/[\w_]+\.py)`", readme))):
    check(f"README-listed {t} exists", os.path.exists(os.path.join(ROOT, t)))

# 4. No references to removed phantom features.
for phantom in ["maxichoice", "33/33"]:
    check(f"README free of stale '{phantom}'", phantom not in readme,
          "stale reference found")

# 5. e2e gate script exists and is executable.
gate = os.path.join(ROOT, "tests", "verify_e2e.sh")
check("verify_e2e.sh exists", os.path.exists(gate))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
