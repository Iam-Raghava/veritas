"""README doctest: every Python code block in README.md must run.

Documentation that doesn't run is documentation that lies.
Run: python3 tests/test_readme.py
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _test_file(filename: str) -> int:
    """Test all Python code blocks in a markdown file. Returns failure count."""
    path = os.path.join(ROOT, filename)
    if not os.path.exists(path):
        print(f"  skip: {filename} not found")
        return 0
    text = open(path).read()
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    if not blocks:
        print(f"  skip: no python blocks in {filename}")
        return 0
    failed = 0
    for i, b in enumerate(blocks):
        code = "import sys; sys.path.insert(0, '.')\n" + b
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            path = f.name
        try:
            p = subprocess.run(
                [sys.executable, path], capture_output=True, text=True,
                timeout=30, cwd=ROOT)
            if p.returncode == 0:
                print(f"  ok: README block {i}")
            else:
                failed += 1
                err = (p.stderr.strip().splitlines() or [""])[-1][:120]
                print(f"  FAIL: README block {i}: {err}")
        finally:
            os.unlink(path)
            # Blocks may create scratch DBs; don't leak them.
            for scratch in ("memory.db",):
                sp = os.path.join(ROOT, scratch)
                if os.path.exists(sp):
                    os.unlink(sp)
    print(f"  {len(blocks) - failed}/{len(blocks)} blocks in {filename} run")
    return failed


def main():
    total_failed = 0
    # Test README.md, ARTICLE.md, and ARTICLE_FRONTIER.md.
    # Documentation that doesn't run is documentation that lies.
    for filename in ["README.md", "ARTICLE.md", "ARTICLE_FRONTIER.md"]:
        total_failed += _test_file(filename)
    if total_failed:
        print(f"TOTAL: {total_failed} failed blocks")
        sys.exit(1)
    print("All documentation code blocks pass.")


if __name__ == "__main__":
    main()
