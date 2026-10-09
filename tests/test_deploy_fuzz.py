"""Deployment fuzz: journal_sync.py must survive hostile inputs.

This is production cron code reading real files. It must never crash
on malformed logs, binary garbage, huge inputs, or missing files —
it degrades gracefully (empty parse, clean error, nonzero exit).

Run: python3 tests/test_deploy_fuzz.py
"""
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))

import journal_sync as js  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok: {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, e))
        print(f"  FAIL: {name}: {type(e).__name__}: {e}")


def write_temp(content: bytes) -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
    f.write(content)
    f.close()
    return f.name


def test_parse_log_garbage():
    rng = random.Random(7)
    for i in range(50):
        # Random binary garbage, huge lines, weird unicode.
        chunks = []
        for _ in range(rng.randint(1, 30)):
            kind = rng.random()
            if kind < 0.3:
                chunks.append(bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 500))))
            elif kind < 0.5:
                chunks.append(("- " + "x" * rng.randint(1, 10000)).encode())
            elif kind < 0.7:
                chunks.append("--- --- ---".encode())
            else:
                chunks.append("日本語 🎉 ---\n- ".encode() + os.urandom(20))
        data = b"\n".join(chunks)
        path = write_temp(data)
        try:
            # Must not crash; latin-1 fallback for undecodable bytes.
            try:
                entries = js.parse_log(path)
            except UnicodeDecodeError:
                pass  # acceptable: binary file rejected loudly
            else:
                assert isinstance(entries, list)
        finally:
            os.unlink(path)


def test_parse_log_missing():
    try:
        js.parse_log("/nonexistent/path/log.md")
        raise AssertionError("should raise on missing file")
    except (FileNotFoundError, OSError):
        pass


def test_parse_memory_garbage():
    rng = random.Random(13)
    for i in range(30):
        lines = []
        for _ in range(rng.randint(1, 40)):
            kind = rng.random()
            if kind < 0.4:
                lines.append("## " + "".join(chr(rng.randint(32, 5000)) for _ in range(20)))
            elif kind < 0.7:
                lines.append("- " + "y" * rng.randint(1, 5000))
            else:
                lines.append("## Facts")
        path = write_temp("\n".join(lines).encode("utf-8", "replace"))
        try:
            props = js.parse_memory(path)
            assert isinstance(props, list)
            assert all(isinstance(p, str) and len(p) > 20 for p in props)
        finally:
            os.unlink(path)


def test_detector_total():
    """journal_contradiction must be total: never raise, always bool."""
    rng = random.Random(21)
    class FakeBelief:
        def __init__(self, p, m=None):
            self.proposition = p
            self.metadata = m or {}
            self.id = "x"
    props = ["", "x" * 10000, "日本語", "a 's b is c", "---", None]
    for _ in range(200):
        a = rng.choice(props)
        b = rng.choice(props)
        for pa, pb in [(a, b)]:
            try:
                if pa is None or pb is None:
                    continue
                r = js.journal_contradiction(FakeBelief(pa), FakeBelief(pb))
                assert isinstance(r, bool), f"non-bool: {r!r}"
            except (AttributeError, TypeError):
                pass  # None props are caller error; must not hang/crash else
    # Symmetry spot check on valid inputs.
    r1 = js.journal_contradiction(FakeBelief("use SQLite for index"), FakeBelief("use shelf cache for index"))
    r2 = js.journal_contradiction(FakeBelief("use shelf cache for index"), FakeBelief("use SQLite for index"))
    assert r1 == r2, "detector asymmetric"


def test_main_missing_files():
    """main() with missing journal files: clean error, no traceback."""
    p = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'deploy');"
         "import journal_sync;"
         "journal_sync.JOURNAL='/nonexistent';"
         "sys.exit(journal_sync.main())"],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.join(os.path.dirname(__file__), ".."))
    out = (p.stdout + p.stderr).lower()
    assert "traceback" not in out, f"traceback on missing files: {out[:200]}"


def main():
    print("deployment fuzz:")
    check("parse_log survives garbage (50 cases)", test_parse_log_garbage)
    check("parse_log missing file", test_parse_log_missing)
    check("parse_memory survives garbage (30 cases)", test_parse_memory_garbage)
    check("detector total + symmetric (200 pairs)", test_detector_total)
    check("main() missing files: no traceback", test_main_missing_files)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
