"""Live deployment: Veritas truth-maintains the agent journal itself.

Reads the journal's decision log + MEMORY.md, asserts every decision and
fact as a belief in a persistent SQLite store, and reports contradictions.
RESOLVED decisions are treated as new evidence: the losing side is linked
as an explicit contradiction so the principled path (entrenchment
comparison, cascade, audit) does the retraction — never silent deletion.

Run:  python3 deploy/journal_sync.py [--report]
The daily snapshot runs this automatically; it stays silent when the
store is consistent and writes deploy/report.md when it isn't.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore, heuristic_contradiction, save  # noqa: E402
from veritas import register_detector  # noqa: E402

JOURNAL = os.path.expanduser("~/workspace/agent-journal")
LOG = os.path.join(JOURNAL, "decisions", "log.md")
MEMORY = os.path.expanduser("~/MEMORY.md")
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DEPLOY_DIR, "journal.db")
REPORT = os.path.join(DEPLOY_DIR, "report.md")

# ---------------------------------------------------------------------------
# Journal-domain contradiction detector.
# ---------------------------------------------------------------------------

_ACTION = re.compile(
    r"^\s*(?:-\s*)?(?P<verb>use|avoid|build|do not build|don't build|"
    r"do not cache|don't cache|do not use|don't use|add|implement|keep|"
    r"declined)\b\s+(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)

_STOPWORDS = {
    "the", "a", "an", "for", "of", "in", "on", "to", "and", "or", "with",
    "it", "its", "is", "as", "by", "at", "from", "that", "this", "over",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().rstrip("."))


def _keywords(s: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", _norm(s))
        if len(w) > 3 and w not in _STOPWORDS
    }


def _core_object(obj: str) -> str:
    """The claim before any justification clause."""
    return re.split(r"[:;]", obj, maxsplit=1)[0]


def _same_object(a_obj: str, b_obj: str) -> bool:
    ka, kb = _keywords(_core_object(a_obj)), _keywords(_core_object(b_obj))
    if not ka or not kb:
        return False
    return len(ka & kb) / len(ka | kb) > 0.4


def _polarity(verb: str) -> bool:
    v = verb.lower()
    return not (
        v.startswith("do not") or v.startswith("don't") or v in
        {"avoid", "declined", "remove", "drop", "reject"}
    )


def _action_parts(prop: str) -> tuple[str, str] | None:
    m = _ACTION.match(prop)
    if not m:
        return None
    return _norm(m.group("verb")), m.group("obj")


def journal_contradiction(a, b) -> bool:
    """heuristic + 'Use X' vs 'Avoid X' action opposition."""
    if heuristic_contradiction(a, b):
        return True
    pa, pb = _action_parts(a.proposition), _action_parts(b.proposition)
    if pa and pb and _same_object(pa[1], pb[1]):
        if _polarity(pa[0]) != _polarity(pb[0]):
            return True
    return False


register_detector("journal", journal_contradiction)

# ---------------------------------------------------------------------------
# Parsing.
# ---------------------------------------------------------------------------

_RESOLVED = re.compile(
    r"RESOLVED\s*\((?P<topic>[^)]+)\):\s*chose\s+(?P<winner>.+?)\s+over\s+"
    r"(?P<loser>.+?)(?:\s+because|\s*$)",
    re.IGNORECASE,
)


def parse_log(path: str) -> list[dict]:
    entries: list[dict] = []
    stamp = "unknown"
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^---\s*(\S+)\s*---\s*$", line)
            if m:
                stamp = m.group(1)
                continue
            s = line.strip()
            if s.startswith("- ") and len(s) > 4:
                text = s[2:].strip()
                rm = _RESOLVED.search(text)
                entries.append({
                    "timestamp": stamp,
                    "text": text,
                    "resolved": rm.groupdict() if rm else None,
                })
    return entries


def parse_memory(path: str) -> list[str]:
    props: list[str] = []
    section = ""
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("## "):
                section = line.strip()
            elif line.strip().startswith("- ") and section in (
                "## Facts", "## Preferences", "## Boundaries", "## Commitments"
            ):
                prop = line.strip()[2:].strip()
                prop = re.sub(r"\s*\([^()]*\d{4}[^()]*\)\.?\s*$", "", prop).strip()
                if len(prop) > 20:
                    props.append(prop)
    return props


def _loser_match(belief_text: str, topic: str, loser: str) -> bool:
    """Does this belief assert the losing side of a RESOLVED decision?"""
    parts = _action_parts(belief_text)
    if not parts or not _polarity(parts[0]):
        return False  # only positive actions can assert the loser
    obj_kw = _keywords(_core_object(parts[1]))
    topic_kw = _keywords(topic)
    loser_kw = _keywords(loser)
    # Must be about the topic AND the losing option.
    return len(obj_kw & topic_kw) >= 1 and len(obj_kw & loser_kw) >= 1


# ---------------------------------------------------------------------------
# Sync.
# ---------------------------------------------------------------------------

def main() -> int:
    report_mode = "--report" in sys.argv
    store = BeliefStore(contradiction_fn=journal_contradiction)
    store.detector_name = "journal"

    entries = parse_log(LOG)
    normal = [e for e in entries if not e["resolved"]]
    resolved = [e for e in entries if e["resolved"]]

    # Pass 1: ordinary decisions + memory facts.
    asserted: dict[str, object] = {}  # text -> Belief
    for e in normal:
        b = store.assert_belief(
            e["text"], confidence=0.85,
            source=f"journal decision {e['timestamp']}",
            source_reliability=0.7,
            metadata={"timestamp": e["timestamp"]},
        )
        if b is not None:
            asserted[e["text"]] = b

    for prop in parse_memory(MEMORY):
        store.assert_belief(
            prop, confidence=0.85,
            source="MEMORY.md", source_reliability=0.9,
        )

    # Pass 2: RESOLVED decisions link the losing side explicitly, so the
    # principled contraction path (not manual deletion) does the work.
    for e in resolved:
        r = e["resolved"]
        assert r is not None
        loser_ids = [
            b.id for text, b in asserted.items()
            if b.is_active and _loser_match(text, r["topic"], r["loser"])
        ]
        store.assert_belief(
            e["text"], confidence=0.95,
            source=f"journal resolution {e['timestamp']}",
            source_reliability=0.95,
            metadata={"timestamp": e["timestamp"], "contradicts": loser_ids},
        )

    pairs = store.find_contradictions()
    stats = store.stats()
    save(store, DB, detector_name="journal")

    lines = [
        "# Veritas journal audit",
        "",
        f"Beliefs: {stats['active']} active, {stats['retracted']} retracted, "
        f"{stats['total']} total. Consistent: {stats['consistent']}.",
        "",
    ]
    if pairs:
        lines.append(f"## {len(pairs)} live contradiction(s) need resolution")
        lines.append("")
        for a, b in pairs:
            ea, eb = store.entrenchment_of(a), store.entrenchment_of(b)
            lines.append(f"- **CONFLICT** (e={ea:.2f}) {a.proposition[:100]}")
            lines.append(f"  vs (e={eb:.2f}) {b.proposition[:100]}")
            lines.append(
                "  → record a `RESOLVED (<topic>): chose X over Y because Z` "
                "decision to contract the losing side."
            )
    else:
        lines.append("No live contradictions. Memory is consistent.")
    lines.append("")
    lines.append("## Retractions this run")
    for ev in store.audit.of_type("retracted")[-8:]:
        lines.append(f"- {ev.proposition[:90]}")
        lines.append(f"  ↳ {ev.reason[:110]}")
    casc = store.audit.of_type("cascade_retracted")
    if casc:
        lines.append("")
        lines.append("## Cascades")
        for ev in casc[-5:]:
            lines.append(f"- {ev.proposition[:90]}")

    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")

    if pairs or report_mode:
        print("\n".join(lines))
    else:
        print(f"veritas: {stats['active']} active, {stats['retracted']} "
              f"retracted, consistent={stats['consistent']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
