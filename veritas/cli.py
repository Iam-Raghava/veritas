"""Veritas command-line interface.

Every command loads the store from SQLite, acts, and saves it back —
the CLI is a thin shell over BeliefStore, so everything it does is
audited exactly like programmatic use.

    veritas init
    veritas assert "Acme's CEO is Jane Smith" --source "tech blog" --reliability 0.65
    veritas list
    veritas assert "Acme's CEO is John Doe" --source "press release" --reliability 0.97
    veritas contradictions
    veritas audit
    veritas stats
"""
from __future__ import annotations

import argparse
import os
import sys

from . import load, save
from .store import BeliefStore

DEFAULT_DB = os.environ.get("VERITAS_DB", os.path.join(os.getcwd(), "veritas.db"))


def _open(db: str) -> BeliefStore:
    if not os.path.exists(db):
        raise SystemExit(
            f"no store at {db}. Run `veritas init` first "
            f"(or set VERITAS_DB)."
        )
    return load(db)


def cmd_init(args: argparse.Namespace) -> int:
    if os.path.exists(args.db) and not args.force:
        print(f"store already exists at {args.db} (use --force to overwrite)")
        return 1
    save(BeliefStore(), args.db)
    print(f"initialized empty Veritas store at {args.db}")
    return 0


def cmd_assert(args: argparse.Namespace) -> int:
    store = _open(args.db)
    just = [j.strip() for j in (args.justify or "").split(",") if j.strip()]
    belief = store.assert_belief(
        args.proposition,
        confidence=args.confidence,
        source=args.source,
        source_reliability=args.reliability,
        justifications=just,
        ground=args.ground,
    )
    save(store, args.db, detector_name=store.detector_name)
    if belief is None:
        print("REJECTED: a more entrenched contradictor holds (see `veritas audit`)")
        return 2
    print(f"asserted [{belief.id}] entrenchment={store.entrenchment_of(belief):.3f}")
    retracted = [e for e in store.audit if e.event_type in ("retracted", "cascade_retracted")]
    if retracted:
        print(f"contracted {len(retracted)} belief(s):")
        for e in retracted[-5:]:
            print(f"  - {e.proposition} ({e.reason[:80]})")
    return 0


def cmd_retract(args: argparse.Namespace) -> int:
    store = _open(args.db)
    retracted = store.retract_belief(args.id, reason=args.reason)
    save(store, args.db, detector_name=store.detector_name)
    if not retracted:
        print(f"no active belief {args.id}")
        return 1
    print(f"retracted {len(retracted)} belief(s):")
    for b in retracted:
        print(f"  - [{b.id}] {b.proposition}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    store = _open(args.db)
    beliefs = store.all_beliefs() if args.all else store.active_beliefs()
    for b in beliefs:
        mark = " " if b.is_active else "x"
        print(f"[{mark}] [{b.id}] (e={store.entrenchment_of(b):.3f}) {b.proposition}")
        print(f"      src={b.source} rel={b.source_reliability} "
              f"conf={b.confidence} just={b.justifications or '-'}")
    print(f"{len(beliefs)} belief(s), {len(store.active_beliefs())} active")
    return 0


def cmd_contradictions(args: argparse.Namespace) -> int:
    store = _open(args.db)
    pairs = store.find_contradictions()
    if not pairs:
        print("consistent: no contradictions among active beliefs")
        return 0
    print(f"{len(pairs)} contradiction(s):")
    for a, b in pairs:
        print(f"  [{a.id}] {a.proposition}")
        print(f"  [{b.id}] {b.proposition}")
        print()
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    store = _open(args.db)
    events = store.audit.events_for(args.belief) if args.belief else list(store.audit)
    for e in events:
        print(f"[{e.seq:03d}] {e.event_type:16s} [{e.belief_id}] {e.proposition}")
        print(f"       {e.reason}")
    print(f"{len(events)} event(s)")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    store = _open(args.db)
    for k, v in store.stats().items():
        print(f"{k:14s} {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="veritas", description="Truth maintenance for AI agents")
    p.add_argument("--db", default=DEFAULT_DB, help="store file (default: ./veritas.db or $VERITAS_DB)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a new empty store")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("assert", help="assert a belief (contracts contradictors)")
    s.add_argument("proposition")
    s.add_argument("--source", default="cli")
    s.add_argument("--reliability", type=float, default=0.5)
    s.add_argument("--confidence", type=float, default=0.8)
    s.add_argument("--justify", default="", help="comma-separated belief IDs")
    s.add_argument("--ground", action="store_true",
                   help="mark as direct observation (survives orphaning)")
    s.set_defaults(fn=cmd_assert)

    s = sub.add_parser("retract", help="retract a belief (propagates)")
    s.add_argument("id")
    s.add_argument("--reason", default="manual retraction via CLI")
    s.set_defaults(fn=cmd_retract)

    s = sub.add_parser("list", help="list beliefs")
    s.add_argument("--all", action="store_true", help="include retracted")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("contradictions", help="show contradicting pairs")
    s.set_defaults(fn=cmd_contradictions)

    s = sub.add_parser("audit", help="show audit log")
    s.add_argument("--belief", default=None, help="filter to one belief ID")
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("stats", help="store statistics")
    s.set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
