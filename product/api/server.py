"""Veritas Product API — REST service for the Deterministic Governance
& Consistency Layer for AI Memory.

Zero dependencies beyond Python stdlib + veritas.

Endpoints:
  POST /api/beliefs        — assert a belief {proposition, source, reliability}
  GET  /api/beliefs        — list active beliefs
  GET  /api/beliefs/{id}   — belief detail with history
  POST /api/beliefs/{id}/retract — retract a belief
  GET  /api/audit          — audit log (all events)
  POST /api/query          — "what do you believe about X?" {question}
  POST /api/ingest         — ingest text, auto-extract claims {text}
  GET  /api/stats          — store statistics
  GET  /health             — health check

  GET  /                   — web playground UI

Run: python3 product/api/server.py [--port 8000]
"""

from __future__ import annotations

import json
import sys
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from veritas import BeliefStore, MemoryGovernor

# Single shared store (in-memory). For persistence, use ?persist=path.
STORE = BeliefStore()
GOVERNOR = MemoryGovernor(STORE)
START_TIME = time.time()

UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")


class Handler(BaseHTTPRequestHandler):
    server_version = "VeritasAPI/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def _send(self, code: int, obj, content_type="application/json"):
        body = json.dumps(obj, indent=1).encode() if isinstance(obj, (dict, list)) else obj
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Serve UI.
        if path == "/" or path == "/index.html":
            return self._serve_ui("index.html")
        if path.startswith("/ui/"):
            return self._serve_ui(path[4:])

        # API.
        if path == "/health":
            return self._send(200, {"status": "ok",
                                    "uptime_s": round(time.time() - START_TIME, 1),
                                    "version": "0.16.0"})
        if path == "/api/beliefs":
            beliefs = [
                {"id": b.id, "proposition": b.proposition,
                 "is_active": b.is_active,
                 "entrenchment": round(STORE.entrenchment_of(b), 3),
                 "source": b.source, "timestamp": b.timestamp}
                for b in STORE._beliefs.values() if b.is_active
            ]
            return self._send(200, {"beliefs": beliefs, "count": len(beliefs)})
        if path.startswith("/api/beliefs/"):
            bid = path.split("/")[3]
            b = STORE._beliefs.get(bid)
            if not b:
                return self._send(404, {"error": "not found"})
            # History from audit log.
            history = [e for e in STORE._audit._events
                       if e.get("belief_id") == bid]
            return self._send(200, {
                "id": b.id, "proposition": b.proposition,
                "is_active": b.is_active,
                "entrenchment": round(STORE.entrenchment_of(b), 3),
                "source": b.source, "timestamp": b.timestamp,
                "justifications": [list(js) for js in b.justifications],
                "history": history,
            })
        if path == "/api/audit":
            qs = parse_qs(parsed.query)
            limit = int(qs.get("limit", ["100"])[0])
            events = STORE._audit._events[-limit:]
            return self._send(200, {"events": events, "count": len(events)})
        if path == "/api/stats":
            active = sum(1 for b in STORE._beliefs.values() if b.is_active)
            retracted = sum(1 for b in STORE._beliefs.values() if not b.is_active)
            return self._send(200, {
                "active_beliefs": active,
                "retracted_beliefs": retracted,
                "total_beliefs": len(STORE._beliefs),
                "audit_events": len(STORE._audit._events),
                "uptime_s": round(time.time() - START_TIME, 1),
            })
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self._read_json()
        except Exception:
            return self._send(400, {"error": "invalid JSON"})

        if path == "/api/beliefs":
            prop = data.get("proposition", "").strip()
            if not prop:
                return self._send(400, {"error": "proposition required"})
            b = STORE.assert_belief(
                prop,
                source=data.get("source", "api"),
                source_reliability=float(data.get("reliability", 0.7)),
                confidence=float(data.get("confidence", 0.7)),
                ground=bool(data.get("ground", True)),
            )
            if b is None:
                return self._send(200, {"status": "rejected",
                                        "reason": "contradicted by stronger belief"})
            return self._send(201, {
                "id": b.id, "proposition": b.proposition,
                "is_active": b.is_active,
                "entrenchment": round(STORE.entrenchment_of(b), 3),
            })

        if path.startswith("/api/beliefs/") and path.endswith("/retract"):
            bid = path.split("/")[3]
            ok = STORE.retract_belief(bid)
            return self._send(200, {"retracted": bool(ok)})

        if path == "/api/ingest":
            text = data.get("text", "").strip()
            if not text:
                return self._send(400, {"error": "text required"})
            before = len([b for b in STORE._beliefs.values() if b.is_active])
            GOVERNOR.ingest_text(
                text,
                source=data.get("source", "api-ingest"),
                source_reliability=float(data.get("reliability", 0.7)),
                ground=True,
            )
            after = len([b for b in STORE._beliefs.values() if b.is_active])
            return self._send(200, {
                "ingested": True,
                "new_active_beliefs": after - before,
                "total_active": after,
            })

        if path == "/api/query":
            question = data.get("question", "").strip()
            if not question:
                return self._send(400, {"error": "question required"})
            # Simple: find active belief with highest word overlap.
            qwords = set(question.lower().split())
            best, best_score = None, 0
            for b in STORE._beliefs.values():
                if not b.is_active:
                    continue
                score = len(qwords & set(b.proposition.lower().split()))
                if score > best_score:
                    best, best_score = b, score
            if not best:
                return self._send(200, {"answer": None,
                                        "reason": "no matching belief"})
            return self._send(200, {
                "answer": best.proposition,
                "belief_id": best.id,
                "entrenchment": round(STORE.entrenchment_of(best), 3),
            })

        return self._send(404, {"error": "not found"})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_ui(self, filename: str):
        # Prevent directory traversal.
        safe = os.path.basename(filename)
        fpath = os.path.join(UI_DIR, safe)
        if not os.path.isfile(fpath):
            return self._send(404, {"error": "UI not found"})
        ctype = "text/html" if safe.endswith(".html") else \
                "application/javascript" if safe.endswith(".js") else \
                "text/css" if safe.endswith(".css") else \
                "application/octet-stream"
        with open(fpath, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = HTTPServer((args.host, args.port), Handler)
    print(f"Veritas API on http://{args.host}:{args.port}  "
          f"(UI at /)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
