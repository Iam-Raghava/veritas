"""Tests for the AML Add/Search wrapper: governance behavior end to end."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Isolate the DB per test run.
_tmp = tempfile.mkdtemp()
os.environ["VERITAS_DB"] = os.path.join(_tmp, "test_aml.db")

from fastapi.testclient import TestClient  # noqa: E402

from aml.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_add_and_governed_search():
    # Grounded fact first.
    r = client.post("/add", json={"items": [
        {"text": "The CEO is Alice.", "source": "board_minutes",
         "source_reliability": 0.95, "timestamp": 1000.0},
    ]})
    assert r.json()["added"] == 1

    # Weak rumor contradicts it -> rejected, Alice stays.
    r = client.post("/add", json={"items": [
        {"text": "The CEO is Bob.", "source": "rumor",
         "source_reliability": 0.2, "timestamp": 2000.0},
    ]})
    body = r.json()
    assert body["rejected"] == 1, body

    # Search returns Alice only; Bob never surfaces.
    r = client.post("/search", json={"query": "who is the CEO?", "top_k": 5})
    ev = r.json()["evidence"]
    assert len(ev) == 1, ev
    assert "Alice" in ev[0]["text"]

    # Strong update from the same source supersedes Alice -> Carol retracts her.
    r = client.post("/add", json={"items": [
        {"text": "The CEO is Carol.", "source": "board_minutes",
         "source_reliability": 0.95, "timestamp": 3000.0},
    ]})
    body = r.json()
    assert body["superseded"] == 1, body
    assert body["added"] == 1, body

    r = client.post("/search", json={"query": "who is the CEO?", "top_k": 5})
    ev = r.json()["evidence"]
    assert len(ev) == 1, ev
    assert "Carol" in ev[0]["text"], ev
    print("governance: stale Alice retracted, current Carol served")


def test_temporal_as_of():
    client.post("/add", json={"items": [
        {"text": "Office is in Austin.", "source": "hr",
         "source_reliability": 0.9, "timestamp": 100.0},
    ]})
    # Query scoped before the belief's valid_from -> no evidence.
    r = client.post("/search", json={
        "query": "where is the office?", "top_k": 5, "as_of": 50.0})
    assert r.json()["count"] == 0
    # Scoped after -> evidence present.
    r = client.post("/search", json={
        "query": "where is the office?", "top_k": 5, "as_of": 150.0})
    assert r.json()["count"] >= 1
    print("temporal: as_of scoping works")


if __name__ == "__main__":
    test_health()
    test_add_and_governed_search()
    test_temporal_as_of()
    print("ALL AML SERVICE TESTS PASSED")
