"""Tests for the DashScope text-embedding-v4 embedder (mocked HTTP, no key).

Covers: missing key -> load() None; interface parity with the local
DenseEmbedder (embed shape, L2 normalization, empty input); batching;
graceful failure on API errors.
"""

import io
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aml"))

import numpy as np
from dense_dashscope import DashScopeEmbedder, MODEL, ENDPOINT


DIMS = 8


def _resp(texts, dims=DIMS):
    data = [
        {"index": i, "embedding": [float(i + 1)] * dims} for i in range(len(texts))
    ]
    return json.dumps({"data": data}).encode()


class _FakeHTTP:
    """urlopen replacement: records requests, replays canned embeddings."""

    def __init__(self, dims=DIMS, fail=False):
        self.calls = []
        self.dims = dims
        self.fail = fail

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode())
        self.calls.append(body)
        if self.fail:
            raise ConnectionError("boom")
        resp = _resp(body["input"], self.dims)

        class Ctx:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

            def read(self_):
                return resp

        return Ctx()


class TestDashScopeEmbedder(unittest.TestCase):
    def test_no_key_returns_none(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("DASHSCOPE_API_KEY", "VERITAS_DASHSCOPE_KEY")}
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(DashScopeEmbedder.load())

    def test_alt_key_name_accepted(self):
        fake = _FakeHTTP()
        env = {"VERITAS_DASHSCOPE_KEY": "k"}
        with patch.dict(os.environ, env, clear=True):
            with patch("urllib.request.urlopen", fake):
                emb = DashScopeEmbedder.load()
        self.assertIsNotNone(emb)
        self.assertEqual(emb.dims, DIMS)

    def test_embed_shape_and_normalization(self):
        fake = _FakeHTTP()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "k"}, clear=True):
            with patch("urllib.request.urlopen", fake):
                emb = DashScopeEmbedder("k")
                vecs = emb.embed(["hello", "world"])
        self.assertEqual(vecs.shape, (2, DIMS))
        norms = np.linalg.norm(vecs, axis=1)
        self.assertTrue(np.allclose(norms, 1.0))
        # endpoint + model + float encoding asserted on the wire;
        # 2 calls: constructor probe + the embed itself
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["model"], MODEL)

    def test_empty_input(self):
        fake = _FakeHTTP()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "k"}, clear=True):
            with patch("urllib.request.urlopen", fake):
                emb = DashScopeEmbedder("k")
        out = emb.embed([])
        self.assertEqual(out.shape, (0, DIMS))
        # no extra HTTP call for empty input (probe only)
        self.assertEqual(len(fake.calls), 1)

    def test_batching(self):
        fake = _FakeHTTP()
        env = {"DASHSCOPE_API_KEY": "k", "VERITAS_DASHSCOPE_BATCH": "40"}
        with patch.dict(os.environ, env, clear=True):
            with patch("urllib.request.urlopen", fake):
                emb = DashScopeEmbedder("k")
                vecs = emb.embed([f"t{i}" for i in range(100)])
        self.assertEqual(vecs.shape, (100, DIMS))
        # 1 probe + ceil(100/40) = 4 calls total
        self.assertEqual(len(fake.calls), 4)
        self.assertEqual(sum(len(c["input"]) for c in fake.calls[1:]), 100)

    def test_api_failure_load_returns_none(self):
        fake = _FakeHTTP(fail=True)
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "k"}, clear=True):
            with patch("urllib.request.urlopen", fake):
                with patch("sys.stdout", io.StringIO()):
                    self.assertIsNone(DashScopeEmbedder.load())


if __name__ == "__main__":
    unittest.main()
