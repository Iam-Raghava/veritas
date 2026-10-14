"""Observability for Veritas: metrics and operation tracing.

Production agent runtimes need visibility into belief-base health:
how often contradictions fire, retraction rates, cache hit rates,
and slow operations. This module provides a lightweight,
dependency-free metrics collector.

Usage:
    from veritas.observability import Metrics
    store = BeliefStore(metrics=Metrics())
    ...
    print(store.metrics.summary())
"""

from __future__ import annotations

import collections
import threading
import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Metrics:
    """Thread-safe metrics collector.

    All mutations go through the internal lock, so counters are safe
    even when called outside the store's lock (e.g., entrenchment_of
    is a public method that does not hold the store lock).
    """

    assertions: int = 0
    rejections: int = 0
    retractions: int = 0
    cascades: int = 0
    cascade_size_total: int = 0
    contradictions_found: int = 0
    constraint_violations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    # Operation latencies (seconds): name -> bounded deque of samples.
    # Bounded to prevent unbounded memory growth on long-running agents.
    latencies: dict[str, "collections.deque[float]"] = field(
        default_factory=dict)
    # Event counters by type.
    events: Counter = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock,
                                  repr=False, compare=False)

    def incr(self, name: str, amount: int = 1) -> None:
        """Thread-safe counter increment (e.g., incr("cache_hits"))."""
        with self._lock:
            setattr(self, name, getattr(self, name) + amount)

    def record(self, event: str, latency: float | None = None) -> None:
        with self._lock:
            self.events[event] += 1
            if latency is not None:
                import collections
                dq = self.latencies.get(event)
                if dq is None:
                    dq = collections.deque(maxlen=1000)
                    self.latencies[event] = dq
                dq.append(latency)

    def summary(self) -> dict:
        with self._lock:
            out = {
                "assertions": self.assertions,
                "rejections": self.rejections,
                "retractions": self.retractions,
                "cascades": self.cascades,
                "avg_cascade_size": (
                    self.cascade_size_total / self.cascades if self.cascades else 0
                ),
                "contradictions_found": self.contradictions_found,
                "constraint_violations": self.constraint_violations,
                "cache_hit_rate": (
                    self.cache_hits / (self.cache_hits + self.cache_misses)
                    if (self.cache_hits + self.cache_misses) else 0
                ),
            }
            for name, samples in self.latencies.items():
                if samples:
                    s = sorted(samples)
                    out[f"latency_{name}_p50"] = s[len(s) // 2]
                    out[f"latency_{name}_p99"] = s[int(len(s) * 0.99)]
            return out


class timed:
    """Context manager to time an operation and record it."""

    def __init__(self, metrics: Metrics | None, event: str) -> None:
        self.metrics = metrics
        self.event = event
        self.start = 0.0

    def __enter__(self) -> "timed":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if self.metrics is not None:
            self.metrics.record(self.event, time.perf_counter() - self.start)
