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

import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Metrics:
    """Thread-safe-ish metrics collector (use with store's lock)."""

    assertions: int = 0
    rejections: int = 0
    retractions: int = 0
    cascades: int = 0
    cascade_size_total: int = 0
    contradictions_found: int = 0
    constraint_violations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    # Operation latencies (seconds): name -> list of samples.
    latencies: dict[str, list[float]] = field(default_factory=dict)
    # Event counters by type.
    events: Counter = field(default_factory=Counter)

    def record(self, event: str, latency: float | None = None) -> None:
        self.events[event] += 1
        if latency is not None:
            self.latencies.setdefault(event, []).append(latency)

    def summary(self) -> dict:
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
                out[f"latency_{name}_p50"] = sorted(samples)[len(samples) // 2]
                out[f"latency_{name}_p99"] = sorted(samples)[int(len(samples) * 0.99)]
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
