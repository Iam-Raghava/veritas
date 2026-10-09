"""Confidence calibration tracker for Veritas.

A belief's confidence should predict its survival: high-confidence
beliefs should rarely be retracted, low-confidence ones often.
This module tracks calibration over time, enabling:
- Detecting overconfident sources
- Adjusting source reliability automatically
- Reporting calibration curves

Well-calibrated confidence is essential for production ML systems.
"""

from __future__ import annotations

from collections import defaultdict


class CalibrationTracker:
    """Track confidence vs actual survival rates."""

    def __init__(self, bins: int = 10) -> None:
        self.bins = bins
        # bin -> [asserted, survived]
        self._stats: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        # source -> [asserted, survived]
        self._by_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def record_assertion(
        self, confidence: float, source: str, belief_id: str
    ) -> None:
        """Record a new belief assertion."""
        b = min(int(confidence * self.bins), self.bins - 1)
        self._stats[b][0] += 1
        self._by_source[source][0] += 1

    def record_survival(self, confidence: float, source: str) -> None:
        """Record that a belief survived (was not retracted)."""
        b = min(int(confidence * self.bins), self.bins - 1)
        self._stats[b][1] += 1
        self._by_source[source][1] += 1

    def calibration_curve(self) -> list[tuple[float, float]]:
        """Return (confidence_bin_center, survival_rate) pairs."""
        out = []
        for b in range(self.bins):
            asserted, survived = self._stats[b]
            if asserted > 0:
                center = (b + 0.5) / self.bins
                out.append((center, survived / asserted))
        return out

    def is_calibrated(self, tolerance: float = 0.2) -> bool:
        """Check if confidence predicts survival within tolerance."""
        for conf, rate in self.calibration_curve():
            if abs(conf - rate) > tolerance:
                return False
        return True

    def source_reliability(self) -> dict[str, float]:
        """Empirical survival rate by source."""
        return {
            s: survived / asserted if asserted else 0.0
            for s, (asserted, survived) in self._by_source.items()
        }
