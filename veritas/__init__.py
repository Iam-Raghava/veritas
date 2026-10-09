"""Veritas: truth maintenance for AI agents.

The missing piece in agent memory: when new evidence contradicts old
beliefs, *automatically* decide what to retract using a principled policy
(epistemic entrenchment), propagate the retraction through belief
dependencies, and audit everything.
"""
from .audit import AuditEvent, AuditLog
from .belief import Belief
from .detect import (
    ContradictionFn,
    Detector,
    HeuristicDetector,
    as_function,
    functional_parts,
    heuristic_contradiction,
    subject_object_parts,
)
from .detect_nli import HuggingFaceNLIDetector
from .detect_semantic import SemanticDetector
from .entrenchment import (
    corroboration_score,
    entrenchment,
    independent_entrenchment,
    order_by_entrenchment,
    recency_score,
)
from .persist import load, register_detector, save
from .store import BeliefStore
from .async_store import AsyncBeliefStore
from .observability import Metrics

__version__ = "0.7.0"

__all__ = [
    "Belief",
    "BeliefStore",
    "AsyncBeliefStore",
    "Metrics",
    "AuditEvent",
    "AuditLog",
    "ContradictionFn",
    "Detector",
    "HeuristicDetector",
    "HuggingFaceNLIDetector",
    "SemanticDetector",
    "as_function",
    "functional_parts",
    "subject_object_parts",
    "heuristic_contradiction",
    "entrenchment",
    "independent_entrenchment",
    "corroboration_score",
    "recency_score",
    "order_by_entrenchment",
    "save",
    "load",
    "register_detector",
]
