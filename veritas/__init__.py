"""Veritas: truth maintenance for AI agents.

The missing piece in agent memory: when new evidence contradicts old
beliefs, *automatically* decide what to retract using a principled policy
(epistemic entrenchment), propagate the retraction through belief
dependencies, and audit everything.
"""
from .audit import AuditEvent, AuditLog, CONSTRAINT_ERROR
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
from .detect_llm import LLMJudgeDetector
from .timetravel import TimeTravel
from .export import to_json, to_dot, to_graphml
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
from .explain import Explainer
from .calibration import CalibrationTracker
from .policies import Policy, describe_policy

__version__ = "0.11.0"

__all__ = [
    "Belief",
    "BeliefStore",
    "AsyncBeliefStore",
    "Metrics",
    "Explainer",
    "CalibrationTracker",
    "Policy",
    "describe_policy",
    "AuditEvent",
    "AuditLog",
    "CONSTRAINT_ERROR",
    "ContradictionFn",
    "Detector",
    "HeuristicDetector",
    "HuggingFaceNLIDetector",
    "SemanticDetector",
    "LLMJudgeDetector",
    "TimeTravel",
    "to_json",
    "to_dot",
    "to_graphml",
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
