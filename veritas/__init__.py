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
from .detect_hybrid import HybridDetector
from .timetravel import TimeTravel
from .distributed import DistributedBeliefStore, SQLiteEventLog, EventLog
from .distributed_gossip import GossipEventLog
from .export import to_json, to_dot, to_graphml

# BFT is optional (requires cryptography). Lazy via __getattr__ (PEP 562)
# so the import doesn't fail when cryptography is missing.
_BFT_AVAILABLE = False
try:
    import veritas.distributed_bft as _bft_module
    _BFT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _bft_module = None  # type: ignore[assignment]


def __getattr__(name: str):  # type: ignore[no-redef]
    if name == "ByzantineTolerantLog" and _bft_module is not None:
        return _bft_module.ByzantineTolerantLog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from .entrenchment import (
    corroboration_score,
    entrenchment,
    independent_entrenchment,
    order_by_entrenchment,
    recency_score,
)
from .persist import load, register_detector, save
from .persist_backend import (
    PersistenceBackend,
    SQLiteBackend,
    get_backend,
)
from .extractor import Claim, ClaimExtractor, HeuristicClaimExtractor
from .extractor import (
    ExtractedClaim,
    HeuristicExtractor,
    StructuredLLMExtractor,
)
from .detect_vector import VectorPrefilterDetector
from .store import BeliefStore
from .async_store import AsyncBeliefStore
from .observability import Metrics
from .explain import Explainer
from .calibration import CalibrationTracker
from .policies import Policy, describe_policy

__version__ = "0.14.1"

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
    "HybridDetector",
    "SemanticDetector",
    "LLMJudgeDetector",
    "TimeTravel",
    "DistributedBeliefStore",
    "SQLiteEventLog",
    "EventLog",
    "GossipEventLog",
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
    "Claim",
    "ClaimExtractor",
    "HeuristicClaimExtractor",
    "ExtractedClaim",
    "HeuristicExtractor",
    "StructuredLLMExtractor",
    "VectorPrefilterDetector",
    "PersistenceBackend",
    "SQLiteBackend",
    "get_backend",
]

if _BFT_AVAILABLE:
    __all__.append("ByzantineTolerantLog")
