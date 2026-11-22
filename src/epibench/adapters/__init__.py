"""Built-in memory adapters and reference bounds for epibench."""

from epibench.adapters.base import (
    ADAPTER_CONTRACT_VERSION,
    BaseMemoryAdapter,
    IngestResult,
    RetrievalResult,
    count_tokens,
    filter_sessions_as_of,
    format_session_text,
    is_on_or_before_as_of,
    parse_as_of_cutoff,
)
from epibench.adapters.full_context_adapter import (
    FullContextAdapter,
    FullContextUnboundedAdapter,
    RecentContextAdapter,
)
from epibench.adapters.hybrid_rag_adapter import HybridRAGAdapter
from epibench.adapters.mem0_adapter import Mem0Adapter
from epibench.adapters.reference_adapters import NullMemoryAdapter, OracleAdapter
from epibench.adapters.vector_rag_adapter import VectorRAGAdapter

__all__ = [
    "ADAPTER_CONTRACT_VERSION",
    "BaseMemoryAdapter",
    "FullContextAdapter",
    "FullContextUnboundedAdapter",
    "HybridRAGAdapter",
    "IngestResult",
    "Mem0Adapter",
    "NullMemoryAdapter",
    "OracleAdapter",
    "RecentContextAdapter",
    "RetrievalResult",
    "VectorRAGAdapter",
    "count_tokens",
    "filter_sessions_as_of",
    "format_session_text",
    "is_on_or_before_as_of",
    "parse_as_of_cutoff",
]
