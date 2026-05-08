"""
agent-memory-lab — five memory patterns for LLM agents.

Each pattern is a single file (`patterns/<name>.py`) and exposes the
shared interface:

    add(msg: Message) -> None
    view() -> list[Message]

Two of the five additionally expose explicit recall:

    vector_retrieval.VectorRetrieval.query(text, k)
    structured_episodic.StructuredEpisodic.recall_episodes(situation, tags, k)

Top-level imports for convenience:

    from patterns import (
        SlidingWindow,
        SummaryCompression,
        HierarchicalSummary,
        VectorRetrieval,
        StructuredEpisodic,
        Message,
    )

Note: each pattern module also exports its own `Message` dataclass with
identical structure. The top-level `Message` re-exports the
sliding_window variant — they are interchangeable in practice but
mypy will flag cross-module passing if you mix them. Pick one and stay
with it inside any given codebase.
"""
from patterns.sliding_window import SlidingWindow, Message
from patterns.summary_compression import SummaryCompression
from patterns.hierarchical_summary import HierarchicalSummary
from patterns.vector_retrieval import VectorRetrieval
from patterns.structured_episodic import StructuredEpisodic, Episode
from patterns.memory_writer import MemoryWriter, WriteOutcome, ListBackend
from patterns.checkpoint import CheckpointManager, RunState, MemoryBackend, FileBackend
from patterns.tool_fence import (
    Action,
    ToolFence,
    Verdict,
    RuleFn,
    block_pattern_rule,
    warn_pattern_rule,
    sensitive_path_rule,
    task_alignment_rule,
    deobfuscate,
)
from patterns.agent_tx import (
    CompensationEntry,
    MemorySagaBackend,
    SQLiteSagaBackend,
    SagaBackend,
    SagaLog,
    compensable,
    run_saga,
)
from patterns.agent_coord import (
    FileLock,
    WorkQueue,
    Task,
    EventBus,
    Barrier,
    DriftMonitor,
)
from patterns.agent_guard import (
    AgentGuard,
    BudgetGate,
    NoProgressDetector,
    ProgressStatus,
    RateLimitRetrier,
    ToolCallValidator,
    ValidationResult,
)

__all__ = [
    "SlidingWindow",
    "SummaryCompression",
    "HierarchicalSummary",
    "VectorRetrieval",
    "StructuredEpisodic",
    "Episode",
    "MemoryWriter",
    "WriteOutcome",
    "ListBackend",
    "Message",
    "CheckpointManager",
    "RunState",
    "MemoryBackend",
    "FileBackend",
    "Action",
    "ToolFence",
    "Verdict",
    "RuleFn",
    "block_pattern_rule",
    "warn_pattern_rule",
    "sensitive_path_rule",
    "task_alignment_rule",
    "deobfuscate",
    "CompensationEntry",
    "MemorySagaBackend",
    "SQLiteSagaBackend",
    "SagaBackend",
    "SagaLog",
    "compensable",
    "run_saga",
    "FileLock",
    "WorkQueue",
    "Task",
    "EventBus",
    "Barrier",
    "DriftMonitor",
    "AgentGuard",
    "BudgetGate",
    "NoProgressDetector",
    "ProgressStatus",
    "RateLimitRetrier",
    "ToolCallValidator",
    "ValidationResult",
]
