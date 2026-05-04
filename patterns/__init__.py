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

__all__ = [
    "SlidingWindow",
    "SummaryCompression",
    "HierarchicalSummary",
    "VectorRetrieval",
    "StructuredEpisodic",
    "Episode",
    "Message",
]
