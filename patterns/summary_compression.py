"""
Summary-compression memory: when the non-system buffer exceeds `trigger`
messages, summarize the oldest portion (keeping `keep` most-recent verbatim)
and store the summary in place.

When to use:
    Medium-length tasks where cost control matters but you can't afford
    to just drop old context. Pay one summarization call per trigger to
    keep the window bounded.

When NOT to use:
    Very long sessions spanning sub-problems (hierarchical_summary).
    Tasks where old context must be recalled by topic, not recency
    (vector_retrieval). One-shot short tasks (sliding_window is cheaper).

Interface (shared by every pattern in this repo):
    add(msg: Message) -> None
    view() -> list[Message]

The summarizer is a pluggable callable, not baked in: you pass whatever
function you like (`lambda msgs: openai_summarize(msgs)` or similar).
The default `_mock_summarize` lets you run the demo and tests without
any LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


SummarizeFn = Callable[[list[Message]], str]
"""Given a list of Messages, return a single summary string."""


@dataclass
class SummaryCompression:
    """
    Keeps system messages verbatim, keeps the `keep` most-recent verbatim,
    and summarizes everything in between into one rolling summary message.

    Compression triggers when the non-system buffer exceeds `trigger`.
    After each compression the oldest portion is folded into a new summary
    message that sits between system messages and recent ones.
    """

    summarize: SummarizeFn
    trigger: int = 40
    keep: int = 10
    _system: list[Message] = field(default_factory=list)
    _summaries: list[Message] = field(default_factory=list)
    _recent: list[Message] = field(default_factory=list)

    def add(self, msg: Message) -> None:
        if msg.role == "system":
            self._system.append(msg)
            return
        self._recent.append(msg)
        if len(self._recent) > self.trigger:
            self._compress()

    def view(self) -> list[Message]:
        return [*self._system, *self._summaries, *self._recent]

    def __len__(self) -> int:
        return len(self._system) + len(self._summaries) + len(self._recent)

    def _compress(self) -> None:
        to_compress = self._recent[: -self.keep]
        self._recent = self._recent[-self.keep :]
        summary_text = self.summarize(to_compress)
        self._summaries.append(
            Message(
                role="assistant",
                content=f"<previous_summary>\n{summary_text}\n</previous_summary>",
            )
        )


def _mock_summarize(msgs: list[Message]) -> str:
    """Placeholder summarizer for demos and tests. Pass a real LLM call in production."""
    preview = " | ".join(m.content[:20] for m in msgs[:3])
    return f"[{len(msgs)} earlier messages; opening turns: {preview} ...]"


def _demo() -> None:
    mem = SummaryCompression(summarize=_mock_summarize, trigger=5, keep=2)
    mem.add(Message("system", "you are a helpful assistant"))
    for i in range(8):
        mem.add(Message("user", f"message number {i}"))
        mem.add(Message("assistant", f"reply to message {i}"))

    view = mem.view()
    print(f"total in view: {len(view)} (system + summaries + last {mem.keep} recent)")
    for m in view:
        body = m.content if len(m.content) < 70 else m.content[:67] + "..."
        print(f"  [{m.role:9s}] {body}")


if __name__ == "__main__":
    _demo()
