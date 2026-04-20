"""
Hierarchical-summary memory: a pyramid of rolling summaries.

Raw messages at the leaves are periodically rolled up into L1 summaries;
L1 summaries roll up into L2; L2 into L3; and so on. Recent raw messages
stay verbatim at the tail. The result: older context is retained at
decreasing detail rather than forgotten outright.

When to use:
    Multi-day sessions. Tasks where a short, high-level summary of
    activity from "two days ago" is useful context alongside the last
    few minutes of verbatim detail.

When NOT to use:
    Tasks where old context is NOT useful (use sliding_window — cheaper).
    Tasks where old context must be recalled by topic, not recency
    (use vector_retrieval — planned).
    Short tasks (summary_compression is simpler and sufficient).

Interface (shared by every pattern in this repo):
    add(msg: Message) -> None
    view() -> list[Message]
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


@dataclass
class HierarchicalSummary:
    """
    Maintains a pyramid: raw recent tail -> L1 summaries -> L2 summaries -> ...

    Tuning:
    - `leaf_chunk` — how many raw messages beyond `keep_recent` before one gets rolled into an L1 summary.
    - `fanout` — how many summaries at any level accumulate before rolling up.
    - `keep_recent` — how many raw messages always stay verbatim at the tail.
    """

    summarize: SummarizeFn
    leaf_chunk: int = 10
    fanout: int = 5
    keep_recent: int = 5

    _system: list[Message] = field(default_factory=list)
    # _hierarchy[n] holds the L(n+1) summaries
    _hierarchy: list[list[Message]] = field(default_factory=list)
    _recent: list[Message] = field(default_factory=list)

    def add(self, msg: Message) -> None:
        if msg.role == "system":
            self._system.append(msg)
            return
        self._recent.append(msg)
        overflow = len(self._recent) - self.keep_recent
        while overflow >= self.leaf_chunk:
            chunk = self._recent[: self.leaf_chunk]
            self._recent = self._recent[self.leaf_chunk :]
            summary = self._wrap_summary(chunk, level=1)
            self._push_to_level(summary, level_index=0)
            overflow = len(self._recent) - self.keep_recent

    def view(self) -> list[Message]:
        parts: list[Message] = list(self._system)
        # highest level first — oldest, most compressed context up front
        for level_msgs in reversed(self._hierarchy):
            parts.extend(level_msgs)
        parts.extend(self._recent)
        return parts

    def __len__(self) -> int:
        levels_total = sum(len(level) for level in self._hierarchy)
        return len(self._system) + levels_total + len(self._recent)

    def _wrap_summary(self, msgs: list[Message], level: int) -> Message:
        return Message(
            role="assistant",
            content=f"<L{level}_summary>\n{self.summarize(msgs)}\n</L{level}_summary>",
        )

    def _push_to_level(self, summary_msg: Message, level_index: int) -> None:
        while len(self._hierarchy) <= level_index:
            self._hierarchy.append([])
        self._hierarchy[level_index].append(summary_msg)
        if len(self._hierarchy[level_index]) >= self.fanout:
            chunk = self._hierarchy[level_index][: self.fanout]
            self._hierarchy[level_index] = self._hierarchy[level_index][self.fanout :]
            higher = self._wrap_summary(chunk, level=level_index + 2)
            self._push_to_level(higher, level_index=level_index + 1)


def _mock_summarize(msgs: list[Message]) -> str:
    """Stand-in summarizer. In production pass an LLM call."""
    preview = " | ".join(m.content[:15] for m in msgs[:2])
    return f"[{len(msgs)} items; opens: {preview} ...]"


def _demo() -> None:
    mem = HierarchicalSummary(
        summarize=_mock_summarize, leaf_chunk=4, fanout=3, keep_recent=2
    )
    mem.add(Message("system", "you are a helpful assistant"))
    for i in range(30):
        mem.add(Message("user", f"message {i}"))

    view = mem.view()
    print(f"total in view: {len(view)} (system + {len(mem._hierarchy)} levels + {len(mem._recent)} recent)")
    for m in view:
        body = m.content if len(m.content) < 80 else m.content[:77] + "..."
        print(f"  [{m.role:9s}] {body}")


if __name__ == "__main__":
    _demo()
