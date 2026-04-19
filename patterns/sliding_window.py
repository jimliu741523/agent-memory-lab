"""
Sliding-window memory: keep the most recent N non-system messages.

When to use:
    Short-to-medium tasks where anything older than N turns is not worth
    paying to re-send. Cheapest possible memory. Zero dependencies.

When NOT to use:
    If the answer at turn 50 depends on a fact introduced at turn 3, this
    pattern will quietly drop it. Reach for summary_compression, vector_retrieval,
    or hierarchical_summary instead.

Interface (shared by every pattern in this repo):
    add(msg: Message) -> None
    view() -> list[Message]          # returns the messages to send to the LLM
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass
class SlidingWindow:
    """Keep the most recent `window` non-system messages. System messages are always kept."""

    window: int = 20
    _system: list[Message] = field(default_factory=list)
    _recent: list[Message] = field(default_factory=list)

    def add(self, msg: Message) -> None:
        if msg.role == "system":
            self._system.append(msg)
            return
        self._recent.append(msg)
        if len(self._recent) > self.window:
            # keep only the last `window` entries
            self._recent = self._recent[-self.window :]

    def view(self) -> list[Message]:
        return [*self._system, *self._recent]

    def __len__(self) -> int:
        return len(self._system) + len(self._recent)


def _demo() -> None:
    mem = SlidingWindow(window=3)
    mem.add(Message("system", "you are a helpful assistant"))
    for i in range(5):
        mem.add(Message("user", f"msg {i}"))
        mem.add(Message("assistant", f"reply {i}"))

    view = mem.view()
    print(f"total in view: {len(view)} (1 system + up to {mem.window} recent)")
    for m in view:
        print(f"  [{m.role:9s}] {m.content}")


if __name__ == "__main__":
    _demo()
