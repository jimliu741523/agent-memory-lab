"""
examples/compose.py — sliding window + vector retrieval, composed.

The five patterns in `patterns/` aren't exclusive; the production
shape is usually a composition. This example shows the most common
one: keep recent turns verbatim with `SlidingWindow` (fast, cheap),
*and* archive every overflow turn into `VectorRetrieval` so older
specific facts can still be recalled by topic.

Run:
    python -m examples.compose
"""
from __future__ import annotations

from patterns import (
    Message,
    SlidingWindow,
    VectorRetrieval,
)
from patterns.vector_retrieval import _hash_bow_embed


class HybridMemory:
    """
    Composes:
      - SlidingWindow(window=5)        for the live recent tail
      - VectorRetrieval(keep_recent=0) as an archive of everything else

    Every add() lands in *both*. View() returns the recent tail.
    Query() reaches into the vector archive for topical recall.

    No magic — just the same callables side by side.
    """

    def __init__(self, window: int = 5):
        self.recent = SlidingWindow(window=window)
        self.archive = VectorRetrieval(embed=_hash_bow_embed, keep_recent=0)

    def add(self, msg: Message) -> None:
        self.recent.add(msg)
        self.archive.add(msg)

    def view(self) -> list[Message]:
        return self.recent.view()

    def query(self, text: str, k: int = 3) -> list[Message]:
        return self.archive.query(text, k=k)


def _demo() -> None:
    mem = HybridMemory(window=3)
    mem.add(Message("system", "you are an ops assistant"))
    mem.add(Message("user", "the launch code is ALPHA-7742"))
    mem.add(Message("assistant", "noted"))
    for i in range(20):
        mem.add(Message("user", f"please summarize document #{i}"))
        mem.add(Message("assistant", f"summary {i}"))
    mem.add(Message("user", "what is the launch code?"))

    print("=== view() — sliding window tail (recent) ===")
    for m in mem.view():
        print(f"  [{m.role:9s}] {m.content[:60]}")

    print("\n=== query('launch code') — vector archive recall ===")
    for m in mem.query("launch code", k=2):
        print(f"  [{m.role:9s}] {m.content[:60]}")


if __name__ == "__main__":
    _demo()
