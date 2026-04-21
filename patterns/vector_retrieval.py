"""
Vector-retrieval memory.

When older turns need to be recalled by TOPIC (semantic similarity) rather
than recency, vector retrieval stores each older turn with its embedding
and fetches the top-K most similar at query time.

When to use:
    Large/long conversations where the right context for the current
    turn depends on what was said about a specific subject earlier, not
    on how recently it was said.

When NOT to use:
    Short tasks (sliding_window is cheaper). Tasks where recency is a
    good proxy for relevance (summary_compression). Very long sessions
    where you want graceful detail decay (hierarchical_summary).

The embedder is pluggable (same design as summarize_fn in
summary_compression). A stdlib hash-bag-of-words fallback is included
for tests and demos — it's NOT semantically meaningful in the way a
real embedder is, but it's deterministic and dependency-free.

Interface:
    add(msg: Message) -> None      # same as every pattern
    view() -> list[Message]        # default: system + recent tail
    query(text, k) -> list[Message]   # NEW: semantic recall from archive

Callers who want semantic retrieval call query() explicitly; view()
still returns the boring recency-based slice for drop-in compatibility
with the shared interface.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


EmbedFn = Callable[[str], list[float]]


def _hash_bow_embed(text: str, dim: int = 256) -> list[float]:
    """
    Deterministic stdlib embedder: hashed bag-of-words, unit-normalized.

    NOT a semantically meaningful embedding. Two sentences share vector
    mass only to the extent they share literal tokens. Included so the
    module runs / tests pass without external dependencies. For real use,
    pass an embedder from sentence-transformers / OpenAI / Voyage / Cohere.
    """
    tokens = text.lower().split()
    vec = [0.0] * dim
    for token in tokens:
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class VectorRetrieval:
    """
    Keeps system messages verbatim, a tail of recent messages verbatim,
    and archives older messages with their embeddings for similarity
    retrieval via query().
    """

    embed: EmbedFn
    keep_recent: int = 10
    _system: list[Message] = field(default_factory=list)
    _archive: list[tuple[Message, list[float]]] = field(default_factory=list)
    _recent: list[Message] = field(default_factory=list)

    def add(self, msg: Message) -> None:
        if msg.role == "system":
            self._system.append(msg)
            return
        self._recent.append(msg)
        while len(self._recent) > self.keep_recent:
            oldest = self._recent.pop(0)
            self._archive.append((oldest, self.embed(oldest.content)))

    def view(self) -> list[Message]:
        return [*self._system, *self._recent]

    def query(self, text: str, k: int = 3) -> list[Message]:
        if not self._archive:
            return []
        q_emb = self.embed(text)
        scored = [(_cosine(q_emb, emb), msg) for msg, emb in self._archive]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [msg for _, msg in scored[:k]]

    def __len__(self) -> int:
        return len(self._system) + len(self._archive) + len(self._recent)


def _demo() -> None:
    mem = VectorRetrieval(embed=_hash_bow_embed, keep_recent=3)
    mem.add(Message("system", "you are helpful"))
    mem.add(Message("user", "Let's discuss Python garbage collection"))
    mem.add(Message("assistant", "Python uses reference counting plus a cycle collector"))
    mem.add(Message("user", "What about Rust's borrow checker"))
    mem.add(Message("assistant", "Rust uses ownership and borrowing at compile time"))
    mem.add(Message("user", "How does JavaScript handle memory"))
    mem.add(Message("assistant", "JS uses mark-and-sweep garbage collection"))
    mem.add(Message("user", "Let's talk about something else, like databases"))
    mem.add(Message("assistant", "Databases are stateful systems that store data"))

    print("=== view() returns system + recent tail ===")
    for m in mem.view():
        print(f"  [{m.role:9s}] {m.content[:60]}")

    print("\n=== query('Python memory management', k=2) — semantic recall from archive ===")
    for m in mem.query("Python memory management", k=2):
        print(f"  [{m.role:9s}] {m.content[:60]}")


if __name__ == "__main__":
    _demo()
