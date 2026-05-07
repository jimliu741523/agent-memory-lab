"""
memory_writer.py — write-path middleware for agent memory stores (RP-3).

Wraps any backend that exposes add(msg)/view() and adds three write-path
operations before forwarding:

  (a) Salience scoring — a pluggable SalienceFn scores each incoming
      message against the current store; messages below salience_threshold
      are dropped and return WriteOutcome.DROPPED_LOW_SALIENCE.
  (b) Contradiction detection — a pluggable ContradictFn returns the
      conflicting existing message (or None). Resolution is pluggable:
        latest_wins  — suppress the old message, commit the new one.
        merge        — suppress the old, commit a merged message.
        flag_for_review — hold the new message; caller uses release()/discard().
  (c) TTL / decay — optional ttl turn count; view() filters out messages
      older than ttl turns. System messages are always kept. ttl=0 disables.

Interface:
    add(msg: Message) -> WriteOutcome
    view() -> list[Message]
    held -> list[Message]          # messages waiting for human review
    release(msg) -> WriteOutcome   # promote held message to backend
    discard(msg) -> None           # drop held message

Backend contract (structural — no import required):
    Any object with add(msg) -> None and view() -> list[Message].
    All five patterns in this repo qualify.

RP-3: memory write-path filtering, contradiction handling, learned
forgetting. See RESEARCH_PRIORITY.md RP-3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class WriteOutcome(Enum):
    COMMITTED = auto()
    DROPPED_LOW_SALIENCE = auto()
    FLAGGED_CONTRADICTION = auto()
    MERGED = auto()
    HELD_FOR_REVIEW = auto()


SalienceFn = Callable[[Message, list[Message]], float]
"""Score a candidate message against current context. Returns 0.0–1.0."""

ContradictFn = Callable[[Message, list[Message]], Optional[Message]]
"""Return the existing message that the candidate contradicts, or None."""

MergeFn = Callable[[Message, Message], Message]
"""Given (new_msg, old_msg), return a merged replacement."""

ResolutionStrategy = Literal["latest_wins", "merge", "flag_for_review"]


# ── default pluggable functions (no LLM required) ────────────────────────────

def _always_salient(_msg: Message, _ctx: list[Message]) -> float:
    """Default: every message is maximally salient."""
    return 1.0


def _no_contradiction(_msg: Message, _ctx: list[Message]) -> Optional[Message]:
    """Default: no contradictions detected."""
    return None


def _naive_merge(new: Message, old: Message) -> Message:
    """Simple content concatenation for demo/test use. Replace with LLM call in production."""
    return Message(role=new.role, content=f"{old.content} [superseded by: {new.content}]")


# ── core middleware ───────────────────────────────────────────────────────────

@dataclass
class MemoryWriter:
    """
    Write-path middleware. Wraps any memory backend and intercepts each
    add() call to apply salience gating, contradiction resolution, and TTL.
    """

    backend: object
    salience_threshold: float = 0.0
    salience_fn: SalienceFn = field(default=_always_salient)
    contradict_fn: ContradictFn = field(default=_no_contradiction)
    merge_fn: MergeFn = field(default=_naive_merge)
    resolution: ResolutionStrategy = "latest_wins"
    ttl: int = 0

    _turn: int = field(default=0, init=False, repr=False)
    _turn_added: dict = field(default_factory=dict, init=False, repr=False)
    _suppressed: set = field(default_factory=set, init=False, repr=False)
    _held: list = field(default_factory=list, init=False, repr=False)

    def add(self, msg: Message) -> WriteOutcome:
        if msg.role == "system":
            self.backend.add(msg)
            self._turn_added[msg] = 0
            return WriteOutcome.COMMITTED

        self._turn += 1
        context = self.view()

        score = self.salience_fn(msg, context)
        if score < self.salience_threshold:
            return WriteOutcome.DROPPED_LOW_SALIENCE

        contradicting = self.contradict_fn(msg, context)
        if contradicting is not None:
            return self._resolve(msg, contradicting)

        self._commit(msg)
        return WriteOutcome.COMMITTED

    def view(self) -> list[Message]:
        msgs = self.backend.view()
        if self._suppressed:
            msgs = [m for m in msgs if m not in self._suppressed]
        if self.ttl > 0:
            msgs = [
                m for m in msgs
                if m.role == "system"
                or self._turn_added.get(m, 0) > self._turn - self.ttl
            ]
        return msgs

    @property
    def held(self) -> list[Message]:
        return list(self._held)

    def release(self, msg: Message) -> WriteOutcome:
        """Commit a held message to the backend."""
        if msg not in self._held:
            raise ValueError(f"Message not in held queue: {msg!r}")
        self._held.remove(msg)
        self._commit(msg)
        return WriteOutcome.COMMITTED

    def discard(self, msg: Message) -> None:
        """Drop a held message without committing it."""
        if msg not in self._held:
            raise ValueError(f"Message not in held queue: {msg!r}")
        self._held.remove(msg)

    # ── private helpers ──────────────────────────────────────────────────────

    def _commit(self, msg: Message) -> None:
        self.backend.add(msg)
        self._turn_added[msg] = self._turn

    def _resolve(self, new: Message, old: Message) -> WriteOutcome:
        if self.resolution == "latest_wins":
            self._suppressed.add(old)
            self._commit(new)
            return WriteOutcome.FLAGGED_CONTRADICTION
        if self.resolution == "merge":
            self._suppressed.add(old)
            merged = self.merge_fn(new, old)
            self._commit(merged)
            return WriteOutcome.MERGED
        # "flag_for_review"
        self._held.append(new)
        return WriteOutcome.HELD_FOR_REVIEW


# ── minimal list backend for demo / testing ──────────────────────────────────

@dataclass
class ListBackend:
    """Trivial append-only backend. Useful for tests and demos."""
    _store: list = field(default_factory=list)

    def add(self, msg: Message) -> None:
        self._store.append(msg)

    def view(self) -> list[Message]:
        return list(self._store)


# ── demo ─────────────────────────────────────────────────────────────────────

def _demo() -> None:
    # --- salience gate demo ---
    print("=== salience gate ===")

    def _score(msg: Message, _ctx: list[Message]) -> float:
        return 0.0 if "noise" in msg.content.lower() else 1.0

    mw = MemoryWriter(backend=ListBackend(), salience_fn=_score, salience_threshold=0.5)
    results = [
        mw.add(Message("user", "important task: redesign the schema")),
        mw.add(Message("user", "noise: random chatter nobody cares about")),
        mw.add(Message("assistant", "I'll redesign the schema")),
    ]
    print(f"outcomes: {[r.name for r in results]}")
    print(f"view ({len(mw.view())} msgs): {[m.content[:30] for m in mw.view()]}")

    # --- contradiction demo ---
    print("\n=== latest_wins contradiction ===")

    def _contradicts(msg: Message, ctx: list[Message]) -> Optional[Message]:
        if "sky is" in msg.content:
            for m in ctx:
                if "sky is" in m.content and m.content != msg.content:
                    return m
        return None

    mw2 = MemoryWriter(backend=ListBackend(), contradict_fn=_contradicts, resolution="latest_wins")
    mw2.add(Message("user", "the sky is green"))
    out = mw2.add(Message("user", "the sky is blue"))
    print(f"outcome: {out.name}")
    print(f"view ({len(mw2.view())} msgs): {[m.content for m in mw2.view()]}")

    # --- TTL demo ---
    print("\n=== TTL decay (ttl=3) ===")
    mw3 = MemoryWriter(backend=ListBackend(), ttl=3)
    for i in range(6):
        mw3.add(Message("user", f"turn {i}"))
    print(f"view after 6 turns with ttl=3 ({len(mw3.view())} msgs): {[m.content for m in mw3.view()]}")


if __name__ == "__main__":
    _demo()
