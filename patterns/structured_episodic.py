"""
Structured-episodic memory.

Unlike vector_retrieval (which fuzzy-matches past text by semantic
similarity), structured_episodic records completed mini-tasks as typed
records — ``Episode(situation, action, outcome, tags)`` — and retrieves
them by structured key match. Think of it as the agent's case-file
cabinet: "I've deployed to prod before; the last time tests were red
and I deployed anyway, here's what happened."

When to use:
    Multi-session agents that repeat similar tasks and benefit from
    recalling how a past attempt under the *same* conditions turned
    out. Good fit when the situation has a natural typed structure
    (task name, environment, user id, language, ...) rather than a
    free-text query.

When NOT to use:
    Single-session chat where recency is enough (sliding_window).
    Cases where the retrieval query is free text with no natural
    schema (vector_retrieval is a better fit).

Recall ranks episodes by:
    1. number of ``situation`` key/value matches,
    2. plus number of ``tags`` intersected,
    3. ties broken by recency (newest first).

Interface:
    add(msg: Message)          -> None        # standard message buffer
    view() -> list[Message]                   # system + recent tail
    record_episode(...)        -> Episode     # structured recall target
    recall_episodes(...)       -> list[Episode]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Episode:
    situation: dict[str, str]
    action: str
    outcome: str
    tags: tuple[str, ...] = ()

    def __hash__(self) -> int:
        return hash((tuple(sorted(self.situation.items())), self.action, self.outcome, self.tags))


@dataclass
class StructuredEpisodic:
    """
    Typed episodic memory: records completed mini-tasks and recalls past
    episodes by structured key match rather than text similarity.
    """

    keep_recent: int = 10
    _system: list[Message] = field(default_factory=list)
    _recent: list[Message] = field(default_factory=list)
    _episodes: list[Episode] = field(default_factory=list)

    def add(self, msg: Message) -> None:
        if msg.role == "system":
            self._system.append(msg)
            return
        self._recent.append(msg)
        while len(self._recent) > self.keep_recent:
            self._recent.pop(0)

    def view(self) -> list[Message]:
        return [*self._system, *self._recent]

    def record_episode(
        self,
        situation: dict[str, str],
        action: str,
        outcome: str,
        tags: tuple[str, ...] = (),
    ) -> Episode:
        ep = Episode(
            situation=dict(situation),
            action=action,
            outcome=outcome,
            tags=tuple(tags),
        )
        self._episodes.append(ep)
        return ep

    def recall_episodes(
        self,
        situation: Optional[dict[str, str]] = None,
        tags: Optional[tuple[str, ...]] = None,
        k: int = 5,
    ) -> list[Episode]:
        if not self._episodes:
            return []
        sit = situation or {}
        want_tags = set(tags or ())

        scored: list[tuple[int, int, Episode]] = []
        for idx, ep in enumerate(self._episodes):
            matches = sum(1 for key, value in sit.items() if ep.situation.get(key) == value)
            overlap = len(want_tags & set(ep.tags))
            score = matches + overlap
            if score == 0 and (sit or want_tags):
                continue
            # second key: idx preserves insertion order; negated so newer ranks ahead of older at equal score
            scored.append((score, idx, ep))

        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [ep for _, _, ep in scored[:k]]

    def __len__(self) -> int:
        return len(self._system) + len(self._recent) + len(self._episodes)


def _demo() -> None:
    mem = StructuredEpisodic(keep_recent=3)
    mem.add(Message("system", "you are a deploy assistant"))

    mem.record_episode(
        situation={"task": "deploy", "env": "prod", "tests": "green"},
        action="ran `./deploy prod` directly",
        outcome="clean rollout, 0 incidents",
        tags=("safe-path", "deploy"),
    )
    mem.record_episode(
        situation={"task": "deploy", "env": "prod", "tests": "red"},
        action="overrode the gate and deployed anyway (URGENT label)",
        outcome="rollback within 12 minutes; two customer reports",
        tags=("incident", "deploy", "lesson"),
    )
    mem.record_episode(
        situation={"task": "deploy", "env": "staging", "tests": "red"},
        action="deployed to staging to reproduce the failure",
        outcome="repro confirmed; fix landed before prod attempt",
        tags=("debug", "deploy"),
    )

    print("=== recall: same situation as the red-tests prod deploy ===")
    for ep in mem.recall_episodes(
        situation={"task": "deploy", "env": "prod", "tests": "red"}
    ):
        print(f"  situation={ep.situation}")
        print(f"    -> action:  {ep.action}")
        print(f"    -> outcome: {ep.outcome}")
        print(f"    -> tags:    {ep.tags}")

    print("\n=== recall by tag: lessons learned ===")
    for ep in mem.recall_episodes(tags=("lesson",)):
        print(f"  {ep.outcome}  (situation={ep.situation})")


if __name__ == "__main__":
    _demo()
