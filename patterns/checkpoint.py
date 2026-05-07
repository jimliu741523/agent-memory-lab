"""
checkpoint.py — framework-agnostic agent-run checkpoint / pause / resume (RP-1).

Provides a portable JSON schema for agent run state and two primitives:

    save(run: RunState) -> token: str
    load(token: str) -> RunState

The schema captures everything needed to pause and resume a long-running
agent across process boundaries, context-window resets, or framework
restarts:

    RunState
      run_id       — stable identifier for the logical run
      step         — monotonic counter of steps executed so far
      tool_queue   — pending tool calls not yet dispatched (list of dicts)
      pending      — in-flight tool calls awaiting response (list of dicts)
      memory_snap  — opaque list[dict] snapshot of the agent's memory context
      stream_cursor — byte offset or message index in the stream at pause time
      metadata     — arbitrary dict for framework-specific extensions
      created_at   — ISO-8601 UTC timestamp

Tokens are SHA-256 hashes of the canonical JSON; they are content-addressed
and immutable — the same RunState always produces the same token.

Backends (structural contract: store(token, json_str) / fetch(token) -> str):
    MemoryBackend  — in-process dict; safe for tests and single-turn demos.
    FileBackend    — one JSON file per checkpoint in a directory.

RP-1: Durable agent runs: checkpoint / pause / resume primitive.
See RESEARCH_PRIORITY.md RP-1.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# ── RunState schema ───────────────────────────────────────────────────────────

@dataclass
class RunState:
    """Portable snapshot of an agent run at a single pause point."""

    run_id: str
    step: int
    tool_queue: list[dict[str, Any]] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    memory_snap: list[dict[str, Any]] = field(default_factory=list)
    stream_cursor: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> str:
        """Canonical JSON representation (sorted keys, no trailing whitespace)."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "RunState":
        data = json.loads(raw)
        return cls(**data)


# ── token helpers ─────────────────────────────────────────────────────────────

def _token(run: RunState) -> str:
    """SHA-256 hex digest of the canonical JSON (content-addressed)."""
    digest = hashlib.sha256(run.to_json().encode()).hexdigest()
    return digest


# ── backend contract (structural — no ABC required) ───────────────────────────
#
#   store(token: str, payload: str) -> None
#   fetch(token: str) -> str          # raises KeyError if not found


@dataclass
class MemoryBackend:
    """In-process dict. Safe for tests and single-process demos."""
    _store: dict[str, str] = field(default_factory=dict)

    def store(self, token: str, payload: str) -> None:
        self._store[token] = payload

    def fetch(self, token: str) -> str:
        try:
            return self._store[token]
        except KeyError:
            raise KeyError(f"Checkpoint not found: {token[:12]}…")


@dataclass
class FileBackend:
    """One JSON file per checkpoint under `directory`."""
    directory: str

    def __post_init__(self) -> None:
        os.makedirs(self.directory, exist_ok=True)

    def _path(self, token: str) -> str:
        return os.path.join(self.directory, f"{token}.json")

    def store(self, token: str, payload: str) -> None:
        with open(self._path(token), "w", encoding="utf-8") as fh:
            fh.write(payload)

    def fetch(self, token: str) -> str:
        path = self._path(token)
        if not os.path.exists(path):
            raise KeyError(f"Checkpoint not found: {token[:12]}…")
        with open(path, encoding="utf-8") as fh:
            return fh.read()


# ── CheckpointManager ─────────────────────────────────────────────────────────

@dataclass
class CheckpointManager:
    """
    save() / load() façade over a pluggable backend.

    Usage::

        mgr = CheckpointManager(backend=MemoryBackend())
        run = RunState(run_id="run-001", step=3,
                       tool_queue=[{"name": "search", "args": {"q": "hello"}}])
        token = mgr.save(run)           # persist, return content-addressed token
        restored = mgr.load(token)      # round-trip identical RunState
    """

    backend: object

    def save(self, run: RunState) -> str:
        """Serialize *run* and persist via the backend. Returns the token."""
        token = _token(run)
        self.backend.store(token, run.to_json())
        return token

    def load(self, token: str) -> RunState:
        """Deserialize the checkpoint identified by *token*. Raises KeyError if absent."""
        raw = self.backend.fetch(token)
        return RunState.from_json(raw)

    def advance(self, token: str, **updates: Any) -> str:
        """
        Convenience: load a checkpoint, apply field updates, save and return the new token.

        Typical use::

            token = mgr.advance(token, step=run.step + 1, tool_queue=new_queue)
        """
        run = self.load(token)
        for k, v in updates.items():
            setattr(run, k, v)
        return self.save(run)


# ── demo ──────────────────────────────────────────────────────────────────────

def _demo() -> None:
    print("=== CheckpointManager (MemoryBackend) ===")
    mgr = CheckpointManager(backend=MemoryBackend())

    run = RunState(
        run_id="demo-run-001",
        step=0,
        tool_queue=[{"name": "web_search", "args": {"q": "agent checkpoint"}}],
        memory_snap=[{"role": "user", "content": "Research agent checkpoints."}],
        metadata={"framework": "raw_loop", "model": "claude-sonnet-4-6"},
    )

    token = mgr.save(run)
    print(f"saved  token: {token[:16]}…")

    restored = mgr.load(token)
    print(f"loaded run_id={restored.run_id!r} step={restored.step}")

    print("\n=== advance: dispatch first tool, bump step ===")
    dispatched = mgr.load(token)
    dispatched.step += 1
    call = dispatched.tool_queue.pop(0)
    dispatched.pending.append(call)
    token2 = mgr.save(dispatched)
    run2 = mgr.load(token2)
    print(f"step={run2.step} pending={run2.pending} queue={run2.tool_queue}")

    print("\n=== idempotent: same state → same token ===")
    run_copy = RunState.from_json(run.to_json())
    token_copy = mgr.save(run_copy)
    print(f"original token == copy token: {token == token_copy}")


if __name__ == "__main__":
    _demo()
