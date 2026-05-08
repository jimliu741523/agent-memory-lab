"""
Pattern 11 — AgentCoord: multi-agent coordination primitives.

Anchored to RP-5 (arxiv 2601.04170, 2605.03310, 2505.21298, 2503.13657,
2604.19540 + practitioner gist documenting ~20k LOC of custom infra for 2
parallel agents). Fills the gap: no OSS library provides these primitives
as composable, framework-agnostic Python objects.

Five primitives:
  FileLock   — advisory file lock with TTL lease and stale-lock recovery
  WorkQueue  — atomic claim/release task queue (SQLite-backed)
  EventBus   — in-process pub/sub for cross-agent signals
  Barrier    — phase gate: blocks until N agents have all arrived
  DriftMonitor — tracks semantic drift across agent turns via embedding deltas
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional


# ---------------------------------------------------------------------------
# FileLock
# ---------------------------------------------------------------------------

class FileLock:
    """
    Advisory file lock with TTL lease.

    Uses O_CREAT|O_EXCL for atomic creation (POSIX-safe). A lock whose
    lease has expired is treated as stale and can be re-acquired by any caller.
    """

    def __init__(self, path: str | Path, lease_secs: float = 30.0) -> None:
        self.path = Path(path)
        self.lease_secs = lease_secs
        self._lock_path = Path(str(path) + ".lock")

    # --- public API ---------------------------------------------------------

    def acquire(self, owner: str = "") -> bool:
        """Try to acquire the lock. Returns True if acquired."""
        now = time.time()
        if self._lock_path.exists():
            try:
                data = json.loads(self._lock_path.read_text())
                if now - data.get("ts", 0) < self.lease_secs:
                    return False  # held and fresh
                self._lock_path.unlink(missing_ok=True)
            except (json.JSONDecodeError, OSError):
                try:
                    self._lock_path.unlink(missing_ok=True)
                except OSError:
                    pass

        try:
            fd = os.open(
                str(self._lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            try:
                os.write(fd, json.dumps({"owner": owner, "ts": now}).encode())
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            return False

    def release(self, owner: str = "") -> None:
        """Release the lock. Pass the same owner string used in acquire()."""
        if not self._lock_path.exists():
            return
        try:
            data = json.loads(self._lock_path.read_text())
            if data.get("owner") == owner or owner == "":
                self._lock_path.unlink(missing_ok=True)
        except (json.JSONDecodeError, OSError):
            try:
                self._lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def is_stale(self) -> bool:
        """Return True if a lock file exists AND its lease has expired."""
        if not self._lock_path.exists():
            return False
        try:
            data = json.loads(self._lock_path.read_text())
            return time.time() - data.get("ts", 0) >= self.lease_secs
        except (json.JSONDecodeError, OSError):
            return True

    @contextmanager
    def locked(
        self, owner: str = "", timeout: float = 5.0
    ) -> Generator[None, None, None]:
        """Context manager. Raises TimeoutError if lock not acquired within timeout."""
        deadline = time.time() + timeout
        while True:
            if self.acquire(owner):
                try:
                    yield
                finally:
                    self.release(owner)
                return
            if time.time() > deadline:
                raise TimeoutError(
                    f"FileLock on {self.path} not acquired within {timeout}s"
                )
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# WorkQueue
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    payload: Any
    status: str = "pending"  # pending | claimed | done | failed


class WorkQueue:
    """
    Atomic claim/release task queue.

    Each claim() is wrapped in a SQLite transaction so two concurrent workers
    cannot claim the same task. Stale leases (claimed_at older than lease_secs)
    are automatically returned to 'pending' at the start of each claim() call.

    Default db_path=':memory:' is convenient for tests; pass a file path for
    crash-durable persistence.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_db()

    # --- internal -----------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path, check_same_thread=False
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                payload     TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                claimed_by  TEXT,
                claimed_at  REAL,
                lease_secs  REAL
            )
            """
        )
        conn.commit()

    # --- public API ---------------------------------------------------------

    def put(self, task_id: str, payload: Any) -> None:
        """Enqueue a task. Silently ignored if task_id already exists."""
        conn = self._conn()
        with self._write_lock:
            conn.execute(
                "INSERT OR IGNORE INTO tasks (id, payload) VALUES (?, ?)",
                (task_id, json.dumps(payload)),
            )
            conn.commit()

    def claim(self, worker_id: str, lease_secs: float = 60.0) -> Optional[Task]:
        """
        Atomically claim one pending task. Returns None if no pending tasks.
        Automatically reclaims tasks whose leases have expired.
        """
        conn = self._conn()
        now = time.time()
        with self._write_lock:
            with conn:  # SQLite transaction
                # Reclaim tasks whose stored lease has expired (claimed_at + lease_secs < now).
                conn.execute(
                    "UPDATE tasks "
                    "SET status='pending', claimed_by=NULL, claimed_at=NULL, lease_secs=NULL "
                    "WHERE status='claimed' "
                    "AND claimed_at IS NOT NULL AND lease_secs IS NOT NULL "
                    "AND claimed_at + lease_secs < ?",
                    (now,),
                )
                row = conn.execute(
                    "SELECT id, payload FROM tasks WHERE status='pending' LIMIT 1"
                ).fetchone()
                if row is None:
                    return None
                task_id, payload_json = row
                conn.execute(
                    "UPDATE tasks "
                    "SET status='claimed', claimed_by=?, claimed_at=?, lease_secs=? "
                    "WHERE id=? AND status='pending'",
                    (worker_id, now, lease_secs, task_id),
                )
        return Task(id=task_id, payload=json.loads(payload_json), status="claimed")

    def release(
        self, task_id: str, *, done: bool = False, failed: bool = False
    ) -> None:
        """
        Release a claimed task. Pass done=True or failed=True to mark terminal
        state; otherwise the task returns to 'pending'.
        """
        status = "done" if done else ("failed" if failed else "pending")
        conn = self._conn()
        with self._write_lock:
            conn.execute(
                "UPDATE tasks "
                "SET status=?, claimed_by=NULL, claimed_at=NULL, lease_secs=NULL "
                "WHERE id=?",
                (status, task_id),
            )
            conn.commit()

    def stats(self) -> Dict[str, int]:
        """Return counts by status: {'pending': N, 'claimed': M, 'done': K, ...}"""
        conn = self._conn()
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM tasks GROUP BY status"
        ).fetchall()
        return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

_EventHandler = Callable[[str, Any], None]


class EventBus:
    """
    In-process publish/subscribe event bus for cross-agent signals.
    Thread-safe. Each publish() call invokes all handlers synchronously
    in the calling thread.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[_EventHandler]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event: str, handler: _EventHandler) -> None:
        with self._lock:
            self._handlers.setdefault(event, []).append(handler)

    def publish(self, event: str, payload: Any = None) -> int:
        """Fire all handlers for event. Returns number of handlers invoked."""
        with self._lock:
            handlers = list(self._handlers.get(event, []))
        for h in handlers:
            h(event, payload)
        return len(handlers)

    def unsubscribe(self, event: str, handler: _EventHandler) -> bool:
        """Remove a specific handler. Returns True if it was found."""
        with self._lock:
            lst = self._handlers.get(event, [])
            if handler in lst:
                lst.remove(handler)
                return True
        return False


# ---------------------------------------------------------------------------
# Barrier
# ---------------------------------------------------------------------------

class Barrier:
    """
    Phase gate: blocks until exactly n_agents have all called .wait().
    Reusable across multiple phases. Thread-safe. Does not break on timeout
    (unlike threading.Barrier), so surviving agents can retry.
    """

    def __init__(self, n_agents: int) -> None:
        self.n_agents = n_agents
        self._cond = threading.Condition(threading.Lock())
        self._count = 0
        self._generation = 0

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Block until n_agents have arrived. Returns True on success,
        False if timeout expired before all agents arrived.
        A timed-out caller's slot is returned so the barrier can still fire
        for future arrivals.
        """
        with self._cond:
            gen = self._generation
            self._count += 1
            if self._count >= self.n_agents:
                self._count = 0
                self._generation += 1
                self._cond.notify_all()
                return True
            result = self._cond.wait_for(
                lambda: self._generation != gen, timeout=timeout
            )
            if not result:
                self._count -= 1  # reclaim the slot on timeout
            return result


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

@dataclass
class _DriftSample:
    agent_id: str
    turn: int
    embedding: List[float]
    ts: float = field(default_factory=time.time)


class DriftMonitor:
    """
    Tracks semantic drift across agent turns by comparing consecutive intent
    embeddings via cosine distance.

    The caller is responsible for producing embeddings (e.g. via a fast
    embedding model). DriftMonitor is model-agnostic; it only compares
    successive vectors. When the delta exceeds threshold, on_drift is called.

    Anchored to arxiv 2601.04170 (36.9% of production failures attributed to
    coordination drift) and arxiv 2605.03310 (drift monitor + conflict
    detection as a required architectural layer).
    """

    def __init__(
        self,
        threshold: float = 0.3,
        on_drift: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.threshold = threshold
        self.on_drift = on_drift
        self._history: Dict[str, List[_DriftSample]] = {}
        self._lock = threading.Lock()

    def record(
        self, agent_id: str, turn: int, embedding: List[float]
    ) -> Optional[float]:
        """
        Record a new intent embedding for agent_id at turn.
        Returns cosine-distance delta from the previous sample, or None for
        the first sample. Fires on_drift callback if delta > threshold.
        """
        sample = _DriftSample(agent_id=agent_id, turn=turn, embedding=embedding)
        with self._lock:
            history = self._history.setdefault(agent_id, [])
            if not history:
                history.append(sample)
                return None
            prev = history[-1]
            history.append(sample)

        delta = self._cosine_distance(prev.embedding, sample.embedding)
        if delta > self.threshold and self.on_drift is not None:
            self.on_drift(agent_id, delta)
        return delta

    def drift_history(self, agent_id: str) -> List[float]:
        """Return list of successive drift deltas for agent_id."""
        with self._lock:
            history = self._history.get(agent_id, [])
            if len(history) < 2:
                return []
            return [
                self._cosine_distance(history[i - 1].embedding, history[i].embedding)
                for i in range(1, len(history))
            ]

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 1.0
        return 1.0 - dot / (na * nb)
