"""
Pattern 10 — SagaLog: saga-pattern compensation for agent tool calls.

Problem: agent tool calls can succeed in mutating external state (database
INSERT, file write, API POST) even when the overall agent run subsequently
fails.  At that point there is no compensation path — the state is mutated
but the agent result is an error.

This module provides three composable primitives:

    @compensable(compensation_fn)   — pairs any tool with its rollback function
    SagaLog                         — records successful calls; reverse-replays on failure
    run_saga(log)                   — context manager; auto-compensates on any exception

The compensation function receives ``(result, *original_args, **original_kwargs)``
so it has full access to what was passed to the tool and what it returned.

Pluggable log backends are supported via the ``SagaBackend`` protocol; the
default ``MemorySagaBackend`` keeps entries in-process.  A ``SQLiteSagaBackend``
is included for crash-durable sagas.
"""

import functools
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CompensationEntry:
    """One recorded tool execution plus its rollback handle."""
    tool_name: str
    args: Tuple[Any, ...]
    kwargs: dict
    result: Any
    compensation_fn: Callable


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class SagaBackend:
    """Minimal interface a SagaLog backend must implement."""

    def append(self, entry: CompensationEntry) -> None:
        raise NotImplementedError

    def snapshot(self) -> List[CompensationEntry]:
        """Return a stable ordered copy (oldest first)."""
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class MemorySagaBackend(SagaBackend):
    """Thread-safe in-memory backend (default)."""

    def __init__(self) -> None:
        self._entries: List[CompensationEntry] = []
        self._lock = threading.Lock()

    def append(self, entry: CompensationEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def snapshot(self) -> List[CompensationEntry]:
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class SQLiteSagaBackend(SagaBackend):
    """Crash-durable backend that persists tool names to SQLite.

    Only ``tool_name`` is stored durably; ``compensation_fn``, ``args``,
    ``kwargs``, and ``result`` live in-memory.  The durable record lets an
    operator audit which tools ran; compensation itself still requires the
    in-memory entries (the live ``CompensationEntry`` objects are the source of
    truth for rollback).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS saga_log "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, tool_name TEXT NOT NULL)"
        )
        self._conn.commit()
        self._entries: List[CompensationEntry] = []
        self._lock = threading.Lock()

    def append(self, entry: CompensationEntry) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO saga_log (tool_name) VALUES (?)", (entry.tool_name,))
            self._conn.commit()
            self._entries.append(entry)

    def snapshot(self) -> List[CompensationEntry]:
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM saga_log")
            self._conn.commit()
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def recorded_tool_names(self) -> List[str]:
        """Return tool names from the durable SQLite record (for auditing)."""
        with self._lock:
            rows = self._conn.execute("SELECT tool_name FROM saga_log ORDER BY id").fetchall()
            return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# SagaLog
# ---------------------------------------------------------------------------

class SagaLog:
    """Records successful compensable tool calls; replays them in reverse on failure.

    Usage::

        saga = SagaLog()

        @compensable(lambda result, path, **_: os.remove(path), log=saga)
        def write_file(path: str, content: str) -> str: ...

        write_file("out.txt", "hello")
        # later, on failure:
        saga.compensate()

    Or with the context manager for automatic rollback::

        with run_saga(SagaLog()) as saga:
            write_file("out.txt", "hello", _saga=saga)
            db_insert("record", _saga=saga)
            # if anything raises, both calls are automatically compensated
    """

    def __init__(self, backend: Optional[SagaBackend] = None) -> None:
        self._backend: SagaBackend = backend if backend is not None else MemorySagaBackend()

    def record(self, entry: CompensationEntry) -> None:
        self._backend.append(entry)

    def compensate(self) -> List[str]:
        """Run all compensations in reverse order.

        Returns the names of tools whose compensation functions completed
        without raising.  Compensation errors are swallowed — best-effort
        rollback is intentional (the saga pattern cannot guarantee atomicity
        across external services, only best-effort undo).
        """
        entries = list(reversed(self._backend.snapshot()))
        compensated: List[str] = []
        for entry in entries:
            try:
                entry.compensation_fn(entry.result, *entry.args, **entry.kwargs)
                compensated.append(entry.tool_name)
            except Exception:
                pass
        return compensated

    def entries(self) -> List[CompensationEntry]:
        return self._backend.snapshot()

    def clear(self) -> None:
        self._backend.clear()

    def __len__(self) -> int:
        return len(self._backend)


# ---------------------------------------------------------------------------
# @compensable decorator
# ---------------------------------------------------------------------------

def compensable(
    compensation_fn: Callable,
    log: Optional[SagaLog] = None,
) -> Callable:
    """Decorator pairing a tool function with its rollback/compensation function.

    Args:
        compensation_fn: Called as ``compensation_fn(result, *args, **kwargs)``
            with the original positional args and kwargs of the tool call.
        log: Optional default :class:`SagaLog`.  Can be overridden per-call
            with the ``_saga`` keyword argument.

    The decorated function's return value passes through unchanged.  If the
    function raises, nothing is recorded (no side-effect, no compensation
    needed for that step).

    Example::

        saga = SagaLog()

        @compensable(lambda result, path, **_: os.remove(path), log=saga)
        def write_file(path: str, content: str) -> str:
            with open(path, "w") as f:
                f.write(content)
            return path
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, _saga: Optional[SagaLog] = None, **kwargs: Any) -> Any:
            effective_log = _saga if _saga is not None else log
            result = fn(*args, **kwargs)
            if effective_log is not None:
                effective_log.record(CompensationEntry(
                    tool_name=fn.__name__,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    compensation_fn=compensation_fn,
                ))
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# run_saga context manager
# ---------------------------------------------------------------------------

@contextmanager
def run_saga(saga: SagaLog) -> Generator[SagaLog, None, None]:
    """Context manager that auto-compensates on any exception.

    The saga log is yielded so tools can be called with ``_saga=saga``
    inside the block::

        with run_saga(SagaLog()) as saga:
            write_file("a.txt", "hello", _saga=saga)
            db_insert("record_42", _saga=saga)
            # if anything below raises, write_file and db_insert are undone

    The original exception is re-raised after compensation completes.
    """
    try:
        yield saga
    except Exception:
        saga.compensate()
        raise


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    print("=== agent_tx demo ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        created_files: list = []

        def _undo_write(result: str, path: str, content: str) -> None:
            if os.path.exists(path):
                os.remove(path)
                print(f"  compensated: removed {os.path.basename(path)}")

        @compensable(_undo_write)
        def write_file(path: str, content: str) -> str:
            with open(path, "w") as f:
                f.write(content)
            created_files.append(path)
            print(f"  tool: wrote {os.path.basename(path)}")
            return path

        # Scenario 1 — happy path (no compensation needed)
        print("Scenario 1: happy path")
        saga = SagaLog()
        with run_saga(saga):
            write_file(os.path.join(tmpdir, "step1.txt"), "hello", _saga=saga)
            write_file(os.path.join(tmpdir, "step2.txt"), "world", _saga=saga)
        print(f"  completed OK; {len(saga)} entries recorded, 0 compensated\n")

        # Scenario 2 — failure mid-workflow triggers rollback
        print("Scenario 2: failure after step 1 of 3")
        saga2 = SagaLog()
        try:
            with run_saga(saga2):
                write_file(os.path.join(tmpdir, "s2_step1.txt"), "a", _saga=saga2)
                raise RuntimeError("validation failed at step 2")
        except RuntimeError:
            pass
        remaining = [f for f in os.listdir(tmpdir) if f.startswith("s2_")]
        print(f"  files left behind: {remaining or 'none — clean rollback'}\n")

        print(f"Total files on disk: {os.listdir(tmpdir)}")
