"""
Pattern 13 — McpPipe: tool-call composition without LLM round-tripping.

Anchored to RP-2 (HN signal: 6+ independent mentions of the MCP context-cost
problem in a single thread; arxiv 2605.00827 "Separating Intelligence from
Execution" demonstrates >99% per-execution token cost reduction via DAG-based
piping; GitHub langchain-ai/langchain#34130 — 29 reactions; multiple HN
threads on quadratic context cost from tool-output materialization).

Problem: MCP tool calls return values as content that the LLM sees in full —
there is no handle-based or pipe mechanism. A chain of N dependent tool calls
burns O(N * output_size) tokens even when the LLM only needs the final result.

McpPipe wraps any Python callable (representing a tool) and provides:
  ResultHandle  — opaque content-addressed ID for a stored tool result
  HandleStore   — thread-safe store mapping handle IDs to raw results
  McpPipe       — orchestrator:
      .call(tool_fn, ...)              -> ResultHandle  (execute + store)
      .transform(handle, expr)         -> ResultHandle  (filter/reshape in-store)
      .pipe(handle, expr, tool_fn, **) -> ResultHandle  (chain without roundtrip)
      .materialize(handle)             -> Any           (retrieve raw value)
      .token_savings(handle_ids)       -> int           (bytes kept out of context)

The LLM never sees intermediate results — it only receives the final
materialized value when the agent explicitly calls .materialize().
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Union


# ---------------------------------------------------------------------------
# ResultHandle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResultHandle:
    """Opaque reference to a stored tool result.

    handle_id:    UUID string, unique per call (same content → different IDs)
    content_hash: SHA-256 hex digest of the serialized value (content-addressed)
    byte_size:    serialized byte size — used to report token savings
    """
    handle_id: str
    content_hash: str
    byte_size: int


def _serialize(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode()


def _make_handle(value: Any) -> ResultHandle:
    raw = _serialize(value)
    return ResultHandle(
        handle_id=str(uuid.uuid4()),
        content_hash=hashlib.sha256(raw).hexdigest(),
        byte_size=len(raw),
    )


# ---------------------------------------------------------------------------
# HandleStore
# ---------------------------------------------------------------------------

class HandleStore:
    """Thread-safe mapping from handle_id -> raw result."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: Dict[str, Any] = {}

    def put(self, handle: ResultHandle, value: Any) -> None:
        with self._lock:
            self._store[handle.handle_id] = value

    def get(self, handle: ResultHandle) -> Any:
        with self._lock:
            if handle.handle_id not in self._store:
                raise KeyError(f"Handle {handle.handle_id!r} not found in store")
            return self._store[handle.handle_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Path expression evaluator (no external deps)
# ---------------------------------------------------------------------------

def _apply_expr(value: Any, expr: Union[str, Callable[[Any], Any]]) -> Any:
    """Apply a dot-path expression or callable to a value.

    Expr forms:
      callable     — called with value directly
      "."          — identity, returns value unchanged
      ".key"       — dict key access
      ".key.sub"   — nested dict key access
      ".[0]"       — list index access
      ".key.[2].sub" — mixed dict + list navigation

    Examples:
      _apply_expr({"a": {"b": 1}}, ".a.b")         -> 1
      _apply_expr([10, 20, 30], ".[1]")             -> 20
      _apply_expr({"r": ["x", "y"]}, ".r.[0]")     -> "x"
      _apply_expr(data, lambda v: v["n"] * 2)       -> computed
    """
    if callable(expr):
        return expr(value)
    if not isinstance(expr, str):
        raise TypeError(f"expr must be str or callable, got {type(expr)!r}")
    current = value
    for part in expr.split(".")[1:]:   # split on ".", skip empty leading segment
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            current = current[int(part[1:-1])]
        else:
            current = current[part]
    return current


# ---------------------------------------------------------------------------
# McpPipe
# ---------------------------------------------------------------------------

class McpPipe:
    """Tool-call composition middleware — keeps intermediate results out of LLM context.

    Usage::

        pipe = McpPipe()

        # execute tool A, get a handle (LLM never sees the raw result)
        h_a = pipe.call(search_tool, query="agent memory")

        # extract one field — no LLM roundtrip, no tokens spent
        h_url = pipe.transform(h_a, ".results.[0].url")

        # feed into tool B, again without surfacing to LLM
        h_b = pipe.pipe(h_url, ".", fetch_tool, inject_key="url")

        # only materialize when the agent genuinely needs to reason about it
        final = pipe.materialize(h_b)

        # report bytes kept out of context
        savings = pipe.token_savings([h_a, h_url])
    """

    def __init__(self) -> None:
        self._store = HandleStore()
        self._materializations = 0

    def call(self, tool_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> ResultHandle:
        """Execute tool_fn, store result, return handle (result not surfaced to LLM)."""
        result = tool_fn(*args, **kwargs)
        handle = _make_handle(result)
        self._store.put(handle, result)
        return handle

    def transform(
        self,
        handle: ResultHandle,
        expr: Union[str, Callable[[Any], Any]],
    ) -> ResultHandle:
        """Apply expr to the stored result; store the transformed value as a new handle.

        The original handle remains valid — both old and new handles coexist.
        """
        raw = self._store.get(handle)
        transformed = _apply_expr(raw, expr)
        new_handle = _make_handle(transformed)
        self._store.put(new_handle, transformed)
        return new_handle

    def pipe(
        self,
        handle: ResultHandle,
        expr: Union[str, Callable[[Any], Any]],
        tool_fn: Callable[..., Any],
        inject_key: str = "input",
        **extra_kwargs: Any,
    ) -> ResultHandle:
        """Transform handle's value, then feed it directly into tool_fn.

        Equivalent to::

            intermediate = transform(handle, expr)
            result = tool_fn(**{inject_key: materialize(intermediate)}, **extra_kwargs)

        but without incrementing the materialization counter — the intermediate
        value never surfaces to the LLM context.

        Args:
            handle:       source handle
            expr:         path expression or callable to extract/reshape the value
            tool_fn:      callable to invoke with the transformed value
            inject_key:   kwarg name under which to pass the transformed value
            **extra_kwargs: additional kwargs forwarded to tool_fn
        """
        raw = self._store.get(handle)
        intermediate = _apply_expr(raw, expr)
        result = tool_fn(**{inject_key: intermediate}, **extra_kwargs)
        new_handle = _make_handle(result)
        self._store.put(new_handle, result)
        return new_handle

    def materialize(self, handle: ResultHandle) -> Any:
        """Retrieve stored value — the only point where data surfaces to the caller."""
        self._materializations += 1
        return self._store.get(handle)

    def token_savings(self, handles: Sequence[ResultHandle]) -> int:
        """Total serialized bytes of results kept out of LLM context."""
        return sum(h.byte_size for h in handles)

    def stats(self) -> Dict[str, int]:
        """Summary counters: handles stored and explicit materialization calls."""
        return {
            "handles_stored": len(self._store),
            "materializations": self._materializations,
        }
