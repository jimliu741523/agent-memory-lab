"""
Pattern 12 — AgentGuard: agent runtime guard middleware.

Anchored to RP-6 (arxiv 2601.06112 ReliabilityBench, arxiv 2601.16280
tool-invocation failure taxonomy, Datadog 2026 8.4M rate-limit errors/month,
$437 overnight runaway + $47k 11-day runaway incidents).

Four composable primitives:
  ToolCallValidator  — JSON-Schema-strict arg validation + repair hints (no deps)
  BudgetGate         — hierarchical cost ceilings (session/agent/sub-agent scope)
  NoProgressDetector — n-gram Jaccard stall detector across tool-output turns
  RateLimitRetrier   — exponential-backoff + jitter retry scheduler

AgentGuard composes all four into a single middleware object.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type


# ---------------------------------------------------------------------------
# ToolCallValidator
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    repair_hint: str = ""


class ToolCallValidator:
    """
    JSON-Schema-strict tool argument validator with repair hints.

    Validates tool call args against registered per-tool schemas without
    any external dependencies. Supported schema keywords: type, properties,
    required, additionalProperties, enum, minimum, maximum, minLength,
    maxLength, items (for arrays).
    """

    _TYPE_MAP: Dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    def __init__(self) -> None:
        self._schemas: Dict[str, dict] = {}

    def register(self, tool_name: str, schema: dict) -> None:
        """Register a JSON Schema dict for a tool's arguments object."""
        self._schemas[tool_name] = schema

    def validate(self, tool_name: str, args: dict) -> ValidationResult:
        """Validate args against the registered schema. Unknown tools pass."""
        if tool_name not in self._schemas:
            return ValidationResult(ok=True, errors=[], repair_hint="")
        schema = self._schemas[tool_name]
        errors: List[str] = []
        self._check_object(args, schema, "args", errors)
        hint = self._build_hint(errors, schema) if errors else ""
        return ValidationResult(ok=not errors, errors=errors, repair_hint=hint)

    # -- internal validators -------------------------------------------------

    def _check_type(self, val: Any, vtype: str, path: str, errors: List[str]) -> None:
        if vtype == "integer":
            if isinstance(val, bool) or not isinstance(val, int):
                errors.append(f"{path}: expected integer, got {type(val).__name__}")
        elif vtype == "number":
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                errors.append(f"{path}: expected number, got {type(val).__name__}")
        elif vtype == "boolean":
            if not isinstance(val, bool):
                errors.append(f"{path}: expected boolean, got {type(val).__name__}")
        else:
            py_type = self._TYPE_MAP.get(vtype)
            if py_type and not isinstance(val, py_type):
                errors.append(f"{path}: expected {vtype}, got {type(val).__name__}")

    def _check_value(self, val: Any, schema: dict, path: str, errors: List[str]) -> None:
        vtype = schema.get("type")
        if vtype:
            pre = len(errors)
            self._check_type(val, vtype, path, errors)
            if len(errors) > pre:
                return  # type wrong; skip deeper checks

        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{path}: {val!r} not in enum {schema['enum']}")

        if not isinstance(val, bool) and isinstance(val, (int, float)):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{path}: {val} < minimum {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{path}: {val} > maximum {schema['maximum']}")

        if isinstance(val, str):
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{path}: length {len(val)} < minLength {schema['minLength']}")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{path}: length {len(val)} > maxLength {schema['maxLength']}")

        if isinstance(val, list) and "items" in schema:
            for i, item in enumerate(val):
                self._check_value(item, schema["items"], f"{path}[{i}]", errors)

        if isinstance(val, dict):
            self._check_object(val, schema, path, errors)

    def _check_object(self, obj: Any, schema: dict, path: str, errors: List[str]) -> None:
        if not isinstance(obj, dict):
            return
        required = schema.get("required", [])
        props = schema.get("properties", {})
        add_props = schema.get("additionalProperties", True)

        for req in required:
            if req not in obj:
                errors.append(f"{path}.{req}: required field missing")

        for key, val in obj.items():
            if key in props:
                self._check_value(val, props[key], f"{path}.{key}", errors)
            elif add_props is False:
                errors.append(f"{path}.{key}: unexpected additional property")

    def _build_hint(self, errors: List[str], schema: dict) -> str:
        lines = ["Fix the following issues:"]
        for e in errors:
            lines.append(f"  • {e}")
        required = schema.get("required", [])
        if required:
            lines.append(f"Required fields: {required}")
        props = schema.get("properties", {})
        if props:
            lines.append(f"Expected types: { {k: v.get('type', 'any') for k, v in props.items()} }")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BudgetGate
# ---------------------------------------------------------------------------

class BudgetGate:
    """
    Hierarchical, thread-safe cost ceiling enforcer.

    Scopes are arbitrary strings: "session", "agent:worker1",
    "sub-agent:worker1.fetch". set_ceiling() registers a spending cap;
    record_cost() accumulates spend and fires on_exceed callbacks exactly
    once per crossing; check_scope() returns False when the cap is breached.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ceilings: Dict[str, float] = {}
        self._spent: Dict[str, float] = {}
        self._callbacks: Dict[str, List[Callable[[str, float, float], None]]] = {}

    def set_ceiling(self, scope: str, ceiling: float) -> None:
        with self._lock:
            self._ceilings[scope] = ceiling
            self._spent.setdefault(scope, 0.0)

    def register_on_exceed(
        self, scope: str, callback: Callable[[str, float, float], None]
    ) -> None:
        """Register callback(scope, spent, ceiling) fired on first crossing."""
        with self._lock:
            self._callbacks.setdefault(scope, []).append(callback)

    def record_cost(self, scope: str, amount: float) -> None:
        """Accumulate cost for scope; fire callbacks if ceiling first crossed."""
        with self._lock:
            prev = self._spent.get(scope, 0.0)
            new = prev + amount
            self._spent[scope] = new
            ceiling = self._ceilings.get(scope)
            if ceiling is not None and prev <= ceiling < new:
                to_fire: List[Callable] = list(self._callbacks.get(scope, []))
                c_snap, n_snap = ceiling, new
            else:
                to_fire = []
                c_snap = n_snap = 0.0
        for cb in to_fire:
            cb(scope, n_snap, c_snap)

    def check_scope(self, scope: str) -> bool:
        """Return True if scope is within its ceiling (or has no ceiling)."""
        with self._lock:
            ceiling = self._ceilings.get(scope)
            if ceiling is None:
                return True
            return self._spent.get(scope, 0.0) <= ceiling

    def spent(self, scope: str) -> float:
        with self._lock:
            return self._spent.get(scope, 0.0)

    def remaining(self, scope: str) -> Optional[float]:
        """Return budget remaining, or None if no ceiling set for scope."""
        with self._lock:
            ceiling = self._ceilings.get(scope)
            if ceiling is None:
                return None
            return max(0.0, ceiling - self._spent.get(scope, 0.0))


# ---------------------------------------------------------------------------
# NoProgressDetector
# ---------------------------------------------------------------------------

class ProgressStatus(Enum):
    PROGRESSING = "progressing"
    STALLED = "stalled"
    INSUFFICIENT_DATA = "insufficient_data"


class NoProgressDetector:
    """
    N-gram Jaccard stall detector for agent tool-output streams.

    observe() returns STALLED when `stall_window` consecutive output
    pairs all exceed `similarity_threshold`. Character n-grams are used
    for fast, dependency-free similarity that catches semantic repetition
    even when minor wording varies.
    """

    def __init__(
        self,
        stall_window: int = 3,
        similarity_threshold: float = 0.85,
        ngram_n: int = 3,
    ) -> None:
        if stall_window < 1:
            raise ValueError("stall_window must be >= 1")
        self._stall_window = stall_window
        self._sim_threshold = similarity_threshold
        self._ngram_n = ngram_n
        self._history: List[str] = []
        self._stall_run: int = 0

    def observe(self, output: str) -> ProgressStatus:
        """Record a tool output; return progress status."""
        self._history.append(output)
        if len(self._history) < 2:
            return ProgressStatus.INSUFFICIENT_DATA
        sim = self._jaccard(self._history[-2], self._history[-1])
        if sim >= self._sim_threshold:
            self._stall_run += 1
        else:
            self._stall_run = 0
        if self._stall_run >= self._stall_window:
            return ProgressStatus.STALLED
        if len(self._history) <= self._stall_window:
            return ProgressStatus.INSUFFICIENT_DATA
        return ProgressStatus.PROGRESSING

    def reset(self) -> None:
        self._history.clear()
        self._stall_run = 0

    def _ngrams(self, text: str) -> set:
        n = self._ngram_n
        if len(text) < n:
            return {text} if text else set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def _jaccard(self, a: str, b: str) -> float:
        sa, sb = self._ngrams(a), self._ngrams(b)
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# RateLimitRetrier
# ---------------------------------------------------------------------------

class RateLimitRetrier:
    """
    Exponential-backoff + jitter retry scheduler for rate-limit errors.

    Default exc_types catches ALL exceptions; narrow it to the exact
    exception types from your HTTP/SDK layer (e.g. anthropic.RateLimitError).
    Override is_rate_limit_error() for status-code-level inspection.
    """

    def __init__(
        self,
        max_retries: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exc_types: Tuple[Type[Exception], ...] = (Exception,),
        on_retry: Optional[Callable[[int, float, Exception], None]] = None,
    ) -> None:
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._exc_types = exc_types
        self._on_retry = on_retry

    def is_rate_limit_error(self, exc: Exception) -> bool:
        """Override to add status-code inspection (e.g. check exc.status == 429)."""
        return isinstance(exc, self._exc_types)

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call fn(*args, **kwargs), retrying on rate-limit errors with backoff."""
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if attempt >= self._max_retries or not self.is_rate_limit_error(exc):
                    raise
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                jitter = delay * 0.1 * (2 * random.random() - 1)
                wait = max(0.0, delay + jitter)
                if self._on_retry:
                    self._on_retry(attempt + 1, wait, exc)
                time.sleep(wait)
                attempt += 1


# ---------------------------------------------------------------------------
# AgentGuard — composite middleware
# ---------------------------------------------------------------------------

class AgentGuard:
    """
    Composite runtime guard: validator + budget + progress + retry.

    Drop in as middleware between the agent loop and the tool registry.
    Each primitive is independently accessible and can be used standalone.
    """

    def __init__(
        self,
        validator: Optional[ToolCallValidator] = None,
        budget: Optional[BudgetGate] = None,
        progress: Optional[NoProgressDetector] = None,
        retrier: Optional[RateLimitRetrier] = None,
    ) -> None:
        self.validator = validator if validator is not None else ToolCallValidator()
        self.budget = budget if budget is not None else BudgetGate()
        self.progress = progress if progress is not None else NoProgressDetector()
        self.retrier = retrier if retrier is not None else RateLimitRetrier()
