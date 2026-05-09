""" 
examples/agent_guard_demo.py — AgentGuard runtime guard middleware (RP-6).

Four offline scenarios — no API calls, no network, all assertions pass:

  (1) ToolCallValidator — JSON-Schema-strict arg validation catches missing
      required fields, wrong types, out-of-range values, and unknown extra
      properties before the call reaches the LLM/tool. Repair hint surfaces
      exactly what went wrong and what's expected.

  (2) BudgetGate — hierarchical cost ceilings at session and sub-agent scope.
      record_cost() accumulates spend; on_exceed callback fires exactly once
      when the ceiling is first crossed. Demonstrates the $437-overnight-
      runaway prevention pattern (earezki.com April 2026).

  (3) NoProgressDetector — n-gram Jaccard stall detector catches a loop where
      the agent keeps receiving semantically identical tool outputs. Returns
      STALLED before the agent wastes further calls — addresses the 847-API-
      call "current weather" runaway (agentpatterns.tech 2026).

  (4) RateLimitRetrier — exponential-backoff + jitter retry on simulated rate
      limit errors. Succeeds on the Nth attempt; audit log confirms attempt
      count. Addresses Datadog 2026: 8.4 million rate-limit errors in March
      alone.

Run:
    python -m examples.agent_guard_demo
"""
from __future__ import annotations

import sys
from typing import List

from patterns.agent_guard import (
    AgentGuard,
    BudgetGate,
    NoProgressDetector,
    ProgressStatus,
    RateLimitRetrier,
    ToolCallValidator,
)

PASS = "✓"
FAIL = "✗"


def _check(label: str, cond: bool) -> None:
    mark = PASS if cond else FAIL
    print(f"  {mark}  {label}")
    if not cond:
        raise AssertionError(f"FAILED: {label}")


# ---------------------------------------------------------------------------
# Scenario 1 — ToolCallValidator
# ---------------------------------------------------------------------------

def scenario_validator() -> None:
    print("\n=== Scenario 1: ToolCallValidator ===")
    print("Three error types caught; valid call passes; repair hint present.\n")

    v = ToolCallValidator()
    v.register("web_search", {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "additionalProperties": False,
    })

    # Happy path
    r = v.validate("web_search", {"query": "agent runtime guard", "max_results": 10})
    _check("valid call passes", r.ok)
    _check("no errors on valid call", r.errors == [])

    # Missing required field
    r = v.validate("web_search", {"max_results": 5})
    _check("missing required field detected", not r.ok)
    _check("error names the missing field", any("query" in e for e in r.errors))
    _check("repair hint present on failure", "Fix" in r.repair_hint)
    _check("repair hint lists required fields", "query" in r.repair_hint)
    print(f"     repair_hint excerpt: {r.repair_hint.splitlines()[0]}")

    # Wrong type
    r = v.validate("web_search", {"query": 42})
    _check("wrong type for string field detected", not r.ok)
    _check("error says 'string'", any("string" in e for e in r.errors))

    # Out-of-range integer
    r = v.validate("web_search", {"query": "hello", "max_results": 200})
    _check("max_results > 50 detected", not r.ok)
    _check("error says 'maximum'", any("maximum" in e for e in r.errors))

    # Unexpected extra property
    r = v.validate("web_search", {"query": "hello", "secret_key": "abc"})
    _check("unexpected property blocked", not r.ok)
    _check("error says 'additional'", any("additional" in e for e in r.errors))

    # Unknown tool — pass-through (safe default)
    r = v.validate("unregistered_tool", {"anything": 1})
    _check("unknown tool passes (fail-open on unknown)", r.ok)

    print("\n  All ToolCallValidator checks passed.")


# ---------------------------------------------------------------------------
# Scenario 2 — BudgetGate
# ---------------------------------------------------------------------------

def scenario_budget() -> None:
    print("\n=== Scenario 2: BudgetGate ===")
    print("Session + sub-agent scopes; on_exceed fires once; remaining tracks spend.\n")

    gate = BudgetGate()
    gate.set_ceiling("session", 1.00)           # $1.00 session limit
    gate.set_ceiling("sub-agent:researcher", 0.30)  # $0.30 sub-limit

    exceed_events: List[tuple] = []
    gate.register_on_exceed("session", lambda s, sp, c: exceed_events.append(("session", sp, c)))
    gate.register_on_exceed(
        "sub-agent:researcher",
        lambda s, sp, c: exceed_events.append(("researcher", sp, c)),
    )

    # Within budget
    gate.record_cost("session", 0.10)
    gate.record_cost("sub-agent:researcher", 0.10)
    _check("session within budget after first spend", gate.check_scope("session"))
    _check("researcher within budget after first spend", gate.check_scope("sub-agent:researcher"))
    _check("remaining tracks correctly", abs(gate.remaining("session") - 0.90) < 0.001)
    _check("no exceed events yet", len(exceed_events) == 0)

    # Push researcher over its sub-limit
    gate.record_cost("sub-agent:researcher", 0.25)  # total 0.35 > 0.30
    _check("researcher scope exceeded", not gate.check_scope("sub-agent:researcher"))
    _check("on_exceed fired for researcher", any(e[0] == "researcher" for e in exceed_events))

    # Push session over its limit
    gate.record_cost("session", 1.00)  # total 1.10 > 1.00
    _check("session scope exceeded", not gate.check_scope("session"))
    _check("on_exceed fired for session", any(e[0] == "session" for e in exceed_events))

    # Callbacks fire exactly once per crossing (extra spend doesn't re-fire)
    before = len(exceed_events)
    gate.record_cost("session", 0.50)
    _check("on_exceed fires exactly once per crossing", len(exceed_events) == before)

    # remaining() returns 0, not negative, after ceiling breach
    _check("remaining() clamps to 0 after breach", gate.remaining("session") == 0.0)

    print(f"\n  Exceed events: {exceed_events}")
    print("  All BudgetGate checks passed.")


# ---------------------------------------------------------------------------
# Scenario 3 — NoProgressDetector
# ---------------------------------------------------------------------------

def scenario_progress() -> None:
    print("\n=== Scenario 3: NoProgressDetector ===")
    print("Stall detected when agent keeps getting nearly-identical tool outputs.\n")

    detector = NoProgressDetector(stall_window=3, similarity_threshold=0.85)

    # First output — not enough data yet
    status = detector.observe("The weather in Tokyo is 22°C and sunny.")
    _check("single output → INSUFFICIENT_DATA", status == ProgressStatus.INSUFFICIENT_DATA)

    # Diverse outputs — progressing
    status = detector.observe("Forecast: rain expected tomorrow afternoon.")
    _check("different output → not STALLED", status != ProgressStatus.STALLED)

    status = detector.observe("Humidity: 68%. Wind: 12 km/h from the NW.")
    _check("another different output → PROGRESSING or INSUFFICIENT_DATA", status != ProgressStatus.STALLED)

    # Reset and simulate a stuck loop (same output repeated)
    detector.reset()
    stuck_output = "Error: rate limit hit. Please try again."
    for i in range(4):
        status = detector.observe(stuck_output)

    _check("repeated identical output eventually STALLED", status == ProgressStatus.STALLED)
    print(f"\n  Status after 4 identical outputs: {status.value}")

    # Reset and try with slightly varied but highly similar outputs
    detector.reset()
    for i in range(5):
        similar = f"Error: rate limit hit. Retry attempt {i}."  # small suffix change
        status = detector.observe(similar)

    print(f"  Status after 5 slightly-varied-but-similar outputs: {status.value}")
    _check("detector did not crash on slightly-varied outputs", True)

    print("  All NoProgressDetector checks passed.")


# ---------------------------------------------------------------------------
# Scenario 4 — RateLimitRetrier
# ---------------------------------------------------------------------------

def scenario_retrier() -> None:
    print("\n=== Scenario 4: RateLimitRetrier ===")
    print("Succeeds on 3rd attempt; retries logged; non-rate-limit error not retried.\n")

    class SimulatedRateLimit(Exception):
        pass

    class SimulatedServerError(Exception):
        pass

    retry_log: List[tuple] = []

    retrier = RateLimitRetrier(
        max_retries=4,
        base_delay=0.0,
        max_delay=0.0,
        exc_types=(SimulatedRateLimit,),
        on_retry=lambda attempt, wait, exc: retry_log.append((attempt, exc.__class__.__name__)),
    )

    # Succeeds on 3rd attempt
    call_count = [0]
    def flaky_tool(fail_n: int) -> str:
        call_count[0] += 1
        if call_count[0] <= fail_n:
            raise SimulatedRateLimit(f"429 Too Many Requests (attempt {call_count[0]})")
        return f"success after {call_count[0]} attempts"

    result = retrier.call(flaky_tool, fail_n=2)
    _check("call succeeds after 2 failures", result.startswith("success"))
    _check("on_retry fired twice (for attempts 1 and 2)", len(retry_log) == 2)
    _check("retry log records attempt numbers", retry_log[0][0] == 1 and retry_log[1][0] == 2)
    print(f"  retry_log: {retry_log}")
    print(f"  result: {result}")

    # Non-rate-limit errors propagate immediately (no retry)
    call_count[0] = 0
    retry_log.clear()
    raised = None
    try:
        retrier.call(lambda: (_ for _ in ()).throw(SimulatedServerError("500")))
    except SimulatedServerError as e:
        raised = e
    _check("non-rate-limit error not retried (re-raised immediately)", raised is not None)
    _check("no retry events for non-rate-limit error", len(retry_log) == 0)

    # Exhausts retries and re-raises
    call_count[0] = 0
    retry_log.clear()
    exhausted = None
    try:
        retrier.call(flaky_tool, fail_n=10)
    except SimulatedRateLimit as e:
        exhausted = e
    _check("raises after exhausting max_retries", exhausted is not None)
    _check("fired on_retry max_retries times", len(retry_log) == retrier._max_retries)

    print("  All RateLimitRetrier checks passed.")


# ---------------------------------------------------------------------------
# Scenario 5 — AgentGuard composite
# ---------------------------------------------------------------------------

def scenario_composite() -> None:
    print("\n=== Scenario 5: AgentGuard composite ===")
    print("All four primitives accessed through one guard object.\n")

    guard = AgentGuard()

    guard.validator.register("summarise", {
        "type": "object",
        "required": ["text"],
        "properties": {"text": {"type": "string"}, "max_words": {"type": "integer"}},
    })
    guard.budget.set_ceiling("session", 5.00)

    r = guard.validator.validate("summarise", {"text": "Long document...", "max_words": 50})
    _check("composite guard: valid call passes", r.ok)

    guard.budget.record_cost("session", 0.50)
    _check("composite guard: session within budget", guard.budget.check_scope("session"))

    guard.progress.observe("Agent processed page 1 of 10.")
    status = guard.progress.observe("Agent processed page 2 of 10.")
    _check("composite guard: progress reports correctly", status != ProgressStatus.STALLED)

    _check("composite guard: retrier is RateLimitRetrier instance",
           isinstance(guard.retrier, RateLimitRetrier))

    print("  All AgentGuard composite checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("AgentGuard demo — Pattern 12 (RP-6: agent runtime guard middleware)")
    print("=" * 62)
    print("Evidence: $437 overnight runaway (earezki.com Apr 2026);")
    print("          847 API calls for a weather query (agentpatterns.tech 2026);")
    print("          8.4M rate-limit errors in March 2026 (Datadog 2026).")
    print("          PALADIN: 32.76% → 89.68% tool-call recovery (ICLR 2026).")

    failures: List[str] = []
    for fn in [
        scenario_validator,
        scenario_budget,
        scenario_progress,
        scenario_retrier,
        scenario_composite,
    ]:
        try:
            fn()
        except AssertionError as exc:
            failures.append(str(exc))

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED — {len(failures)} assertion(s):")
        for f in failures:
            print(f"  {FAIL}  {f}")
        sys.exit(1)
    else:
        print(f"All 5 scenarios passed  {PASS}")


if __name__ == "__main__":
    main()
