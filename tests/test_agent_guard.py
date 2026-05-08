"""Tests for patterns/agent_guard.py (Pattern 12 — AgentGuard, RP-6)."""
import threading
import unittest

from patterns.agent_guard import (
    AgentGuard,
    BudgetGate,
    NoProgressDetector,
    ProgressStatus,
    RateLimitRetrier,
    ToolCallValidator,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# ToolCallValidator — basic schema checks
# ---------------------------------------------------------------------------

class TestToolCallValidatorBasic(unittest.TestCase):
    def setUp(self):
        self.v = ToolCallValidator()
        self.v.register("search", {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "additionalProperties": False,
        })

    def test_valid_args_passes(self):
        r = self.v.validate("search", {"query": "hello", "limit": 10})
        self.assertTrue(r.ok)
        self.assertEqual(r.errors, [])

    def test_optional_field_absent_passes(self):
        r = self.v.validate("search", {"query": "hello"})
        self.assertTrue(r.ok)

    def test_missing_required_field(self):
        r = self.v.validate("search", {"limit": 5})
        self.assertFalse(r.ok)
        self.assertTrue(any("query" in e for e in r.errors))

    def test_wrong_type_for_string(self):
        r = self.v.validate("search", {"query": 123})
        self.assertFalse(r.ok)
        self.assertTrue(any("string" in e for e in r.errors))

    def test_wrong_type_for_integer(self):
        r = self.v.validate("search", {"query": "q", "limit": "ten"})
        self.assertFalse(r.ok)
        self.assertTrue(any("integer" in e for e in r.errors))

    def test_minimum_violation(self):
        r = self.v.validate("search", {"query": "q", "limit": 0})
        self.assertFalse(r.ok)
        self.assertTrue(any("minimum" in e for e in r.errors))

    def test_maximum_violation(self):
        r = self.v.validate("search", {"query": "q", "limit": 200})
        self.assertFalse(r.ok)
        self.assertTrue(any("maximum" in e for e in r.errors))

    def test_additional_property_blocked(self):
        r = self.v.validate("search", {"query": "q", "extra": "x"})
        self.assertFalse(r.ok)
        self.assertTrue(any("additional" in e for e in r.errors))

    def test_unknown_tool_always_passes(self):
        r = self.v.validate("unknown_tool", {"anything": 1})
        self.assertTrue(r.ok)

    def test_repair_hint_present_on_failure(self):
        r = self.v.validate("search", {})
        self.assertFalse(r.ok)
        self.assertIn("Fix", r.repair_hint)
        self.assertIn("query", r.repair_hint)

    def test_repair_hint_empty_on_success(self):
        r = self.v.validate("search", {"query": "q"})
        self.assertTrue(r.ok)
        self.assertEqual(r.repair_hint, "")

    def test_multiple_errors_collected(self):
        r = self.v.validate("search", {"query": 1, "limit": 0, "extra": "x"})
        self.assertFalse(r.ok)
        self.assertGreaterEqual(len(r.errors), 2)


# ---------------------------------------------------------------------------
# ToolCallValidator — type edge cases
# ---------------------------------------------------------------------------

class TestToolCallValidatorTypes(unittest.TestCase):
    def _make(self, schema):
        v = ToolCallValidator()
        v.register("t", schema)
        return v

    def test_boolean_accepted(self):
        v = self._make({"properties": {"flag": {"type": "boolean"}}})
        self.assertTrue(v.validate("t", {"flag": True}).ok)

    def test_bool_rejected_for_integer(self):
        v = self._make({"properties": {"n": {"type": "integer"}}})
        self.assertFalse(v.validate("t", {"n": True}).ok)

    def test_float_accepted_for_number(self):
        v = self._make({"properties": {"x": {"type": "number"}}})
        self.assertTrue(v.validate("t", {"x": 3.14}).ok)

    def test_int_accepted_for_number(self):
        v = self._make({"properties": {"x": {"type": "number"}}})
        self.assertTrue(v.validate("t", {"x": 3}).ok)

    def test_bool_rejected_for_number(self):
        v = self._make({"properties": {"x": {"type": "number"}}})
        self.assertFalse(v.validate("t", {"x": False}).ok)

    def test_enum_valid_value_passes(self):
        v = self._make({"properties": {"mode": {"enum": ["read", "write"]}}})
        self.assertTrue(v.validate("t", {"mode": "read"}).ok)

    def test_enum_invalid_value_fails(self):
        v = self._make({"properties": {"mode": {"enum": ["read", "write"]}}})
        r = v.validate("t", {"mode": "delete"})
        self.assertFalse(r.ok)
        self.assertTrue(any("enum" in e for e in r.errors))

    def test_min_length_violation(self):
        v = self._make({"properties": {"s": {"type": "string", "minLength": 3}}})
        self.assertFalse(v.validate("t", {"s": "ab"}).ok)

    def test_max_length_violation(self):
        v = self._make({"properties": {"s": {"type": "string", "maxLength": 5}}})
        self.assertFalse(v.validate("t", {"s": "toolong"}).ok)

    def test_array_items_type_ok(self):
        v = self._make({"properties": {"ids": {"type": "array", "items": {"type": "integer"}}}})
        self.assertTrue(v.validate("t", {"ids": [1, 2, 3]}).ok)

    def test_array_items_type_violation(self):
        v = self._make({"properties": {"ids": {"type": "array", "items": {"type": "integer"}}}})
        r = v.validate("t", {"ids": [1, 2, "three"]})
        self.assertFalse(r.ok)
        self.assertTrue(any("[2]" in e for e in r.errors))

    def test_null_type_accepted(self):
        v = self._make({"properties": {"v": {"type": "null"}}})
        self.assertTrue(v.validate("t", {"v": None}).ok)

    def test_null_type_rejected_for_string(self):
        v = self._make({"properties": {"v": {"type": "string"}}})
        self.assertFalse(v.validate("t", {"v": None}).ok)


# ---------------------------------------------------------------------------
# BudgetGate
# ---------------------------------------------------------------------------

class TestBudgetGate(unittest.TestCase):
    def test_under_ceiling_returns_true(self):
        bg = BudgetGate()
        bg.set_ceiling("session", 10.0)
        bg.record_cost("session", 5.0)
        self.assertTrue(bg.check_scope("session"))

    def test_at_ceiling_returns_true(self):
        bg = BudgetGate()
        bg.set_ceiling("session", 10.0)
        bg.record_cost("session", 10.0)
        self.assertTrue(bg.check_scope("session"))

    def test_over_ceiling_returns_false(self):
        bg = BudgetGate()
        bg.set_ceiling("session", 10.0)
        bg.record_cost("session", 10.01)
        self.assertFalse(bg.check_scope("session"))

    def test_no_ceiling_always_true(self):
        bg = BudgetGate()
        bg.record_cost("session", 1000.0)
        self.assertTrue(bg.check_scope("session"))

    def test_callback_fires_on_exceed(self):
        bg = BudgetGate()
        bg.set_ceiling("agent:w1", 5.0)
        fired = []
        bg.register_on_exceed("agent:w1", lambda s, sp, c: fired.append((s, sp, c)))
        bg.record_cost("agent:w1", 3.0)
        self.assertEqual(fired, [])
        bg.record_cost("agent:w1", 3.0)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], "agent:w1")
        self.assertGreater(fired[0][1], 5.0)

    def test_callback_fires_exactly_once(self):
        bg = BudgetGate()
        bg.set_ceiling("s", 1.0)
        count = []
        bg.register_on_exceed("s", lambda *a: count.append(1))
        bg.record_cost("s", 2.0)  # crosses
        bg.record_cost("s", 2.0)  # already over; prev > ceiling
        self.assertEqual(len(count), 1)

    def test_remaining_under_ceiling(self):
        bg = BudgetGate()
        bg.set_ceiling("s", 10.0)
        bg.record_cost("s", 3.0)
        self.assertAlmostEqual(bg.remaining("s"), 7.0)

    def test_remaining_no_ceiling_is_none(self):
        bg = BudgetGate()
        self.assertIsNone(bg.remaining("s"))

    def test_remaining_zero_when_exceeded(self):
        bg = BudgetGate()
        bg.set_ceiling("s", 5.0)
        bg.record_cost("s", 10.0)
        self.assertEqual(bg.remaining("s"), 0.0)

    def test_spent_accumulates(self):
        bg = BudgetGate()
        bg.record_cost("x", 1.5)
        bg.record_cost("x", 2.5)
        self.assertAlmostEqual(bg.spent("x"), 4.0)

    def test_multiple_scopes_independent(self):
        bg = BudgetGate()
        bg.set_ceiling("a", 5.0)
        bg.set_ceiling("b", 5.0)
        bg.record_cost("a", 6.0)
        self.assertFalse(bg.check_scope("a"))
        self.assertTrue(bg.check_scope("b"))

    def test_thread_safe_concurrent_record(self):
        bg = BudgetGate()
        bg.set_ceiling("s", 100_000.0)
        errors = []

        def add():
            for _ in range(100):
                try:
                    bg.record_cost("s", 1.0)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=add) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertAlmostEqual(bg.spent("s"), 1000.0)


# ---------------------------------------------------------------------------
# NoProgressDetector
# ---------------------------------------------------------------------------

class TestNoProgressDetector(unittest.TestCase):
    def test_first_observation_insufficient(self):
        d = NoProgressDetector()
        self.assertEqual(d.observe("hello"), ProgressStatus.INSUFFICIENT_DATA)

    def test_two_observations_insufficient_when_window_is_3(self):
        d = NoProgressDetector(stall_window=3)
        d.observe("alpha")
        self.assertEqual(d.observe("beta"), ProgressStatus.INSUFFICIENT_DATA)

    def test_different_outputs_progressing_after_window(self):
        d = NoProgressDetector(stall_window=3)
        d.observe("alpha")
        d.observe("beta")
        d.observe("gamma")
        status = d.observe("delta")
        self.assertEqual(status, ProgressStatus.PROGRESSING)

    def test_identical_outputs_trigger_stalled(self):
        d = NoProgressDetector(stall_window=3, similarity_threshold=0.9)
        txt = "the result is exactly the same output as before"
        d.observe(txt)
        d.observe(txt)
        d.observe(txt)
        status = d.observe(txt)
        self.assertEqual(status, ProgressStatus.STALLED)

    def test_near_identical_triggers_stalled(self):
        d = NoProgressDetector(stall_window=2, similarity_threshold=0.8, ngram_n=3)
        d.observe("search returned no results found in database")
        d.observe("search returned no results found in database.")
        status = d.observe("search returned no results found in database..")
        self.assertEqual(status, ProgressStatus.STALLED)

    def test_reset_clears_history_and_stall_run(self):
        d = NoProgressDetector(stall_window=2, similarity_threshold=0.9)
        txt = "identical output text here"
        for _ in range(4):
            d.observe(txt)
        d.reset()
        self.assertEqual(d.observe(txt), ProgressStatus.INSUFFICIENT_DATA)

    def test_stall_window_one(self):
        d = NoProgressDetector(stall_window=1, similarity_threshold=0.9)
        d.observe("aaa bbb ccc ddd eee fff ggg hhh")
        status = d.observe("aaa bbb ccc ddd eee fff ggg hhh")
        self.assertEqual(status, ProgressStatus.STALLED)

    def test_recovery_resets_stall_run(self):
        d = NoProgressDetector(stall_window=3, similarity_threshold=0.9)
        txt = "same output every single time, no change"
        d.observe(txt)
        d.observe(txt)
        d.observe(txt)
        d.observe(txt)  # STALLED — stall_run reaches 3
        # new different output breaks the stall run
        d.observe("completely different result came back from tool")
        d.observe("another varied result that differs significantly")
        # one more near-identical: stall_run resets to 1, not yet ≥ 3
        status = d.observe("another varied result that differs significantly!")
        self.assertNotEqual(status, ProgressStatus.STALLED)

    def test_empty_strings_treated_as_similar(self):
        d = NoProgressDetector(stall_window=1, similarity_threshold=0.9)
        d.observe("")
        status = d.observe("")
        self.assertEqual(status, ProgressStatus.STALLED)

    def test_invalid_stall_window_raises(self):
        with self.assertRaises(ValueError):
            NoProgressDetector(stall_window=0)


# ---------------------------------------------------------------------------
# RateLimitRetrier
# ---------------------------------------------------------------------------

class TestRateLimitRetrier(unittest.TestCase):
    def test_success_on_first_call(self):
        r = RateLimitRetrier()
        self.assertEqual(r.call(lambda: 42), 42)

    def test_retries_on_matching_exception(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("rate limited")
            return "ok"

        r = RateLimitRetrier(max_retries=4, base_delay=0.0, exc_types=(ValueError,))
        self.assertEqual(r.call(flaky), "ok")
        self.assertEqual(len(attempts), 3)

    def test_raises_after_max_retries_exhausted(self):
        attempts = []

        def always_fail():
            attempts.append(1)
            raise ValueError("rate limited")

        r = RateLimitRetrier(max_retries=2, base_delay=0.0, exc_types=(ValueError,))
        with self.assertRaises(ValueError):
            r.call(always_fail)
        self.assertEqual(len(attempts), 3)  # initial + 2 retries

    def test_non_matching_exception_not_retried(self):
        attempts = []

        def fail_with_type_error():
            attempts.append(1)
            raise TypeError("wrong type")

        r = RateLimitRetrier(max_retries=4, base_delay=0.0, exc_types=(ValueError,))
        with self.assertRaises(TypeError):
            r.call(fail_with_type_error)
        self.assertEqual(len(attempts), 1)

    def test_on_retry_callback_invoked(self):
        retries = []

        def flaky():
            if len(retries) < 2:
                raise ValueError()
            return "done"

        r = RateLimitRetrier(
            max_retries=3, base_delay=0.0,
            exc_types=(ValueError,),
            on_retry=lambda attempt, wait, exc: retries.append(attempt),
        )
        r.call(flaky)
        self.assertEqual(retries, [1, 2])

    def test_passes_positional_and_keyword_args(self):
        def add(a, b=0):
            return a + b

        r = RateLimitRetrier()
        self.assertEqual(r.call(add, 3, b=4), 7)

    def test_is_rate_limit_error_override(self):
        class Http429Error(Exception):
            pass

        class StrictRetrier(RateLimitRetrier):
            def is_rate_limit_error(self, exc):
                return isinstance(exc, Http429Error)

        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise Http429Error("429")
            return "done"

        r = StrictRetrier(max_retries=4, base_delay=0.0)
        self.assertEqual(r.call(flaky), "done")
        self.assertEqual(len(attempts), 3)


# ---------------------------------------------------------------------------
# AgentGuard — composite
# ---------------------------------------------------------------------------

class TestAgentGuard(unittest.TestCase):
    def test_default_construction_creates_all_components(self):
        g = AgentGuard()
        self.assertIsInstance(g.validator, ToolCallValidator)
        self.assertIsInstance(g.budget, BudgetGate)
        self.assertIsInstance(g.progress, NoProgressDetector)
        self.assertIsInstance(g.retrier, RateLimitRetrier)

    def test_custom_components_injected_by_identity(self):
        v, b, p, r = ToolCallValidator(), BudgetGate(), NoProgressDetector(), RateLimitRetrier()
        g = AgentGuard(validator=v, budget=b, progress=p, retrier=r)
        self.assertIs(g.validator, v)
        self.assertIs(g.budget, b)
        self.assertIs(g.progress, p)
        self.assertIs(g.retrier, r)

    def test_composite_validation_and_budget_workflow(self):
        g = AgentGuard()
        g.validator.register("fetch", {
            "required": ["url"],
            "properties": {"url": {"type": "string"}},
        })
        g.budget.set_ceiling("session", 1.0)

        r = g.validator.validate("fetch", {"url": "https://example.com"})
        self.assertTrue(r.ok)

        r_bad = g.validator.validate("fetch", {"url": 999})
        self.assertFalse(r_bad.ok)

        g.budget.record_cost("session", 0.5)
        self.assertTrue(g.budget.check_scope("session"))
        g.budget.record_cost("session", 0.6)
        self.assertFalse(g.budget.check_scope("session"))

    def test_composite_progress_observation(self):
        g = AgentGuard()
        self.assertEqual(g.progress.observe("first"), ProgressStatus.INSUFFICIENT_DATA)

    def test_composite_retrier_success(self):
        g = AgentGuard()
        self.assertEqual(g.retrier.call(lambda: "result"), "result")


if __name__ == "__main__":
    unittest.main()
