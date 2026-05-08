"""Tests for patterns/agent_tx.py (Pattern 10 — SagaLog)."""

import threading
import unittest

from patterns.agent_tx import (
    CompensationEntry,
    MemorySagaBackend,
    SQLiteSagaBackend,
    SagaLog,
    compensable,
    run_saga,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_comp(result, *args, **kwargs):
    pass


def _make_entry(name="t", result=None):
    return CompensationEntry(name, (), {}, result, _noop_comp)


# ---------------------------------------------------------------------------
# SagaLog basics
# ---------------------------------------------------------------------------

class TestSagaLogBasics(unittest.TestCase):
    def test_empty_log_has_zero_length(self):
        self.assertEqual(len(SagaLog()), 0)

    def test_record_increments_length(self):
        saga = SagaLog()
        saga.record(_make_entry())
        self.assertEqual(len(saga), 1)

    def test_entries_returns_copy(self):
        saga = SagaLog()
        saga.record(_make_entry())
        snapshot = saga.entries()
        self.assertEqual(len(snapshot), 1)
        snapshot.clear()  # mutating the copy should not affect the log
        self.assertEqual(len(saga), 1)

    def test_clear_resets_length(self):
        saga = SagaLog()
        saga.record(_make_entry())
        saga.clear()
        self.assertEqual(len(saga), 0)

    def test_entries_preserves_insertion_order(self):
        saga = SagaLog()
        for name in ("a", "b", "c"):
            saga.record(_make_entry(name))
        names = [e.tool_name for e in saga.entries()]
        self.assertEqual(names, ["a", "b", "c"])


# ---------------------------------------------------------------------------
# SagaLog.compensate
# ---------------------------------------------------------------------------

class TestSagaLogCompensate(unittest.TestCase):
    def test_compensate_returns_tool_names(self):
        saga = SagaLog()
        called = []

        def comp(result, *args, **kwargs):
            called.append(result)

        saga.record(CompensationEntry("tool_a", (), {}, "a", comp))
        saga.record(CompensationEntry("tool_b", (), {}, "b", comp))
        names = saga.compensate()
        self.assertEqual(names, ["tool_b", "tool_a"])

    def test_compensate_reverse_order(self):
        saga = SagaLog()
        order = []
        for label in ("first", "second", "third"):
            saga.record(CompensationEntry(
                label, (), {}, None,
                lambda r, *a, lbl=label, **k: order.append(lbl)
            ))
        saga.compensate()
        self.assertEqual(order, ["third", "second", "first"])

    def test_compensate_receives_result_and_args(self):
        saga = SagaLog()
        received = {}

        def comp(result, x, y, z=None):
            received.update(result=result, x=x, y=y, z=z)

        saga.record(CompensationEntry("tool", (1, 2), {"z": 3}, "ret", comp))
        saga.compensate()
        self.assertEqual(received, {"result": "ret", "x": 1, "y": 2, "z": 3})

    def test_compensate_swallows_errors(self):
        saga = SagaLog()
        saga.record(CompensationEntry("bad", (), {}, None, lambda r, *a, **k: 1 / 0))
        names = saga.compensate()  # must not raise
        self.assertEqual(names, [])  # failed compensation not counted as successful

    def test_compensate_empty_log_returns_empty_list(self):
        self.assertEqual(SagaLog().compensate(), [])

    def test_compensate_partial_failure_continues(self):
        saga = SagaLog()
        called = []
        saga.record(CompensationEntry("ok_a", (), {}, None, lambda r, *a, **k: called.append("a")))
        saga.record(CompensationEntry("bad",  (), {}, None, lambda r, *a, **k: 1 / 0))
        saga.record(CompensationEntry("ok_b", (), {}, None, lambda r, *a, **k: called.append("b")))
        names = saga.compensate()
        # reverse order: ok_b, bad (fails), ok_a
        self.assertIn("b", called)
        self.assertIn("a", called)
        self.assertIn("ok_b", names)
        self.assertIn("ok_a", names)

    def test_compensate_does_not_clear_log(self):
        saga = SagaLog()
        saga.record(_make_entry())
        saga.compensate()
        self.assertEqual(len(saga), 1)  # entries still accessible after compensation


# ---------------------------------------------------------------------------
# @compensable decorator
# ---------------------------------------------------------------------------

class TestCompensableDecorator(unittest.TestCase):
    def test_return_value_passes_through(self):
        @compensable(_noop_comp, log=SagaLog())
        def add(x, y):
            return x + y

        self.assertEqual(add(2, 3), 5)

    def test_records_on_success(self):
        saga = SagaLog()

        @compensable(_noop_comp, log=saga)
        def noop(x):
            return x

        noop(42)
        self.assertEqual(len(saga), 1)

    def test_does_not_record_on_exception(self):
        saga = SagaLog()

        @compensable(_noop_comp, log=saga)
        def boom(x):
            raise ValueError("fail")

        with self.assertRaises(ValueError):
            boom(1)
        self.assertEqual(len(saga), 0)

    def test_no_log_does_not_record(self):
        @compensable(_noop_comp)
        def noop(x):
            return x

        result = noop(7)
        self.assertEqual(result, 7)  # still works; just nothing recorded

    def test_per_call_saga_overrides_default(self):
        default_saga = SagaLog()
        per_call_saga = SagaLog()

        @compensable(_noop_comp, log=default_saga)
        def noop(x):
            return x

        noop(1, _saga=per_call_saga)
        self.assertEqual(len(default_saga), 0)
        self.assertEqual(len(per_call_saga), 1)

    def test_compensation_fn_called_with_correct_args(self):
        saga = SagaLog()
        received = {}

        def comp(result, path, content, flag=False):
            received.update(result=result, path=path, content=content, flag=flag)

        @compensable(comp, log=saga)
        def write(path, content, flag=False):
            return f"wrote:{path}"

        write("a.txt", "hello", flag=True)
        saga.compensate()
        self.assertEqual(received, {
            "result": "wrote:a.txt",
            "path": "a.txt",
            "content": "hello",
            "flag": True,
        })

    def test_functools_wraps_preserves_metadata(self):
        @compensable(_noop_comp)
        def my_tool(x):
            """A tool docstring."""
            return x

        self.assertEqual(my_tool.__name__, "my_tool")
        self.assertEqual(my_tool.__doc__, "A tool docstring.")

    def test_multiple_calls_all_recorded(self):
        saga = SagaLog()

        @compensable(_noop_comp, log=saga)
        def step(n):
            return n

        step(1)
        step(2)
        step(3)
        self.assertEqual(len(saga), 3)

    def test_entry_stores_tool_name(self):
        saga = SagaLog()

        @compensable(_noop_comp, log=saga)
        def specific_tool(x):
            return x

        specific_tool(0)
        self.assertEqual(saga.entries()[0].tool_name, "specific_tool")

    def test_entry_stores_result(self):
        saga = SagaLog()

        @compensable(_noop_comp, log=saga)
        def produce(x):
            return x * 2

        produce(21)
        self.assertEqual(saga.entries()[0].result, 42)


# ---------------------------------------------------------------------------
# run_saga context manager
# ---------------------------------------------------------------------------

class TestRunSaga(unittest.TestCase):
    def test_no_exception_does_not_compensate(self):
        called = []
        saga = SagaLog()
        saga.record(CompensationEntry("t", (), {}, None, lambda r, *a, **k: called.append("c")))

        with run_saga(saga):
            pass  # no exception

        self.assertEqual(called, [])

    def test_exception_triggers_compensation(self):
        called = []
        saga = SagaLog()
        saga.record(CompensationEntry("t", (), {}, None, lambda r, *a, **k: called.append("c")))

        with self.assertRaises(RuntimeError):
            with run_saga(saga):
                raise RuntimeError("agent failed")

        self.assertEqual(called, ["c"])

    def test_exception_is_reraised(self):
        with self.assertRaises(ValueError):
            with run_saga(SagaLog()):
                raise ValueError("oops")

    def test_yields_saga_log_instance(self):
        with run_saga(SagaLog()) as saga:
            self.assertIsInstance(saga, SagaLog)

    def test_end_to_end_rollback(self):
        """3-step workflow fails after step 2; steps 1 and 2 are rolled back."""
        db: list = []
        files: list = []

        def undo_db_insert(result, record):
            db.remove(record)

        def undo_file_write(result, path, content):
            if path in files:
                files.remove(path)

        @compensable(undo_db_insert)
        def db_insert(record):
            db.append(record)
            return f"inserted:{record}"

        @compensable(undo_file_write)
        def file_write(path, content):
            files.append(path)
            return f"wrote:{path}"

        saga = SagaLog()
        with self.assertRaises(RuntimeError):
            with run_saga(saga):
                db_insert("user_42", _saga=saga)
                file_write("report.pdf", "data", _saga=saga)
                raise RuntimeError("validation failed after 2 steps")

        self.assertEqual(db, [])
        self.assertEqual(files, [])

    def test_partial_rollback_only_compensates_completed_steps(self):
        """Steps 1 and 2 complete; step 3 never called; only 1+2 are compensated."""
        executed = []
        compensated = []

        def comp(result, step_id):
            compensated.append(step_id)

        @compensable(comp)
        def do_step(step_id):
            executed.append(step_id)
            return step_id

        saga = SagaLog()
        with self.assertRaises(RuntimeError):
            with run_saga(saga):
                do_step(1, _saga=saga)
                do_step(2, _saga=saga)
                raise RuntimeError("step 3 not reached")

        self.assertEqual(executed, [1, 2])
        self.assertEqual(compensated, [2, 1])  # reverse order


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class TestMemorySagaBackend(unittest.TestCase):
    def test_append_and_len(self):
        b = MemorySagaBackend()
        b.append(_make_entry())
        self.assertEqual(len(b), 1)

    def test_snapshot_returns_copy(self):
        b = MemorySagaBackend()
        b.append(_make_entry("x"))
        snap = b.snapshot()
        snap.clear()
        self.assertEqual(len(b), 1)

    def test_clear(self):
        b = MemorySagaBackend()
        b.append(_make_entry())
        b.clear()
        self.assertEqual(len(b), 0)


class TestSQLiteSagaBackend(unittest.TestCase):
    def test_append_and_len(self):
        b = SQLiteSagaBackend()
        b.append(_make_entry("sql_tool"))
        self.assertEqual(len(b), 1)

    def test_recorded_tool_names(self):
        b = SQLiteSagaBackend()
        for name in ("alpha", "beta", "gamma"):
            b.append(_make_entry(name))
        self.assertEqual(b.recorded_tool_names(), ["alpha", "beta", "gamma"])

    def test_clear_removes_sqlite_rows(self):
        b = SQLiteSagaBackend()
        b.append(_make_entry())
        b.clear()
        self.assertEqual(b.recorded_tool_names(), [])
        self.assertEqual(len(b), 0)

    def test_saga_log_with_sqlite_backend(self):
        backend = SQLiteSagaBackend()
        saga = SagaLog(backend=backend)
        called = []

        @compensable(lambda r, x: called.append(x), log=saga)
        def tool(x):
            return x

        tool(99)
        saga.compensate()
        self.assertEqual(called, [99])
        self.assertEqual(backend.recorded_tool_names(), ["tool"])


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):
    def test_concurrent_records_all_registered(self):
        saga = SagaLog()
        threads = [
            threading.Thread(target=saga.record, args=(_make_entry(f"t{i}"),))
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(saga), 50)


if __name__ == "__main__":
    unittest.main()
