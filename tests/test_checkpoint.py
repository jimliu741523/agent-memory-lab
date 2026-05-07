"""
Tests for patterns/checkpoint.py (RP-1: checkpoint / pause / resume).

Run from repo root:
    python -m unittest discover tests -v
"""
import os
import tempfile
import unittest

from patterns.checkpoint import (
    CheckpointManager,
    FileBackend,
    MemoryBackend,
    RunState,
    _token,
)


class TestRunStateSerialization(unittest.TestCase):
    def _run(self, **kw) -> RunState:
        return RunState(run_id="r1", step=0, **kw)

    def test_round_trip_empty_fields(self):
        run = self._run()
        restored = RunState.from_json(run.to_json())
        self.assertEqual(run.run_id, restored.run_id)
        self.assertEqual(run.step, restored.step)
        self.assertEqual(run.tool_queue, restored.tool_queue)
        self.assertEqual(run.created_at, restored.created_at)

    def test_round_trip_full_fields(self):
        run = RunState(
            run_id="run-abc",
            step=7,
            tool_queue=[{"name": "search", "args": {"q": "test"}}],
            pending=[{"name": "fetch", "args": {"url": "http://x"}}],
            memory_snap=[{"role": "user", "content": "hello"}],
            stream_cursor=42,
            metadata={"framework": "raw"},
        )
        restored = RunState.from_json(run.to_json())
        self.assertEqual(restored.step, 7)
        self.assertEqual(restored.tool_queue, run.tool_queue)
        self.assertEqual(restored.pending, run.pending)
        self.assertEqual(restored.memory_snap, run.memory_snap)
        self.assertEqual(restored.stream_cursor, 42)
        self.assertEqual(restored.metadata, {"framework": "raw"})

    def test_canonical_json_is_sorted(self):
        raw = self._run().to_json()
        # Canonical JSON must not start with a field after 'c' to confirm sort
        self.assertTrue(raw.startswith('{"created_at"'))

    def test_token_is_deterministic(self):
        ts = "2026-01-01T00:00:00+00:00"
        run = RunState(run_id="x", step=0, created_at=ts)
        run2 = RunState.from_json(run.to_json())
        self.assertEqual(_token(run), _token(run2))


class TestMemoryBackend(unittest.TestCase):
    def _mgr(self):
        return CheckpointManager(backend=MemoryBackend())

    def test_save_returns_token(self):
        mgr = self._mgr()
        token = mgr.save(RunState(run_id="r", step=0))
        self.assertIsInstance(token, str)
        self.assertEqual(len(token), 64)  # SHA-256 hex

    def test_load_recovers_identical_state(self):
        mgr = self._mgr()
        run = RunState(
            run_id="run-42",
            step=3,
            tool_queue=[{"name": "tool_a", "args": {}}],
        )
        token = mgr.save(run)
        restored = mgr.load(token)
        self.assertEqual(restored.run_id, "run-42")
        self.assertEqual(restored.step, 3)
        self.assertEqual(restored.tool_queue, [{"name": "tool_a", "args": {}}])

    def test_load_unknown_token_raises_key_error(self):
        mgr = self._mgr()
        with self.assertRaises(KeyError):
            mgr.load("a" * 64)

    def test_same_state_same_token_idempotent(self):
        mgr = self._mgr()
        run = RunState(run_id="idem", step=1, created_at="2026-01-01T00:00:00+00:00")
        t1 = mgr.save(run)
        t2 = mgr.save(RunState.from_json(run.to_json()))
        self.assertEqual(t1, t2)

    def test_different_step_different_token(self):
        mgr = self._mgr()
        base_ts = "2026-01-01T00:00:00+00:00"
        t1 = mgr.save(RunState(run_id="r", step=0, created_at=base_ts))
        t2 = mgr.save(RunState(run_id="r", step=1, created_at=base_ts))
        self.assertNotEqual(t1, t2)

    def test_advance_increments_step(self):
        mgr = self._mgr()
        run = RunState(run_id="adv", step=5)
        token = mgr.save(run)
        token2 = mgr.advance(token, step=6)
        restored = mgr.load(token2)
        self.assertEqual(restored.step, 6)

    def test_advance_preserves_other_fields(self):
        mgr = self._mgr()
        run = RunState(run_id="adv2", step=0, metadata={"x": 1})
        token = mgr.save(run)
        token2 = mgr.advance(token, step=1)
        restored = mgr.load(token2)
        self.assertEqual(restored.run_id, "adv2")
        self.assertEqual(restored.metadata, {"x": 1})


class TestFileBackend(unittest.TestCase):
    def test_save_and_load_via_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(backend=FileBackend(tmpdir))
            run = RunState(run_id="file-run", step=2)
            token = mgr.save(run)
            # Confirm file exists on disk
            expected = os.path.join(tmpdir, f"{token}.json")
            self.assertTrue(os.path.exists(expected))
            # Round-trip
            restored = mgr.load(token)
            self.assertEqual(restored.run_id, "file-run")
            self.assertEqual(restored.step, 2)

    def test_file_backend_missing_token_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(backend=FileBackend(tmpdir))
            with self.assertRaises(KeyError):
                mgr.load("b" * 64)

    def test_file_backend_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "checkpoints", "v1")
            mgr = CheckpointManager(backend=FileBackend(nested))
            run = RunState(run_id="nested", step=0)
            mgr.save(run)
            self.assertTrue(os.path.isdir(nested))


if __name__ == "__main__":
    unittest.main()
