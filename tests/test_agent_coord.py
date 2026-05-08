"""Tests for Pattern 11 — AgentCoord (RP-5)."""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from patterns.agent_coord import (
    Barrier,
    DriftMonitor,
    EventBus,
    FileLock,
    WorkQueue,
)


# ---------------------------------------------------------------------------
# FileLock
# ---------------------------------------------------------------------------

class TestFileLock(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.target = Path(self.tmpdir) / "resource.txt"

    def test_acquire_and_release(self):
        fl = FileLock(self.target, lease_secs=10)
        self.assertTrue(fl.acquire("a1"))
        self.assertFalse(fl.acquire("a2"))
        fl.release("a1")
        self.assertTrue(fl.acquire("a2"))

    def test_double_acquire_same_owner_blocked(self):
        fl = FileLock(self.target, lease_secs=10)
        fl.acquire("a1")
        self.assertFalse(fl.acquire("a1"))  # already held

    def test_stale_lease_allows_reacquire(self):
        fl = FileLock(self.target, lease_secs=0.05)
        self.assertTrue(fl.acquire("a1"))
        time.sleep(0.1)
        self.assertTrue(fl.is_stale())
        self.assertTrue(fl.acquire("a2"))

    def test_fresh_lock_not_stale(self):
        fl = FileLock(self.target, lease_secs=10)
        fl.acquire("a1")
        self.assertFalse(fl.is_stale())

    def test_no_lock_file_not_stale(self):
        fl = FileLock(self.target, lease_secs=10)
        self.assertFalse(fl.is_stale())

    def test_context_manager_acquires_and_releases(self):
        fl = FileLock(self.target, lease_secs=10)
        with fl.locked("a1"):
            self.assertFalse(fl.acquire("a2"))
        self.assertTrue(fl.acquire("a3"))

    def test_context_manager_releases_on_exception(self):
        fl = FileLock(self.target, lease_secs=10)
        try:
            with fl.locked("a1"):
                raise ValueError("boom")
        except ValueError:
            pass
        self.assertTrue(fl.acquire("a2"))

    def test_context_manager_timeout(self):
        fl = FileLock(self.target, lease_secs=10)
        fl.acquire("a1")
        with self.assertRaises(TimeoutError):
            with fl.locked("a2", timeout=0.1):
                pass

    def test_release_wrong_owner_no_effect(self):
        fl = FileLock(self.target, lease_secs=10)
        fl.acquire("a1")
        fl.release("other")  # different owner — should not remove
        # Lock should still be held
        # Note: our impl only releases if owner matches OR owner is ""
        # "other" != "a1", so lock remains
        self.assertFalse(fl.acquire("a2"))

    def test_release_empty_owner_force_release(self):
        fl = FileLock(self.target, lease_secs=10)
        fl.acquire("a1")
        fl.release("")   # empty string = force release
        self.assertTrue(fl.acquire("a2"))


# ---------------------------------------------------------------------------
# WorkQueue
# ---------------------------------------------------------------------------

class TestWorkQueue(unittest.TestCase):
    def test_put_and_claim(self):
        q = WorkQueue()
        q.put("t1", {"x": 1})
        task = q.claim("w1")
        self.assertIsNotNone(task)
        self.assertEqual(task.id, "t1")
        self.assertEqual(task.payload, {"x": 1})
        self.assertEqual(task.status, "claimed")

    def test_empty_queue_returns_none(self):
        q = WorkQueue()
        self.assertIsNone(q.claim("w1"))

    def test_two_tasks_two_workers(self):
        q = WorkQueue()
        q.put("t1", "a")
        q.put("t2", "b")
        t1 = q.claim("w1")
        t2 = q.claim("w2")
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)
        self.assertNotEqual(t1.id, t2.id)
        self.assertIsNone(q.claim("w3"))

    def test_release_returns_to_pending(self):
        q = WorkQueue()
        q.put("t1", "x")
        t = q.claim("w1")
        q.release(t.id)
        reclaimed = q.claim("w2")
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.id, "t1")

    def test_release_done(self):
        q = WorkQueue()
        q.put("t1", "x")
        t = q.claim("w1")
        q.release(t.id, done=True)
        self.assertEqual(q.stats().get("done", 0), 1)
        self.assertIsNone(q.claim("w2"))

    def test_release_failed(self):
        q = WorkQueue()
        q.put("t1", "x")
        t = q.claim("w1")
        q.release(t.id, failed=True)
        self.assertEqual(q.stats().get("failed", 0), 1)

    def test_stale_lease_auto_reclaimed(self):
        q = WorkQueue()
        q.put("t1", "x")
        q.claim("w1", lease_secs=0.05)
        time.sleep(0.15)
        recovered = q.claim("w2", lease_secs=60)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.id, "t1")

    def test_idempotent_put(self):
        q = WorkQueue()
        q.put("t1", "a")
        q.put("t1", "b")  # same id — second ignored
        q.claim("w1")
        self.assertIsNone(q.claim("w2"))

    def test_stats_pending_done(self):
        q = WorkQueue()
        q.put("t1", "a")
        q.put("t2", "b")
        q.put("t3", "c")
        t = q.claim("w1")
        q.release(t.id, done=True)
        s = q.stats()
        self.assertEqual(s.get("pending", 0), 2)
        self.assertEqual(s.get("done", 0), 1)

    def test_sqlite_file_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            q = WorkQueue(db_path)
            q.put("t1", {"key": "val"})
            # New WorkQueue object pointing to same file
            q2 = WorkQueue(db_path)
            task = q2.claim("w-after-restart")
            self.assertIsNotNone(task)
            self.assertEqual(task.payload, {"key": "val"})
        finally:
            os.unlink(db_path)

    def test_concurrent_claim_no_double_assign(self):
        q = WorkQueue()
        for i in range(10):
            q.put(f"t{i}", i)
        claimed_ids = []
        lock = threading.Lock()

        def worker():
            task = q.claim("w", lease_secs=60)
            if task:
                with lock:
                    claimed_ids.append(task.id)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(claimed_ids), len(set(claimed_ids)))  # no duplicates


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class TestEventBus(unittest.TestCase):
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("done", lambda e, p: received.append(p))
        bus.publish("done", {"agent": "a1"})
        self.assertEqual(received, [{"agent": "a1"}])

    def test_no_subscribers_returns_zero(self):
        bus = EventBus()
        self.assertEqual(bus.publish("event", "x"), 0)

    def test_returns_handler_count(self):
        bus = EventBus()
        bus.subscribe("e", lambda e, p: None)
        bus.subscribe("e", lambda e, p: None)
        self.assertEqual(bus.publish("e", None), 2)

    def test_multiple_subscribers_all_called(self):
        bus = EventBus()
        log = []
        bus.subscribe("e", lambda e, p: log.append("h1"))
        bus.subscribe("e", lambda e, p: log.append("h2"))
        bus.publish("e", None)
        self.assertIn("h1", log)
        self.assertIn("h2", log)

    def test_unsubscribe_removes_handler(self):
        bus = EventBus()
        log = []
        def h(e, p): log.append(p)
        bus.subscribe("e", h)
        bus.unsubscribe("e", h)
        bus.publish("e", 99)
        self.assertEqual(log, [])

    def test_unsubscribe_returns_false_if_not_found(self):
        bus = EventBus()
        def h(e, p): pass
        self.assertFalse(bus.unsubscribe("e", h))

    def test_different_events_isolated(self):
        bus = EventBus()
        log = []
        bus.subscribe("a", lambda e, p: log.append("a"))
        bus.publish("b", None)
        self.assertEqual(log, [])

    def test_thread_safe_concurrent_publish(self):
        bus = EventBus()
        counts = [0]
        lock = threading.Lock()
        def h(e, p):
            with lock:
                counts[0] += 1
        bus.subscribe("tick", h)
        threads = [threading.Thread(target=lambda: bus.publish("tick", None)) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(counts[0], 20)

    def test_payload_none_allowed(self):
        bus = EventBus()
        received = []
        bus.subscribe("e", lambda e, p: received.append(p))
        bus.publish("e")
        self.assertEqual(received, [None])


# ---------------------------------------------------------------------------
# Barrier
# ---------------------------------------------------------------------------

class TestBarrier(unittest.TestCase):
    def test_two_agents_both_succeed(self):
        barrier = Barrier(2)
        results = []

        def agent(delay):
            time.sleep(delay)
            results.append(barrier.wait(timeout=2.0))

        t1 = threading.Thread(target=agent, args=(0.0,))
        t2 = threading.Thread(target=agent, args=(0.05,))
        t1.start(); t2.start()
        t1.join(); t2.join()
        self.assertEqual(results, [True, True])

    def test_three_agents_all_succeed(self):
        barrier = Barrier(3)
        results = []
        lock = threading.Lock()

        def agent():
            ok = barrier.wait(timeout=2.0)
            with lock:
                results.append(ok)

        threads = [threading.Thread(target=agent) for _ in range(3)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(sorted(results), [True, True, True])

    def test_timeout_returns_false(self):
        barrier = Barrier(3)
        result = barrier.wait(timeout=0.1)
        self.assertFalse(result)

    def test_timeout_does_not_corrupt_barrier(self):
        barrier = Barrier(2)
        # First call times out
        r1 = barrier.wait(timeout=0.05)
        self.assertFalse(r1)
        # Subsequent two calls should still succeed
        results = []
        t1 = threading.Thread(target=lambda: results.append(barrier.wait(timeout=2.0)))
        t2 = threading.Thread(target=lambda: results.append(barrier.wait(timeout=2.0)))
        t1.start(); t2.start()
        t1.join(); t2.join()
        self.assertEqual(results, [True, True])

    def test_reusable_across_phases(self):
        barrier = Barrier(2)
        phase1 = []
        phase2 = []

        def agent():
            barrier.wait(timeout=2.0)
            phase1.append(1)
            barrier.wait(timeout=2.0)
            phase2.append(1)

        t1 = threading.Thread(target=agent)
        t2 = threading.Thread(target=agent)
        t1.start(); t2.start()
        t1.join(); t2.join()
        self.assertEqual(len(phase1), 2)
        self.assertEqual(len(phase2), 2)


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class TestDriftMonitor(unittest.TestCase):
    def test_first_record_returns_none(self):
        dm = DriftMonitor()
        self.assertIsNone(dm.record("a1", 0, [1.0, 0.0]))

    def test_identical_embeddings_zero_drift(self):
        dm = DriftMonitor()
        dm.record("a1", 0, [1.0, 0.0])
        delta = dm.record("a1", 1, [1.0, 0.0])
        self.assertAlmostEqual(delta, 0.0, places=5)

    def test_orthogonal_embeddings_max_drift(self):
        dm = DriftMonitor()
        dm.record("a1", 0, [1.0, 0.0])
        delta = dm.record("a1", 1, [0.0, 1.0])
        self.assertAlmostEqual(delta, 1.0, places=5)

    def test_opposite_embeddings_max_drift(self):
        dm = DriftMonitor()
        dm.record("a1", 0, [1.0, 0.0])
        delta = dm.record("a1", 1, [-1.0, 0.0])
        self.assertAlmostEqual(delta, 2.0, places=5)

    def test_on_drift_callback_fires_above_threshold(self):
        fired = []
        dm = DriftMonitor(threshold=0.3, on_drift=lambda aid, d: fired.append((aid, d)))
        dm.record("a1", 0, [1.0, 0.0])
        dm.record("a1", 1, [0.0, 1.0])  # delta = 1.0 > 0.3
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], "a1")
        self.assertAlmostEqual(fired[0][1], 1.0, places=5)

    def test_on_drift_not_fired_below_threshold(self):
        fired = []
        dm = DriftMonitor(threshold=0.5, on_drift=lambda a, d: fired.append(d))
        dm.record("a1", 0, [1.0, 0.0])
        dm.record("a1", 1, [0.98, 0.1])  # small drift
        self.assertEqual(fired, [])

    def test_drift_history_two_deltas(self):
        dm = DriftMonitor(threshold=1.5)  # high threshold, no callback
        dm.record("a1", 0, [1.0, 0.0])
        dm.record("a1", 1, [0.0, 1.0])  # delta ≈ 1.0
        dm.record("a1", 2, [1.0, 0.0])  # delta ≈ 1.0
        history = dm.drift_history("a1")
        self.assertEqual(len(history), 2)
        self.assertAlmostEqual(history[0], 1.0, places=5)
        self.assertAlmostEqual(history[1], 1.0, places=5)

    def test_multiple_agents_independent(self):
        dm = DriftMonitor(threshold=0.3)
        dm.record("a1", 0, [1.0, 0.0])
        dm.record("a2", 0, [0.0, 1.0])
        d1 = dm.record("a1", 1, [0.0, 1.0])  # a1 drifts
        d2 = dm.record("a2", 1, [0.0, 1.0])  # a2 stable
        self.assertAlmostEqual(d1, 1.0, places=5)
        self.assertAlmostEqual(d2, 0.0, places=5)

    def test_zero_vector_distance_is_one(self):
        dm = DriftMonitor()
        dm.record("a1", 0, [0.0, 0.0])
        delta = dm.record("a1", 1, [1.0, 0.0])
        self.assertEqual(delta, 1.0)

    def test_empty_history_returns_empty_list(self):
        dm = DriftMonitor()
        self.assertEqual(dm.drift_history("nonexistent"), [])

    def test_single_sample_history_returns_empty_list(self):
        dm = DriftMonitor()
        dm.record("a1", 0, [1.0, 0.0])
        self.assertEqual(dm.drift_history("a1"), [])

    def test_thread_safe_concurrent_records(self):
        dm = DriftMonitor(threshold=2.0)  # never fires
        errors = []

        def record_many(agent_id):
            try:
                for i in range(20):
                    dm.record(agent_id, i, [float(i), 0.0])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many, args=(f"a{i}",)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
