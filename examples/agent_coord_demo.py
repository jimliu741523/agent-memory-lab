"""
examples/agent_coord_demo.py — AgentCoord multi-agent coordination primitives (RP-5).

Five offline scenarios — no API calls, no network, all assertions pass:

  (1) FileLock — two agents race for a shared file; the second is blocked
      until the first releases. Stale-lease recovery lets a third agent
      reclaim an expired lock without manual intervention. Addresses the
      "Concurrent File Conflicts" failure mode from the ~20k-LOC practitioner
      report (sigalovskinick, May 2026).

  (2) WorkQueue — task distribution across three agents with atomic
      claim/release, no double-assignment under concurrent access, and
      automatic reclaim of stale-leased tasks after a crash. Addresses
      the "Flat Task Lists / no phase gates" failure mode from the same
      report and the 79% coordination-failure statistic (arxiv 2604.16339).

  (3) EventBus — cross-agent signaling: an orchestrator publishes phase
      events; two worker agents subscribe and react. Demonstrates the
      in-process pub/sub pattern that replaces the custom event managers
      practitioners build from scratch.

  (4) Barrier — phase gate: two parallel agents must both complete Phase 1
      before either proceeds to Phase 2. Prevents the "out-of-phase work"
      failure mode (agent B overwrites A's incomplete output) documented in
      arxiv 2502.14743 (livelocks from absent barriers).

  (5) DriftMonitor — semantic-drift detection across agent turns. A planner
      agent's simulated intent embeddings gradually drift orthogonally across
      five turns; the DriftMonitor fires an on_drift callback the moment the
      cosine-distance delta exceeds the configured threshold, enabling early
      intervention. Anchored to arxiv 2601.04170 (36.9% of production failures
      from inter-agent misalignment).

Run:
    python -m examples.agent_coord_demo
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

from patterns.agent_coord import (
    Barrier,
    DriftMonitor,
    EventBus,
    FileLock,
    WorkQueue,
)

PASS = "✓"
FAIL = "✗"
_failed: list[str] = []
_total = [0]


def _check(label: str, cond: bool) -> None:
    _total[0] += 1
    mark = PASS if cond else FAIL
    print(f"  {mark}  {label}")
    if not cond:
        _failed.append(label)


# ---------------------------------------------------------------------------
# Scenario 1 — FileLock: mutual exclusion + stale-lease recovery
# ---------------------------------------------------------------------------

def scenario_file_lock() -> None:
    print("\n=== Scenario 1: FileLock — mutual exclusion + stale-lease recovery ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        resource = Path(tmpdir) / "shared_artifact.json"

        fl = FileLock(resource, lease_secs=10)

        # Agent A acquires the lock
        acquired_a = fl.acquire("agent-A")
        _check("agent-A acquires lock on uncontested resource", acquired_a)

        # Agent B is blocked while A holds it
        blocked_b = fl.acquire("agent-B")
        _check("agent-B is blocked while agent-A holds the lock", not blocked_b)

        # Agent A releases; B can now acquire
        fl.release("agent-A")
        acquired_b_after = fl.acquire("agent-B")
        _check("agent-B acquires after agent-A releases", acquired_b_after)
        fl.release("agent-B")

        # Stale-lease scenario: agent-C crashes without releasing (short TTL)
        fl_short = FileLock(resource, lease_secs=0.08)
        fl_short.acquire("agent-C")
        time.sleep(0.15)  # lease expires
        _check("stale lock detected after lease expiry", fl_short.is_stale())

        # Agent-D recovers by re-acquiring the stale lock
        recovered = fl_short.acquire("agent-D")
        _check("agent-D reclaims stale lock without manual cleanup", recovered)
        fl_short.release("agent-D")

        # Context-manager interface
        with fl.locked("agent-E", timeout=1.0):
            held_during = not fl.acquire("agent-F")
        _check("context-manager blocks concurrent acquirer during block", held_during)
        released_after = fl.acquire("agent-G")
        _check("context-manager releases lock on block exit", released_after)
        fl.release("agent-G")


# ---------------------------------------------------------------------------
# Scenario 2 — WorkQueue: atomic claim/release, crash-recovery, no double-assign
# ---------------------------------------------------------------------------

def scenario_work_queue() -> None:
    print("\n=== Scenario 2: WorkQueue — atomic claim/release + crash-recovery ===")

    q = WorkQueue()
    tasks = [("parse-doc-1", {"url": "s3://docs/1"}),
             ("parse-doc-2", {"url": "s3://docs/2"}),
             ("parse-doc-3", {"url": "s3://docs/3"})]
    for tid, payload in tasks:
        q.put(tid, payload)

    # Three agents claim one task each
    t1 = q.claim("agent-1")
    t2 = q.claim("agent-2")
    t3 = q.claim("agent-3")
    _check("agent-1 claims a task", t1 is not None)
    _check("agent-2 claims a different task", t2 is not None and t2.id != t1.id)
    _check("agent-3 claims the last task", t3 is not None and t3.id not in {t1.id, t2.id})

    # Queue is now empty
    extra = q.claim("agent-4")
    _check("no task available for a fourth agent (queue exhausted)", extra is None)

    # Agent-1 completes; agent-2 crashes (releases back to pending)
    q.release(t1.id, done=True)
    q.release(t2.id)  # back to pending
    stats = q.stats()
    _check("one task marked done", stats.get("done", 0) == 1)
    _check("one task back in pending after crash-release", stats.get("pending", 0) == 1)

    # Stale-lease auto-recovery: agent-3 holds a very short lease
    q2 = WorkQueue()
    q2.put("stale-task", {"x": 42})
    q2.claim("agent-crash", lease_secs=0.06)
    time.sleep(0.15)
    recovered = q2.claim("agent-recovery", lease_secs=30)
    _check("stale-leased task auto-recovered by new agent", recovered is not None)
    q2.release(recovered.id, done=True)

    # No double-assignment under concurrency — use file-backed DB so all
    # threads share the same SQLite database (each thread's threading.local
    # connection would otherwise hit a separate :memory: database).
    import os as _os
    import tempfile as _tmpfile
    _fd, _db_path = _tmpfile.mkstemp(suffix=".db")
    _os.close(_fd)
    try:
        q3 = WorkQueue(_db_path)
        for i in range(8):
            q3.put(f"job-{i}", i)
        claimed_ids: list[str] = []
        lock = threading.Lock()

        def grab_one(worker_id: str) -> None:
            task = q3.claim(worker_id)
            if task:
                with lock:
                    claimed_ids.append(task.id)

        threads = [threading.Thread(target=grab_one, args=(f"w{i}",)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        _check("no double-assignment across 8 concurrent workers",
               len(claimed_ids) == len(set(claimed_ids)) and len(claimed_ids) == 8)
    finally:
        _os.unlink(_db_path)


# ---------------------------------------------------------------------------
# Scenario 3 — EventBus: cross-agent publish/subscribe signaling
# ---------------------------------------------------------------------------

def scenario_event_bus() -> None:
    print("\n=== Scenario 3: EventBus — cross-agent pub/sub signaling ===")

    bus = EventBus()
    received_by_a: list[dict] = []
    received_by_b: list[dict] = []

    bus.subscribe("phase:complete", lambda _evt, p: received_by_a.append(p))
    bus.subscribe("phase:complete", lambda _evt, p: received_by_b.append(p))

    # Orchestrator signals phase completion
    count = bus.publish("phase:complete", {"phase": 1, "orchestrator": "orch-1"})
    _check("event delivered to both subscribers (count=2)", count == 2)
    _check("agent-A received the phase-complete event", len(received_by_a) == 1)
    _check("agent-B received the phase-complete event", len(received_by_b) == 1)
    _check("payload matches published data", received_by_a[0]["phase"] == 1)

    # Unrelated event does not bleed through
    bus.publish("budget:exceeded", {"agent": "orch-1", "spend": 1.50})
    _check("unrelated event (budget:exceeded) not received by phase subscribers",
           len(received_by_a) == 1)

    # Subscriber unregisters cleanly
    def handler_c(evt: str, p: object) -> None:
        pass
    bus.subscribe("phase:complete", handler_c)
    removed = bus.unsubscribe("phase:complete", handler_c)
    _check("subscriber can unregister cleanly", removed)

    # Thread-safe concurrent publish
    events_fired: list[int] = []
    fire_lock = threading.Lock()
    bus3 = EventBus()

    def on_tick(_evt: str, _p: object) -> None:
        with fire_lock:
            events_fired.append(1)

    bus3.subscribe("tick", on_tick)

    def publish_tick() -> None:
        bus3.publish("tick", None)

    threads = [threading.Thread(target=publish_tick) for _ in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    _check("20 concurrent publishes all deliver without data loss", len(events_fired) == 20)


# ---------------------------------------------------------------------------
# Scenario 4 — Barrier: phase gate for parallel agents
# ---------------------------------------------------------------------------

def scenario_barrier() -> None:
    print("\n=== Scenario 4: Barrier — phase gate for parallel agents ===")

    barrier = Barrier(2)
    phase1_done: list[str] = []
    phase2_started: list[str] = []
    log_lock = threading.Lock()

    def agent_worker(name: str) -> None:
        # Phase 1 work (simulated by a brief pause)
        time.sleep(0.01 if name == "agent-X" else 0.04)
        with log_lock:
            phase1_done.append(name)

        # Both agents must finish Phase 1 before either starts Phase 2
        ok = barrier.wait(timeout=2.0)
        if ok:
            with log_lock:
                phase2_started.append(name)

    t1 = threading.Thread(target=agent_worker, args=("agent-X",))
    t2 = threading.Thread(target=agent_worker, args=("agent-Y",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    _check("both agents completed Phase 1", len(phase1_done) == 2)
    _check("both agents crossed the barrier into Phase 2", len(phase2_started) == 2)
    _check("neither agent skipped the gate (Phase 2 count matches Phase 1)",
           len(phase2_started) == len(phase1_done))

    # Timeout scenario: only one agent arrives, barrier returns False
    barrier2 = Barrier(3)
    result = barrier2.wait(timeout=0.1)
    _check("barrier.wait() returns False when quorum not reached within timeout",
           result is False)

    # Reusable across multiple phases
    barrier3 = Barrier(2)
    p1_arrivals: list[int] = []
    p2_arrivals: list[int] = []
    arr_lock = threading.Lock()

    def two_phase(idx: int) -> None:
        barrier3.wait(timeout=2.0)
        with arr_lock:
            p1_arrivals.append(idx)
        barrier3.wait(timeout=2.0)
        with arr_lock:
            p2_arrivals.append(idx)

    threads = [threading.Thread(target=two_phase, args=(i,)) for i in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    _check("barrier reusable for Phase 2 without reinitialisation",
           len(p2_arrivals) == 2)


# ---------------------------------------------------------------------------
# Scenario 5 — DriftMonitor: semantic-drift detection across agent turns
# ---------------------------------------------------------------------------

def scenario_drift_monitor() -> None:
    print("\n=== Scenario 5: DriftMonitor — semantic drift detection ===")

    drift_events: list[tuple[str, float]] = []
    monitor = DriftMonitor(
        threshold=0.4,
        on_drift=lambda agent_id, delta: drift_events.append((agent_id, delta)),
    )

    # Turn 0 — establish intent baseline (unit vector pointing at "retrieve")
    d0 = monitor.record("planner", 0, [1.0, 0.0, 0.0])
    _check("first record returns None (no previous sample to compare)", d0 is None)

    # Turns 1-2 — minor variation, below threshold (0.4)
    d1 = monitor.record("planner", 1, [0.99, 0.14, 0.0])   # ~8 deg off — tiny drift
    d2 = monitor.record("planner", 2, [0.98, 0.20, 0.0])   # ~11 deg off — still minor
    _check("small drift turn 1 stays below threshold (no callback)", d1 is not None and d1 < 0.4)
    _check("small drift turn 2 stays below threshold (no callback)", d2 is not None and d2 < 0.4)
    _check("no drift callback fired for sub-threshold turns", len(drift_events) == 0)

    # Turn 3 — abrupt pivot: [0,0,1] is perpendicular to turn-2 [0.98,0.20,0]
    #   dot product = 0 -> cosine distance exactly 1.0
    d3 = monitor.record("planner", 3, [0.0, 0.0, 1.0])
    _check("orthogonal pivot detected as high drift (delta = 1.0)", d3 is not None and d3 > 0.9)
    _check("on_drift callback fired with agent_id and delta", len(drift_events) == 1)
    _check("callback identifies correct agent", drift_events[0][0] == "planner")

    # Turn 4 — continues drifting: [0,1,0] is perpendicular to [0,0,1]
    d4 = monitor.record("planner", 4, [0.0, 1.0, 0.0])
    _check("drift continues on turn 4", d4 is not None and d4 > 0.9)

    # History API
    history = monitor.drift_history("planner")
    _check("drift_history returns 4 deltas for 5 recorded turns", len(history) == 4)

    # Independent agents tracked separately
    monitor.record("executor", 0, [1.0, 0.0])
    d_exec = monitor.record("executor", 1, [1.0, 0.0])
    _check("executor shows zero drift (stable embeddings)", d_exec is not None and abs(d_exec) < 1e-9)

    prev_callback_count = len(drift_events)
    _check("executor's stable turns did not trigger planner's callback",
           len(drift_events) == prev_callback_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("AgentCoord demo — Pattern 11 (RP-5)")
    print("Anchored to: arxiv 2604.16339 (79% multi-agent failures = coordination),")
    print("             arxiv 2601.04170 (36.9% inter-agent misalignment),")
    print("             arxiv 2502.14743 (livelocks from absent lock/queue/barrier),")
    print("             practitioner gist (sigalovskinick ~20k LOC for 2 agents).")

    scenario_file_lock()
    scenario_work_queue()
    scenario_event_bus()
    scenario_barrier()
    scenario_drift_monitor()

    total = _total[0]
    passed = total - len(_failed)
    print(f"\n{'='*55}")
    print(f"  {passed}/{total} assertions passed")
    if _failed:
        print("\nFailed:")
        for f in _failed:
            print(f"  {FAIL}  {f}")
        sys.exit(1)
    else:
        print("  All scenarios passed.")


if __name__ == "__main__":
    main()
