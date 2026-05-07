"""
examples/checkpoint_resume_demo.py — crash-and-resume for a multi-step agent loop.

Demonstrates the core value of CheckpointManager (RP-1): a running agent loop
that saves state after every step survives a process crash and can resume
exactly where it left off — without rerunning completed steps or losing
tool-call results.

The sequence:

    Phase 1:  agent runs steps 0–2, crashes on step 3
              → 4 checkpoint files survive on disk

    Phase 2:  "new process" loads the latest checkpoint (step 3)
              → resumes steps 3–4, done

No API calls, no external dependencies.

Run:
    python -m examples.checkpoint_resume_demo
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

from patterns import CheckpointManager, RunState, FileBackend

# Simulated planner task list; real agents derive this from the LLM.
TASKS: list[tuple[str, dict]] = [
    ("web_search",    {"q": "agent checkpoint architectures 2026"}),
    ("fetch_url",     {"url": "https://arxiv.org/abs/2503.12434"}),
    ("summarize",     {"doc_id": "doc-001"}),
    ("extract_facts", {"doc_id": "doc-001", "fields": ["title", "year", "key_claims"]}),
    ("write_report",  {"title": "Agent Durability Survey", "sections": 3}),
]

CRASH_AT_STEP = 3  # raise RuntimeError before executing this step number


def _fake_execute(tool_name: str, args: dict) -> dict:
    return {"tool": tool_name, "status": "ok", "result_id": f"r-{tool_name[:6]}-1"}


def run_agent(mgr: CheckpointManager, resume_token: Optional[str] = None) -> None:
    """
    Run the research loop, saving a checkpoint after every step.

    Pass resume_token to skip completed steps and pick up mid-run.
    """
    if resume_token is not None:
        state = mgr.load(resume_token)
        print(f"  [resume] loaded checkpoint  step={state.step}  token={resume_token[:12]}…")
    else:
        state = RunState(
            run_id="demo-research-001",
            step=0,
            tool_queue=[{"name": t, "args": a} for t, a in TASKS],
            memory_snap=[{"role": "system", "content": "You are a research agent."}],
            metadata={"framework": "raw_loop", "model": "claude-sonnet-4-6"},
        )
        token = mgr.save(state)
        print(f"  [init]   saved initial checkpoint  token={token[:12]}…")

    while state.tool_queue:
        call = state.tool_queue[0]

        # Simulated crash mid-loop — only on first run, not on resume.
        if state.step == CRASH_AT_STEP and resume_token is None:
            raise RuntimeError(
                f"process killed at step {state.step} "
                f"before {call['name']!r} dispatched"
            )

        result = _fake_execute(call["name"], call["args"])
        print(
            f"  step {state.step:02d}  {call['name']}  →  "
            f"result_id={result['result_id']!r}"
        )

        # Advance: dequeue call, record result, bump step.
        new_queue = state.tool_queue[1:]
        new_snap = state.memory_snap + [{"role": "tool", "content": str(result)}]
        state = RunState(
            run_id=state.run_id,
            step=state.step + 1,
            tool_queue=new_queue,
            pending=[],
            memory_snap=new_snap,
            stream_cursor=state.stream_cursor,
            metadata=state.metadata,
            created_at=state.created_at,
        )
        token = mgr.save(state)
        print(f"           checkpoint saved  token={token[:12]}…")

    print(f"\n  [done]  run_id={state.run_id!r}  steps_completed={state.step}")


def _latest_token(directory: str) -> str:
    """Token for the most-recently written checkpoint file in *directory*."""
    files = [
        (os.path.getmtime(os.path.join(directory, f)), f)
        for f in os.listdir(directory)
        if f.endswith(".json")
    ]
    if not files:
        raise FileNotFoundError("no checkpoint files found")
    return max(files)[1][:-5]  # strip .json suffix → token


def _demo() -> None:
    with tempfile.TemporaryDirectory() as ckpt_dir:
        mgr = CheckpointManager(backend=FileBackend(ckpt_dir))

        # ── Phase 1: run until crash ──────────────────────────────────────────
        print("=" * 60)
        print("Phase 1 — initial run (will crash at step 3)")
        print("=" * 60)
        try:
            run_agent(mgr, resume_token=None)
        except RuntimeError as exc:
            print(f"\n  [CRASH]  {exc}")
            files = sorted(os.listdir(ckpt_dir))
            print(f"  {len(files)} checkpoint file(s) survive on disk:")
            for f in files:
                print(f"    {f[:18]}…json")

        # ── Phase 2: new process — load latest checkpoint and resume ──────────
        print()
        print("=" * 60)
        print("Phase 2 — new process, resuming from last checkpoint")
        print("=" * 60)
        latest = _latest_token(ckpt_dir)
        run_agent(mgr, resume_token=latest)

        print()
        print("Steps 0–2 were not re-executed.")
        print("The agent resumed at exactly the step that was queued when it crashed.")


if __name__ == "__main__":
    _demo()
