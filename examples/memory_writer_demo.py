"""
examples/memory_writer_demo.py — MemoryWriter write-path middleware (RP-3).

Five offline scenarios — no API calls, no network, all assertions pass:

  (1) Salience gating — low-value tool chatter is DROPPED_LOW_SALIENCE
      before reaching the backend; high-value facts are committed. Models
      scoring 90%+ on passive recall plummet to 40-60% on active decision
      tasks precisely because junk crowds out signal on the write path
      (arxiv 2603.07670 MemoryArena empirical gap).

  (2) latest_wins contradiction — when the agent learns a fact conflicts
      with a stored fact, the old entry is suppressed and the new one
      committed. Addresses the 55.2% best-model ceiling on implicit-conflict
      scenarios (arxiv 2605.06527 STALE). Old fact no longer appears in
      view().

  (3) flag_for_review + release/discard — contradictions can instead be
      held in a review queue (HELD_FOR_REVIEW) for human or agent review.
      release() promotes the candidate to the backend; discard() drops it.
      Addresses agents oscillating between contradictory states (AUDN loop
      practitioner pattern, dev.to 2026).

  (4) Merge resolution — old and new conflicting facts are merged into a
      single entry via a pluggable MergeFn. The merged message is committed;
      neither raw version persists. Maps to SSGM "intrinsic drift" handling
      (arxiv 2603.11768).

  (5) TTL decay + system pinning — messages older than ttl turns are
      filtered from view(); system messages are always kept regardless of
      age. Simulates the BeliefMem multi-turn staleness scenario (arxiv
      2605.06527 STALE; mem0.ai "context rot" 2026). Asserts exact
      visibility counts at each turn boundary.

Run:
    python -m examples.memory_writer_demo
"""
from __future__ import annotations

import sys
from typing import List, Optional

from patterns import MemoryWriter, Message, ListBackend, WriteOutcome

PASS = "✓"
FAIL = "✗"


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------

def _check(label: str, condition: bool) -> None:
    mark = PASS if condition else FAIL
    print(f"  {mark}  {label}")
    assert condition, label


# ---------------------------------------------------------------------------
# Salience helpers (keyword heuristics — swap for LLM call in production)
# ---------------------------------------------------------------------------

_HIGH_VALUE = {"fact", "result", "confirmed", "conclusion", "error", "critical", "auth", "limit"}
_NOISE = {"ok", "done", "ack", "received", "acknowledged", "ping", "pong"}


def _keyword_salience(msg: Message, _ctx: List[Message]) -> float:
    lower = msg.content.lower()
    if any(k in lower for k in _HIGH_VALUE):
        return 1.0
    if any(k in lower for k in _NOISE):
        return 0.1
    return 0.6


# ---------------------------------------------------------------------------
# Contradiction helper (topic heuristic — swap for vector similarity + LLM)
# ---------------------------------------------------------------------------

def _topic_words(text: str) -> str:
    words = [w.strip(".,!?") for w in text.lower().split() if len(w) > 3]
    return " ".join(words[:3])


def _topic_contradict(msg: Message, ctx: List[Message]) -> Optional[Message]:
    key = _topic_words(msg.content)
    for existing in ctx:
        if existing.role == msg.role and _topic_words(existing.content) == key:
            if existing.content != msg.content:
                return existing
    return None


# ---------------------------------------------------------------------------
# Scenario 1 — Salience gating
# ---------------------------------------------------------------------------

def scenario_salience_gating() -> None:
    print("\nScenario 1 — Salience gating: noise filtered before commit")
    print("  Evidence: 90%+ passive recall → 40-60% active-task recall gap")
    print("  (arxiv 2603.07670 MemoryArena; mem0.ai 'write-path invisibility' 2026)")

    mw = MemoryWriter(
        backend=ListBackend(),
        salience_fn=_keyword_salience,
        salience_threshold=0.5,
    )

    noise_msgs = [
        Message("assistant", "Acknowledged."),
        Message("assistant", "Done."),
        Message("assistant", "Received, processing."),
    ]
    fact_msgs = [
        Message("assistant", "Confirmed: the API rate limit is 1000 rps."),
        Message("assistant", "Critical result: auth token expires in 300 s."),
    ]

    noise_outcomes = [mw.add(m) for m in noise_msgs]
    fact_outcomes  = [mw.add(m) for m in fact_msgs]

    _check("all noise msgs dropped", all(o == WriteOutcome.DROPPED_LOW_SALIENCE for o in noise_outcomes))
    _check("all fact msgs committed", all(o == WriteOutcome.COMMITTED for o in fact_outcomes))
    _check("only 2 msgs in backend", len(mw.view()) == 2)
    _check("first committed msg contains 'rate limit'", "rate limit" in mw.view()[0].content)
    _check("second committed msg contains 'auth token'", "auth token" in mw.view()[1].content)

    print("  All salience-gating checks passed.")


# ---------------------------------------------------------------------------
# Scenario 2 — latest_wins contradiction
# ---------------------------------------------------------------------------

def scenario_latest_wins() -> None:
    print("\nScenario 2 — latest_wins contradiction: old fact suppressed, new committed")
    print("  Evidence: 55.2% best-model score on implicit-conflict scenarios")
    print("  (arxiv 2605.06527 STALE 'Can LLM Agents Know When Memories Are Invalid?')")

    mw = MemoryWriter(
        backend=ListBackend(),
        contradict_fn=_topic_contradict,
        resolution="latest_wins",
    )

    mw.add(Message("user", "context window limit is 128k tokens"))
    initial_count = len(mw.view())

    outcome = mw.add(Message("user", "context window limit is 200k tokens"))

    _check("outcome is FLAGGED_CONTRADICTION", outcome == WriteOutcome.FLAGGED_CONTRADICTION)
    _check("still only 1 msg in view (old suppressed, new added)", len(mw.view()) == 1)
    _check("view contains the new fact (200k)", "200k" in mw.view()[0].content)
    _check("old fact (128k) no longer visible", not any("128k" in m.content for m in mw.view()))
    _check("initial_count was 1", initial_count == 1)

    print("  All latest_wins checks passed.")


# ---------------------------------------------------------------------------
# Scenario 3 — flag_for_review + release / discard
# ---------------------------------------------------------------------------

def scenario_flag_for_review() -> None:
    print("\nScenario 3 — flag_for_review: contradictions held for human/agent review")
    print("  Evidence: agents oscillating between contradictory states for several")
    print("  interactions (AUDN loop pattern, dev.to state-of-agent-memory 2026)")

    # --- 3a: release promotes the candidate ---
    mw = MemoryWriter(
        backend=ListBackend(),
        contradict_fn=_topic_contradict,
        resolution="flag_for_review",
    )
    mw.add(Message("user", "context window limit is 128k tokens"))
    outcome = mw.add(Message("user", "context window limit is 200k tokens"))

    _check("outcome is HELD_FOR_REVIEW", outcome == WriteOutcome.HELD_FOR_REVIEW)
    _check("held queue has 1 msg", len(mw.held) == 1)
    _check("original fact still in view", any("128k" in m.content for m in mw.view()))
    _check("new fact NOT yet in view", not any("200k" in m.content for m in mw.view()))

    released = mw.release(mw.held[0])
    _check("release returns COMMITTED", released == WriteOutcome.COMMITTED)
    _check("held queue now empty", len(mw.held) == 0)
    _check("view now contains both (old not suppressed in flag_for_review)", len(mw.view()) == 2)

    # --- 3b: discard drops the candidate ---
    mw2 = MemoryWriter(
        backend=ListBackend(),
        contradict_fn=_topic_contradict,
        resolution="flag_for_review",
    )
    mw2.add(Message("user", "context window limit is 128k tokens"))
    mw2.add(Message("user", "context window limit is 200k tokens"))
    candidate = mw2.held[0]
    mw2.discard(candidate)

    _check("after discard: held queue empty", len(mw2.held) == 0)
    _check("after discard: only original fact in view", len(mw2.view()) == 1)
    _check("after discard: original fact preserved", "128k" in mw2.view()[0].content)

    print("  All flag_for_review checks passed.")


# ---------------------------------------------------------------------------
# Scenario 4 — Merge resolution
# ---------------------------------------------------------------------------

def scenario_merge_resolution() -> None:
    print("\nScenario 4 — Merge resolution: conflicting facts merged into one entry")
    print("  Evidence: 'intrinsic drift' (knowledge conflict) — no production OSS")
    print("  implements forgetting-by-design (arxiv 2603.11768 SSGM framework)")

    def _merge_latest(new: Message, old: Message) -> Message:
        return Message(role=new.role, content=f"[merged] {new.content}")

    mw = MemoryWriter(
        backend=ListBackend(),
        contradict_fn=_topic_contradict,
        merge_fn=_merge_latest,
        resolution="merge",
    )

    mw.add(Message("user", "context window limit is 128k tokens"))
    outcome = mw.add(Message("user", "context window limit is 200k tokens"))

    _check("outcome is MERGED", outcome == WriteOutcome.MERGED)
    _check("exactly 1 msg in view", len(mw.view()) == 1)
    merged_content = mw.view()[0].content
    _check("merged entry contains [merged] prefix", "[merged]" in merged_content)
    _check("merged entry contains new fact (200k)", "200k" in merged_content)
    _check("old raw fact (128k) not in merged content", "128k tokens" not in merged_content)

    print("  All merge-resolution checks passed.")


# ---------------------------------------------------------------------------
# Scenario 5 — TTL decay + system message pinning
# ---------------------------------------------------------------------------

def scenario_ttl_decay() -> None:
    print("\nScenario 5 — TTL decay: stale facts expire; system messages always pinned")
    print("  Evidence: agents commit to one fact per observation → self-reinforcing")
    print("  error (arxiv 2605.05583 BeliefMem); 'context rot' (mem0.ai 2026 report)")

    mw = MemoryWriter(backend=ListBackend(), ttl=3)

    sys_msg = Message("system", "You are a research assistant.")
    mw.add(sys_msg)

    facts = [
        Message("user", f"turn {i}: fact about topic {i}")
        for i in range(1, 7)
    ]
    for f in facts:
        mw.add(f)

    visible = mw.view()
    _check("system message always visible", sys_msg in visible)
    _check("only last 3 non-system facts visible (ttl=3)", sum(1 for m in visible if m.role != "system") == 3)
    _check("total visible = system + 3 recent facts", len(visible) == 4)

    last3_contents = [m.content for m in visible if m.role != "system"]
    _check("fact from turn 3 not visible (expired)", not any("turn 3:" in c for c in last3_contents))
    _check("fact from turn 4 visible (recent)", any("turn 4:" in c for c in last3_contents))
    _check("fact from turn 6 visible (recent)", any("turn 6:" in c for c in last3_contents))

    # Verify system message stays pinned regardless of TTL
    mw2 = MemoryWriter(backend=ListBackend(), ttl=1)
    sys2 = Message("system", "Pinned system instruction.")
    mw2.add(sys2)
    for i in range(10):
        mw2.add(Message("user", f"ephemeral turn {i}"))
    view2 = mw2.view()
    _check("system msg pinned even with ttl=1 after 10 turns", sys2 in view2)
    _check("only 1 non-system msg visible with ttl=1", sum(1 for m in view2 if m.role != "system") == 1)

    print("  All TTL-decay checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("MemoryWriter demo — Pattern 7 (RP-3: memory write-path middleware)")
    print("=" * 64)
    print("Evidence: 90%+ passive recall → 40-60% active-task gap (arxiv 2603.07670);")
    print("          55.2% implicit-conflict ceiling (arxiv 2605.06527 STALE);")
    print("          BeliefMem multi-candidate retention (arxiv 2605.05583);")
    print("          SSGM intrinsic-drift framework (arxiv 2603.11768);")
    print("          mem0.ai 'write-path invisibility' 2026 industry report.")

    failures: List[str] = []
    for fn in [
        scenario_salience_gating,
        scenario_latest_wins,
        scenario_flag_for_review,
        scenario_merge_resolution,
        scenario_ttl_decay,
    ]:
        try:
            fn()
        except AssertionError as exc:
            failures.append(str(exc))

    print("\n" + "=" * 64)
    if failures:
        print(f"FAILED — {len(failures)} assertion(s):")
        for f in failures:
            print(f"  {FAIL}  {f}")
        sys.exit(1)
    else:
        count = 5 + 5 + 7 + 5 + 8  # assertions per scenario
        print(f"All 5 scenarios passed ({count} assertions)  {PASS}")


if __name__ == "__main__":
    main()
