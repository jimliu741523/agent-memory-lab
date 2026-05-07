"""
examples/memory_writer_sketch.py — MemoryWriter write-path middleware (RP-3).

Shows three write-path problems that long-lived agents routinely hit:

  (a) Junk accumulation — low-salience tool chatter crowds out facts.
  (b) Contradictions — the agent later "learns" a fact conflicts with
      something already stored, and needs a resolution strategy.
  (c) Stale facts — knowledge gathered many turns ago is no longer
      relevant; TTL decay removes it automatically.

All three are handled by MemoryWriter with no LLM required at write time.
Swap the keyword heuristics below for LLM calls in production.

Run:
    python -m examples.memory_writer_sketch
"""
from __future__ import annotations

from typing import Optional

from patterns import MemoryWriter, Message, ListBackend, WriteOutcome


# ── Pluggable salience scorer ─────────────────────────────────────────────────
# In production: call an embedding model or an LLM judge.
# Here: keyword heuristics that are obvious and offline.

_HIGH_VALUE = {"fact", "result", "confirmed", "conclusion", "error", "critical"}
_NOISE      = {"ok", "done", "ack", "received", "acknowledged", "ping", "pong"}

def keyword_salience(msg: Message, _ctx: list[Message]) -> float:
    lower = msg.content.lower()
    if any(k in lower for k in _HIGH_VALUE):
        return 1.0
    if any(k in lower for k in _NOISE):
        return 0.1
    return 0.6  # neutral


# ── Pluggable contradiction detector ─────────────────────────────────────────
# In production: vector similarity + LLM entailment check.
# Here: same leading noun phrase → assumed to be about the same topic.

def _topic(text: str) -> str:
    """First 3 significant words as a proxy topic key."""
    words = [w.strip(".,") for w in text.lower().split() if len(w) > 3]
    return " ".join(words[:3])

def topic_contradict(msg: Message, ctx: list[Message]) -> Optional[Message]:
    key = _topic(msg.content)
    for existing in ctx:
        if existing.role == msg.role and _topic(existing.content) == key:
            if existing.content != msg.content:
                return existing
    return None


# ── Demo ──────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'─' * 56}")
    print(f"  {title}")
    print(f"{'─' * 56}")


def _demo() -> None:

    # ── (a) Salience gating ───────────────────────────────────────────────────
    _section("(a) Salience gating — noise filtered before commit")

    mw = MemoryWriter(
        backend=ListBackend(),
        salience_fn=keyword_salience,
        salience_threshold=0.5,
    )

    turns = [
        Message("assistant", "Acknowledged, processing your request."),         # noise
        Message("assistant", "Tool call ack — received."),                       # noise
        Message("assistant", "Confirmed: the API rate limit is 1 000 rps."),     # HIGH
        Message("assistant", "Done."),                                           # noise
        Message("assistant", "Critical result: auth token expires in 300 s."),  # HIGH
    ]

    print(f"\n  {'turn':>4}  {'outcome':25}  content[:50]")
    print(f"  {'':>4}  {'─' * 25}  {'─' * 50}")
    for i, msg in enumerate(turns):
        outcome = mw.add(msg)
        print(f"  {i:4d}  {outcome.name:25}  {msg.content[:50]!r}")

    committed = mw.view()
    print(f"\n  {len(committed)}/{len(turns)} messages committed to memory:")
    for m in committed:
        print(f"    [{m.role}] {m.content[:60]}")

    # ── (b) Contradiction resolution ──────────────────────────────────────────
    _section("(b) Contradiction resolution — latest_wins vs flag_for_review")

    # latest_wins: old fact suppressed, new fact committed
    mw_lw = MemoryWriter(
        backend=ListBackend(),
        contradict_fn=topic_contradict,
        resolution="latest_wins",
    )
    mw_lw.add(Message("user", "context window limit is 128k tokens"))
    r = mw_lw.add(Message("user", "context window limit is 200k tokens"))
    print(f"\n  latest_wins outcome : {r.name}")
    print(f"  memory after conflict ({len(mw_lw.view())} msg):")
    for m in mw_lw.view():
        print(f"    {m.content}")

    # flag_for_review: new fact held until human/agent reviews
    mw_fr = MemoryWriter(
        backend=ListBackend(),
        contradict_fn=topic_contradict,
        resolution="flag_for_review",
    )
    mw_fr.add(Message("user", "context window limit is 128k tokens"))
    r = mw_fr.add(Message("user", "context window limit is 200k tokens"))
    print(f"\n  flag_for_review outcome : {r.name}")
    print(f"  held queue ({len(mw_fr.held)} msg): {[m.content for m in mw_fr.held]}")
    # Resolve: release the new fact
    released = mw_fr.release(mw_fr.held[0])
    print(f"  after release → {released.name}; memory: {[m.content for m in mw_fr.view()]}")

    # ── (c) TTL / decay ───────────────────────────────────────────────────────
    _section("(c) TTL decay — stale facts expire, system messages pinned")

    mw_ttl = MemoryWriter(backend=ListBackend(), ttl=3)
    mw_ttl.add(Message("system", "You are a research assistant."))  # pinned

    facts = [
        "turn 1: API endpoint is /v1/search",
        "turn 2: rate limit confirmed 500 rps",
        "turn 3: pagination uses cursor token",
        "turn 4: auth header format is Bearer <token>",
        "turn 5: max page size is 100 items",
    ]
    for fact in facts:
        mw_ttl.add(Message("user", fact))

    visible = mw_ttl.view()
    print(f"\n  After {len(facts)} turns with ttl=3:")
    print(f"  {len(visible)} message(s) visible (system + last 3):")
    for m in visible:
        print(f"    [{m.role:9}] {m.content}")

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("Summary")
    print("""
  MemoryWriter sits between any tool loop and any memory backend.
  Swap the heuristic functions for LLM calls to get production-grade
  salience + contradiction detection with no API-surface changes.

  Patterns:
    patterns/memory_writer.py  — MemoryWriter implementation (RP-3)
    patterns/checkpoint.py     — CheckpointManager (RP-1)
""")


if __name__ == "__main__":
    _demo()
