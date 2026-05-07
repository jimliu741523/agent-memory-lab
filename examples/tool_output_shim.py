"""
examples/tool_output_shim.py — summary compression as a tool-output filter.

Addresses AP-22 (context pollution from raw tool output) from
agentic-anti-patterns: agents that pipe unfiltered tool responses verbatim
into context crowd out earlier task constraints by mid-task, because recent
tokens receive more model attention than early ones.

The shim here acts as a thin filtering layer between a tool call and the
context window:
  1. Raw tool output arrives (potentially large).
  2. The shim compacts it immediately with a pluggable one-shot summarizer.
  3. Only the compact summary enters the conversation context.
  4. All summaries are accumulated in a SummaryCompression store so even
     the compressed history is further rolled up as the session grows.

Run:
    python -m examples.tool_output_shim
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from patterns import Message, SummaryCompression


# ---------------------------------------------------------------------------
# Shim
# ---------------------------------------------------------------------------

@dataclass
class ToolOutputShim:
    """
    Wraps SummaryCompression with a two-level filter:
      - Level 1 (per-call): raw tool output → compact snippet via `compact_fn`.
      - Level 2 (rolling): compact snippets fed into SummaryCompression so
        even the compressed history stays bounded over very long sessions.

    Parameters
    ----------
    compact_fn : callable
        ``(text: str) -> str`` — converts one raw tool-output string into a
        compact snippet. Swap in a real LLM call; the mock below works offline.
    max_raw_chars : int
        Raw outputs shorter than this bypass `compact_fn` and go in verbatim.
    """

    compact_fn: Callable[[str], str]
    max_raw_chars: int = 400
    _store: SummaryCompression = field(init=False)

    def __post_init__(self) -> None:
        def _batch_summarize(msgs: list[Message]) -> str:
            combined = " | ".join(m.content[:60] for m in msgs)
            return f"[batch-summary: {combined}]"

        self._store = SummaryCompression(
            summarize=_batch_summarize,
            trigger=8,
            keep=4,
        )

    def record(self, tool_name: str, raw_output: str) -> str:
        """
        Filter `raw_output` and return a compact version for the context.

        Short outputs are kept verbatim; long outputs are compacted first.
        The compacted text is also archived in the rolling store.
        """
        if len(raw_output) <= self.max_raw_chars:
            snippet = raw_output
        else:
            snippet = self.compact_fn(raw_output)

        tag = f"[{tool_name}] {snippet}"
        self._store.add(Message(role="tool", content=tag))
        return tag

    def view(self) -> list[Message]:
        """Primary context — bounded by SummaryCompression."""
        return self._store.view()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _mock_compact(text: str) -> str:
    """Keeps first 120 chars — stand-in for a real LLM summarizer."""
    text = text.strip()
    return text[:120] + ("…" if len(text) > 120 else "")


def _fake_search(query: str) -> str:
    """Returns a large, noisy blob to simulate AP-22 raw tool output."""
    return "\n".join(
        f"Result {i}: [{query}] metadata={'x' * 180}"
        for i in range(1, 6)
    )


def _demo() -> None:
    task_constraint = (
        "Fix the auth bug WITHOUT modifying the /v1 public API endpoints."
    )
    shim = ToolOutputShim(compact_fn=_mock_compact, max_raw_chars=200)

    print(f"Task: {task_constraint}\n")
    print(f"{'Turn':>4}  {'raw chars':>9}  {'compact chars':>13}  {'reduction':>9}")
    print("-" * 44)

    for i in range(1, 11):
        raw = _fake_search(f"auth debug strategy {i}")
        compact = shim.record("web_search", raw)
        raw_c = len(raw)
        cmp_c = len(compact)
        print(
            f"{i:4d}  {raw_c:9d}  {cmp_c:13d}  {1 - cmp_c / raw_c:9.0%}"
        )

    print(f"\nContext messages visible to model: {len(shim.view())}")
    print("(capped by SummaryCompression — original 10 tool outputs rolled up)\n")
    print("Context tail:")
    for msg in shim.view()[-4:]:
        preview = msg.content[:72].replace("\n", " ")
        print(f"  [{msg.role:9s}] {preview}")

    print(
        "\nThe task constraint lives in the system prompt — pinned at position 0"
        " and never crowded out by tool output."
    )


if __name__ == "__main__":
    _demo()
