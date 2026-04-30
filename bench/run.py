"""
Recall micro-benchmark for the five memory patterns.

Design goal: a deterministic, stdlib-only harness that runs a fixed
50-turn conversation through each pattern and asks whether a fact
injected at turn 3 survives to the final context window at turn 50.

This is intentionally a *micro*-bench — not a ranking of agent
frameworks. It exists to make the pattern tradeoffs legible:

    - sliding_window   forgets by age
    - summary_compression  retains gist, loses detail
    - hierarchical_summary retains decay-gracefully
    - vector_retrieval surfaces on topical similarity
    - structured_episodic recalls by structured key

Each pattern is evaluated on:
    - recall_raw:      the target fact appears verbatim in the final
                       view() (or query() for the two patterns that
                       expose one).
    - context_chars:   total chars in the final view (token proxy
                       without a tokenizer dep).
    - extra_calls:     number of callback invocations (summarize,
                       embed) induced by running the 50-turn script.

Run:
    python -m bench.run
    python -m bench.run --pattern sliding_window
    python -m bench.run --output bench/results/results.md
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Ensure `patterns/` is importable when `python -m bench.run` is invoked
# from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patterns.sliding_window import SlidingWindow, Message as SWMessage
from patterns.summary_compression import (
    SummaryCompression,
    Message as SCMessage,
)
from patterns.hierarchical_summary import (
    HierarchicalSummary,
    Message as HSMessage,
)
from patterns.vector_retrieval import (
    VectorRetrieval,
    Message as VRMessage,
    _hash_bow_embed,
)
from patterns.structured_episodic import (
    StructuredEpisodic,
    Message as SEMessage,
)


TARGET_FACT = "launch code is ALPHA-7742"


def build_script() -> list[tuple[str, str]]:
    """
    Fifty turns. Turn 3 (user) carries the target fact. Turns 4..49 are
    filler. Turn 50 asks the recall question.

    Returns a list of (role, content) — role strings are the shared
    Message Role Literal for every pattern.
    """
    script: list[tuple[str, str]] = [
        ("system", "you are a helpful assistant for operations"),
        ("user", "hello, we need to coordinate a launch today"),
        ("user", f"note this: {TARGET_FACT}"),  # turn 3 — the target
        ("assistant", "noted."),
    ]
    fillers = [
        "what's the weather like in Taipei",
        "summarize yesterday's standup",
        "draft a follow-up email to the vendor",
        "find last week's deployment logs",
        "what is 17 * 23",
        "translate 'hello' into five languages",
        "rewrite this paragraph in a formal tone",
        "list three risks for a new microservice",
    ]
    for i in range(46):
        user = fillers[i % len(fillers)]
        script.append(("user", f"{user} (turn {i + 5})"))
        script.append(("assistant", f"ack (turn {i + 5})"))
    # Recall question at turn 50 (stored but not asked of the model —
    # we just score against final view).
    script.append(("user", "what was the launch code?"))
    return script


@dataclass
class Result:
    pattern: str
    recall_raw: bool
    context_chars: int
    extra_calls: int
    notes: str = ""

    def row(self) -> str:
        ok = "yes" if self.recall_raw else "no"
        return f"| {self.pattern} | {ok} | {self.context_chars} | {self.extra_calls} | {self.notes} |"


def _counting(fn: Callable) -> tuple[Callable, list[int]]:
    counter = [0]

    def wrapped(*args, **kwargs):
        counter[0] += 1
        return fn(*args, **kwargs)

    return wrapped, counter


def _run_sliding_window(script: list[tuple[str, str]]) -> Result:
    mem = SlidingWindow(window=10)
    for role, content in script:
        mem.add(SWMessage(role, content))
    view = mem.view()
    text = "\n".join(m.content for m in view)
    return Result(
        pattern="sliding_window",
        recall_raw=TARGET_FACT in text,
        context_chars=len(text),
        extra_calls=0,
        notes="window=10",
    )


def _run_summary_compression(script: list[tuple[str, str]]) -> Result:
    def _summarize(msgs):
        preview = " | ".join(m.content[:20] for m in msgs[:3])
        return f"[{len(msgs)} earlier msgs; preview: {preview}]"

    counting_summarize, counter = _counting(_summarize)
    mem = SummaryCompression(summarize=counting_summarize, trigger=20, keep=10)
    for role, content in script:
        mem.add(SCMessage(role, content))
    view = mem.view()
    text = "\n".join(m.content for m in view)
    return Result(
        pattern="summary_compression",
        recall_raw=TARGET_FACT in text,
        context_chars=len(text),
        extra_calls=counter[0],
        notes="trigger=20 keep=10 (mock summarizer)",
    )


def _run_hierarchical_summary(script: list[tuple[str, str]]) -> Result:
    def _summarize(msgs):
        return f"<rollup of {len(msgs)}>"

    counting_summarize, counter = _counting(_summarize)
    mem = HierarchicalSummary(
        summarize=counting_summarize, leaf_chunk=10, fanout=3, keep_recent=5
    )
    for role, content in script:
        mem.add(HSMessage(role, content))
    view = mem.view()
    text = "\n".join(m.content for m in view)
    return Result(
        pattern="hierarchical_summary",
        recall_raw=TARGET_FACT in text,
        context_chars=len(text),
        extra_calls=counter[0],
        notes="leaf=10 fanout=3 keep_recent=5 (mock)",
    )


def _run_vector_retrieval(script: list[tuple[str, str]]) -> Result:
    counting_embed, counter = _counting(_hash_bow_embed)
    mem = VectorRetrieval(embed=counting_embed, keep_recent=10)
    for role, content in script:
        mem.add(VRMessage(role, content))

    # Recall path uses query(), not view(), for this pattern.
    hits = mem.query("launch code", k=3)
    view = mem.view()
    base_text = "\n".join(m.content for m in view)
    query_text = "\n".join(m.content for m in hits)
    combined = base_text + "\n" + query_text
    return Result(
        pattern="vector_retrieval",
        recall_raw=TARGET_FACT in combined,
        context_chars=len(base_text),
        extra_calls=counter[0],
        notes="keep_recent=10 hash-BOW embed, recall via query()",
    )


def _run_structured_episodic(script: list[tuple[str, str]]) -> Result:
    mem = StructuredEpisodic(keep_recent=10)
    for role, content in script:
        mem.add(SEMessage(role, content))
        # Record an episode any time the user states a code-like fact
        # (domain-specific trigger; models the "agent saves an episode
        # when something structured is established" pattern).
        if role == "user" and "code is" in content:
            # parse "... the launch code is ALPHA-7742"
            after = content.split("code is", 1)[1].strip().rstrip(".")
            mem.record_episode(
                situation={"task": "launch", "field": "code"},
                action="stored",
                outcome=after,
                tags=("launch", "code"),
            )

    hits = mem.recall_episodes(situation={"task": "launch", "field": "code"}, k=1)
    view = mem.view()
    base_text = "\n".join(m.content for m in view)
    recall_text = " ".join(h.outcome for h in hits)
    combined = base_text + " " + recall_text
    # The harness treats TARGET_FACT as recalled if its *answer portion*
    # is present; the structured episode stores only the answer, not the
    # full original sentence.
    recalled = "ALPHA-7742" in combined
    return Result(
        pattern="structured_episodic",
        recall_raw=recalled,
        context_chars=len(base_text),
        extra_calls=0,
        notes="keep_recent=10 recall via recall_episodes()",
    )


RUNNERS: dict[str, Callable[[list[tuple[str, str]]], Result]] = {
    "sliding_window": _run_sliding_window,
    "summary_compression": _run_summary_compression,
    "hierarchical_summary": _run_hierarchical_summary,
    "vector_retrieval": _run_vector_retrieval,
    "structured_episodic": _run_structured_episodic,
}


def render(results: list[Result]) -> str:
    header = (
        "| pattern | recall (turn 3 fact) | final-context chars | extra callback calls | notes |\n"
        "|---|---|---|---|---|\n"
    )
    body = "\n".join(r.row() for r in results)
    return header + body + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="agent-memory-lab recall micro-bench")
    parser.add_argument(
        "--pattern",
        default="all",
        choices=["all", *RUNNERS.keys()],
        help="which pattern to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="write the rendered markdown table to this path as well as stdout",
    )
    args = parser.parse_args()

    script = build_script()
    names = list(RUNNERS) if args.pattern == "all" else [args.pattern]
    results = [RUNNERS[name](script) for name in names]

    rendered = render(results)
    print(f"# agent-memory-lab recall bench ({len(script)} turns, target fact at turn 3)\n")
    print(rendered)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            f"# agent-memory-lab recall bench\n\n"
            f"- Turns: {len(script)}\n"
            f"- Target fact injected at turn 3 (user message)\n"
            f"- Deterministic, stdlib-only, no network\n\n"
            + rendered
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
