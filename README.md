# agent-memory-lab

> Five memory patterns for LLM agents, implemented minimally and compared head-to-head.

Agent memory is usually discussed at two unhelpful extremes: "just stuff everything in the context" on one end, and dense research papers on the other. This repo sits in between — five small, readable implementations you can drop into your own agent, plus a tiny benchmark so you can see which one fits your use case before committing.

## The five patterns

| # | Pattern | When to reach for it | Status |
|---|---|---|---|
| 1 | [Sliding window](./patterns/sliding_window.py) | Short tasks, cheapest, forgets anything older than N turns | ✅ v0 |
| 2 | [Summary compression](./patterns/summary_compression.py) | Long tasks where you can afford one summarization pass per K turns | ✅ v0 |
| 3 | [Vector retrieval](./patterns/vector_retrieval.py) | Large knowledge base; you need the *relevant* turns, not the *recent* turns | 🚧 |
| 4 | [Hierarchical summary](./patterns/hierarchical_summary.py) | Very long sessions; build a pyramid of summaries | 🚧 |
| 5 | [Structured episodic](./patterns/structured_episodic.py) | Multi-session agents; store "episodes" as structured records, query by attribute | 🚧 |

## Design principles

- **One file per pattern.** Read it top to bottom in 5 minutes.
- **No external ML deps for the core pattern.** Vector retrieval uses a pluggable embedder; everything else is stdlib.
- **Same interface for all patterns** (`add(msg)`, `view() -> list[Message]`) so swapping is one line.
- **Tests over docs.** If a pattern needs paragraphs of prose to explain behavior, it's not minimal yet.

## Benchmark (coming)

See [`bench/`](./bench/) for the plan. Rough idea: a fixed multi-turn agent task where the ground truth requires information introduced at turn 3 and recalled at turn 50. Each memory pattern gets the same task; we measure:

- **Recall accuracy** — did the right info survive to turn 50?
- **Tokens per turn** — cost proxy.
- **Latency overhead** — some patterns add a summarization or embedding call per turn.

Benchmarks will be reproducible with `python bench/run.py --pattern sliding_window`.

## Quickstart

```python
from patterns.sliding_window import SlidingWindow, Message

mem = SlidingWindow(window=20)
mem.add(Message(role="user", content="hello"))
mem.add(Message(role="assistant", content="hi"))
# ... later ...
messages_for_llm = mem.view()
```

## Installation

Zero runtime dependencies for the shipped patterns. Other patterns document their deps in their own docstrings.

```
git clone https://github.com/jimliu741523/agent-memory-lab
cd agent-memory-lab
python -m patterns.sliding_window        # runs the module's built-in demo
python -m patterns.summary_compression   # demo with a mock summarizer
python -m unittest discover tests -v     # stdlib-only tests for all patterns
```

## Roadmap

See [`patterns/README.md`](./patterns/README.md) for per-pattern notes and [`bench/README.md`](./bench/README.md) for the benchmark plan.

## Contributing

A new pattern is welcome if it:
- Fits in one file, ~150 lines or less
- Exposes the standard `add` / `view` interface
- Comes with a docstring explaining when to use it *and when not to*
- Has at least one test in `tests/`

## License

MIT.
