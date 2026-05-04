# agent-memory-lab

> Five memory patterns for LLM agents, implemented minimally and compared head-to-head.

Agent memory is usually discussed at two unhelpful extremes: "just stuff everything in the context" on one end, and dense research papers on the other. This repo sits in between — five small, readable implementations you can drop into your own agent, plus a tiny benchmark so you can see which one fits your use case before committing.

## The five patterns

| # | Pattern | When to reach for it | Status |
|---|---|---|---|
| 1 | [Sliding window](./patterns/sliding_window.py) | Short tasks, cheapest, forgets anything older than N turns | ✅ v0 |
| 2 | [Summary compression](./patterns/summary_compression.py) | Long tasks where you can afford one summarization pass per K turns | ✅ v0 |
| 3 | [Vector retrieval](./patterns/vector_retrieval.py) | Large knowledge base; you need the *relevant* turns, not the *recent* turns | ✅ v0 |
| 4 | [Hierarchical summary](./patterns/hierarchical_summary.py) | Very long sessions; build a pyramid of summaries that degrade gracefully with age | ✅ v0 |
| 5 | [Structured episodic](./patterns/structured_episodic.py) | Multi-session agents; store "episodes" as structured records, query by attribute | ✅ v0 |

## Design principles

- **One file per pattern.** Read it top to bottom in 5 minutes.
- **No external ML deps for the core pattern.** Vector retrieval uses a pluggable embedder; everything else is stdlib.
- **Same interface for all patterns** (`add(msg)`, `view() -> list[Message]`) so swapping is one line.
- **Tests over docs.** If a pattern needs paragraphs of prose to explain behavior, it's not minimal yet.

## Benchmark

A 50-turn recall micro-bench (deterministic, stdlib-only, no network) is in [`bench/run.py`](./bench/run.py). A target fact is injected at turn 3; each pattern is asked to surface it at the end. Latest run ([`bench/results/results.md`](./bench/results/results.md)):

| pattern | recall (turn-3 fact) | final-context chars | extra callback calls |
|---|---|---|---|
| sliding_window | no | 303 | 0 |
| summary_compression | no | 1451 | 7 |
| hierarchical_summary | no | 225 | 13 |
| vector_retrieval | **yes** | 303 | 87 (one embed per archived msg) |
| structured_episodic | **yes** | 303 | 0 |

Reads as expected: the recency-only patterns drop the early fact; the two patterns with explicit recall (`query()` and `recall_episodes()`) surface it on demand. `summary_compression` keeps a longer rolling buffer but didn't preserve the specific token. `hierarchical_summary` compresses the most aggressively. The numbers are not a horse race — they make the *tradeoff* between recall, context cost, and per-turn callback work concrete.

```
python -m bench.run                            # all patterns, table to stdout
python -m bench.run --pattern sliding_window   # one pattern only
python -m bench.run --output bench/results/results.md
```

This is a micro-bench for legibility, not a ranking. It's not designed to argue any pattern is "best" — only to show the shape of each pattern's compromise.

## Quickstart

Single pattern (any of the five — same interface):

```python
from patterns import SlidingWindow, Message

mem = SlidingWindow(window=20)
mem.add(Message(role="user", content="hello"))
mem.add(Message(role="assistant", content="hi"))
messages_for_llm = mem.view()
```

Composed (recent + topic recall — the production shape, see [`examples/compose.py`](./examples/compose.py)):

```python
from patterns import SlidingWindow, VectorRetrieval, Message
from patterns.vector_retrieval import _hash_bow_embed

recent = SlidingWindow(window=20)
archive = VectorRetrieval(embed=_hash_bow_embed, keep_recent=0)

def add(msg):
    recent.add(msg); archive.add(msg)

# add(...) for a long conversation, then:
context = recent.view()                          # what the LLM sees
extra   = archive.query("specific old fact", k=3)  # topical recall on demand
```

The two return the same `Message` shape, so concatenating `context + extra` is one line.

## Installation

Zero runtime dependencies for the shipped patterns. Other patterns document their deps in their own docstrings.

```
git clone https://github.com/jimliu741523/agent-memory-lab
cd agent-memory-lab
python -m patterns.sliding_window        # runs the module's built-in demo
python -m patterns.summary_compression   # demo with a mock summarizer
python -m patterns.hierarchical_summary  # pyramid of rolling summaries demo
python -m patterns.vector_retrieval      # semantic recall with stdlib hash-BOW embedder
python -m patterns.structured_episodic   # typed episode records, recall by structured key match
python -m unittest discover tests -v     # stdlib-only tests for all patterns (26/26)
```

## Roadmap

See [`patterns/README.md`](./patterns/README.md) for per-pattern notes and [`bench/README.md`](./bench/README.md) for the benchmark plan.

## Contributing

A new pattern is welcome if it:
- Fits in one file, ~150 lines or less
- Exposes the standard `add` / `view` interface
- Comes with a docstring explaining when to use it *and when not to*
- Has at least one test in `tests/`

## Related

- [`agentic-anti-patterns`](https://github.com/jimliu741523/agentic-anti-patterns) catalogs agent failure modes; the patterns in this repo are concrete mitigations for:
  - [AP-05 Context bloat → cost explosion](https://github.com/jimliu741523/agentic-anti-patterns#ap-05--context-bloat--cost-explosion) — sliding-window and summary-compression bound the context; hierarchical summary (planned) handles very long sessions
  - [AP-08 Memory poisoning](https://github.com/jimliu741523/agentic-anti-patterns#ap-08--memory-poisoning) — provenance-tagging and trust-tier authorization would apply to any of the stores here
- [`self-evolving-agent`](https://github.com/jimliu741523/self-evolving-agent) — sibling experiment in agent self-improvement; shares the pluggable-callable design (this repo for `summarize_fn`, self-evolving-agent for `ModelFn`).

## License

MIT.
