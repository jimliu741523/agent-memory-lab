# Patterns

Each file is a single memory pattern. Pick the one that matches your agent's failure mode.

## Selection guide (decision tree)

Walk top-down; first match wins. The head-to-head bench in [`bench/run.py`](../bench/) backs the recall column with measured numbers.

```
Is the conversation < 50 turns AND you can afford to drop old context?
└── yes  → sliding_window
└── no   → continue ↓

Do you need to recall a SPECIFIC older fact at query time
(rather than the gist of older context)?
└── yes  → does the query have a natural typed shape (task=X, env=prod, ...)?
           └── yes → structured_episodic
           └── no  → vector_retrieval
└── no   → continue ↓

Is the session very long AND you want graceful detail decay
(recent verbatim, older becomes a single rolled-up summary)?
└── yes  → hierarchical_summary
└── no   → summary_compression
```

### Cheat-sheet by failure mode

What does your agent lose most often? That's the pattern you want.

- **Forgets recent turns under cost pressure** → `sliding_window.py`
- **Forgets older context but can afford a periodic summarize call** → `summary_compression.py`
- **Forgets in a way correlated with topic, not recency** → `vector_retrieval.py`
- **Sessions span hours/days; recall needs to degrade gracefully with age** → `hierarchical_summary.py`
- **Same task type repeats across sessions with structured outcomes** → `structured_episodic.py`

### Bench-measured tradeoff (10-seed run, target fact at turn 3 of 50)

| pattern | recall pass-rate | context chars (min/avg/max) | per-turn callback cost |
|---|---|---|---|
| sliding_window | 0/10 | 273/311/323 | 0 |
| summary_compression | 0/10 | 1393/1446/1473 | summarize call per `trigger` overflow |
| hierarchical_summary | 0/10 | 203/233/251 | summarize call per leaf chunk + cascade |
| vector_retrieval | 10/10 | 273/311/323 | embed call per archived msg |
| structured_episodic | 10/10 | 273/311/323 | 0 |

Read this as: if the recall task is *retrieve a specific old fact*, only `vector_retrieval` and `structured_episodic` solve it — the recency-only patterns intentionally drop old turns. Compose them: `sliding_window` for the live context tail + `vector_retrieval` for retrieval is a real production shape this repo deliberately doesn't hide behind a single class.

## Shared interface

```python
class Pattern:
    def add(self, msg: Message) -> None: ...
    def view(self) -> list[Message]: ...
```

Drop-in replaceable. Your agent code doesn't change when you swap patterns.

## Status

| Pattern | File | Status |
|---|---|---|
| Sliding window | `sliding_window.py` | ✅ v0 |
| Summary compression | `summary_compression.py` | ✅ v0, pluggable summarizer + mock for demo |
| Vector retrieval | `vector_retrieval.py` | ✅ v0, pluggable embedder + stdlib hash-BOW fallback |
| Hierarchical summary | `hierarchical_summary.py` | ✅ v0, pyramid rollup with cascading fanout |
| Structured episodic | `structured_episodic.py` | ✅ v0, typed `Episode(situation, action, outcome, tags)` with structured-key recall |

All five share the same `add(msg) / view() -> list[Message]` interface. Two add explicit recall: `vector_retrieval.query(text, k)` and `structured_episodic.recall_episodes(situation, tags, k)`.

## Design rules

1. One file per pattern. Imports only from stdlib, except where absolutely necessary (e.g. vector retrieval needs an embedder — document the choice).
2. Same public interface. New entry points get a strong reason in the PR.
3. Behaviour documented in the docstring, including *when not to use it*. Patterns have failure modes; name them.
4. A `__main__`-guarded `_demo()` function that shows the pattern in one terminal run.
