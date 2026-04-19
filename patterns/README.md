# Patterns

Each file is a single memory pattern. Pick the one that matches your agent's failure mode.

## Selection guide

Ask yourself: **what kind of information does my agent lose most often?**

- "It forgets stuff from 20 turns ago but I'm cost-constrained" → `sliding_window.py`
- "It forgets old stuff but I can afford one LLM call per N turns" → `summary_compression.py` (planned)
- "It has to search a large set of past facts, not all of them" → `vector_retrieval.py` (planned)
- "Sessions are days long, and recall needs to span them" → `hierarchical_summary.py` (planned)
- "I need to recall structured facts (user preferences, past decisions) across sessions" → `structured_episodic.py` (planned)

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
| Sliding window | `sliding_window.py` | ✅ v0, tests coming |
| Summary compression | `summary_compression.py` | ✅ v0, pluggable summarizer + mock for demo |
| Vector retrieval | `vector_retrieval.py` | 🚧 skeleton pending |
| Hierarchical summary | `hierarchical_summary.py` | 🚧 skeleton pending |
| Structured episodic | `structured_episodic.py` | 🚧 skeleton pending |

## Design rules

1. One file per pattern. Imports only from stdlib, except where absolutely necessary (e.g. vector retrieval needs an embedder — document the choice).
2. Same public interface. New entry points get a strong reason in the PR.
3. Behaviour documented in the docstring, including *when not to use it*. Patterns have failure modes; name them.
4. A `__main__`-guarded `_demo()` function that shows the pattern in one terminal run.
