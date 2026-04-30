# Benchmark

**Status: v0 implemented** in [`run.py`](./run.py). Latest results: [`results/results.md`](./results/results.md). This file documents the spec; the harness now matches it.

## The task

A multi-turn agent task constructed so the final answer depends on information introduced early in the conversation.

- Turns 1–5: background facts injected (user identity, project constraints, key dates)
- Turns 6–49: noise (unrelated chat, tool calls, tangents)
- Turn 50: a question whose correct answer requires the turn-3 fact

Each memory pattern runs the same task; we compare:

| metric | what it measures |
|---|---|
| **recall accuracy** | did the turn-3 fact survive to turn 50? (pass/fail) |
| **tokens per turn (avg)** | cost proxy |
| **wall-clock per turn (p50, p95)** | latency |
| **extra LLM calls** | e.g. summary_compression triggers a summarization call every K turns |

## Reproducibility

```
python bench/run.py --pattern sliding_window --seed 42
python bench/run.py --pattern all --output results.csv
```

- Deterministic: fixed seed, canned LLM responses (no real model calls in the default run).
- Optional real-model run with `--model claude-sonnet-4-6` etc.

## Output

A CSV + a rendered Markdown table checked into `bench/results/`.

## Non-goals

- Production benchmark. This is a micro-bench for making the memory-pattern tradeoff legible, not for ranking agent frameworks.
- Agreement with any specific paper's numbers.
