# Pattern 6 — long-horizon (planned, design notes)

> **Status: design notes only.** No code yet. This file is the spec so
> the eventual implementation has a target. Tracked separately because
> the failure mode and mitigations are larger than any single existing
> pattern in this repo.

## What it's for

Agents that run for hours-to-days — long-horizon coding agents,
multi-day research synthesisers, autonomous ops debuggers. Their
defining property is **lossy state compression that accumulates over
time**: every existing pattern in this repo (sliding window,
summary compression, hierarchical summary, vector retrieval, structured
episodic) is "good enough for one session". Long-horizon is what
happens when "session" becomes a unit you stop being able to ignore.

The exact failure mode the pattern needs to mitigate is documented in
[`agentic-anti-patterns#AP-21 — Long-horizon agent state collapse`](https://github.com/jimliu741523/agentic-anti-patterns#ap-21--long-horizon-agent-state-collapse).
This pattern is the architectural answer.

## Key invariants the pattern must hold

1. **Provenance through rollups.** Every fact that survives a rollup
   carries a citation back to the tool call / scraped URL / subagent
   transcript / earlier memory entry that produced it. A fact whose
   provenance has been compressed away is treated as suspect.
2. **Outcome-preserving compression.** When a hypothesis or sub-task is
   rolled up, *what happened to it* is preserved. "Hypothesis A:
   rejected because X failed" remains addressable. Plain "investigated
   A" is not an acceptable rollup of an attempted-and-rejected step.
3. **Checkpoint + rollback as first-class.** Snapshots of full state at
   regular cadence; rollback to a prior checkpoint is a primitive the
   planner can choose. Without this, "compound the error" is the only
   path forward when state has drifted.
4. **Liveness ≠ progress.** The pattern surfaces a metric that lets the
   *outside* observer distinguish "agent is producing tokens" from
   "agent is producing correct tokens against the original spec". The
   distinction has to be machine-readable, not just narratively
   inferrable.

## Proposed shape (to be implemented)

```python
@dataclass
class Episode:
    fact: str
    sources: tuple[Provenance, ...]   # ← never empty; provenance carried through
    outcome: Optional[str] = None     # ← what happened to this fact / hypothesis

class LongHorizon:
    keep_recent: int
    checkpoint_every: int  # turns
    summarize: SummarizeFn  # outcome-preserving signature
    drift_gate: SpecDriftFn  # called every N turns; can raise PauseForReview

    def add(self, msg: Message) -> None: ...
    def view(self) -> list[Message]: ...
    def checkpoint(self) -> CheckpointID: ...
    def rollback(self, to: CheckpointID) -> None: ...
    def query(self, text: str, k: int) -> list[Episode]: ...
```

The `SummarizeFn` signature changes from `list[Message] -> str` (used
by the other four summarising patterns) to
`list[Episode] -> list[Episode]` — the summariser is required to
*emit* episodes-with-provenance, not opaque prose. This is the
load-bearing API choice; everything else follows from it.

## Bench plan (for when this lands)

The current `bench/run.py` measures recall at turn 50. Long-horizon
extends this:

- **Provenance survival**: at hour H, can the agent name the original
  source of a fact it injected at hour 0?
- **Contradiction detection rate**: rebuttal pass at hour H — ask the
  agent to argue against its current top conclusion using only its
  own memory. Sharp coherence drop indicates decoherence.
- **Rollback cost**: the cost-of-recovery from a deliberately-injected
  bad branch at hour K. A pattern that supports rollback should make
  this much cheaper than a pattern that doesn't.
- **Spec-drift gate hit rate**: how often does the drift gate
  correctly pause when the agent's current state has wandered from
  the original task spec.

These will require a synthetic harness that simulates many turns
without spending real-LLM tokens. Probably extends `bench/run.py`
with `--horizon` flag.

## Why not just "hierarchical summary plus vector retrieval"?

Both already exist in this repo. Composing them gets you closer than
either alone, but neither solves:
- provenance preservation through cascading rollups (hierarchical
  drops it; vector retrieval was never about provenance to begin with)
- outcome-preserving compression (the existing `summarize_fn` shape
  encourages opaque prose)
- checkpoint / rollback as first-class operations (no existing pattern
  exposes them)
- a drift-against-original-spec gate (out of scope of any single
  pattern)

A long-horizon pattern is a different shape, not a parameter sweep
over the existing ones.

## Status / next steps

- [ ] Land `Provenance` + `Episode` dataclasses in `patterns/long_horizon.py`
- [ ] Implement `LongHorizon` with rollup that preserves outcome
- [ ] Add checkpoint/rollback primitives backed by deep-copy snapshots
- [ ] Stub a `SpecDriftFn` interface (real implementation comes later)
- [ ] Extend bench with `--horizon` mode
- [ ] Add to CHANGELOG once `make test` covers all four invariants

This file gets deleted once the implementation lands and the relevant
content moves into the module docstring + the main README.
