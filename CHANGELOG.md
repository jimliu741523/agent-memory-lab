# CHANGELOG

Newest on top. Tracks user-visible changes to patterns, the bench, and the test suite. Cosmetic doc edits are not logged.

## 2026-05-04
- **`bench/run.py --multi-seed N`** added — aggregate recall pass-rate and char range across N filler shuffles. 10-seed run confirms recall behaviour is stable: vector_retrieval and structured_episodic hit 10/10, recency-only patterns 0/10.
- **`bench/run.py --seed`** added — reproducible filler shuffling. Same seed = same script.
- **`patterns/__init__.py`** added — top-level imports (`from patterns import SlidingWindow, ...`).

## 2026-04-30
- **`bench/run.py` v0** shipped — deterministic stdlib-only 50-turn recall harness. Bumps `bench/` from "plan only" to v0.
- README: results table embedded with measured numbers.

## 2026-04-23
- **structured_episodic** pattern (5 of 5) added — typed `Episode(situation, action, outcome, tags)` with structured-key recall. The 5-pattern set is now complete.
- 7 new tests; suite at 26/26.

## 2026-04-21
- **vector_retrieval** pattern (4 of 5) added — pluggable embedder + stdlib hash-BOW fallback; `query()` for semantic recall over an archive of older messages.

## 2026-04-20
- **hierarchical_summary** pattern (3 of 5) added — pyramid of rolling summaries with cascading fanout, graceful detail decay over very long sessions.
- Cross-link section to `agentic-anti-patterns` (AP-05 context bloat, AP-08 memory poisoning).

## 2026-04-19
- **summary_compression** pattern (2 of 5) added — pluggable summarizer + mock for offline runs.
- **sliding_window** pattern (1 of 5) shipped at repo creation.
- Initial commit: README, LICENSE, two patterns, stdlib unittest suite.
