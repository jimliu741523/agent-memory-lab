# CHANGELOG

Newest on top. Tracks user-visible changes to patterns, the bench, and the test suite. Cosmetic doc edits are not logged.

## 2026-05-14
- **`bench/run.py --model NAME`** added — the long-promised real-LLM bench mode. When set, `summary_compression` and `hierarchical_summary` use the real Anthropic model as their `summarize_fn` callback instead of the mock. Lazy-imports `anthropic`; clean SystemExit if SDK or `ANTHROPIC_API_KEY` is missing. Default (mock) mode unchanged — still deterministic stdlib-only. The bench now measures whether *summarization quality* (not just size) affects downstream recall — previously testable only by reading code.

## 2026-05-09
- **`examples/mcp_agent_auth_demo.py`** added — four offline scenarios for Pattern 15 (McpAgentAuth, RP-9): (1) scope narrowing — privilege-escalation attempt clamped at construction time (researcher/writer/escalator all receive ⊆ parent scope); (2) parent revocation cascade — revoking the orchestrator token immediately invalidates all registered descendants (4→0 active in one call); (3) delegation chain audit log — 6-event compliance export (2 issued + 3 tool_call + 1 revoked), parent_id linkage preserved for full delegation-tree reconstruction; (4) session cleanup — `revoke_all()` drains 4 live credentials at session end, `active_count() == 0` confirmed. All assertions pass.
- **`patterns/mcp_agent_auth.py`** added — `McpAgentAuth` MCP agent credential lifecycle middleware (Pattern 15). Three composable stdlib-only primitives: `SubAgentToken` (short-lived scoped credential derived from a parent token; two privilege-escalation invariants enforced at construction — scope can only narrow via intersection with parent scopes, TTL can only shorten vs parent remaining lifetime; HMAC-SHA256 self-signed token string for transmission; revoke() fires callbacks exactly once, thread-safe); `DelegationChain` (immutable thread-safe audit log of issued/tool_call/revoked events; `to_json()` for compliance hand-off; `events_for_token()` / `events_for_agent()` for postmortem filtering); `RevocationPropagator` (cascades revocations from parent to all registered descendants; `revoke_all()` for session cleanup; optionally auto-records all lifecycle events to a DelegationChain). Addresses RP-9 (SecurityWeek Claude Code OAuth theft via MCP hijacking; PocketOS production DB wipe in <10 s via leaked static credential; arxiv 2604.23280 OAuth 2.0 multi-hop delegation gap; Auth0 MCP Auth GA May 2026 confirms platform-layer market, pip-installable middleware gap remains open). 51 new tests; 416/416 suite passes. Stdlib only, zero external deps.
- **`patterns/tool_schema_compiler.py`** added — `ToolSchemaCompiler` tool-schema wire-format adaptation (Pattern 14). Compiles verbose JSON Schema tool definitions to token-efficient formats matched to the target model family's parsing capability. Addresses the 0%→84.4% Phi-4 14B accuracy gap (arxiv 2605.04107 TSCG) and the 14-month-open mcp-sdk adapter request (#235, 18 reactions). Four TSCG operator types: `BooleanOp`, `NumericOp`, `SelectionOp`, `CompositeOp`. Three pluggable output formats: `structured_text` (default; best for small models), `compressed_json` (minified; reliable for GPT-4/Claude), `markdown_table` (chain-of-thought readable). Built-in per-model profile registry maps phi/qwen/llama→structured_text, gpt4/claude→compressed_json; fully overridable via `register_model_profile()`. Schema validation runs before compilation and catches FastMCP #226-class bugs (malformed required field, broken enum list, min>max). `compile_all()` batch-compiles all registered tools. `classify_params()` exposes per-parameter operator-type introspection. 69 new tests; 365/365 suite passes. Stdlib only, zero external deps. Anchors RP-8.

## 2026-05-08
- **`patterns/mcp_pipe.py`** added — `McpPipe` tool-call composition middleware (Pattern 13). `ResultHandle` (opaque content-addressed ID — SHA-256 hash + UUID), `HandleStore` (thread-safe store), `McpPipe` orchestrator: `call` executes any callable and stores result as a handle; `transform` applies a dot-path or callable to produce a new handle; `pipe` chains two tools without surfacing the intermediate to the LLM; `materialize` is the single point where data surfaces; `token_savings` reports bytes kept out of context. Dot-path supports dict key traversal (`.key.sub`), list index (`.[0]`), mixed (`.results.[0].url`), and plain callables — stdlib only. 63 new tests; 296/296 suite passes. Addresses RP-2 (tool-call composition without round-tripping through context).
- **`patterns/agent_tx.py`** added — `SagaLog` saga-pattern compensation middleware (Pattern 10). Three primitives: `@compensable(compensation_fn)` decorator pairs any tool with its rollback function; `SagaLog` records each successful call in order; `run_saga(log)` context manager auto-compensates in reverse on any exception. Two pluggable backends: `MemorySagaBackend` (default, thread-safe) and `SQLiteSagaBackend` (crash-durable audit trail; stores tool names durably so an operator can audit which tools ran before a process crash). Fixes a falsy-backend footgun (`or` vs `is not None` guard). 36 new tests; suite at 127/127. Addresses RP-7 (agentic side-effect atomicity — compensation-safe tool execution).
- **`examples/agent_tx_demo.py`** added — three offline scenarios: (1) happy path (no compensation); (2) 4-step workflow fails at step 4, steps 1–3 rolled back in reverse order, system left clean; (3) `SQLiteSagaBackend` audit trail demo. All assertions pass.

## 2026-05-07
- **`patterns/checkpoint.py`** added — `CheckpointManager` framework-agnostic checkpoint / pause / resume primitive (Pattern 8). Portable `RunState` JSON schema (run_id, step, tool_queue, pending, in-flight calls, memory_snap, stream_cursor, metadata). `save(run) -> token` and `load(token) -> run` with content-addressed SHA-256 tokens; `advance()` shortcut for step advancement. `MemoryBackend` (in-process dict) and `FileBackend` (one JSON file per checkpoint) included. 14 new tests; suite at 52/52. Anchors RP-1 (durable agent runs).
- **`patterns/memory_writer.py`** added — `MemoryWriter` write-path middleware (Pattern 7). Three composable write-path primitives: salience-gating (configurable `SalienceFn`, drops low-signal messages before commit), contradiction resolution (`latest_wins` / `merge` / `flag_for_review` strategies via pluggable `ContradictFn`), and TTL decay (turn-counted expiry, implements learned forgetting). Wraps any `add()/view()` backend. `ListBackend` included for demos and tests. 12 new tests; suite at 38/38. Addresses RP-3 (memory write-path: filtering, contradiction handling, learned forgetting).
- **`examples/tool_output_shim.py`** added — demonstrates using `SummaryCompression` as a two-level tool-output filter (per-call compaction + rolling rollup) to mitigate AP-22 (context pollution from raw tool output). Shows 88%+ per-call reduction and rolling context cap across a 10-call debugging session; pairs with the AP-22 entry in `agentic-anti-patterns`.

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
