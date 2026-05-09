"""
McpAgentAuth demo — offline, no API calls required.

Anchored to RP-9 (SecurityWeek: Claude Code OAuth tokens stolen via MCP
hijacking; PocketOS production DB wiped in <10 s by agent holding a
blanket-authority static secret; arxiv 2604.23280 — OAuth 2.0 handles
one-hop delegation but not multi-hop agent chains; Auth0 MCP Auth GA
May 2026 confirms market-layer validation; pip-installable middleware
gap confirmed open).

Four scenarios:
  1. Scope narrowing — child tokens can only hold a subset of parent scopes;
     attempted privilege escalation is silently clamped at construction time.
  2. Parent revocation cascade — revoking the orchestrator token automatically
     revokes all registered descendants within the session.
  3. Delegation chain audit log — every issued/tool_call/revoked event is
     captured and exported as compliance-ready JSON.
  4. Session cleanup — revoke_all() drains every live credential at session end;
     active_count() confirms no live tokens remain.

Run: python examples/mcp_agent_auth_demo.py
"""
from __future__ import annotations

import json
import time

from patterns.mcp_agent_auth import (
    DelegationChain,
    RevocationPropagator,
    SigningKey,
    SubAgentToken,
)


# ---------------------------------------------------------------------------
# Scenario 1: Scope narrowing — no privilege escalation possible
# ---------------------------------------------------------------------------

def scenario_scope_narrowing() -> None:
    print("=" * 65)
    print("SCENARIO 1 — Scope narrowing (privilege escalation blocked)")
    print("=" * 65)
    print("Context: PocketOS production DB wiped in <10 s because an agent")
    print("held a blanket-authority credential with read+write+delete+admin.")
    print("SubAgentToken enforces scope ⊆ parent_scope at construction time.")
    print()

    key = SigningKey()

    # Orchestrator gets a full-authority session token
    orch = SubAgentToken(
        "orchestrator",
        ttl_secs=300,
        scope_mask=["read", "write", "delete", "admin"],
        signing_key=key,
    )
    print(f"  Orchestrator token : {orch}")

    # Sub-agent A: researcher — needs only read
    researcher = SubAgentToken(
        "agent:researcher",
        ttl_secs=60,
        scope_mask=["read"],
        parent=orch,
    )
    print(f"  Researcher token   : {researcher}")

    # Sub-agent B: writer — needs read+write but NOT delete/admin
    writer = SubAgentToken(
        "agent:writer",
        ttl_secs=60,
        scope_mask=["read", "write"],
        parent=orch,
    )
    print(f"  Writer token       : {writer}")

    # Sub-agent C: attempts privilege escalation — requests delete+admin
    # even though its parent (writer) only holds read+write
    escalator = SubAgentToken(
        "agent:escalator",
        ttl_secs=60,
        scope_mask=["delete", "admin", "read"],  # tries to include delete/admin
        parent=writer,                            # writer's parent lacks these
    )
    print(f"  Escalator token    : {escalator}")
    print()

    # Assertions
    assert researcher.has_scope("read"), "researcher must have read"
    assert not researcher.has_scope("write"), "researcher must NOT have write"
    assert not researcher.has_scope("delete"), "researcher must NOT have delete"

    assert writer.has_scope("read") and writer.has_scope("write")
    assert not writer.has_scope("delete") and not writer.has_scope("admin")

    # Escalation attempt: effective scope = {"delete","admin","read"} ∩ {"read","write"} = {"read"}
    assert escalator.has_scope("read"), "escalator retains read from intersection"
    assert not escalator.has_scope("delete"), "escalation to delete BLOCKED"
    assert not escalator.has_scope("admin"), "escalation to admin BLOCKED"

    print("  [PASS] researcher: read only — write/delete/admin blocked")
    print("  [PASS] writer: read+write — delete/admin blocked")
    print("  [PASS] escalator: requested delete+admin, clamped to read only")
    print()


# ---------------------------------------------------------------------------
# Scenario 2: Parent revocation cascades to all children
# ---------------------------------------------------------------------------

def scenario_cascade_revocation() -> None:
    print("=" * 65)
    print("SCENARIO 2 — Parent revocation cascade")
    print("=" * 65)
    print("Context: SecurityWeek (2026) — Claude Code OAuth tokens extractable")
    print("via MCP hijacking. If the parent session token is compromised,")
    print("all derived sub-agent tokens must be invalidated immediately.")
    print()

    chain = DelegationChain()
    prop = RevocationPropagator(chain=chain)

    # Build a 3-level delegation tree:
    #   orchestrator → planner → executor-a
    #                           → executor-b
    orch = SubAgentToken("orchestrator", ttl_secs=300,
                         scope_mask=["read", "write", "execute"])
    planner = SubAgentToken("agent:planner", ttl_secs=120,
                            scope_mask=["read", "execute"], parent=orch)
    exec_a = SubAgentToken("agent:exec-a", ttl_secs=60,
                           scope_mask=["execute"], parent=planner)
    exec_b = SubAgentToken("agent:exec-b", ttl_secs=60,
                           scope_mask=["execute"], parent=planner)

    for tok in (orch, planner, exec_a, exec_b):
        prop.register(tok)

    print(f"  Before revocation: active={prop.active_count()} tokens")

    # Record some tool calls before revocation
    chain.record_tool_call(exec_a, "read_file", {"path": "/workspace/plan.md"})
    chain.record_tool_call(exec_b, "run_python", {"snippet": "compute_result()"})

    # Compromise detected — revoke the orchestrator token
    print("  [!] Orchestrator token compromised — revoking ...")
    orch.revoke()

    print(f"  After revocation : active={prop.active_count()} tokens")
    print()

    assert orch.is_revoked, "orchestrator must be revoked"
    assert planner.is_revoked, "planner must cascade-revoke"
    assert exec_a.is_revoked, "exec-a must cascade-revoke"
    assert exec_b.is_revoked, "exec-b must cascade-revoke"
    assert prop.active_count() == 0, "zero live tokens after cascade"

    print("  [PASS] revoking orchestrator cascaded to planner, exec-a, exec-b")
    print("  [PASS] active_count() == 0 (all credentials drained)")
    print()


# ---------------------------------------------------------------------------
# Scenario 3: Delegation chain audit log
# ---------------------------------------------------------------------------

def scenario_audit_log() -> None:
    print("=" * 65)
    print("SCENARIO 3 — Delegation chain audit log (compliance export)")
    print("=" * 65)
    print("Context: arxiv 2604.23280 — 'OAuth 2.0 handles one-hop delegation")
    print("well but lacks multi-hop chaining, cross-domain async flows, and")
    print("any mapping between OAuth scopes and agent capabilities.'")
    print("DelegationChain provides the missing audit trail.")
    print()

    chain = DelegationChain()
    prop = RevocationPropagator(chain=chain)

    orch = SubAgentToken("orchestrator", ttl_secs=120,
                         scope_mask=["read", "write"])
    worker = SubAgentToken("agent:worker", ttl_secs=30,
                           scope_mask=["read"], parent=orch)

    for tok in (orch, worker):
        prop.register(tok)

    # Simulate tool usage
    chain.record_tool_call(orch, "list_files", {"dir": "/workspace"})
    chain.record_tool_call(worker, "read_file", {"path": "/workspace/notes.md"})
    chain.record_tool_call(worker, "read_file", {"path": "/workspace/config.yaml"})

    # Revoke worker mid-session
    worker.revoke()

    # Export compliance log
    log_json = chain.to_json()
    events = json.loads(log_json)

    print(f"  Total audit events : {len(events)}")
    event_types = [e["event_type"] for e in events]
    agents_seen = sorted({e["agent_id"] for e in events})
    print(f"  Event types        : {event_types}")
    print(f"  Agents recorded    : {agents_seen}")
    print()

    # Per-agent filtering
    orch_events = chain.events_for_agent("orchestrator")
    worker_events = chain.events_for_agent("agent:worker")
    print(f"  Orchestrator events : {[e.event_type for e in orch_events]}")
    print(f"  Worker events       : {[e.event_type for e in worker_events]}")
    print()

    # Assertions
    issued_agents = [e["agent_id"] for e in events if e["event_type"] == "issued"]
    assert "orchestrator" in issued_agents
    assert "agent:worker" in issued_agents

    tool_calls = [e for e in events if e["event_type"] == "tool_call"]
    assert len(tool_calls) == 3, f"expected 3 tool_call events, got {len(tool_calls)}"

    revoked = [e for e in events if e["event_type"] == "revoked"]
    assert len(revoked) == 1 and revoked[0]["agent_id"] == "agent:worker"

    # Parent/child linkage preserved in log
    worker_issued = next(e for e in events
                         if e["event_type"] == "issued" and e["agent_id"] == "agent:worker")
    orch_issued = next(e for e in events
                       if e["event_type"] == "issued" and e["agent_id"] == "orchestrator")
    assert worker_issued["parent_id"] == orch_issued["token_id"], (
        "worker's parent_id must match orchestrator's token_id in the log"
    )

    print("  [PASS] 2 issued + 3 tool_call + 1 revoked events captured")
    print("  [PASS] parent_id linkage preserved for full delegation tree reconstruction")
    print("  [PASS] events filterable by agent_id for postmortem analysis")
    print()


# ---------------------------------------------------------------------------
# Scenario 4: Session cleanup — revoke_all() drains all live credentials
# ---------------------------------------------------------------------------

def scenario_session_cleanup() -> None:
    print("=" * 65)
    print("SCENARIO 4 — Session cleanup (revoke_all at session end)")
    print("=" * 65)
    print("Context: GitGuardian (April 2026) — 64% of agent credentials")
    print("confirmed valid four years after detection; 'ephemeral credential")
    print("broker model' = credentials expire at end of the owning session.")
    print("revoke_all() implements this at the Python process boundary.")
    print()

    prop = RevocationPropagator()

    tokens = []
    # Orchestrator
    root = SubAgentToken("orchestrator", ttl_secs=300,
                         scope_mask=["read", "write", "execute"])
    prop.register(root)
    tokens.append(root)

    # Three concurrent sub-agents with different scopes
    for name, scopes in [
        ("agent:search",  ["read"]),
        ("agent:writer",  ["read", "write"]),
        ("agent:runner",  ["execute"]),
    ]:
        tok = SubAgentToken(name, ttl_secs=60,
                            scope_mask=scopes, parent=root)
        prop.register(tok)
        tokens.append(tok)

    print(f"  Active before cleanup : {prop.active_count()}")

    revoked_count = prop.revoke_all()

    print(f"  Revoked by revoke_all : {revoked_count}")
    print(f"  Active after cleanup  : {prop.active_count()}")
    print()

    assert prop.active_count() == 0, "all tokens must be dead after revoke_all"
    assert revoked_count == 4, f"expected 4 revoked, got {revoked_count}"
    for tok in tokens:
        assert tok.is_revoked, f"{tok.agent_id} must be revoked"
        assert not tok.is_valid, f"{tok.agent_id} must not be valid"

    print("  [PASS] revoke_all() revoked 4 tokens (1 orchestrator + 3 sub-agents)")
    print("  [PASS] active_count() == 0 — no live credentials remain post-session")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("McpAgentAuth demo")
    print("Pairs with Pattern 15 in agent-memory-lab (RP-9)")
    print()

    scenario_scope_narrowing()
    scenario_cascade_revocation()
    scenario_audit_log()
    scenario_session_cleanup()

    print("=" * 65)
    print("All scenarios passed.")
    print("=" * 65)
