"""
Pattern 15 — McpAgentAuth: MCP agent credential lifecycle middleware.

Anchored to RP-9 (SecurityWeek Claude Code OAuth theft via MCP hijacking;
PocketOS production DB wiped in <10 s via static agent credential;
arxiv 2604.23280 OAuth 2.0 multi-hop delegation gap; Auth0 MCP Auth GA
May 2026 confirms the problem at platform layer — pip-installable gap open).

Three composable primitives (stdlib only, zero external deps):

  SubAgentToken       — short-lived scoped credential derived from a parent;
                        scope can only narrow, TTL can only shorten vs parent
  DelegationChain     — immutable thread-safe audit log (issued / tool_call /
                        revoked events) serialisable to JSON for compliance
  RevocationPropagator — cascades revocations parent → all registered children;
                         revoke_all() cleans up the entire agent session

Signing backend: HMAC-SHA256 via stdlib (hashlib + hmac).  The sign/verify
pair in SigningKey is isolated so a production deployment can swap in
SPIFFE/WIMSE or cloud IAM token exchange without touching the rest of the API.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# SigningKey — HMAC-SHA256 self-signed tokens (dev/test backend)
# ---------------------------------------------------------------------------

class SigningKey:
    """HMAC-SHA256 signing key.  Produces self-verifying token strings without
    any external crypto library.  Auto-generates a cryptographically random
    32-byte secret when none is provided."""

    def __init__(self, secret: Optional[bytes] = None) -> None:
        self._secret = secret or secrets.token_bytes(32)

    def sign(self, claims: Dict[str, Any]) -> str:
        payload = (
            base64.urlsafe_b64encode(
                json.dumps(claims, sort_keys=True).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        sig = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}.{sig}"

    def verify(self, token_str: str) -> Optional[Dict[str, Any]]:
        """Return the claims dict if signature is valid, else None."""
        try:
            payload, sig = token_str.rsplit(".", 1)
            expected = hmac.new(
                self._secret, payload.encode(), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(sig, expected):
                return None
            padded = payload + "=" * (-len(payload) % 4)
            return json.loads(base64.urlsafe_b64decode(padded))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# SubAgentToken
# ---------------------------------------------------------------------------

class SubAgentToken:
    """Short-lived scoped credential derived from a parent token.

    Two privilege-escalation invariants are enforced at construction time:
      1. Scope narrows — effective scope = requested ∩ parent scope.
      2. TTL shortens  — effective expiry = min(now + ttl_secs, parent.expires_at).

    Revocation fires registered callbacks exactly once and never raises.
    """

    def __init__(
        self,
        agent_id: str,
        ttl_secs: float,
        scope_mask: Optional[List[str]] = None,
        parent: Optional["SubAgentToken"] = None,
        signing_key: Optional[SigningKey] = None,
    ) -> None:
        self._signing_key = signing_key or (
            parent._signing_key if parent is not None else SigningKey()
        )
        now = time.time()

        # Scope narrowing
        if parent is not None:
            parent_scopes: set = set(parent.claims["scope"])
            requested: set = set(scope_mask) if scope_mask is not None else parent_scopes
            effective_scope = sorted(requested & parent_scopes)
        else:
            effective_scope = sorted(scope_mask) if scope_mask is not None else []

        # TTL shortening
        parent_exp = parent.claims["exp"] if parent is not None else float("inf")
        expires_at = min(now + ttl_secs, parent_exp)

        self.claims: Dict[str, Any] = {
            "tid": secrets.token_hex(16),
            "pid": parent.claims["tid"] if parent is not None else None,
            "agent": agent_id,
            "iat": now,
            "exp": expires_at,
            "scope": effective_scope,
        }
        self._token_str = self._signing_key.sign(self.claims)
        self._revoked = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[["SubAgentToken"], None]] = []

    # --- identity / lifetime ---

    @property
    def token_id(self) -> str:
        return self.claims["tid"]

    @property
    def agent_id(self) -> str:
        return self.claims["agent"]

    @property
    def expires_at(self) -> float:
        return self.claims["exp"]

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.claims["exp"]

    @property
    def is_revoked(self) -> bool:
        return self._revoked

    @property
    def is_valid(self) -> bool:
        return not self._revoked and not self.is_expired

    # --- scope ---

    def has_scope(self, scope: str) -> bool:
        return scope in self.claims["scope"]

    # --- revocation ---

    def on_revoke(self, callback: Callable[["SubAgentToken"], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def revoke(self) -> None:
        with self._lock:
            if self._revoked:
                return
            self._revoked = True
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(self)
            except Exception:
                pass

    # --- serialisation ---

    def as_str(self) -> str:
        """Signed token string for transmission to a sub-agent."""
        return self._token_str

    def __repr__(self) -> str:
        status = (
            "valid" if self.is_valid else ("revoked" if self._revoked else "expired")
        )
        ttl = max(0.0, self.claims["exp"] - time.time())
        return (
            f"SubAgentToken(agent={self.claims['agent']!r}, "
            f"status={status}, ttl={ttl:.1f}s, "
            f"scopes={self.claims['scope']})"
        )


# ---------------------------------------------------------------------------
# DelegationChain
# ---------------------------------------------------------------------------

@dataclass
class DelegationEvent:
    event_type: str          # "issued" | "tool_call" | "revoked"
    timestamp: float
    agent_id: str
    token_id: str
    parent_id: Optional[str] = None
    tool_name: Optional[str] = None
    scope: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DelegationChain:
    """Immutable thread-safe audit log of orchestrator → sub-agent → tool events.

    Serialises to JSON for compliance hand-off and postmortem analysis.
    Designed to be passed to RevocationPropagator so every token lifecycle
    event (issued, tool_call, revoked) is captured automatically.
    """

    def __init__(self) -> None:
        self._events: List[DelegationEvent] = []
        self._lock = threading.Lock()

    def record_issued(self, token: SubAgentToken) -> None:
        self._append(DelegationEvent(
            event_type="issued",
            timestamp=time.time(),
            agent_id=token.agent_id,
            token_id=token.token_id,
            parent_id=token.claims.get("pid"),
            scope=list(token.claims["scope"]),
        ))

    def record_tool_call(
        self,
        token: SubAgentToken,
        tool_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._append(DelegationEvent(
            event_type="tool_call",
            timestamp=time.time(),
            agent_id=token.agent_id,
            token_id=token.token_id,
            tool_name=tool_name,
            metadata=metadata or {},
        ))

    def record_revoked(self, token: SubAgentToken) -> None:
        self._append(DelegationEvent(
            event_type="revoked",
            timestamp=time.time(),
            agent_id=token.agent_id,
            token_id=token.token_id,
        ))

    def to_json(self) -> str:
        with self._lock:
            return json.dumps(
                [
                    {
                        k: v
                        for k, v in {
                            "event_type": e.event_type,
                            "timestamp": e.timestamp,
                            "agent_id": e.agent_id,
                            "token_id": e.token_id,
                            "parent_id": e.parent_id,
                            "tool_name": e.tool_name,
                            "scope": e.scope,
                            "metadata": e.metadata or None,
                        }.items()
                        if v is not None
                    }
                    for e in self._events
                ],
                indent=2,
            )

    def events_for_token(self, token_id: str) -> List[DelegationEvent]:
        with self._lock:
            return [e for e in self._events if e.token_id == token_id]

    def events_for_agent(self, agent_id: str) -> List[DelegationEvent]:
        with self._lock:
            return [e for e in self._events if e.agent_id == agent_id]

    def _append(self, event: DelegationEvent) -> None:
        with self._lock:
            self._events.append(event)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


# ---------------------------------------------------------------------------
# RevocationPropagator
# ---------------------------------------------------------------------------

class RevocationPropagator:
    """Cascades token revocations from parent to all registered descendants.

    Call register() on every SubAgentToken issued during a session.
    When any token is revoked (timeout, exception, explicit call), every
    registered child of that token is revoked automatically.
    Call revoke_all() at session end to drain all live credentials.

    Optionally accepts a DelegationChain to auto-record issued and revoked
    events without extra bookkeeping in caller code.
    """

    def __init__(self, chain: Optional[DelegationChain] = None) -> None:
        self._chain = chain
        self._children: Dict[str, List[SubAgentToken]] = {}
        self._all: List[SubAgentToken] = []
        self._lock = threading.Lock()

    def register(self, token: SubAgentToken) -> None:
        """Wire up parent→child propagation and log to DelegationChain if set."""
        if self._chain is not None:
            self._chain.record_issued(token)

        with self._lock:
            self._all.append(token)
            parent_id = token.claims.get("pid")
            if parent_id:
                self._children.setdefault(parent_id, []).append(token)

        token.on_revoke(self._propagate)
        if self._chain is not None:
            token.on_revoke(self._chain.record_revoked)

    def _propagate(self, revoked_token: SubAgentToken) -> None:
        with self._lock:
            children = list(self._children.get(revoked_token.token_id, []))
        for child in children:
            if not child.is_revoked:
                child.revoke()

    def revoke_all(self) -> int:
        """Revoke every registered token that is still alive.  Returns count."""
        with self._lock:
            targets = [t for t in self._all if not t.is_revoked]
        count = 0
        for token in targets:
            token.revoke()
            count += 1
        return count

    def active_count(self) -> int:
        """Number of registered tokens that are currently valid."""
        with self._lock:
            return sum(1 for t in self._all if t.is_valid)

    def __repr__(self) -> str:
        with self._lock:
            total = len(self._all)
        return f"RevocationPropagator(total={total}, active={self.active_count()})"
