"""Tests for Pattern 15 — McpAgentAuth (RP-9)."""
import json
import threading
import time
import unittest

from patterns.mcp_agent_auth import (
    DelegationChain,
    DelegationEvent,
    RevocationPropagator,
    SigningKey,
    SubAgentToken,
)


# ---------------------------------------------------------------------------
# SigningKey tests
# ---------------------------------------------------------------------------

class TestSigningKey(unittest.TestCase):

    def test_sign_and_verify_roundtrip(self):
        key = SigningKey()
        claims = {"a": 1, "b": "hello"}
        token_str = key.sign(claims)
        result = key.verify(token_str)
        self.assertEqual(result, claims)

    def test_verify_returns_none_on_tampered_payload(self):
        key = SigningKey()
        token_str = key.sign({"x": 1})
        bad = "AAAA" + token_str[4:]
        self.assertIsNone(key.verify(bad))

    def test_verify_returns_none_on_tampered_signature(self):
        key = SigningKey()
        token_str = key.sign({"x": 1})
        payload, sig = token_str.rsplit(".", 1)
        bad = f"{payload}.{'a' * len(sig)}"
        self.assertIsNone(key.verify(bad))

    def test_verify_returns_none_for_wrong_key(self):
        k1 = SigningKey()
        k2 = SigningKey()
        token_str = k1.sign({"x": 1})
        self.assertIsNone(k2.verify(token_str))

    def test_fixed_secret_is_deterministic(self):
        secret = b"fixed-secret-32-bytes-xxxxxxxxxxx"
        k1 = SigningKey(secret)
        k2 = SigningKey(secret)
        token_str = k1.sign({"y": 2})
        self.assertIsNotNone(k2.verify(token_str))

    def test_verify_returns_none_on_garbage(self):
        key = SigningKey()
        self.assertIsNone(key.verify("notavalidtoken"))
        self.assertIsNone(key.verify(""))

    def test_auto_generated_secret_differs(self):
        k1 = SigningKey()
        k2 = SigningKey()
        self.assertNotEqual(k1._secret, k2._secret)


# ---------------------------------------------------------------------------
# SubAgentToken tests
# ---------------------------------------------------------------------------

class TestSubAgentToken(unittest.TestCase):

    def _root(self, agent_id="orch", ttl=60, scopes=None):
        return SubAgentToken(agent_id, ttl, scope_mask=scopes)

    def test_basic_creation(self):
        tok = self._root(scopes=["read", "write"])
        self.assertEqual(tok.agent_id, "orch")
        self.assertIsNotNone(tok.token_id)
        self.assertTrue(tok.is_valid)
        self.assertFalse(tok.is_revoked)
        self.assertFalse(tok.is_expired)

    def test_is_valid_initially(self):
        tok = self._root()
        self.assertTrue(tok.is_valid)

    def test_revoke_makes_invalid(self):
        tok = self._root()
        tok.revoke()
        self.assertFalse(tok.is_valid)
        self.assertTrue(tok.is_revoked)

    def test_revoke_idempotent(self):
        tok = self._root()
        tok.revoke()
        tok.revoke()  # should not raise
        self.assertTrue(tok.is_revoked)

    def test_callback_fires_on_revoke(self):
        fired = []
        tok = self._root()
        tok.on_revoke(lambda t: fired.append(t.token_id))
        tok.revoke()
        self.assertEqual(fired, [tok.token_id])

    def test_callback_fires_exactly_once(self):
        fired = []
        tok = self._root()
        tok.on_revoke(lambda t: fired.append(1))
        tok.revoke()
        tok.revoke()
        self.assertEqual(fired, [1])

    def test_callback_exception_does_not_propagate(self):
        tok = self._root()
        tok.on_revoke(lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
        tok.revoke()  # should not raise
        self.assertTrue(tok.is_revoked)

    def test_has_scope_true(self):
        tok = self._root(scopes=["read", "write"])
        self.assertTrue(tok.has_scope("read"))
        self.assertTrue(tok.has_scope("write"))

    def test_has_scope_false(self):
        tok = self._root(scopes=["read"])
        self.assertFalse(tok.has_scope("delete"))

    def test_scope_narrowing_from_parent(self):
        parent = self._root(scopes=["read", "write", "delete"])
        child = SubAgentToken("worker", 30, scope_mask=["read", "delete", "admin"],
                              parent=parent)
        # Only read and delete (intersection); admin not in parent → excluded
        self.assertIn("read", child.claims["scope"])
        self.assertIn("delete", child.claims["scope"])
        self.assertNotIn("admin", child.claims["scope"])
        self.assertNotIn("write", child.claims["scope"])

    def test_scope_defaults_to_parent_scope_when_none(self):
        parent = self._root(scopes=["read", "write"])
        child = SubAgentToken("worker", 30, parent=parent)
        self.assertEqual(sorted(child.claims["scope"]), ["read", "write"])

    def test_ttl_shortens_relative_to_parent(self):
        parent = SubAgentToken("orch", ttl_secs=100, scope_mask=["read"])
        child = SubAgentToken("worker", ttl_secs=200, parent=parent)
        # child ttl would be 200 s but parent only lives 100 s — cap at parent
        self.assertLessEqual(child.expires_at, parent.expires_at + 0.01)

    def test_ttl_shorter_than_parent_is_honoured(self):
        parent = SubAgentToken("orch", ttl_secs=100, scope_mask=["read"])
        child = SubAgentToken("worker", ttl_secs=10, parent=parent)
        self.assertAlmostEqual(
            child.expires_at, time.time() + 10, delta=2
        )

    def test_parent_id_recorded(self):
        parent = self._root()
        child = SubAgentToken("worker", 30, parent=parent)
        self.assertEqual(child.claims["pid"], parent.token_id)

    def test_root_token_has_no_parent_id(self):
        tok = self._root()
        self.assertIsNone(tok.claims["pid"])

    def test_as_str_verifiable(self):
        key = SigningKey()
        tok = SubAgentToken("orch", 60, scope_mask=["read"], signing_key=key)
        verified = key.verify(tok.as_str())
        self.assertIsNotNone(verified)
        self.assertEqual(verified["agent"], "orch")

    def test_signing_key_inherited_from_parent(self):
        key = SigningKey()
        parent = SubAgentToken("orch", 60, signing_key=key)
        child = SubAgentToken("worker", 30, parent=parent)
        # child should be verifiable by the same key
        verified = key.verify(child.as_str())
        self.assertIsNotNone(verified)

    def test_repr_contains_status(self):
        tok = self._root()
        self.assertIn("valid", repr(tok))
        tok.revoke()
        self.assertIn("revoked", repr(tok))

    def test_thread_safe_revoke(self):
        tok = self._root()
        fired = []
        tok.on_revoke(lambda t: fired.append(1))
        threads = [threading.Thread(target=tok.revoke) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # callback must fire exactly once despite concurrent revokes
        self.assertEqual(fired, [1])


# ---------------------------------------------------------------------------
# DelegationChain tests
# ---------------------------------------------------------------------------

class TestDelegationChain(unittest.TestCase):

    def _tok(self, agent_id="orch", scopes=None):
        return SubAgentToken(agent_id, 60, scope_mask=scopes or ["read"])

    def test_initially_empty(self):
        chain = DelegationChain()
        self.assertEqual(len(chain), 0)

    def test_record_issued(self):
        chain = DelegationChain()
        tok = self._tok()
        chain.record_issued(tok)
        self.assertEqual(len(chain), 1)
        events = chain.events_for_token(tok.token_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "issued")
        self.assertEqual(events[0].agent_id, tok.agent_id)

    def test_record_tool_call(self):
        chain = DelegationChain()
        tok = self._tok()
        chain.record_tool_call(tok, "search_web", {"query": "hello"})
        events = chain.events_for_token(tok.token_id)
        self.assertEqual(events[0].event_type, "tool_call")
        self.assertEqual(events[0].tool_name, "search_web")
        self.assertEqual(events[0].metadata, {"query": "hello"})

    def test_record_revoked(self):
        chain = DelegationChain()
        tok = self._tok()
        chain.record_revoked(tok)
        events = chain.events_for_token(tok.token_id)
        self.assertEqual(events[0].event_type, "revoked")

    def test_events_for_agent(self):
        chain = DelegationChain()
        tok1 = self._tok("alice")
        tok2 = self._tok("bob")
        chain.record_issued(tok1)
        chain.record_issued(tok2)
        chain.record_tool_call(tok1, "write_file")
        alice_events = chain.events_for_agent("alice")
        self.assertEqual(len(alice_events), 2)

    def test_events_for_token_filters_correctly(self):
        chain = DelegationChain()
        tok1 = self._tok("a")
        tok2 = self._tok("b")
        chain.record_issued(tok1)
        chain.record_issued(tok2)
        self.assertEqual(len(chain.events_for_token(tok1.token_id)), 1)
        self.assertEqual(len(chain.events_for_token(tok2.token_id)), 1)

    def test_to_json_is_valid_json(self):
        chain = DelegationChain()
        tok = self._tok()
        chain.record_issued(tok)
        chain.record_tool_call(tok, "search_web")
        chain.record_revoked(tok)
        parsed = json.loads(chain.to_json())
        self.assertEqual(len(parsed), 3)
        types = [e["event_type"] for e in parsed]
        self.assertEqual(types, ["issued", "tool_call", "revoked"])

    def test_to_json_omits_none_fields(self):
        chain = DelegationChain()
        tok = self._tok()
        chain.record_revoked(tok)
        parsed = json.loads(chain.to_json())
        self.assertNotIn("tool_name", parsed[0])

    def test_parent_id_recorded_in_issued_event(self):
        chain = DelegationChain()
        parent = self._tok("orch")
        child = SubAgentToken("worker", 30, parent=parent)
        chain.record_issued(child)
        events = chain.events_for_token(child.token_id)
        self.assertEqual(events[0].parent_id, parent.token_id)

    def test_thread_safe_concurrent_records(self):
        chain = DelegationChain()
        toks = [self._tok(f"agent-{i}") for i in range(20)]

        def record(tok):
            chain.record_issued(tok)
            chain.record_tool_call(tok, "some_tool")

        threads = [threading.Thread(target=record, args=(t,)) for t in toks]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(len(chain), 40)

    def test_tool_call_no_metadata_defaults_to_empty(self):
        chain = DelegationChain()
        tok = self._tok()
        chain.record_tool_call(tok, "list_files")
        events = chain.events_for_token(tok.token_id)
        # metadata should be omitted from JSON (empty dict → falsy → None → omitted)
        parsed = json.loads(chain.to_json())
        self.assertNotIn("metadata", parsed[0])


# ---------------------------------------------------------------------------
# RevocationPropagator tests
# ---------------------------------------------------------------------------

class TestRevocationPropagator(unittest.TestCase):

    def _root(self, scopes=None):
        return SubAgentToken("orch", 60, scope_mask=scopes or ["read", "write"])

    def _child(self, parent, agent_id="worker", ttl=30):
        return SubAgentToken(agent_id, ttl, parent=parent)

    def test_register_records_in_chain(self):
        chain = DelegationChain()
        prop = RevocationPropagator(chain)
        tok = self._root()
        prop.register(tok)
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain.events_for_token(tok.token_id)[0].event_type, "issued")

    def test_active_count_tracks_valid_tokens(self):
        prop = RevocationPropagator()
        t1 = self._root()
        t2 = self._root()
        prop.register(t1)
        prop.register(t2)
        self.assertEqual(prop.active_count(), 2)
        t1.revoke()
        self.assertEqual(prop.active_count(), 1)

    def test_revoke_parent_cascades_to_child(self):
        prop = RevocationPropagator()
        parent = self._root()
        child = self._child(parent)
        prop.register(parent)
        prop.register(child)
        parent.revoke()
        self.assertTrue(child.is_revoked)

    def test_revoke_parent_cascades_two_levels(self):
        prop = RevocationPropagator()
        grandparent = self._root()
        parent = self._child(grandparent, "mid")
        child = self._child(parent, "leaf")
        prop.register(grandparent)
        prop.register(parent)
        prop.register(child)
        grandparent.revoke()
        self.assertTrue(parent.is_revoked)
        self.assertTrue(child.is_revoked)

    def test_revoking_child_does_not_revoke_parent(self):
        prop = RevocationPropagator()
        parent = self._root()
        child = self._child(parent)
        prop.register(parent)
        prop.register(child)
        child.revoke()
        self.assertFalse(parent.is_revoked)

    def test_chain_records_revoked_on_cascade(self):
        chain = DelegationChain()
        prop = RevocationPropagator(chain)
        parent = self._root()
        child = self._child(parent)
        prop.register(parent)
        prop.register(child)
        parent.revoke()
        revoked_events = [
            e for e in json.loads(chain.to_json())
            if e["event_type"] == "revoked"
        ]
        revoked_ids = {e["token_id"] for e in revoked_events}
        self.assertIn(parent.token_id, revoked_ids)
        self.assertIn(child.token_id, revoked_ids)

    def test_revoke_all_revokes_every_live_token(self):
        prop = RevocationPropagator()
        tokens = [self._root() for _ in range(5)]
        for t in tokens:
            prop.register(t)
        count = prop.revoke_all()
        self.assertEqual(count, 5)
        self.assertEqual(prop.active_count(), 0)

    def test_revoke_all_skips_already_revoked(self):
        prop = RevocationPropagator()
        t1 = self._root()
        t2 = self._root()
        prop.register(t1)
        prop.register(t2)
        t1.revoke()
        count = prop.revoke_all()
        self.assertEqual(count, 1)

    def test_repr_shows_counts(self):
        prop = RevocationPropagator()
        tok = self._root()
        prop.register(tok)
        r = repr(prop)
        self.assertIn("total=1", r)
        self.assertIn("active=1", r)

    def test_no_chain_does_not_raise(self):
        prop = RevocationPropagator()  # no chain
        tok = self._root()
        prop.register(tok)
        tok.revoke()  # should not raise

    def test_thread_safe_concurrent_register_and_revoke_all(self):
        prop = RevocationPropagator()
        tokens = [self._root() for _ in range(50)]

        def reg(t):
            prop.register(t)

        threads = [threading.Thread(target=reg, args=(t,)) for t in tokens]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        count = prop.revoke_all()
        self.assertEqual(count, 50)
        self.assertEqual(prop.active_count(), 0)

    def test_sibling_tokens_independent_revocation(self):
        prop = RevocationPropagator()
        parent = self._root()
        child_a = SubAgentToken("worker-a", 30, parent=parent)
        child_b = SubAgentToken("worker-b", 30, parent=parent)
        prop.register(parent)
        prop.register(child_a)
        prop.register(child_b)
        child_a.revoke()
        self.assertFalse(child_b.is_revoked)
        self.assertFalse(parent.is_revoked)


# ---------------------------------------------------------------------------
# Integration: full session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle(unittest.TestCase):

    def test_full_session_with_chain_and_propagator(self):
        chain = DelegationChain()
        prop = RevocationPropagator(chain)

        # Orchestrator token
        orch = SubAgentToken("orchestrator", 120, scope_mask=["read", "write", "delete"])
        prop.register(orch)

        # Two specialised sub-agents with narrowed scopes
        reader = SubAgentToken("reader-agent", 60, scope_mask=["read"], parent=orch)
        writer = SubAgentToken("writer-agent", 60, scope_mask=["write"], parent=orch)
        prop.register(reader)
        prop.register(writer)

        # Reader can read but not write
        self.assertTrue(reader.has_scope("read"))
        self.assertFalse(reader.has_scope("write"))

        # Writer can write but not delete (not in its scope_mask)
        self.assertTrue(writer.has_scope("write"))
        self.assertFalse(writer.has_scope("delete"))

        # Tool calls recorded
        chain.record_tool_call(reader, "fetch_document", {"doc_id": "abc"})
        chain.record_tool_call(writer, "update_document", {"doc_id": "abc"})

        # All tokens valid mid-session
        self.assertEqual(prop.active_count(), 3)

        # Session ends — revoke all
        revoked = prop.revoke_all()
        self.assertEqual(revoked, 3)
        self.assertEqual(prop.active_count(), 0)

        # Audit log complete
        parsed = json.loads(chain.to_json())
        event_types = [e["event_type"] for e in parsed]
        self.assertIn("issued", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("revoked", event_types)

    def test_cascade_on_orchestrator_failure(self):
        chain = DelegationChain()
        prop = RevocationPropagator(chain)

        orch = SubAgentToken("orch", 120, scope_mask=["read", "write"])
        sub1 = SubAgentToken("sub1", 60, parent=orch)
        sub2 = SubAgentToken("sub2", 60, parent=orch)
        leaf = SubAgentToken("leaf", 30, parent=sub1)

        for t in [orch, sub1, sub2, leaf]:
            prop.register(t)

        # Orchestrator hits an error, session torn down
        orch.revoke()

        # All descendants should be revoked
        self.assertTrue(sub1.is_revoked)
        self.assertTrue(sub2.is_revoked)
        self.assertTrue(leaf.is_revoked)
        self.assertEqual(prop.active_count(), 0)


if __name__ == "__main__":
    unittest.main()
