"""
Stdlib-only tests for every shipped memory pattern.

Run from the repo root:
    python -m unittest discover tests -v
"""
import unittest

from patterns.sliding_window import SlidingWindow, Message as SWMessage
from patterns.summary_compression import (
    SummaryCompression,
    Message as SCMessage,
    _mock_summarize,
)
from patterns.hierarchical_summary import (
    HierarchicalSummary,
    Message as HSMessage,
    _mock_summarize as _mock_hs,
)
from patterns.vector_retrieval import (
    VectorRetrieval,
    Message as VRMessage,
    _hash_bow_embed,
)
from patterns.structured_episodic import (
    StructuredEpisodic,
    Message as SEMessage,
    Episode,
)


class TestSlidingWindow(unittest.TestCase):
    def test_system_messages_always_kept(self):
        mem = SlidingWindow(window=1)
        mem.add(SWMessage("system", "sys"))
        mem.add(SWMessage("user", "u1"))
        mem.add(SWMessage("assistant", "a1"))
        mem.add(SWMessage("user", "u2"))

        view = mem.view()
        self.assertEqual(view[0].role, "system")
        self.assertEqual(view[0].content, "sys")

    def test_window_bounds_by_recency(self):
        mem = SlidingWindow(window=3)
        for i in range(10):
            mem.add(SWMessage("user", f"m{i}"))

        view = mem.view()
        self.assertEqual([m.content for m in view], ["m7", "m8", "m9"])

    def test_empty_state(self):
        mem = SlidingWindow(window=5)
        self.assertEqual(mem.view(), [])
        self.assertEqual(len(mem), 0)

    def test_system_does_not_consume_window(self):
        mem = SlidingWindow(window=2)
        mem.add(SWMessage("system", "s1"))
        mem.add(SWMessage("system", "s2"))
        mem.add(SWMessage("user", "u1"))
        mem.add(SWMessage("user", "u2"))
        mem.add(SWMessage("user", "u3"))

        view = mem.view()
        systems = [m for m in view if m.role == "system"]
        users = [m for m in view if m.role == "user"]
        self.assertEqual(len(systems), 2)
        self.assertEqual([m.content for m in users], ["u2", "u3"])


class TestSummaryCompression(unittest.TestCase):
    def test_compression_triggers(self):
        mem = SummaryCompression(
            summarize=_mock_summarize, trigger=3, keep=1
        )
        for i in range(5):
            mem.add(SCMessage("user", f"m{i}"))

        view = mem.view()
        has_summary = any("previous_summary" in m.content for m in view)
        self.assertTrue(has_summary, f"expected at least one summary in {view}")

    def test_keep_bounds_recent(self):
        mem = SummaryCompression(
            summarize=_mock_summarize, trigger=3, keep=2
        )
        for i in range(6):
            mem.add(SCMessage("user", f"m{i}"))

        view = mem.view()
        recent_users = [
            m.content
            for m in view
            if m.role == "user" and "previous_summary" not in m.content
        ]
        # after compression, at most `keep`-sized non-summary tail remains
        self.assertLessEqual(len(recent_users), 2)
        # and those are the *most recent*
        self.assertIn("m5", recent_users)

    def test_system_preserved_across_compressions(self):
        mem = SummaryCompression(summarize=_mock_summarize, trigger=3, keep=1)
        mem.add(SCMessage("system", "sys"))
        for i in range(10):
            mem.add(SCMessage("user", f"m{i}"))

        view = mem.view()
        self.assertEqual(view[0].role, "system")
        self.assertEqual(view[0].content, "sys")

    def test_no_compression_below_trigger(self):
        mem = SummaryCompression(summarize=_mock_summarize, trigger=10, keep=2)
        for i in range(5):
            mem.add(SCMessage("user", f"m{i}"))

        view = mem.view()
        has_summary = any("previous_summary" in m.content for m in view)
        self.assertFalse(has_summary)


class TestHierarchicalSummary(unittest.TestCase):
    def test_no_rollup_under_threshold(self):
        mem = HierarchicalSummary(
            summarize=_mock_hs, leaf_chunk=5, fanout=3, keep_recent=2
        )
        for i in range(5):
            mem.add(HSMessage("user", f"m{i}"))
        # not enough overflow beyond keep_recent to trigger a rollup
        self.assertEqual(len(mem._hierarchy), 0)

    def test_rollup_triggers_l1(self):
        mem = HierarchicalSummary(
            summarize=_mock_hs, leaf_chunk=3, fanout=10, keep_recent=1
        )
        for i in range(10):
            mem.add(HSMessage("user", f"m{i}"))
        # at least one L1 summary must exist at hierarchy index 0
        self.assertGreater(len(mem._hierarchy), 0)
        self.assertGreater(len(mem._hierarchy[0]), 0)
        # its content carries the L1 marker
        self.assertIn("L1_summary", mem._hierarchy[0][0].content)

    def test_cascades_to_higher_levels(self):
        import re

        mem = HierarchicalSummary(
            summarize=_mock_hs, leaf_chunk=2, fanout=2, keep_recent=1
        )
        for i in range(20):
            mem.add(HSMessage("user", f"m{i}"))

        # enough messages with fanout=2 MUST produce at least one L2+ summary
        # somewhere. The rolled-up summary may itself have cascaded further,
        # so we can't assume level[1] holds it; check all non-empty levels.
        levels_seen: set[int] = set()
        for level in mem._hierarchy:
            for msg in level:
                for match in re.finditer(r"<L(\d+)_summary>", msg.content):
                    levels_seen.add(int(match.group(1)))
        self.assertTrue(
            any(l >= 2 for l in levels_seen),
            f"no L2+ summary after cascade; saw only {sorted(levels_seen)}",
        )

    def test_system_preserved_and_order(self):
        mem = HierarchicalSummary(
            summarize=_mock_hs, leaf_chunk=2, fanout=2, keep_recent=1
        )
        mem.add(HSMessage("system", "sys"))
        for i in range(10):
            mem.add(HSMessage("user", f"m{i}"))
        view = mem.view()
        self.assertEqual(view[0].role, "system")
        self.assertEqual(view[0].content, "sys")
        # last message in view must be a recent raw user message
        self.assertEqual(view[-1].role, "user")

    def test_keep_recent_always_verbatim_at_tail(self):
        mem = HierarchicalSummary(
            summarize=_mock_hs, leaf_chunk=3, fanout=5, keep_recent=2
        )
        for i in range(10):
            mem.add(HSMessage("user", f"m{i}"))
        view = mem.view()
        # last keep_recent items are raw user messages with the most recent content
        tail = view[-2:]
        self.assertTrue(all(m.role == "user" for m in tail))
        self.assertEqual(tail[-1].content, "m9")


class TestVectorRetrieval(unittest.TestCase):
    def test_view_returns_system_plus_recent(self):
        mem = VectorRetrieval(embed=_hash_bow_embed, keep_recent=3)
        mem.add(VRMessage("system", "sys"))
        for i in range(5):
            mem.add(VRMessage("user", f"m{i}"))
        view = mem.view()
        self.assertEqual(view[0].role, "system")
        # last 3 recent are kept
        self.assertEqual([m.content for m in view[1:]], ["m2", "m3", "m4"])

    def test_oldest_archived_after_overflow(self):
        mem = VectorRetrieval(embed=_hash_bow_embed, keep_recent=2)
        for i in range(5):
            mem.add(VRMessage("user", f"m{i}"))
        # 5 added, 2 recent remain, 3 archived
        self.assertEqual(len(mem._recent), 2)
        self.assertEqual(len(mem._archive), 3)
        # archived contents are the oldest 3
        archived_contents = [pair[0].content for pair in mem._archive]
        self.assertEqual(archived_contents, ["m0", "m1", "m2"])

    def test_query_returns_most_similar(self):
        mem = VectorRetrieval(embed=_hash_bow_embed, keep_recent=1)
        mem.add(VRMessage("user", "python garbage collection uses reference counting"))
        mem.add(VRMessage("user", "rust borrow checker prevents data races"))
        mem.add(VRMessage("user", "javascript has mark and sweep gc"))
        mem.add(VRMessage("user", "dummy recent"))  # kept in _recent, not in archive

        # python-related query should surface the python message from archive
        hits = mem.query("python memory management", k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("python", hits[0].content)

    def test_query_empty_archive_returns_empty_list(self):
        mem = VectorRetrieval(embed=_hash_bow_embed, keep_recent=10)
        mem.add(VRMessage("user", "hi"))
        self.assertEqual(mem.query("anything", k=3), [])

    def test_query_does_not_return_recent_or_system(self):
        mem = VectorRetrieval(embed=_hash_bow_embed, keep_recent=2)
        mem.add(VRMessage("system", "python context"))  # system, should never appear in query
        mem.add(VRMessage("user", "archive python old"))
        mem.add(VRMessage("user", "recent one"))  # in recent
        mem.add(VRMessage("user", "recent two python"))  # in recent — but python-ish
        hits = mem.query("python", k=5)
        # only archived items come back; recent/system are not eligible
        for m in hits:
            self.assertNotEqual(m.role, "system")
            self.assertNotIn(m.content, ["recent one", "recent two python"])

    def test_hash_embed_is_deterministic(self):
        a = _hash_bow_embed("hello world")
        b = _hash_bow_embed("hello world")
        self.assertEqual(a, b)
        # different text should differ
        c = _hash_bow_embed("completely other tokens")
        self.assertNotEqual(a, c)


class TestStructuredEpisodic(unittest.TestCase):
    def _seed(self) -> StructuredEpisodic:
        mem = StructuredEpisodic(keep_recent=5)
        mem.record_episode(
            situation={"task": "deploy", "env": "prod", "tests": "green"},
            action="deploy",
            outcome="clean",
            tags=("safe-path",),
        )
        mem.record_episode(
            situation={"task": "deploy", "env": "prod", "tests": "red"},
            action="override",
            outcome="rollback",
            tags=("incident", "lesson"),
        )
        mem.record_episode(
            situation={"task": "deploy", "env": "staging", "tests": "red"},
            action="repro",
            outcome="fix found",
            tags=("debug",),
        )
        return mem

    def test_view_returns_system_plus_recent_tail(self):
        mem = StructuredEpisodic(keep_recent=2)
        mem.add(SEMessage("system", "sys"))
        for i in range(5):
            mem.add(SEMessage("user", f"m{i}"))
        view = mem.view()
        self.assertEqual(view[0].role, "system")
        self.assertEqual([m.content for m in view[1:]], ["m3", "m4"])

    def test_record_returns_episode_and_stores_it(self):
        mem = StructuredEpisodic()
        ep = mem.record_episode(
            situation={"task": "index"}, action="ran", outcome="ok"
        )
        self.assertIsInstance(ep, Episode)
        self.assertEqual(len(mem._episodes), 1)

    def test_recall_ranks_exact_situation_match_highest(self):
        mem = self._seed()
        hits = mem.recall_episodes(
            situation={"task": "deploy", "env": "prod", "tests": "red"}, k=3
        )
        # top hit must be the exact (prod, red) episode
        self.assertEqual(hits[0].outcome, "rollback")

    def test_recall_by_tag_only(self):
        mem = self._seed()
        hits = mem.recall_episodes(tags=("incident",), k=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].outcome, "rollback")

    def test_recall_empty_returns_empty(self):
        mem = StructuredEpisodic()
        self.assertEqual(
            mem.recall_episodes(situation={"task": "deploy"}), []
        )

    def test_recall_skips_nonmatching_when_query_given(self):
        mem = self._seed()
        # a totally unrelated situation with no overlapping keys or tags -> no hits
        hits = mem.recall_episodes(
            situation={"task": "train", "dataset": "imagenet"},
            tags=("ml",),
        )
        self.assertEqual(hits, [])

    def test_recency_breaks_ties_at_equal_score(self):
        mem = StructuredEpisodic()
        # three episodes, each matches one key -> all score 1
        mem.record_episode(
            situation={"task": "deploy"}, action="a1", outcome="old"
        )
        mem.record_episode(
            situation={"task": "deploy"}, action="a2", outcome="mid"
        )
        mem.record_episode(
            situation={"task": "deploy"}, action="a3", outcome="new"
        )
        hits = mem.recall_episodes(situation={"task": "deploy"}, k=3)
        self.assertEqual([h.outcome for h in hits], ["new", "mid", "old"])


class TestMemoryWriter(unittest.TestCase):
    """Tests for patterns.memory_writer.MemoryWriter (RP-3)."""

    def _make_backend(self):
        from patterns.memory_writer import ListBackend
        return ListBackend()

    def _make_msg(self, role, content):
        from patterns.memory_writer import Message
        return Message(role, content)

    def _make_mw(self, **kwargs):
        from patterns.memory_writer import MemoryWriter
        return MemoryWriter(backend=self._make_backend(), **kwargs)

    def test_default_commits_message(self):
        from patterns.memory_writer import WriteOutcome
        mw = self._make_mw()
        out = mw.add(self._make_msg("user", "hello"))
        self.assertEqual(out, WriteOutcome.COMMITTED)
        self.assertEqual(len(mw.view()), 1)

    def test_system_always_committed(self):
        from patterns.memory_writer import WriteOutcome
        def _zero_salience(msg, ctx):
            return 0.0
        mw = self._make_mw(salience_fn=_zero_salience, salience_threshold=0.5)
        out = mw.add(self._make_msg("system", "be helpful"))
        self.assertEqual(out, WriteOutcome.COMMITTED)
        self.assertIn("be helpful", [m.content for m in mw.view()])

    def test_low_salience_drops_message(self):
        from patterns.memory_writer import WriteOutcome
        def _low(msg, ctx):
            return 0.1
        mw = self._make_mw(salience_fn=_low, salience_threshold=0.5)
        out = mw.add(self._make_msg("user", "boring"))
        self.assertEqual(out, WriteOutcome.DROPPED_LOW_SALIENCE)
        self.assertEqual(mw.view(), [])

    def test_high_salience_passes_through(self):
        from patterns.memory_writer import WriteOutcome
        def _high(msg, ctx):
            return 1.0
        mw = self._make_mw(salience_fn=_high, salience_threshold=0.5)
        out = mw.add(self._make_msg("user", "important"))
        self.assertEqual(out, WriteOutcome.COMMITTED)
        self.assertEqual(len(mw.view()), 1)

    def test_latest_wins_suppresses_old_contradicting_message(self):
        from patterns.memory_writer import Message, WriteOutcome
        M = self._make_msg
        old_msg = M("user", "sky is green")

        def _contradicts(msg, ctx):
            for m in ctx:
                if "sky is" in m.content and m.content != msg.content:
                    return m
            return None

        mw = self._make_mw(contradict_fn=_contradicts, resolution="latest_wins")
        mw.add(old_msg)
        out = mw.add(M("user", "sky is blue"))
        self.assertEqual(out, WriteOutcome.FLAGGED_CONTRADICTION)
        contents = [m.content for m in mw.view()]
        self.assertIn("sky is blue", contents)
        self.assertNotIn("sky is green", contents)

    def test_merge_resolution_produces_merged_message(self):
        from patterns.memory_writer import Message, WriteOutcome
        M = self._make_msg

        def _contradicts(msg, ctx):
            for m in ctx:
                if "sky is" in m.content and m.content != msg.content:
                    return m
            return None

        def _merge(new, old):
            return Message(new.role, f"[merged] {old.content} + {new.content}")

        mw = self._make_mw(
            contradict_fn=_contradicts,
            merge_fn=_merge,
            resolution="merge",
        )
        mw.add(M("user", "sky is green"))
        out = mw.add(M("user", "sky is blue"))
        self.assertEqual(out, WriteOutcome.MERGED)
        contents = [m.content for m in mw.view()]
        self.assertTrue(any("[merged]" in c for c in contents))
        self.assertNotIn("sky is green", contents)

    def test_flag_for_review_holds_message(self):
        from patterns.memory_writer import WriteOutcome
        M = self._make_msg

        def _contradicts(msg, ctx):
            for m in ctx:
                if "sky is" in m.content and m.content != msg.content:
                    return m
            return None

        mw = self._make_mw(contradict_fn=_contradicts, resolution="flag_for_review")
        mw.add(M("user", "sky is green"))
        out = mw.add(M("user", "sky is blue"))
        self.assertEqual(out, WriteOutcome.HELD_FOR_REVIEW)
        self.assertEqual(len(mw.held), 1)
        self.assertEqual(mw.held[0].content, "sky is blue")
        self.assertNotIn("sky is blue", [m.content for m in mw.view()])

    def test_release_promotes_held_message(self):
        from patterns.memory_writer import WriteOutcome
        M = self._make_msg

        def _contradicts(msg, ctx):
            return ctx[0] if ctx else None

        mw = self._make_mw(contradict_fn=_contradicts, resolution="flag_for_review")
        mw.add(M("user", "first"))
        mw.add(M("user", "second"))
        held = mw.held[0]
        out = mw.release(held)
        self.assertEqual(out, WriteOutcome.COMMITTED)
        self.assertEqual(len(mw.held), 0)
        self.assertIn(held.content, [m.content for m in mw.view()])

    def test_discard_drops_held_message(self):
        M = self._make_msg

        def _contradicts(msg, ctx):
            return ctx[0] if ctx else None

        mw = self._make_mw(contradict_fn=_contradicts, resolution="flag_for_review")
        mw.add(M("user", "first"))
        mw.add(M("user", "second"))
        held = mw.held[0]
        mw.discard(held)
        self.assertEqual(len(mw.held), 0)
        self.assertNotIn(held.content, [m.content for m in mw.view()])

    def test_ttl_expires_old_messages(self):
        mw = self._make_mw(ttl=2)
        M = self._make_msg
        for i in range(5):
            mw.add(M("user", f"turn {i}"))
        visible = [m.content for m in mw.view()]
        self.assertIn("turn 4", visible)
        self.assertIn("turn 3", visible)
        self.assertNotIn("turn 0", visible)
        self.assertNotIn("turn 1", visible)
        self.assertNotIn("turn 2", visible)

    def test_ttl_zero_keeps_all_messages(self):
        mw = self._make_mw(ttl=0)
        M = self._make_msg
        for i in range(5):
            mw.add(M("user", f"turn {i}"))
        self.assertEqual(len(mw.view()), 5)

    def test_ttl_preserves_system_messages(self):
        mw = self._make_mw(ttl=1)
        M = self._make_msg
        mw.add(M("system", "always here"))
        for i in range(5):
            mw.add(M("user", f"turn {i}"))
        visible = [m.content for m in mw.view()]
        self.assertIn("always here", visible)


if __name__ == "__main__":
    unittest.main()
