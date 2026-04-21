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


if __name__ == "__main__":
    unittest.main()
