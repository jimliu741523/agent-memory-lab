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


if __name__ == "__main__":
    unittest.main()
