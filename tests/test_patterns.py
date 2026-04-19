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


if __name__ == "__main__":
    unittest.main()
