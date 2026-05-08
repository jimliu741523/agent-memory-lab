"""Tests for patterns/mcp_pipe.py (Pattern 13 — McpPipe, RP-2)."""
import threading
import unittest

from patterns.mcp_pipe import (
    HandleStore,
    McpPipe,
    ResultHandle,
    _apply_expr,
    _make_handle,
    _serialize,
)


# ---------------------------------------------------------------------------
# _serialize + _make_handle
# ---------------------------------------------------------------------------

class TestSerialize(unittest.TestCase):
    def test_returns_bytes(self):
        self.assertIsInstance(_serialize({"a": 1}), bytes)

    def test_deterministic(self):
        self.assertEqual(_serialize({"b": 2, "a": 1}), _serialize({"a": 1, "b": 2}))

    def test_different_values_different_bytes(self):
        self.assertNotEqual(_serialize(1), _serialize(2))


class TestMakeHandle(unittest.TestCase):
    def test_returns_result_handle(self):
        h = _make_handle({"key": "val"})
        self.assertIsInstance(h, ResultHandle)

    def test_frozen(self):
        h = _make_handle(42)
        with self.assertRaises((AttributeError, TypeError)):
            h.handle_id = "mutated"  # type: ignore[misc]

    def test_same_content_same_hash_different_id(self):
        h1 = _make_handle({"a": 1})
        h2 = _make_handle({"a": 1})
        self.assertEqual(h1.content_hash, h2.content_hash)
        self.assertNotEqual(h1.handle_id, h2.handle_id)

    def test_different_content_different_hash(self):
        h1 = _make_handle({"a": 1})
        h2 = _make_handle({"a": 2})
        self.assertNotEqual(h1.content_hash, h2.content_hash)

    def test_byte_size_positive(self):
        h = _make_handle({"results": ["a", "b", "c"]})
        self.assertGreater(h.byte_size, 0)

    def test_handle_id_is_uuid_string(self):
        import uuid
        h = _make_handle("test")
        # must be parseable as UUID
        uuid.UUID(h.handle_id)

    def test_content_hash_is_hex_64_chars(self):
        h = _make_handle("anything")
        self.assertEqual(len(h.content_hash), 64)
        int(h.content_hash, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# HandleStore
# ---------------------------------------------------------------------------

class TestHandleStore(unittest.TestCase):
    def test_put_and_get(self):
        store = HandleStore()
        h = _make_handle("hello")
        store.put(h, "hello")
        self.assertEqual(store.get(h), "hello")

    def test_get_missing_raises_key_error(self):
        store = HandleStore()
        h = _make_handle("x")
        with self.assertRaises(KeyError):
            store.get(h)

    def test_error_message_contains_handle_id(self):
        store = HandleStore()
        h = _make_handle("y")
        with self.assertRaises(KeyError) as ctx:
            store.get(h)
        self.assertIn(h.handle_id, str(ctx.exception))

    def test_len_empty(self):
        store = HandleStore()
        self.assertEqual(len(store), 0)

    def test_len_after_puts(self):
        store = HandleStore()
        store.put(_make_handle(1), 1)
        store.put(_make_handle(2), 2)
        self.assertEqual(len(store), 2)

    def test_overwrite_same_handle(self):
        store = HandleStore()
        h = _make_handle("v1")
        store.put(h, "v1")
        store.put(h, "v2")
        self.assertEqual(store.get(h), "v2")
        self.assertEqual(len(store), 1)

    def test_thread_safe_concurrent_puts(self):
        store = HandleStore()
        handles = []
        errors = []

        def worker(i):
            try:
                h = _make_handle(f"item-{i}")
                store.put(h, i)
                handles.append(h)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(store), 30)


# ---------------------------------------------------------------------------
# _apply_expr
# ---------------------------------------------------------------------------

class TestApplyExpr(unittest.TestCase):
    def test_identity_dot(self):
        self.assertEqual(_apply_expr({"a": 1}, "."), {"a": 1})

    def test_dict_key(self):
        self.assertEqual(_apply_expr({"key": "val"}, ".key"), "val")

    def test_nested_dict(self):
        d = {"a": {"b": {"c": 42}}}
        self.assertEqual(_apply_expr(d, ".a.b.c"), 42)

    def test_list_index(self):
        self.assertEqual(_apply_expr([10, 20, 30], ".[1]"), 20)

    def test_list_index_zero(self):
        self.assertEqual(_apply_expr(["first", "second"], ".[0]"), "first")

    def test_mixed_dict_and_list(self):
        data = {"results": [{"url": "http://a.com"}, {"url": "http://b.com"}]}
        self.assertEqual(_apply_expr(data, ".results.[0].url"), "http://a.com")

    def test_mixed_nested_list(self):
        data = {"items": [[1, 2], [3, 4]]}
        self.assertEqual(_apply_expr(data, ".items.[1].[0]"), 3)

    def test_callable_expr(self):
        self.assertEqual(_apply_expr({"n": 5}, lambda v: v["n"] * 2), 10)

    def test_callable_with_list(self):
        self.assertEqual(_apply_expr([1, 2, 3], sum), 6)

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            _apply_expr({}, 42)

    def test_invalid_type_message(self):
        with self.assertRaises(TypeError) as ctx:
            _apply_expr({}, 3.14)
        self.assertIn("callable", str(ctx.exception))


# ---------------------------------------------------------------------------
# McpPipe.call
# ---------------------------------------------------------------------------

class TestMcpPipeCall(unittest.TestCase):
    def setUp(self):
        self.pipe = McpPipe()

    def test_call_returns_result_handle(self):
        h = self.pipe.call(lambda: {"result": 1})
        self.assertIsInstance(h, ResultHandle)

    def test_call_stores_result(self):
        h = self.pipe.call(lambda n: n + 1, n=4)
        self.assertEqual(self.pipe.materialize(h), 5)

    def test_call_with_positional_args(self):
        def add(a, b):
            return a + b
        h = self.pipe.call(add, 3, 4)
        self.assertEqual(self.pipe.materialize(h), 7)

    def test_call_with_none_result(self):
        h = self.pipe.call(lambda: None)
        self.assertIsNone(self.pipe.materialize(h))

    def test_call_increments_handle_count(self):
        self.pipe.call(lambda: 1)
        self.pipe.call(lambda: 2)
        self.assertEqual(self.pipe.stats()["handles_stored"], 2)

    def test_call_does_not_increment_materializations(self):
        self.pipe.call(lambda: "x")
        self.assertEqual(self.pipe.stats()["materializations"], 0)


# ---------------------------------------------------------------------------
# McpPipe.transform
# ---------------------------------------------------------------------------

class TestMcpPipeTransform(unittest.TestCase):
    def setUp(self):
        self.pipe = McpPipe()

    def test_transform_dict_path(self):
        h = self.pipe.call(lambda: {"results": [{"url": "http://a.com", "score": 0.9}]})
        h2 = self.pipe.transform(h, ".results.[0].url")
        self.assertEqual(self.pipe.materialize(h2), "http://a.com")

    def test_transform_callable(self):
        h = self.pipe.call(lambda: [1, 2, 3, 4, 5])
        h2 = self.pipe.transform(h, lambda xs: sum(xs))
        self.assertEqual(self.pipe.materialize(h2), 15)

    def test_original_handle_still_valid_after_transform(self):
        h = self.pipe.call(lambda: {"a": 1, "b": 2})
        h2 = self.pipe.transform(h, ".a")
        self.assertEqual(self.pipe.materialize(h), {"a": 1, "b": 2})
        self.assertEqual(self.pipe.materialize(h2), 1)

    def test_transform_produces_new_handle(self):
        h = self.pipe.call(lambda: {"x": 99})
        h2 = self.pipe.transform(h, ".x")
        self.assertNotEqual(h.handle_id, h2.handle_id)

    def test_transform_adds_to_store(self):
        h = self.pipe.call(lambda: {"x": 99})
        self.pipe.transform(h, ".x")
        self.assertEqual(self.pipe.stats()["handles_stored"], 2)

    def test_transform_identity_path(self):
        original = {"data": "unchanged"}
        h = self.pipe.call(lambda: original)
        h2 = self.pipe.transform(h, ".")
        self.assertEqual(self.pipe.materialize(h2), original)

    def test_chained_transforms(self):
        h = self.pipe.call(lambda: {"a": {"b": [10, 20, 30]}})
        h2 = self.pipe.transform(h, ".a")
        h3 = self.pipe.transform(h2, ".b.[2]")
        self.assertEqual(self.pipe.materialize(h3), 30)

    def test_transform_missing_handle_raises(self):
        fake = ResultHandle("nonexistent", "abc" * 21 + "ab", 0)
        with self.assertRaises(KeyError):
            self.pipe.transform(fake, ".")


# ---------------------------------------------------------------------------
# McpPipe.pipe
# ---------------------------------------------------------------------------

class TestMcpPipePipe(unittest.TestCase):
    def setUp(self):
        self.pipe = McpPipe()

    def test_pipe_basic(self):
        def search(q):
            return {"results": ["doc1", "doc2", "doc3"]}

        def fetch(input):
            return f"content of {input}"

        h_search = self.pipe.call(search, q="bar")
        h_result = self.pipe.pipe(h_search, ".results.[0]", fetch)
        self.assertEqual(self.pipe.materialize(h_result), "content of doc1")

    def test_pipe_with_extra_kwargs(self):
        def enrich(input, prefix):
            return f"{prefix}:{input}"

        h = self.pipe.call(lambda: "base_value")
        h2 = self.pipe.pipe(h, ".", enrich, inject_key="input", prefix="TAGGED")
        self.assertEqual(self.pipe.materialize(h2), "TAGGED:base_value")

    def test_pipe_custom_inject_key(self):
        def process(data):
            return len(data)

        h = self.pipe.call(lambda: [1, 2, 3, 4, 5])
        h2 = self.pipe.pipe(h, ".", process, inject_key="data")
        self.assertEqual(self.pipe.materialize(h2), 5)

    def test_pipe_chain_three_steps(self):
        """Three dependent tools chained without LLM roundtrip."""
        def step1():
            return {"items": [10, 20, 30]}

        def step2(input):
            return {"doubled": [x * 2 for x in input]}

        def step3(input):
            return sum(input)

        h1 = self.pipe.call(step1)
        h2 = self.pipe.pipe(h1, ".items", step2)
        h3 = self.pipe.pipe(h2, ".doubled", step3)
        self.assertEqual(self.pipe.materialize(h3), 120)
        # Only 3 handles created (3 tool calls, 0 transforms)
        self.assertEqual(self.pipe.stats()["handles_stored"], 3)

    def test_pipe_callable_expr(self):
        def summarize(input):
            return {"summary": f"total={input}"}

        h = self.pipe.call(lambda: [5, 10, 15])
        h2 = self.pipe.pipe(h, sum, summarize)
        self.assertEqual(self.pipe.materialize(h2), {"summary": "total=30"})

    def test_pipe_does_not_count_as_materialization(self):
        h = self.pipe.call(lambda: {"v": 1})
        self.pipe.pipe(h, ".v", lambda input: input * 10)
        # pipe should NOT increment materializations
        self.assertEqual(self.pipe.stats()["materializations"], 0)

    def test_pipe_missing_handle_raises(self):
        fake = ResultHandle("bad-id", "abc" * 21 + "ab", 0)
        with self.assertRaises(KeyError):
            self.pipe.pipe(fake, ".", lambda input: input)

    def test_pipe_result_is_independently_materializable(self):
        h = self.pipe.call(lambda: {"n": 7})
        h2 = self.pipe.pipe(h, ".n", lambda input: input ** 2)
        self.assertEqual(self.pipe.materialize(h2), 49)


# ---------------------------------------------------------------------------
# McpPipe.materialize
# ---------------------------------------------------------------------------

class TestMcpPipeMaterialize(unittest.TestCase):
    def test_materialize_increments_counter(self):
        pipe = McpPipe()
        h = pipe.call(lambda: "ok")
        pipe.materialize(h)
        pipe.materialize(h)
        self.assertEqual(pipe.stats()["materializations"], 2)

    def test_materialize_missing_handle_raises(self):
        pipe = McpPipe()
        fake = ResultHandle("nonexistent", "abc" * 21 + "ab", 0)
        with self.assertRaises(KeyError):
            pipe.materialize(fake)

    def test_materialize_string(self):
        pipe = McpPipe()
        h = pipe.call(lambda: "hello world")
        self.assertEqual(pipe.materialize(h), "hello world")

    def test_materialize_int(self):
        pipe = McpPipe()
        h = pipe.call(lambda: 42)
        self.assertEqual(pipe.materialize(h), 42)

    def test_materialize_list(self):
        pipe = McpPipe()
        h = pipe.call(lambda: [1, 2, 3])
        self.assertEqual(pipe.materialize(h), [1, 2, 3])


# ---------------------------------------------------------------------------
# token_savings
# ---------------------------------------------------------------------------

class TestTokenSavings(unittest.TestCase):
    def test_savings_positive_for_large_result(self):
        pipe = McpPipe()
        h = pipe.call(lambda: {"data": "x" * 1000})
        savings = pipe.token_savings([h])
        self.assertGreater(savings, 1000)

    def test_savings_empty_list(self):
        self.assertEqual(McpPipe().token_savings([]), 0)

    def test_savings_multiple_handles(self):
        pipe = McpPipe()
        h1 = pipe.call(lambda: {"r": "a" * 100})
        h2 = pipe.call(lambda: {"r": "b" * 200})
        savings = pipe.token_savings([h1, h2])
        self.assertGreater(savings, 300)

    def test_savings_is_sum_of_byte_sizes(self):
        pipe = McpPipe()
        h1 = pipe.call(lambda: 1)
        h2 = pipe.call(lambda: 2)
        self.assertEqual(pipe.token_savings([h1, h2]), h1.byte_size + h2.byte_size)


# ---------------------------------------------------------------------------
# Content addressing
# ---------------------------------------------------------------------------

class TestContentAddressing(unittest.TestCase):
    def test_same_value_same_hash(self):
        pipe = McpPipe()
        h1 = pipe.call(lambda x: {"val": x}, x=42)
        h2 = pipe.call(lambda x: {"val": x}, x=42)
        self.assertEqual(h1.content_hash, h2.content_hash)
        self.assertNotEqual(h1.handle_id, h2.handle_id)

    def test_different_value_different_hash(self):
        pipe = McpPipe()
        h1 = pipe.call(lambda: {"a": 1})
        h2 = pipe.call(lambda: {"a": 2})
        self.assertNotEqual(h1.content_hash, h2.content_hash)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats(unittest.TestCase):
    def test_initial_stats(self):
        pipe = McpPipe()
        self.assertEqual(pipe.stats(), {"handles_stored": 0, "materializations": 0})

    def test_stats_after_operations(self):
        pipe = McpPipe()
        h = pipe.call(lambda: {"x": 1})
        pipe.transform(h, ".x")
        pipe.materialize(h)
        stats = pipe.stats()
        self.assertEqual(stats["handles_stored"], 2)
        self.assertEqual(stats["materializations"], 1)


if __name__ == "__main__":
    unittest.main()
