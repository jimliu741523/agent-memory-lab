"""
Tests for Pattern 14 — ToolSchemaCompiler (RP-8).

Covers: schema validation, operator classification, all three output formats,
model-profile routing, compile_all, token estimation, edge cases.
"""
import json
import unittest

from patterns.tool_schema_compiler import (
    CompiledSchema,
    ModelProfile,
    OperatorType,
    SchemaValidationError,
    ToolSchemaCompiler,
    _classify_param,
    _token_estimate,
    _validate_schema,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

FULL_SCHEMA = {
    "description": "Search the web for current information.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
                "description": "Maximum number of results.",
            },
            "safe_search": {
                "type": "boolean",
                "description": "Filter explicit content.",
            },
            "category": {
                "type": "string",
                "enum": ["news", "web", "images"],
                "description": "Content category filter.",
            },
            "filters": {
                "type": "object",
                "properties": {
                    "date_from": {"type": "string"},
                    "region":    {"type": "string"},
                },
                "description": "Nested filter object.",
            },
        },
        "required": ["query"],
    },
}

NO_PARAMS_SCHEMA: dict = {
    "description": "List available models.",
    "parameters": {},
}


# ---------------------------------------------------------------------------
# _classify_param
# ---------------------------------------------------------------------------

class TestClassifyParam(unittest.TestCase):
    def test_boolean(self) -> None:
        self.assertEqual(_classify_param({"type": "boolean"}), OperatorType.BOOLEAN)

    def test_numeric_integer(self) -> None:
        self.assertEqual(_classify_param({"type": "integer"}), OperatorType.NUMERIC)

    def test_numeric_number(self) -> None:
        self.assertEqual(_classify_param({"type": "number"}), OperatorType.NUMERIC)

    def test_selection_via_enum(self) -> None:
        result = _classify_param({"type": "string", "enum": ["a", "b"]})
        self.assertEqual(result, OperatorType.SELECTION)

    def test_selection_no_explicit_type(self) -> None:
        result = _classify_param({"enum": [1, 2, 3]})
        self.assertEqual(result, OperatorType.SELECTION)

    def test_composite_object_type(self) -> None:
        result = _classify_param({"type": "object", "properties": {}})
        self.assertEqual(result, OperatorType.COMPOSITE)

    def test_composite_via_properties_only(self) -> None:
        result = _classify_param({"properties": {"x": {"type": "string"}}})
        self.assertEqual(result, OperatorType.COMPOSITE)

    def test_unknown_string(self) -> None:
        result = _classify_param({"type": "string"})
        self.assertEqual(result, OperatorType.UNKNOWN)

    def test_union_type_list_first_non_null(self) -> None:
        result = _classify_param({"type": ["integer", "null"]})
        self.assertEqual(result, OperatorType.NUMERIC)

    def test_all_null_type_list(self) -> None:
        result = _classify_param({"type": ["null"]})
        self.assertEqual(result, OperatorType.UNKNOWN)


# ---------------------------------------------------------------------------
# _validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema(unittest.TestCase):
    def test_valid_schema_passes(self) -> None:
        _validate_schema("search_web", FULL_SCHEMA)  # no exception

    def test_empty_parameters_passes(self) -> None:
        _validate_schema("list_models", {"description": "List models.", "parameters": {}})

    def test_required_not_list_raises(self) -> None:
        bad = {
            "parameters": {
                "type": "object",
                "required": "query",
                "properties": {"query": {"type": "string"}},
            }
        }
        with self.assertRaises(SchemaValidationError) as ctx:
            _validate_schema("bad_tool", bad)
        self.assertIn("required", str(ctx.exception))

    def test_enum_not_list_raises(self) -> None:
        bad = {
            "parameters": {
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": "news"}},
            }
        }
        with self.assertRaises(SchemaValidationError):
            _validate_schema("bad_tool", bad)

    def test_minimum_greater_than_maximum_raises(self) -> None:
        bad = {
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 100, "maximum": 10}
                },
            }
        }
        with self.assertRaises(SchemaValidationError):
            _validate_schema("bad_tool", bad)

    def test_top_level_type_not_object_raises(self) -> None:
        bad = {
            "parameters": {
                "type": "array",
                "properties": {"x": {"type": "string"}},
            }
        }
        with self.assertRaises(SchemaValidationError):
            _validate_schema("bad_tool", bad)

    def test_properties_not_dict_raises(self) -> None:
        bad = {
            "parameters": {
                "type": "object",
                "properties": "not_a_dict",
            }
        }
        with self.assertRaises(SchemaValidationError):
            _validate_schema("bad_tool", bad)

    def test_property_schema_not_dict_raises(self) -> None:
        bad = {
            "parameters": {
                "type": "object",
                "properties": {"query": "just_a_string"},
            }
        }
        with self.assertRaises(SchemaValidationError):
            _validate_schema("bad_tool", bad)


# ---------------------------------------------------------------------------
# ToolSchemaCompiler — registration
# ---------------------------------------------------------------------------

class TestCompilerRegistration(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = ToolSchemaCompiler()

    def test_register_and_list(self) -> None:
        self.compiler.register_tool("search_web", FULL_SCHEMA)
        self.assertIn("search_web", self.compiler.registered_tools)

    def test_register_invalid_raises(self) -> None:
        with self.assertRaises(SchemaValidationError):
            self.compiler.register_tool("bad", {
                "parameters": {"type": "object", "required": 123}
            })

    def test_compile_unregistered_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.compiler.compile("nonexistent")

    def test_invalid_format_raises(self) -> None:
        self.compiler.register_tool("search_web", FULL_SCHEMA)
        with self.assertRaises(ValueError):
            self.compiler.compile("search_web", format="xml")


# ---------------------------------------------------------------------------
# structured_text format
# ---------------------------------------------------------------------------

class TestStructuredTextFormat(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = ToolSchemaCompiler()
        self.compiler.register_tool("search_web", FULL_SCHEMA)

    def _compile(self) -> CompiledSchema:
        return self.compiler.compile("search_web", format="structured_text")

    def test_tool_line_present(self) -> None:
        text = self._compile().text
        self.assertIn("TOOL: search_web", text)

    def test_desc_line_present(self) -> None:
        text = self._compile().text
        self.assertIn("DESC:", text)

    def test_params_header_present(self) -> None:
        text = self._compile().text
        self.assertIn("PARAMS:", text)

    def test_required_labeled(self) -> None:
        text = self._compile().text
        self.assertIn("required", text)

    def test_optional_labeled(self) -> None:
        text = self._compile().text
        self.assertIn("optional", text)

    def test_numeric_range_present(self) -> None:
        text = self._compile().text
        self.assertIn("1..20", text)

    def test_boolean_type_labeled(self) -> None:
        text = self._compile().text
        self.assertIn("boolean", text)

    def test_selection_choices_present(self) -> None:
        text = self._compile().text
        self.assertIn("news|web|images", text)

    def test_composite_present(self) -> None:
        text = self._compile().text
        self.assertIn("object", text)

    def test_default_shown(self) -> None:
        text = self._compile().text
        self.assertIn("default=", text)

    def test_no_params_shows_none(self) -> None:
        self.compiler.register_tool("list_models", NO_PARAMS_SCHEMA)
        text = self.compiler.compile("list_models", format="structured_text").text
        self.assertIn("(none)", text)

    def test_format_field_correct(self) -> None:
        result = self._compile()
        self.assertEqual(result.format, "structured_text")

    def test_token_estimate_positive(self) -> None:
        result = self._compile()
        self.assertGreater(result.token_estimate, 0)


# ---------------------------------------------------------------------------
# compressed_json format
# ---------------------------------------------------------------------------

class TestCompressedJsonFormat(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = ToolSchemaCompiler()
        self.compiler.register_tool("search_web", FULL_SCHEMA)

    def _compile(self) -> CompiledSchema:
        return self.compiler.compile("search_web", format="compressed_json")

    def test_output_is_valid_json(self) -> None:
        text = self._compile().text
        parsed = json.loads(text)
        self.assertIn("name", parsed)

    def test_no_whitespace_in_output(self) -> None:
        text = self._compile().text
        self.assertNotIn("  ", text)  # minified — no double spaces

    def test_has_parameters_key(self) -> None:
        text = self._compile().text
        parsed = json.loads(text)
        self.assertIn("parameters", parsed)

    def test_format_field_correct(self) -> None:
        self.assertEqual(self._compile().format, "compressed_json")

    def test_description_truncated_at_depth(self) -> None:
        long_desc = "A" * 200
        schema = {
            "description": long_desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string", "description": long_desc}
                },
                "required": [],
            },
        }
        self.compiler.register_tool("long_desc", schema)
        result = self.compiler.compile("long_desc", format="compressed_json")
        parsed = json.loads(result.text)
        prop_desc = (
            parsed.get("parameters", {})
            .get("properties", {})
            .get("x", {})
            .get("description", "")
        )
        self.assertLessEqual(len(prop_desc), 45)


# ---------------------------------------------------------------------------
# markdown_table format
# ---------------------------------------------------------------------------

class TestMarkdownTableFormat(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = ToolSchemaCompiler()
        self.compiler.register_tool("search_web", FULL_SCHEMA)

    def _compile(self) -> CompiledSchema:
        return self.compiler.compile("search_web", format="markdown_table")

    def test_has_heading(self) -> None:
        self.assertIn("### search_web", self._compile().text)

    def test_has_table_header(self) -> None:
        text = self._compile().text
        self.assertIn("| Param |", text)
        self.assertIn("| Type |", text)

    def test_has_separator_row(self) -> None:
        text = self._compile().text
        self.assertIn("|----", text)

    def test_boolean_type_cell(self) -> None:
        self.assertIn("boolean", self._compile().text)

    def test_enum_type_cell(self) -> None:
        self.assertIn("enum (", self._compile().text)

    def test_numeric_range_cell(self) -> None:
        self.assertIn("[1..20]", self._compile().text)

    def test_no_params_shows_message(self) -> None:
        self.compiler.register_tool("list_models", NO_PARAMS_SCHEMA)
        text = self.compiler.compile("list_models", format="markdown_table").text
        self.assertIn("No parameters", text)

    def test_format_field_correct(self) -> None:
        self.assertEqual(self._compile().format, "markdown_table")


# ---------------------------------------------------------------------------
# Model-profile routing
# ---------------------------------------------------------------------------

class TestModelProfileRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = ToolSchemaCompiler()
        self.compiler.register_tool("search_web", FULL_SCHEMA)

    def test_phi_routes_to_structured_text(self) -> None:
        result = self.compiler.compile("search_web", model="phi")
        self.assertEqual(result.format, "structured_text")

    def test_qwen_routes_to_structured_text(self) -> None:
        result = self.compiler.compile("search_web", model="qwen")
        self.assertEqual(result.format, "structured_text")

    def test_llama_routes_to_structured_text(self) -> None:
        result = self.compiler.compile("search_web", model="llama")
        self.assertEqual(result.format, "structured_text")

    def test_gpt4_routes_to_compressed_json(self) -> None:
        result = self.compiler.compile("search_web", model="gpt4")
        self.assertEqual(result.format, "compressed_json")

    def test_claude_routes_to_compressed_json(self) -> None:
        result = self.compiler.compile("search_web", model="claude")
        self.assertEqual(result.format, "compressed_json")

    def test_unknown_model_defaults_to_structured_text(self) -> None:
        result = self.compiler.compile("search_web", model="mixtral")
        self.assertEqual(result.format, "structured_text")

    def test_format_override_takes_precedence_over_model(self) -> None:
        result = self.compiler.compile(
            "search_web", format="markdown_table", model="phi"
        )
        self.assertEqual(result.format, "markdown_table")

    def test_register_custom_profile(self) -> None:
        self.compiler.register_model_profile(
            "custom_model", ModelProfile("custom_model", "markdown_table")
        )
        result = self.compiler.compile("search_web", model="custom_model")
        self.assertEqual(result.format, "markdown_table")

    def test_model_profile_invalid_format_raises(self) -> None:
        with self.assertRaises(ValueError):
            ModelProfile("bad", "xml_format")

    def test_model_case_insensitive(self) -> None:
        result = self.compiler.compile("search_web", model="PHI")
        self.assertEqual(result.format, "structured_text")


# ---------------------------------------------------------------------------
# compile_all
# ---------------------------------------------------------------------------

class TestCompileAll(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = ToolSchemaCompiler()
        self.compiler.register_tool("search_web", FULL_SCHEMA)
        self.compiler.register_tool("list_models", NO_PARAMS_SCHEMA)

    def test_returns_all_tools(self) -> None:
        results = self.compiler.compile_all()
        names = {r.tool_name for r in results}
        self.assertEqual(names, {"search_web", "list_models"})

    def test_consistent_format(self) -> None:
        results = self.compiler.compile_all(format="markdown_table")
        for r in results:
            self.assertEqual(r.format, "markdown_table")

    def test_model_routing_in_compile_all(self) -> None:
        results = self.compiler.compile_all(model="phi")
        for r in results:
            self.assertEqual(r.format, "structured_text")


# ---------------------------------------------------------------------------
# classify_params
# ---------------------------------------------------------------------------

class TestClassifyParams(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = ToolSchemaCompiler()
        self.compiler.register_tool("search_web", FULL_SCHEMA)

    def test_classify_returns_correct_types(self) -> None:
        result = self.compiler.classify_params("search_web")
        self.assertEqual(result["query"],       OperatorType.UNKNOWN)   # string, no enum
        self.assertEqual(result["max_results"], OperatorType.NUMERIC)
        self.assertEqual(result["safe_search"], OperatorType.BOOLEAN)
        self.assertEqual(result["category"],    OperatorType.SELECTION)
        self.assertEqual(result["filters"],     OperatorType.COMPOSITE)

    def test_classify_unregistered_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.compiler.classify_params("nonexistent")


# ---------------------------------------------------------------------------
# Token estimate
# ---------------------------------------------------------------------------

class TestTokenEstimate(unittest.TestCase):
    def test_single_word(self) -> None:
        self.assertEqual(_token_estimate("hello"), 1)

    def test_multiword(self) -> None:
        self.assertEqual(_token_estimate("hello world foo"), 3)

    def test_empty_string(self) -> None:
        self.assertEqual(_token_estimate(""), 1)

    def test_structured_text_fewer_tokens_than_raw_json(self) -> None:
        compiler = ToolSchemaCompiler()
        compiler.register_tool("search_web", FULL_SCHEMA)
        st = compiler.compile("search_web", format="structured_text")
        import json as _json
        raw_json_tokens = _token_estimate(
            _json.dumps({"name": "search_web", **FULL_SCHEMA}, indent=2)
        )
        self.assertLess(st.token_estimate, raw_json_tokens)


# ---------------------------------------------------------------------------
# CompiledSchema dataclass
# ---------------------------------------------------------------------------

class TestCompiledSchema(unittest.TestCase):
    def test_fields_populated(self) -> None:
        compiler = ToolSchemaCompiler()
        compiler.register_tool("search_web", FULL_SCHEMA)
        result = compiler.compile("search_web")
        self.assertEqual(result.tool_name, "search_web")
        self.assertIsInstance(result.text, str)
        self.assertGreater(len(result.text), 0)
        self.assertIsInstance(result.token_estimate, int)

    def test_anthropic_input_schema_alias(self) -> None:
        schema_with_input_schema = {
            "description": "A tool using Anthropic input_schema key.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Input prompt."},
                },
                "required": ["prompt"],
            },
        }
        compiler = ToolSchemaCompiler()
        compiler.register_tool("anthropic_tool", schema_with_input_schema)
        result = compiler.compile("anthropic_tool", format="structured_text")
        self.assertIn("prompt", result.text)
        self.assertIn("required", result.text)


if __name__ == "__main__":
    unittest.main()
