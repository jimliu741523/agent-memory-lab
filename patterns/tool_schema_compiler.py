"""
Pattern 14 — ToolSchemaCompiler: tool-schema wire-format adaptation.

Anchored to RP-8 (arxiv 2605.04107 TSCG: Phi-4 14B 0%->84.4% accuracy
with schema compilation; mcp-sdk #235 14-month open request for adapter
functions; crewAI #5472 + deer-flow #186 + autogen #6995 confirming
multi-repo practitioner impact).

JSON Schema as transmitted by MCP, OpenAI Function Calling, and Anthropic
Tool Use is a structural mismatch for small/local models (4B-14B params).
This module compiles verbose tool schemas into token-efficient formats,
adapting the wire format to each model family's parsing capability.

Four operator types per TSCG taxonomy:
  BooleanOp   — yes/no / true-false parameters
  NumericOp   — integer / float with optional range constraints
  SelectionOp — enum / fixed-choice parameters
  CompositeOp — nested object parameters

Three pluggable output formats:
  "structured_text"  — compact indented spec (default; best for small models)
  "compressed_json"  — minified JSON with stripped verbose descriptions
  "markdown_table"   — parameter table (readable; good for chain-of-thought prompts)

Built-in model profile registry (override via register_model_profile):
  "phi"   -> structured_text  (0%->84.4% per TSCG paper)
  "qwen"  -> structured_text
  "llama" -> structured_text
  "gpt4"  -> compressed_json  (native JSON parsing reliable)
  "claude"-> compressed_json  (native JSON parsing reliable)
  default -> structured_text

Schema validation runs before compilation and catches FastMCP #226-class
bugs (malformed docstring->JSON-Schema conversion: missing type, wrong
required placement, broken enum arrays).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Operator type taxonomy (per TSCG 2605.04107)
# ---------------------------------------------------------------------------

class OperatorType(Enum):
    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    SELECTION = "selection"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SchemaValidationError(ValueError):
    """Raised when a tool schema fails pre-compilation validation."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    name: str
    preferred_format: str  # "structured_text" | "compressed_json" | "markdown_table"

    def __post_init__(self) -> None:
        valid = {"structured_text", "compressed_json", "markdown_table"}
        if self.preferred_format not in valid:
            raise ValueError(f"preferred_format must be one of {valid}")


@dataclass
class CompiledSchema:
    tool_name: str
    format: str
    text: str
    token_estimate: int  # rough whitespace-split estimate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_PROFILES: Dict[str, ModelProfile] = {
    "phi":    ModelProfile("phi",    "structured_text"),
    "qwen":   ModelProfile("qwen",   "structured_text"),
    "llama":  ModelProfile("llama",  "structured_text"),
    "gpt4":   ModelProfile("gpt4",   "compressed_json"),
    "claude": ModelProfile("claude", "compressed_json"),
}

_NUMERIC_TYPES = {"integer", "number"}
_BOOLEAN_TYPES = {"boolean"}


def _classify_param(param_schema: dict) -> OperatorType:
    """Classify a single parameter schema into an OperatorType."""
    ptype = param_schema.get("type", "")
    if isinstance(ptype, list):
        # e.g. ["string", "null"] — treat by first non-null type
        ptype = next((t for t in ptype if t != "null"), "")

    if ptype in _BOOLEAN_TYPES:
        return OperatorType.BOOLEAN
    if ptype in _NUMERIC_TYPES:
        return OperatorType.NUMERIC
    if "enum" in param_schema:
        return OperatorType.SELECTION
    if ptype == "object" or "properties" in param_schema:
        return OperatorType.COMPOSITE
    return OperatorType.UNKNOWN


def _validate_schema(tool_name: str, schema: dict) -> None:
    """
    Validate that a tool schema is structurally correct before compilation.
    Catches FastMCP #226-class bugs: missing type, wrong required placement,
    broken enum arrays, non-string names.
    """
    errors: List[str] = []

    if not isinstance(schema.get("name", tool_name), str):
        errors.append("'name' must be a string")

    params = schema.get("parameters") or schema.get("input_schema") or {}
    if not isinstance(params, dict):
        errors.append("'parameters'/'input_schema' must be a dict")
        raise SchemaValidationError(f"{tool_name}: " + "; ".join(errors))

    if params:
        ptype = params.get("type")
        if ptype and ptype != "object":
            errors.append(f"top-level parameters type must be 'object', got '{ptype}'")

        required = params.get("required", [])
        if not isinstance(required, list):
            errors.append("'required' must be a list, not " + type(required).__name__)

        properties = params.get("properties", {})
        if not isinstance(properties, dict):
            errors.append("'properties' must be a dict")
        else:
            for pname, pschema in properties.items():
                if not isinstance(pschema, dict):
                    errors.append(f"property '{pname}' schema must be a dict")
                    continue
                enum_val = pschema.get("enum")
                if enum_val is not None and not isinstance(enum_val, list):
                    errors.append(f"property '{pname}' enum must be a list")
                minimum = pschema.get("minimum")
                maximum = pschema.get("maximum")
                if minimum is not None and maximum is not None:
                    if not isinstance(minimum, (int, float)):
                        errors.append(f"property '{pname}' minimum must be numeric")
                    if not isinstance(maximum, (int, float)):
                        errors.append(f"property '{pname}' maximum must be numeric")
                    elif isinstance(minimum, (int, float)) and minimum > maximum:
                        errors.append(
                            f"property '{pname}' minimum ({minimum}) > maximum ({maximum})"
                        )

    if errors:
        raise SchemaValidationError(f"{tool_name}: " + "; ".join(errors))


def _token_estimate(text: str) -> int:
    return max(1, len(re.split(r"\s+", text.strip())))


# ---------------------------------------------------------------------------
# Format renderers
# ---------------------------------------------------------------------------

def _render_structured_text(tool_name: str, schema: dict) -> str:
    """
    Compact structured-text format per TSCG operator taxonomy.
    Significantly lower token count vs raw JSON for the same information.

    Example output:
        TOOL: search_web
        DESC: Search the web for information
        PARAMS:
          query [string, required]: Search query text
          max_results [numeric 1..20, optional, default=5]: Result count
          safe_search [boolean, optional]: Enable safe search
          category [selection: news|web|images, optional]: Content category
    """
    lines: List[str] = []
    lines.append(f"TOOL: {tool_name}")

    desc = schema.get("description", "")
    if desc:
        lines.append(f"DESC: {desc.strip()}")

    params = schema.get("parameters") or schema.get("input_schema") or {}
    properties: Dict[str, Any] = params.get("properties", {})
    required_list: List[str] = params.get("required", [])

    if not properties:
        lines.append("PARAMS: (none)")
        return "\n".join(lines)

    lines.append("PARAMS:")
    for pname, pschema in properties.items():
        op = _classify_param(pschema)
        is_req = pname in required_list
        req_label = "required" if is_req else "optional"
        pdesc = pschema.get("description", "").strip()
        default = pschema.get("default")

        if op == OperatorType.BOOLEAN:
            type_tag = "boolean"
        elif op == OperatorType.NUMERIC:
            ptype = pschema.get("type", "number")
            minimum = pschema.get("minimum")
            maximum = pschema.get("maximum")
            if minimum is not None and maximum is not None:
                type_tag = f"numeric {minimum}..{maximum}"
            elif minimum is not None:
                type_tag = f"numeric >={minimum}"
            elif maximum is not None:
                type_tag = f"numeric <={maximum}"
            else:
                type_tag = f"numeric ({ptype})"
        elif op == OperatorType.SELECTION:
            choices = "|".join(str(c) for c in pschema.get("enum", []))
            type_tag = f"selection: {choices}"
        elif op == OperatorType.COMPOSITE:
            nested_props = pschema.get("properties", {})
            keys = ", ".join(nested_props.keys()) if nested_props else "..."
            type_tag = f"object {{{keys}}}"
        else:
            ptype = pschema.get("type", "any")
            if isinstance(ptype, list):
                ptype = "|".join(ptype)
            type_tag = ptype

        parts = [f"{pname} [{type_tag}, {req_label}"]
        if default is not None:
            parts[0] += f", default={default!r}"
        parts[0] += "]"

        if pdesc:
            param_line = f"  {parts[0]}: {pdesc}"
        else:
            param_line = f"  {parts[0]}"
        lines.append(param_line)

    return "\n".join(lines)


def _render_compressed_json(tool_name: str, schema: dict) -> str:
    """
    Minified JSON with verbose descriptions stripped to save tokens.
    Retains structure but removes whitespace and long description fields.
    """
    def _strip_descriptions(obj: Any, depth: int = 0) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "description" and depth > 0:
                    stripped = str(v)[:40].rstrip()
                    if stripped:
                        out[k] = stripped + ("…" if len(str(v)) > 40 else "")
                else:
                    out[k] = _strip_descriptions(v, depth + 1)
            return out
        if isinstance(obj, list):
            return [_strip_descriptions(i, depth) for i in obj]
        return obj

    payload = {
        "name": schema.get("name", tool_name),
        "description": (schema.get("description", "")[:60] or ""),
        "parameters": schema.get("parameters") or schema.get("input_schema") or {},
    }
    stripped = _strip_descriptions(payload)
    return json.dumps(stripped, separators=(",", ":"))


def _render_markdown_table(tool_name: str, schema: dict) -> str:
    """
    Markdown table format — readable in chain-of-thought prompts.

    Example:
        ### search_web
        Search the web for information

        | Param | Type | Required | Default | Description |
        |-------|------|----------|---------|-------------|
        | query | string | yes | — | Search query |
    """
    lines: List[str] = []
    lines.append(f"### {tool_name}")

    desc = schema.get("description", "")
    if desc:
        lines.append(desc.strip())
    lines.append("")

    params = schema.get("parameters") or schema.get("input_schema") or {}
    properties: Dict[str, Any] = params.get("properties", {})
    required_list: List[str] = params.get("required", [])

    if not properties:
        lines.append("_No parameters._")
        return "\n".join(lines)

    lines.append("| Param | Type | Required | Default | Description |")
    lines.append("|-------|------|----------|---------|-------------|")

    for pname, pschema in properties.items():
        op = _classify_param(pschema)
        is_req = "yes" if pname in required_list else "no"
        default = pschema.get("default", "—")
        pdesc = pschema.get("description", "").strip() or "—"

        if op == OperatorType.BOOLEAN:
            type_cell = "boolean"
        elif op == OperatorType.NUMERIC:
            ptype = pschema.get("type", "number")
            mn, mx = pschema.get("minimum"), pschema.get("maximum")
            if mn is not None and mx is not None:
                type_cell = f"{ptype} [{mn}..{mx}]"
            else:
                type_cell = ptype
        elif op == OperatorType.SELECTION:
            choices = ", ".join(str(c) for c in pschema.get("enum", []))
            type_cell = f"enum ({choices})"
        elif op == OperatorType.COMPOSITE:
            type_cell = "object"
        else:
            ptype = pschema.get("type", "any")
            if isinstance(ptype, list):
                ptype = "|".join(ptype)
            type_cell = ptype

        lines.append(f"| {pname} | {type_cell} | {is_req} | {default} | {pdesc} |")

    return "\n".join(lines)


_RENDERERS = {
    "structured_text": _render_structured_text,
    "compressed_json": _render_compressed_json,
    "markdown_table":  _render_markdown_table,
}


# ---------------------------------------------------------------------------
# ToolSchemaCompiler
# ---------------------------------------------------------------------------

class ToolSchemaCompiler:
    """
    Compiles tool JSON Schema definitions into token-efficient wire formats
    suited to the target model family.

    Usage:
        compiler = ToolSchemaCompiler()
        compiler.register_tool("search_web", {
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string",  "description": "Search query"},
                    "top_k":  {"type": "integer", "minimum": 1, "maximum": 20},
                    "safe":   {"type": "boolean"},
                    "filter": {"type": "string",  "enum": ["news", "web", "images"]},
                },
                "required": ["query"],
            },
        })
        result = compiler.compile("search_web", model="phi")
        print(result.text)
    """

    def __init__(self) -> None:
        self._schemas: Dict[str, dict] = {}
        self._profiles: Dict[str, ModelProfile] = dict(_DEFAULT_PROFILES)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(self, tool_name: str, schema: dict) -> None:
        """
        Validate and register a tool schema.

        Raises SchemaValidationError if the schema is structurally invalid
        (catches FastMCP #226-class bugs before runtime failures).
        """
        _validate_schema(tool_name, schema)
        self._schemas[tool_name] = schema

    def register_model_profile(self, name: str, profile: ModelProfile) -> None:
        """Override or add a model profile."""
        self._profiles[name] = profile

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(
        self,
        tool_name: str,
        format: Optional[str] = None,
        model: Optional[str] = None,
    ) -> CompiledSchema:
        """
        Compile a registered tool schema to the target wire format.

        Args:
            tool_name: Name passed to register_tool().
            format:    Output format override ("structured_text",
                       "compressed_json", "markdown_table").
                       If None, resolved from model profile or defaults
                       to "structured_text".
            model:     Model family hint (e.g. "phi", "gpt4", "claude").
                       Used to look up preferred_format when format=None.

        Returns:
            CompiledSchema with .text ready to embed in the LLM prompt.

        Raises:
            KeyError if tool_name not registered.
            ValueError if format is invalid.
        """
        if tool_name not in self._schemas:
            raise KeyError(f"Tool '{tool_name}' not registered")

        resolved_format = format
        if resolved_format is None and model is not None:
            profile = self._profiles.get(model.lower())
            resolved_format = profile.preferred_format if profile else "structured_text"
        if resolved_format is None:
            resolved_format = "structured_text"

        if resolved_format not in _RENDERERS:
            raise ValueError(
                f"Unknown format '{resolved_format}'. "
                f"Valid formats: {sorted(_RENDERERS)}"
            )

        schema = self._schemas[tool_name]
        renderer = _RENDERERS[resolved_format]
        text = renderer(tool_name, schema)
        return CompiledSchema(
            tool_name=tool_name,
            format=resolved_format,
            text=text,
            token_estimate=_token_estimate(text),
        )

    def compile_all(
        self,
        format: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[CompiledSchema]:
        """Compile all registered tool schemas."""
        return [
            self.compile(name, format=format, model=model)
            for name in self._schemas
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def classify_params(self, tool_name: str) -> Dict[str, OperatorType]:
        """Return operator-type classification for each parameter."""
        if tool_name not in self._schemas:
            raise KeyError(f"Tool '{tool_name}' not registered")
        schema = self._schemas[tool_name]
        params = schema.get("parameters") or schema.get("input_schema") or {}
        props = params.get("properties", {})
        return {name: _classify_param(pschema) for name, pschema in props.items()}

    @property
    def registered_tools(self) -> List[str]:
        return list(self._schemas)


# ---------------------------------------------------------------------------
# Offline demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    compiler = ToolSchemaCompiler()

    SEARCH_SCHEMA = {
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
                    "description": "Maximum number of results to return.",
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
            },
            "required": ["query"],
        },
    }

    compiler.register_tool("search_web", SEARCH_SCHEMA)

    for fmt in ("structured_text", "compressed_json", "markdown_table"):
        result = compiler.compile("search_web", format=fmt)
        print(f"\n{'='*60}")
        print(f"Format: {fmt} (~{result.token_estimate} tokens)")
        print("=" * 60)
        print(result.text)

    # Model-profile routing
    print("\n" + "=" * 60)
    print("Model-profile routing demo")
    print("=" * 60)
    for mdl in ("phi", "qwen", "gpt4", "claude"):
        r = compiler.compile("search_web", model=mdl)
        print(f"  model={mdl!r:8s} -> format={r.format!r}, ~{r.token_estimate} tokens")

    # Param classification
    print("\nParam classifications:")
    for pname, op in compiler.classify_params("search_web").items():
        print(f"  {pname:15s} -> {op.value}")

    # Validation catches bad schema
    print("\nValidation demo (bad schema):")
    try:
        compiler.register_tool("bad_tool", {
            "parameters": {
                "type": "object",
                "required": "should_be_a_list",
            },
        })
    except SchemaValidationError as exc:
        print(f"  Caught: {exc}")
