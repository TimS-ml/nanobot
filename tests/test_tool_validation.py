# Tests for tool parameter validation, type casting, and the exec tool path extraction.
# Covers required fields, type checks, range/enum constraints, nested objects, and
# the cast_params helper that coerces LLM-provided strings into correct Python types.

from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool


# Sample tool with a rich JSON Schema used to exercise all validation paths
class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


# Verify that omitting a required parameter produces a validation error
def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


# Verify that wrong types and out-of-range values are caught
def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


# Verify that enum constraint violations and minLength violations are reported
def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


# Verify validation recurses into nested objects and array items
def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


# Verify that extra/unknown fields are silently ignored (no errors)
def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


# Verify the registry returns an error string instead of executing when params are invalid
async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


# Verify that Windows-style absolute paths (C:\...) are correctly extracted
def test_exec_extract_absolute_paths_keeps_full_windows_path() -> None:
    cmd = r"type C:\user\workspace\txt"
    paths = ExecTool._extract_absolute_paths(cmd)
    assert paths == [r"C:\user\workspace\txt"]


# Verify that relative paths like .venv/bin/python are not mistaken for absolute
def test_exec_extract_absolute_paths_ignores_relative_posix_segments() -> None:
    cmd = ".venv/bin/python script.py"
    paths = ExecTool._extract_absolute_paths(cmd)
    assert "/bin/python" not in paths


# Verify that POSIX absolute paths (/tmp/...) are correctly extracted from commands
def test_exec_extract_absolute_paths_captures_posix_absolute_paths() -> None:
    cmd = "cat /tmp/data.txt > /tmp/out.txt"
    paths = ExecTool._extract_absolute_paths(cmd)
    assert "/tmp/data.txt" in paths
    assert "/tmp/out.txt" in paths


# --- cast_params tests ---


class CastTestTool(Tool):
    """Minimal tool for testing cast_params."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    @property
    def name(self) -> str:
        return "cast_test"

    @property
    def description(self) -> str:
        return "test tool for casting"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._schema

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


# Verify that string "42" is cast to int 42 for integer-typed parameters
def test_cast_params_string_to_int() -> None:
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
    )
    result = tool.cast_params({"count": "42"})
    assert result["count"] == 42
    assert isinstance(result["count"], int)


# Verify that string "3.14" is cast to float 3.14 for number-typed parameters
def test_cast_params_string_to_number() -> None:
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"rate": {"type": "number"}},
        }
    )
    result = tool.cast_params({"rate": "3.14"})
    assert result["rate"] == 3.14
    assert isinstance(result["rate"], float)


# Verify that "true"/"false"/"1" strings are cast to Python booleans
def test_cast_params_string_to_bool() -> None:
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
        }
    )
    assert tool.cast_params({"enabled": "true"})["enabled"] is True
    assert tool.cast_params({"enabled": "false"})["enabled"] is False
    assert tool.cast_params({"enabled": "1"})["enabled"] is True


# Verify that string items inside an array are cast to the declared item type
def test_cast_params_array_items() -> None:
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {
                "nums": {"type": "array", "items": {"type": "integer"}},
            },
        }
    )
    result = tool.cast_params({"nums": ["1", "2", "3"]})
    assert result["nums"] == [1, 2, 3]


# Verify that casting recurses into nested object properties
def test_cast_params_nested_object() -> None:
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "port": {"type": "integer"},
                        "debug": {"type": "boolean"},
                    },
                },
            },
        }
    )
    result = tool.cast_params({"config": {"port": "8080", "debug": "true"}})
    assert result["config"]["port"] == 8080
    assert result["config"]["debug"] is True


# Verify that Python bool True is NOT silently cast to int (isinstance(True, int) is True in Python)
def test_cast_params_bool_not_cast_to_int() -> None:
    """Booleans should not be silently cast to integers."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
    )
    result = tool.cast_params({"count": True})
    assert result["count"] is True
    errors = tool.validate_params(result)
    assert any("count should be integer" in e for e in errors)


# Verify that empty string "" is preserved as-is for string-typed parameters
def test_cast_params_preserves_empty_string() -> None:
    """Empty strings should be preserved for string type."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
    )
    result = tool.cast_params({"name": ""})
    assert result["name"] == ""


# Verify that various falsy string representations are cast to False
def test_cast_params_bool_string_false() -> None:
    """Test that 'false', '0', 'no' strings convert to False."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
        }
    )
    assert tool.cast_params({"flag": "false"})["flag"] is False
    assert tool.cast_params({"flag": "False"})["flag"] is False
    assert tool.cast_params({"flag": "0"})["flag"] is False
    assert tool.cast_params({"flag": "no"})["flag"] is False
    assert tool.cast_params({"flag": "NO"})["flag"] is False


# Verify that unrecognized strings like "random" are left as-is (validation catches them)
def test_cast_params_bool_string_invalid() -> None:
    """Invalid boolean strings should not be cast."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
        }
    )
    # Invalid strings should be preserved (validation will catch them)
    result = tool.cast_params({"flag": "random"})
    assert result["flag"] == "random"
    result = tool.cast_params({"flag": "maybe"})
    assert result["flag"] == "maybe"


# Verify that non-numeric strings like "abc" are preserved when cast to integer fails
def test_cast_params_invalid_string_to_int() -> None:
    """Invalid strings should not be cast to integer."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
    )
    result = tool.cast_params({"count": "abc"})
    assert result["count"] == "abc"  # Original value preserved
    result = tool.cast_params({"count": "12.5.7"})
    assert result["count"] == "12.5.7"


# Verify that non-numeric strings are preserved when cast to number fails
def test_cast_params_invalid_string_to_number() -> None:
    """Invalid strings should not be cast to number."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"rate": {"type": "number"}},
        }
    )
    result = tool.cast_params({"rate": "not_a_number"})
    assert result["rate"] == "not_a_number"


# Verify that Python booleans are rejected for number-typed parameters
def test_validate_params_bool_not_accepted_as_number() -> None:
    """Booleans should not pass number validation."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"rate": {"type": "number"}},
        }
    )
    errors = tool.validate_params({"rate": False})
    assert any("rate should be number" in e for e in errors)


# Verify that None values are preserved across all parameter types
def test_cast_params_none_values() -> None:
    """Test None handling for different types."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "items": {"type": "array"},
                "config": {"type": "object"},
            },
        }
    )
    result = tool.cast_params(
        {
            "name": None,
            "count": None,
            "items": None,
            "config": None,
        }
    )
    # None should be preserved for all types
    assert result["name"] is None
    assert result["count"] is None
    assert result["items"] is None
    assert result["config"] is None


# Verify that scalar values are NOT auto-wrapped into arrays (validation catches type mismatch)
def test_cast_params_single_value_not_auto_wrapped_to_array() -> None:
    """Single values should NOT be automatically wrapped into arrays."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"items": {"type": "array"}},
        }
    )
    # Non-array values should be preserved (validation will catch them)
    result = tool.cast_params({"items": 5})
    assert result["items"] == 5  # Not wrapped to [5]
    result = tool.cast_params({"items": "text"})
    assert result["items"] == "text"  # Not wrapped to ["text"]
