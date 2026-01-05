"""Tests for structured output parsing and CLI adapters."""

import json

import pytest

from supervisor.core.models import ImplementationOutput, PlanOutput, ReviewOutput
from supervisor.core.parser import (
    ClaudeAdapter,
    CodexAdapter,
    GeminiAdapter,
    GenericOutput,
    InvalidOutputError,
    ParsingError,
    extract_json_block,
    get_adapter,
    parse_role_output,
    parse_worker_output,
)


class TestExtractJsonBlock:
    """Tests for JSON block extraction."""

    def test_extract_simple_json_block(self):
        """Extract JSON from markdown with code block."""
        output = """
        Here's my analysis:

        ```json
        {"status": "SUCCESS", "files_created": ["foo.py"]}
        ```
        """
        result = extract_json_block(output)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["status"] == "SUCCESS"

    def test_last_json_block_used(self):
        """When multiple JSON blocks, last one is used."""
        output = """
        First attempt (wrong):
        ```json
        {"status": "FAILED"}
        ```

        Fixed version:
        ```json
        {"status": "SUCCESS", "files_created": []}
        ```
        """
        result = extract_json_block(output)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["status"] == "SUCCESS"

    def test_no_json_block_returns_none(self):
        """No JSON block returns None."""
        output = "Just some text without any JSON."
        result = extract_json_block(output)
        assert result is None

    def test_raw_json_not_detected(self):
        """Raw JSON without fences is NOT detected (strict mode)."""
        output = '{"status": "SUCCESS"}'
        result = extract_json_block(output)
        assert result is None

    def test_json_block_with_extra_whitespace(self):
        """JSON block with whitespace is extracted correctly."""
        output = """
        ```json

        {
            "status": "SUCCESS"
        }

        ```
        """
        result = extract_json_block(output)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["status"] == "SUCCESS"


class TestParseWorkerOutput:
    """Tests for worker output parsing with schema validation."""

    def test_parse_plan_output(self):
        """Parse valid planner output."""
        output = """
        Here's the plan:

        ```json
        {
            "status": "COMPLETE",
            "phases": [{"title": "Phase 1", "components": []}],
            "estimated_components": 5,
            "risks": ["Timeline risk"]
        }
        ```
        """
        result = parse_worker_output(output, PlanOutput)
        assert result.status == "COMPLETE"
        assert result.estimated_components == 5

    def test_parse_implementation_output(self):
        """Parse valid implementer output."""
        output = """
        ```json
        {
            "status": "SUCCESS",
            "action_taken": "Implemented feature",
            "files_created": ["src/new.py"],
            "files_modified": ["src/main.py"]
        }
        ```
        """
        result = parse_worker_output(output, ImplementationOutput)
        assert result.status == "SUCCESS"
        assert "src/new.py" in result.files_created

    def test_parse_review_output(self):
        """Parse valid reviewer output."""
        output = """
        ```json
        {
            "status": "APPROVED",
            "review_status": "APPROVED",
            "issues": [],
            "suggestions": ["Consider adding more tests"]
        }
        ```
        """
        result = parse_worker_output(output, ReviewOutput)
        assert result.status == "APPROVED"
        assert "Consider adding more tests" in result.suggestions

    def test_no_marker_fallback(self):
        """Magic strings like 'APPROVED' are NOT parsed."""
        output = "REVIEW_STATUS: APPROVED\n\nLooks good!"
        with pytest.raises(ParsingError, match="No JSON block found"):
            parse_worker_output(output, ReviewOutput)

    def test_invalid_json_syntax(self):
        """Invalid JSON syntax raises ParsingError."""
        output = """
        ```json
        {"status": "SUCCESS", invalid}
        ```
        """
        with pytest.raises(ParsingError, match="Invalid JSON"):
            parse_worker_output(output, ImplementationOutput)

    def test_schema_validation_failure(self):
        """Missing required fields raises InvalidOutputError."""
        output = """
        ```json
        {"status": "SUCCESS"}
        ```
        """
        # ImplementationOutput requires action_taken
        with pytest.raises(InvalidOutputError, match="action_taken"):
            parse_worker_output(output, ImplementationOutput)

    def test_invalid_status_value(self):
        """Invalid status enum value raises InvalidOutputError."""
        output = """
        ```json
        {
            "status": "INVALID_STATUS",
            "action_taken": "Something"
        }
        ```
        """
        with pytest.raises(InvalidOutputError):
            parse_worker_output(output, ImplementationOutput)


class TestGenericOutput:
    """Tests for GenericOutput model."""

    def test_generic_output_accepts_any_fields(self):
        """GenericOutput accepts any JSON fields."""
        output = """
        ```json
        {
            "status": "CUSTOM",
            "custom_field": "value",
            "nested": {"key": "value"}
        }
        ```
        """
        result = parse_worker_output(output, GenericOutput)
        assert result.status == "CUSTOM"
        assert result.custom_field == "value"

    def test_generic_output_default_status(self):
        """GenericOutput defaults status to UNKNOWN."""
        output = """
        ```json
        {"some_field": "value"}
        ```
        """
        result = parse_worker_output(output, GenericOutput)
        assert result.status == "UNKNOWN"


class TestParseRoleOutput:
    """Tests for role-based output parsing."""

    def test_known_role_uses_specific_parser(self):
        """Known roles use their specific parsers."""
        output = """
        ```json
        {
            "status": "SUCCESS",
            "action_taken": "Did something",
            "files_created": []
        }
        ```
        """
        result = parse_role_output("implementer", output)
        assert isinstance(result, ImplementationOutput)

    def test_unknown_role_uses_generic_output(self):
        """Unknown roles use GenericOutput."""
        output = """
        ```json
        {"status": "CUSTOM", "data": "value"}
        ```
        """
        result = parse_role_output("custom-role", output)
        assert isinstance(result, GenericOutput)
        assert result.status == "CUSTOM"


class TestClaudeAdapter:
    """Tests for Claude CLI adapter."""

    def test_parse_raw_markdown(self):
        """Claude outputs raw markdown - parse directly."""
        stdout = """
        Here's my implementation:

        ```json
        {
            "status": "SUCCESS",
            "action_taken": "Created file"
        }
        ```
        """
        adapter = ClaudeAdapter()
        result = adapter.parse_output(stdout, ImplementationOutput)
        assert result.status == "SUCCESS"


class TestCodexAdapter:
    """Tests for Codex CLI adapter."""

    def test_extract_from_jsonl_result_event(self):
        """Extract model text from Codex JSONL result event."""
        # Codex JSONL format
        stdout = (
            '{"type": "start", "payload": {}}\n'
            '{"type": "message", "payload": {"text": "Working..."}}\n'
            '{"type": "result", "payload": {"text": "```json\\n{\\"status\\": \\"SUCCESS\\", \\"action_taken\\": \\"Done\\"}\\n```"}}'
        )
        adapter = CodexAdapter()
        result = adapter.parse_output(stdout, ImplementationOutput)
        assert result.status == "SUCCESS"

    def test_no_result_event_raises_error(self):
        """Missing result event raises ParsingError."""
        stdout = '{"type": "start", "payload": {}}\n{"type": "message", "payload": {"text": "Working..."}}'
        adapter = CodexAdapter()
        with pytest.raises(ParsingError, match="No 'result' event"):
            adapter.parse_output(stdout, ImplementationOutput)


class TestGeminiAdapter:
    """Tests for Gemini CLI adapter."""

    def test_extract_from_json_envelope(self):
        """Extract model text from Gemini JSON envelope."""
        stdout = json.dumps({
            "output": "```json\n{\"status\": \"SUCCESS\", \"action_taken\": \"Done\"}\n```",
            "tokens_used": 100,
        })
        adapter = GeminiAdapter()
        result = adapter.parse_output(stdout, ImplementationOutput)
        assert result.status == "SUCCESS"

    def test_fallback_to_raw_if_not_json(self):
        """If not JSON envelope, treat as raw markdown."""
        stdout = """
        ```json
        {"status": "SUCCESS", "action_taken": "Done"}
        ```
        """
        adapter = GeminiAdapter()
        result = adapter.parse_output(stdout, ImplementationOutput)
        assert result.status == "SUCCESS"


class TestGetAdapter:
    """Tests for adapter factory."""

    def test_get_claude_adapter(self):
        """Get Claude adapter by name."""
        adapter = get_adapter("claude")
        assert isinstance(adapter, ClaudeAdapter)

    def test_get_codex_adapter(self):
        """Get Codex adapter by name."""
        adapter = get_adapter("codex")
        assert isinstance(adapter, CodexAdapter)

    def test_get_gemini_adapter(self):
        """Get Gemini adapter by name."""
        adapter = get_adapter("gemini")
        assert isinstance(adapter, GeminiAdapter)

    def test_unknown_cli_returns_claude_adapter(self):
        """Unknown CLI defaults to Claude adapter."""
        adapter = get_adapter("unknown")
        assert isinstance(adapter, ClaudeAdapter)
