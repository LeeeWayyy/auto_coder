"""Structured output parsing for AI worker responses.

STRICT MODE: No fallback to marker detection. JSON-only validation.

Phase 2 additions:
- CLI adapters for extracting model text from CLI-specific output formats
- GenericOutput for unknown/overlay roles
- ROLE_SCHEMAS mapping for schema lookup by role name
"""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError

from supervisor.core.models import (
    ImplementationOutput,
    PlanOutput,
    ReviewOutput,
)


# --- Generic output schema for unknown/overlay roles ---


class GenericOutput(BaseModel):
    """Generic output for overlay/custom roles.

    Accepts any JSON structure while maintaining Pydantic validation.
    All adapters require a Pydantic model (not dict) for model_validate*.
    """

    model_config = {"extra": "allow"}
    status: str = Field(default="UNKNOWN")


# --- Role to schema mapping ---


ROLE_SCHEMAS: dict[str, type[BaseModel]] = {
    "planner": PlanOutput,
    "implementer": ImplementationOutput,
    "reviewer": ReviewOutput,
}


class ParsingError(Exception):
    """Failed to parse structured output from worker response."""

    pass


class InvalidOutputError(Exception):
    """Worker output doesn't match expected schema."""

    pass


T = TypeVar("T", bound=BaseModel)


def extract_json_block(raw_output: str) -> str | None:
    """Extract JSON block from worker output.

    STRICT MODE: Only accepts explicit ```json code blocks.
    No fallback to raw JSON detection - this prevents spoofing attacks
    where malicious content could be injected as "valid JSON".

    Workers MUST format their output as:
        ```json
        { "status": "...", ... }
        ```
    """
    # ONLY accept explicit ```json blocks - no fallbacks
    json_block_pattern = r"```json\s*([\s\S]*?)\s*```"
    matches = re.findall(json_block_pattern, raw_output)

    if not matches:
        return None

    # Return the last JSON block (most likely to be the structured output)
    return matches[-1].strip()


def parse_worker_output(raw_output: str, schema: type[T]) -> T:
    """Extract and validate structured output from worker response.

    STRICT MODE: No fallback to marker detection.

    Args:
        raw_output: Raw text output from AI worker
        schema: Pydantic model to validate against

    Returns:
        Validated output as schema instance

    Raises:
        ParsingError: If no JSON block found
        InvalidOutputError: If JSON doesn't match schema
    """
    json_str = extract_json_block(raw_output)
    if not json_str:
        raise ParsingError(
            "No JSON block found in worker output. "
            "Worker must include output in ```json ... ``` block."
        )

    try:
        return schema.model_validate_json(json_str)
    except ValidationError as e:
        # Pydantic ValidationError includes JSON syntax errors (type=json_invalid)
        error_str = str(e)
        if "json_invalid" in error_str.lower() or "invalid json" in error_str.lower():
            raise ParsingError(f"Invalid JSON in output: {e}")
        raise InvalidOutputError(f"Output doesn't match {schema.__name__} schema: {e}")
    except json.JSONDecodeError as e:
        raise ParsingError(f"Invalid JSON in output: {e}")


def parse_plan_output(raw_output: str) -> PlanOutput:
    """Parse planner role output."""
    return parse_worker_output(raw_output, PlanOutput)


def parse_implementation_output(raw_output: str) -> ImplementationOutput:
    """Parse implementer role output."""
    return parse_worker_output(raw_output, ImplementationOutput)


def parse_review_output(raw_output: str) -> ReviewOutput:
    """Parse reviewer role output.

    For reviews, we require strict JSON-only validation.
    NO magic string fallback - this prevents spoofing.
    """
    return parse_worker_output(raw_output, ReviewOutput)


# Role name to parser mapping
ROLE_PARSERS = {
    "planner": parse_plan_output,
    "implementer": parse_implementation_output,
    "reviewer": parse_review_output,
}


def parse_role_output(role_name: str, raw_output: str) -> BaseModel:
    """Parse output based on role name.

    Uses ROLE_PARSERS for known roles, GenericOutput for unknown/overlay roles.
    This is SAFE because:
    1. extract_json_block() only accepts fenced ```json blocks
    2. GenericOutput.model_validate() runs Pydantic validation
    3. extra="allow" allows any fields, but they're still type-checked
    """
    parser = ROLE_PARSERS.get(role_name)
    if parser:
        return parser(raw_output)

    # Unknown roles use GenericOutput - still validates fenced JSON and Pydantic model
    return parse_worker_output(raw_output, GenericOutput)


# --- CLI Adapters for extracting model text from CLI-specific output formats ---


class CLIAdapter:
    """Base adapter for CLI-specific output handling.

    IMPORTANT: Adapters must extract MODEL TEXT from CLI output format,
    then use parse_worker_output() to find fenced ```json blocks.
    """

    def parse_output(self, stdout: str, schema: type[T]) -> T:
        """Parse CLI output and validate against schema."""
        raise NotImplementedError

    def _extract_model_text(self, stdout: str) -> str:
        """Extract model text from CLI-specific output format.

        Override in subclasses for CLIs that wrap output in JSON/JSONL.
        """
        return stdout  # Default: stdout IS the model text


class ClaudeAdapter(CLIAdapter):
    """Adapter for Claude Code CLI output.

    Claude Code with -p outputs raw markdown (no JSON wrapper).
    Model text IS the stdout.
    """

    def parse_output(self, stdout: str, schema: type[T]) -> T:
        # Claude outputs raw markdown - use directly
        model_text = self._extract_model_text(stdout)
        return parse_worker_output(model_text, schema)


class CodexAdapter(CLIAdapter):
    """Adapter for Codex CLI output (JSONL events).

    Codex with --json emits JSONL events. The model text is in the
    'text' field of the 'result' event.

    SECURITY: No direct payload validation - extract text then parse fenced JSON.
    """

    def _extract_model_text(self, stdout: str) -> str:
        """Extract model text from Codex JSONL output."""
        lines = stdout.strip().split("\n")
        for line in reversed(lines):
            try:
                event = json.loads(line)
                if event.get("type") == "result":
                    # Model text is in the 'text' field of the payload
                    # Codex JSONL: {"type": "result", "payload": {"text": "..."}}
                    payload = event.get("payload", {})
                    model_text = payload.get("text") or payload.get("output") or ""
                    if model_text:
                        return model_text
            except json.JSONDecodeError:
                continue

        raise ParsingError("No 'result' event with text found in Codex JSONL output.")

    def parse_output(self, stdout: str, schema: type[T]) -> T:
        # Extract model text from JSONL, then parse fenced JSON
        model_text = self._extract_model_text(stdout)
        return parse_worker_output(model_text, schema)


class GeminiAdapter(CLIAdapter):
    """Adapter for Gemini CLI output (JSON envelope).

    Gemini with -o json outputs a JSON envelope. The model text is
    in the 'output' or 'text' field.
    """

    def _extract_model_text(self, stdout: str) -> str:
        """Extract model text from Gemini JSON envelope."""
        try:
            envelope = json.loads(stdout)
            # Gemini JSON: {"output": "...", ...} or {"text": "..."}
            model_text = envelope.get("output") or envelope.get("text") or ""
            if model_text:
                return model_text
        except json.JSONDecodeError:
            pass

        # If not valid JSON, assume raw markdown output (fallback)
        return stdout

    def parse_output(self, stdout: str, schema: type[T]) -> T:
        # Extract model text from JSON envelope, then parse fenced JSON
        model_text = self._extract_model_text(stdout)
        return parse_worker_output(model_text, schema)


def get_adapter(cli: str) -> CLIAdapter:
    """Get appropriate adapter for CLI type."""
    adapters: dict[str, CLIAdapter] = {
        "claude": ClaudeAdapter(),
        "codex": CodexAdapter(),
        "gemini": GeminiAdapter(),
    }
    return adapters.get(cli, ClaudeAdapter())
