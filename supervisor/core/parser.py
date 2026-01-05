"""Structured output parsing for AI worker responses.

STRICT MODE: No fallback to marker detection. JSON-only validation.
"""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from supervisor.core.models import (
    ImplementationOutput,
    PlanOutput,
    ReviewOutput,
)


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
    """Parse output based on role name."""
    parser = ROLE_PARSERS.get(role_name)
    if parser:
        return parser(raw_output)

    # Default: try to extract any JSON as dict
    json_str = extract_json_block(raw_output)
    if json_str:
        return json.loads(json_str)

    raise ParsingError(f"No parser for role '{role_name}' and no JSON found")
