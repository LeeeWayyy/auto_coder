# Output Parsing

**File:** `supervisor/core/parser.py`

## Overview

Structured output parsing for AI worker responses. Operates in **STRICT MODE**: JSON-only validation with no fallback to marker detection.

## Design Philosophy

Workers MUST format their output as:

```
... any text ...

```json
{
  "status": "...",
  ...
}
```
```

This strict requirement:
- Prevents spoofing attacks where malicious content could be injected as "valid JSON"
- Ensures consistent, parseable output from all workers
- Makes debugging easier by requiring explicit structure

## Key Functions

### `extract_json_block()`

Extracts JSON block from worker output.

```python
def extract_json_block(raw_output: str) -> str | None
```

- Only accepts explicit ` ```json ` code blocks
- Returns the **last** JSON block found (most likely to be structured output)
- Returns `None` if no JSON block found

### `parse_worker_output()`

Generic parser with schema validation:

```python
def parse_worker_output(raw_output: str, schema: type[T]) -> T
```

- Extracts JSON block
- Validates against Pydantic schema
- Raises `ParsingError` or `InvalidOutputError` on failure

### Role-Specific Parsers

```python
def parse_plan_output(raw_output: str) -> PlanOutput
def parse_implementation_output(raw_output: str) -> ImplementationOutput
def parse_review_output(raw_output: str) -> ReviewOutput
```

### `parse_role_output()`

Dispatches to appropriate parser based on role name:

```python
def parse_role_output(role_name: str, raw_output: str) -> BaseModel
```

Uses `GenericOutput` for unknown roles (Phase 2).

## Role Parser Mapping

```python
ROLE_PARSERS = {
    "planner": parse_plan_output,
    "implementer": parse_implementation_output,
    "reviewer": parse_review_output,
}
```

## Role Schema Mapping (Phase 2)

```python
ROLE_SCHEMAS: dict[str, type[BaseModel]] = {
    "planner": PlanOutput,
    "implementer": ImplementationOutput,
    "reviewer": ReviewOutput,
}
```

Used by `_get_schema_for_role()` in engine for schema selection. CLI adapters are schema-agnostic and receive the schema as a parameter from the caller.

## Schema Resolution (Phase 2)

Schema selection is owned by the **engine**, not adapters:

1. **Base roles** (planner, implementer, reviewer): Use their dedicated schema from `ROLE_SCHEMAS`
2. **Overlay roles** with `base_role`: Resolve to base role's schema via `_get_schema_for_role()`
3. **Unknown roles** (no matching name or base_role): Fall back to `GenericOutput`

## GenericOutput (Phase 2)

For truly unknown roles that have no `base_role` pointing to a known schema:

```python
class GenericOutput(BaseModel):
    """Accepts any JSON object with arbitrary fields (extra=allow)."""
    model_config = {"extra": "allow"}
    status: str = Field(default="UNKNOWN")
```

This allows unknown roles to use custom output fields while maintaining validation. Note: Only JSON objects are valid; arrays/scalars will fail Pydantic validation.

## CLI Adapters (Phase 2)

CLI adapters extract model text from CLI-specific output formats before parsing fenced JSON.

### Base Class

```python
class CLIAdapter:
    def parse_output(self, stdout: str, schema: type[T]) -> T:
        """Parse CLI output and validate against schema."""
        raise NotImplementedError

    def _extract_model_text(self, stdout: str) -> str:
        """Extract model text from CLI-specific output format."""
        return stdout  # Default: stdout IS the model text
```

### `ClaudeAdapter`

Claude Code with `-p` outputs raw markdown (no JSON wrapper):

```python
class ClaudeAdapter(CLIAdapter):
    def parse_output(self, stdout: str, schema: type[T]) -> T:
        # Claude outputs raw markdown - use directly
        return parse_worker_output(stdout, schema)
```

### `CodexAdapter`

Codex with `--json` emits JSONL events:

```python
class CodexAdapter(CLIAdapter):
    def _extract_model_text(self, stdout: str) -> str:
        # Find result event in JSONL (searches from end)
        # {"type": "result", "payload": {"text": "..."}}
        for line in reversed(stdout.strip().split("\n")):
            try:
                event = json.loads(line)
                if event.get("type") == "result":
                    payload = event.get("payload", {})
                    # Fallback: text -> output
                    model_text = payload.get("text") or payload.get("output")
                    if model_text:
                        return model_text
            except json.JSONDecodeError:
                continue  # Skip malformed JSONL lines
        raise ParsingError("No result event with text found")

    def parse_output(self, stdout: str, schema: type[T]) -> T:
        model_text = self._extract_model_text(stdout)
        return parse_worker_output(model_text, schema)
```

### `GeminiAdapter`

Gemini with `-o json` outputs a JSON envelope:

```python
class GeminiAdapter(CLIAdapter):
    def _extract_model_text(self, stdout: str) -> str:
        # {"output": "...", ...}
        try:
            envelope = json.loads(stdout)
            return envelope.get("output") or envelope.get("text") or stdout
        except json.JSONDecodeError:
            return stdout  # Fallback to raw if not JSON

    def parse_output(self, stdout: str, schema: type[T]) -> T:
        model_text = self._extract_model_text(stdout)
        return parse_worker_output(model_text, schema)
```

### `get_adapter()`

Factory function:

```python
def get_adapter(cli: str) -> CLIAdapter:
    adapters = {
        "claude": ClaudeAdapter(),
        "codex": CodexAdapter(),
        "gemini": GeminiAdapter(),
    }
    return adapters.get(cli, ClaudeAdapter())
```

## Error Classes

### `ParsingError`

Failed to parse structured output:
- No JSON block found
- Invalid JSON syntax

### `InvalidOutputError`

Output doesn't match expected schema:
- Missing required fields
- Wrong field types
- Validation constraints violated

## Output Schemas

Defined in `supervisor/core/models.py`:

### `PlanOutput`

```python
class PlanOutput(BaseModel):
    status: str = Field(..., pattern="^(COMPLETE|NEEDS_REFINEMENT|BLOCKED)$")
    phases: list[dict[str, Any]]
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    estimated_components: int
    risks: list[str] = Field(default_factory=list)
    next_step: str | None = None
```

### `ImplementationOutput`

```python
class ImplementationOutput(BaseModel):
    status: str = Field(..., pattern="^(SUCCESS|PARTIAL|FAILED|BLOCKED)$")
    action_taken: str
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    tests_written: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_step: str | None = None
```

### `ReviewOutput`

```python
class ReviewOutput(BaseModel):
    status: str = Field(..., pattern="^(APPROVED|CHANGES_REQUESTED|REJECTED)$")
    review_status: str  # Same as status (for compatibility)
    issues: list[ReviewIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    security_concerns: list[str] = Field(default_factory=list)
    next_step: str | None = None
```

## Usage Example

```python
from supervisor.core.parser import parse_role_output, get_adapter
from supervisor.core.models import ImplementationOutput

# Direct parsing (for testing)
raw_output = '''
I've implemented the feature:

```json
{
  "status": "SUCCESS",
  "action_taken": "Added authentication module",
  "files_created": ["src/auth.py"],
  "files_modified": ["src/main.py"]
}
```
'''

result = parse_role_output("implementer", raw_output)
# Returns: ImplementationOutput instance

# Using CLI adapter (production)
adapter = get_adapter("claude")
result = adapter.parse_output(cli_stdout, ImplementationOutput)
```
