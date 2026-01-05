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

Falls back to generic JSON extraction for unknown roles.

## Role Parser Mapping

```python
ROLE_PARSERS = {
    "planner": parse_plan_output,
    "implementer": parse_implementation_output,
    "reviewer": parse_review_output,
}
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
    status: str
    phases: list[PhaseSpec]
    estimated_complexity: str
```

### `ImplementationOutput`

```python
class ImplementationOutput(BaseModel):
    status: str
    files_modified: list[str]
    summary: str
```

### `ReviewOutput`

```python
class ReviewOutput(BaseModel):
    status: str  # "approved" or "changes_requested"
    issues: list[ReviewIssue]
    summary: str
```

## Usage Example

```python
from supervisor.core.parser import parse_role_output

raw_output = '''
I've analyzed the code and here's my plan:

```json
{
  "status": "complete",
  "phases": [...],
  "estimated_complexity": "medium"
}
```
'''

result = parse_role_output("planner", raw_output)
# Returns: PlanOutput instance
```
