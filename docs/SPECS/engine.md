# Execution Engine

**File:** `supervisor/core/engine.py`

## Overview

The Execution Engine is the central coordinator for the Supervisor orchestrator. It manages the execution of AI worker roles with full context packing, Docker sandbox isolation, retry policies, and circuit breaker patterns.

## Key Classes

### `ExecutionEngine`

Main execution engine that coordinates:
- Context packing for AI prompts
- Worker invocation via Docker sandbox
- Output parsing and validation
- Gate verification (tests, lint)
- State updates via event sourcing

#### Constructor

```python
ExecutionEngine(
    repo_path: Path,
    db: Database | None = None,
    sandbox_config: SandboxConfig | None = None,
)
```

- `repo_path`: Absolute path to the git repository
- `db`: Optional database instance (defaults to `.supervisor/state.db`)
- `sandbox_config`: Optional sandbox configuration (defaults to allowing `.worktrees` only)

#### Key Methods

##### `run_role()`

```python
def run_role(
    role_name: str,
    task_description: str,
    workflow_id: str,
    step_id: str | None = None,
    target_files: list[str] | None = None,
    extra_context: dict[str, str] | None = None,
    retry_policy: RetryPolicy | None = None,
    gates: list[str] | None = None,
) -> BaseModel
```

Executes a role with full isolation and error handling:
1. Packs context using templates (Phase 2) or legacy method (once, from main repo)
2. Creates isolated git worktree (per retry attempt)
3. Executes CLI in Docker sandbox
4. Parses output using CLI adapter (Phase 2)
5. Runs gates in worktree (only if caller passes gates parameter)
6. Applies changes to main tree only after gates pass

##### `_get_template_for_role()` (Phase 2)

Maps role to Jinja2 template, resolving overlays via `base_role`:

```python
def _get_template_for_role(self, role: RoleConfig) -> str | None:
    templates = {
        "planner": "planning.j2",
        "implementer": "implement.j2",
        "reviewer": "review_strict.j2",
    }

    # Try exact role name (for base roles)
    if role.name in templates:
        return templates[role.name]

    # Try base_role (for overlays - handles multi-level extends)
    if role.base_role and role.base_role in templates:
        return templates[role.base_role]

    # Unknown role - fallback to legacy pack_context
    return None
```

##### `_get_schema_for_role()` (Phase 2)

Gets output schema for role, resolving overlays via `base_role`:

```python
def _get_schema_for_role(self, role: RoleConfig) -> type[BaseModel]:
    # Try exact role name (for base roles)
    if role.name in ROLE_SCHEMAS:
        return ROLE_SCHEMAS[role.name]

    # Try base_role (for overlays)
    if role.base_role and role.base_role in ROLE_SCHEMAS:
        return ROLE_SCHEMAS[role.base_role]

    # Unknown role - use GenericOutput
    return GenericOutput
```

### `RetryPolicy`

Configuration for retry behavior with exponential backoff.

```python
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    initial_delay: float = 2.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.1
```

### `CircuitBreaker`

Prevents runaway error loops by tracking failures per step.

- **Thread-safe**: Uses `threading.Lock` for protection
- **Auto-cleanup**: Removes stale keys to prevent memory growth
- **Configurable**: `max_failures=5`, `reset_timeout=300s`

### `ErrorClassifier`

Classifies parsing/validation errors for appropriate handling (uses `ErrorCategory` enum):
- `NETWORK`: Network errors, timeouts (retry same)
- `VALIDATION`: Parsing errors (retry with feedback)
- `LOGIC`: Fatal errors (escalate)

## Execution Flow (Phase 2)

```
run_role()
    |
    +--> load_role()                  # With schema validation
    |
    +--> _get_template_for_role()     # Resolve overlay -> base template
    |
    +--> build_full_prompt()          # Template-based (known roles)
    |    or pack_context()            # Legacy (unknown roles)
    |
    +--> _execute_cli()               # Docker sandbox
    |
    +--> get_adapter() + parse_output()  # CLI-specific parsing
    |    +-- _get_schema_for_role()   # Resolve overlay -> base schema
    |
    +--> run_gates()                  # In worktree
    |
    +--> apply_changes()              # To main tree
```

## Phase 2 Integration

### Template-Based Prompts

```python
# In run_role()
template_name = self._get_template_for_role(role)
if template_name:
    # Template flow for known roles (planner, implementer, reviewer)
    # Also works for overlays via base_role
    prompt = self.context_packer.build_full_prompt(
        template_name,
        role,
        task_description,
        target_files,
        extra_context,
    )
else:
    # Legacy fallback for truly unknown roles
    prompt = self.context_packer.pack_context(...)
```

### CLI Adapter Parsing

```python
# In run_role()
adapter = get_adapter(role.cli)
schema = self._get_schema_for_role(role)
output = adapter.parse_output(result.stdout, schema)
```

This ensures:
- Claude raw markdown is parsed directly
- Codex JSONL events have model text extracted first
- Gemini JSON envelopes are unwrapped first
- All outputs require fenced ```json blocks

## Security Features

1. **Docker Sandbox**: All CLI execution happens in isolated containers
2. **Worktree Isolation**: Each step runs in a clean git worktree
3. **Gate Verification**: Changes only applied after gates pass
4. **Thread-Safe**: CLI client initialization uses double-checked locking
5. **Template Allowlist**: Only package templates can be rendered (Phase 2)

## Error Handling

- **Retry with Feedback**: On parsing errors, includes warning that file changes were discarded
- **Circuit Breaker**: Prevents infinite retry loops
- **DB Failure Recovery**: Logs critical errors if DB update fails after git apply

## Dependencies

- `supervisor.core.context.ContextPacker`
- `supervisor.core.workspace.IsolatedWorkspace`
- `supervisor.core.parser.get_adapter`, `ROLE_SCHEMAS`, `GenericOutput`
- `supervisor.sandbox.executor.SandboxedLLMClient`

## Usage Example

```python
from supervisor.core.engine import ExecutionEngine, RetryPolicy
from pathlib import Path

engine = ExecutionEngine(repo_path=Path("/path/to/repo"))

# Run implementer role with gates
result = engine.run_role(
    role_name="python-implementer",  # Overlay extends implementer
    task_description="Add authentication middleware",
    workflow_id="workflow-123",
    target_files=["src/auth.py"],
    gates=["test", "lint", "mypy"],
    retry_policy=RetryPolicy(max_attempts=3),
)

# Result is ImplementationOutput (resolved via base_role)
print(result.status)  # "SUCCESS"
print(result.files_modified)  # ["src/auth.py"]
```
