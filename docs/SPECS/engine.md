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
1. Creates isolated git worktree
2. Packs context and executes CLI in Docker sandbox
3. Parses and validates output
4. Runs gates in worktree before applying changes
5. Applies changes to main tree only after gates pass

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

Classifies errors for appropriate handling:
- `TRANSIENT`: Network errors, timeouts (retry same)
- `VALIDATION`: Parsing errors (retry with feedback)
- `LOGIC`: Fatal errors (escalate)

## Security Features

1. **Docker Sandbox**: All CLI execution happens in isolated containers
2. **Worktree Isolation**: Each step runs in a clean git worktree
3. **Gate Verification**: Changes only applied after gates pass
4. **Thread-Safe**: CLI client initialization uses double-checked locking

## Error Handling

- **Retry with Feedback**: On parsing errors, includes warning that file changes were discarded
- **Circuit Breaker**: Prevents infinite retry loops
- **DB Failure Recovery**: Logs critical errors if DB update fails after git apply

## Dependencies

- `supervisor.core.context.ContextPacker`
- `supervisor.core.workspace.IsolatedWorkspace`
- `supervisor.core.parser.parse_role_output`
- `supervisor.sandbox.executor.SandboxedLLMClient`
