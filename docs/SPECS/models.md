# Data Models

**File:** `supervisor/core/models.py`

## Overview

Pydantic-based data models for the Supervisor orchestrator. Provides schema-enforced structured outputs for AI workers and workflow state management.

## Status Enums

### `StepStatus`

Status of a workflow step:

| Value | Description |
|-------|-------------|
| `PENDING` | Not yet started |
| `IN_PROGRESS` | Currently executing |
| `COMPLETED` | Successfully finished |
| `FAILED` | Execution failed |
| `BLOCKED` | Waiting on dependency |
| `AWAITING_APPROVAL` | Needs human review |

### `ComponentStatus`

Status of a component in hierarchical workflow:

| Value | Description |
|-------|-------------|
| `PENDING` | Not started |
| `IMPLEMENTING` | Code being written |
| `TESTING` | Tests running |
| `REVIEW` | In code review |
| `COMPLETED` | Done |
| `FAILED` | Failed |

### `PhaseStatus`

Status of a phase (rolled up from components):

- `PENDING`, `IN_PROGRESS`, `COMPLETED`, `FAILED`

### `FeatureStatus`

Status of a feature (rolled up from phases):

- `PLANNING`, `IN_PROGRESS`, `REVIEW`, `COMPLETED`, `FAILED`

## Workflow State Models

### `Step`

A single step in a workflow:

```python
class Step(BaseModel):
    id: str
    workflow_id: str
    role: str
    status: StepStatus = StepStatus.PENDING
    gates: list[str] = []
    context: dict[str, Any] = {}
    output: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
```

### `WorkflowState`

Current state of a workflow execution:

```python
class WorkflowState(BaseModel):
    id: str
    current_step: str
    status: StepStatus = StepStatus.PENDING
    context: dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
```

## Hierarchical Workflow Models

### `Component`

Atomic unit of work:

```python
class Component(BaseModel):
    id: str  # e.g., "P1T5-PH1-C1"
    phase_id: str
    title: str
    files: list[str] = []
    depends_on: list[str] = []
    status: ComponentStatus = ComponentStatus.PENDING
    assigned_role: str | None = None
    output: dict[str, Any] | None = None
    error: str | None = None
```

### `Phase`

Groups related components:

```python
class Phase(BaseModel):
    id: str  # e.g., "P1T5-PH1"
    feature_id: str
    title: str
    sequence: int  # Execution order
    status: PhaseStatus = PhaseStatus.PENDING
    interfaces: list[Interface] = []
```

### `Feature`

Top-level task:

```python
class Feature(BaseModel):
    id: str  # e.g., "P1T5"
    title: str
    description: str = ""
    status: FeatureStatus = FeatureStatus.PLANNING
    completed_at: datetime | None = None
```

### `Interface`

Interface definition locked before parallel implementation:

```python
class Interface(BaseModel):
    name: str
    type: str  # "typescript", "openapi", "python_protocol"
    definition: str
    locked: bool = False
```

## Worker Output Models

Schema-enforced outputs from AI workers.

### `PlanOutput`

Planner role output:

```python
class PlanOutput(BaseModel):
    status: str  # COMPLETE, NEEDS_REFINEMENT, BLOCKED
    phases: list[dict[str, Any]]
    dependencies: list[dict[str, Any]] = []
    estimated_components: int
    risks: list[str] = []
    next_step: str | None = None
```

### `ImplementationOutput`

Implementer role output:

```python
class ImplementationOutput(BaseModel):
    status: str  # SUCCESS, PARTIAL, FAILED, BLOCKED
    action_taken: str
    files_created: list[str] = []
    files_modified: list[str] = []
    tests_written: list[str] = []
    blockers: list[str] = []
    next_step: str | None = None
```

### `ReviewOutput`

Reviewer role output:

```python
class ReviewOutput(BaseModel):
    status: str  # APPROVED, CHANGES_REQUESTED, REJECTED
    review_status: str  # Same as status
    issues: list[ReviewIssue] = []
    suggestions: list[str] = []
    security_concerns: list[str] = []
    next_step: str | None = None
```

### `ReviewIssue`

Single issue found during review:

```python
class ReviewIssue(BaseModel):
    severity: str  # critical, high, medium, low
    file: str
    line: int | None = None
    description: str
    suggestion: str | None = None
```

## Error Models

### `ErrorCategory`

Six categories of errors:

| Category | Description |
|----------|-------------|
| `NETWORK` | Timeouts, DNS, connection failures |
| `CLI_ERROR` | Non-zero exit codes, CLI crashes |
| `VALIDATION` | Output doesn't match schema |
| `PARSING` | Can't extract structured data |
| `LOGIC` | Wrong/incomplete result |
| `RESOURCE` | Out of memory, disk full |

### `ErrorAction`

What to do when error occurs:

| Action | Description |
|--------|-------------|
| `RETRY_SAME` | Retry with same context |
| `RETRY_WITH_FEEDBACK` | Retry with error appended |
| `RETRY_ONCE` | Single retry then escalate |
| `ESCALATE` | Immediate human intervention |
