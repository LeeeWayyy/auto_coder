# SQLite Event Sourcing

**File:** `supervisor/core/state.py`

## Overview

SQLite-based state management using event sourcing pattern. The events table is the write model (source of truth), while other tables are read models (projections) rebuilt from events.

## Architecture

```
Events (Source of Truth)
    │
    ├── Workflows (Projection)
    ├── Steps (Projection)
    ├── Features (Projection)
    ├── Phases (Projection)
    ├── Components (Projection)
    └── Checkpoints (Recovery)
```

## Key Classes

### `Database`

Main database class with event sourcing.

```python
db = Database(db_path=".supervisor/state.db")
```

#### Event Methods

```python
# Append event (also updates projections)
event_id = db.append_event(event)

# Query events
events = db.get_events(workflow_id, event_types=None)
```

#### Query Methods

```python
db.get_step(step_id) -> Step | None
db.get_feature(feature_id) -> Feature | None
db.get_phases(feature_id) -> list[Phase]
db.get_components(feature_id) -> list[Component]
db.get_components_by_status(status) -> list[Component]
```

#### Update Methods

```python
db.update_step(step_id, status, output=None, error=None)
db.update_component(component_id, status, output=None, error=None)
```

#### Checkpoint Methods

```python
checkpoint_id = db.create_checkpoint(workflow_id, step_id, git_sha, context)
checkpoint = db.get_latest_checkpoint(workflow_id)
```

### `Event`

Immutable event in the event log:

```python
class Event(BaseModel):
    id: int | None = None
    workflow_id: str
    event_type: EventType
    role: str | None = None
    step_id: str | None = None
    component_id: str | None = None
    status: str | None = None
    payload: dict[str, Any] = {}
    timestamp: datetime
```

### `EventType`

Enum of all event types:

| Category | Events |
|----------|--------|
| Workflow | `WORKFLOW_STARTED`, `WORKFLOW_COMPLETED`, `WORKFLOW_FAILED` |
| Step | `STEP_STARTED`, `STEP_COMPLETED`, `STEP_FAILED`, `STEP_RETRIED` |
| Gate | `GATE_PASSED`, `GATE_FAILED` |
| Feature | `FEATURE_CREATED`, `FEATURE_COMPLETED` |
| Phase | `PHASE_CREATED`, `PHASE_COMPLETED` |
| Component | `COMPONENT_CREATED`, `COMPONENT_STARTED`, `COMPONENT_COMPLETED`, `COMPONENT_FAILED` |
| Checkpoint | `CHECKPOINT_CREATED`, `CHECKPOINT_RESTORED` |
| Approval | `APPROVAL_REQUESTED`, `APPROVAL_GRANTED`, `APPROVAL_DENIED` |

## Database Schema

### Events Table (Source of Truth)

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    role TEXT,
    step_id TEXT,
    component_id TEXT,
    status TEXT,
    payload JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Projection Tables

- `workflows`: Current workflow state
- `steps`: Step execution state
- `features`: Feature-level tracking
- `phases`: Phase-level tracking (sequences)
- `components`: Component-level tracking (files, dependencies)
- `checkpoints`: Recovery points

### Metrics Table

Stores execution performance metrics for adaptive routing and dashboards:
- `metrics` (role, cli, task_type, success, duration, retry_count, etc.)

### Graph Workflow Tables

Used by the declarative graph engine and Studio:
- `graph_workflows`: Stored workflow definitions (JSON)
- `graph_executions`: Execution instances and status
- `node_executions`: Per-node execution state with version counters
- `loop_counters`: Loop iteration tracking

## Concurrency Features

### WAL Mode

```sql
PRAGMA journal_mode=WAL
```

Enables concurrent readers and writers without blocking.

### Busy Timeout

```python
conn = sqlite3.connect(db_path, timeout=30.0)
conn.execute("PRAGMA busy_timeout = 30000")
```

30-second timeout before failing with "database is locked".

### Event Ordering

Events are ordered by `id` (autoincrement) rather than `timestamp` to ensure deterministic ordering even with identical timestamps.

## JSON Serialization

`_SafeJSONEncoder` handles:
- `datetime` objects (ISO format)
- Pydantic `BaseModel` instances
- `Path` objects

## Automatic Cascade Updates

When components complete, the system automatically:
1. Checks if all components in phase are complete
2. If so, marks phase as complete
3. Checks if all phases in feature are complete
4. If so, marks feature as complete
