"""SQLite state management with event sourcing.

The events table is the write model (source of truth).
Other tables are read models (projections) rebuilt from events.
"""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from supervisor.core.models import (
    Component,
    ComponentStatus,
    Feature,
    FeatureStatus,
    Phase,
    PhaseStatus,
    Step,
    StepStatus,
)


class EventType(str, Enum):
    """Types of events in the event log."""

    # Workflow events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"

    # Step events
    STEP_STARTED = "step_started"
    STEP_APPLYING = "step_applying"  # Emitted before applying changes (crash recovery)
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_RETRIED = "step_retried"

    # Gate events
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"

    # Feature/Phase/Component events (hierarchical)
    FEATURE_CREATED = "feature_created"
    FEATURE_COMPLETED = "feature_completed"
    PHASE_CREATED = "phase_created"
    PHASE_COMPLETED = "phase_completed"
    COMPONENT_CREATED = "component_created"
    COMPONENT_STARTED = "component_started"
    COMPONENT_COMPLETED = "component_completed"
    COMPONENT_FAILED = "component_failed"

    # Checkpoint events
    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_RESTORED = "checkpoint_restored"

    # Human intervention
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_SKIPPED = "approval_skipped"  # FIX (v26): Distinct event for SKIP decisions

    # Metadata events (Phase 5)
    METADATA_SET = "metadata_set"  # FIX (v24): For SKIP persistence


def _utc_now() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


class _SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime and Pydantic models.

    Prevents TypeError when serializing payloads containing:
    - datetime objects (converted to ISO format strings)
    - Pydantic BaseModel instances (converted via model_dump)
    - Path objects (converted to strings)
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _safe_json_dumps(obj: Any) -> str:
    """Serialize object to JSON string, handling datetime and Pydantic models."""
    return json.dumps(obj, cls=_SafeJSONEncoder)


class Event(BaseModel):
    """Immutable event in the event log."""

    id: int | None = None
    workflow_id: str
    event_type: EventType
    role: str | None = None
    step_id: str | None = None
    component_id: str | None = None
    status: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)


class Database:
    """SQLite database with event sourcing for workflow state."""

    SCHEMA = """
    -- Event log (immutable, source of truth)
    CREATE TABLE IF NOT EXISTS events (
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

    -- Workflow state (projection)
    CREATE TABLE IF NOT EXISTS workflows (
        id TEXT PRIMARY KEY,
        current_step TEXT,
        status TEXT NOT NULL,
        context JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_by_event_id INTEGER DEFAULT 0  -- Ordering guard for parallel updates
    );

    -- Steps (projection)
    CREATE TABLE IF NOT EXISTS steps (
        id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL,
        role TEXT NOT NULL,
        status TEXT NOT NULL,
        gates JSON,
        context JSON,
        output JSON,
        error TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_by_event_id INTEGER DEFAULT 0,  -- Ordering guard for parallel updates
        FOREIGN KEY (workflow_id) REFERENCES workflows(id)
    );

    -- Features (hierarchical - projection)
    CREATE TABLE IF NOT EXISTS features (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        updated_by_event_id INTEGER DEFAULT 0  -- Ordering guard for parallel updates
    );

    -- Phases (hierarchical - projection)
    CREATE TABLE IF NOT EXISTS phases (
        id TEXT PRIMARY KEY,
        feature_id TEXT NOT NULL,
        title TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        status TEXT NOT NULL,
        interfaces JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_by_event_id INTEGER DEFAULT 0,  -- Ordering guard for parallel updates
        FOREIGN KEY (feature_id) REFERENCES features(id)
    );

    -- Components (hierarchical - projection)
    -- FIX (PR review): Added description column
    CREATE TABLE IF NOT EXISTS components (
        id TEXT PRIMARY KEY,
        phase_id TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT DEFAULT '',
        files JSON,
        depends_on JSON,
        status TEXT NOT NULL,
        assigned_role TEXT,
        output JSON,
        error TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_by_event_id INTEGER DEFAULT 0,  -- Ordering guard for parallel updates
        FOREIGN KEY (phase_id) REFERENCES phases(id)
    );

    -- Checkpoints for recovery
    CREATE TABLE IF NOT EXISTS checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workflow_id TEXT NOT NULL,
        step_id TEXT,
        git_sha TEXT,
        context_snapshot JSON,
        status TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (workflow_id) REFERENCES workflows(id)
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_events_workflow ON events(workflow_id);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    CREATE INDEX IF NOT EXISTS idx_components_phase ON components(phase_id);
    CREATE INDEX IF NOT EXISTS idx_components_status ON components(status);
    CREATE INDEX IF NOT EXISTS idx_phases_feature ON phases(feature_id);

    -- Metrics table for performance tracking
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        -- Dimensions
        role TEXT NOT NULL,
        cli TEXT NOT NULL,
        task_type TEXT,
        workflow_id TEXT,

        -- Measures
        success BOOLEAN NOT NULL,
        duration_seconds REAL NOT NULL,
        retry_count INTEGER DEFAULT 0,
        token_usage INTEGER,
        error_category TEXT,

        -- For adaptive routing
        model_score REAL
    );

    CREATE INDEX IF NOT EXISTS idx_metrics_role_cli ON metrics(role, cli);
    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

    -- Graph workflow definitions (stored as JSON)
    CREATE TABLE IF NOT EXISTS graph_workflows (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        definition JSON NOT NULL,
        version TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Graph workflow execution instances
    CREATE TABLE IF NOT EXISTS graph_executions (
        id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL,
        graph_id TEXT NOT NULL,
        status TEXT CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        error TEXT,
        FOREIGN KEY (graph_id) REFERENCES graph_workflows(id)
    );

    -- Individual node execution states
    CREATE TABLE IF NOT EXISTS node_executions (
        id TEXT PRIMARY KEY,
        execution_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        node_type TEXT NOT NULL,
        status TEXT CHECK(status IN ('pending', 'ready', 'running', 'completed', 'failed', 'skipped')),
        input_data JSON,
        output_data JSON,
        error TEXT,
        version INTEGER DEFAULT 0,  -- Tracks state changes (incremented on each status update)
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        UNIQUE(execution_id, node_id),
        FOREIGN KEY (execution_id) REFERENCES graph_executions(id)
    );

    -- Loop iteration counters (prevent infinite loops)
    CREATE TABLE IF NOT EXISTS loop_counters (
        execution_id TEXT NOT NULL,
        loop_key TEXT NOT NULL,
        iteration_count INTEGER DEFAULT 0,
        PRIMARY KEY (execution_id, loop_key),
        FOREIGN KEY (execution_id) REFERENCES graph_executions(id)
    );

    -- Indexes for graph execution queries
    CREATE INDEX IF NOT EXISTS idx_node_exec_status ON node_executions(execution_id, status);
    CREATE INDEX IF NOT EXISTS idx_node_ready ON node_executions(execution_id, status) WHERE status = 'ready';
    CREATE INDEX IF NOT EXISTS idx_graph_exec_status ON graph_executions(workflow_id, status);
    """

    def __init__(self, db_path: str | Path = ".supervisor/state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema and enable WAL mode for better concurrency."""
        with self._connect() as conn:
            # Enable WAL (Write-Ahead Logging) mode for better concurrent access
            # WAL allows readers and writers to operate simultaneously without blocking
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(self.SCHEMA)

        # Verify projection sync on startup
        self._verify_projection_sync()

    def record_metric(
        self,
        role: str,
        cli: str,
        workflow_id: str,
        success: bool,
        duration_seconds: float,
        task_type: str = "other",
        retry_count: int = 0,
        token_usage: int | None = None,
        error_category: str | None = None,
    ) -> None:
        """Record a metric for role execution.

        NEW method for Phase 5 - uses _connect() context manager like existing methods.
        FIX (v15 - Gemini): MUST use `with self._connect() as conn:` pattern.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics (
                    role, cli, task_type, workflow_id,
                    success, duration_seconds, retry_count,
                    token_usage, error_category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    role,
                    cli,
                    task_type,
                    workflow_id,
                    success,
                    duration_seconds,
                    retry_count,
                    token_usage,
                    error_category,
                ),
            )
            # commit handled by context manager

    def get_metrics(
        self,
        days: int = 30,
        role: str | None = None,
        cli: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query metrics with optional filters.

        NEW method for Phase 5 - supports metrics aggregation.
        FIX (v15 - Gemini): MUST use `with self._connect() as conn:` pattern.
        FIX (v14 - Codex): Use SQLite datetime() for cross-format comparison.
        """
        with self._connect() as conn:
            # FIX (v14): Use datetime() for format-agnostic timestamp comparison
            query = f"SELECT * FROM metrics WHERE timestamp > datetime('now', '-{days} days')"
            params: list[Any] = []

            if role:
                query += " AND role = ?"
                params.append(role)
            if cli:
                query += " AND cli = ?"
                params.append(cli)

            query += " ORDER BY timestamp DESC"
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

    def _verify_projection_sync(self) -> None:
        """Verify projections are in sync with event log.

        Checks that the max updated_by_event_id in projections matches
        the max event id. If out of sync, logs a warning (projections
        may be stale from a previous crash during projection update).
        """
        with self._connect() as conn:
            # Get max event id
            max_event = conn.execute("SELECT MAX(id) FROM events").fetchone()[0]
            if max_event is None:
                return  # No events yet, nothing to verify

            # Get max updated_by_event_id from each projection table
            # Some projections may not have been updated if they weren't affected
            projection_tables = ["workflows", "steps", "features", "phases", "components"]
            max_projection_id = 0

            for table in projection_tables:
                try:
                    result = conn.execute(
                        f"SELECT MAX(updated_by_event_id) FROM {table}"
                    ).fetchone()[0]
                    if result and result > max_projection_id:
                        max_projection_id = result
                except sqlite3.OperationalError:
                    pass  # Table doesn't have the column yet (migration pending)

            # If projections are behind, log a warning
            if max_projection_id < max_event:
                import sys

                print(
                    f"WARNING: Projection sync mismatch detected. "
                    f"Events: {max_event}, Projections: {max_projection_id}. "
                    f"Some projection updates may have failed. "
                    f"Consider replaying events or investigating the last {max_event - max_projection_id} events.",
                    file=sys.stderr,
                )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections.

        Uses a 30-second busy timeout to handle concurrent access gracefully
        instead of immediately failing with "database is locked".
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Additional busy timeout via PRAGMA for better concurrency handling
        conn.execute("PRAGMA busy_timeout = 30000")
        # Enable foreign key enforcement to maintain referential integrity
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.OperationalError as e:
            conn.rollback()
            if "database is locked" in str(e):
                raise sqlite3.OperationalError(
                    f"Database locked after 30s timeout. Check for long-running transactions: {e}"
                ) from e
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Explicit transaction context for atomic operations."""
        with self._connect() as conn:
            yield conn

    # --- Event Sourcing ---

    def append_event(self, event: Event) -> int:
        """Append an event to the log and update projections.

        IMPORTANT: Event insertion and projection updates are handled separately
        to ensure the event is never lost even if projection update fails.
        The event log is the source of truth - projections can be rebuilt.

        RACE CONDITION FIX: Projections use updated_by_event_id to ensure
        out-of-order projection updates don't regress state. Each projection
        update only applies if it's from a newer event than the last one
        that updated that row.
        """
        # Step 1: Insert event (commits immediately)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (workflow_id, event_type, role, step_id,
                                    component_id, status, payload, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.workflow_id,
                    event.event_type.value,
                    event.role,
                    event.step_id,
                    event.component_id,
                    event.status,
                    _safe_json_dumps(event.payload),
                    event.timestamp.isoformat(),
                ),
            )
            event_id = cursor.lastrowid

        # Step 2: Update projections in separate transaction
        # If this fails, the event is still recorded (source of truth preserved)
        # Projections can be rebuilt from events if needed
        # Pass event_id for ordering guard
        try:
            with self._connect() as conn:
                self._update_projections(conn, event, event_id)
        except Exception as e:
            # Log projection failure but don't lose the event
            import sys

            print(
                f"WARNING: Projection update failed for event {event_id} "
                f"({event.event_type.value}): {e}. "
                f"Event is recorded; projections may need rebuilding.",
                file=sys.stderr,
            )

        return event_id  # type: ignore

    def _update_projections(self, conn: sqlite3.Connection, event: Event, event_id: int) -> None:
        """Update read models based on event.

        RACE CONDITION FIX: Uses updated_by_event_id ordering guard.
        Updates only apply if event_id > current updated_by_event_id.
        This prevents out-of-order projection updates from regressing state.

        For INSERT operations, we use INSERT OR IGNORE + conditional UPDATE
        to handle race conditions where multiple events try to insert.
        """
        match event.event_type:
            case EventType.WORKFLOW_STARTED:
                # Insert if not exists, then conditionally update
                conn.execute(
                    """
                    INSERT OR IGNORE INTO workflows
                    (id, current_step, status, context, updated_by_event_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event.workflow_id,
                        event.payload.get("initial_step"),
                        "active",
                        _safe_json_dumps(event.payload.get("context", {})),
                        event_id,
                    ),
                )
                # Conditional update for race condition safety
                conn.execute(
                    """
                    UPDATE workflows SET current_step = ?, status = ?, context = ?,
                        updated_at = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        event.payload.get("initial_step"),
                        "active",
                        _safe_json_dumps(event.payload.get("context", {})),
                        _utc_now().isoformat(),
                        event_id,
                        event.workflow_id,
                        event_id,
                    ),
                )

            case EventType.STEP_STARTED:
                # Insert if not exists
                conn.execute(
                    """
                    INSERT OR IGNORE INTO steps
                    (id, workflow_id, role, status, gates, context, updated_by_event_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.step_id,
                        event.workflow_id,
                        event.role,
                        StepStatus.IN_PROGRESS.value,
                        _safe_json_dumps(event.payload.get("gates", [])),
                        _safe_json_dumps(event.payload.get("context", {})),
                        event_id,
                    ),
                )
                # Conditional update for race condition safety
                conn.execute(
                    """
                    UPDATE steps SET workflow_id = ?, role = ?, status = ?,
                        gates = ?, context = ?, updated_at = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        event.workflow_id,
                        event.role,
                        StepStatus.IN_PROGRESS.value,
                        _safe_json_dumps(event.payload.get("gates", [])),
                        _safe_json_dumps(event.payload.get("context", {})),
                        _utc_now().isoformat(),
                        event_id,
                        event.step_id,
                        event_id,
                    ),
                )
                # Conditional update for workflow current_step
                conn.execute(
                    """
                    UPDATE workflows SET current_step = ?, updated_at = ?,
                        updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        event.step_id,
                        _utc_now().isoformat(),
                        event_id,
                        event.workflow_id,
                        event_id,
                    ),
                )

            case EventType.STEP_APPLYING:
                # Mark step as applying changes (for crash recovery detection)
                # Conditional update for race condition safety
                conn.execute(
                    """
                    UPDATE steps SET status = ?, updated_at = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        "applying",
                        _utc_now().isoformat(),
                        event_id,
                        event.step_id,
                        event_id,
                    ),
                )

            case EventType.STEP_COMPLETED:
                # Conditional update for race condition safety
                conn.execute(
                    """
                    UPDATE steps SET status = ?, output = ?, updated_at = ?,
                        updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        StepStatus.COMPLETED.value,
                        _safe_json_dumps(event.payload.get("output", {})),
                        _utc_now().isoformat(),
                        event_id,
                        event.step_id,
                        event_id,
                    ),
                )

            case EventType.STEP_FAILED:
                # Conditional update for race condition safety
                conn.execute(
                    """
                    UPDATE steps SET status = ?, error = ?, updated_at = ?,
                        updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        StepStatus.FAILED.value,
                        event.payload.get("error"),
                        _utc_now().isoformat(),
                        event_id,
                        event.step_id,
                        event_id,
                    ),
                )

            case EventType.FEATURE_CREATED:
                # Insert if not exists
                conn.execute(
                    """
                    INSERT OR IGNORE INTO features
                    (id, title, description, status, updated_by_event_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event.payload["id"],
                        event.payload["title"],
                        event.payload.get("description", ""),
                        FeatureStatus.PLANNING.value,
                        event_id,
                    ),
                )
                # Conditional update for plan corrections/retries
                conn.execute(
                    """
                    UPDATE features SET title = ?, description = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        event.payload["title"],
                        event.payload.get("description", ""),
                        event_id,
                        event.payload["id"],
                        event_id,
                    ),
                )

            case EventType.FEATURE_COMPLETED:
                # Update feature status to COMPLETED with completed_at timestamp
                conn.execute(
                    """
                    UPDATE features SET status = ?, completed_at = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        FeatureStatus.COMPLETED.value,
                        _utc_now().isoformat(),
                        event_id,
                        event.payload.get("feature_id", event.workflow_id),
                        event_id,
                    ),
                )

            case EventType.PHASE_CREATED:
                # Insert if not exists
                conn.execute(
                    """
                    INSERT OR IGNORE INTO phases
                    (id, feature_id, title, sequence, status, interfaces, updated_by_event_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.payload["id"],
                        event.payload["feature_id"],
                        event.payload["title"],
                        event.payload["sequence"],
                        PhaseStatus.PENDING.value,
                        _safe_json_dumps(event.payload.get("interfaces", [])),
                        event_id,
                    ),
                )
                # Conditional update for plan corrections/retries
                conn.execute(
                    """
                    UPDATE phases SET feature_id = ?, title = ?, sequence = ?,
                        interfaces = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        event.payload["feature_id"],
                        event.payload["title"],
                        event.payload["sequence"],
                        _safe_json_dumps(event.payload.get("interfaces", [])),
                        event_id,
                        event.payload["id"],
                        event_id,
                    ),
                )

            case EventType.COMPONENT_CREATED:
                # Insert if not exists
                # FIX (PR review): Include description column
                conn.execute(
                    """
                    INSERT OR IGNORE INTO components (id, phase_id, title, description, files,
                                           depends_on, status, assigned_role, updated_by_event_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.payload["id"],
                        event.payload["phase_id"],
                        event.payload["title"],
                        event.payload.get("description", ""),
                        _safe_json_dumps(event.payload.get("files", [])),
                        _safe_json_dumps(event.payload.get("depends_on", [])),
                        ComponentStatus.PENDING.value,
                        event.payload.get("assigned_role"),
                        event_id,
                    ),
                )
                # Conditional update for plan corrections/retries
                conn.execute(
                    """
                    UPDATE components SET phase_id = ?, title = ?, description = ?, files = ?,
                        depends_on = ?, assigned_role = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        event.payload["phase_id"],
                        event.payload["title"],
                        event.payload.get("description", ""),
                        _safe_json_dumps(event.payload.get("files", [])),
                        _safe_json_dumps(event.payload.get("depends_on", [])),
                        event.payload.get("assigned_role"),
                        event_id,
                        event.payload["id"],
                        event_id,
                    ),
                )
                # RACE FIX: Reset parent phase to IN_PROGRESS if it was COMPLETED
                # This prevents the race where a component is added to a phase
                # that's concurrently being marked COMPLETED
                conn.execute(
                    """
                    UPDATE phases SET status = ?, updated_by_event_id = ?
                    WHERE id = ? AND status = ? AND updated_by_event_id < ?
                    """,
                    (
                        PhaseStatus.IN_PROGRESS.value,
                        event_id,
                        event.payload["phase_id"],
                        PhaseStatus.COMPLETED.value,
                        event_id,
                    ),
                )

            case EventType.COMPONENT_STARTED:
                # Conditional update for race condition safety
                conn.execute(
                    """
                    UPDATE components SET status = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        ComponentStatus.IMPLEMENTING.value,
                        event_id,
                        event.component_id,
                        event_id,
                    ),
                )

            case EventType.COMPONENT_COMPLETED:
                # Conditional update for race condition safety
                conn.execute(
                    """
                    UPDATE components SET status = ?, output = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        ComponentStatus.COMPLETED.value,
                        _safe_json_dumps(event.payload.get("output", {})),
                        event_id,
                        event.component_id,
                        event_id,
                    ),
                )
                # Check if phase is complete (pass event_id for ordering guard)
                self._check_phase_completion(conn, event.component_id, event_id)
                # FIX (Codex review): Also check for phase failure rollup
                # If a component failed earlier while others were pending, and now
                # all remaining components are complete, we need to mark phase as FAILED
                self._check_phase_failure(conn, event.component_id, event_id)

            case EventType.COMPONENT_FAILED:
                # Conditional update for race condition safety
                conn.execute(
                    """
                    UPDATE components SET status = ?, error = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    (
                        ComponentStatus.FAILED.value,
                        event.payload.get("error"),
                        event_id,
                        event.component_id,
                        event_id,
                    ),
                )
                # FIX (PR review): Roll up failure to phase/feature status
                self._check_phase_failure(conn, event.component_id, event_id)

            case EventType.WORKFLOW_COMPLETED:
                # Conditional update for race condition safety
                conn.execute(
                    """
                    UPDATE workflows SET status = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    ("completed", event_id, event.workflow_id, event_id),
                )

            case EventType.WORKFLOW_FAILED:
                # Conditional update for race condition safety
                conn.execute(
                    """
                    UPDATE workflows SET status = ?, updated_by_event_id = ?
                    WHERE id = ? AND updated_by_event_id < ?
                    """,
                    ("failed", event_id, event.workflow_id, event_id),
                )

    def _check_phase_completion(
        self, conn: sqlite3.Connection, component_id: str, event_id: int
    ) -> None:
        """Check if all components in a phase are complete.

        Uses ordering guard to prevent stale completion updates from
        overwriting newer state changes.
        """
        # Get the phase for this component
        row = conn.execute(
            "SELECT phase_id FROM components WHERE id = ?", (component_id,)
        ).fetchone()
        if not row:
            return

        phase_id = row["phase_id"]

        # Check if all components in this phase are complete
        incomplete = conn.execute(
            "SELECT COUNT(*) FROM components WHERE phase_id = ? AND status != ?",
            (phase_id, ComponentStatus.COMPLETED.value),
        ).fetchone()[0]

        if incomplete == 0:
            # Conditional update with ordering guard
            conn.execute(
                """
                UPDATE phases SET status = ?, updated_by_event_id = ?
                WHERE id = ? AND updated_by_event_id < ?
                """,
                (PhaseStatus.COMPLETED.value, event_id, phase_id, event_id),
            )
            # Check if feature is complete
            self._check_feature_completion(conn, phase_id, event_id)

    def _check_feature_completion(
        self, conn: sqlite3.Connection, phase_id: str, event_id: int
    ) -> None:
        """Check if all phases in a feature are complete.

        Uses ordering guard to prevent stale completion updates from
        overwriting newer state changes.
        """
        row = conn.execute("SELECT feature_id FROM phases WHERE id = ?", (phase_id,)).fetchone()
        if not row:
            return

        feature_id = row["feature_id"]

        incomplete = conn.execute(
            "SELECT COUNT(*) FROM phases WHERE feature_id = ? AND status != ?",
            (feature_id, PhaseStatus.COMPLETED.value),
        ).fetchone()[0]

        if incomplete == 0:
            # Conditional update with ordering guard
            conn.execute(
                """
                UPDATE features SET status = ?, completed_at = ?, updated_by_event_id = ?
                WHERE id = ? AND updated_by_event_id < ?
                """,
                (
                    FeatureStatus.COMPLETED.value,
                    _utc_now().isoformat(),
                    event_id,
                    feature_id,
                    event_id,
                ),
            )

    def _check_phase_failure(
        self, conn: sqlite3.Connection, component_id: str, event_id: int
    ) -> None:
        """Check if a phase should be marked as FAILED after a component failure.

        FIX (PR review): Roll up component failures to phase/feature status.
        A phase is marked FAILED when all components are terminal (COMPLETED/FAILED)
        and at least one is FAILED.
        """
        # Get the phase for this component
        row = conn.execute(
            "SELECT phase_id FROM components WHERE id = ?", (component_id,)
        ).fetchone()
        if not row:
            return

        phase_id = row["phase_id"]

        # Check if any components are still pending or in progress
        non_terminal = conn.execute(
            """
            SELECT COUNT(*) FROM components
            WHERE phase_id = ? AND status NOT IN (?, ?)
            """,
            (phase_id, ComponentStatus.COMPLETED.value, ComponentStatus.FAILED.value),
        ).fetchone()[0]

        if non_terminal > 0:
            # Some components still running - don't mark phase as failed yet
            return

        # All components are terminal - check if any failed
        failed_count = conn.execute(
            "SELECT COUNT(*) FROM components WHERE phase_id = ? AND status = ?",
            (phase_id, ComponentStatus.FAILED.value),
        ).fetchone()[0]

        if failed_count > 0:
            # Mark phase as FAILED
            conn.execute(
                """
                UPDATE phases SET status = ?, updated_by_event_id = ?
                WHERE id = ? AND updated_by_event_id < ?
                """,
                (PhaseStatus.FAILED.value, event_id, phase_id, event_id),
            )
            # Check if feature should also be marked as failed
            self._check_feature_failure(conn, phase_id, event_id)

    def _check_feature_failure(
        self, conn: sqlite3.Connection, phase_id: str, event_id: int
    ) -> None:
        """Check if a feature should be marked as FAILED after a phase failure.

        FIX (PR review): Roll up phase failures to feature status.
        A feature is marked FAILED when all phases are terminal (COMPLETED/FAILED)
        and at least one is FAILED.
        """
        row = conn.execute("SELECT feature_id FROM phases WHERE id = ?", (phase_id,)).fetchone()
        if not row:
            return

        feature_id = row["feature_id"]

        # Check if any phases are still pending or in progress
        non_terminal = conn.execute(
            """
            SELECT COUNT(*) FROM phases
            WHERE feature_id = ? AND status NOT IN (?, ?)
            """,
            (feature_id, PhaseStatus.COMPLETED.value, PhaseStatus.FAILED.value),
        ).fetchone()[0]

        if non_terminal > 0:
            # Some phases still running - don't mark feature as failed yet
            return

        # All phases are terminal - check if any failed
        failed_count = conn.execute(
            "SELECT COUNT(*) FROM phases WHERE feature_id = ? AND status = ?",
            (feature_id, PhaseStatus.FAILED.value),
        ).fetchone()[0]

        if failed_count > 0:
            # Mark feature as FAILED
            conn.execute(
                """
                UPDATE features SET status = ?, updated_by_event_id = ?
                WHERE id = ? AND updated_by_event_id < ?
                """,
                (FeatureStatus.FAILED.value, event_id, feature_id, event_id),
            )

    # --- Query Methods ---

    def get_events(
        self, workflow_id: str, event_types: list[EventType] | None = None
    ) -> list[Event]:
        """Get events for a workflow, optionally filtered by type."""
        with self._connect() as conn:
            if event_types:
                placeholders = ",".join("?" * len(event_types))
                rows = conn.execute(
                    f"""
                    SELECT * FROM events
                    WHERE workflow_id = ? AND event_type IN ({placeholders})
                    ORDER BY id
                    """,
                    [workflow_id] + [et.value for et in event_types],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events WHERE workflow_id = ? ORDER BY id",
                    (workflow_id,),
                ).fetchall()

            return [self._row_to_event(row) for row in rows]

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        """Convert database row to Event."""
        return Event(
            id=row["id"],
            workflow_id=row["workflow_id"],
            event_type=EventType(row["event_type"]),
            role=row["role"],
            step_id=row["step_id"],
            component_id=row["component_id"],
            status=row["status"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def get_step(self, step_id: str) -> Step | None:
        """Get a step by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM steps WHERE id = ?", (step_id,)).fetchone()
            if not row:
                return None
            return Step(
                id=row["id"],
                workflow_id=row["workflow_id"],
                role=row["role"],
                status=StepStatus(row["status"]),
                gates=json.loads(row["gates"]) if row["gates"] else [],
                context=json.loads(row["context"]) if row["context"] else {},
                output=json.loads(row["output"]) if row["output"] else None,
                error=row["error"],
            )

    def update_step(
        self,
        step_id: str,
        status: StepStatus,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update step status and output.

        DEPRECATED: This method bypasses the event-sourcing pattern and
        ordering guards. Prefer using append_event() with appropriate
        EventType (STEP_COMPLETED, STEP_FAILED) instead.

        WARNING: Direct updates can race with event-sourced updates and
        potentially regress state. Use only when you need to update a step
        without recording an event (e.g., during migration/recovery).
        """
        import warnings

        warnings.warn(
            "update_step bypasses event-sourcing. Use append_event() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE steps SET status = ?, output = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    _safe_json_dumps(output) if output is not None else None,
                    error,
                    _utc_now().isoformat(),
                    step_id,
                ),
            )

    def get_feature(self, feature_id: str) -> Feature | None:
        """Get a feature by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM features WHERE id = ?", (feature_id,)).fetchone()
            if not row:
                return None
            return Feature(
                id=row["id"],
                title=row["title"],
                description=row["description"] or "",
                status=FeatureStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                completed_at=(
                    datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                ),
            )

    def get_phases(self, feature_id: str) -> list[Phase]:
        """Get all phases for a feature."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM phases WHERE feature_id = ? ORDER BY sequence",
                (feature_id,),
            ).fetchall()
            return [
                Phase(
                    id=row["id"],
                    feature_id=row["feature_id"],
                    title=row["title"],
                    sequence=row["sequence"],
                    status=PhaseStatus(row["status"]),
                    interfaces=json.loads(row["interfaces"]) if row["interfaces"] else [],
                )
                for row in rows
            ]

    def get_components(self, feature_id: str) -> list[Component]:
        """Get all components for a feature (across all phases)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.* FROM components c
                JOIN phases p ON c.phase_id = p.id
                WHERE p.feature_id = ?
                ORDER BY p.sequence, c.id
                """,
                (feature_id,),
            ).fetchall()
            return [self._row_to_component(row) for row in rows]

    def get_components_by_status(self, status: ComponentStatus) -> list[Component]:
        """Get all components with a given status."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM components WHERE status = ?", (status.value,)
            ).fetchall()
            return [self._row_to_component(row) for row in rows]

    def get_component(self, component_id: str) -> Component | None:
        """Get a component by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM components WHERE id = ?", (component_id,)).fetchone()
            if not row:
                return None
            return self._row_to_component(row)

    def _row_to_component(self, row: sqlite3.Row) -> Component:
        """Convert database row to Component."""
        # FIX (PR review): Include description field
        return Component(
            id=row["id"],
            phase_id=row["phase_id"],
            title=row["title"],
            description=row["description"] or "",
            files=json.loads(row["files"]) if row["files"] else [],
            depends_on=json.loads(row["depends_on"]) if row["depends_on"] else [],
            status=ComponentStatus(row["status"]),
            assigned_role=row["assigned_role"],
            output=json.loads(row["output"]) if row["output"] else None,
            error=row["error"],
        )

    def update_component(
        self,
        component_id: str,
        status: ComponentStatus,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update component status and output.

        DEPRECATED: This method bypasses the event-sourcing pattern and
        ordering guards. Prefer using append_event() with appropriate
        EventType (COMPONENT_COMPLETED, COMPONENT_FAILED) instead.

        WARNING: Direct updates can race with event-sourced updates and
        potentially regress state. Use only when you need to update a component
        without recording an event (e.g., during migration/recovery).
        """
        import warnings

        warnings.warn(
            "update_component bypasses event-sourcing. Use append_event() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        with self._connect() as conn:
            conn.execute(
                "UPDATE components SET status = ?, output = ?, error = ? WHERE id = ?",
                (
                    status.value,
                    _safe_json_dumps(output) if output is not None else None,
                    error,
                    component_id,
                ),
            )

    # --- Phase 4: Helper Methods for Feature/Phase/Component Creation ---

    def create_feature(
        self,
        feature_id: str,
        title: str,
        description: str = "",
    ) -> None:
        """Create a new feature via event sourcing.

        Args:
            feature_id: Unique feature identifier (e.g., "F-ABC12345")
            title: Feature title
            description: Detailed description
        """
        self.append_event(
            Event(
                workflow_id=feature_id,
                event_type=EventType.FEATURE_CREATED,
                payload={
                    "id": feature_id,
                    "title": title,
                    "description": description,
                },
            )
        )

    def create_phase(
        self,
        phase_id: str,
        feature_id: str,
        title: str,
        sequence: int,
        interfaces: dict | list | None = None,
    ) -> None:
        """Create a new phase via event sourcing.

        Args:
            phase_id: Unique phase identifier (e.g., "F-ABC12345-PH1")
            feature_id: Parent feature ID
            title: Phase title
            sequence: Execution order (1-based)
            interfaces: Interface definitions for this phase
        """
        self.append_event(
            Event(
                workflow_id=feature_id,
                event_type=EventType.PHASE_CREATED,
                payload={
                    "id": phase_id,
                    "feature_id": feature_id,
                    "title": title,
                    "sequence": sequence,
                    "interfaces": interfaces or [],
                },
            )
        )

    def create_component(
        self,
        component_id: str,
        phase_id: str,
        title: str,
        files: list[str] | None = None,
        depends_on: list[str] | None = None,
        assigned_role: str = "implementer",
        description: str = "",
        feature_id: str | None = None,
    ) -> None:
        """Create a new component via event sourcing.

        Args:
            component_id: Unique component identifier (e.g., "F-ABC12345-PH1-C1")
            phase_id: Parent phase ID
            title: Component title
            files: List of files this component will create/modify
            depends_on: List of component IDs this depends on
            assigned_role: Role to execute this component (default: "implementer")
            description: Component description
            feature_id: Parent feature ID (optional, avoids DB lookup if provided)
        """
        # FIX (PR review): Accept feature_id directly to avoid unnecessary DB lookup
        # Caller (workflow.py) already knows the feature_id, so pass it through
        if feature_id is None:
            # Fallback: Get feature_id from phase_id (for backwards compatibility)
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT feature_id FROM phases WHERE id = ?", (phase_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"Phase '{phase_id}' not found")
                feature_id = row["feature_id"]

        self.append_event(
            Event(
                workflow_id=feature_id,
                event_type=EventType.COMPONENT_CREATED,
                payload={
                    "id": component_id,
                    "phase_id": phase_id,
                    "title": title,
                    "files": files or [],
                    "depends_on": depends_on or [],
                    "assigned_role": assigned_role,
                    "description": description,
                },
            )
        )

    def get_phase(self, phase_id: str) -> Phase | None:
        """Get a phase by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM phases WHERE id = ?", (phase_id,)).fetchone()
            if not row:
                return None
            return Phase(
                id=row["id"],
                feature_id=row["feature_id"],
                title=row["title"],
                sequence=row["sequence"],
                status=PhaseStatus(row["status"]),
                interfaces=json.loads(row["interfaces"]) if row["interfaces"] else [],
            )

    def update_feature_status(
        self,
        feature_id: str,
        status: FeatureStatus,
    ) -> None:
        """Update feature status directly (for workflow state transitions).

        Note: For completion, prefer relying on automatic rollup from
        _check_feature_completion(). Use this for explicit state changes
        like PLANNING -> IN_PROGRESS -> REVIEW.

        FIX (Codex review): Use MAX(id) not MAX(id)+1 to avoid skipping next event.
        Setting to MAX(id)+1 would equal the next event's ID, causing projections
        to skip that event since they only update when updated_by_event_id < event_id.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE features SET status = ?, updated_by_event_id = (
                    SELECT COALESCE(MAX(id), 0) FROM events
                )
                WHERE id = ?
                """,
                (status.value, feature_id),
            )

    def update_phase_status(
        self,
        phase_id: str,
        status: PhaseStatus,
    ) -> None:
        """Update phase status directly (for workflow state transitions).

        FIX (Codex review): Use MAX(id) not MAX(id)+1 to avoid skipping next event.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE phases SET status = ?, updated_by_event_id = (
                    SELECT COALESCE(MAX(id), 0) FROM events
                )
                WHERE id = ?
                """,
                (status.value, phase_id),
            )

    def update_component_dependencies(
        self,
        component_id: str,
        depends_on: list[str],
    ) -> None:
        """Update component dependencies (for dependency ID remapping).

        Used during Phase 4 workflow to remap symbolic IDs to generated IDs.

        FIX (Codex review): Use MAX(id) not MAX(id)+1 to avoid skipping next event.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE components SET depends_on = ?, updated_by_event_id = (
                    SELECT COALESCE(MAX(id), 0) FROM events
                )
                WHERE id = ?
                """,
                (_safe_json_dumps(depends_on), component_id),
            )

    # --- Checkpoint Methods ---

    def create_checkpoint(
        self,
        workflow_id: str,
        step_id: str | None,
        git_sha: str,
        context: dict[str, Any],
        status: str = "active",
    ) -> int:
        """Create a checkpoint for recovery."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO checkpoints (workflow_id, step_id, git_sha, context_snapshot, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (workflow_id, step_id, git_sha, _safe_json_dumps(context), status),
            )
            return cursor.lastrowid  # type: ignore

    def get_latest_checkpoint(self, workflow_id: str) -> dict[str, Any] | None:
        """Get the most recent checkpoint for a workflow."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE workflow_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (workflow_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "workflow_id": row["workflow_id"],
                "step_id": row["step_id"],
                "git_sha": row["git_sha"],
                "context": json.loads(row["context_snapshot"]),
                "status": row["status"],
                "timestamp": row["timestamp"],
            }
