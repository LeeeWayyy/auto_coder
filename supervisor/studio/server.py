"""FastAPI backend for Supervisor Studio.

This module provides:
- REST API for workflow CRUD operations
- WebSocket for real-time execution updates
- Static file serving for the frontend

Architecture Notes (Phase 3):
- Real-time WebSocket updates use in-memory callbacks registered with
  GraphOrchestrator. This means Studio can only provide real-time visualization
  for executions started by this Studio server instance. Executions started via
  CLI or other processes will show in the list (from DB) but won't receive
  real-time updates. Cross-process real-time updates would require implementing
  a pub/sub mechanism (e.g., DB polling, event table) in a future phase.
- There is no auto-resume logic for executions on server restart. Running
  executions from a previous server session will remain in their last known
  state until manually cancelled or restarted. This is a known limitation
  suitable for local development use cases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from supervisor.core.engine import ExecutionEngine
from supervisor.core.graph_engine import GraphOrchestrator
from supervisor.core.graph_schema import WorkflowGraph
from supervisor.core.state import Database
from supervisor.core.worker import WorkflowWorker
from supervisor.sandbox.executor import DockerNotAvailableError, EgressNotConfiguredError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Supervisor Studio API",
    description="API for visual workflow management",
    version="1.0.0",
)

# Expose host CLI mode to the frontend (for warning banner).
HOST_CLI_MODE = os.environ.get("SUPERVISOR_USE_HOST_CLI") == "1"


# CORS for local development - restricted to known origins
# NOTE: This is intentionally restrictive. The studio is designed for localhost-only use.
# If deployed behind a reverse proxy, client_host will be the proxy IP (often 127.0.0.1),
# which could allow remote access. This is a known limitation - production deployment
# requires proper authentication which is out of scope for this local development tool.
def _get_allowed_origins() -> list[str]:
    """Build allowed origins list including the configured port."""
    # Base origins for common development ports
    origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]

    # Add configured port from CLI (allows custom --port to work)
    configured_port = os.environ.get("SUPERVISOR_STUDIO_PORT")
    if configured_port and configured_port not in ("3000", "5173", "8000"):
        origins.extend(
            [
                f"http://localhost:{configured_port}",
                f"http://127.0.0.1:{configured_port}",
            ]
        )

    return origins


ALLOWED_ORIGINS = _get_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def validate_request_origin(request: Request, call_next):
    """
    Validate request origin for security.

    For state-changing requests (POST/PUT/DELETE):
    - Browser requests: Validate Origin header against allowed list
    - Non-browser requests: Must come from localhost

    For read-only requests (GET):
    - If Origin header present: Validate against allowed list
    - If no Origin: Allow from localhost only (prevents data exposure if bound externally)

    This provides defense-in-depth even without authentication. The primary protection
    is localhost-only binding; this middleware is a second layer if that's bypassed.
    """
    # Skip validation for WebSocket upgrade requests (handled separately)
    if request.headers.get("upgrade", "").lower() == "websocket":
        return await call_next(request)

    origin = request.headers.get("origin")
    client_host = request.client.host if request.client else None

    if request.method in ["POST", "PUT", "DELETE"]:
        # State-changing requests require strict validation
        if origin:
            if origin not in ALLOWED_ORIGINS:
                return JSONResponse(
                    status_code=403,
                    content={"detail": f"Origin '{origin}' not allowed"},
                )
        else:
            # No Origin - must be localhost
            if client_host not in ("127.0.0.1", "localhost", "::1"):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Origin header required for non-localhost requests"},
                )
    elif request.method == "GET":
        # Read-only requests: validate if Origin present, otherwise require localhost
        # This prevents data leakage if server is accidentally bound to external interface
        if origin:
            if origin not in ALLOWED_ORIGINS:
                return JSONResponse(
                    status_code=403,
                    content={"detail": f"Origin '{origin}' not allowed"},
                )
        else:
            # No Origin - allow localhost only (blocks external data access)
            if client_host not in ("127.0.0.1", "localhost", "::1"):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Non-localhost access without Origin header not allowed"},
                )

    return await call_next(request)


# Global instances - initialized lazily
_db: Database | None = None
_engine: ExecutionEngine | None = None
_orchestrator: GraphOrchestrator | None = None
_project_root: Path | None = None


def _find_project_root() -> Path:
    """Find the project root by looking for .supervisor directory.

    Walks up from cwd looking for .supervisor/state.db.
    Falls back to cwd if not found (init will create it).
    """
    global _project_root
    if _project_root is not None:
        return _project_root

    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".supervisor" / "state.db").exists():
            _project_root = parent
            return parent
        if (parent / ".supervisor").exists():
            _project_root = parent
            return parent

    # Fall back to cwd - Database will create .supervisor if needed
    _project_root = current
    return current


def get_db() -> Database:
    """Get or create database instance."""
    global _db
    if _db is None:
        root = _find_project_root()
        _db = Database(root / ".supervisor" / "state.db")
    return _db


def get_orchestrator() -> GraphOrchestrator:
    """Get or create orchestrator instance."""
    global _engine, _orchestrator
    if _orchestrator is None:
        db = get_db()
        root = _find_project_root()
        _engine = ExecutionEngine(root)
        _orchestrator = GraphOrchestrator(db, _engine, _engine.gate_executor, _engine.gate_loader)
    return _orchestrator


# ========== API Models ==========


class WorkflowCreateRequest(BaseModel):
    """Request to create a new workflow"""

    graph: WorkflowGraph


class WorkflowUpdateRequest(BaseModel):
    """Request to update an existing workflow"""

    graph: WorkflowGraph


class ExecutionRequest(BaseModel):
    """Request to execute a workflow"""

    graph_id: str
    workflow_id: str | None = None
    input_data: dict[str, Any] = Field(default_factory=dict)


class HumanResponseRequest(BaseModel):
    """Request to respond to a human-in-the-loop node"""

    node_id: str
    action: Literal["approve", "reject", "edit"]
    feedback: str | None = None
    edited_data: dict[str, Any] | None = None


class ExecutionResponse(BaseModel):
    """Execution status response"""

    execution_id: str
    workflow_id: str
    graph_id: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class NodeStatusResponse(BaseModel):
    """Node execution status"""

    node_id: str
    node_type: str
    status: str
    output: dict[str, Any] | None = None
    error: str | None = None
    version: int


# ========== Workflow CRUD Endpoints ==========


@app.get("/api/workflows")
def list_workflows() -> list[dict[str, Any]]:
    """List all saved workflows (excludes execution snapshots)."""
    db = get_db()
    with db._connect() as conn:
        # Filter out execution snapshots (id starts with 'snapshot_')
        rows = conn.execute(
            """
            SELECT definition FROM graph_workflows
            WHERE id NOT LIKE 'snapshot_%'
            ORDER BY updated_at DESC
        """
        ).fetchall()

    return [json.loads(row[0]) for row in rows]


@app.get("/api/workflows/{graph_id}")
def get_workflow(graph_id: str) -> dict[str, Any]:
    """Get a specific workflow by ID."""
    db = get_db()
    with db._connect() as conn:
        row = conn.execute(
            "SELECT definition FROM graph_workflows WHERE id = ?", (graph_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return json.loads(row[0])


@app.post("/api/workflows", status_code=201)
def create_workflow(request: WorkflowCreateRequest) -> dict[str, Any]:
    """Create a new workflow."""
    graph_id = request.graph.id.strip()
    # Prevent reserved prefix to avoid collision with execution snapshots
    if graph_id.startswith("snapshot_"):
        raise HTTPException(
            status_code=400,
            detail="Workflow ID cannot start with 'snapshot_' (reserved for execution snapshots)",
        )
    if not graph_id:
        raise HTTPException(status_code=400, detail="Workflow ID cannot be empty")

    graph = request.graph
    if graph.id != graph_id:
        graph = graph.model_copy(update={"id": graph_id})

    # Validate graph
    errors = graph.validate_graph()
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})

    db = get_db()

    # Save to database with conflict handling
    try:
        with db._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_workflows (id, name, definition, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    graph_id,
                    graph.name,
                    graph.model_dump_json(),
                    graph.version,
                    datetime.now(UTC),
                    datetime.now(UTC),
                ),
            )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail=f"Workflow with ID '{graph_id}' already exists")

    return graph.model_dump()


@app.put("/api/workflows/{graph_id}")
def update_workflow(graph_id: str, request: WorkflowUpdateRequest) -> dict[str, Any]:
    """Update an existing workflow (snapshots are immutable and cannot be updated)."""
    if request.graph.id != graph_id:
        raise HTTPException(status_code=400, detail="Workflow ID mismatch")

    # Reject updates to execution snapshots (they are immutable)
    if graph_id.startswith("snapshot_"):
        raise HTTPException(
            status_code=403, detail="Cannot modify execution snapshots - they are immutable"
        )

    # Validate
    errors = request.graph.validate_graph()
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})

    db = get_db()

    # Update
    with db._connect() as conn:
        result = conn.execute(
            """
            UPDATE graph_workflows
            SET name = ?, definition = ?, version = ?, updated_at = ?
            WHERE id = ?
        """,
            (
                request.graph.name,
                request.graph.model_dump_json(),
                request.graph.version,
                datetime.now(UTC),
                graph_id,
            ),
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Workflow not found")

    return request.graph.model_dump()


@app.delete("/api/workflows/{graph_id}")
def delete_workflow(graph_id: str) -> dict[str, str]:
    """Delete a workflow (snapshots are protected and cannot be deleted directly)."""
    # Reject deletion of execution snapshots
    if graph_id.startswith("snapshot_"):
        raise HTTPException(
            status_code=403,
            detail="Cannot delete execution snapshots directly - delete the execution instead",
        )

    db = get_db()

    with db._connect() as conn:
        result = conn.execute("DELETE FROM graph_workflows WHERE id = ?", (graph_id,))

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Workflow not found")

    return {"status": "deleted", "id": graph_id}


# ========== Execution Endpoints ==========


@app.post("/api/execute")
async def execute_workflow(request: ExecutionRequest) -> ExecutionResponse:
    """Start workflow execution."""
    db = get_db()
    try:
        orchestrator = get_orchestrator()
    except (DockerNotAvailableError, EgressNotConfiguredError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Sandbox preflight failed. Start Docker Desktop or configure egress rules. ({exc})",
        ) from exc

    # Load workflow by graph_id
    def load_workflow():
        with db._connect() as conn:
            row = conn.execute(
                "SELECT definition FROM graph_workflows WHERE id = ?", (request.graph_id,)
            ).fetchone()
        return row

    row = await run_in_threadpool(load_workflow)

    if not row:
        raise HTTPException(status_code=404, detail="Workflow graph not found")

    workflow = WorkflowGraph.model_validate_json(row[0])

    # workflow_id is the user-provided label; defaults to graph_id
    run_label = request.workflow_id or request.graph_id

    # Start execution
    try:
        execution_id = await orchestrator.start_workflow(
            workflow, run_label, initial_inputs=request.input_data
        )
    except (DockerNotAvailableError, EgressNotConfiguredError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Sandbox preflight failed. Start Docker Desktop or configure egress rules. ({exc})",
        ) from exc

    # The engine creates an immutable snapshot with ID "snapshot_{execution_id}"
    snapshot_graph_id = f"snapshot_{execution_id}"

    # Register status callback for WebSocket broadcasting
    loop = asyncio.get_running_loop()

    async def on_status_change(node_id: str, status: str, output: dict, version: int):
        """Broadcast node status change to WebSocket clients.

        Version is passed directly from the orchestrator's transaction to ensure
        consistency between status and version (no race condition possible).
        """
        try:
            node_type = None
            for node in workflow.nodes:
                if node.id == node_id:
                    node_type = node.type
                    break

            await run_in_threadpool(
                db.append_execution_event,
                execution_id,
                "node_update",
                node_id,
                node_type,
                status,
                {"output": output},
                version,
            )

            await manager.broadcast(
                execution_id,
                {
                    "type": "node_update",
                    "node_id": node_id,
                    "status": status,
                    "output": output,
                    "version": version,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        except Exception as e:
            # Log error but don't propagate - avoid breaking WebSocket stream
            import logging

            logging.getLogger(__name__).warning(
                f"Error broadcasting status change for {execution_id}/{node_id}: {e}"
            )

    def sync_callback(node_id: str, status: str, output: dict, version: int):
        # Use run_coroutine_threadsafe since this is called from worker threads
        asyncio.run_coroutine_threadsafe(on_status_change(node_id, status, output, version), loop)

    orchestrator.register_status_callback(execution_id, sync_callback)

    # Start background worker
    asyncio.create_task(_run_execution_with_events(execution_id))

    # Read actual started_at from the database
    def get_started_at():
        with db._connect() as conn:
            row = conn.execute(
                "SELECT started_at FROM graph_executions WHERE id = ?", (execution_id,)
            ).fetchone()
        return row[0] if row else datetime.now(UTC).isoformat()

    started_at = await run_in_threadpool(get_started_at)

    return ExecutionResponse(
        execution_id=execution_id,
        workflow_id=run_label,
        graph_id=snapshot_graph_id,
        status="running",
        started_at=started_at,
    )


async def _run_execution_with_events(execution_id: str):
    """Background task to run workflow and broadcast completion."""
    db = get_db()
    orchestrator = get_orchestrator()
    worker = WorkflowWorker(orchestrator)

    try:
        await worker.run_until_complete(execution_id)

        # Re-read actual status and completion time to avoid race with cancellation
        def get_final_state():
            with db._connect() as conn:
                row = conn.execute(
                    "SELECT status, completed_at, error FROM graph_executions WHERE id = ?",
                    (execution_id,),
                ).fetchone()
            return row if row else (None, None, None)

        final_status, completed_at, error = await run_in_threadpool(get_final_state)

        # Only broadcast if not already cancelled
        if final_status and final_status != "cancelled":
            await run_in_threadpool(
                db.append_execution_event,
                execution_id,
                "execution_complete",
                None,
                None,
                final_status,
                {"completed_at": completed_at, "error": error},
                0,
            )
            payload: dict[str, Any] = {
                "type": "execution_complete",
                "status": final_status,
            }
            if completed_at:
                payload["completed_at"] = completed_at
            if error:
                payload["error"] = error
            await manager.broadcast(execution_id, payload)
    except Exception as e:
        logger.error(f"Execution {execution_id} failed: {e}")
        # Persist failure to DB (not just broadcast) so API responses are consistent
        error_msg = str(e)

        def mark_failed():
            with db._connect() as conn:
                conn.execute(
                    """
                    UPDATE graph_executions
                    SET status = 'failed', error = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND status = 'running'
                    """,
                    (error_msg, execution_id),
                )
                # Note: db._connect() context manager handles commit on exit

        await run_in_threadpool(mark_failed)

        # Re-check status after marking failed - if it was cancelled in the meantime,
        # don't broadcast "failed" (the cancel endpoint already broadcast cancellation)
        def get_status_after_fail():
            with db._connect() as conn:
                row = conn.execute(
                    "SELECT status FROM graph_executions WHERE id = ?", (execution_id,)
                ).fetchone()
            return row[0] if row else None

        final_status = await run_in_threadpool(get_status_after_fail)
        if final_status == "failed":
            await manager.broadcast(
                execution_id,
                {
                    "type": "execution_complete",
                    "status": "failed",
                    "error": error_msg,
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
    finally:
        orchestrator.unregister_status_callback(execution_id)


@app.post("/api/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str) -> dict[str, str]:
    """
    Cancel a running execution.

    NOTE: This sets the execution status to 'cancelled' but does NOT stop
    currently running tasks. Running tasks will complete in the background
    but their results will not advance the workflow.
    """
    db = get_db()
    orchestrator = get_orchestrator()

    result = await run_in_threadpool(orchestrator.cancel_execution, execution_id)

    if not result:
        # Check if execution exists
        def check_execution():
            with db._connect() as conn:
                row = conn.execute(
                    "SELECT status FROM graph_executions WHERE id = ?", (execution_id,)
                ).fetchone()
            return row

        row = await run_in_threadpool(check_execution)

        if not row:
            raise HTTPException(status_code=404, detail="Execution not found")

        raise HTTPException(status_code=400, detail=f"Cannot cancel execution in '{row[0]}' state")

    # Update running nodes to 'skipped' and broadcast final states to WebSocket clients.
    # This ensures the UI doesn't show nodes stuck in "running" state after cancellation.
    # Using 'skipped' because it's already a valid NodeStatus (cancelled nodes didn't complete).
    def get_final_nodes_and_cancel_running():
        with db._connect() as conn:
            # First, update any running nodes to skipped status
            conn.execute(
                """
                UPDATE node_executions
                SET status = 'skipped', version = version + 1
                WHERE execution_id = ? AND status = 'running'
                """,
                (execution_id,),
            )
            # Note: db._connect() context manager handles commit on exit

            # Then fetch all final node states
            rows = conn.execute(
                """
                SELECT node_id, status, version FROM node_executions
                WHERE execution_id = ?
                """,
                (execution_id,),
            ).fetchall()
        return [{"node_id": r[0], "status": r[1], "version": r[2]} for r in rows]

    final_nodes = await run_in_threadpool(get_final_nodes_and_cancel_running)

    await run_in_threadpool(
        db.append_execution_event,
        execution_id,
        "execution_complete",
        None,
        None,
        "cancelled",
        {"completed_at": datetime.now(UTC).isoformat()},
        0,
    )

    await manager.broadcast(
        execution_id,
        {"type": "execution_complete", "status": "cancelled", "final_nodes": final_nodes},
    )

    return {"status": "cancelled", "execution_id": execution_id}


@app.post("/api/executions/{execution_id}/respond")
async def respond_to_human_node(
    execution_id: str, request: HumanResponseRequest
) -> dict[str, str]:
    """Handle human response to interrupted execution."""
    db = get_db()

    def get_status():
        with db._connect() as conn:
            row = conn.execute(
                "SELECT status FROM graph_executions WHERE id = ?", (execution_id,)
            ).fetchone()
        return row[0] if row else None

    status = await run_in_threadpool(get_status)
    if not status:
        raise HTTPException(status_code=404, detail="Execution not found")
    if status != "interrupted":
        raise HTTPException(
            status_code=400,
            detail=f"Execution not interrupted (current status: {status})",
        )

    # TODO: Forward response to supervisor engine and update status accordingly.
    # This is a placeholder response until backend orchestration support is added.
    _ = request
    return {"status": "resumed"}


@app.get("/api/executions")
def list_executions(
    workflow_id: str | None = None,
    source_graph_id: str | None = None,
    limit: int = 50,
) -> list[ExecutionResponse]:
    """
    List recent executions with optional filtering.

    Args:
        workflow_id: Filter by user-provided run label (exact match)
        source_graph_id: Filter by original graph ID (before snapshot).
            NOTE: This filter works by matching workflow_id, which defaults to
            graph_id when not explicitly provided. If the execution was started
            with a custom workflow_id different from the graph_id, this filter
            will not find it. This is a known limitation of the current schema
            which does not store source_graph_id as a separate column.
        limit: Maximum number of results (default 50)
    """
    db = get_db()

    with db._connect() as conn:
        base_query = """
            SELECT id, workflow_id, graph_id, status, started_at, completed_at, error
            FROM graph_executions
        """
        conditions = []
        params: list[Any] = []

        if workflow_id:
            conditions.append("workflow_id = ?")
            params.append(workflow_id)
        elif source_graph_id:
            # Use workflow_id filter (defaults to graph_id when not overridden)
            # Note: workflow_id takes precedence if both are provided
            conditions.append("workflow_id = ?")
            params.append(source_graph_id)

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(base_query, params).fetchall()

    return [
        ExecutionResponse(
            execution_id=row[0],
            workflow_id=row[1],
            graph_id=row[2],
            status=row[3],
            started_at=str(row[4]) if row[4] else None,
            completed_at=str(row[5]) if row[5] else None,
            error=row[6],
        )
        for row in rows
    ]


@app.get("/api/executions/{execution_id}")
def get_execution(execution_id: str) -> ExecutionResponse:
    """Get execution status."""
    db = get_db()

    with db._connect() as conn:
        row = conn.execute(
            """
            SELECT workflow_id, graph_id, status, started_at, completed_at, error
            FROM graph_executions
            WHERE id = ?
        """,
            (execution_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Execution not found")

    return ExecutionResponse(
        execution_id=execution_id,
        workflow_id=row[0],
        graph_id=row[1],
        status=row[2],
        started_at=str(row[3]) if row[3] else None,
        completed_at=str(row[4]) if row[4] else None,
        error=row[5],
    )


@app.get("/api/executions/{execution_id}/nodes")
def get_execution_nodes(
    execution_id: str, since_version: int | None = None
) -> list[dict[str, Any]]:
    """
    Get node statuses for an execution.

    Args:
        execution_id: The execution ID
        since_version: Optional - only return nodes with version > this value.
            IMPORTANT: Node versions are incremented per-node, not globally.
            Using since_version may miss updates on nodes with lower versions
            than the cutoff. For complete state, omit this parameter.
            This filter is primarily useful for detecting if ANY updates have
            occurred (non-empty response means something changed), not for
            incremental state synchronization. For real-time updates, use the
            WebSocket endpoint instead.
    """
    db = get_db()

    with db._connect() as conn:
        if since_version is not None:
            rows = conn.execute(
                """
                SELECT node_id, node_type, status, output_data, error, version
                FROM node_executions
                WHERE execution_id = ? AND version > ?
                ORDER BY version DESC
            """,
                (execution_id, since_version),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT node_id, node_type, status, output_data, error, version
                FROM node_executions
                WHERE execution_id = ?
            """,
                (execution_id,),
            ).fetchall()

    return [
        {
            "node_id": row[0],
            "node_type": row[1],
            "status": row[2],
            "output": json.loads(row[3]) if row[3] else None,
            "error": row[4],
            "version": row[5],
        }
        for row in rows
    ]


@app.get("/api/executions/{execution_id}/history")
def get_execution_history(
    execution_id: str, since_id: int | None = None, limit: int = 500
) -> list[dict[str, Any]]:
    """Get execution event history for time-travel debugging."""
    db = get_db()
    if limit < 1:
        limit = 1
    if limit > 5000:
        limit = 5000
    return db.get_execution_events(execution_id, since_id=since_id, limit=limit)


@app.get("/api/executions/{execution_id}/stream")
async def stream_execution_events(
    execution_id: str, request: Request, since_id: int | None = None
) -> StreamingResponse:
    """Stream execution events via Server-Sent Events (SSE)."""
    db = get_db()
    last_id = since_id or 0

    async def event_generator():
        nonlocal last_id
        while True:
            if await request.is_disconnected():
                break
            events = await run_in_threadpool(
                db.get_execution_events, execution_id, last_id, 200
            )
            for event in events:
                last_id = max(last_id, int(event.get("id", last_id)))
                yield f"event: execution_event\ndata: {json.dumps(event)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ========== WebSocket for Live Updates ==========


class ConnectionManager:
    """
    Manage WebSocket connections with event-driven broadcasting.

    This uses a publish-subscribe model:
    - Clients connect and receive the initial state
    - The orchestrator's status callback broadcasts updates when node states change
    - All connected clients receive updates simultaneously
    """

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, execution_id: str):
        await websocket.accept()
        if execution_id not in self.active_connections:
            self.active_connections[execution_id] = []
        self.active_connections[execution_id].append(websocket)

    def disconnect(self, websocket: WebSocket, execution_id: str):
        if execution_id in self.active_connections:
            try:
                self.active_connections[execution_id].remove(websocket)
            except ValueError:
                pass
            if not self.active_connections[execution_id]:
                del self.active_connections[execution_id]

    async def broadcast(self, execution_id: str, message: dict):
        """Broadcast message to all clients watching this execution."""
        if execution_id not in self.active_connections:
            return

        connections = self.active_connections[execution_id][:]

        async def safe_send(conn: WebSocket):
            try:
                await conn.send_json(message)
            except Exception:
                self.disconnect(conn, execution_id)

        await asyncio.gather(*[safe_send(c) for c in connections], return_exceptions=True)


manager = ConnectionManager()


@app.websocket("/ws/executions/{execution_id}")
async def execution_websocket(websocket: WebSocket, execution_id: str):
    """
    WebSocket for live execution updates.

    Protocol:
    1. On connect: Send initial state with all node statuses
    2. During execution: Receive broadcasted node_update and execution_complete events
    3. Client can send "ping" to check connection, server responds with "pong"
    """
    # Validate Origin/host for WebSocket connections
    origin = websocket.headers.get("origin")
    client_host = websocket.client.host if websocket.client else None

    if origin:
        if origin not in ALLOWED_ORIGINS:
            await websocket.close(code=4003, reason="Origin not allowed")
            return
    else:
        if client_host not in ("127.0.0.1", "localhost", "::1"):
            await websocket.close(
                code=4003, reason="Non-localhost connections require Origin header"
            )
            return

    await manager.connect(websocket, execution_id)

    try:
        # Send initial state - handle missing execution gracefully
        # Wrap sync DB calls in threadpool to avoid blocking the event loop
        try:
            nodes = await run_in_threadpool(get_execution_nodes, execution_id, None)
            execution = await run_in_threadpool(get_execution, execution_id)
        except HTTPException as e:
            await websocket.send_json({"type": "error", "code": e.status_code, "detail": e.detail})
            await websocket.close(code=4004, reason=str(e.detail))
            return

        await websocket.send_json(
            {"type": "initial_state", "nodes": nodes, "status": execution.status}
        )

        # If already complete, send completion and close explicitly
        if execution.status in ["completed", "failed", "cancelled"]:
            await websocket.send_json({"type": "execution_complete", "status": execution.status})
            await websocket.close(code=1000, reason="Execution already complete")
            return

        # Keep connection alive and handle client messages
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if msg == "ping":
                    # Respond with JSON so client can parse consistently
                    await websocket.send_json({"type": "pong"})
            except TimeoutError:
                # Send heartbeat to detect dead connections
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, execution_id)


# ========== Static File Serving ==========

frontend_dir = Path(__file__).parent / "frontend" / "dist"

if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")

    @app.get("/")
    async def serve_index():
        import re

        index_path = frontend_dir / "index.html"
        content = index_path.read_text()
        if HOST_CLI_MODE:
            # Use case-insensitive regex for robustness against HTML minification
            content = re.sub(
                r"</head>",
                "<script>window.__SUPERVISOR_HOST_CLI__ = true;</script></head>",
                content,
                count=1,
                flags=re.IGNORECASE,
            )
        return Response(content, media_type="text/html")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """
        Serve SPA for client-side routing.

        SECURITY: Validates resolved path is within frontend_dir to prevent
        path traversal attacks.
        """
        # Skip API and WebSocket routes
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(status_code=404, detail="Not found")

        requested_path = (frontend_dir / full_path).resolve()
        if not requested_path.is_relative_to(frontend_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")

        if requested_path.exists() and requested_path.is_file():
            return FileResponse(requested_path)
        return FileResponse(frontend_dir / "index.html")
else:

    @app.get("/")
    async def studio_not_built():
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Studio UI not built.",
                "next_steps": [
                    "cd supervisor/studio/frontend",
                    "npm install",
                    "npm run dev  # open http://127.0.0.1:5173",
                    "npm run build  # serve from backend",
                ],
            },
        )

    @app.get("/favicon.ico")
    async def favicon_missing():
        return JSONResponse(status_code=404, content={"detail": "No favicon (UI not built)"})
