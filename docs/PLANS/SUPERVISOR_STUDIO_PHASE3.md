# Supervisor Studio - Phase 3: Web UI Studio

**Status:** Planning
**Objective:** Build a web-based visual editor for designing, executing, and monitoring workflows.

**Prerequisites:** Phase 1 (Engine) and Phase 2 (CLI Visualization) completed.

---

## 3.1 Overview

Phase 3 delivers a full web-based Supervisor Studio with:
- Visual drag-and-drop workflow editor (ReactFlow)
- Real-time execution monitoring (WebSockets with event-driven updates)
- Workflow management (CRUD operations)
- Node configuration panels
- Execution history and cancellation support

---

## 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Supervisor Studio                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Frontend   │    │   Backend    │    │   Core Engine        │  │
│  │   (React)    │◄──►│   (FastAPI)  │◄──►│   (GraphOrchestrator)│  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│        │                    │                      │                │
│        │ WebSocket          │ REST API             │ Database      │
│        │ (Event-driven)     │                      │               │
│        ▼                    ▼                      ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  ReactFlow   │    │ Connection   │    │   SQLite             │  │
│  │  @tanstack   │    │  Manager     │    │   (State Store)      │  │
│  │  react-query │    │ (In-memory)  │    │                      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

> **Note:** The MVP implementation uses an in-memory `ConnectionManager` for WebSocket
> pub/sub with event-driven broadcasting. Redis can be added later for horizontal scaling.

---

## 3.2.1 Security Considerations

> **IMPORTANT: MVP Security Scope**
>
> This MVP does **not** include full authentication/authorization. Security measures:
>
> 1. **Localhost-only binding:** Server binds to `127.0.0.1` by default
> 2. **Origin validation:** CSRF-style protection via `Origin` header checking
> 3. **Non-localhost warning:** CLI prompts for confirmation if binding to external address
>
> **Before deploying to any shared or production environment:**
> - Add authentication middleware (e.g., OAuth2, JWT, session-based auth)
> - Implement per-workflow/execution authorization checks
> - Add WebSocket handshake authentication
> - Use HTTPS with proper certificates
>
> The `task_template` field allows arbitrary code execution by design (it's a workflow
> orchestrator). Never expose this service on a network without proper authentication.

---

## 3.3 Required Core Engine Modifications

**IMPORTANT:** Phase 3 requires the following modifications to `GraphOrchestrator` in
`supervisor/core/graph_engine.py`. These must be implemented before starting the Web UI.

### 3.3.1 Node Status Change Callback

Add callback support to `_set_node_status_guarded` to enable real-time WebSocket updates.

**IMPORTANT:** The callback must be in `_set_node_status_guarded`, NOT `_set_node_status`.
The guarded version handles all runtime status transitions (PENDING→READY, RUNNING→COMPLETED,
RUNNING→FAILED, etc.). The basic `_set_node_status` is only used for initial setup.

```python
class GraphOrchestrator:
    def __init__(self, db, engine, gate_executor, gate_loader):
        # ... existing init ...
        self._status_callbacks: Dict[str, Callable] = {}  # execution_id -> callback

    def register_status_callback(self, execution_id: str, callback: Callable[[str, str, dict], None]):
        """Register a callback to be invoked when node status changes.

        Args:
            execution_id: The execution to monitor
            callback: Function(node_id, status, output) called on status change

        Note: Callbacks are invoked from threadpool threads. Use
        asyncio.run_coroutine_threadsafe() for async callbacks.
        """
        self._status_callbacks[execution_id] = callback

    def unregister_status_callback(self, execution_id: str):
        """Remove callback for an execution."""
        self._status_callbacks.pop(execution_id, None)

    def _set_node_status_guarded(self, conn, execution_id: str, node_id: str,
                                  status: NodeStatus, expected_status: NodeStatus, ...):
        # ... existing implementation (returns True if update succeeded) ...
        return updated  # Return before callback - callback invoked after transaction commits
```

**IMPORTANT: Callback Timing and Implementation Paths**

The callback must be invoked AFTER the transaction commits and output is persisted, not inside
`_set_node_status_guarded`. If invoked too early, the callback may read NULL output.

**All status transition paths requiring callback invocation:**

| Transition | Location in graph_engine.py | When to invoke callback |
|------------|---------------------------|------------------------|
| PENDING → READY | `_update_dependents()` after evaluating dependencies | After transaction commits |
| READY → RUNNING | `claim_next_node()` when claiming node for execution | After node claimed |
| RUNNING → COMPLETED | `_execute_node()` after task completes successfully | After output persisted |
| RUNNING → FAILED | `_execute_node()` when task fails or gate fails | After error persisted |
| * → SKIPPED | `_skip_downstream_nodes()` for conditional branches | After skip persisted |

**Implementation pattern for each path:**

```python
# Example: In the task completion code path (after _run_in_transaction completes)
def _complete_node_execution(self, execution_id: str, node_id: str, output: dict):
    # Persist status and output in transaction
    def persist(conn):
        updated = self._set_node_status_guarded(conn, execution_id, node_id, NodeStatus.COMPLETED, ...)
        if updated:
            self._set_node_output(conn, execution_id, node_id, output)
        return updated

    was_updated = self._run_in_transaction(persist)

    # Invoke callback AFTER transaction commits (output is now readable)
    if was_updated and execution_id in self._status_callbacks:
        try:
            callback = self._status_callbacks[execution_id]
            callback(node_id, "completed", output)  # Pass output directly, don't re-read
        except Exception as e:
            import logging
            logging.warning(f"Status callback error: {e}")
```

**Implementation note:** Each of the paths above must be modified to invoke the callback.
The implementer should grep for `_set_node_status_guarded` calls and ensure each one
has callback invocation after its enclosing transaction commits.

### 3.3.2 Execution Cancellation

Add cancellation support to stop workflow execution:

```python
def cancel_execution(self, execution_id: str) -> bool:
    """Cancel a running execution.

    Sets execution status to 'cancelled' and prevents new nodes from being claimed.
    NOTE: This does NOT stop currently running tasks - they will complete but their
    results will be ignored. This is a known limitation for the MVP.

    Returns:
        True if cancellation was successful, False if execution was not running
    """
    with self.db._connect() as conn:
        result = conn.execute("""
            UPDATE graph_executions
            SET status = 'cancelled', completed_at = ?
            WHERE id = ? AND status = 'running'
        """, (datetime.utcnow(), execution_id))

        if result.rowcount == 0:
            return False

        # Unregister any status callback
        self.unregister_status_callback(execution_id)
        return True
```

> **Known Limitation:** Cancellation sets the execution status to `cancelled` but does NOT
> terminate currently running tasks (threads/subprocesses). Running tasks will complete in
> the background but their results will not advance the workflow. Future versions may add
> proper task interruption.

---

## 3.4 Database Schema

The database tables required for Phase 3 **already exist** in `supervisor/core/state.py`.
Verify the following schema is present:

```sql
-- Graph workflow definitions (ALREADY EXISTS)
CREATE TABLE IF NOT EXISTS graph_workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    definition JSON NOT NULL,
    version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Graph workflow execution instances (ALREADY EXISTS)
CREATE TABLE IF NOT EXISTS graph_executions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,    -- User-provided run label
    graph_id TEXT NOT NULL,       -- References graph_workflows.id
    status TEXT CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT,
    FOREIGN KEY (graph_id) REFERENCES graph_workflows(id)
);

-- Node execution states (ALREADY EXISTS)
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
```

**Note:** The existing schema uses:
- `workflow_id` (not `run_label`) in `graph_executions`
- `version` INTEGER (not `updated_at` timestamp) in `node_executions` for change tracking
- No `logs` column - log capture is deferred to a future version

---

## 3.5 Backend API Server

**File:** `supervisor/studio/server.py` (New)

### FastAPI Application

```python
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supervisor.core.graph_schema import WorkflowGraph, Node, Edge
from supervisor.core.graph_engine import GraphOrchestrator
from supervisor.core.state import Database
from supervisor.core.engine import ExecutionEngine
from typing import List, Dict, Any, Optional
import asyncio
import json
import sqlite3
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="Supervisor Studio API",
    description="API for visual workflow management",
    version="1.0.0"
)

# CORS for local development - restricted to known origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security middleware for all requests
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
    origin = request.headers.get("origin")
    client_host = request.client.host if request.client else None

    if request.method in ["POST", "PUT", "DELETE"]:
        # State-changing requests require strict validation
        if origin:
            if origin not in ALLOWED_ORIGINS:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=403,
                    content={"detail": f"Origin '{origin}' not allowed"}
                )
        else:
            # No Origin - must be localhost
            if client_host not in ("127.0.0.1", "localhost", "::1"):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Origin header required for non-localhost requests"}
                )
    elif request.method == "GET":
        # Read-only requests: validate if Origin present, otherwise require localhost
        # This prevents data leakage if server is accidentally bound to external interface
        if origin:
            if origin not in ALLOWED_ORIGINS:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=403,
                    content={"detail": f"Origin '{origin}' not allowed"}
                )
        else:
            # No Origin - allow localhost only (blocks external data access)
            if client_host not in ("127.0.0.1", "localhost", "::1"):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Non-localhost access without Origin header not allowed"}
                )

    return await call_next(request)


# Global instances
db = Database()
engine = ExecutionEngine(Path("."))
orchestrator = GraphOrchestrator(db, engine, engine.gate_executor, engine.gate_loader)
```

### API Models

```python
class WorkflowCreateRequest(BaseModel):
    """Request to create a new workflow"""
    graph: WorkflowGraph

class WorkflowUpdateRequest(BaseModel):
    """Request to update an existing workflow"""
    graph: WorkflowGraph

class ExecutionRequest(BaseModel):
    """Request to execute a workflow"""
    graph_id: str                       # Required: ID of the graph definition to execute
    workflow_id: Optional[str] = None   # Optional: User-provided run label (defaults to graph_id)
    input_data: Dict[str, Any] = Field(default_factory=dict)

class ExecutionResponse(BaseModel):
    """Execution status response"""
    execution_id: str
    workflow_id: str   # User-provided run label (e.g., "nightly-run-1")
    graph_id: str      # Graph snapshot ID - references the workflow definition in graph_workflows
                       # at the time of execution. Use this to load the workflow graph for display.
                       # Note: This is a snapshot; editing the original workflow won't affect running executions.
    status: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

class NodeStatusResponse(BaseModel):
    """Node execution status"""
    node_id: str
    node_type: str
    status: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    version: int  # Use version for incremental updates
```

### Workflow CRUD Endpoints

**IMPORTANT:** Endpoints use `def` (not `async def`) because they perform blocking
synchronous database operations. FastAPI automatically runs `def` endpoints in a
thread pool, preventing event loop blocking.

```python
@app.get("/api/workflows", response_model=List[WorkflowGraph])
def list_workflows():
    """List all saved workflows (excludes execution snapshots)."""
    with db._connect() as conn:
        # Filter out execution snapshots (id starts with 'snapshot_')
        # Snapshots are immutable copies created for each execution
        rows = conn.execute("""
            SELECT definition FROM graph_workflows
            WHERE id NOT LIKE 'snapshot_%'
            ORDER BY updated_at DESC
        """).fetchall()

    return [WorkflowGraph.model_validate_json(row[0]) for row in rows]


@app.get("/api/workflows/{graph_id}", response_model=WorkflowGraph)
def get_workflow(graph_id: str):
    """Get a specific workflow by ID."""
    with db._connect() as conn:
        row = conn.execute(
            "SELECT definition FROM graph_workflows WHERE id = ?",
            (graph_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return WorkflowGraph.model_validate_json(row[0])


@app.post("/api/workflows", response_model=WorkflowGraph, status_code=201)
def create_workflow(request: WorkflowCreateRequest):
    """Create a new workflow."""
    # Prevent reserved prefix to avoid collision with execution snapshots
    if request.graph.id.startswith("snapshot_"):
        raise HTTPException(
            status_code=400,
            detail="Workflow ID cannot start with 'snapshot_' (reserved for execution snapshots)"
        )

    # Validate graph
    errors = request.graph.validate_graph()
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})

    # Save to database with conflict handling
    try:
        with db._connect() as conn:
            conn.execute("""
                INSERT INTO graph_workflows (id, name, definition, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                request.graph.id,
                request.graph.name,
                request.graph.model_dump_json(),
                request.graph.version,
                datetime.utcnow(),
                datetime.utcnow()
            ))
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail=f"Workflow with ID '{request.graph.id}' already exists"
        )

    return request.graph


@app.put("/api/workflows/{graph_id}", response_model=WorkflowGraph)
def update_workflow(graph_id: str, request: WorkflowUpdateRequest):
    """Update an existing workflow (snapshots are immutable and cannot be updated)."""
    if request.graph.id != graph_id:
        raise HTTPException(status_code=400, detail="Workflow ID mismatch")

    # Reject updates to execution snapshots (they are immutable)
    if graph_id.startswith("snapshot_"):
        raise HTTPException(
            status_code=403,
            detail="Cannot modify execution snapshots - they are immutable"
        )

    # Validate
    errors = request.graph.validate_graph()
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})

    # Update
    with db._connect() as conn:
        result = conn.execute("""
            UPDATE graph_workflows
            SET name = ?, definition = ?, version = ?, updated_at = ?
            WHERE id = ?
        """, (
            request.graph.name,
            request.graph.model_dump_json(),
            request.graph.version,
            datetime.utcnow(),
            graph_id
        ))

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Workflow not found")

    return request.graph


@app.delete("/api/workflows/{graph_id}")
def delete_workflow(graph_id: str):
    """Delete a workflow (snapshots are protected and cannot be deleted directly)."""
    # Reject deletion of execution snapshots (they should be cleaned up via execution deletion)
    if graph_id.startswith("snapshot_"):
        raise HTTPException(
            status_code=403,
            detail="Cannot delete execution snapshots directly - delete the execution instead"
        )

    with db._connect() as conn:
        result = conn.execute(
            "DELETE FROM graph_workflows WHERE id = ?",
            (graph_id,)
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Workflow not found")

    return {"status": "deleted", "id": graph_id}
```

### Execution Endpoints

```python
@app.post("/api/execute", response_model=ExecutionResponse)
async def execute_workflow(request: ExecutionRequest):
    """Start workflow execution."""
    from starlette.concurrency import run_in_threadpool

    # Load workflow by graph_id
    def load_workflow():
        with db._connect() as conn:
            row = conn.execute(
                "SELECT definition FROM graph_workflows WHERE id = ?",
                (request.graph_id,)
            ).fetchone()
        return row

    row = await run_in_threadpool(load_workflow)

    if not row:
        raise HTTPException(status_code=404, detail="Workflow graph not found")

    workflow = WorkflowGraph.model_validate_json(row[0])

    # workflow_id is the user-provided label; defaults to graph_id
    run_label = request.workflow_id or request.graph_id

    # Start execution
    execution_id = await orchestrator.start_workflow(
        workflow,
        run_label,
        initial_inputs=request.input_data
    )

    # IMPORTANT: The engine's start_workflow creates an immutable snapshot with ID
    # "snapshot_{execution_id}" in graph_workflows. This snapshot is referenced by
    # graph_executions.graph_id. This behavior already exists in GraphOrchestrator._save_workflow()
    # (see supervisor/core/graph_engine.py lines 160-178, 429-437).
    # DO NOT modify this - it ensures executions are not affected by later workflow edits.
    snapshot_graph_id = f"snapshot_{execution_id}"

    # Register status callback for WebSocket broadcasting
    # IMPORTANT: Callbacks are invoked from threadpool contexts (via asyncio.to_thread),
    # so we must use run_coroutine_threadsafe, NOT create_task
    loop = asyncio.get_running_loop()

    async def on_status_change(node_id: str, status: str, output: dict):
        await manager.broadcast(execution_id, {
            "type": "node_update",
            "node_id": node_id,
            "status": status,
            "output": output,
            "timestamp": datetime.utcnow().isoformat()
        })

    def sync_callback(node_id: str, status: str, output: dict):
        # Use run_coroutine_threadsafe since this is called from worker threads
        asyncio.run_coroutine_threadsafe(on_status_change(node_id, status, output), loop)

    orchestrator.register_status_callback(execution_id, sync_callback)

    # Start background worker
    asyncio.create_task(_run_execution_with_events(execution_id))

    # Read actual started_at from the database (not utcnow) for consistency
    def get_started_at():
        with db._connect() as conn:
            row = conn.execute(
                "SELECT started_at FROM graph_executions WHERE id = ?",
                (execution_id,)
            ).fetchone()
        return row[0] if row else datetime.utcnow().isoformat()

    started_at = await run_in_threadpool(get_started_at)

    return ExecutionResponse(
        execution_id=execution_id,
        workflow_id=run_label,
        graph_id=snapshot_graph_id,  # Return snapshot ID, not original graph_id
        status="running",
        started_at=started_at
    )


async def _run_execution_with_events(execution_id: str):
    """Background task to run workflow and broadcast completion."""
    from supervisor.core.worker import WorkflowWorker
    from starlette.concurrency import run_in_threadpool

    worker = WorkflowWorker(orchestrator)

    try:
        await worker.run_until_complete(execution_id)

        # Re-read actual status to avoid race with cancellation
        # (cancellation may have already broadcast "cancelled")
        def get_status():
            with db._connect() as conn:
                row = conn.execute(
                    "SELECT status FROM graph_executions WHERE id = ?",
                    (execution_id,)
                ).fetchone()
            return row[0] if row else None

        final_status = await run_in_threadpool(get_status)

        # Only broadcast if not already cancelled (cancellation broadcasts its own event)
        if final_status and final_status != "cancelled":
            await manager.broadcast(execution_id, {
                "type": "execution_complete",
                "status": final_status
            })
    except Exception as e:
        await manager.broadcast(execution_id, {
            "type": "execution_complete",
            "status": "failed",
            "error": str(e)
        })
    finally:
        orchestrator.unregister_status_callback(execution_id)


@app.post("/api/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """
    Cancel a running execution.

    NOTE: This sets the execution status to 'cancelled' but does NOT stop
    currently running tasks. Running tasks will complete in the background
    but their results will not advance the workflow.
    """
    from starlette.concurrency import run_in_threadpool

    result = await run_in_threadpool(orchestrator.cancel_execution, execution_id)

    if not result:
        # Check if execution exists
        def check_execution():
            with db._connect() as conn:
                row = conn.execute(
                    "SELECT status FROM graph_executions WHERE id = ?",
                    (execution_id,)
                ).fetchone()
            return row

        row = await run_in_threadpool(check_execution)

        if not row:
            raise HTTPException(status_code=404, detail="Execution not found")

        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel execution in '{row[0]}' state"
        )

    # Broadcast final node states to WebSocket clients
    # This ensures UI shows correct state (running nodes should appear as they were)
    def get_final_nodes():
        with db._connect() as conn:
            rows = conn.execute("""
                SELECT node_id, status, version FROM node_executions
                WHERE execution_id = ?
            """, (execution_id,)).fetchall()
        return [{"node_id": r[0], "status": r[1], "version": r[2]} for r in rows]

    final_nodes = await run_in_threadpool(get_final_nodes)

    await manager.broadcast(execution_id, {
        "type": "execution_complete",
        "status": "cancelled",
        "final_nodes": final_nodes  # Include final node states for UI update
    })

    return {"status": "cancelled", "execution_id": execution_id}


@app.get("/api/executions", response_model=List[ExecutionResponse])
def list_executions(
    workflow_id: Optional[str] = None,  # Filter by user-provided run label
    source_graph_id: Optional[str] = None,  # Filter by original graph ID (extracts from snapshot)
    limit: int = 50
):
    """
    List recent executions with optional filtering.

    Args:
        workflow_id: Filter by user-provided run label (exact match)
        source_graph_id: Filter by original graph ID (before snapshot). Matches executions
                        where the snapshot was created from this graph.
        limit: Maximum number of results (default 50)

    Note: graph_id in results is the snapshot ID (snapshot_{execution_id}), not the original.
    Use source_graph_id filter to find all runs of a particular workflow definition.
    """
    with db._connect() as conn:
        base_query = """
            SELECT id, workflow_id, graph_id, status, started_at, completed_at, error
            FROM graph_executions
        """
        conditions = []
        params = []

        if workflow_id:
            conditions.append("workflow_id = ?")
            params.append(workflow_id)

        if source_graph_id:
            # LIMITATION: source_graph_id filter uses workflow_id which defaults to graph_id
            # if not provided by the user. This works for the common case where users don't
            # override workflow_id. For better filtering, a future version could add a
            # source_graph_id column to graph_executions storing the original graph ID.
            #
            # Current behavior: Matches executions where workflow_id == source_graph_id
            # This is correct when: User runs workflow without specifying custom workflow_id
            # This may miss: Executions where user provided a custom workflow_id different from graph_id
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
            started_at=row[4],
            completed_at=row[5],
            error=row[6]
        )
        for row in rows
    ]


@app.get("/api/executions/{execution_id}", response_model=ExecutionResponse)
def get_execution(execution_id: str):
    """Get execution status."""
    with db._connect() as conn:
        row = conn.execute("""
            SELECT workflow_id, graph_id, status, started_at, completed_at, error
            FROM graph_executions
            WHERE id = ?
        """, (execution_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Execution not found")

    return ExecutionResponse(
        execution_id=execution_id,
        workflow_id=row[0],
        graph_id=row[1],
        status=row[2],
        started_at=row[3],
        completed_at=row[4],
        error=row[5]
    )


@app.get("/api/executions/{execution_id}/nodes")
def get_execution_nodes(execution_id: str, since_version: Optional[int] = None):
    """
    Get node statuses for an execution.

    Args:
        execution_id: The execution ID
        since_version: Optional - only return nodes with version > this value
    """
    with db._connect() as conn:
        if since_version is not None:
            rows = conn.execute("""
                SELECT node_id, node_type, status, output_data, error, version
                FROM node_executions
                WHERE execution_id = ? AND version > ?
                ORDER BY version DESC
            """, (execution_id, since_version)).fetchall()
        else:
            rows = conn.execute("""
                SELECT node_id, node_type, status, output_data, error, version
                FROM node_executions
                WHERE execution_id = ?
            """, (execution_id,)).fetchall()

    return [
        {
            "node_id": row[0],
            "node_type": row[1],
            "status": row[2],
            "output": json.loads(row[3]) if row[3] else None,
            "error": row[4],
            "version": row[5]
        }
        for row in rows
    ]
```

### WebSocket for Live Updates (Event-Driven)

```python
from starlette.concurrency import run_in_threadpool

class ConnectionManager:
    """
    Manage WebSocket connections with event-driven broadcasting.

    This uses a publish-subscribe model:
    - Clients connect and receive the initial state
    - The orchestrator's status callback broadcasts updates when node states change
    - All connected clients receive updates simultaneously
    """

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

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
    # Validate Origin/host for WebSocket connections (same protection as HTTP)
    origin = websocket.headers.get("origin")
    client_host = websocket.client.host if websocket.client else None

    if origin:
        # Browser request - validate Origin
        if origin not in ALLOWED_ORIGINS:
            await websocket.close(code=4003, reason="Origin not allowed")
            return
    else:
        # No Origin - only allow from localhost (CLI/scripts)
        if client_host not in ("127.0.0.1", "localhost", "::1"):
            await websocket.close(code=4003, reason="Non-localhost connections require Origin header")
            return

    await manager.connect(websocket, execution_id)

    try:
        # Send initial state - handle missing execution gracefully
        try:
            nodes = await run_in_threadpool(get_execution_nodes, execution_id, None)
            execution = await run_in_threadpool(get_execution, execution_id)
        except HTTPException as e:
            # Execution not found - close with specific reason
            await websocket.send_json({
                "type": "error",
                "code": e.status_code,
                "detail": e.detail
            })
            await websocket.close(code=4004, reason=str(e.detail))
            return

        await websocket.send_json({
            "type": "initial_state",
            "nodes": nodes,
            "status": execution.status
        })

        # If already complete, send completion and close
        if execution.status in ["completed", "failed", "cancelled"]:
            await websocket.send_json({
                "type": "execution_complete",
                "status": execution.status
            })
            return

        # Keep connection alive and handle client messages
        # Updates come via broadcast() from the orchestrator callback
        while True:
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30s heartbeat timeout
                )
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat to detect dead connections
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        import logging
        logging.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, execution_id)
```

### Static File Serving

```python
# Serve frontend build
frontend_dir = Path(__file__).parent / "frontend" / "dist"

if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(frontend_dir / "index.html")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """
        Serve SPA for client-side routing.

        SECURITY: Validates resolved path is within frontend_dir to prevent
        path traversal attacks.
        """
        requested_path = (frontend_dir / full_path).resolve()
        if not requested_path.is_relative_to(frontend_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")

        if requested_path.exists() and requested_path.is_file():
            return FileResponse(requested_path)
        return FileResponse(frontend_dir / "index.html")
```

---

## 3.6 Frontend Application

**Directory:** `supervisor/studio/frontend/`

### Project Structure

```
frontend/
├── package.json
├── vite.config.ts
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/
│   │   └── client.ts          # API client
│   ├── hooks/
│   │   ├── useWorkflows.ts    # React Query hooks
│   │   └── useExecution.ts
│   ├── components/
│   │   ├── WorkflowCanvas.tsx # ReactFlow canvas
│   │   ├── NodePalette.tsx    # Drag-and-drop node palette
│   │   ├── NodeConfigPanel.tsx # Node configuration sidebar
│   │   ├── ExecutionMonitor.tsx # Live execution view
│   │   └── nodes/
│   │       ├── TaskNode.tsx
│   │       ├── GateNode.tsx
│   │       ├── BranchNode.tsx
│   │       ├── MergeNode.tsx
│   │       └── ParallelNode.tsx
│   ├── pages/
│   │   ├── WorkflowList.tsx
│   │   ├── WorkflowEditor.tsx
│   │   └── ExecutionView.tsx
│   └── types/
│       └── workflow.ts        # TypeScript types
```

### Vite Configuration (Dev Proxy)

**File:** `vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API and WebSocket requests to the backend during development
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
});
```

### Dependencies (package.json)

```json
{
  "name": "supervisor-studio",
  "version": "1.0.0",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "reactflow": "^11.10.0",
    "@tanstack/react-query": "^5.0.0",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-tabs": "^1.0.4",
    "lucide-react": "^0.292.0",
    "zustand": "^4.4.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0"
  }
}
```

### API Client

**File:** `src/api/client.ts`

```typescript
import { WorkflowGraph, Execution, NodeExecution } from '../types/workflow';

const API_BASE = '/api';

export const apiClient = {
  // Workflows
  async getWorkflows(): Promise<WorkflowGraph[]> {
    const response = await fetch(`${API_BASE}/workflows`);
    if (!response.ok) throw new Error('Failed to fetch workflows');
    return response.json();
  },

  async getWorkflow(id: string): Promise<WorkflowGraph> {
    const response = await fetch(`${API_BASE}/workflows/${id}`);
    if (!response.ok) throw new Error('Failed to fetch workflow');
    return response.json();
  },

  async createWorkflow(workflow: WorkflowGraph): Promise<WorkflowGraph> {
    const response = await fetch(`${API_BASE}/workflows`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: workflow }),
    });
    if (!response.ok) {
      const error = await response.json();
      if (response.status === 409) {
        throw new Error(`Workflow ID '${workflow.id}' already exists`);
      }
      throw new Error(error.detail?.validation_errors?.join(', ') || 'Failed to create');
    }
    return response.json();
  },

  async updateWorkflow(id: string, workflow: WorkflowGraph): Promise<WorkflowGraph> {
    const response = await fetch(`${API_BASE}/workflows/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: workflow }),
    });
    if (!response.ok) throw new Error('Failed to update workflow');
    return response.json();
  },

  async deleteWorkflow(id: string): Promise<void> {
    const response = await fetch(`${API_BASE}/workflows/${id}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete workflow');
  },

  // Executions
  async executeWorkflow(
    graphId: string,
    inputData: Record<string, unknown> = {},
    workflowId?: string
  ): Promise<Execution> {
    const response = await fetch(`${API_BASE}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        graph_id: graphId,
        workflow_id: workflowId,
        input_data: inputData
      }),
    });
    if (!response.ok) throw new Error('Failed to execute workflow');
    return response.json();
  },

  async cancelExecution(executionId: string): Promise<void> {
    const response = await fetch(`${API_BASE}/executions/${executionId}/cancel`, {
      method: 'POST',
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to cancel execution');
    }
  },

  async getExecution(id: string): Promise<Execution> {
    const response = await fetch(`${API_BASE}/executions/${id}`);
    if (!response.ok) throw new Error('Failed to fetch execution');
    return response.json();
  },

  async getExecutionNodes(id: string, sinceVersion?: number): Promise<NodeExecution[]> {
    const url = sinceVersion !== undefined
      ? `${API_BASE}/executions/${id}/nodes?since_version=${sinceVersion}`
      : `${API_BASE}/executions/${id}/nodes`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch nodes');
    return response.json();
  },
};
```

### React Query Hooks

**File:** `src/hooks/useWorkflows.ts`

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { WorkflowGraph } from '../types/workflow';
import { useCallback, useEffect, useRef } from 'react';

export function useWorkflows() {
  return useQuery({
    queryKey: ['workflows'],
    queryFn: () => apiClient.getWorkflows(),
  });
}

export function useWorkflow(id: string) {
  return useQuery({
    queryKey: ['workflows', id],
    queryFn: () => apiClient.getWorkflow(id),
    enabled: !!id,
  });
}

export function useCreateWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (workflow: WorkflowGraph) => apiClient.createWorkflow(workflow),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workflows'] });
    },
  });
}

export function useUpdateWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, workflow }: { id: string; workflow: WorkflowGraph }) =>
      apiClient.updateWorkflow(id, workflow),

    onMutate: async ({ id, workflow }) => {
      await queryClient.cancelQueries({ queryKey: ['workflows', id] });
      const previousWorkflow = queryClient.getQueryData<WorkflowGraph>(['workflows', id]);
      queryClient.setQueryData(['workflows', id], workflow);
      return { previousWorkflow };
    },

    onError: (err, { id }, context) => {
      if (context?.previousWorkflow) {
        queryClient.setQueryData(['workflows', id], context.previousWorkflow);
      }
    },

    onSettled: (data, error, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['workflows', id] });
    },
  });
}

export function useDeleteWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => apiClient.deleteWorkflow(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workflows'] });
    },
  });
}

export function useDebouncedWorkflowUpdate() {
  const updateMutation = useUpdateWorkflow();
  const mutationRef = useRef(updateMutation);
  mutationRef.current = updateMutation;
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

  const debouncedUpdate = useCallback((id: string, workflow: WorkflowGraph) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = setTimeout(() => {
      mutationRef.current.mutate({ id, workflow });
    }, 500);
  }, []);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return debouncedUpdate;
}
```

### Workflow Canvas (ReactFlow)

**File:** `src/components/WorkflowCanvas.tsx`

```typescript
import React, { useCallback, useEffect, useState, useRef } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
  Connection,
  Node,
  Edge,
  NodeTypes,
  OnNodesChange,
  OnConnect,
} from 'reactflow';
import 'reactflow/dist/style.css';

import TaskNode from './nodes/TaskNode';
import GateNode from './nodes/GateNode';
import BranchNode from './nodes/BranchNode';
import MergeNode from './nodes/MergeNode';
import ParallelNode from './nodes/ParallelNode';
import HumanNode from './nodes/HumanNode';
import { useWorkflow, useDebouncedWorkflowUpdate } from '../hooks/useWorkflows';
import { WorkflowGraph, Node as WorkflowNode } from '../types/workflow';

const nodeTypes: NodeTypes = {
  task: TaskNode,
  gate: GateNode,
  branch: BranchNode,
  merge: MergeNode,
  parallel: ParallelNode,
  human: HumanNode,
};

interface Props {
  workflowId: string;
  onNodeSelect?: (node: WorkflowNode | null) => void;
  onEdgeSelect?: (edge: import('../types/workflow').Edge | null) => void;
  executionStatuses?: Record<string, string>;
  readOnly?: boolean;  // If true, disables all editing (for execution monitoring)
}

function WorkflowCanvasInner({ workflowId, onNodeSelect, onEdgeSelect, executionStatuses, readOnly }: Props) {
  const { data: workflow, isLoading } = useWorkflow(workflowId);
  const debouncedUpdate = useDebouncedWorkflowUpdate();
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();
  const isInitialized = useRef(false);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Initialize nodes/edges when workflow loads
  useEffect(() => {
    if (workflow && !isInitialized.current) {
      const initialNodes = workflow.nodes.map(node => ({
        id: node.id,
        type: node.type,
        position: node.ui_metadata?.position || { x: 0, y: 0 },
        data: {
          label: node.label || node.id,
          config: node.task_config || node.gate_config || node.branch_config || {},
          status: executionStatuses?.[node.id],
        },
      }));

      const initialEdges = workflow.edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.condition
          ? `${edge.condition.field} ${edge.condition.operator} ${edge.condition.value}`
          : undefined,
        animated: !!edge.condition,
        style: edge.is_loop_edge ? { stroke: '#f59e0b' } : undefined,
      }));

      setNodes(initialNodes);
      setEdges(initialEdges);
      isInitialized.current = true;
    }
  }, [workflow, setNodes, setEdges, executionStatuses]);

  useEffect(() => {
    isInitialized.current = false;
  }, [workflowId]);

  // Granular status update - only update changed nodes
  useEffect(() => {
    if (!executionStatuses) return;

    setNodes(currentNodes =>
      currentNodes.map(node => {
        const newStatus = executionStatuses[node.id];
        if (node.data.status !== newStatus) {
          return {
            ...node,
            data: { ...node.data, status: newStatus }
          };
        }
        return node;
      })
    );
  }, [executionStatuses, setNodes]);

  const handleNodesChange: OnNodesChange = useCallback((changes) => {
    // In read-only mode, only allow selection changes, not structural changes
    if (readOnly) {
      const selectChanges = changes.filter(c => c.type === 'select');
      if (selectChanges.length > 0) {
        onNodesChange(selectChanges);
      }
      return;
    }

    onNodesChange(changes);

    if (!workflow) return;

    // Handle node deletions
    const removeChanges = changes.filter(c => c.type === 'remove');
    if (removeChanges.length > 0) {
      const removedIds = new Set(removeChanges.map(c => c.id));
      const updatedNodes = workflow.nodes.filter(n => !removedIds.has(n.id));
      // Also remove edges connected to deleted nodes
      const updatedEdges = workflow.edges.filter(
        e => !removedIds.has(e.source) && !removedIds.has(e.target)
      );
      debouncedUpdate(workflow.id, { ...workflow, nodes: updatedNodes, edges: updatedEdges });
      return;
    }

    // Handle position changes
    const positionChanges = changes.filter(c => c.type === 'position' && c.position);
    if (positionChanges.length > 0) {
      const updatedNodes = workflow.nodes.map(node => {
        const change = positionChanges.find(c => c.id === node.id);
        if (change && change.type === 'position' && change.position) {
          return {
            ...node,
            ui_metadata: { ...node.ui_metadata, position: change.position }
          };
        }
        return node;
      });

      debouncedUpdate(workflow.id, { ...workflow, nodes: updatedNodes });
    }
  }, [workflow, onNodesChange, debouncedUpdate, readOnly]);

  const handleEdgesChange = useCallback((changes: import('reactflow').EdgeChange[]) => {
    // In read-only mode, only allow selection changes
    if (readOnly) {
      const selectChanges = changes.filter(c => c.type === 'select');
      if (selectChanges.length > 0) {
        onEdgesChange(selectChanges);
      }
      return;
    }

    onEdgesChange(changes);

    if (!workflow) return;

    // Handle edge deletions
    const removeChanges = changes.filter(c => c.type === 'remove');
    if (removeChanges.length > 0) {
      const removedIds = new Set(removeChanges.map(c => c.id));
      const updatedEdges = workflow.edges.filter(e => !removedIds.has(e.id));
      debouncedUpdate(workflow.id, { ...workflow, edges: updatedEdges });
    }
  }, [workflow, onEdgesChange, debouncedUpdate, readOnly]);

  const onConnect: OnConnect = useCallback((connection: Connection) => {
    if (!workflow || !connection.source || !connection.target) return;

    const newEdge = {
      id: crypto.randomUUID(),
      source: connection.source,
      target: connection.target,
    };

    setEdges(eds => addEdge({ ...connection, id: newEdge.id }, eds));

    const updatedWorkflow = {
      ...workflow,
      edges: [...workflow.edges, newEdge]
    };
    debouncedUpdate(workflow.id, updatedWorkflow);
  }, [workflow, setEdges, debouncedUpdate]);

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id);
    setSelectedEdge(null);
    const workflowNode = workflow?.nodes.find(n => n.id === node.id);
    onNodeSelect?.(workflowNode || null);
    onEdgeSelect?.(null);
  }, [workflow, onNodeSelect, onEdgeSelect]);

  const onEdgeClick = useCallback((event: React.MouseEvent, edge: Edge) => {
    setSelectedEdge(edge.id);
    setSelectedNode(null);
    const workflowEdge = workflow?.edges.find(e => e.id === edge.id);
    onEdgeSelect?.(workflowEdge || null);
    onNodeSelect?.(null);
  }, [workflow, onNodeSelect, onEdgeSelect]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    if (!workflow) return;

    const nodeType = event.dataTransfer.getData('application/reactflow');
    if (!nodeType) return;

    const position = screenToFlowPosition({
      x: event.clientX,
      y: event.clientY,
    });

    const newNodeId = crypto.randomUUID();
    const newNode: WorkflowNode = {
      id: newNodeId,
      type: nodeType as WorkflowNode['type'],
      label: `New ${nodeType}`,
      ui_metadata: { position },
      ...(nodeType === 'task' && { task_config: { role: 'implementer', task_template: '{input}' } }),
      ...(nodeType === 'gate' && { gate_config: { gate_type: 'test_gate' } }),
      ...(nodeType === 'branch' && { branch_config: { condition: { field: 'status', operator: '==', value: 'success', max_iterations: 10 }, on_true: '', on_false: '' } }),
      ...(nodeType === 'merge' && { merge_config: { wait_for: 'all', merge_strategy: 'union' } }),
      ...(nodeType === 'parallel' && { parallel_config: { branches: [], wait_for: 'all' } }),
      ...(nodeType === 'human' && { human_config: { title: 'Approval Required' } }),
    };

    setNodes(nodes => [...nodes, {
      id: newNodeId,
      type: nodeType,
      position,
      data: {
        label: newNode.label,
        config: newNode.task_config || newNode.gate_config || newNode.branch_config || {},
      },
    }]);

    const updatedWorkflow = {
      ...workflow,
      nodes: [...workflow.nodes, newNode]
    };
    debouncedUpdate(workflow.id, updatedWorkflow);
  }, [workflow, screenToFlowPosition, debouncedUpdate, setNodes]);

  if (isLoading) {
    return <div className="flex items-center justify-center h-full">Loading...</div>;
  }

  return (
    <div
      className="w-full h-full"
      ref={reactFlowWrapper}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        onConnect={readOnly ? undefined : onConnect}
        onNodeClick={onNodeClick}
        onEdgeClick={onEdgeClick}
        nodeTypes={nodeTypes}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
        nodesConnectable={!readOnly}
        nodesDraggable={!readOnly}
        elementsSelectable={true}
        deleteKeyCode={readOnly ? null : 'Delete'}
      >
        <Background gap={15} />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
}

export default function WorkflowCanvas(props: Props) {
  return (
    <ReactFlowProvider>
      <WorkflowCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
```

### Custom Node Components

**File:** `src/components/nodes/TaskNode.tsx`

```typescript
import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Play } from 'lucide-react';

interface TaskNodeData {
  label: string;
  config: {
    role: string;
  };
  status?: string;
}

function TaskNode({ data, selected }: NodeProps<TaskNodeData>) {
  const statusColors: Record<string, string> = {
    completed: 'border-green-500 bg-green-50',
    running: 'border-blue-500 bg-blue-50 animate-pulse',
    failed: 'border-red-500 bg-red-50',
    ready: 'border-yellow-500 bg-yellow-50',
    pending: 'border-gray-300 bg-gray-50',
    skipped: 'border-gray-400 bg-gray-100',
  };

  const borderColor = statusColors[data.status || 'pending'] || statusColors.pending;

  return (
    <div className={`
      px-4 py-2 rounded-lg border-2 shadow-sm min-w-[150px]
      ${borderColor}
      ${selected ? 'ring-2 ring-blue-400' : ''}
    `}>
      <Handle type="target" position={Position.Top} className="w-3 h-3" />

      <div className="flex items-center gap-2">
        <Play className="w-4 h-4 text-cyan-600" />
        <div>
          <div className="font-medium text-sm">{data.label}</div>
          <div className="text-xs text-gray-500">role: {data.config.role}</div>
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </div>
  );
}

export default memo(TaskNode);
```

### Execution Monitor Component

**File:** `src/components/ExecutionMonitor.tsx`

```typescript
import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import WorkflowCanvas from './WorkflowCanvas';
import { apiClient } from '../api/client';

interface NodeStatus {
  node_id: string;
  status: string;
  output?: unknown;
}

function buildWebSocketUrl(path: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
}

export default function ExecutionMonitor() {
  const { executionId } = useParams<{ executionId: string }>();
  const [nodeStatuses, setNodeStatuses] = useState<Record<string, string>>({});
  const [graphId, setGraphId] = useState<string | null>(null);
  const [executionStatus, setExecutionStatus] = useState<string>('running');
  const wsRef = useRef<WebSocket | null>(null);

  const handleCancel = useCallback(async () => {
    if (!executionId) return;
    try {
      await apiClient.cancelExecution(executionId);
    } catch (e) {
      console.error('Failed to cancel:', e);
    }
  }, [executionId]);

  useEffect(() => {
    if (!executionId) return;

    apiClient.getExecution(executionId).then(exec => {
      setGraphId(exec.graph_id);
      setExecutionStatus(exec.status);
    });

    const wsUrl = buildWebSocketUrl(`/ws/executions/${executionId}`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'initial_state') {
        const statuses: Record<string, string> = {};
        data.nodes.forEach((node: NodeStatus) => {
          statuses[node.node_id] = node.status;
        });
        setNodeStatuses(statuses);
        if (data.status) {
          setExecutionStatus(data.status);
        }
      }

      if (data.type === 'node_update') {
        setNodeStatuses(prev => ({
          ...prev,
          [data.node_id]: data.status
        }));
      }

      if (data.type === 'execution_complete') {
        setExecutionStatus(data.status);
        // Update node statuses with final state (especially for cancellation)
        if (data.final_nodes) {
          const finalStatuses: Record<string, string> = {};
          data.final_nodes.forEach((node: { node_id: string; status: string }) => {
            finalStatuses[node.node_id] = node.status;
          });
          setNodeStatuses(finalStatuses);
        }
      }
    };

    return () => ws.close();
  }, [executionId]);

  if (!graphId) {
    return <div>Loading...</div>;
  }

  const isComplete = ['completed', 'failed', 'cancelled'].includes(executionStatus);

  return (
    <div className="h-screen flex flex-col">
      <header className="bg-white border-b px-4 py-2 flex items-center justify-between">
        <h1 className="text-lg font-semibold">
          Execution: {executionId?.slice(0, 8)}...
        </h1>
        <div className="flex items-center gap-3">
          {!isComplete && (
            <button
              onClick={handleCancel}
              className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200"
            >
              Cancel
            </button>
          )}
          <span className={`px-2 py-1 rounded text-sm ${
            executionStatus === 'completed' ? 'bg-green-100 text-green-800' :
            executionStatus === 'failed' ? 'bg-red-100 text-red-800' :
            executionStatus === 'cancelled' ? 'bg-gray-100 text-gray-800' :
            'bg-blue-100 text-blue-800'
          }`}>
            {executionStatus.charAt(0).toUpperCase() + executionStatus.slice(1)}
          </span>
        </div>
      </header>

      <div className="flex-1">
        <WorkflowCanvas
          workflowId={graphId}
          executionStatuses={nodeStatuses}
          readOnly={true}  // Execution views are always read-only (snapshots are immutable)
        />
      </div>
    </div>
  );
}
```

---

## 3.7 CLI Integration

**File:** `supervisor/cli.py` (Extend)

```python
@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1 for security)")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def studio(host: str, port: int):
    """Start Supervisor Studio web interface.

    SECURITY WARNING: This server has no authentication. By default it binds to
    127.0.0.1 (localhost only). Do not expose to the network without adding
    authentication - workflows can execute arbitrary code.
    """
    import uvicorn
    from supervisor.studio.server import app

    # Security warning for non-localhost binding
    if host not in ("127.0.0.1", "localhost"):
        click.secho(
            "\n WARNING: Binding to non-localhost address without authentication!",
            fg="yellow", bold=True
        )
        click.secho(
            "   This server allows arbitrary code execution via workflows.",
            fg="yellow"
        )
        click.secho(
            "   Only proceed if you understand the security implications.\n",
            fg="yellow"
        )
        if not click.confirm("Continue anyway?"):
            raise SystemExit(1)

    click.echo(f"Starting Supervisor Studio on http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")

    uvicorn.run(app, host=host, port=port)
```

---

## 3.8 Build & Deployment

### Build Frontend

```bash
cd supervisor/studio/frontend
npm install
npm run build
```

### Run Studio (Development)

```bash
# Terminal 1: Start backend
supervisor studio --port 8000

# Terminal 2: Start frontend dev server (with proxy)
cd supervisor/studio/frontend
npm run dev
```

### Run Studio (Production)

```bash
# Build frontend first
cd supervisor/studio/frontend && npm run build

# Start server (serves built frontend)
supervisor studio --port 8000
```

Then open http://localhost:8000 (production) or http://localhost:5173 (dev) in your browser.

---

## 3.9 Feature Summary

| Feature | Component | Description |
|---------|-----------|-------------|
| Workflow List | `WorkflowList.tsx` | View/create/delete workflows |
| Visual Editor | `WorkflowCanvas.tsx` | Drag-drop graph editing with ReactFlow |
| Node Palette | `NodePalette.tsx` | Drag new nodes onto canvas |
| Node Config | `NodeConfigPanel.tsx` | Configure selected node |
| Edge Config | `EdgeConfigPanel.tsx` | Configure edges (loop edges) |
| Live Monitor | `ExecutionMonitor.tsx` | Real-time execution visualization |
| Cancel Execution | API + UI | Stop running workflows (status only) |
| WebSocket | `server.py` | Event-driven push updates via orchestrator callback |
| REST API | `server.py` | CRUD for workflows, executions |

---

## 3.10 Implementation Checklist

Before implementing Phase 3, complete these prerequisites:

**Core Engine Modifications (`supervisor/core/graph_engine.py`):**

- [ ] Add `_status_callbacks` dict to `GraphOrchestrator.__init__`
- [ ] Add `register_status_callback(execution_id, callback)` method
- [ ] Add `unregister_status_callback(execution_id)` method
- [ ] Add `cancel_execution(execution_id)` method (sets status to 'cancelled')

**Callback Invocation Points (add callback after each transaction commits):**

- [ ] `_update_dependents()` - invoke callback when node becomes READY
- [ ] `claim_next_node()` - invoke callback when node transitions to RUNNING
- [ ] `_execute_node()` - invoke callback on COMPLETED or FAILED
- [ ] `_skip_downstream_nodes()` - invoke callback when node is SKIPPED

**Database Verification:**

- [ ] Verify `graph_workflows` table exists
- [ ] Verify `graph_executions` table exists
- [ ] Verify `node_executions` table exists with `version` column

**Security Verification:**

- [ ] Server binds to `127.0.0.1` by default
- [ ] CLI warns when binding to non-localhost address

---

**End of Phase 3 Plan**
