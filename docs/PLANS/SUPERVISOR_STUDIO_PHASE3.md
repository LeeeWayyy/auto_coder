# Supervisor Studio - Phase 3: Web UI Studio

**Status:** Planning
**Objective:** Build a web-based visual editor for designing, executing, and monitoring workflows.

**Prerequisites:** Phase 1 (Engine) and Phase 2 (CLI Visualization) completed.

---

## 3.1 Overview

Phase 3 delivers a full web-based Supervisor Studio with:
- Visual drag-and-drop workflow editor (ReactFlow)
- Real-time execution monitoring (WebSockets)
- Workflow management (CRUD operations)
- Node configuration panels
- Execution history and analytics

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
│        ▼                    ▼                      ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  ReactFlow   │    │  Redis       │    │   SQLite             │  │
│  │  @tanstack   │    │  (Pub/Sub)   │    │   (State Store)      │  │
│  │  react-query │    │              │    │                      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

> **Note:** The MVP implementation uses an in-memory `ConnectionManager` for WebSocket
> pub/sub. Redis can be added later for horizontal scaling of backend servers.

---

## 3.3 Backend API Server

**File:** `supervisor/studio/server.py` (New)

### FastAPI Application

```python
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
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
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="Supervisor Studio API",
    description="API for visual workflow management",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    graph_id: str                      # Required: ID of the graph definition to execute
    workflow_id: Optional[str] = None  # Optional: User-provided run label (defaults to graph_id)
    # Use default_factory to avoid mutable default sharing across requests
    input_data: Dict[str, Any] = Field(default_factory=dict)  # Initial input for entry node

class ExecutionResponse(BaseModel):
    """Execution status response"""
    execution_id: str
    workflow_id: str  # User-provided run label (e.g., "nightly-run-1")
    graph_id: str     # Graph definition ID - use this to load the workflow graph
    status: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

class NodeStatusUpdate(BaseModel):
    """WebSocket message for node status update"""
    type: str = "node_update"
    node_id: str
    status: str
    output: Optional[Dict[str, Any]] = None
    timestamp: str
```

### Workflow CRUD Endpoints

**IMPORTANT:** Endpoints use `def` (not `async def`) because they perform blocking
synchronous database operations. FastAPI automatically runs `def` endpoints in a
thread pool, preventing event loop blocking. Using `async def` with sync DB calls
would freeze the server.

```python
@app.get("/api/workflows", response_model=List[WorkflowGraph])
def list_workflows():
    """List all saved workflows.

    NOTE: Using `def` (not async def) because DB operations are synchronous.
    FastAPI runs sync endpoints in a thread pool automatically.
    """
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT definition FROM graph_workflows
            ORDER BY updated_at DESC
        """).fetchall()

    return [WorkflowGraph.model_validate_json(row[0]) for row in rows]


@app.get("/api/workflows/{workflow_id}", response_model=WorkflowGraph)
def get_workflow(workflow_id: str):
    """Get a specific workflow by ID."""
    with db._connect() as conn:
        row = conn.execute(
            "SELECT definition FROM graph_workflows WHERE id = ?",
            (workflow_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return WorkflowGraph.model_validate_json(row[0])


@app.post("/api/workflows", response_model=WorkflowGraph)
def create_workflow(request: WorkflowCreateRequest):
    """Create a new workflow."""
    # Validate graph
    errors = request.graph.validate_graph()
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})

    # Save to database
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

    return request.graph


@app.put("/api/workflows/{workflow_id}", response_model=WorkflowGraph)
def update_workflow(workflow_id: str, request: WorkflowUpdateRequest):
    """Update an existing workflow."""
    if request.graph.id != workflow_id:
        raise HTTPException(status_code=400, detail="Workflow ID mismatch")

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
            workflow_id
        ))

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Workflow not found")

    return request.graph


@app.delete("/api/workflows/{workflow_id}")
def delete_workflow(workflow_id: str):
    """Delete a workflow."""
    with db._connect() as conn:
        result = conn.execute(
            "DELETE FROM graph_workflows WHERE id = ?",
            (workflow_id,)
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Workflow not found")

    return {"status": "deleted", "id": workflow_id}
```

### Execution Endpoints

```python
@app.post("/api/execute", response_model=ExecutionResponse)
async def execute_workflow(request: ExecutionRequest):
    """
    Start workflow execution.

    NOTE: This endpoint is async because it starts a background task.
    The initial DB read is wrapped in run_in_threadpool.
    """
    from starlette.concurrency import run_in_threadpool

    # Load workflow by graph_id (schema definition ID)
    def load_workflow():
        with db._connect() as conn:
            row = conn.execute(
                "SELECT definition FROM graph_workflows WHERE id = ?",
                (request.graph_id,)  # Use graph_id to load the definition
            ).fetchone()
        return row

    row = await run_in_threadpool(load_workflow)

    if not row:
        raise HTTPException(status_code=404, detail="Workflow graph not found")

    workflow = WorkflowGraph.model_validate_json(row[0])

    # workflow_id is the run label; defaults to graph_id if not provided
    run_label = request.workflow_id or request.graph_id

    # Start execution - pass initial input_data for entry node
    execution_id = await orchestrator.start_workflow(
        workflow,
        run_label,                        # User's run label
        initial_inputs=request.input_data # Seed entry node with input
    )

    # Start background worker
    asyncio.create_task(_run_execution(execution_id))

    return ExecutionResponse(
        execution_id=execution_id,
        workflow_id=run_label,           # User's run label
        graph_id=request.graph_id,       # Graph definition ID for loading
        status="running",
        started_at=datetime.utcnow().isoformat()
    )


async def _run_execution(execution_id: str):
    """Background task to run workflow execution"""
    from supervisor.core.worker import WorkflowWorker
    worker = WorkflowWorker(orchestrator)
    await worker.run_until_complete(execution_id)


@app.get("/api/executions", response_model=List[ExecutionResponse])
def list_executions(workflow_id: Optional[str] = None, limit: int = 50):
    """List recent executions."""
    with db._connect() as conn:
        if workflow_id:
            rows = conn.execute("""
                SELECT id, workflow_id, graph_id, status, started_at, completed_at, error
                FROM graph_executions
                WHERE workflow_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (workflow_id, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, workflow_id, graph_id, status, started_at, completed_at, error
                FROM graph_executions
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

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
def get_execution_nodes(execution_id: str):
    """Get all node statuses for an execution."""
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT node_id, node_type, status, output_data, error
            FROM node_executions
            WHERE execution_id = ?
        """, (execution_id,)).fetchall()

    return [
        {
            "node_id": row[0],
            "node_type": row[1],
            "status": row[2],
            "output": json.loads(row[3]) if row[3] else None,
            "error": row[4]
        }
        for row in rows
    ]
```

### WebSocket for Live Updates

```python
from starlette.concurrency import run_in_threadpool

class ConnectionManager:
    """Manage WebSocket connections."""

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
                pass  # Already removed

    async def broadcast(self, execution_id: str, message: dict):
        """Broadcast to all connections concurrently."""
        if execution_id not in self.active_connections:
            return

        connections = self.active_connections[execution_id]
        if not connections:
            return

        # Use asyncio.gather for concurrent sends (don't let slow client block others)
        async def safe_send(conn: WebSocket):
            try:
                await conn.send_json(message)
            except Exception:
                # Connection likely closed
                self.disconnect(conn, execution_id)

        await asyncio.gather(*[safe_send(c) for c in connections], return_exceptions=True)


manager = ConnectionManager()


@app.websocket("/ws/executions/{execution_id}")
async def execution_websocket(websocket: WebSocket, execution_id: str):
    """
    WebSocket for live execution updates.

    IMPORTANT: Properly handles client disconnection by:
    1. Using asyncio.wait_for with timeout on receive
    2. Catching WebSocketDisconnect for graceful cleanup
    3. Running DB queries in thread pool to avoid blocking
    """
    await manager.connect(websocket, execution_id)

    try:
        # Send initial state (run DB query in thread pool)
        nodes = await run_in_threadpool(get_execution_nodes, execution_id)
        await websocket.send_json({
            "type": "initial_state",
            "nodes": nodes
        })

        # Main loop: poll for updates and check for client messages
        while True:
            # Non-blocking check for client disconnect/ping
            # This ensures we detect disconnection promptly
            try:
                # Wait for message with short timeout (allows update cycle to continue)
                msg = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.5
                )
                # Handle ping/pong or other client messages
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # No message received - normal, continue update loop
                pass

            # Check execution status (run in thread pool)
            status = await run_in_threadpool(
                orchestrator.get_execution_status, execution_id
            )

            if status in ["completed", "failed", "cancelled"]:
                await websocket.send_json({
                    "type": "execution_complete",
                    "status": status
                })
                break

            # Get latest node statuses (run in thread pool)
            nodes = await run_in_threadpool(get_execution_nodes, execution_id)
            await websocket.send_json({
                "type": "status_update",
                "nodes": nodes
            })

    except WebSocketDisconnect:
        # Client disconnected gracefully
        pass
    except Exception as e:
        # Log error but don't crash
        import logging
        logging.error(f"WebSocket error: {e}")
    finally:
        # Always cleanup connection
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
        path traversal attacks (e.g., GET /../../etc/passwd).
        """
        # Resolve to absolute path and validate containment
        requested_path = (frontend_dir / full_path).resolve()
        if not requested_path.is_relative_to(frontend_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")

        if requested_path.exists() and requested_path.is_file():
            return FileResponse(requested_path)
        return FileResponse(frontend_dir / "index.html")
```

---

## 3.4 Frontend Application

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
    "zustand": "^4.4.0",
    "lodash": "^4.17.21"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@types/lodash": "^4.14.0",
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
    inputData: any = {},
    workflowId?: string  // Optional run label
  ): Promise<Execution> {
    const response = await fetch(`${API_BASE}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        graph_id: graphId,           // Required: schema definition ID
        workflow_id: workflowId,     // Optional: user-provided run label
        input_data: inputData
      }),
    });
    if (!response.ok) throw new Error('Failed to execute workflow');
    return response.json();
  },

  async getExecution(id: string): Promise<Execution> {
    const response = await fetch(`${API_BASE}/executions/${id}`);
    if (!response.ok) throw new Error('Failed to fetch execution');
    return response.json();
  },

  async getExecutionNodes(id: string): Promise<NodeExecution[]> {
    const response = await fetch(`${API_BASE}/executions/${id}/nodes`);
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
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { debounce } from 'lodash';

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

    // Optimistic update for instant UI feedback
    onMutate: async ({ id, workflow }) => {
      await queryClient.cancelQueries({ queryKey: ['workflows', id] });
      const previousWorkflow = queryClient.getQueryData<WorkflowGraph>(['workflows', id]);
      queryClient.setQueryData(['workflows', id], workflow);
      return { previousWorkflow };
    },

    onError: (err, { id }, context) => {
      // Rollback on error
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

// Debounced update for drag operations
export function useDebouncedWorkflowUpdate() {
  const updateMutation = useUpdateWorkflow();
  // Use ref to hold latest mutation to avoid stale closures
  const mutationRef = useRef(updateMutation);
  mutationRef.current = updateMutation;

  // Create stable debounced function
  const debouncedUpdate = useMemo(
    () =>
      debounce((id: string, workflow: WorkflowGraph) => {
        mutationRef.current.mutate({ id, workflow });
      }, 500),
    [] // Empty deps - function is stable, uses ref for latest mutation
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      debouncedUpdate.cancel();
    };
  }, [debouncedUpdate]);

  return debouncedUpdate;
}
```

### Workflow Canvas (ReactFlow)

**File:** `src/components/WorkflowCanvas.tsx`

```typescript
import React, { useCallback, useEffect, useMemo, useState, useRef } from 'react';
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
  OnEdgesChange,
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
}

function WorkflowCanvasInner({ workflowId, onNodeSelect, onEdgeSelect, executionStatuses }: Props) {
  const { data: workflow, isLoading } = useWorkflow(workflowId);
  const debouncedUpdate = useDebouncedWorkflowUpdate();
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  // Convert workflow to ReactFlow format
  const initialNodes = useMemo(() => {
    if (!workflow) return [];
    return workflow.nodes.map(node => ({
      id: node.id,
      type: node.type,
      position: node.ui_metadata?.position || { x: 0, y: 0 },
      data: {
        label: node.label || node.id,
        config: node.task_config || node.gate_config || node.branch_config || {},
        status: executionStatuses?.[node.id],
      },
      selected: node.id === selectedNode,
    }));
  }, [workflow, executionStatuses, selectedNode]);

  const initialEdges = useMemo(() => {
    if (!workflow) return [];
    return workflow.edges.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.condition
        ? `${edge.condition.field} ${edge.condition.operator} ${edge.condition.value}`
        : undefined,
      animated: !!edge.condition,
      style: edge.is_loop_edge ? { stroke: '#f59e0b' } : undefined,
    }));
  }, [workflow]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // CRITICAL: Sync props to ReactFlow state when workflow or execution status changes
  // useNodesState/useEdgesState only use initial values on first render;
  // subsequent prop changes must be synced explicitly via useEffect
  useEffect(() => {
    setNodes(initialNodes);
  }, [initialNodes, setNodes]);

  useEffect(() => {
    setEdges(initialEdges);
  }, [initialEdges, setEdges]);

  // Handle node position changes (debounced)
  const handleNodesChange: OnNodesChange = useCallback((changes) => {
    onNodesChange(changes);

    // Debounce position updates to server
    if (!workflow) return;

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
  }, [workflow, onNodesChange, debouncedUpdate]);

  // Handle new edge connections
  const onConnect: OnConnect = useCallback((connection: Connection) => {
    if (!workflow || !connection.source || !connection.target) return;

    const newEdge = {
      id: `e-${connection.source}-${connection.target}`,
      source: connection.source,
      target: connection.target,
    };

    setEdges(eds => addEdge(connection, eds));

    // Save to server
    const updatedWorkflow = {
      ...workflow,
      edges: [...workflow.edges, newEdge]
    };
    debouncedUpdate(workflow.id, updatedWorkflow);
  }, [workflow, setEdges, debouncedUpdate]);

  // Handle node selection
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id);
    setSelectedEdge(null);  // Deselect edge when selecting node
    const workflowNode = workflow?.nodes.find(n => n.id === node.id);
    onNodeSelect?.(workflowNode || null);
    onEdgeSelect?.(null);
  }, [workflow, onNodeSelect, onEdgeSelect]);

  // Handle edge selection (for configuring loop edges)
  const onEdgeClick = useCallback((event: React.MouseEvent, edge: Edge) => {
    setSelectedEdge(edge.id);
    setSelectedNode(null);  // Deselect node when selecting edge
    const workflowEdge = workflow?.edges.find(e => e.id === edge.id);
    onEdgeSelect?.(workflowEdge || null);
    onNodeSelect?.(null);
  }, [workflow, onNodeSelect, onEdgeSelect]);

  // Drag-and-drop handlers for adding new nodes from NodePalette
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    if (!workflow) return;

    const nodeType = event.dataTransfer.getData('application/reactflow');
    if (!nodeType) return;

    // Calculate drop position in flow coordinates
    const position = screenToFlowPosition({
      x: event.clientX,
      y: event.clientY,
    });

    // Create new node with unique ID
    const newNodeId = `${nodeType}_${Date.now()}`;
    const newNode: WorkflowNode = {
      id: newNodeId,
      type: nodeType as any,
      label: `New ${nodeType}`,
      ui_metadata: { position },
      // Add default config based on type
      ...(nodeType === 'task' && { task_config: { role: 'implementer', task_template: '{input}' } }),
      ...(nodeType === 'gate' && { gate_config: { gate_type: 'test_gate' } }),
      ...(nodeType === 'branch' && { branch_config: { condition: { field: 'status', operator: '==', value: 'success', max_iterations: 10 }, on_true: null, on_false: null } }),
      ...(nodeType === 'merge' && { merge_config: { wait_for: 'all', merge_strategy: 'union' } }),
      ...(nodeType === 'parallel' && { parallel_config: { branches: [], wait_for: 'all' } }),
      ...(nodeType === 'human' && { human_config: { title: 'Approval Required' } }),
    };

    // Update workflow with new node
    const updatedWorkflow = {
      ...workflow,
      nodes: [...workflow.nodes, newNode]
    };
    debouncedUpdate(workflow.id, updatedWorkflow);
  }, [workflow, screenToFlowPosition, debouncedUpdate]);

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
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onEdgeClick={onEdgeClick}
        nodeTypes={nodeTypes}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
      >
        <Background gap={15} />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
}

// Wrap with ReactFlowProvider for useReactFlow hook
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

**File:** `src/components/nodes/BranchNode.tsx`

```typescript
import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { GitBranch } from 'lucide-react';

interface BranchNodeData {
  label: string;
  config: {
    condition?: {
      field: string;
      operator: string;
      value: any;
    };
  };
  status?: string;
}

function BranchNode({ data, selected }: NodeProps<BranchNodeData>) {
  return (
    <div className={`
      px-4 py-2 rounded-lg border-2 border-purple-500 bg-purple-50 shadow-sm
      ${selected ? 'ring-2 ring-blue-400' : ''}
    `}>
      <Handle type="target" position={Position.Top} className="w-3 h-3" />

      <div className="flex items-center gap-2">
        <GitBranch className="w-4 h-4 text-purple-600" />
        <div>
          <div className="font-medium text-sm">{data.label}</div>
          {data.config.condition && (
            <div className="text-xs text-gray-500">
              {data.config.condition.field} {data.config.condition.operator} {data.config.condition.value}
            </div>
          )}
        </div>
      </div>

      {/* Two output handles for true/false */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="true"
        className="w-3 h-3 left-1/4"
        style={{ left: '25%' }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="false"
        className="w-3 h-3 left-3/4"
        style={{ left: '75%' }}
      />
    </div>
  );
}

export default memo(BranchNode);
```

### Execution Monitor Component

**File:** `src/components/ExecutionMonitor.tsx`

```typescript
import React, { useEffect, useState, useRef } from 'react';
import { useParams } from 'react-router-dom';
import WorkflowCanvas from './WorkflowCanvas';
import { apiClient } from '../api/client';

interface NodeStatus {
  node_id: string;
  status: string;
  output?: any;
}

/**
 * Build WebSocket URL from current location.
 * Handles both http/https and uses correct protocol (ws/wss).
 */
function buildWebSocketUrl(path: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
}

export default function ExecutionMonitor() {
  const { executionId } = useParams<{ executionId: string }>();
  const [nodeStatuses, setNodeStatuses] = useState<Record<string, string>>({});
  const [workflowId, setWorkflowId] = useState<string | null>(null);
  const [isComplete, setIsComplete] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!executionId) return;

    // Get initial execution info - use graph_id to load workflow definition
    apiClient.getExecution(executionId).then(exec => {
      // Use graph_id (not workflow_id) to load the graph definition
      // workflow_id is the user's run label, graph_id is the schema ID
      setWorkflowId(exec.graph_id);
    });

    // Connect to WebSocket for live updates
    // Use dynamic URL based on current location (supports https/wss)
    const wsUrl = buildWebSocketUrl(`/ws/executions/${executionId}`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'initial_state' || data.type === 'status_update') {
        const statuses: Record<string, string> = {};
        data.nodes.forEach((node: NodeStatus) => {
          statuses[node.node_id] = node.status;
        });
        setNodeStatuses(statuses);
      }

      if (data.type === 'execution_complete') {
        setIsComplete(true);
      }
    };

    return () => ws.close();
  }, [executionId]);

  if (!workflowId) {
    return <div>Loading...</div>;
  }

  return (
    <div className="h-screen flex flex-col">
      <header className="bg-white border-b px-4 py-2 flex items-center justify-between">
        <h1 className="text-lg font-semibold">
          Execution: {executionId?.slice(0, 8)}...
        </h1>
        <span className={`px-2 py-1 rounded text-sm ${
          isComplete ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'
        }`}>
          {isComplete ? 'Completed' : 'Running'}
        </span>
      </header>

      <div className="flex-1">
        <WorkflowCanvas
          workflowId={workflowId}
          executionStatuses={nodeStatuses}
        />
      </div>
    </div>
  );
}
```

---

## 3.5 Node Palette (Drag & Drop)

**File:** `src/components/NodePalette.tsx`

```typescript
import React from 'react';
import { Play, Shield, GitBranch, Merge, Split, Folder, User } from 'lucide-react';

// IMPORTANT: Use static class strings for Tailwind CSS.
// Dynamic template strings like `border-${color}-300` get purged by Tailwind
// because the compiler can't detect them at build time.
const nodeTypes = [
  {
    type: 'task',
    label: 'Task',
    icon: Play,
    containerClass: 'border-cyan-300 bg-cyan-50',
    iconClass: 'text-cyan-600'
  },
  {
    type: 'gate',
    label: 'Gate',
    icon: Shield,
    containerClass: 'border-yellow-300 bg-yellow-50',
    iconClass: 'text-yellow-600'
  },
  {
    type: 'branch',
    label: 'Branch',
    icon: GitBranch,
    containerClass: 'border-purple-300 bg-purple-50',
    iconClass: 'text-purple-600'
  },
  {
    type: 'merge',
    label: 'Merge',
    icon: Merge,
    containerClass: 'border-blue-300 bg-blue-50',
    iconClass: 'text-blue-600'
  },
  {
    type: 'parallel',
    label: 'Parallel',
    icon: Split,
    containerClass: 'border-green-300 bg-green-50',
    iconClass: 'text-green-600'
  },
  {
    type: 'subgraph',
    label: 'Subgraph',
    icon: Folder,
    containerClass: 'border-gray-300 bg-gray-50',
    iconClass: 'text-gray-600'
  },
  {
    type: 'human',
    label: 'Human',
    icon: User,
    containerClass: 'border-red-300 bg-red-50',
    iconClass: 'text-red-600'
  },
];

interface Props {
  onDragStart: (type: string) => void;
}

export default function NodePalette({ onDragStart }: Props) {
  return (
    <div className="w-48 bg-white border-r p-4">
      <h3 className="font-semibold text-sm text-gray-600 mb-3">Node Types</h3>
      <div className="space-y-2">
        {nodeTypes.map(({ type, label, icon: Icon, containerClass, iconClass }) => (
          <div
            key={type}
            className={`flex items-center gap-2 p-2 rounded border cursor-grab hover:bg-gray-50 active:cursor-grabbing ${containerClass}`}
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/reactflow', type);
              onDragStart(type);
            }}
          >
            <Icon className={`w-4 h-4 ${iconClass}`} />
            <span className="text-sm">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## 3.6 Node Configuration Panel

**File:** `src/components/NodeConfigPanel.tsx`

```typescript
import React from 'react';
import { Node } from '../types/workflow';
import { X } from 'lucide-react';

interface Props {
  node: Node;
  onUpdate: (updates: Partial<Node>) => void;
  onClose: () => void;
}

export default function NodeConfigPanel({ node, onUpdate, onClose }: Props) {
  return (
    <div className="w-80 bg-white border-l p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Configure Node</h3>
        <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Common fields */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">ID</label>
          {/* ID is read-only after creation to prevent broken graph references.
              Changing ID would require updating all edges, entry/exit points,
              branch configs (on_true/on_false), and data mappings. */}
          <input
            type="text"
            value={node.id}
            disabled
            className="mt-1 w-full border rounded px-3 py-2 text-sm bg-gray-100 text-gray-500 cursor-not-allowed"
          />
          <p className="mt-1 text-xs text-gray-500">ID cannot be changed after creation</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Label</label>
          <input
            type="text"
            value={node.label || ''}
            onChange={(e) => onUpdate({ label: e.target.value })}
            className="mt-1 w-full border rounded px-3 py-2 text-sm"
          />
        </div>

        {/* Type-specific fields */}
        {node.type === 'task' && node.task_config && (
          <TaskConfigFields
            config={node.task_config}
            onChange={(config) => onUpdate({ task_config: config })}
          />
        )}

        {node.type === 'gate' && node.gate_config && (
          <GateConfigFields
            config={node.gate_config}
            onChange={(config) => onUpdate({ gate_config: config })}
          />
        )}

        {node.type === 'branch' && node.branch_config && (
          <BranchConfigFields
            config={node.branch_config}
            onChange={(config) => onUpdate({ branch_config: config })}
          />
        )}
      </div>
    </div>
  );
}

function TaskConfigFields({ config, onChange }: { config: any; onChange: (c: any) => void }) {
  return (
    <>
      <div>
        <label className="block text-sm font-medium text-gray-700">Role</label>
        <select
          value={config.role}
          onChange={(e) => onChange({ ...config, role: e.target.value })}
          className="mt-1 w-full border rounded px-3 py-2 text-sm"
        >
          <option value="planner">Planner</option>
          <option value="implementer">Implementer</option>
          <option value="reviewer">Reviewer</option>
          <option value="debugger">Debugger</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Task Template</label>
        <textarea
          value={config.task_template || '{input}'}
          onChange={(e) => onChange({ ...config, task_template: e.target.value })}
          className="mt-1 w-full border rounded px-3 py-2 text-sm"
          rows={3}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Gates</label>
        <input
          type="text"
          value={config.gates?.join(', ') || ''}
          onChange={(e) => onChange({
            ...config,
            gates: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
          })}
          placeholder="test_gate, lint_gate"
          className="mt-1 w-full border rounded px-3 py-2 text-sm"
        />
      </div>
    </>
  );
}

function GateConfigFields({ config, onChange }: { config: any; onChange: (c: any) => void }) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700">Gate Type</label>
      <select
        value={config.gate_type}
        onChange={(e) => onChange({ ...config, gate_type: e.target.value })}
        className="mt-1 w-full border rounded px-3 py-2 text-sm"
      >
        <option value="test_gate">Test Gate</option>
        <option value="lint_gate">Lint Gate</option>
        <option value="security_gate">Security Gate</option>
        <option value="build_gate">Build Gate</option>
      </select>
    </div>
  );
}

function BranchConfigFields({ config, onChange }: { config: any; onChange: (c: any) => void }) {
  return (
    <>
      <div>
        <label className="block text-sm font-medium text-gray-700">Condition Field</label>
        <input
          type="text"
          value={config.condition?.field || ''}
          onChange={(e) => onChange({
            ...config,
            condition: { ...config.condition, field: e.target.value }
          })}
          className="mt-1 w-full border rounded px-3 py-2 text-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Operator</label>
        <select
          value={config.condition?.operator || '=='}
          onChange={(e) => onChange({
            ...config,
            condition: { ...config.condition, operator: e.target.value }
          })}
          className="mt-1 w-full border rounded px-3 py-2 text-sm"
        >
          <option value="==">equals (==)</option>
          <option value="!=">not equals (!=)</option>
          <option value=">">greater than (&gt;)</option>
          <option value="<">less than (&lt;)</option>
          <option value="in">in list</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Value</label>
        <input
          type="text"
          value={config.condition?.value || ''}
          onChange={(e) => onChange({
            ...config,
            condition: { ...config.condition, value: e.target.value }
          })}
          className="mt-1 w-full border rounded px-3 py-2 text-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Max Iterations</label>
        <input
          type="number"
          value={config.condition?.max_iterations || 10}
          onChange={(e) => onChange({
            ...config,
            condition: { ...config.condition, max_iterations: parseInt(e.target.value) }
          })}
          className="mt-1 w-full border rounded px-3 py-2 text-sm"
        />
      </div>
    </>
  );
}
```

### Edge Configuration Panel

**File:** `src/components/EdgeConfigPanel.tsx`

```typescript
import React from 'react';
import { Edge } from '../types/workflow';
import { X, RotateCcw } from 'lucide-react';

interface Props {
  edge: Edge;
  onUpdate: (updates: Partial<Edge>) => void;
  onClose: () => void;
}

/**
 * Panel for configuring edge properties, especially is_loop_edge.
 * Loop edges are critical for cycle control - without them, cycles
 * will fail validation with "Cycle without loop control" error.
 */
export default function EdgeConfigPanel({ edge, onUpdate, onClose }: Props) {
  return (
    <div className="w-80 bg-white border-l p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Configure Edge</h3>
        <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded">
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="space-y-4">
        {/* Edge info (read-only) */}
        <div className="bg-gray-50 rounded p-3 text-sm">
          <div><span className="font-medium">From:</span> {edge.source}</div>
          <div><span className="font-medium">To:</span> {edge.target}</div>
        </div>

        {/* Loop Edge Toggle - CRITICAL for cycle control */}
        <div className="border rounded p-3">
          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={edge.is_loop_edge || false}
              onChange={(e) => onUpdate({ is_loop_edge: e.target.checked })}
              className="mt-1 w-4 h-4"
            />
            <div>
              <div className="flex items-center gap-2 font-medium">
                <RotateCcw className="w-4 h-4 text-orange-500" />
                Loop Edge
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Mark this edge as a loop edge. Loop edges allow cycles in the
                workflow by resetting completed nodes to ready state. Required
                for retry/iteration patterns.
              </p>
            </div>
          </label>
        </div>

        {/* Condition display (if present) */}
        {edge.condition && (
          <div>
            <label className="block text-sm font-medium text-gray-700">Condition</label>
            <div className="mt-1 bg-gray-50 rounded p-2 text-sm font-mono">
              {edge.condition.field} {edge.condition.operator} {JSON.stringify(edge.condition.value)}
            </div>
          </div>
        )}
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
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def studio(host: str, port: int):
    """Start Supervisor Studio web interface"""
    import uvicorn
    from supervisor.studio.server import app

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

### Run Studio

```bash
supervisor studio --port 8000
```

Then open http://localhost:8000 in your browser.

---

## 3.9 Feature Summary

| Feature | Component | Description |
|---------|-----------|-------------|
| Workflow List | `WorkflowList.tsx` | View/create/delete workflows |
| Visual Editor | `WorkflowCanvas.tsx` | Drag-drop graph editing with ReactFlow |
| Node Palette | `NodePalette.tsx` | Drag new nodes onto canvas |
| Node Config | `NodeConfigPanel.tsx` | Configure selected node |
| Live Monitor | `ExecutionMonitor.tsx` | Real-time execution visualization |
| WebSocket | `server.py` | Push updates to frontend |
| REST API | `server.py` | CRUD for workflows, executions |

---

**End of Phase 3 Plan**
