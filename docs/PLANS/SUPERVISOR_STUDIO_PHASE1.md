# Supervisor Studio - Phase 1: Declarative Engine Foundation

**Status:** Planning
**Objective:** Enable workflows to be defined in YAML and executed by a stateless, resumable graph engine.

---

## 1.1 Graph Schema Definition

**File:** `supervisor/core/graph_schema.py` (New)

**Purpose:** Define Pydantic models for declarative graph workflows with security-first design.

### Node Types

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Literal, Union, Set
from enum import Enum
import networkx as nx

class NodeType(str, Enum):
    """Supported node types in workflow graphs"""
    TASK = "task"              # Execute a role (planner, implementer, reviewer, etc.)
    GATE = "gate"              # Verification gate (test, lint, security, etc.)
    BRANCH = "branch"          # Conditional branching with loop support
    MERGE = "merge"            # Synchronization point for parallel branches
    PARALLEL = "parallel"      # Split into parallel execution branches
    SUBGRAPH = "subgraph"      # Nested workflow execution
    HUMAN = "human"            # Human approval/intervention point

class NodeStatus(str, Enum):
    """Execution status for nodes"""
    PENDING = "pending"        # Not yet ready (dependencies not met)
    READY = "ready"            # Ready to execute (all dependencies satisfied)
    RUNNING = "running"        # Currently executing
    COMPLETED = "completed"    # Successfully completed
    FAILED = "failed"          # Execution failed
    SKIPPED = "skipped"        # Skipped due to branch conditions

class BranchOutcome(str, Enum):
    """Possible outcomes from BRANCH nodes"""
    TRUE = "on_true"
    FALSE = "on_false"
    MAX_ITERATIONS = "max_iterations_reached"
```

### Condition Models (Security-First)

```python
class TransitionCondition(BaseModel):
    """
    Safe, declarative condition evaluation for edges.
    NO arbitrary code execution - only structured operators.
    """
    field: str                 # Supports dotted notation: "source_node.field_name"
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "starts_with", "ends_with"]
    value: Union[str, int, float, bool, List[str]]

    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        """Ensure field names are safe (alphanumeric + underscore + dots)"""
        if not v.replace('_', '').replace('.', '').isalnum():
            raise ValueError(f"Invalid field name: {v}")
        return v

class LoopCondition(BaseModel):
    """
    Safe loop condition for BRANCH nodes.
    Replaces unsafe string evaluation with structured conditions.
    """
    field: str                 # Field to evaluate from previous node output
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in", "not_in"]
    value: Union[str, int, float, bool, List[str]]
    max_iterations: int = 10   # CRITICAL: Prevent infinite loops
```

### Edge Model

```python
class Edge(BaseModel):
    """Directed edge between nodes with optional conditions and data mapping"""
    id: str
    source: str                # Source node ID
    target: str                # Target node ID
    condition: Optional[TransitionCondition] = None  # For conditional edges
    data_mapping: Optional[Dict[str, str]] = None    # Map outputs to inputs
    is_loop_edge: bool = False  # Marks edges that form loops (for re-execution)
```

### Node Configuration Models

```python
class TaskNodeConfig(BaseModel):
    """Configuration for TASK nodes - execute a role"""
    role: str                  # Role name (planner, implementer, reviewer, debugger, etc.)
    task_template: str = "{input}"  # Template for task description formatting
    timeout: Optional[int] = None   # Execution timeout in seconds
    gates: List[str] = []      # Gates to run after task (pre-apply safety)
    worktree_id: Optional[str] = None  # Optional worktree for isolation

class GateNodeConfig(BaseModel):
    """Configuration for GATE nodes - verification gates"""
    gate_type: str             # Gate name from gates.yaml (test_gate, lint_gate, etc.)
    auto_approve: bool = False # Skip human approval if gate passes
    requires_worktree: bool = True  # Whether gate needs worktree context

class BranchNodeConfig(BaseModel):
    """
    Configuration for BRANCH nodes.
    Uses declarative LoopCondition instead of unsafe string evaluation.
    """
    condition: LoopCondition   # Safe, structured condition
    on_true: str               # Target node ID if condition is true
    on_false: str              # Target node ID if condition is false

class MergeNodeConfig(BaseModel):
    """
    Configuration for MERGE nodes - synchronization points.
    """
    wait_for: Literal["all", "any"] = "all"  # Wait for all/any incoming edges
    merge_strategy: Literal["union", "intersection", "first"] = "union"
    # union: merge all inputs (later overwrites earlier)
    # intersection: only common keys
    # first: first completed input wins

class ParallelNodeConfig(BaseModel):
    """Configuration for PARALLEL nodes - fan-out execution"""
    branches: List[str]        # List of node IDs to execute in parallel
    wait_for: Literal["all", "any", "first"] = "all"

class SubgraphNodeConfig(BaseModel):
    """Configuration for SUBGRAPH nodes - nested workflows"""
    workflow_name: str         # Name of nested workflow to execute
    input_mapping: Dict[str, str] = {}   # Map parent inputs to child inputs
    output_mapping: Dict[str, str] = {}  # Map child outputs to parent outputs

class HumanNodeConfig(BaseModel):
    """Configuration for HUMAN nodes - approval points"""
    title: str                 # Title shown in approval UI
    description: Optional[str] = None  # Additional context
```

### Node Model

```python
class Node(BaseModel):
    """Generic graph node with type-specific configuration"""
    id: str
    type: NodeType
    label: Optional[str] = None
    description: Optional[str] = None

    # Type-specific configuration (only one should be set based on type)
    task_config: Optional[TaskNodeConfig] = None
    gate_config: Optional[GateNodeConfig] = None
    branch_config: Optional[BranchNodeConfig] = None
    merge_config: Optional[MergeNodeConfig] = None
    parallel_config: Optional[ParallelNodeConfig] = None
    subgraph_config: Optional[SubgraphNodeConfig] = None
    human_config: Optional[HumanNodeConfig] = None

    # UI metadata (position, styling) for visual editor
    ui_metadata: Optional[Dict] = None

    # Readiness policy for nodes with multiple incoming edges
    wait_for_incoming: Literal["all", "any"] = "any"
```

### WorkflowGraph Model

```python
class WorkflowGraph(BaseModel):
    """Complete workflow definition"""
    id: str
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"

    nodes: List[Node]
    edges: List[Edge]

    entry_point: str           # Starting node ID
    exit_points: List[str] = []  # Terminal node IDs (auto-detected if empty)

    # Global configuration
    config: Dict = Field(default_factory=lambda: {
        "max_parallel_nodes": 4,      # Limit concurrent execution
        "fail_fast": True             # Stop on first failure
    })

    def validate_graph(self) -> List[str]:
        """
        Validate graph structure using NetworkX.
        Returns list of validation errors.
        """
        errors = []

        # Check for duplicate node IDs (critical - would corrupt execution state)
        seen_node_ids = set()
        for node in self.nodes:
            if node.id in seen_node_ids:
                errors.append(f"Duplicate node ID: '{node.id}'")
            seen_node_ids.add(node.id)
        node_ids = seen_node_ids

        # Check for duplicate edge IDs
        seen_edge_ids = set()
        for edge in self.edges:
            if edge.id in seen_edge_ids:
                errors.append(f"Duplicate edge ID: '{edge.id}'")
            seen_edge_ids.add(edge.id)

        # Check for duplicate edges (same source->target)
        seen_edge_pairs = set()
        for edge in self.edges:
            pair = (edge.source, edge.target)
            if pair in seen_edge_pairs:
                errors.append(f"Duplicate edge from '{edge.source}' to '{edge.target}'")
            seen_edge_pairs.add(pair)

        # Check entry point exists
        if self.entry_point not in node_ids:
            errors.append(f"Entry point '{self.entry_point}' not found")

        # Check all edge endpoints exist
        for edge in self.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge {edge.id}: source '{edge.source}' not found")
            if edge.target not in node_ids:
                errors.append(f"Edge {edge.id}: target '{edge.target}' not found")

        # Build NetworkX graph for advanced validation
        G = self._to_networkx()

        # Auto-detect exit points if not specified
        if not self.exit_points:
            self.exit_points = [n for n in node_ids if G.out_degree(n) == 0]
            if not self.exit_points:
                errors.append("No exit points found (no terminal nodes)")

        # Check for cycles without loop control
        for cycle in nx.simple_cycles(G):
            has_loop_control = any(
                n.type == NodeType.BRANCH and n.branch_config and
                n.branch_config.condition.max_iterations > 0 and
                any(e.is_loop_edge for e in self.edges if e.source in cycle)
                for n in self.nodes if n.id in cycle
            )
            if not has_loop_control:
                errors.append(f"Cycle without loop control: {' -> '.join(cycle)}")

        # Validate MERGE nodes have multiple incoming edges
        for node in self.nodes:
            if node.type == NodeType.MERGE:
                incoming = [e for e in self.edges if e.target == node.id]
                if len(incoming) < 2:
                    errors.append(f"MERGE node '{node.id}' should have at least 2 incoming edges")

        # Warn about SUBGRAPH nodes (not yet implemented)
        # This prevents users from authoring workflows that will fail at runtime
        for node in self.nodes:
            if node.type == NodeType.SUBGRAPH:
                errors.append(
                    f"SUBGRAPH node '{node.id}' is not yet implemented. "
                    f"Remove this node or use a different workflow pattern."
                )

        return errors

    def _to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph for analysis"""
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.id)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target)
        return G

    def get_terminal_nodes(self) -> Set[str]:
        """Find nodes with no outgoing edges"""
        G = self._to_networkx()
        return {n for n in G.nodes() if G.out_degree(n) == 0}

    def analyze_parallelism(self) -> List[List[str]]:
        """Find nodes that can execute in parallel (topological levels)"""
        G = self._to_networkx()
        return [list(level) for level in nx.topological_generations(G)]

    def find_critical_path(self) -> List[str]:
        """Find longest path through graph"""
        G = self._to_networkx()
        try:
            return nx.dag_longest_path(G)
        except nx.NetworkXError:
            return []  # Has cycles
```

---

## 1.2 Stateless Graph Execution Engine

**File:** `supervisor/core/graph_engine.py` (New)

**Purpose:** Implement a stateless, resumable graph orchestrator.

### Design Principles

1. **Database as Source of Truth**: All state stored in DB, not in-memory
2. **Transaction-Based Execution**: Each node execution is atomic
3. **Crash-Resistant**: Can resume from any point after failure
4. **Worker-Based**: Supports horizontal scaling via multiple workers
5. **Pre-Apply Safety**: Gates run BEFORE changes are applied to main repo
6. **Async/Sync Hygiene**: SQLite is synchronous; blocking calls (DB, approvals) are
   wrapped in `asyncio.to_thread()` for non-blocking behavior in async context

### GraphOrchestrator Class

```python
from supervisor.core.graph_schema import *
from supervisor.core.state import Database
from supervisor.core.engine import ExecutionEngine
from supervisor.core.gate_executor import GateExecutor
from supervisor.core.gate_loader import GateLoader
from supervisor.core.gate_models import GateStatus
from supervisor.core.approval import ApprovalGate
from supervisor.core.interaction import ApprovalDecision
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GraphOrchestrator:
    """
    Stateless workflow graph orchestrator.

    Key Features:
    - No long-running execution loops
    - Each execute_next_batch() call is independent
    - Can resume after crash by querying DB state
    - Supports horizontal scaling via multiple workers
    """

    def __init__(self, db: Database, execution_engine: ExecutionEngine,
                 gate_executor: GateExecutor, gate_loader: GateLoader,
                 max_parallel: int = 4):
        self.db = db
        self.engine = execution_engine
        self.gate_executor = gate_executor
        self.gate_loader = gate_loader
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.approval_gate = ApprovalGate(db)
```

### Database Transaction Helper

```python
    def _transaction(self):
        """
        Synchronous transaction context manager.

        NOTE: SQLite operations are synchronous. In async methods, we wrap
        entire transaction blocks in asyncio.to_thread() when needed for
        non-blocking behavior. For short DB operations, blocking is acceptable.
        """
        return self.db._connect()

    def _run_in_transaction(self, func, *args):
        """Execute a function within a DB transaction."""
        with self.db._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = func(conn, *args)
                conn.commit()
                return result
            except:
                conn.rollback()
                raise

    def _save_workflow(self, conn, workflow: WorkflowGraph):
        """Save workflow definition to database."""
        conn.execute("""
            INSERT OR REPLACE INTO graph_workflows (id, name, definition, version, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            workflow.id,
            workflow.name,
            workflow.model_dump_json(),
            workflow.version
        ))

    def _create_execution(self, conn, execution_id: str, workflow_id: str, graph_id: str):
        """Create a new execution record."""
        conn.execute("""
            INSERT INTO graph_executions (id, workflow_id, graph_id, status, started_at)
            VALUES (?, ?, ?, 'running', CURRENT_TIMESTAMP)
        """, (execution_id, workflow_id, graph_id))

    def _set_node_status(self, conn, execution_id: str, node_id: str,
                        status: NodeStatus, node_type: str = None, error: str = None):
        """Set node execution status."""
        if node_type:
            # Initial creation with type
            conn.execute("""
                INSERT INTO node_executions (id, execution_id, node_id, node_type, status)
                VALUES (?, ?, ?, ?, ?)
            """, (f"{execution_id}_{node_id}", execution_id, node_id, node_type, status.value))
        else:
            # Status update
            conn.execute("""
                UPDATE node_executions SET status=?, error=?
                WHERE execution_id=? AND node_id=?
            """, (status.value, error, execution_id, node_id))

    def _set_node_output(self, conn, execution_id: str, node_id: str, output: Any):
        """Store node output data."""
        conn.execute("""
            UPDATE node_executions SET output_data=?, completed_at=CURRENT_TIMESTAMP
            WHERE execution_id=? AND node_id=?
        """, (json.dumps(output), execution_id, node_id))

    def _get_node_output(self, execution_id: str, node_id: str) -> Any:
        """Get node output from database."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT output_data FROM node_executions WHERE execution_id=? AND node_id=?",
                (execution_id, node_id)
            ).fetchone()
            return json.loads(row[0]) if row and row[0] else None

    def _get_node_status_single(self, execution_id: str, node_id: str) -> str:
        """Get status of a single node."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT status FROM node_executions WHERE execution_id=? AND node_id=?",
                (execution_id, node_id)
            ).fetchone()
            return row[0] if row else NodeStatus.PENDING.value

    def _clear_node_output(self, conn, execution_id: str, node_id: str):
        """Clear node output for loop re-execution."""
        conn.execute("""
            UPDATE node_executions SET output_data=NULL, completed_at=NULL
            WHERE execution_id=? AND node_id=?
        """, (execution_id, node_id))

    def _get_workflow(self, execution_id: str) -> WorkflowGraph:
        """Load workflow definition for an execution."""
        with self.db._connect() as conn:
            row = conn.execute("""
                SELECT w.definition FROM graph_workflows w
                JOIN graph_executions e ON w.id = e.graph_id
                WHERE e.id = ?
            """, (execution_id,)).fetchone()
            return WorkflowGraph.model_validate_json(row[0]) if row else None
```

### Workflow Lifecycle Methods

```python
    async def start_workflow(self, workflow: WorkflowGraph, workflow_id: str,
                            initial_inputs: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new workflow execution.

        Args:
            workflow: The workflow graph definition
            workflow_id: User-provided workflow identifier (run label)
            initial_inputs: Optional initial data for the entry point node

        Returns:
            execution_id: Unique ID for this execution instance
        """
        # Validate graph
        errors = workflow.validate_graph()
        if errors:
            raise ValueError(f"Invalid workflow graph: {errors}")

        import uuid
        execution_id = str(uuid.uuid4())

        # Persist workflow and initialize node states atomically
        # NOTE: Using sync transaction - short DB ops are acceptable to block
        def init_workflow(conn, exec_id, wf_id, wf, inputs):
            self._save_workflow(conn, wf)
            self._create_execution(conn, exec_id, wf_id, wf.id)
            for node in wf.nodes:
                self._set_node_status(conn, exec_id, node.id,
                                     NodeStatus.PENDING, node.type.value)
            self._set_node_status(conn, exec_id, wf.entry_point, NodeStatus.READY)

            # Seed entry point with initial inputs (if provided)
            # This allows the entry node to receive external data
            if inputs:
                conn.execute(
                    "UPDATE node_executions SET input_data=? WHERE execution_id=? AND node_id=?",
                    (json.dumps(inputs), exec_id, wf.entry_point)
                )

        self._run_in_transaction(init_workflow, execution_id, workflow_id, workflow, initial_inputs)

        return execution_id

    async def execute_next_batch(self, execution_id: str) -> int:
        """
        Execute one batch of READY nodes.

        This is called repeatedly by a scheduler/worker.
        Each call is stateless - queries DB for current state.

        Returns:
            Number of nodes successfully executed
        """
        # Check if execution is already terminal
        status = self._get_execution_status(execution_id)
        if status in ["completed", "failed", "cancelled"]:
            return 0

        # Load workflow definition from DB
        workflow = self._get_workflow(execution_id)

        # Claim READY nodes atomically (prevents duplicate execution)
        ready_nodes = await self._claim_ready_nodes(
            execution_id,
            workflow.config.get("max_parallel_nodes", 4)
        )

        if not ready_nodes:
            # No ready nodes - check if workflow should complete
            await self._check_workflow_state(execution_id, workflow)
            return 0

        # Execute batch in parallel
        results = await asyncio.gather(
            *[self._execute_node(execution_id, workflow, nid) for nid in ready_nodes],
            return_exceptions=True
        )

        return sum(1 for r in results if not isinstance(r, Exception))
```

### Node Claiming (Prevents Duplicate Execution)

```python
    async def _claim_ready_nodes(self, execution_id: str, limit: int) -> List[str]:
        """
        Atomically claim READY nodes to prevent duplicate execution.
        Uses optimistic locking via version field.

        NOTE: Connection is created INSIDE the thread function because
        SQLite connections are not thread-safe by default.
        """
        def claim_nodes(db, exec_id, lim):
            # Create connection inside thread for thread safety
            with db._connect() as conn:
                claimed = []
                conn.execute("BEGIN IMMEDIATE")
                try:
                    rows = conn.execute(
                        "SELECT node_id FROM node_executions "
                        "WHERE execution_id=? AND status='ready' LIMIT ?",
                        (exec_id, lim)
                    ).fetchall()

                    for (node_id,) in rows:
                        result = conn.execute(
                            "UPDATE node_executions SET status='running', version=version+1 "
                            "WHERE execution_id=? AND node_id=? AND status='ready'",
                            (exec_id, node_id)
                        )
                        if result.rowcount > 0:
                            claimed.append(node_id)

                    conn.commit()
                except:
                    conn.rollback()
                    raise
                return claimed

        # Pass db reference, not connection - connection created inside thread
        return await asyncio.to_thread(claim_nodes, self.db, execution_id, limit)
```

### Node Execution

```python
    async def _execute_node(self, execution_id: str, workflow: WorkflowGraph,
                           node_id: str) -> bool:
        """
        Execute a single node as an atomic transaction.

        Transaction Steps:
        1. Execute node logic (task/gate/branch/etc.)
        2. Store output in DB
        3. Mark node as COMPLETED/FAILED
        4. Update dependent nodes to READY if conditions met
        """
        node = next(n for n in workflow.nodes if n.id == node_id)

        try:
            # Collect inputs from completed upstream nodes
            # Wrap in to_thread to avoid blocking event loop during DB reads
            input_data = await asyncio.to_thread(
                self._collect_inputs, execution_id, workflow, node
            )

            # Execute based on node type
            if node.type == NodeType.TASK:
                output = await self._execute_task(execution_id, node, input_data)
            elif node.type == NodeType.GATE:
                output = await self._execute_gate(execution_id, node, input_data)
            elif node.type == NodeType.BRANCH:
                output = await self._execute_branch(execution_id, workflow, node, input_data)
            elif node.type == NodeType.MERGE:
                output = self._execute_merge(execution_id, workflow, node)
            elif node.type == NodeType.PARALLEL:
                output = await self._execute_parallel(execution_id, workflow, node)
            elif node.type == NodeType.HUMAN:
                output = await self._execute_human(execution_id, node, input_data)
            elif node.type == NodeType.SUBGRAPH:
                output = await self._execute_subgraph(execution_id, node, input_data)
            else:
                raise ValueError(f"Unknown node type: {node.type}")

            # Handle Pydantic models - serialize with model_dump()
            if hasattr(output, 'model_dump'):
                output = output.model_dump()

            # GATE FAILURE HANDLING: If gate returned passed=False, mark as FAILED
            # This ensures fail_fast can stop execution and downstream conditions work
            if node.type == NodeType.GATE and isinstance(output, dict) and not output.get("passed", True):
                def persist_gate_failure(conn, exec_id, n_id, out):
                    self._set_node_output(conn, exec_id, n_id, out)
                    self._set_node_status(conn, exec_id, n_id, NodeStatus.FAILED,
                                         error=f"Gate failed: {out.get('output', 'unknown')}")

                # Wrap in to_thread to avoid blocking event loop
                await asyncio.to_thread(
                    self._run_in_transaction, persist_gate_failure, execution_id, node_id, output
                )

                # Fail entire workflow if fail_fast is enabled
                if workflow.config.get("fail_fast", True):
                    await self._fail_workflow(execution_id, f"Gate '{node_id}' failed")
                return False  # Indicate failure without raising

            # Persist output and update dependents atomically
            # Wrap in to_thread to avoid blocking event loop
            def persist_completion(conn, exec_id, n_id, out, wf):
                self._set_node_output(conn, exec_id, n_id, out)
                self._set_node_status(conn, exec_id, n_id, NodeStatus.COMPLETED)
                self._update_dependents(conn, exec_id, wf, n_id, out)

            await asyncio.to_thread(
                self._run_in_transaction, persist_completion, execution_id, node_id, output, workflow
            )
            return True

        except Exception as e:
            # Mark as FAILED - wrap in to_thread to avoid blocking
            def persist_failure(conn, exec_id, n_id, err):
                self._set_node_status(conn, exec_id, n_id, NodeStatus.FAILED, error=err)

            await asyncio.to_thread(
                self._run_in_transaction, persist_failure, execution_id, node_id, str(e)
            )

            # Fail entire workflow if fail_fast is enabled
            if workflow.config.get("fail_fast", True):
                await self._fail_workflow(execution_id, str(e))
            raise
```

### Input Collection (Namespaced)

```python
    def _collect_inputs(self, execution_id: str, workflow: WorkflowGraph,
                       node: Node) -> Dict[str, Any]:
        """
        Collect inputs from completed upstream nodes.

        Inputs are namespaced by source node ID to prevent collisions:
        - "source_node.field_name" for namespaced access
        - "field_name" also stored for convenience

        For entry point nodes (no upstream edges), returns seeded input_data
        from start_workflow's initial_inputs parameter.
        """
        merged = {}

        # Check for seeded input_data (used for entry point nodes)
        # This allows external data to be passed into the workflow
        seeded_input = self._get_seeded_input(execution_id, node.id)
        if seeded_input:
            if isinstance(seeded_input, dict):
                merged.update(seeded_input)
            else:
                merged["input"] = seeded_input

        # Collect from upstream nodes
        for edge in [e for e in workflow.edges if e.target == node.id]:
            output = self._get_node_output(execution_id, edge.source)
            if output is None:
                continue

            # Apply data mapping if specified
            if edge.data_mapping:
                output = {
                    target_key: output.get(source_key) if isinstance(output, dict) else output
                    for target_key, source_key in edge.data_mapping.items()
                }

            # Store both namespaced and un-namespaced versions
            if isinstance(output, dict):
                for k, v in output.items():
                    merged[f"{edge.source}.{k}"] = v  # Namespaced
                    merged[k] = v  # Also un-namespaced for convenience
            else:
                merged[edge.source] = output

        return merged

    def _get_seeded_input(self, execution_id: str, node_id: str) -> Any:
        """Get seeded input_data for a node (set during start_workflow)."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT input_data FROM node_executions WHERE execution_id=? AND node_id=?",
                (execution_id, node_id)
            ).fetchone()
            return json.loads(row[0]) if row and row[0] else None
```

### Dependent Node Updates (Handles Loops)

```python
    def _update_dependents(self, conn, execution_id: str, workflow: WorkflowGraph,
                          completed_id: str, output: Any):
        """
        After node completes, update downstream nodes.

        Key Features:
        - Evaluates edge conditions
        - Handles loop re-execution (reset COMPLETED nodes to READY)
        - Respects wait_for_incoming policy
        """
        for edge in [e for e in workflow.edges if e.source == completed_id]:
            # Check edge condition
            if edge.condition:
                if not self._eval_condition(edge.condition, output):
                    continue  # Condition not met, skip this edge

            target = next(n for n in workflow.nodes if n.id == edge.target)
            current = self._get_node_status_single(execution_id, edge.target)

            # LOOP RE-EXECUTION: Reset completed nodes when taking loop edge
            if edge.is_loop_edge and current == NodeStatus.COMPLETED.value:
                self._set_node_status(conn, execution_id, edge.target, NodeStatus.READY)
                self._clear_node_output(conn, execution_id, edge.target)
                continue

            # Skip if already processed
            if current != NodeStatus.PENDING.value:
                continue

            # Check if node is ready based on wait_for_incoming policy
            if self._is_node_ready(execution_id, workflow, edge.target, target):
                self._set_node_status(conn, execution_id, edge.target, NodeStatus.READY)
```

### Task Execution (Pre-Apply Safety)

```python
    async def _execute_task(self, execution_id: str, node: Node,
                           input_data: Dict) -> Any:
        """
        Execute a role-based task.

        SAFETY: Gates are passed to run_role so they execute BEFORE
        changes are applied to the main repository.
        """
        config = node.task_config

        # Format input_data into task description string
        task_description = config.task_template.format(
            input=json.dumps(input_data, indent=2),
            **input_data
        )

        # Get workflow_id from execution record (not execution_id)
        workflow_id = self._get_workflow_id(execution_id)

        async with self.semaphore:  # Limit concurrency
            # CORRECT SIGNATURE: run_role with gates for pre-apply safety
            coro = asyncio.to_thread(
                self.engine.run_role,
                config.role,
                task_description,
                workflow_id,       # User's workflow_id (for metrics/reporting)
                node.id,           # step_id
                None,              # target_files
                None,              # extra_context
                None,              # retry_policy
                config.gates or None,  # gates - run BEFORE apply
                None,              # cli_override
                None               # cancellation_check
            )
            # Only wrap with wait_for if timeout is specified
            # (asyncio.wait_for raises TypeError when timeout=None)
            if config.timeout:
                result = await asyncio.wait_for(coro, timeout=config.timeout)
            else:
                result = await coro

        return result

    def _get_workflow_id(self, execution_id: str) -> str:
        """Get user-provided workflow_id from execution record."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT workflow_id FROM graph_executions WHERE id=?",
                (execution_id,)
            ).fetchone()
            return row[0] if row else execution_id

    def _get_execution_status(self, execution_id: str) -> str:
        """Get current execution status from DB."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT status FROM graph_executions WHERE id=?",
                (execution_id,)
            ).fetchone()
            return row[0] if row else "unknown"

    def _get_all_statuses(self, execution_id: str) -> Dict[str, str]:
        """Get all node statuses for an execution."""
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, status FROM node_executions WHERE execution_id=?",
                (execution_id,)
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    # --- Public Accessors (for external use by CLI/Web) ---

    def get_execution_status(self, execution_id: str) -> str:
        """Public accessor for execution status."""
        return self._get_execution_status(execution_id)

    def get_all_node_statuses(self, execution_id: str) -> Dict[str, str]:
        """Public accessor for all node statuses."""
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, status FROM node_executions WHERE execution_id=?",
                (execution_id,)
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    def get_node_output(self, execution_id: str, node_id: str) -> Any:
        """Public accessor for node output."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT output_data FROM node_executions WHERE execution_id=? AND node_id=?",
                (execution_id, node_id)
            ).fetchone()
            return json.loads(row[0]) if row and row[0] else None
```

### Gate Execution

```python
    async def _execute_gate(self, execution_id: str, node: Node,
                           input_data: Dict) -> Any:
        """
        Execute a verification gate.

        SAFETY: Uses isolated_execution context manager for:
        - Automatic worktree creation under .worktrees/
        - Proper file locking and path validation
        - Guaranteed cleanup after execution
        """
        config = node.gate_config

        def run_gate_isolated():
            # Use public isolated_execution context manager for safety
            with self.engine.workspace.isolated_execution(
                f"gate_{execution_id}_{node.id}"
            ) as ctx:
                gate_config = self.gate_loader.get_gate(config.gate_type)
                result = self.gate_executor.run_gate(
                    gate_config,
                    ctx.worktree_path,  # Safe path under .worktrees/
                    execution_id,
                    node.id
                )
                return result

        # Run in thread to avoid blocking
        result = await asyncio.to_thread(run_gate_isolated)

        passed = result.status == GateStatus.PASSED
        return {
            "passed": passed,
            "output": result.output,
            "test_status": result.status.value
        }
```

### Branch Execution (Loop Control)

```python
    async def _execute_branch(self, execution_id: str, workflow: WorkflowGraph,
                             node: Node, input_data: Dict) -> Any:
        """
        Execute branch logic with loop control.

        Prevents infinite loops via max_iterations.
        """
        config = node.branch_config

        # Find loop edge for counter tracking
        loop_edge = next(
            (e for e in workflow.edges if e.source == node.id and e.is_loop_edge),
            None
        )
        loop_key = f"loop_{loop_edge.id if loop_edge else node.id}"
        iteration = self._get_loop_counter(execution_id, loop_key)

        # Check max iterations
        if iteration >= config.condition.max_iterations:
            return {
                "branch_outcome": BranchOutcome.MAX_ITERATIONS.value,
                "iterations": iteration
            }

        # Evaluate condition
        result = self._eval_condition(config.condition, input_data)
        outcome = BranchOutcome.TRUE.value if result else BranchOutcome.FALSE.value

        # Increment counter when taking loop edge
        if loop_edge:
            target = config.on_true if result else config.on_false
            if loop_edge.target == target:
                self._increment_loop_counter(execution_id, loop_key)

        return {
            "branch_outcome": outcome,
            "condition_result": result,
            "iterations": iteration
        }
```

### Human Node Execution

```python
    async def _execute_human(self, execution_id: str, node: Node,
                            input_data: Dict) -> Any:
        """
        Execute human approval node.

        IMPORTANT: request_approval is a blocking call (waits for user input).
        We wrap it in asyncio.to_thread to avoid blocking the event loop.
        """
        config = node.human_config

        # Prepare changes list from input data
        changes = []
        if config.description:
            changes.append(config.description)
        if isinstance(input_data, dict):
            for key, value in input_data.items():
                if not key.startswith('_'):  # Skip internal fields
                    changes.append(f"{key}: {value}")

        # CORRECT SIGNATURE: request_approval with all optional params
        # Wrapped in to_thread because it blocks waiting for user input
        decision = await asyncio.to_thread(
            self.approval_gate.request_approval,
            feature_id=execution_id,
            title=config.title,
            changes=changes,  # list[str], not single str
            review_summary=json.dumps(input_data, indent=2),
            # Optional params for TUI integration
            bridge=None,       # InteractionBridge for rich TUI (optional)
            diff_lines=None    # Diff lines for display (optional)
        )

        return {"approved": decision == ApprovalDecision.APPROVE}
```

### Subgraph Execution (Nested Workflows)

```python
    async def _execute_subgraph(self, execution_id: str, node: Node,
                                input_data: Dict) -> Any:
        """
        Execute a nested workflow as a subgraph.

        NOTE: Full subgraph execution requires recursive orchestration.
        This is a planned feature - for now, raise NotImplementedError
        with a clear message so users know it's not yet available.

        Future implementation will:
        1. Load the nested workflow by name
        2. Map parent inputs to child inputs
        3. Start a child execution
        4. Wait for child completion
        5. Map child outputs back to parent
        """
        config = node.subgraph_config
        if not config:
            raise ValueError(f"Subgraph node '{node.id}' missing subgraph_config")

        # TODO: Implement nested workflow execution
        # For now, explicitly fail so users know this isn't supported yet
        raise NotImplementedError(
            f"Subgraph execution not yet implemented. "
            f"Node '{node.id}' references workflow '{config.workflow_name}'. "
            f"Remove SUBGRAPH nodes from your workflow or wait for future release."
        )
```

### Skip Propagation

```python
    async def _propagate_skips(self, execution_id: str, workflow: WorkflowGraph):
        """
        Mark nodes as SKIPPED if all their incoming edges are unreachable.

        This handles branch conditions where one path is taken and
        the other path's nodes should be marked as skipped (not deadlocked).
        """
        statuses = self._get_all_statuses(execution_id)
        changed = True

        while changed:
            changed = False
            for node in workflow.nodes:
                if statuses.get(node.id) != NodeStatus.PENDING.value:
                    continue

                incoming = [e for e in workflow.edges if e.target == node.id]
                if not incoming:
                    continue

                # Check if all incoming sources are unreachable
                all_unreachable = True
                for edge in incoming:
                    src_status = statuses.get(edge.source)
                    if src_status == NodeStatus.SKIPPED.value:
                        continue
                    if src_status == NodeStatus.COMPLETED.value:
                        if edge.condition:
                            output = self._get_node_output(execution_id, edge.source)
                            if self._eval_condition(edge.condition, output):
                                all_unreachable = False
                                break
                        else:
                            all_unreachable = False
                            break
                    else:
                        all_unreachable = False
                        break

                if all_unreachable:
                    def mark_skipped(conn, exec_id, n_id):
                        self._set_node_status(conn, exec_id, n_id, NodeStatus.SKIPPED)

                    self._run_in_transaction(mark_skipped, execution_id, node.id)
                    statuses[node.id] = NodeStatus.SKIPPED.value
                    changed = True
```

---

## 1.3 State Schema Extension

**File:** `supervisor/core/state.py` (Extend)

**Purpose:** Add database tables for graph execution with resumability support.

### Database Initialization

Enable WAL (Write-Ahead Logging) mode to prevent "database is locked" errors
when multiple readers (CLI monitor, WebSocket) and writers (workers) access
the database concurrently.

```python
class Database:
    def __init__(self, db_path: str = ".supervisor/state.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database with WAL mode and schema."""
        with self._connect() as conn:
            # Enable WAL mode for better concurrent read/write performance
            # WAL allows readers and writers to operate simultaneously
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout for locks

            # Create tables (see schema below)
            self._create_tables(conn)
```

### SQL Schema

```sql
-- Workflow definitions (stored as JSON)
CREATE TABLE IF NOT EXISTS graph_workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    definition JSON NOT NULL,
    version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workflow execution instances
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
    version INTEGER DEFAULT 0,  -- For optimistic locking
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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_node_exec_status
    ON node_executions(execution_id, status);
CREATE INDEX IF NOT EXISTS idx_node_ready
    ON node_executions(execution_id, status) WHERE status = 'ready';
CREATE INDEX IF NOT EXISTS idx_graph_exec_status
    ON graph_executions(workflow_id, status);
```

---

## 1.4 Worker/Scheduler

**File:** `supervisor/core/worker.py` (New)

**Purpose:** Background worker that polls for READY nodes and executes them.

```python
import asyncio
from supervisor.core.graph_engine import GraphOrchestrator
from supervisor.core.state import Database

class WorkflowWorker:
    """
    Background worker that executes workflows.

    Design:
    - Polls DB for active executions
    - Processes one batch at a time per execution
    - Can run multiple workers for horizontal scaling
    """

    def __init__(self, orchestrator: GraphOrchestrator,
                 poll_interval: float = 1.0):
        self.orchestrator = orchestrator
        self.poll_interval = poll_interval
        self.running = False

    async def run_until_complete(self, execution_id: str) -> str:
        """
        Run a single execution until completion.
        Returns final status.
        """
        while True:
            status = self.orchestrator.get_execution_status(execution_id)
            if status in ["completed", "failed", "cancelled"]:
                return status

            await self.orchestrator.execute_next_batch(execution_id)
            await asyncio.sleep(self.poll_interval)

    async def start_daemon(self):
        """
        Start daemon mode - process all active executions.
        """
        self.running = True

        while self.running:
            try:
                # Find all running executions
                active = self._get_active_executions()

                for execution_id in active:
                    await self.orchestrator.execute_next_batch(execution_id)

            except Exception as e:
                logger.error(f"Worker error: {e}")

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        """Stop the worker daemon"""
        self.running = False
```

---

## 1.5 CLI Integration

**File:** `supervisor/cli.py` (Modify)

**Purpose:** Add CLI commands for graph-based workflows.

```python
@cli.command()
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--workflow-id", required=True, help="Unique workflow execution ID")
@click.option("--validate-only", is_flag=True, help="Only validate, don't execute")
def run_graph(workflow_file: str, workflow_id: str, validate_only: bool):
    """Execute a declarative workflow graph from YAML"""
    import yaml

    # Load workflow YAML
    with open(workflow_file) as f:
        workflow = WorkflowGraph(**yaml.safe_load(f))

    # Validate
    errors = workflow.validate_graph()
    if errors:
        click.secho("Validation errors:", fg="red")
        for error in errors:
            click.echo(f"  - {error}")
        return

    click.secho("Workflow validation passed", fg="green")
    click.echo(f"  Nodes: {len(workflow.nodes)}")
    click.echo(f"  Edges: {len(workflow.edges)}")

    if validate_only:
        return

    # Initialize components
    db = Database()
    engine = ExecutionEngine(Path("."))
    orchestrator = GraphOrchestrator(db, engine, engine.gate_executor, engine.gate_loader)

    # Start and run workflow
    async def run():
        exec_id = await orchestrator.start_workflow(workflow, workflow_id)
        worker = WorkflowWorker(orchestrator)
        return await worker.run_until_complete(exec_id)

    status = asyncio.run(run())

    if status == "completed":
        click.secho("Workflow completed successfully", fg="green")
    else:
        click.secho(f"Workflow {status}", fg="red")
```

---

## YAML Workflow Examples

### Debug Workflow with Retry Loop

```yaml
id: debug-workflow
name: Debug with Retry Loop
version: 1.0.0
entry_point: analyze
exit_points: [review]

nodes:
  - id: analyze
    type: task
    label: Analyze Bug
    task_config:
      role: debugger
      task_template: "Analyze this issue: {input}"

  - id: fix
    type: task
    label: Apply Fix
    task_config:
      role: implementer
      task_template: "Fix based on analysis: {input}"

  - id: test
    type: gate
    label: Run Tests
    gate_config:
      gate_type: test_gate

  - id: check
    type: branch
    label: Check Test Result
    branch_config:
      condition:
        field: test_status
        operator: "=="
        value: "passed"
        max_iterations: 3
      on_true: review
      on_false: analyze

  - id: review
    type: task
    label: Final Review
    task_config:
      role: reviewer
      task_template: "Review: {input}"

edges:
  - { id: e1, source: analyze, target: fix }
  - { id: e2, source: fix, target: test }
  - { id: e3, source: test, target: check }
  - { id: e4, source: check, target: review,
      condition: { field: branch_outcome, operator: "==", value: "on_true" } }
  - { id: e5, source: check, target: analyze, is_loop_edge: true,
      condition: { field: branch_outcome, operator: "==", value: "on_false" } }

config:
  max_parallel_nodes: 2
  fail_fast: true
```

### Parallel Review Workflow

```yaml
id: parallel-review
name: Parallel Code Review
entry_point: implement
exit_points: [merge]

nodes:
  - id: implement
    type: task
    task_config:
      role: implementer
      task_template: "Implement feature: {input}"

  - id: split
    type: parallel
    parallel_config:
      branches: [lint, test, security]
      wait_for: all

  - id: lint
    type: gate
    gate_config: { gate_type: lint_gate }

  - id: test
    type: gate
    gate_config: { gate_type: test_gate }

  - id: security
    type: gate
    gate_config: { gate_type: security_gate }

  - id: join
    type: merge
    merge_config:
      wait_for: all
      merge_strategy: union

  - id: review
    type: human
    human_config:
      title: "Approve Changes"
      description: "All gates passed. Review and approve."

  - id: merge
    type: task
    task_config:
      role: implementer
      task_template: "Merge approved changes"

edges:
  - { id: e1, source: implement, target: split }
  - { id: e2, source: split, target: lint }
  - { id: e3, source: split, target: test }
  - { id: e4, source: split, target: security }
  - { id: e5, source: lint, target: join }
  - { id: e6, source: test, target: join }
  - { id: e7, source: security, target: join }
  - { id: e8, source: join, target: review }
  - { id: e9, source: review, target: merge,
      condition: { field: approved, operator: "==", value: true } }
```

---

## Integration Notes

### Existing Components Used

| Component | Module | Usage |
|-----------|--------|-------|
| `ExecutionEngine` | `supervisor/core/engine.py` | Execute roles with `run_role()` |
| `GateExecutor` | `supervisor/core/gate_executor.py` | Run gates with `run_gate()` |
| `GateLoader` | `supervisor/core/gate_loader.py` | Load gate configurations |
| `ApprovalGate` | `supervisor/core/approval.py` | Human approval requests |
| `Database` | `supervisor/core/state.py` | State persistence |
| `IsolatedWorkspace` | `supervisor/core/workspace.py` | Worktree management |

### Key API Signatures

```python
# ExecutionEngine.run_role (full signature)
run_role(
    role_name: str,
    task_description: str,
    workflow_id: str,
    step_id: str | None = None,
    target_files: list[str] | None = None,
    extra_context: dict | None = None,
    retry_policy: RetryPolicy | None = None,
    gates: list[str] | None = None,
    cli_override: str | None = None,
    cancellation_check: Callable[[], bool] | None = None  # For async cancellation
) -> BaseModel

# GateExecutor.run_gate
run_gate(
    config: GateConfig,
    worktree_path: Path,
    workflow_id: str,
    step_id: str
) -> GateResult

# ApprovalGate.request_approval (full signature)
request_approval(
    feature_id: str,
    title: str,
    changes: list[str],
    review_summary: str,
    component_id: str | None = None,
    bridge: InteractionBridge | None = None,  # For TUI integration
    diff_lines: list[str] | None = None       # For diff display
) -> ApprovalDecision

# Database._connect (context manager - synchronous)
with db._connect() as conn:
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(...)
        conn.commit()
    except:
        conn.rollback()
        raise

# IsolatedWorkspace.isolated_execution (context manager)
with engine.workspace.isolated_execution(context_id: str) as ctx:
    worktree_path = ctx.worktree_path  # Safe path under .worktrees/
    # ... execute in isolation ...
# Automatic cleanup on exit
```

---

**End of Phase 1 Plan**
