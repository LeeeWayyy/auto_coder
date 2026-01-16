"""Stateless graph workflow execution engine.

This module implements a crash-resistant, resumable graph orchestrator that:
- Stores all state in the database (no in-memory state)
- Uses transaction-based execution for atomicity
- Supports horizontal scaling via multiple workers
- Implements pre-apply safety with gates
"""

import asyncio
import json
import logging
import uuid
from typing import Any

from supervisor.core.approval import ApprovalGate
from supervisor.core.gate_executor import GateExecutor
from supervisor.core.gate_loader import GateLoader
from supervisor.core.gate_models import GateStatus
from supervisor.core.graph_schema import (
    BranchOutcome,
    Node,
    NodeStatus,
    NodeType,
    TransitionCondition,
    WorkflowGraph,
)
from supervisor.core.interaction import ApprovalDecision
from supervisor.core.state import Database

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

    def __init__(
        self,
        db: Database,
        execution_engine: Any,  # ExecutionEngine
        gate_executor: GateExecutor,
        gate_loader: GateLoader,
        max_parallel: int = 4,
    ):
        self.db = db
        self.engine = execution_engine
        self.gate_executor = gate_executor
        self.gate_loader = gate_loader
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.approval_gate = ApprovalGate(db)

    # ========== Database Transaction Helpers ==========

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
            except Exception:
                conn.rollback()
                raise

    def _save_workflow(self, conn, workflow: WorkflowGraph):
        """Save workflow definition to database."""
        conn.execute(
            """
            INSERT OR REPLACE INTO graph_workflows (id, name, definition, version, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (workflow.id, workflow.name, workflow.model_dump_json(), workflow.version),
        )

    def _create_execution(self, conn, execution_id: str, workflow_id: str, graph_id: str):
        """Create a new execution record."""
        conn.execute(
            """
            INSERT INTO graph_executions (id, workflow_id, graph_id, status, started_at)
            VALUES (?, ?, ?, 'running', CURRENT_TIMESTAMP)
        """,
            (execution_id, workflow_id, graph_id),
        )

    def _set_node_status(
        self,
        conn,
        execution_id: str,
        node_id: str,
        status: NodeStatus,
        node_type: str = None,
        error: str = None,
    ):
        """Set node execution status."""
        if node_type:
            # Initial creation with type
            conn.execute(
                """
                INSERT INTO node_executions (id, execution_id, node_id, node_type, status)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    f"{execution_id}_{node_id}",
                    execution_id,
                    node_id,
                    node_type,
                    status.value,
                ),
            )
        else:
            # Status update
            conn.execute(
                """
                UPDATE node_executions SET status=?, error=?
                WHERE execution_id=? AND node_id=?
            """,
                (status.value, error, execution_id, node_id),
            )

    def _set_node_output(self, conn, execution_id: str, node_id: str, output: Any):
        """Store node output data."""
        conn.execute(
            """
            UPDATE node_executions SET output_data=?, completed_at=CURRENT_TIMESTAMP
            WHERE execution_id=? AND node_id=?
        """,
            (json.dumps(output), execution_id, node_id),
        )

    def _get_node_output(self, execution_id: str, node_id: str) -> Any:
        """Get node output from database."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT output_data FROM node_executions WHERE execution_id=? AND node_id=?",
                (execution_id, node_id),
            ).fetchone()
            return json.loads(row[0]) if row and row[0] else None

    def _get_node_status_single(self, execution_id: str, node_id: str) -> str:
        """Get status of a single node."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT status FROM node_executions WHERE execution_id=? AND node_id=?",
                (execution_id, node_id),
            ).fetchone()
            return row[0] if row else NodeStatus.PENDING.value

    def _clear_node_output(self, conn, execution_id: str, node_id: str):
        """Clear node output for loop re-execution."""
        conn.execute(
            """
            UPDATE node_executions SET output_data=NULL, completed_at=NULL
            WHERE execution_id=? AND node_id=?
        """,
            (execution_id, node_id),
        )

    def _get_node_status_in_txn(self, conn, execution_id: str, node_id: str) -> str:
        """Get status of a single node within an existing transaction."""
        row = conn.execute(
            "SELECT status FROM node_executions WHERE execution_id=? AND node_id=?",
            (execution_id, node_id),
        ).fetchone()
        return row[0] if row else NodeStatus.PENDING.value

    def _set_node_status_guarded(
        self,
        conn,
        execution_id: str,
        node_id: str,
        new_status: NodeStatus,
        expected_status: NodeStatus,
    ) -> bool:
        """
        Set node status only if current status matches expected.
        Returns True if update was applied, False if status had changed.
        This prevents race conditions where a concurrent worker changed the status.
        """
        result = conn.execute(
            """
            UPDATE node_executions SET status=?, version=version+1
            WHERE execution_id=? AND node_id=? AND status=?
        """,
            (new_status.value, execution_id, node_id, expected_status.value),
        )
        return result.rowcount > 0

    def _get_workflow(self, execution_id: str) -> WorkflowGraph:
        """Load workflow definition for an execution."""
        with self.db._connect() as conn:
            row = conn.execute(
                """
                SELECT w.definition FROM graph_workflows w
                JOIN graph_executions e ON w.id = e.graph_id
                WHERE e.id = ?
            """,
                (execution_id,),
            ).fetchone()
            return WorkflowGraph.model_validate_json(row[0]) if row else None

    def _get_workflow_id(self, execution_id: str) -> str:
        """Get user-provided workflow_id from execution record."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT workflow_id FROM graph_executions WHERE id=?", (execution_id,)
            ).fetchone()
            return row[0] if row else execution_id

    def _get_execution_status(self, execution_id: str) -> str:
        """Get current execution status from DB."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT status FROM graph_executions WHERE id=?", (execution_id,)
            ).fetchone()
            return row[0] if row else "unknown"

    def _get_all_statuses(self, execution_id: str) -> dict[str, str]:
        """Get all node statuses for an execution."""
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, status FROM node_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    def _get_seeded_input(self, execution_id: str, node_id: str) -> Any:
        """Get seeded input_data for a node (set during start_workflow)."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT input_data FROM node_executions WHERE execution_id=? AND node_id=?",
                (execution_id, node_id),
            ).fetchone()
            return json.loads(row[0]) if row and row[0] else None

    def _get_loop_counter(self, execution_id: str, loop_key: str) -> int:
        """Get current iteration count for a loop."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT iteration_count FROM loop_counters WHERE execution_id=? AND loop_key=?",
                (execution_id, loop_key),
            ).fetchone()
            return row[0] if row else 0

    def _increment_loop_counter(self, execution_id: str, loop_key: str):
        """Increment loop iteration counter."""
        with self.db._connect() as conn:
            conn.execute(
                """
                INSERT INTO loop_counters (execution_id, loop_key, iteration_count)
                VALUES (?, ?, 1)
                ON CONFLICT(execution_id, loop_key)
                DO UPDATE SET iteration_count = iteration_count + 1
            """,
                (execution_id, loop_key),
            )

    def _reset_loop_counter(self, execution_id: str, loop_key: str):
        """Reset loop iteration counter (when exiting loop for potential re-entry)."""
        with self.db._connect() as conn:
            conn.execute(
                "DELETE FROM loop_counters WHERE execution_id=? AND loop_key=?",
                (execution_id, loop_key),
            )

    # ========== Public Accessors (for CLI/Web) ==========

    def get_execution_status(self, execution_id: str) -> str:
        """Public accessor for execution status."""
        return self._get_execution_status(execution_id)

    def get_all_node_statuses(self, execution_id: str) -> dict[str, str]:
        """Public accessor for all node statuses."""
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, status FROM node_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    def get_node_output(self, execution_id: str, node_id: str) -> Any:
        """Public accessor for node output."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT output_data FROM node_executions WHERE execution_id=? AND node_id=?",
                (execution_id, node_id),
            ).fetchone()
            return json.loads(row[0]) if row and row[0] else None

    # ========== Workflow Lifecycle ==========

    async def start_workflow(
        self,
        workflow: WorkflowGraph,
        workflow_id: str,
        initial_inputs: dict[str, Any] | None = None,
    ) -> str:
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

        execution_id = str(uuid.uuid4())

        # Persist workflow and initialize node states atomically
        def init_workflow(conn, exec_id, wf_id, wf, inputs):
            self._save_workflow(conn, wf)
            self._create_execution(conn, exec_id, wf_id, wf.id)
            for node in wf.nodes:
                self._set_node_status(conn, exec_id, node.id, NodeStatus.PENDING, node.type.value)
            self._set_node_status(conn, exec_id, wf.entry_point, NodeStatus.READY)

            # Seed entry point with initial inputs (if provided)
            if inputs:
                conn.execute(
                    "UPDATE node_executions SET input_data=? WHERE execution_id=? AND node_id=?",
                    (json.dumps(inputs), exec_id, wf.entry_point),
                )

        # Wrap in to_thread to avoid blocking the event loop
        await asyncio.to_thread(
            self._run_in_transaction,
            init_workflow,
            execution_id,
            workflow_id,
            workflow,
            initial_inputs,
        )

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
            execution_id, workflow.config.get("max_parallel_nodes", 4)
        )

        if not ready_nodes:
            # No ready nodes - check if workflow should complete
            await self._check_workflow_state(execution_id, workflow)
            return 0

        # Execute batch in parallel
        results = await asyncio.gather(
            *[self._execute_node(execution_id, workflow, nid) for nid in ready_nodes],
            return_exceptions=True,
        )

        return sum(1 for r in results if not isinstance(r, Exception))

    async def _claim_ready_nodes(self, execution_id: str, limit: int) -> list[str]:
        """
        Atomically claim READY nodes to prevent duplicate execution.
        Uses optimistic locking via version field.
        """

        def claim_nodes(db, exec_id, max_limit):
            with db._connect() as conn:
                claimed = []
                conn.execute("BEGIN IMMEDIATE")
                try:
                    # Count currently running nodes to enforce global parallelism limit
                    running_count = conn.execute(
                        "SELECT COUNT(*) FROM node_executions "
                        "WHERE execution_id=? AND status='running'",
                        (exec_id,),
                    ).fetchone()[0]

                    # Calculate effective limit (don't claim more than allowed)
                    effective_limit = max(0, max_limit - running_count)
                    if effective_limit == 0:
                        conn.commit()
                        return claimed

                    rows = conn.execute(
                        "SELECT node_id FROM node_executions "
                        "WHERE execution_id=? AND status='ready' LIMIT ?",
                        (exec_id, effective_limit),
                    ).fetchall()

                    for (node_id,) in rows:
                        result = conn.execute(
                            "UPDATE node_executions SET status='running', version=version+1 "
                            "WHERE execution_id=? AND node_id=? AND status='ready'",
                            (exec_id, node_id),
                        )
                        if result.rowcount > 0:
                            claimed.append(node_id)

                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                return claimed

        return await asyncio.to_thread(claim_nodes, self.db, execution_id, limit)

    async def _check_workflow_state(self, execution_id: str, workflow: WorkflowGraph):
        """Check if workflow should complete or mark skipped nodes."""
        # Propagate skips first
        await self._propagate_skips(execution_id, workflow)

        # Check if all terminal
        statuses = self._get_all_statuses(execution_id)
        all_terminal = all(s in ["completed", "failed", "skipped"] for s in statuses.values())

        if all_terminal:
            # Check if any nodes failed
            any_failed = any(s == "failed" for s in statuses.values())

            # Check if any exit point reached successfully
            any_exit_completed = any(statuses.get(ep) == "completed" for ep in workflow.exit_points)

            if any_failed:
                # If any node failed, workflow failed (even if exit reached)
                failed_nodes = [nid for nid, s in statuses.items() if s == "failed"]
                await self._fail_workflow(
                    execution_id, f"Node(s) failed: {', '.join(failed_nodes)}"
                )
            elif any_exit_completed:
                await self._complete_workflow(execution_id)
            else:
                await self._fail_workflow(execution_id, "No exit point reached")

    async def _complete_workflow(self, execution_id: str):
        """Mark workflow as completed."""

        def complete(conn, exec_id):
            conn.execute(
                "UPDATE graph_executions SET status='completed', completed_at=CURRENT_TIMESTAMP WHERE id=?",
                (exec_id,),
            )

        await asyncio.to_thread(self._run_in_transaction, complete, execution_id)

    async def _fail_workflow(self, execution_id: str, error: str):
        """Mark workflow as failed."""

        def fail(conn, exec_id, err):
            conn.execute(
                "UPDATE graph_executions SET status='failed', completed_at=CURRENT_TIMESTAMP, error=? WHERE id=?",
                (err, exec_id),
            )

        await asyncio.to_thread(self._run_in_transaction, fail, execution_id, error)

    # ========== Node Execution ==========

    async def _execute_node(self, execution_id: str, workflow: WorkflowGraph, node_id: str) -> bool:
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
            input_data = await asyncio.to_thread(self._collect_inputs, execution_id, workflow, node)

            # Execute based on node type
            if node.type == NodeType.TASK:
                output = await self._execute_task(execution_id, node, input_data)
            elif node.type == NodeType.GATE:
                output = await self._execute_gate(execution_id, node, input_data)
            elif node.type == NodeType.BRANCH:
                output = await self._execute_branch(execution_id, workflow, node, input_data)
            elif node.type == NodeType.MERGE:
                output = await asyncio.to_thread(self._execute_merge, execution_id, workflow, node)
            elif node.type == NodeType.PARALLEL:
                output = await self._execute_parallel(execution_id, workflow, node)
            elif node.type == NodeType.HUMAN:
                output = await self._execute_human(execution_id, node, input_data)
            elif node.type == NodeType.SUBGRAPH:
                output = await self._execute_subgraph(execution_id, node, input_data)
            else:
                raise ValueError(f"Unknown node type: {node.type}")

            # Handle Pydantic models - serialize with model_dump()
            if hasattr(output, "model_dump"):
                output = output.model_dump()

            # GATE FAILURE HANDLING
            if (
                node.type == NodeType.GATE
                and isinstance(output, dict)
                and not output.get("passed", True)
            ):

                def persist_gate_failure(conn, exec_id, n_id, out):
                    self._set_node_output(conn, exec_id, n_id, out)
                    self._set_node_status(
                        conn,
                        exec_id,
                        n_id,
                        NodeStatus.FAILED,
                        error=f"Gate failed: {out.get('output', 'unknown')}",
                    )

                await asyncio.to_thread(
                    self._run_in_transaction,
                    persist_gate_failure,
                    execution_id,
                    node_id,
                    output,
                )

                if workflow.config.get("fail_fast", True):
                    await self._fail_workflow(execution_id, f"Gate '{node_id}' failed")
                return False

            # Persist output and update dependents atomically
            def persist_completion(conn, exec_id, n_id, out, wf):
                self._set_node_output(conn, exec_id, n_id, out)
                self._set_node_status(conn, exec_id, n_id, NodeStatus.COMPLETED)
                self._update_dependents(conn, exec_id, wf, n_id, out)

            await asyncio.to_thread(
                self._run_in_transaction,
                persist_completion,
                execution_id,
                node_id,
                output,
                workflow,
            )
            return True

        except Exception as e:
            logger.error(f"Node {node_id} failed: {e}")

            def persist_failure(conn, exec_id, n_id, err):
                self._set_node_status(conn, exec_id, n_id, NodeStatus.FAILED, error=err)

            await asyncio.to_thread(
                self._run_in_transaction, persist_failure, execution_id, node_id, str(e)
            )

            if workflow.config.get("fail_fast", True):
                await self._fail_workflow(execution_id, str(e))
            raise

    def _collect_inputs(
        self, execution_id: str, workflow: WorkflowGraph, node: Node
    ) -> dict[str, Any]:
        """
        Collect inputs from completed upstream nodes.

        Inputs are namespaced by source node ID to prevent collisions.
        """
        merged = {}

        # Check for seeded input_data
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

            # Check edge condition before including output
            if edge.condition:
                if not self._eval_condition(edge.condition, output):
                    # Skip this edge - condition not met
                    continue

            # Apply data mapping if specified
            if edge.data_mapping:
                output = {
                    target_key: output.get(source_key) if isinstance(output, dict) else output
                    for target_key, source_key in edge.data_mapping.items()
                }

            # Store both namespaced keys and the full output under node ID
            # Namespaced keys (source.field) prevent collisions between nodes
            # Full output under node ID enables {node_id} template references
            if isinstance(output, dict):
                for k, v in output.items():
                    merged[f"{edge.source}.{k}"] = v  # Namespaced keys
                merged[edge.source] = output  # Full dict for {node_id} templates
            else:
                merged[edge.source] = output

        return merged

    def _update_dependents(
        self,
        conn,
        execution_id: str,
        workflow: WorkflowGraph,
        completed_id: str,
        output: Any,
    ):
        """
        After node completes, update downstream nodes.

        Key Features:
        - Evaluates edge conditions
        - Handles BRANCH routing via on_true/on_false
        - Marks non-selected branch targets as SKIPPED
        - Handles loop re-execution
        - Respects wait_for_incoming policy
        - Uses same transaction for reads/writes to prevent race conditions
        """
        # Get the completed node to check if it's a BRANCH
        completed_node = next((n for n in workflow.nodes if n.id == completed_id), None)

        # Track which target was selected by BRANCH (for skip propagation)
        branch_allowed_target = None

        # First pass: determine allowed target for BRANCH nodes
        if (
            completed_node
            and completed_node.type == NodeType.BRANCH
            and completed_node.branch_config
        ):
            branch_config = completed_node.branch_config
            branch_outcome = output.get("branch_outcome") if isinstance(output, dict) else None

            # Determine allowed target based on branch outcome
            if branch_outcome == BranchOutcome.TRUE.value:
                branch_allowed_target = branch_config.on_true
            elif branch_outcome == BranchOutcome.FALSE.value:
                branch_allowed_target = branch_config.on_false
            elif branch_outcome == BranchOutcome.MAX_ITERATIONS.value:
                # Use selected_target from output (determined by is_loop_edge)
                branch_allowed_target = output.get("selected_target", branch_config.on_false)

            # Mark non-selected branch target as SKIPPED to prevent hanging
            # BUT only if the target doesn't have other valid incoming edges
            if branch_allowed_target:
                non_selected = (
                    branch_config.on_false
                    if branch_allowed_target == branch_config.on_true
                    else branch_config.on_true
                )
                # Guard: only skip if non_selected is different from the allowed target
                # (handles case where on_true == on_false, e.g., do-while loops)
                if non_selected == branch_allowed_target:
                    pass  # Same node - don't skip the target we're about to execute
                elif (
                    self._get_node_status_in_txn(conn, execution_id, non_selected)
                    == NodeStatus.PENDING.value
                ):
                    # Check if non-selected target has other incoming edges
                    non_selected_node = next(
                        (n for n in workflow.nodes if n.id == non_selected), None
                    )
                    other_incoming = [
                        e
                        for e in workflow.edges
                        if e.target == non_selected and e.source != completed_id
                    ]
                    # Only skip if:
                    # 1. No other incoming edges, OR
                    # 2. Node requires all inputs (wait_for_incoming="all")
                    should_skip = not other_incoming or (
                        non_selected_node and non_selected_node.wait_for_incoming == "all"
                    )
                    if should_skip:
                        self._set_node_status_guarded(
                            conn,
                            execution_id,
                            non_selected,
                            NodeStatus.SKIPPED,
                            expected_status=NodeStatus.PENDING,
                        )

        for edge in [e for e in workflow.edges if e.source == completed_id]:
            # BRANCH ROUTING: Skip edges that don't match the branch decision
            if branch_allowed_target and edge.target != branch_allowed_target:
                continue

            # Check edge condition (for non-BRANCH nodes or additional filtering)
            if edge.condition:
                if not self._eval_condition(edge.condition, output):
                    continue

            target = next(n for n in workflow.nodes if n.id == edge.target)
            # Use same connection to prevent race conditions
            current = self._get_node_status_in_txn(conn, execution_id, edge.target)

            # LOOP RE-EXECUTION
            if edge.is_loop_edge and current == NodeStatus.COMPLETED.value:
                # Reset all nodes in the loop cycle, not just the loop target
                # Find all nodes between loop target and this BRANCH node
                loop_nodes = self._find_loop_cycle_nodes(workflow, edge.target, completed_id)
                for loop_node_id in loop_nodes:
                    loop_status = self._get_node_status_in_txn(conn, execution_id, loop_node_id)
                    # Reset both COMPLETED and SKIPPED nodes on loop re-entry
                    # SKIPPED nodes may need to run on subsequent iterations if branch
                    # conditions change (e.g., alternating paths in loop body)
                    if loop_status in (NodeStatus.COMPLETED.value, NodeStatus.SKIPPED.value):
                        self._set_node_status_guarded(
                            conn,
                            execution_id,
                            loop_node_id,
                            NodeStatus.READY,
                            expected_status=NodeStatus(loop_status),
                        )
                        self._clear_node_output(conn, execution_id, loop_node_id)
                continue

            # Skip if already processed
            if current != NodeStatus.PENDING.value:
                continue

            # Check if node is ready (use same connection)
            if self._is_node_ready_in_txn(conn, execution_id, workflow, edge.target, target):
                # Guard with expected status to prevent race
                self._set_node_status_guarded(
                    conn,
                    execution_id,
                    edge.target,
                    NodeStatus.READY,
                    expected_status=NodeStatus.PENDING,
                )

    def _find_loop_cycle_nodes(
        self, workflow: WorkflowGraph, start_node: str, end_node: str
    ) -> list[str]:
        """
        Find all nodes in the loop cycle from start_node back to end_node.

        Only includes nodes that are BOTH:
        1. Reachable from start_node (forward direction)
        2. Can reach end_node (backward direction)

        This excludes side branches that don't lead back to the loop exit.
        Returns list including start_node but excluding end_node (the BRANCH).
        """
        # Build forward adjacency list
        adj: dict[str, list[str]] = {}
        for edge in workflow.edges:
            if edge.source not in adj:
                adj[edge.source] = []
            adj[edge.source].append(edge.target)

        # Build reverse adjacency list
        rev_adj: dict[str, list[str]] = {}
        for edge in workflow.edges:
            if edge.target not in rev_adj:
                rev_adj[edge.target] = []
            rev_adj[edge.target].append(edge.source)

        # Step 1: Find all nodes that can reach end_node (reverse BFS)
        can_reach_end: set[str] = set()
        queue = [end_node]
        while queue:
            node = queue.pop(0)
            if node in can_reach_end:
                continue
            can_reach_end.add(node)
            for pred in rev_adj.get(node, []):
                if pred not in can_reach_end:
                    queue.append(pred)

        # Step 2: Forward BFS from start_node, only including nodes that can reach end
        visited: set[str] = set()
        queue = [start_node]
        loop_nodes: list[str] = []

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if node == end_node:
                # Don't include the BRANCH node itself
                continue

            # Only include if this node can reach end_node
            if node in can_reach_end:
                loop_nodes.append(node)

            # Only explore neighbors that can reach end_node
            for neighbor in adj.get(node, []):
                if neighbor not in visited and neighbor in can_reach_end:
                    queue.append(neighbor)

        return loop_nodes

    def _is_node_ready_in_txn(
        self, conn, execution_id: str, workflow: WorkflowGraph, node_id: str, node: Node
    ) -> bool:
        """
        Check if node is ready based on wait_for_incoming policy.
        Uses existing connection to prevent race conditions.
        """
        incoming = [e for e in workflow.edges if e.target == node_id]
        if not incoming:
            return True

        completed_count = 0
        for edge in incoming:
            src_status = self._get_node_status_in_txn(conn, execution_id, edge.source)
            if src_status == NodeStatus.COMPLETED.value:
                # Check edge condition if present
                if edge.condition:
                    row = conn.execute(
                        "SELECT output_data FROM node_executions WHERE execution_id=? AND node_id=?",
                        (execution_id, edge.source),
                    ).fetchone()
                    output = json.loads(row[0]) if row and row[0] else None
                    if self._eval_condition(edge.condition, output):
                        completed_count += 1
                else:
                    completed_count += 1

        # MERGE nodes use merge_config.wait_for, others use node.wait_for_incoming
        if node.type == NodeType.MERGE and node.merge_config:
            wait_policy = node.merge_config.wait_for
        else:
            wait_policy = node.wait_for_incoming

        if wait_policy == "all":
            return completed_count == len(incoming)
        else:  # "any"
            return completed_count > 0

    def _eval_condition(self, condition: TransitionCondition, data: Any) -> bool:
        """
        Evaluate a transition condition safely.

        Type mismatches and invalid comparisons return False instead of raising.
        Supports both namespaced (source.field) and un-namespaced field lookups.
        """
        if not isinstance(data, dict):
            return False

        # Extract field value with proper precedence:
        # 1. Direct key match (including explicit None values)
        # 2. Dotted notation for nested dict access
        # 3. Un-namespaced convenience match (e.g., "field" matches "source.field")
        value = None
        found = False

        # 1. Direct key match
        if condition.field in data:
            value = data[condition.field]
            found = True

        # 2. Dotted notation for nested dict access
        if not found and "." in condition.field:
            parts = condition.field.split(".")
            temp_val = data
            try:
                for part in parts:
                    temp_val = temp_val[part]
                value = temp_val
                found = True
            except (KeyError, TypeError):
                pass  # Not found via nested access

        # 2b. Fallback: treat dotted field as "source.field" format and extract just the field part
        # This handles edge conditions evaluated against raw source output (flat dicts)
        # e.g., condition field "test.test_status" with data {"test_status": "passed"}
        if not found and "." in condition.field:
            # Try the last part of the dotted field as a direct key
            field_part = condition.field.split(".")[-1]
            if field_part in data:
                value = data[field_part]
                found = True

        # 3. Un-namespaced convenience match (deterministic via sorted keys)
        # NOTE: If multiple upstream nodes provide the same field name, the first
        # alphabetically sorted namespaced key wins. For explicit control, use
        # fully-qualified field names like "node_id.field".
        if not found and "." not in condition.field:
            suffix = f".{condition.field}"
            for key in sorted(data.keys()):  # Sort for deterministic matching
                if key.endswith(suffix):
                    value = data[key]
                    found = True
                    break

        # Evaluate operator with type safety
        try:
            if condition.operator == "==":
                return value == condition.value
            elif condition.operator == "!=":
                return value != condition.value
            elif condition.operator == ">":
                # Comparison operators require compatible types
                if value is None or not isinstance(value, type(condition.value)):
                    return False
                return value > condition.value
            elif condition.operator == "<":
                if value is None or not isinstance(value, type(condition.value)):
                    return False
                return value < condition.value
            elif condition.operator == ">=":
                if value is None or not isinstance(value, type(condition.value)):
                    return False
                return value >= condition.value
            elif condition.operator == "<=":
                if value is None or not isinstance(value, type(condition.value)):
                    return False
                return value <= condition.value
            elif condition.operator == "in":
                # Check if condition.value is iterable (for membership test)
                if not isinstance(condition.value, (str, list, tuple, set)):
                    return False
                return value in condition.value
            elif condition.operator == "not_in":
                if not isinstance(condition.value, (str, list, tuple, set)):
                    return False
                return value not in condition.value
            elif condition.operator == "contains":
                # Support str, list, and dict (check for key)
                if isinstance(value, (dict, str, list)):
                    return condition.value in value
                return False
            elif condition.operator == "starts_with":
                return value.startswith(condition.value) if isinstance(value, str) else False
            elif condition.operator == "ends_with":
                return value.endswith(condition.value) if isinstance(value, str) else False
            else:
                return False
        except (TypeError, AttributeError):
            # Any comparison error returns False instead of crashing
            return False

    async def _propagate_skips(self, execution_id: str, workflow: WorkflowGraph):
        """
        Mark nodes as SKIPPED if they are unreachable.
        Respects wait_for_incoming policy ("all" vs "any").
        """
        statuses = await asyncio.to_thread(self._get_all_statuses, execution_id)

        # Pre-fetch all outputs for completed nodes to avoid N+1 queries
        completed_nodes = [
            nid for nid, status in statuses.items() if status == NodeStatus.COMPLETED.value
        ]
        outputs_cache: dict[str, Any] = {}
        for node_id in completed_nodes:
            outputs_cache[node_id] = await asyncio.to_thread(
                self._get_node_output, execution_id, node_id
            )

        changed = True
        while changed:
            changed = False
            for node in workflow.nodes:
                if statuses.get(node.id) != NodeStatus.PENDING.value:
                    continue

                incoming = [e for e in workflow.edges if e.target == node.id]
                if not incoming:
                    continue

                unreachable_count = 0
                for edge in incoming:
                    src_status = statuses.get(edge.source)
                    is_edge_unreachable = False

                    if src_status in (NodeStatus.SKIPPED.value, NodeStatus.FAILED.value):
                        is_edge_unreachable = True
                    elif src_status == NodeStatus.COMPLETED.value:
                        if edge.condition:
                            output = outputs_cache.get(edge.source)
                            if not self._eval_condition(edge.condition, output):
                                is_edge_unreachable = True

                    if is_edge_unreachable:
                        unreachable_count += 1

                # Determine wait policy (MERGE uses merge_config.wait_for)
                if node.type == NodeType.MERGE and node.merge_config:
                    wait_policy = node.merge_config.wait_for
                else:
                    wait_policy = node.wait_for_incoming

                should_skip = False
                if wait_policy == "all":
                    # If ANY input is unreachable, the node is unreachable
                    if unreachable_count > 0:
                        should_skip = True
                else:  # "any"
                    # ALL inputs must be unreachable
                    if unreachable_count == len(incoming):
                        should_skip = True

                if should_skip:

                    def mark_skipped(conn, exec_id, n_id):
                        self._set_node_status(conn, exec_id, n_id, NodeStatus.SKIPPED)

                    await asyncio.to_thread(
                        self._run_in_transaction, mark_skipped, execution_id, node.id
                    )
                    statuses[node.id] = NodeStatus.SKIPPED.value
                    changed = True

    # ========== Node Type Implementations ==========

    async def _execute_task(self, execution_id: str, node: Node, input_data: dict) -> Any:
        """Execute a role-based task with pre-apply gate safety."""
        config = node.task_config

        # Format task description
        # Use format_map to safely handle arbitrary keys from input_data,
        # including namespaced keys like "node_id.field" which are not valid
        # Python identifiers for **kwargs expansion.
        format_mapping = input_data.copy()
        format_mapping["input"] = json.dumps(input_data, indent=2)
        task_description = config.task_template.format_map(format_mapping)

        # Get workflow_id (wrap in to_thread to avoid blocking)
        workflow_id = await asyncio.to_thread(self._get_workflow_id, execution_id)

        # Use worktree_id if specified to allow context sharing, otherwise node.id
        step_id = config.worktree_id or node.id

        async with self.semaphore:
            coro = asyncio.to_thread(
                self.engine.run_role,
                config.role,
                task_description,
                workflow_id,
                step_id,
                None,
                None,
                None,
                config.gates or None,
                None,
                None,
            )

            if config.timeout:
                result = await asyncio.wait_for(coro, timeout=config.timeout)
            else:
                result = await coro

        return result

    async def _execute_gate(self, execution_id: str, node: Node, input_data: dict) -> Any:
        """Execute a verification gate in isolated worktree."""
        config = node.gate_config

        def run_gate_isolated():
            with self.engine.workspace.isolated_execution(f"gate_{execution_id}_{node.id}") as ctx:
                gate_config = self.gate_loader.get_gate(config.gate_type)
                result = self.gate_executor.run_gate(
                    gate_config, ctx.worktree_path, execution_id, node.id
                )
                return result

        result = await asyncio.to_thread(run_gate_isolated)

        passed = result.status == GateStatus.PASSED
        return {
            "passed": passed,
            "output": result.output,
            "test_status": result.status.value,
        }

    async def _execute_branch(
        self, execution_id: str, workflow: WorkflowGraph, node: Node, input_data: dict
    ) -> Any:
        """
        Execute branch logic with loop control.

        Limitation: This implementation assumes at most ONE outgoing edge from a
        BRANCH node is marked as is_loop_edge=True. If both on_true and on_false
        paths are loop edges (complex multi-loop structures), the behavior is
        undefined. For such cases, use separate BRANCH nodes for each loop.
        """
        config = node.branch_config

        # Find loop edge
        loop_edge = next(
            (e for e in workflow.edges if e.source == node.id and e.is_loop_edge), None
        )
        loop_key = f"loop_{loop_edge.id if loop_edge else node.id}"
        iteration = await asyncio.to_thread(self._get_loop_counter, execution_id, loop_key)

        # Check max iterations - route to non-loop edge
        if iteration >= config.condition.max_iterations:
            # Reset counter for potential re-entry later
            await asyncio.to_thread(self._reset_loop_counter, execution_id, loop_key)
            # Determine non-loop target using is_loop_edge marker
            if loop_edge:
                non_loop_target = (
                    config.on_false if loop_edge.target == config.on_true else config.on_true
                )
            else:
                non_loop_target = config.on_false
            return {
                "branch_outcome": BranchOutcome.MAX_ITERATIONS.value,
                "iterations": iteration,
                "selected_target": non_loop_target,
            }

        # Evaluate condition
        result = self._eval_condition(config.condition, input_data)
        outcome = BranchOutcome.TRUE.value if result else BranchOutcome.FALSE.value

        # Track loop counter: increment when taking loop edge, reset when exiting
        if loop_edge:
            target = config.on_true if result else config.on_false
            if loop_edge.target == target:
                # Taking the loop edge - increment counter
                await asyncio.to_thread(self._increment_loop_counter, execution_id, loop_key)
            else:
                # Exiting the loop - reset counter for potential re-entry
                await asyncio.to_thread(self._reset_loop_counter, execution_id, loop_key)

        return {
            "branch_outcome": outcome,
            "condition_result": result,
            "iterations": iteration,
        }

    def _execute_merge(self, execution_id: str, workflow: WorkflowGraph, node: Node) -> Any:
        """Execute merge logic - combine inputs from multiple branches."""
        config = node.merge_config
        incoming = [e for e in workflow.edges if e.target == node.id]

        merged = {}
        for edge in incoming:
            # Check edge condition before including output
            if edge.condition:
                source_output = self._get_node_output(execution_id, edge.source)
                if not self._eval_condition(edge.condition, source_output):
                    # Skip this edge - condition not met
                    continue
                output = source_output
            else:
                output = self._get_node_output(execution_id, edge.source)

            if output and isinstance(output, dict):
                if config.merge_strategy == "union":
                    merged.update(output)
                elif config.merge_strategy == "first" and not merged:
                    merged = output
                elif config.merge_strategy == "intersection":
                    if not merged:
                        merged = output.copy()
                    else:
                        merged = {k: v for k, v in merged.items() if k in output}

        return merged

    async def _execute_parallel(
        self, execution_id: str, workflow: WorkflowGraph, node: Node
    ) -> Any:
        """
        Execute parallel node - fan-out to multiple branches.

        Design Note: PARALLEL acts as a FORK node that completes immediately,
        allowing all outgoing edges to be followed concurrently. It does NOT
        wait for branches to complete - that's the responsibility of a
        downstream MERGE node.

        The fan-out is determined by outgoing edges in the graph, not the
        branches list in ParallelNodeConfig (which is reserved for future use).

        Pattern: PARALLEL -> [branch1, branch2, ...] -> MERGE
        - PARALLEL: Starts branches (completes immediately)
        - MERGE: Synchronizes branches (waits per merge_config.wait_for)
        """
        # PARALLEL nodes complete immediately - downstream MERGE handles sync
        # The actual parallel execution happens in execute_next_batch
        return {"parallel_started": True, "branches": node.parallel_config.branches}

    async def _execute_human(self, execution_id: str, node: Node, input_data: dict) -> Any:
        """Execute human approval node."""
        config = node.human_config

        # Prepare changes list
        changes = []
        if config.description:
            changes.append(config.description)
        if isinstance(input_data, dict):
            for key, value in input_data.items():
                if not key.startswith("_"):
                    changes.append(f"{key}: {value}")

        # Request approval (blocking call - wrapped in to_thread)
        decision = await asyncio.to_thread(
            self.approval_gate.request_approval,
            feature_id=execution_id,
            title=config.title,
            changes=changes,
            review_summary=json.dumps(input_data, indent=2),
            bridge=None,
            diff_lines=None,
        )

        return {"approved": decision == ApprovalDecision.APPROVE}

    async def _execute_subgraph(self, execution_id: str, node: Node, input_data: dict) -> Any:
        """Execute nested workflow (not yet implemented)."""
        config = node.subgraph_config
        if not config:
            raise ValueError(f"Subgraph node '{node.id}' missing subgraph_config")

        raise NotImplementedError(
            f"Subgraph execution not yet implemented. "
            f"Node '{node.id}' references workflow '{config.workflow_name}'. "
            f"Remove SUBGRAPH nodes from your workflow or wait for future release."
        )
