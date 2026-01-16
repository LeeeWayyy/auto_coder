"""Stateless graph workflow execution engine.

This module implements a crash-resistant, resumable graph orchestrator that:
- Stores all state in the database (no in-memory state)
- Uses transaction-based execution for atomicity
- Supports horizontal scaling via multiple workers
- Implements pre-apply safety with gates
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from supervisor.core.engine import ExecutionEngine

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


class _AttrDict:
    """Wrapper to provide attribute-style access to dict values for format strings."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        value = self._data.get(name, f"<missing:{name}>")
        if isinstance(value, dict):
            return _AttrDict(value)
        return value

    def __str__(self):
        return str(self._data)


class _SafeFormatDict(dict):
    """
    Dict subclass that wraps nested dicts in _AttrDict for attribute-style access.

    This enables format strings like {node_id.field} where node_id is a dict output
    stored by _collect_inputs. Missing keys return a placeholder instead of raising.
    """

    def __getitem__(self, key: str):
        try:
            value = super().__getitem__(key)
            if isinstance(value, dict):
                return _AttrDict(value)
            return value
        except KeyError:
            return f"<missing:{key}>"

    def __missing__(self, key: str):
        return f"<missing:{key}>"


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
        execution_engine: ExecutionEngine,
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
        """Save workflow definition to database.

        Uses ON CONFLICT DO UPDATE instead of REPLACE to preserve FK references
        from graph_executions that reference this workflow.
        """
        conn.execute(
            """
            INSERT INTO graph_workflows (id, name, definition, version, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                definition = excluded.definition,
                version = excluded.version,
                updated_at = CURRENT_TIMESTAMP
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

    def _get_all_outputs(self, execution_id: str) -> dict[str, Any]:
        """Batch fetch all node outputs for an execution (avoids N+1 queries)."""
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, output_data FROM node_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchall()
            return {
                row[0]: json.loads(row[1]) if row[1] else None
                for row in rows
            }

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
            # Use 'is not None' to allow falsy values like 0, False, "", []
            if inputs is not None:
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
            execution_id, workflow.config.max_parallel_nodes
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
        Uses BEGIN IMMEDIATE for atomic claim within the transaction.
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

        # Check if all terminal (wrapped in to_thread to avoid blocking event loop)
        statuses = await asyncio.to_thread(self._get_all_statuses, execution_id)
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
            # Execute based on node type
            # MERGE and PARALLEL handle their own input collection, skip _collect_inputs for them
            if node.type == NodeType.MERGE:
                output = await asyncio.to_thread(self._execute_merge, execution_id, workflow, node)
            elif node.type == NodeType.PARALLEL:
                output = await self._execute_parallel(execution_id, workflow, node)
            else:
                # Collect inputs from completed upstream nodes for other node types
                input_data = await asyncio.to_thread(
                    self._collect_inputs, execution_id, workflow, node
                )
                if node.type == NodeType.TASK:
                    output = await self._execute_task(execution_id, node, input_data)
                elif node.type == NodeType.GATE:
                    output = await self._execute_gate(execution_id, node, input_data)
                elif node.type == NodeType.BRANCH:
                    output = await self._execute_branch(execution_id, workflow, node, input_data)
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
                    # Use guarded update to handle race with loop resets
                    updated = self._set_node_status_guarded(
                        conn, exec_id, n_id, NodeStatus.FAILED, expected_status=NodeStatus.RUNNING
                    )
                    if updated:
                        self._set_node_output(conn, exec_id, n_id, out)
                        # Store error message in a separate update since guarded doesn't take error
                        conn.execute(
                            "UPDATE node_executions SET error=? WHERE execution_id=? AND node_id=?",
                            (f"Gate failed: {out.get('output', 'unknown')}", exec_id, n_id),
                        )
                    return updated

                failed = await asyncio.to_thread(
                    self._run_in_transaction,
                    persist_gate_failure,
                    execution_id,
                    node_id,
                    output,
                )

                if failed and workflow.config.fail_fast:
                    await self._fail_workflow(execution_id, f"Gate '{node_id}' failed")
                return False

            # Persist output and update dependents atomically
            # Use guarded status update to handle race condition where node was
            # reset to PENDING (e.g., by loop restart) while this task was running.
            def persist_completion(conn, exec_id, n_id, out, wf):
                # Guarded update: only complete if still RUNNING
                # This prevents stale completions from overwriting loop resets
                updated = self._set_node_status_guarded(
                    conn, exec_id, n_id, NodeStatus.COMPLETED, expected_status=NodeStatus.RUNNING
                )
                if updated:
                    self._set_node_output(conn, exec_id, n_id, out)
                    self._update_dependents(conn, exec_id, wf, n_id, out)
                return updated

            completed = await asyncio.to_thread(
                self._run_in_transaction,
                persist_completion,
                execution_id,
                node_id,
                output,
                workflow,
            )
            if not completed:
                # Node was reset (e.g., by loop restart) - don't count as success
                logger.info(f"Node {node_id} completion discarded (status changed during execution)")
            return completed

        except Exception as e:
            logger.error(f"Node {node_id} failed: {e}")

            def persist_failure(conn, exec_id, n_id, err):
                # Use guarded update to handle race with loop resets
                updated = self._set_node_status_guarded(
                    conn, exec_id, n_id, NodeStatus.FAILED, expected_status=NodeStatus.RUNNING
                )
                if updated:
                    conn.execute(
                        "UPDATE node_executions SET error=? WHERE execution_id=? AND node_id=?",
                        (err, exec_id, n_id),
                    )
                return updated

            failed = await asyncio.to_thread(
                self._run_in_transaction, persist_failure, execution_id, node_id, str(e)
            )

            if failed and workflow.config.fail_fast:
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
        # Use 'is not None' to allow falsy values like 0, False, "", []
        seeded_input = self._get_seeded_input(execution_id, node.id)
        if seeded_input is not None:
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
                    # Reset COMPLETED, SKIPPED, and RUNNING nodes on loop re-entry
                    # - COMPLETED/SKIPPED: Need to re-execute in the next iteration
                    # - RUNNING: Parallel execution race - task may still be running but
                    #   we need to mark it for re-execution. The running task's completion
                    #   will use guarded status update which will fail (status != RUNNING).
                    if loop_status in (
                        NodeStatus.COMPLETED.value,
                        NodeStatus.SKIPPED.value,
                        NodeStatus.RUNNING.value,
                    ):
                        # Reset to PENDING (not READY) to preserve dependency ordering
                        # Only the loop entry node will be marked READY below
                        self._set_node_status_guarded(
                            conn,
                            execution_id,
                            loop_node_id,
                            NodeStatus.PENDING,
                            expected_status=NodeStatus(loop_status),
                        )
                        self._clear_node_output(conn, execution_id, loop_node_id)

                # CRITICAL: Also reset the BRANCH node itself so it can be re-evaluated
                # on subsequent iterations. Without this, the BRANCH stays COMPLETED
                # and loops only execute once.
                branch_status = self._get_node_status_in_txn(conn, execution_id, completed_id)
                if branch_status == NodeStatus.COMPLETED.value:
                    self._set_node_status_guarded(
                        conn,
                        execution_id,
                        completed_id,
                        NodeStatus.PENDING,
                        expected_status=NodeStatus.COMPLETED,
                    )
                    self._clear_node_output(conn, execution_id, completed_id)

                # Mark only the loop entry node as READY to start the iteration
                entry_status = self._get_node_status_in_txn(conn, execution_id, edge.target)
                if entry_status == NodeStatus.PENDING.value:
                    self._set_node_status_guarded(
                        conn,
                        execution_id,
                        edge.target,
                        NodeStatus.READY,
                        expected_status=NodeStatus.PENDING,
                    )
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
        # Use deque for O(1) popleft operations instead of list.pop(0) which is O(n)
        can_reach_end: set[str] = set()
        queue: deque[str] = deque([end_node])
        while queue:
            node = queue.popleft()
            if node in can_reach_end:
                continue
            can_reach_end.add(node)
            for pred in rev_adj.get(node, []):
                if pred not in can_reach_end:
                    queue.append(pred)

        # Step 2: Forward BFS from start_node, only including nodes that can reach end
        visited: set[str] = set()
        queue = deque([start_node])
        loop_nodes: list[str] = []

        while queue:
            node = queue.popleft()
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

        # 3. Un-namespaced convenience match with ambiguity detection
        # Raises error if multiple upstream nodes provide the same field name
        # to prevent subtle bugs from non-deterministic matching.
        if not found and "." not in condition.field:
            suffix = f".{condition.field}"
            matching_keys = [key for key in data if key.endswith(suffix)]
            if len(matching_keys) > 1:
                raise ValueError(
                    f"Ambiguous field '{condition.field}'. Found in multiple sources: "
                    f"{sorted(matching_keys)}. Use a fully-qualified field name like 'source_node.field'."
                )
            if len(matching_keys) == 1:
                # Warn about implicit suffix matching - could break if upstream changes
                logger.warning(
                    f"Condition field '{condition.field}' matched via suffix to "
                    f"'{matching_keys[0]}'. Consider using fully-qualified name for stability."
                )
                value = data[matching_keys[0]]
                found = True

        # If field was not found, condition cannot match.
        # This prevents typos or missing outputs from accidentally passing conditions
        # like "status != 'failed'" when status is absent (None != 'failed' would be True).
        if not found:
            return False

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

        # Batch fetch all outputs in a single query (avoids N+1)
        outputs_cache = await asyncio.to_thread(self._get_all_outputs, execution_id)

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
                        # Use guarded update to prevent race condition where node was
                        # claimed as READY/RUNNING by another worker between our snapshot
                        # and this update. Only transition PENDING -> SKIPPED.
                        return self._set_node_status_guarded(
                            conn, exec_id, n_id, NodeStatus.SKIPPED, expected_status=NodeStatus.PENDING
                        )

                    updated = await asyncio.to_thread(
                        self._run_in_transaction, mark_skipped, execution_id, node.id
                    )
                    if updated:
                        statuses[node.id] = NodeStatus.SKIPPED.value
                        changed = True
                    # If not updated, node was already claimed - will be handled on next pass

    # ========== Node Type Implementations ==========

    async def _execute_task(self, execution_id: str, node: Node, input_data: dict) -> Any:
        """
        Execute a role-based task with pre-apply gate safety.

        Timeout Limitation (Known Issue):
            When a timeout is configured, asyncio.wait_for cancels the coroutine
            but cannot forcefully stop the underlying thread. The blocking run_role
            call may continue executing in the background.

            Current mitigations:
            - The node is immediately marked as FAILED (no result saved to graph state)
            - A clear TimeoutError is raised with node context
            - The workflow proceeds without waiting for the orphaned thread

            Potential side effects:
            - The timed-out task may still write to files in the working tree
            - This can leave the working tree inconsistent with graph state
            - Downstream nodes may see partial/unexpected file changes

            Recommended solutions (not yet implemented):
            - Use subprocess isolation with SIGKILL for strict timeout enforcement
            - Pass a cancellation token to run_role for cooperative cancellation
            - Run tasks in isolated worktrees that can be discarded on timeout
        """
        config = node.task_config

        # Format task description using a dictionary that supports dot-notation keys.
        # The `_collect_inputs` method creates flattened keys like "node_id.field"
        # which are then used by `format_map`. The `_SafeFormatDict` handles missing
        # keys gracefully by returning a placeholder instead of raising KeyError.
        format_mapping = _SafeFormatDict(input_data)
        format_mapping["input"] = json.dumps(input_data, indent=2)
        task_description = config.task_template.format_map(format_mapping)

        # Get workflow_id (wrap in to_thread to avoid blocking)
        workflow_id = await asyncio.to_thread(self._get_workflow_id, execution_id)

        # Namespace step_id with execution_id AND a unique suffix to avoid:
        # 1. Cross-run state corruption (rerunning same workflow)
        # 2. Zombie task collision (timed-out task vs. re-execution in loop)
        # The UUID ensures each task execution has isolated workspace even if
        # the same node is re-executed while a timed-out thread is still running.
        base_step_id = config.worktree_id or node.id
        unique_suffix = str(uuid.uuid4())[:8]
        step_id = f"{execution_id}_{base_step_id}_{unique_suffix}"

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
                try:
                    result = await asyncio.wait_for(coro, timeout=config.timeout)
                except TimeoutError:
                    # Log a warning about the zombie task - the thread may still be running
                    # and could modify files even though this execution is marked as failed
                    logger.warning(
                        f"ZOMBIE TASK WARNING: Task '{node.id}' timed out after {config.timeout}s. "
                        f"The underlying thread (step_id={step_id}) may still be running in the "
                        f"background and could modify files. Consider using isolated worktrees."
                    )
                    # Re-raise with context
                    raise TimeoutError(
                        f"Task '{node.id}' timed out after {config.timeout}s. "
                        f"Note: The underlying thread may still be running."
                    )
            else:
                result = await coro

        return result

    async def _execute_gate(self, execution_id: str, node: Node, input_data: dict) -> Any:
        """
        Execute a verification gate in isolated worktree.

        Known Limitation:
            Gates run in a fresh worktree created from HEAD, which does NOT include
            uncommitted changes from prior TASK nodes. This means:
            - Gates verify the committed state, not the current working tree
            - A gate may pass while uncommitted changes would actually fail it
            - This is by design for isolation, but can be surprising

            Workarounds:
            - Use TASK nodes with gates=[...] to run gates in the same worktree
            - Have TASK nodes commit changes before GATE verification
            - For Phase 2+, consider adding a "use_main_worktree" option
        """
        config = node.gate_config

        # Get workflow_id for consistent event/artifact tracking
        workflow_id = await asyncio.to_thread(self._get_workflow_id, execution_id)

        def run_gate_isolated():
            with self.engine.workspace.isolated_execution(f"gate_{execution_id}_{node.id}") as ctx:
                gate_config = self.gate_loader.get_gate(config.gate_type)
                result = self.gate_executor.run_gate(
                    gate_config, ctx.worktree_path, workflow_id, node.id
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

    def _get_completion_order(self, execution_id: str, node_ids: list[str]) -> list[str]:
        """
        Get node IDs sorted by their completion timestamp.

        Returns nodes in the order they actually completed, which is essential
        for merge_strategy=first to honor the chronological completion order
        rather than the edge definition order.
        """
        with self.db._connect() as conn:
            placeholders = ",".join("?" * len(node_ids))
            rows = conn.execute(
                f"""
                SELECT node_id, completed_at FROM node_executions
                WHERE execution_id=? AND node_id IN ({placeholders})
                AND completed_at IS NOT NULL
                ORDER BY completed_at ASC
                """,
                (execution_id, *node_ids),
            ).fetchall()
            return [row[0] for row in rows]

    def _execute_merge(self, execution_id: str, workflow: WorkflowGraph, node: Node) -> Any:
        """
        Execute merge logic - combine inputs from multiple branches.

        merge_strategy behavior:
        - "union": Merge all outputs, later values overwrite earlier
        - "first": First branch to complete with non-empty output wins
        - "intersection": Only keys present in all outputs are kept

        Note: Empty outputs ({}) are skipped for all strategies. For "first",
        this means the first branch with actual data wins, not just the first
        to complete. This prevents empty/failed branches from blocking results.
        """
        config = node.merge_config
        incoming = [e for e in workflow.edges if e.target == node.id]

        # For merge_strategy=first, sort edges by actual completion timestamp
        # so the first branch to complete (with non-empty output) wins
        if config.merge_strategy == "first":
            source_ids = [e.source for e in incoming]
            completion_order = self._get_completion_order(execution_id, source_ids)
            # Create a map for sorting
            order_map = {nid: idx for idx, nid in enumerate(completion_order)}
            # Sort edges by completion order (edges with no completion go last)
            incoming = sorted(incoming, key=lambda e: order_map.get(e.source, float("inf")))

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
