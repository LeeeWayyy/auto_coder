"""Live execution monitoring for workflows.

Provides real-time terminal display of workflow execution progress.
"""

import asyncio
import json
from typing import Any, Dict, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from supervisor.cli_ui.graph_renderer import StatusTableRenderer, TerminalGraphRenderer
from supervisor.core.graph_engine import GraphOrchestrator
from supervisor.core.graph_schema import WorkflowGraph
from supervisor.core.state import Database


class LiveExecutionMonitor:
    """
    Real-time terminal UI for monitoring workflow execution.

    Features:
    - Live-updating graph visualization
    - Progress bars for overall completion
    - Node status table
    - Log stream

    Design Notes:
    - Uses asyncio.to_thread for DB queries to avoid blocking event loop
    - Batches DB queries into single call to reduce overhead
    - Reuses Progress widget to avoid flickering
    - Uses public orchestrator methods from Phase 1 (already implemented)
    - Polls at 1 second intervals to balance responsiveness and load
    - Breaks out after MAX_DB_ERRORS consecutive DB failures to prevent infinite hang
    """

    MAX_DB_ERRORS = 5  # Break out after this many consecutive DB errors

    def __init__(self, orchestrator: GraphOrchestrator, db: Database):
        self.orchestrator = orchestrator
        self.db = db
        self.console = Console()
        self.graph_renderer = TerminalGraphRenderer(self.console)
        self.status_renderer = StatusTableRenderer(self.console)
        # Reuse progress widget to avoid recreating each refresh
        self._progress: Progress | None = None
        self._progress_task_id: int | None = None
        self._cancelled = False  # Flag for external cancellation

    def create_layout(self) -> Layout:
        """Create the terminal layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5),
        )

        layout["main"].split_row(
            Layout(name="graph", ratio=1), Layout(name="status", ratio=1)
        )

        return layout

    def cancel(self):
        """Signal the monitor to stop. Called when worker completes/fails."""
        self._cancelled = True

    async def monitor(self, execution_id: str, workflow: WorkflowGraph):
        """
        Monitor execution with live updates.

        NOTE: DB queries are batched and wrapped in asyncio.to_thread to avoid
        blocking the event loop during concurrent worker execution.

        Stops when:
        - Execution status is terminal (completed/failed/cancelled)
        - cancel() is called (worker finished)
        - MAX_DB_ERRORS consecutive DB failures
        """
        layout = self.create_layout()
        consecutive_db_errors = 0

        # Initialize reusable progress bar
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        self._progress_task_id = self._progress.add_task(
            "Nodes: 0/0", total=len(workflow.nodes)
        )

        with Live(layout, console=self.console, refresh_per_second=2) as live:
            while not self._cancelled:
                # Batch all DB queries into single call to reduce overhead
                statuses, outputs, exec_status, had_db_error = await asyncio.to_thread(
                    self._get_execution_snapshot, execution_id
                )

                # Track DB errors - break out if too many consecutive failures
                # NOTE: Only count actual exceptions, not missing rows (normal during startup)
                if had_db_error:
                    consecutive_db_errors += 1
                    if consecutive_db_errors >= self.MAX_DB_ERRORS:
                        self.console.print(
                            "[red]Monitor stopping: too many DB errors[/]"
                        )
                        break
                else:
                    consecutive_db_errors = 0  # Reset on success

                # Update header
                # SECURITY: Escape workflow name to prevent Rich markup injection
                safe_name = escape(workflow.name)
                if exec_status == "running":
                    header_text = f"[bold blue]⟳ Executing:[/] {safe_name}"
                elif exec_status == "completed":
                    header_text = f"[bold green]✓ Completed:[/] {safe_name}"
                elif exec_status == "failed":
                    header_text = f"[bold red]✗ Failed:[/] {safe_name}"
                else:
                    header_text = f"[bold]{safe_name}[/]"

                layout["header"].update(Panel(header_text, style="bold"))

                # Update graph
                graph_tree = self.graph_renderer.render_as_tree(workflow, statuses)
                layout["graph"].update(Panel(graph_tree, title="Workflow Graph"))

                # Update status table
                status_table = self.status_renderer.render_status_table(
                    workflow, execution_id, statuses, outputs
                )
                layout["status"].update(Panel(status_table, title="Node Status"))

                # Update footer with progress (reuse widget to avoid flicker)
                completed = sum(
                    1 for s in statuses.values() if s in ["completed", "skipped"]
                )
                total = len(workflow.nodes)
                self._progress.update(
                    self._progress_task_id,
                    completed=completed,
                    description=f"Nodes: {completed}/{total}",
                )
                layout["footer"].update(Panel(self._progress, title="Progress"))

                # Check if done
                if exec_status in ["completed", "failed", "cancelled"]:
                    await asyncio.sleep(1)  # Show final state
                    break

                # Poll at 1 second intervals to balance responsiveness and DB load
                await asyncio.sleep(1.0)

    def _get_execution_snapshot(
        self, execution_id: str
    ) -> Tuple[Dict[str, str], Dict[str, Any], str, bool]:
        """
        Get complete execution snapshot in a single DB transaction.

        Batches statuses, outputs, and execution status into one call to reduce
        DB overhead and ensure consistent state.

        NOTE: This method is called from asyncio.to_thread() to avoid
        blocking the event loop. Do not call directly from async code.

        Returns:
            Tuple of (statuses, outputs, exec_status, had_db_error)
            - had_db_error is True only when an actual exception occurred,
              not when execution row is simply missing (normal during startup race)
        """
        statuses: Dict[str, str] = {}
        outputs: Dict[str, Any] = {}
        exec_status = "pending"  # Default to pending, not unknown
        had_db_error = False

        try:
            with self.db._connect() as conn:
                # Get node statuses and TRUNCATED outputs in single query
                # Use substr() to limit output_data to 100 chars for performance
                # (StatusTableRenderer only displays ~40 chars anyway)
                rows = conn.execute(
                    "SELECT node_id, status, substr(output_data, 1, 100) FROM node_executions WHERE execution_id=?",
                    (execution_id,),
                ).fetchall()

                for row in rows:
                    node_id, status, output_summary = row
                    statuses[node_id] = status
                    # Parse JSON with error handling (may be truncated/invalid)
                    # NOTE: Catch multiple error types:
                    # - JSONDecodeError: invalid/truncated JSON
                    # - TypeError: json.loads(None) or non-string types
                    # - UnicodeDecodeError: non-UTF-8 bytes from SQLite BLOB/adapters
                    if output_summary:
                        try:
                            outputs[node_id] = json.loads(output_summary)
                        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                            # Truncated JSON or invalid - show summary as string
                            # NOTE: Normalize to string before truncation - SQLite adapters
                            # may return bytes, and "bytes[:40] + '...'" raises TypeError
                            if isinstance(output_summary, (bytes, bytearray)):
                                summary_str = output_summary.decode(
                                    "utf-8", errors="replace"
                                )
                            else:
                                summary_str = str(output_summary)
                            outputs[node_id] = (
                                summary_str[:40] + "..."
                                if len(summary_str) > 40
                                else summary_str
                            )
                    else:
                        outputs[node_id] = None

                # Get execution status
                # NOTE: Missing row is NOT an error - can happen during startup race
                # when worker hasn't created the execution row yet
                exec_row = conn.execute(
                    "SELECT status FROM graph_executions WHERE id=?",
                    (execution_id,),
                ).fetchone()
                if exec_row:
                    exec_status = exec_row[0]
                # else: keep default "pending" - not an error

        except Exception as e:
            # Log error but don't crash the monitor
            # SECURITY: Escape exception message to prevent Rich markup injection
            self.console.print(f"[dim]DB error: {escape(str(e))}[/]")
            had_db_error = True

        return statuses, outputs, exec_status, had_db_error
