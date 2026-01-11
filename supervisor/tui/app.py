"""Terminal User Interface for supervisor workflow control.

FIX (Gemini/Codex review v3): SYNCHRONOUS design using Rich Live display.
Workflow runs in background thread, communicates via InteractionBridge.

USES EXISTING:
- EventType.APPROVAL_REQUESTED - Triggered when approval needed
- EventType.APPROVAL_GRANTED - User approves
- EventType.APPROVAL_DENIED - User rejects
- db.get_phases(feature_id) - EXISTING DB method
- db.get_components(feature_id) - EXISTING DB method
- db.get_events(...) - EXISTING DB method
- ComponentStatus.COMPLETED - CORRECT enum value (not COMPLETE)
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

from supervisor.core.state import Database
from supervisor.core.models import ComponentStatus, FeatureStatus
from supervisor.core.interaction import ApprovalDecision, ApprovalRequest, InteractionBridge

logger = logging.getLogger(__name__)


class SupervisorTUI:
    """Main TUI application for supervisor.

    FIX (Gemini review): SYNCHRONOUS - runs Rich Live in main thread,
    workflow in background thread via InteractionBridge.

    FIX (v12): Accept external bridge instead of creating own instance.
    This allows CLI to share same bridge with WorkflowCoordinator.

    USAGE:
        # Create shared bridge
        bridge = InteractionBridge()

        # Pass same bridge to coordinator AND TUI
        coordinator = WorkflowCoordinator(..., interaction_bridge=bridge)
        tui = SupervisorTUI(db, bridge=bridge)

        tui.run_with_workflow(
            workflow_fn=lambda: coordinator.run_implementation(feature_id),
            feature_id=feature_id,
        )
    """

    def __init__(self, db: Database, bridge: "InteractionBridge | None" = None):
        self.db = db
        self.console = Console()
        # FIX (v12): Use injected bridge or create default for standalone testing
        self.bridge = bridge if bridge is not None else InteractionBridge()
        self._running = False
        self._workflow_thread: threading.Thread | None = None
        self._current_feature_id: str | None = None

    def run_with_workflow(
        self,
        workflow_fn: Callable[[], None],
        feature_id: str,
    ) -> None:
        """Run TUI with workflow executing in background thread.

        Args:
            workflow_fn: Function to run workflow (e.g., coordinator.run_implementation)
            feature_id: Feature being executed (for status display)
        """
        self._running = True
        self._current_feature_id = feature_id

        # Start workflow in background thread
        self._workflow_thread = threading.Thread(
            target=self._run_workflow_wrapper,
            args=(workflow_fn,),
            daemon=True,
        )
        self._workflow_thread.start()

        # Run TUI in main thread (blocking)
        try:
            self._run_tui_loop()
        finally:
            self._running = False

    def _run_workflow_wrapper(self, workflow_fn: Callable[[], None]) -> None:
        """Wrapper to run workflow and catch exceptions."""
        try:
            workflow_fn()
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
        finally:
            self._running = False  # Signal TUI to stop

    def _run_tui_loop(self) -> None:
        """Main TUI loop - runs in main thread."""
        with Live(self._generate_layout(), refresh_per_second=4, console=self.console) as live:
            while self._running:
                live.update(self._generate_layout())

                # Check for pending approvals via bridge (NON-BLOCKING)
                pending = self.bridge.get_pending_requests()
                for request in pending:
                    # Pause Live display for user input
                    live.stop()
                    decision = self._handle_approval_sync(request)
                    self.bridge.submit_decision(request.gate_id, decision)
                    live.start()

                time.sleep(0.25)  # SYNC sleep, not async

    def _generate_layout(self) -> Layout:
        """Generate the TUI layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["header"].update(self._render_header())
        layout["body"].split_row(
            Layout(name="status", ratio=1),
            Layout(name="details", ratio=2),
        )
        layout["status"].update(self._render_workflow_status())
        layout["details"].update(self._render_details())
        layout["footer"].update(self._render_footer())

        return layout

    def _render_header(self) -> Panel:
        """Render header with title and stats."""
        return Panel(
            "[bold blue]Supervisor[/bold blue] - AI Workflow Orchestrator",
            style="blue",
        )

    def _render_workflow_status(self) -> Panel:
        """Render current workflow status.

        FIX (Codex review): Use EXISTING db.get_phases() and db.get_components()
        """
        tree = Tree("[bold]Active Workflows[/bold]")

        if self._current_feature_id:
            feature = self.db.get_feature(self._current_feature_id)
            if feature:
                feature_node = tree.add(f"[cyan]{feature.id}[/cyan]: {feature.title}")

                # FIX: Use EXISTING db.get_phases(feature_id)
                phases = self.db.get_phases(feature.id)
                for phase in phases:
                    status_icon = self._status_icon(phase.status)
                    phase_node = feature_node.add(f"{status_icon} {phase.title}")

                    # FIX: Use db.get_components(feature_id) and filter by phase
                    all_components = self.db.get_components(feature.id)
                    phase_components = [c for c in all_components if c.phase_id == phase.id]
                    # FIX: Use ComponentStatus.COMPLETED (not COMPLETE)
                    completed = sum(1 for c in phase_components if c.status == ComponentStatus.COMPLETED)
                    total = len(phase_components)
                    phase_node.add(f"[dim]{completed}/{total} components[/dim]")

        return Panel(tree, title="Workflow Status")

    def _render_details(self) -> Panel:
        """Render details panel (logs when no approval pending)."""
        return self._render_logs_panel()

    def _render_logs_panel(self) -> Panel:
        """Render recent logs panel.

        FIX (Codex v3): get_events requires workflow_id, has no limit param.
        FIX (v8): PERFORMANCE - Python slice acceptable for MVP since events
        list is bounded by workflow duration. For very long workflows, consider
        adding Database.get_recent_events(workflow_id, limit) with SQL LIMIT.
        """
        table = Table(show_header=True, header_style="bold")
        table.add_column("Time", style="dim")
        table.add_column("Type")
        table.add_column("Details")

        # FIX (Codex v3): get_events(workflow_id) is the actual signature
        # FIX (v8): Fetches all then slices - OK for MVP, SQL LIMIT for scale
        if self._current_feature_id:
            events = self.db.get_events(workflow_id=self._current_feature_id)
            # Show last 20 events (most recent) - Python slice for simplicity
            for event in events[-20:]:
                table.add_row(
                    event.timestamp.strftime("%H:%M:%S") if event.timestamp else "",
                    event.event_type.value,
                    str(event.payload)[:50] + "..." if len(str(event.payload)) > 50 else str(event.payload),
                )

        return Panel(table, title="Recent Events")

    def _status_icon(self, status) -> str:
        """Get icon for status.

        FIX (Codex v3): Map "completed" not "complete" to match ComponentStatus.COMPLETED.
        FIX (v11): Use status.value for Enum types, not str(status).lower().
        """
        status_icons = {
            "pending": "⏳",
            "in_progress": "🔄",
            "implementing": "🔧",
            "testing": "🧪",
            "review": "👀",
            "completed": "✅",  # FIX: was "complete"
            "failed": "❌",
        }
        # FIX (v11): Handle Enum types properly - use .value not str()
        status_key = status.value if hasattr(status, 'value') else str(status).lower()
        return status_icons.get(status_key, "❓")

    def _render_footer(self) -> Panel:
        """Render footer with help."""
        return Panel(
            "[Q]uit  [P]ause  [R]esume  [C]ancel  [?]Help",
            style="dim",
        )

    def _handle_approval_sync(self, request: ApprovalRequest) -> ApprovalDecision:
        """Handle interactive approval request. SYNCHRONOUS.

        FIX (Gemini review): Changed from async to sync method.
        Called from main thread when Live display is paused.
        """
        self.console.print()
        self.console.rule("[bold]Approval Required[/bold]")

        # Show details
        self.console.print(f"\n[bold]Feature:[/bold] {request.feature_id}")
        self.console.print(f"[bold]Gate:[/bold] {request.gate_id}")
        self.console.print(f"[bold]Risk:[/bold] [{self._risk_color(request.risk_level)}]{request.risk_level}[/]")

        self.console.print("\n[bold]Changes:[/bold]")
        for change in request.changes:
            self.console.print(f"  • {change}")

        self.console.print(f"\n[bold]Review Summary:[/bold]\n{request.review_summary}")

        # Get decision (Rich Prompt is blocking, which is fine in main thread)
        # FIX (v27 - Gemini PR review): Removed 'd' for unimplemented EDIT feature
        choice = Prompt.ask(
            "\n[bold]Decision[/bold]",
            choices=["a", "r", "s"],
            default="a",
        )

        decision_map = {
            "a": ApprovalDecision.APPROVE,
            "r": ApprovalDecision.REJECT,
            "s": ApprovalDecision.SKIP,
        }

        decision = decision_map.get(choice, ApprovalDecision.SKIP)

        # Handle rejection reason
        if decision == ApprovalDecision.REJECT:
            reason = Prompt.ask("[bold]Rejection reason[/bold]")
            logger.info(f"Approval rejected: {reason}")

        return decision

    # NOTE (v8): Async request_approval REMOVED - use InteractionBridge.request_approval() instead
    # The workflow thread calls bridge.request_approval() which is SYNC (blocks on Event)
    # TUI main thread polls bridge.get_pending_requests() and calls bridge.submit_decision()

    # NOTE (v8): Duplicate _status_icon REMOVED - defined once in _render_progress_panel
    # The correct mapping uses "completed" (not "complete") to match ComponentStatus enum

    def _risk_color(self, risk: str) -> str:
        """Get color for risk level."""
        colors = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
            "critical": "bold red",
        }
        return colors.get(risk.lower(), "white")

    def stop(self) -> None:
        """Stop the TUI."""
        self._running = False
