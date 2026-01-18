"""CLI entry point for the Supervisor orchestrator.

Commands:
- supervisor init: Initialize project for supervisor
- supervisor plan: Run planner on a task
- supervisor run: Run a workflow
- supervisor workflow: Execute a feature workflow (Phase 5)
- supervisor metrics: View performance metrics (Phase 5)
- supervisor status: Show current workflow status
- supervisor roles: List available roles
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pydantic
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from supervisor.core.approval import ApprovalPolicy

from supervisor.cli_ui.graph_renderer import StatusTableRenderer, TerminalGraphRenderer
from supervisor.cli_ui.live_monitor import LiveExecutionMonitor
from supervisor.cli_ui.node_inspector import NodeInspector
from supervisor.core.engine import ExecutionEngine
from supervisor.core.graph_engine import GraphOrchestrator
from supervisor.core.graph_schema import WorkflowGraph
from supervisor.core.roles import RoleLoader
from supervisor.core.state import Database
from supervisor.core.worker import WorkflowWorker
from supervisor.metrics.aggregator import MetricsAggregator
from supervisor.metrics.dashboard import MetricsDashboard

console = Console()


def get_repo_path() -> Path:
    """Get the repository path (current directory)."""
    return Path.cwd()


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Supervisor - AI CLI Orchestrator.

    Treats AI CLIs (Claude, Codex, Gemini) as stateless workers
    to prevent context dilution in long workflows.
    """
    pass


@main.command()
def init() -> None:
    """Initialize project for supervisor."""
    repo_path = get_repo_path()
    supervisor_dir = repo_path / ".supervisor"

    if supervisor_dir.exists():
        console.print("[yellow]Project already initialized[/yellow]")
        return

    # Create directory structure
    (supervisor_dir / "roles").mkdir(parents=True)
    (supervisor_dir / "templates").mkdir(parents=True)

    # Create default config
    config_path = supervisor_dir / "config.yaml"
    config_path.write_text(
        """# Supervisor configuration for this project
# See: https://github.com/auto-coder/supervisor

# Default CLI to use
default_cli: claude

# Workflow settings
workflow:
  parallel_execution: false
  require_tests: true
  human_approval: true

# Role overrides for this project
roles: {}

# Git settings
git:
  auto_branch: false
  branch_pattern: "ai/{task_id}-{description}"
"""
    )

    # Phase 5: Create timeout/limits config
    limits_yaml = supervisor_dir / "limits.yaml"
    if not limits_yaml.exists():
        limits_yaml.write_text(
            """# Timeout and resource limits configuration
workflow_timeout: 3600  # Total workflow timeout (seconds)
component_timeout: 300  # Per-component timeout (seconds)
role_timeouts:
  planner: 600
  implementer: 300
  reviewer: 180
"""
        )

    # Phase 5: Create adaptive config
    adaptive_yaml = supervisor_dir / "adaptive.yaml"
    if not adaptive_yaml.exists():
        adaptive_yaml.write_text(
            """# Adaptive model selection configuration
adaptive:
  enabled: true
  min_samples_before_adapt: 10
  recalculation_interval: 10
  exploration_rate: 0.1
  score_weights:
    success_rate: 0.6
    avg_duration: 0.4
"""
        )

    # Phase 5: Create approval policy config
    approval_yaml = supervisor_dir / "approval.yaml"
    if not approval_yaml.exists():
        approval_yaml.write_text(
            """# Approval policy configuration
approval:
  auto_approve_low_risk: true
  risk_threshold: medium  # low, medium, high, critical
  require_approval_for:
    - deploy
    - commit
"""
        )

    # Initialize database
    Database(supervisor_dir / "state.db")

    console.print(
        Panel(
            "[green]Project initialized![/green]\n\n"
            f"Created: {supervisor_dir}\n"
            "- config.yaml: Project configuration\n"
            "- limits.yaml: Timeout configuration (Phase 5)\n"
            "- adaptive.yaml: Adaptive routing configuration (Phase 5)\n"
            "- approval.yaml: Approval policy configuration (Phase 5)\n"
            "- roles/: Custom role definitions\n"
            "- state.db: Workflow state database",
            title="Supervisor Initialized",
        )
    )


@main.command()
@click.argument("task", required=True)
@click.option("--workflow-id", "-w", help="Workflow ID (auto-generated if not provided)")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
def plan(task: str, workflow_id: str | None, dry_run: bool) -> None:
    """Run the planner on a task.

    TASK is a description of what you want to build.

    Example:
        supervisor plan "Add user authentication with JWT"
    """
    repo_path = get_repo_path()
    workflow_id = workflow_id or f"wf-{uuid.uuid4().hex[:8]}"

    console.print(f"\n[bold]Planning task:[/bold] {task}")
    console.print(f"[dim]Workflow ID: {workflow_id}[/dim]\n")

    if dry_run:
        console.print("[yellow]Dry run - would execute planner role[/yellow]")
        return

    try:
        engine = ExecutionEngine(repo_path)
        result = engine.run_plan(task, workflow_id)

        # Display results
        console.print(Panel("[green]Planning complete![/green]", title="Status"))

        if hasattr(result, "phases"):
            table = Table(title="Planned Phases")
            table.add_column("Phase", style="cyan")
            table.add_column("Components", style="green")

            for phase in result.phases:
                components = ", ".join(
                    c.get("title", c.get("id", "?")) for c in phase.get("components", [])
                )
                table.add_row(phase.get("title", phase.get("id", "?")), components)

            console.print(table)

        if hasattr(result, "risks") and result.risks:
            console.print("\n[yellow]Risks identified:[/yellow]")
            for risk in result.risks:
                console.print(f"  - {risk}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("role")
@click.argument("task")
@click.option("--workflow-id", "-w", help="Workflow ID")
@click.option("--target", "-t", multiple=True, help="Target files to focus on")
def run(role: str, task: str, workflow_id: str | None, target: tuple[str, ...]) -> None:
    """Run a specific role on a task.

    ROLE is the role name (planner, implementer, reviewer, etc.)
    TASK is the task description.

    Example:
        supervisor run implementer "Add login endpoint" -t src/api/auth.py
    """
    repo_path = get_repo_path()
    workflow_id = workflow_id or f"wf-{uuid.uuid4().hex[:8]}"

    console.print(f"\n[bold]Running role:[/bold] {role}")
    console.print(f"[bold]Task:[/bold] {task}")
    console.print(f"[dim]Workflow ID: {workflow_id}[/dim]\n")

    try:
        engine = ExecutionEngine(repo_path)
        result = engine.run_role(
            role_name=role,
            task_description=task,
            workflow_id=workflow_id,
            target_files=list(target) if target else None,
        )

        console.print(Panel("[green]Role completed![/green]", title="Status"))

        # Display result based on type
        if hasattr(result, "status"):
            console.print(f"[bold]Status:[/bold] {result.status}")

        if hasattr(result, "files_modified") and result.files_modified:
            console.print("\n[bold]Files modified:[/bold]")
            for f in result.files_modified:
                console.print(f"  - {f}")

        if hasattr(result, "files_created") and result.files_created:
            console.print("\n[bold]Files created:[/bold]")
            for f in result.files_created:
                console.print(f"  - {f}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


# FIX (v27 - Gemini PR review): Helper functions for config loading
def _load_approval_config(repo_path: Path) -> ApprovalPolicy | None:
    """Load approval policy from .supervisor/approval.yaml."""
    from supervisor.core.approval import ApprovalPolicy

    config_path = repo_path / ".supervisor/approval.yaml"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)
        approval_cfg = config.get("approval", {})
        return ApprovalPolicy(
            auto_approve_low_risk=approval_cfg.get("auto_approve_low_risk", True),
            risk_threshold=approval_cfg.get("risk_threshold", "medium"),
            require_approval_for=approval_cfg.get("require_approval_for", ["deploy", "commit"]),
        )


def _load_limits_config(repo_path: Path) -> tuple[dict[str, float], float, float]:
    """Load timeout limits from .supervisor/limits.yaml.

    FIX (v27 - Codex PR review): Also load workflow_timeout from config.

    Returns:
        Tuple of (role_timeouts dict, component_timeout, workflow_timeout)
    """
    config_path = repo_path / ".supervisor/limits.yaml"
    role_timeouts: dict[str, float] = {}
    component_timeout = 300.0
    workflow_timeout = 3600.0

    if config_path.exists():
        with open(config_path) as f:
            limits = yaml.safe_load(f)
            role_timeouts = limits.get("role_timeouts", {})
            component_timeout = limits.get("component_timeout", 300.0)
            workflow_timeout = limits.get("workflow_timeout", 3600.0)

    return role_timeouts, component_timeout, workflow_timeout


@main.command()
@click.argument("feature_id")
@click.option("--tui", is_flag=True, help="Run with interactive TUI")
@click.option("--parallel/--sequential", default=True, help="Parallel or sequential execution")
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Workflow timeout in seconds (default: from limits.yaml or 3600)",
)
def workflow(feature_id: str, tui: bool, parallel: bool, timeout: int | None) -> None:
    """Execute a feature workflow (Phase 5).

    FEATURE_ID is the ID of the feature to execute.

    Example:
        supervisor workflow feat-12345678 --tui
    """
    repo_path = get_repo_path()
    db_path = repo_path / ".supervisor/state.db"

    if not db_path.exists():
        console.print("[yellow]No supervisor database found. Run 'supervisor init' first.[/yellow]")
        return

    db = Database(db_path)

    feature = db.get_feature(feature_id)
    if not feature:
        console.print(f"[red]Error:[/red] Feature '{feature_id}' not found")
        sys.exit(1)

    console.print(f"\n[bold]Executing workflow:[/bold] {feature_id}")
    console.print(f"[dim]Mode: {'TUI' if tui else 'CLI'}, Parallel: {parallel}[/dim]\n")

    try:
        from supervisor.core.approval import ApprovalGate
        from supervisor.core.interaction import InteractionBridge
        from supervisor.core.workflow import WorkflowCoordinator

        engine = ExecutionEngine(repo_path)

        # FIX (v27 - Gemini PR review): Use helper functions for config loading
        policy = _load_approval_config(repo_path)
        role_timeouts, component_timeout, config_workflow_timeout = _load_limits_config(repo_path)

        # FIX (v27 - Codex PR review): Use config value unless --timeout was explicitly provided
        # FIX (v27 - Codex PR review): Use None default to allow explicit override to 3600
        effective_timeout = config_workflow_timeout if timeout is None else float(timeout)

        bridge = InteractionBridge() if tui else None
        approval_gate = ApprovalGate(db, policy=policy)

        coordinator = WorkflowCoordinator(
            engine=engine,
            db=db,
            repo_path=repo_path,
            workflow_timeout=effective_timeout,
            component_timeout=component_timeout,
            role_timeouts=role_timeouts,
            approval_gate=approval_gate,
            interaction_bridge=bridge,
        )

        if tui:
            from supervisor.tui.app import SupervisorTUI

            tui_app = SupervisorTUI(db, bridge=bridge)
            tui_app.run_with_workflow(
                workflow_fn=lambda: coordinator.run_implementation(feature_id, parallel=parallel),
                feature_id=feature_id,
            )
        else:
            # Run without TUI
            coordinator.run_implementation(feature_id, parallel=parallel)

        console.print(Panel("[green]Workflow complete![/green]", title="Status"))

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option("--days", type=int, default=30, help="Number of days to analyze")
@click.option("--live", is_flag=True, help="Live updating display")
def metrics(days: int, live: bool) -> None:
    """View performance metrics (Phase 5).

    Example:
        supervisor metrics --days 7
        supervisor metrics --live
    """
    repo_path = get_repo_path()
    db_path = repo_path / ".supervisor/state.db"

    if not db_path.exists():
        console.print("[yellow]No supervisor database found. Run 'supervisor init' first.[/yellow]")
        return

    try:
        db = Database(db_path)
        aggregator = MetricsAggregator(db)
        dashboard = MetricsDashboard(aggregator)

        # FIX (v27 - Gemini PR review): Handle --live option
        if live:
            console.print("[yellow]Live mode is not yet implemented.[/yellow]")
        dashboard.show(days=days)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
def roles() -> None:
    """List available roles."""
    loader = RoleLoader()
    available = loader.list_available_roles()

    table = Table(title="Available Roles")
    table.add_column("Role", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("CLI", style="green")

    for role_name in available:
        try:
            role = loader.load_role(role_name)
            table.add_row(role.name, role.description, role.cli)
        except Exception:
            table.add_row(role_name, "[red]Error loading[/red]", "?")

    console.print(table)


@main.command()
@click.argument("workflow_file", type=click.Path(exists=True))
def visualize(workflow_file: str) -> None:
    """Visualize a workflow graph in the terminal."""
    from pydantic import ValidationError
    from rich.markup import escape

    # Load and validate YAML with error handling
    try:
        with open(workflow_file) as f:
            data = yaml.safe_load(f)
        workflow = WorkflowGraph(**data)
    except yaml.YAMLError as e:
        console.print(f"[red]Invalid YAML:[/] {escape(str(e))}")
        return
    except ValidationError as e:
        console.print("[red]Schema validation failed:[/]")
        for err in e.errors():
            console.print(f"  [red]• {escape(str(err))}[/]")
        return
    except (TypeError, ValueError) as e:
        console.print(f"[red]Invalid workflow data:[/] {escape(str(e))}")
        return

    renderer = TerminalGraphRenderer(console)

    # Show graph tree
    tree = renderer.render_as_tree(workflow)
    console.print(tree)

    # Show summary - SECURITY: escape user-controlled values
    console.print()
    console.print(f"[bold]Nodes:[/] {len(workflow.nodes)}")
    console.print(f"[bold]Edges:[/] {len(workflow.edges)}")
    console.print(f"[bold]Entry:[/] {escape(workflow.entry_point)}")
    exit_str = (
        ", ".join(escape(e) for e in workflow.exit_points)
        if workflow.exit_points
        else "(auto-detect)"
    )
    console.print(f"[bold]Exits:[/] {exit_str}")

    # Validate and show any issues
    errors = workflow.validate_graph()
    if errors:
        console.print("\n[red bold]Validation Errors:[/]")
        for error in errors:
            # SECURITY: escape error messages that may contain user data
            console.print(f"  [red]• {escape(str(error))}[/]")
    else:
        console.print("\n[green]✓ Graph is valid[/]")


@main.command()
@click.option("--workflow-id", "-w", help="Filter by workflow ID")
@click.option("--execution-id", "-e", help="Show specific execution")
def status(workflow_id: str | None, execution_id: str | None) -> None:
    """Show workflow execution status."""
    from rich.markup import escape

    repo_path = get_repo_path()
    db_path = repo_path / ".supervisor" / "state.db"

    if not db_path.exists():
        console.print("[yellow]No supervisor database found. Run 'supervisor init' first.[/yellow]")
        return

    db = Database(db_path)

    if execution_id:
        # SECURITY: Escape execution_id for all Rich output
        safe_exec_id = escape(execution_id)

        # Show specific execution
        try:
            with db._connect() as conn:
                exec_row = conn.execute(
                    "SELECT status, started_at, completed_at, error FROM graph_executions WHERE id=?",
                    (execution_id,),
                ).fetchone()

                if not exec_row:
                    console.print(f"[red]Execution '{safe_exec_id}' not found[/]")
                    return

                workflow_row = conn.execute(
                    """
                    SELECT w.definition FROM graph_workflows w
                    JOIN graph_executions e ON w.id = e.graph_id WHERE e.id = ?
                """,
                    (execution_id,),
                ).fetchone()
        except Exception as e:
            console.print(f"[red]Database error: {e}[/]")
            return

        # Handle missing workflow definition gracefully
        if not workflow_row:
            console.print(f"[red]Workflow definition not found for execution '{safe_exec_id}'[/]")
            return

        workflow = WorkflowGraph.model_validate_json(workflow_row[0])

        # Show execution info
        status_color = {
            "running": "blue",
            "completed": "green",
            "failed": "red",
            "cancelled": "yellow",
        }.get(exec_row[0], "white")

        # SECURITY: Escape error message which may contain user data
        safe_error = escape(str(exec_row[3])) if exec_row[3] else "-"

        console.print(
            Panel(
                f"[bold]Status:[/] [{status_color}]{exec_row[0]}[/]\n"
                f"[bold]Started:[/] {exec_row[1]}\n"
                f"[bold]Completed:[/] {exec_row[2] or '-'}\n"
                f"[bold]Error:[/] {safe_error}",
                title=f"Execution: {safe_exec_id[:8]}...",
            )
        )

        # Show node status table
        renderer = StatusTableRenderer(console)
        statuses = {}
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, status FROM node_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchall()
            statuses = {r[0]: r[1] for r in rows}

        table = renderer.render_status_table(workflow, execution_id, statuses)
        console.print(table)
    else:
        # List recent executions (existing functionality)
        console.print(
            Panel(
                f"Database: {db_path}\nWorkflow ID filter: {workflow_id or 'All'}",
                title="Supervisor Status",
            )
        )


@main.command()
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--workflow-id", required=True, help="Unique workflow execution ID")
@click.option("--validate-only", is_flag=True, help="Only validate, don't execute")
@click.option("--live", is_flag=True, help="Show live execution monitor")
def run_graph(workflow_file: str, workflow_id: str, validate_only: bool, live: bool) -> None:
    """Execute a declarative workflow graph from YAML."""
    # Load workflow YAML with proper error handling
    try:
        with open(workflow_file) as f:
            workflow_dict = yaml.safe_load(f)
            if not isinstance(workflow_dict, dict):
                console.print(
                    f"[red]Error: Invalid YAML content in '{workflow_file}'. "
                    f"Expected a dictionary, got {type(workflow_dict).__name__}.[/red]"
                )
                sys.exit(1)
            workflow = WorkflowGraph(**workflow_dict)
    except yaml.YAMLError as e:
        console.print(f"[red]Error parsing YAML file '{workflow_file}':[/red]")
        console.print(f"  {e}")
        sys.exit(1)
    except pydantic.ValidationError as e:
        console.print("[red]Error validating workflow schema:[/red]")
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            console.print(f"  - {loc}: {err['msg']}")
        sys.exit(1)
    except (ValueError, TypeError) as e:
        console.print("[red]Error validating workflow:[/red]")
        console.print(f"  {e}")
        sys.exit(1)

    # Validate
    errors = workflow.validate_graph()
    if errors:
        console.print("[red]Validation errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        sys.exit(1)

    console.print("[green]Workflow validation passed[/green]")
    console.print(f"  Nodes: {len(workflow.nodes)}")
    console.print(f"  Edges: {len(workflow.edges)}")

    if validate_only:
        return

    # Initialize components with shared database instance
    repo_path = get_repo_path()
    db_path = repo_path / ".supervisor" / "state.db"
    db = Database(db_path)
    engine = ExecutionEngine(repo_path, db=db)
    orchestrator = GraphOrchestrator(db, engine, engine.gate_executor, engine.gate_loader)

    # Start and run workflow
    async def run():
        exec_id = await orchestrator.start_workflow(workflow, workflow_id)
        console.print(f"[blue]Started execution: {exec_id}[/blue]")

        if live:
            # Run with live monitor
            # Use explicit task management so we can cancel monitor when worker finishes
            monitor = LiveExecutionMonitor(orchestrator, db)
            worker = WorkflowWorker(orchestrator)

            async def run_worker_and_cancel_monitor():
                """Run worker, then signal monitor to stop."""
                try:
                    result = await worker.run_until_complete(exec_id)
                    return result
                finally:
                    # Always cancel monitor when worker exits (success or failure)
                    monitor.cancel()

            try:
                # Run worker and monitor concurrently
                # Worker cancels monitor when done via the cancel() method
                results = await asyncio.gather(
                    run_worker_and_cancel_monitor(),
                    monitor.monitor(exec_id, workflow),
                    return_exceptions=True,
                )
                worker_result = results[0]
                if isinstance(worker_result, Exception):
                    raise worker_result
                return worker_result
            except Exception as e:
                monitor.cancel()  # Ensure monitor stops on error
                console.print(f"[red]Error during execution: {e}[/]")
                return "failed"
        else:
            # Existing non-live execution
            worker = WorkflowWorker(orchestrator)
            return await worker.run_until_complete(exec_id)

    final_status = asyncio.run(run())

    if final_status == "completed":
        console.print("[green]Workflow completed successfully[/green]")
    else:
        console.print(f"[red]Workflow {final_status}[/red]")
        sys.exit(1)


@main.command()
@click.argument("execution_id")
@click.option("--node", "-n", help="Specific node ID to inspect")
@click.option("--interactive", "-i", is_flag=True, help="Enter interactive inspection mode")
def inspect(execution_id: str, node: str | None, interactive: bool) -> None:
    """Inspect an execution's node details.

    By default, shows a summary of all nodes (one-shot).
    Use --node to inspect a specific node.
    Use --interactive for REPL-style exploration.
    """
    repo_path = get_repo_path()
    db_path = repo_path / ".supervisor" / "state.db"

    if not db_path.exists():
        console.print("[yellow]No supervisor database found. Run 'supervisor init' first.[/yellow]")
        return

    db = Database(db_path)

    # Get workflow
    try:
        with db._connect() as conn:
            row = conn.execute(
                """
                SELECT w.definition FROM graph_workflows w
                JOIN graph_executions e ON w.id = e.graph_id
                WHERE e.id = ?
            """,
                (execution_id,),
            ).fetchone()
    except Exception as e:
        click.secho(f"Database error: {e}", fg="red")
        return

    if not row:
        click.secho(f"Execution '{execution_id}' not found", fg="red")
        return

    workflow = WorkflowGraph.model_validate_json(row[0])
    inspector = NodeInspector(db)

    if interactive:
        inspector.inspect_interactive(execution_id, workflow)
    else:
        inspector.inspect_node(execution_id, workflow, node)


@main.command()
def version() -> None:
    """Show version information."""
    from supervisor import __version__

    console.print(f"Supervisor v{__version__}")
    console.print("AI CLI Orchestrator")


if __name__ == "__main__":
    main()
