"""CLI entry point for the Supervisor orchestrator.

Commands:
- supervisor init: Initialize project for supervisor
- supervisor plan: Run planner on a task
- supervisor run: Run a workflow
- supervisor status: Show current workflow status
- supervisor roles: List available roles
"""

import sys
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from supervisor.core.engine import ExecutionEngine
from supervisor.core.roles import RoleLoader
from supervisor.core.state import Database

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

    # Initialize database
    db = Database(supervisor_dir / "state.db")

    console.print(
        Panel(
            "[green]Project initialized![/green]\n\n"
            f"Created: {supervisor_dir}\n"
            "- config.yaml: Project configuration\n"
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
                    c.get("title", c.get("id", "?"))
                    for c in phase.get("components", [])
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
@click.option("--workflow-id", "-w", help="Show specific workflow")
def status(workflow_id: str | None) -> None:
    """Show current workflow status."""
    repo_path = get_repo_path()
    db_path = repo_path / ".supervisor/state.db"

    if not db_path.exists():
        console.print("[yellow]No supervisor database found. Run 'supervisor init' first.[/yellow]")
        return

    db = Database(db_path)

    # For now, just show that the database exists
    console.print(Panel(
        f"Database: {db_path}\n"
        f"Workflow ID filter: {workflow_id or 'All'}",
        title="Supervisor Status",
    ))


@main.command()
def version() -> None:
    """Show version information."""
    from supervisor import __version__

    console.print(f"Supervisor v{__version__}")
    console.print("AI CLI Orchestrator")


if __name__ == "__main__":
    main()
