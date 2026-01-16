# Supervisor Studio - Phase 2: CLI Visualization (Terminal UI)

**Status:** Planning
**Objective:** Add terminal-based visualization for workflow graphs using Rich and textual libraries.

**Prerequisites:** Phase 1 (Declarative Engine Foundation) completed.

---

## 2.1 Overview

Phase 2 adds rich terminal UI capabilities for:
- Visualizing workflow graphs as ASCII art
- Real-time execution monitoring
- Interactive node inspection
- Status dashboards

This provides immediate value before the full web UI is built.

---

## 2.2 Graph ASCII Rendering

**File:** `supervisor/cli/graph_renderer.py` (New)

**Purpose:** Render workflow graphs as ASCII diagrams in the terminal.

### Dependencies

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from supervisor.core.graph_schema import WorkflowGraph, Node, NodeType, NodeStatus
from typing import Dict, Optional
import networkx as nx
```

### Graph Renderer Class

```python
from rich.markup import escape

class TerminalGraphRenderer:
    """
    Renders workflow graphs as ASCII art in the terminal.

    Features:
    - Topological layout (left-to-right flow)
    - Color-coded node types
    - Status indicators
    - Edge labels for conditions
    """

    # Node type symbols and colors
    NODE_STYLES = {
        NodeType.TASK: ("[ ]", "cyan"),
        NodeType.GATE: ("[G]", "yellow"),
        NodeType.BRANCH: ("[?]", "magenta"),
        NodeType.MERGE: ("[M]", "blue"),
        NodeType.PARALLEL: ("[P]", "green"),
        NodeType.SUBGRAPH: ("[S]", "white"),
        NodeType.HUMAN: ("[H]", "red"),
    }

    # Status colors - use STRING KEYS for DB compatibility
    # DB returns status as strings, not NodeStatus enums
    STATUS_COLORS = {
        "pending": "dim",
        "ready": "yellow",
        "running": "blue bold",
        "completed": "green",
        "failed": "red bold",
        "skipped": "dim strikethrough",
    }

    @staticmethod
    def _normalize_status(status) -> str:
        """Normalize status to string for consistent lookup."""
        if isinstance(status, NodeStatus):
            return status.value
        return str(status) if status else "pending"

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def render_graph(self, workflow: WorkflowGraph,
                    statuses: Optional[Dict[str, NodeStatus]] = None) -> str:
        """
        Render workflow graph as ASCII.

        Args:
            workflow: The workflow graph to render
            statuses: Optional dict of node_id -> current status

        Returns:
            ASCII representation of the graph
        """
        G = workflow._to_networkx()

        # Get topological levels for layout
        try:
            levels = list(nx.topological_generations(G))
        except nx.NetworkXUnfeasible:
            # Has cycles - use simple layout
            levels = [[n.id for n in workflow.nodes]]

        lines = []
        max_width = 0

        # Render each level
        for level_idx, level in enumerate(levels):
            level_nodes = []

            for node_id in level:
                node = next(n for n in workflow.nodes if n.id == node_id)
                symbol, color = self.NODE_STYLES.get(node.type, ("[ ]", "white"))

                # SECURITY: Escape node labels to prevent Rich markup injection
                safe_label = escape(node.label or node_id)

                # Apply status color if available
                # Normalize status to string for consistent lookup
                status = self._normalize_status(statuses.get(node_id)) if statuses else None
                if status and status != "pending":
                    status_color = self.STATUS_COLORS.get(status, "white")
                    node_str = f"[{status_color}]{symbol} {safe_label}[/]"
                else:
                    node_str = f"[{color}]{symbol} {safe_label}[/]"

                level_nodes.append(node_str)

            level_line = "  |  ".join(level_nodes)
            lines.append(level_line)
            max_width = max(max_width, len(level_line))

            # Add connector arrows if not last level
            if level_idx < len(levels) - 1:
                lines.append("  " + "  |  " * len(level_nodes))
                lines.append("  " + "  v  " * len(level_nodes))

        return "\n".join(lines)

    def render_as_tree(self, workflow: WorkflowGraph,
                      statuses: Optional[Dict[str, NodeStatus]] = None) -> Tree:
        """
        Render workflow as a Rich Tree (hierarchical view).
        """
        tree = Tree(f"[bold]{workflow.name}[/] (v{workflow.version})")

        # Add entry point
        entry = next(n for n in workflow.nodes if n.id == workflow.entry_point)
        self._add_node_to_tree(tree, workflow, entry, statuses, visited=set())

        return tree

    def _add_node_to_tree(self, parent: Tree, workflow: WorkflowGraph,
                         node: Node, statuses: Optional[Dict[str, str]],
                         visited: set):
        """Recursively add nodes to tree"""
        # SECURITY: Escape node labels to prevent Rich markup injection
        safe_label = escape(node.label or node.id)
        safe_id = escape(node.id)

        if node.id in visited:
            parent.add(f"[dim]↩ {safe_id} (loop)[/]")
            return

        visited.add(node.id)

        symbol, color = self.NODE_STYLES.get(node.type, ("[ ]", "white"))
        # Normalize status to string for consistent lookup
        status = self._normalize_status(statuses.get(node.id)) if statuses else None
        status_indicator = ""

        if status and status != "pending":
            status_color = self.STATUS_COLORS.get(status, "white")
            if status == "completed":
                status_indicator = " ✓"
            elif status == "failed":
                status_indicator = " ✗"
            elif status == "running":
                status_indicator = " ⟳"

            node_text = f"[{status_color}]{symbol} {safe_label}{status_indicator}[/]"
        else:
            node_text = f"[{color}]{symbol} {safe_label}[/]"

        branch = parent.add(node_text)

        # Add children
        outgoing = [e for e in workflow.edges if e.source == node.id]
        for edge in outgoing:
            child_node = next((n for n in workflow.nodes if n.id == edge.target), None)
            if child_node:
                # Add condition label if present
                if edge.condition:
                    condition_label = f"[dim]({edge.condition.field} {edge.condition.operator} {edge.condition.value})[/]"
                    edge_branch = branch.add(condition_label)
                    self._add_node_to_tree(edge_branch, workflow, child_node,
                                          statuses, visited.copy())
                else:
                    self._add_node_to_tree(branch, workflow, child_node,
                                          statuses, visited.copy())
```

### Status Table Renderer

```python
class StatusTableRenderer:
    """Renders node execution status as a Rich table"""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def render_status_table(self, workflow: WorkflowGraph,
                           execution_id: str,
                           statuses: Dict[str, str],
                           outputs: Optional[Dict[str, any]] = None) -> Table:
        """
        Render execution status as a table.
        """
        table = Table(title=f"Execution: {execution_id[:8]}...")

        table.add_column("Node", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Status", justify="center")
        table.add_column("Output", max_width=40)

        for node in workflow.nodes:
            status = statuses.get(node.id, "pending")
            output = outputs.get(node.id, "") if outputs else ""

            # Color-code status
            if status == "completed":
                status_text = "[green]✓ Completed[/]"
            elif status == "failed":
                status_text = "[red]✗ Failed[/]"
            elif status == "running":
                status_text = "[blue]⟳ Running[/]"
            elif status == "ready":
                status_text = "[yellow]○ Ready[/]"
            elif status == "skipped":
                status_text = "[dim]⊘ Skipped[/]"
            else:
                status_text = "[dim]○ Pending[/]"

            # Truncate output
            output_str = str(output)[:37] + "..." if len(str(output)) > 40 else str(output)

            table.add_row(
                node.label or node.id,
                node.type.value,
                status_text,
                output_str
            )

        return table
```

---

## 2.3 Live Execution Monitor

**File:** `supervisor/cli/live_monitor.py` (New)

**Purpose:** Real-time terminal display of workflow execution.

```python
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from supervisor.core.graph_engine import GraphOrchestrator
from supervisor.core.state import Database
import asyncio

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
    - Reuses Progress widget to avoid flickering
    - Uses public orchestrator methods (not private _get_* methods)
    """

    def __init__(self, orchestrator: GraphOrchestrator, db: Database):
        self.orchestrator = orchestrator
        self.db = db
        self.console = Console()
        self.graph_renderer = TerminalGraphRenderer(self.console)
        self.status_renderer = StatusTableRenderer(self.console)
        # Reuse progress widget to avoid recreating each refresh
        self._progress = None
        self._progress_task_id = None

    def create_layout(self) -> Layout:
        """Create the terminal layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )

        layout["main"].split_row(
            Layout(name="graph", ratio=1),
            Layout(name="status", ratio=1)
        )

        return layout

    async def monitor(self, execution_id: str, workflow: WorkflowGraph):
        """
        Monitor execution with live updates.

        NOTE: DB queries are wrapped in asyncio.to_thread to avoid
        blocking the event loop during concurrent worker execution.
        """
        layout = self.create_layout()

        # Initialize reusable progress bar
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        self._progress_task_id = self._progress.add_task("Nodes: 0/0", total=len(workflow.nodes))

        with Live(layout, console=self.console, refresh_per_second=2) as live:
            while True:
                # Get current statuses - offload to thread to avoid blocking
                statuses = await asyncio.to_thread(self._get_statuses, execution_id)
                outputs = await asyncio.to_thread(self._get_outputs, execution_id)
                exec_status = await asyncio.to_thread(
                    self.orchestrator.get_execution_status, execution_id
                )

                # Update header
                if exec_status == "running":
                    header_text = f"[bold blue]⟳ Executing:[/] {workflow.name}"
                elif exec_status == "completed":
                    header_text = f"[bold green]✓ Completed:[/] {workflow.name}"
                elif exec_status == "failed":
                    header_text = f"[bold red]✗ Failed:[/] {workflow.name}"
                else:
                    header_text = f"[bold]{workflow.name}[/]"

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
                completed = sum(1 for s in statuses.values() if s in ["completed", "skipped"])
                total = len(workflow.nodes)
                self._progress.update(
                    self._progress_task_id,
                    completed=completed,
                    description=f"Nodes: {completed}/{total}"
                )
                layout["footer"].update(Panel(self._progress, title="Progress"))

                # Check if done
                if exec_status in ["completed", "failed", "cancelled"]:
                    await asyncio.sleep(1)  # Show final state
                    break

                await asyncio.sleep(0.5)

    def _get_statuses(self, execution_id: str) -> Dict[str, str]:
        """
        Get all node statuses.

        NOTE: This method is called from asyncio.to_thread() to avoid
        blocking the event loop. Do not call directly from async code.
        """
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, status FROM node_executions WHERE execution_id=?",
                (execution_id,)
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    def _get_outputs(self, execution_id: str) -> Dict[str, any]:
        """
        Get all node outputs.

        NOTE: This method is called from asyncio.to_thread() to avoid
        blocking the event loop. Do not call directly from async code.
        """
        import json
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, output_data FROM node_executions WHERE execution_id=?",
                (execution_id,)
            ).fetchall()
            return {
                r[0]: json.loads(r[1]) if r[1] else None
                for r in rows
            }
```

### Required Public Accessors on GraphOrchestrator

To avoid tight coupling and private member access, add these public methods to
`GraphOrchestrator` (Phase 1):

```python
class GraphOrchestrator:
    # ... existing code ...

    def get_execution_status(self, execution_id: str) -> str:
        """Public accessor for execution status."""
        return self._get_execution_status(execution_id)

    def get_all_node_statuses(self, execution_id: str) -> Dict[str, str]:
        """Public accessor for all node statuses."""
        return self._get_all_statuses(execution_id)

    def get_node_output(self, execution_id: str, node_id: str) -> Any:
        """Public accessor for node output."""
        return self._get_node_output(execution_id, node_id)
```

---

## 2.4 Interactive Node Inspector

**File:** `supervisor/cli/node_inspector.py` (New)

**Purpose:** Interactive exploration of node details and outputs.

```python
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.text import Text
from supervisor.core.graph_schema import WorkflowGraph, Node
from supervisor.core.state import Database
import json

class NodeInspector:
    """
    Interactive node inspection in the terminal.

    Features:
    - View node configuration
    - View execution output (formatted JSON)
    - View error details
    - Navigate between nodes
    """

    def __init__(self, db: Database):
        self.db = db
        self.console = Console()

    def _get_statuses(self, execution_id: str) -> Dict[str, str]:
        """Get all node statuses for display."""
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, status FROM node_executions WHERE execution_id=?",
                (execution_id,)
            ).fetchall()
            return {row[0]: row[1] for row in rows}

    def inspect(self, execution_id: str, workflow: WorkflowGraph):
        """
        Interactive inspection loop.
        """
        self.console.print("\n[bold]Node Inspector[/]")
        self.console.print("Commands: [node_id], 'list', 'graph', 'quit'\n")

        while True:
            cmd = Prompt.ask("[cyan]inspect[/]")

            if cmd == "quit" or cmd == "q":
                break
            elif cmd == "list" or cmd == "l":
                self._list_nodes(workflow)
            elif cmd == "graph" or cmd == "g":
                renderer = TerminalGraphRenderer(self.console)
                statuses = self._get_statuses(execution_id)
                tree = renderer.render_as_tree(workflow, statuses)
                self.console.print(tree)
            else:
                # Try to find node
                node = next((n for n in workflow.nodes if n.id == cmd), None)
                if node:
                    self._inspect_node(execution_id, node)
                else:
                    self.console.print(f"[red]Node '{cmd}' not found[/]")

    def _list_nodes(self, workflow: WorkflowGraph):
        """List all nodes"""
        from rich.table import Table

        table = Table(title="Workflow Nodes")
        table.add_column("ID")
        table.add_column("Type")
        table.add_column("Label")

        for node in workflow.nodes:
            table.add_row(node.id, node.type.value, node.label or "-")

        self.console.print(table)

    def _inspect_node(self, execution_id: str, node: Node):
        """Show detailed node information"""
        # Get execution data
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT status, input_data, output_data, error FROM node_executions "
                "WHERE execution_id=? AND node_id=?",
                (execution_id, node.id)
            ).fetchone()

        if not row:
            self.console.print(f"[yellow]No execution data for {node.id}[/]")
            return

        status, input_data, output_data, error = row

        # Node info panel - use Rich Group to preserve renderables
        # DO NOT stringify Rich objects - use Group to compose them
        info_parts = [
            Text.from_markup(f"[bold]ID:[/] {node.id}"),
            Text.from_markup(f"[bold]Type:[/] {node.type.value}"),
            Text.from_markup(f"[bold]Label:[/] {node.label or '-'}"),
            Text.from_markup(f"[bold]Status:[/] {status}"),
        ]

        # Add type-specific config
        config = (node.task_config or node.gate_config or node.branch_config or
                 node.merge_config or node.parallel_config or node.subgraph_config or
                 node.human_config)
        if config:
            info_parts.append(Text(""))  # Empty line
            info_parts.append(Text.from_markup("[bold]Configuration:[/]"))
            config_json = json.dumps(config.model_dump(), indent=2)
            info_parts.append(Syntax(config_json, "json", theme="monokai"))

        # Use Group to compose multiple renderables without stringifying
        self.console.print(Panel(Group(*info_parts), title=f"Node: {node.id}"))

        # Input data
        if input_data:
            input_json = json.loads(input_data)
            self.console.print(Panel(
                Syntax(json.dumps(input_json, indent=2), "json", theme="monokai"),
                title="Input Data"
            ))

        # Output data
        if output_data:
            output_json = json.loads(output_data)
            self.console.print(Panel(
                Syntax(json.dumps(output_json, indent=2), "json", theme="monokai"),
                title="Output Data"
            ))

        # Error
        if error:
            self.console.print(Panel(
                f"[red]{error}[/]",
                title="Error",
                style="red"
            ))
```

---

## 2.5 CLI Commands

**File:** `supervisor/cli.py` (Extend)

**Purpose:** Add CLI commands for terminal UI features.

```python
@cli.command()
@click.argument("workflow_file", type=click.Path(exists=True))
def visualize(workflow_file: str):
    """Visualize a workflow graph in the terminal"""
    import yaml

    with open(workflow_file) as f:
        workflow = WorkflowGraph(**yaml.safe_load(f))

    console = Console()
    renderer = TerminalGraphRenderer(console)

    # Show graph tree
    tree = renderer.render_as_tree(workflow)
    console.print(tree)

    # Show summary
    console.print()
    console.print(f"[bold]Nodes:[/] {len(workflow.nodes)}")
    console.print(f"[bold]Edges:[/] {len(workflow.edges)}")
    console.print(f"[bold]Entry:[/] {workflow.entry_point}")
    console.print(f"[bold]Exits:[/] {', '.join(workflow.exit_points or ['(auto-detect)'])}")

    # Validate and show any issues
    errors = workflow.validate_graph()
    if errors:
        console.print("\n[red bold]Validation Errors:[/]")
        for error in errors:
            console.print(f"  [red]• {error}[/]")
    else:
        console.print("\n[green]✓ Graph is valid[/]")


@cli.command()
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--workflow-id", required=True)
@click.option("--live", is_flag=True, help="Show live execution monitor")
def run_graph(workflow_file: str, workflow_id: str, live: bool):
    """Execute a workflow with optional live monitoring"""
    import yaml

    with open(workflow_file) as f:
        workflow = WorkflowGraph(**yaml.safe_load(f))

    # Validate
    errors = workflow.validate_graph()
    if errors:
        console = Console()
        console.print("[red]Validation errors:[/]")
        for error in errors:
            console.print(f"  • {error}")
        return

    # Initialize
    db = Database()
    engine = ExecutionEngine(Path("."))
    orchestrator = GraphOrchestrator(db, engine, engine.gate_executor, engine.gate_loader)

    async def run():
        exec_id = await orchestrator.start_workflow(workflow, workflow_id)

        if live:
            # Run with live monitor - use asyncio.gather for proper lifecycle management
            monitor = LiveExecutionMonitor(orchestrator, db)
            worker = WorkflowWorker(orchestrator)

            # Run worker and monitor concurrently with gather
            # If either fails, both are properly cleaned up
            try:
                results = await asyncio.gather(
                    worker.run_until_complete(exec_id),
                    monitor.monitor(exec_id, workflow),
                    return_exceptions=True  # Don't cancel on first exception
                )
                # Worker result is first
                worker_result = results[0]
                if isinstance(worker_result, Exception):
                    raise worker_result
                return worker_result
            except Exception as e:
                # Ensure worker is stopped if monitor fails
                console = Console()
                console.print(f"[red]Error during execution: {e}[/]")
                return "failed"
        else:
            # Simple execution
            worker = WorkflowWorker(orchestrator)
            return await worker.run_until_complete(exec_id)

    status = asyncio.run(run())

    if status == "completed":
        click.secho("✓ Workflow completed", fg="green")
    else:
        click.secho(f"✗ Workflow {status}", fg="red")


@cli.command()
@click.argument("execution_id")
def inspect(execution_id: str):
    """Interactively inspect an execution"""
    db = Database()

    # Get workflow
    with db._connect() as conn:
        row = conn.execute("""
            SELECT w.definition FROM graph_workflows w
            JOIN graph_executions e ON w.id = e.graph_id
            WHERE e.id = ?
        """, (execution_id,)).fetchone()

    if not row:
        click.secho(f"Execution '{execution_id}' not found", fg="red")
        return

    workflow = WorkflowGraph.model_validate_json(row[0])

    inspector = NodeInspector(db)
    inspector.inspect(execution_id, workflow)


@cli.command()
@click.argument("execution_id")
def status(execution_id: str):
    """Show execution status"""
    db = Database()
    console = Console()

    # Get workflow and status
    with db._connect() as conn:
        exec_row = conn.execute(
            "SELECT status, started_at, completed_at, error FROM graph_executions WHERE id=?",
            (execution_id,)
        ).fetchone()

        if not exec_row:
            console.print(f"[red]Execution '{execution_id}' not found[/]")
            return

        workflow_row = conn.execute("""
            SELECT w.definition FROM graph_workflows w
            JOIN graph_executions e ON w.id = e.graph_id WHERE e.id = ?
        """, (execution_id,)).fetchone()

    workflow = WorkflowGraph.model_validate_json(workflow_row[0])

    # Show execution info
    status_color = {
        "running": "blue",
        "completed": "green",
        "failed": "red",
        "cancelled": "yellow"
    }.get(exec_row[0], "white")

    console.print(Panel(
        f"[bold]Status:[/] [{status_color}]{exec_row[0]}[/]\n"
        f"[bold]Started:[/] {exec_row[1]}\n"
        f"[bold]Completed:[/] {exec_row[2] or '-'}\n"
        f"[bold]Error:[/] {exec_row[3] or '-'}",
        title=f"Execution: {execution_id[:8]}..."
    ))

    # Show node status table
    renderer = StatusTableRenderer(console)
    statuses = {}
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT node_id, status FROM node_executions WHERE execution_id=?",
            (execution_id,)
        ).fetchall()
        statuses = {r[0]: r[1] for r in rows}

    table = renderer.render_status_table(workflow, execution_id, statuses)
    console.print(table)
```

---

## 2.6 Example Usage

### Visualize a Workflow

```bash
$ supervisor visualize workflows/debug-loop.yaml

Workflow: Debug with Retry Loop (v1.0.0)
├── [ ] Analyze Bug
│   └── [ ] Apply Fix
│       └── [G] Run Tests
│           └── [?] Check Test Result
│               ├── (test_status == passed)
│               │   └── [ ] Final Review
│               └── (test_status != passed)
│                   └── ↩ analyze (loop)

Nodes: 5
Edges: 5
Entry: analyze
Exits: review

✓ Graph is valid
```

### Run with Live Monitor

```bash
$ supervisor run-graph workflows/debug-loop.yaml --workflow-id debug-123 --live

┌─ Executing: Debug with Retry Loop ────────────────────────────────┐
│                                                                    │
├─ Workflow Graph ─────────────┬─ Node Status ─────────────────────┤
│                              │                                    │
│ ├── [✓] Analyze Bug          │ Node        Type    Status        │
│ │   └── [⟳] Apply Fix        │ ─────────────────────────────────  │
│ │       └── [ ] Run Tests    │ analyze     task    ✓ Completed   │
│ │           └── [ ] Check    │ fix         task    ⟳ Running     │
│ │               ├── [ ] ...  │ test        gate    ○ Pending     │
│                              │ check       branch  ○ Pending     │
│                              │ review      task    ○ Pending     │
│                              │                                    │
├─ Progress ───────────────────┴────────────────────────────────────┤
│ ⟳ Nodes: 1/5  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  20%     │
└───────────────────────────────────────────────────────────────────┘
```

### Inspect Execution

```bash
$ supervisor inspect abc123

Node Inspector
Commands: [node_id], 'list', 'graph', 'quit'

inspect> list
┌─────────┬─────────┬───────────────┐
│ ID      │ Type    │ Label         │
├─────────┼─────────┼───────────────┤
│ analyze │ task    │ Analyze Bug   │
│ fix     │ task    │ Apply Fix     │
│ test    │ gate    │ Run Tests     │
│ check   │ branch  │ Check Result  │
│ review  │ task    │ Final Review  │
└─────────┴─────────┴───────────────┘

inspect> fix
┌─ Node: fix ───────────────────────────────────────────────────────┐
│ ID: fix                                                            │
│ Type: task                                                         │
│ Label: Apply Fix                                                   │
│ Status: completed                                                  │
│                                                                    │
│ Configuration:                                                     │
│ {                                                                  │
│   "role": "implementer",                                           │
│   "task_template": "Fix based on analysis: {input}"               │
│ }                                                                  │
└────────────────────────────────────────────────────────────────────┘
┌─ Output Data ──────────────────────────────────────────────────────┐
│ {                                                                  │
│   "files_modified": ["src/bug.py"],                               │
│   "changes": "Fixed null pointer check"                           │
│ }                                                                  │
└────────────────────────────────────────────────────────────────────┘

inspect> quit
```

---

## 2.7 Dependencies

Add to `pyproject.toml`:

```toml
[project.dependencies]
rich = ">=13.0.0"
```

Optional for advanced TUI:
```toml
[project.optional-dependencies]
tui = [
    "textual>=0.50.0",  # For full TUI application
]
```

---

**End of Phase 2 Plan**
