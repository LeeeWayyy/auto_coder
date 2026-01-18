# Supervisor Studio - Phase 2: CLI Visualization (Terminal UI)

**Status:** Implemented
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

**File:** `supervisor/cli_ui/graph_renderer.py` (New)

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

    Performance Notes:
    - Uses node_map for O(1) lookups instead of O(N) list scans
    - networkx graph is built with string node IDs via _to_networkx() which uses
      G.add_node(node.id), so topological_generations() returns strings not Node objects
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

    def _build_node_map(self, workflow: WorkflowGraph) -> Dict[str, Node]:
        """Build O(1) lookup map for nodes by ID."""
        return {n.id: n for n in workflow.nodes}

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
        node_map = self._build_node_map(workflow)

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
                node = node_map.get(node_id)
                if not node:
                    continue  # Skip missing nodes gracefully
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
                      statuses: Optional[Dict[str, NodeStatus]] = None,
                      max_depth: int = 50) -> Tree:
        """
        Render workflow as a Rich Tree (hierarchical view).

        Args:
            workflow: The workflow graph to render
            statuses: Optional dict of node_id -> current status
            max_depth: Maximum tree depth to prevent exponential blow-up (default: 50)

        Returns empty tree with error message if entry point is missing.
        """
        # SECURITY: Escape workflow name and version to prevent Rich markup injection
        safe_name = escape(workflow.name)
        safe_version = escape(workflow.version)
        tree = Tree(f"[bold]{safe_name}[/] (v{safe_version})")

        # Build node map for O(1) lookups
        node_map = self._build_node_map(workflow)

        # Find entry point with graceful fallback
        entry = node_map.get(workflow.entry_point)
        if not entry:
            tree.add("[red]Error: Entry point node not found[/]")
            return tree

        self._add_node_to_tree(tree, workflow, entry, statuses, node_map, visited=set(), depth=0, max_depth=max_depth)

        return tree

    def _add_node_to_tree(self, parent: Tree, workflow: WorkflowGraph,
                         node: Node, statuses: Optional[Dict[str, str]],
                         node_map: Dict[str, Node], visited: set,
                         depth: int = 0, max_depth: int = 50):
        """Recursively add nodes to tree with depth limiting to prevent exponential blow-up."""
        # Depth guard to prevent exponential blow-up on wide DAGs
        if depth >= max_depth:
            parent.add("[dim]... (max depth reached)[/]")
            return
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

        # Add children using node_map for O(1) lookup
        outgoing = [e for e in workflow.edges if e.source == node.id]
        for edge in outgoing:
            child_node = node_map.get(edge.target)
            if child_node:
                # Add condition label if present
                # SECURITY: Escape condition values to prevent Rich markup injection
                if edge.condition:
                    safe_field = escape(str(edge.condition.field))
                    safe_op = escape(str(edge.condition.operator))
                    safe_val = escape(str(edge.condition.value)[:50])  # Truncate long values
                    condition_label = f"[dim]({safe_field} {safe_op} {safe_val})[/]"
                    edge_branch = branch.add(condition_label)
                    self._add_node_to_tree(edge_branch, workflow, child_node,
                                          statuses, node_map, visited.copy(), depth + 1, max_depth)
                else:
                    self._add_node_to_tree(branch, workflow, child_node,
                                          statuses, node_map, visited.copy(), depth + 1, max_depth)
```

### Status Table Renderer

```python
class StatusTableRenderer:
    """Renders node execution status as a Rich table.

    SECURITY: All user-controlled strings (node labels, outputs, execution_id) are escaped
    to prevent Rich markup injection.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def render_status_table(self, workflow: WorkflowGraph,
                           execution_id: str,
                           statuses: Dict[str, str],
                           outputs: Optional[Dict[str, any]] = None) -> Table:
        """
        Render execution status as a table.
        """
        # SECURITY: Escape execution_id before truncating to prevent markup injection
        safe_exec_id = escape(execution_id[:8])
        table = Table(title=f"Execution: {safe_exec_id}...")

        table.add_column("Node", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Status", justify="center")
        table.add_column("Output", max_width=40)

        for node in workflow.nodes:
            status = statuses.get(node.id, "pending")
            # Handle None values explicitly - dict.get returns None if key exists with None value
            val = outputs.get(node.id) if outputs else None
            output = val if val is not None else ""

            # SECURITY: Escape node label to prevent Rich markup injection
            safe_label = escape(node.label or node.id)

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

            # SECURITY: Escape output and truncate
            output_str = escape(str(output))
            if len(output_str) > 40:
                output_str = output_str[:37] + "..."

            table.add_row(
                safe_label,
                node.type.value,
                status_text,
                output_str
            )

        return table
```

---

## 2.3 Live Execution Monitor

**File:** `supervisor/cli_ui/live_monitor.py` (New)

**Purpose:** Real-time terminal display of workflow execution.

```python
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markup import escape
from supervisor.core.graph_engine import GraphOrchestrator
from supervisor.core.graph_schema import WorkflowGraph
from supervisor.core.state import Database
from supervisor.cli_ui.graph_renderer import TerminalGraphRenderer, StatusTableRenderer
from typing import Dict, Any, Tuple
import asyncio
import json

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
        self._progress = None
        self._progress_task_id = None
        self._cancelled = False  # Flag for external cancellation

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
        self._progress_task_id = self._progress.add_task("Nodes: 0/0", total=len(workflow.nodes))

        with Live(layout, console=self.console, refresh_per_second=2) as live:
            while not self._cancelled:
                # Batch all DB queries into single call to reduce overhead
                statuses, outputs, exec_status = await asyncio.to_thread(
                    self._get_execution_snapshot, execution_id
                )

                # Track DB errors - break out if too many consecutive failures
                if exec_status == "unknown":
                    consecutive_db_errors += 1
                    if consecutive_db_errors >= self.MAX_DB_ERRORS:
                        self.console.print("[red]Monitor stopping: too many DB errors[/]")
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

                # Poll at 1 second intervals to balance responsiveness and DB load
                await asyncio.sleep(1.0)

    def _get_execution_snapshot(self, execution_id: str) -> Tuple[Dict[str, str], Dict[str, Any], str]:
        """
        Get complete execution snapshot in a single DB transaction.

        Batches statuses, outputs, and execution status into one call to reduce
        DB overhead and ensure consistent state.

        NOTE: This method is called from asyncio.to_thread() to avoid
        blocking the event loop. Do not call directly from async code.

        Returns:
            Tuple of (statuses, outputs, exec_status)
        """
        statuses: Dict[str, str] = {}
        outputs: Dict[str, Any] = {}
        exec_status = "unknown"

        try:
            with self.db._connect() as conn:
                # Get node statuses and TRUNCATED outputs in single query
                # Use substr() to limit output_data to 100 chars for performance
                # (StatusTableRenderer only displays ~40 chars anyway)
                rows = conn.execute(
                    "SELECT node_id, status, substr(output_data, 1, 100) FROM node_executions WHERE execution_id=?",
                    (execution_id,)
                ).fetchall()

                for row in rows:
                    node_id, status, output_summary = row
                    statuses[node_id] = status
                    # Parse JSON with error handling (may be truncated/invalid)
                    if output_summary:
                        try:
                            outputs[node_id] = json.loads(output_summary)
                        except json.JSONDecodeError:
                            # Truncated JSON or invalid - show summary as string
                            outputs[node_id] = output_summary[:40] + "..." if len(output_summary) > 40 else output_summary
                    else:
                        outputs[node_id] = None

                # Get execution status
                exec_row = conn.execute(
                    "SELECT status FROM graph_executions WHERE id=?",
                    (execution_id,)
                ).fetchone()
                if exec_row:
                    exec_status = exec_row[0]

        except Exception as e:
            # Log error but don't crash the monitor
            self.console.print(f"[dim]DB error: {e}[/]")

        return statuses, outputs, exec_status
```

---

## 2.4 Node Inspector

**File:** `supervisor/cli_ui/node_inspector.py` (New)

**Purpose:** Node inspection with both one-shot and interactive modes.

```python
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from supervisor.core.graph_schema import WorkflowGraph, Node
from supervisor.core.state import Database
from supervisor.cli_ui.graph_renderer import TerminalGraphRenderer
from typing import Dict, Optional
import json

class NodeInspector:
    """
    Node inspection in the terminal.

    Features:
    - One-shot mode: View specific node or all nodes and exit
    - Interactive mode: REPL-style navigation between nodes
    - View node configuration
    - View execution output (formatted JSON with error handling)
    - View error details

    SECURITY: All user-controlled strings are escaped to prevent Rich markup injection.

    NOTE: input_data is only persisted for the entry point node in Phase 1.
    Other nodes' inputs are computed dynamically and not stored.
    This is a known limitation of the current engine design.
    """

    def __init__(self, db: Database):
        self.db = db
        self.console = Console()

    def _get_statuses(self, execution_id: str) -> Dict[str, str]:
        """Get all node statuses for display."""
        try:
            with self.db._connect() as conn:
                rows = conn.execute(
                    "SELECT node_id, status FROM node_executions WHERE execution_id=?",
                    (execution_id,)
                ).fetchall()
                return {row[0]: row[1] for row in rows}
        except Exception:
            return {}

    def inspect_node(self, execution_id: str, workflow: WorkflowGraph,
                    node_id: Optional[str] = None):
        """
        One-shot inspection: show node details and exit.

        Args:
            execution_id: The execution to inspect
            workflow: The workflow graph
            node_id: Specific node to inspect, or None to show all nodes
        """
        if node_id:
            # Show specific node
            node = next((n for n in workflow.nodes if n.id == node_id), None)
            if node:
                self._inspect_node(execution_id, node)
            else:
                self.console.print(f"[red]Node '{escape(node_id)}' not found[/]")
        else:
            # Show all nodes summary
            self._list_nodes(workflow)
            statuses = self._get_statuses(execution_id)
            for node in workflow.nodes:
                status = statuses.get(node.id, "pending")
                safe_id = escape(node.id)
                safe_label = escape(node.label or node.id)
                self.console.print(f"  {safe_id}: {safe_label} [{status}]")

    def inspect_interactive(self, execution_id: str, workflow: WorkflowGraph):
        """
        Interactive inspection loop (REPL mode).

        Use this for debugging sessions where you want to explore multiple nodes.
        """
        self.console.print("\n[bold]Node Inspector (Interactive Mode)[/]")
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
                    self.console.print(f"[red]Node '{escape(cmd)}' not found[/]")

    def _list_nodes(self, workflow: WorkflowGraph):
        """List all nodes"""
        table = Table(title="Workflow Nodes")
        table.add_column("ID")
        table.add_column("Type")
        table.add_column("Label")

        for node in workflow.nodes:
            # SECURITY: Escape user-controlled strings
            table.add_row(
                escape(node.id),
                node.type.value,
                escape(node.label) if node.label else "-"
            )

        self.console.print(table)

    def _inspect_node(self, execution_id: str, node: Node):
        """Show detailed node information"""
        # SECURITY: Escape node ID and label
        safe_id = escape(node.id)
        safe_label = escape(node.label or '-')

        # Get execution data
        try:
            with self.db._connect() as conn:
                row = conn.execute(
                    "SELECT status, input_data, output_data, error FROM node_executions "
                    "WHERE execution_id=? AND node_id=?",
                    (execution_id, node.id)
                ).fetchone()
        except Exception as e:
            self.console.print(f"[red]DB error: {e}[/]")
            return

        if not row:
            self.console.print(f"[yellow]No execution data for {safe_id}[/]")
            return

        status, input_data, output_data, error = row

        # Node info panel - use Rich Group to preserve renderables
        # DO NOT stringify Rich objects - use Group to compose them
        info_parts = [
            Text.from_markup(f"[bold]ID:[/] {safe_id}"),
            Text.from_markup(f"[bold]Type:[/] {node.type.value}"),
            Text.from_markup(f"[bold]Label:[/] {safe_label}"),
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
        self.console.print(Panel(Group(*info_parts), title=f"Node: {safe_id}"))

        # Input data with JSON error handling
        if input_data:
            try:
                input_json = json.loads(input_data)
                self.console.print(Panel(
                    Syntax(json.dumps(input_json, indent=2), "json", theme="monokai"),
                    title="Input Data"
                ))
            except json.JSONDecodeError:
                self.console.print(Panel(
                    f"[dim]<Invalid JSON: {escape(str(input_data)[:100])}>[/]",
                    title="Input Data"
                ))

        # Output data with JSON error handling
        if output_data:
            try:
                output_json = json.loads(output_data)
                self.console.print(Panel(
                    Syntax(json.dumps(output_json, indent=2), "json", theme="monokai"),
                    title="Output Data"
                ))
            except json.JSONDecodeError:
                self.console.print(Panel(
                    f"[dim]<Invalid JSON: {escape(str(output_data)[:100])}>[/]",
                    title="Output Data"
                ))

        # Error - escape to prevent markup injection
        if error:
            self.console.print(Panel(
                f"[red]{escape(str(error))}[/]",
                title="Error",
                style="red"
            ))
```

---

## 2.5 CLI Commands

**File:** `supervisor/cli.py` (Extend existing commands)

**Purpose:** Extend existing CLI commands with terminal UI features.

**NOTE:** The `run_graph` and `status` commands already exist in `supervisor/cli.py`.
This section shows the **additions/modifications** needed, not complete replacements.
The existing `--validate-only` flag in `run_graph` is preserved.

### New Imports

Add to existing imports in `supervisor/cli.py`:

```python
from supervisor.cli_ui.graph_renderer import TerminalGraphRenderer, StatusTableRenderer
from supervisor.cli_ui.live_monitor import LiveExecutionMonitor
from supervisor.cli_ui.node_inspector import NodeInspector
```

### New Command: visualize

```python
@main.command()
@click.argument("workflow_file", type=click.Path(exists=True))
def visualize(workflow_file: str):
    """Visualize a workflow graph in the terminal"""
    import yaml
    from pydantic import ValidationError
    from rich.markup import escape

    console = Console()

    # Load and validate YAML with error handling
    try:
        with open(workflow_file) as f:
            data = yaml.safe_load(f)
        workflow = WorkflowGraph(**data)
    except yaml.YAMLError as e:
        console.print(f"[red]Invalid YAML:[/] {escape(str(e))}")
        return
    except ValidationError as e:
        console.print(f"[red]Schema validation failed:[/]")
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
    exit_str = ', '.join(escape(e) for e in workflow.exit_points) if workflow.exit_points else '(auto-detect)'
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
```

### Extend Existing: run_graph

Add `--live` option to the existing `run_graph` command. The existing
`--validate-only` flag is preserved.

```python
@main.command()
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--workflow-id", required=True, help="Unique workflow execution ID")
@click.option("--validate-only", is_flag=True, help="Only validate, don't execute")
@click.option("--live", is_flag=True, help="Show live execution monitor")
def run_graph(workflow_file: str, workflow_id: str, validate_only: bool, live: bool):
    """Execute a declarative workflow graph from YAML."""
    # ... existing YAML loading and validation code preserved ...

    if validate_only:
        return

    # Initialize (existing code)
    repo_path = get_repo_path()
    db_path = repo_path / ".supervisor" / "state.db"
    db = Database(db_path)
    engine = ExecutionEngine(repo_path, db=db)
    orchestrator = GraphOrchestrator(db, engine, engine.gate_executor, engine.gate_loader)

    async def run():
        exec_id = await orchestrator.start_workflow(workflow, workflow_id)

        if live:
            # NEW: Run with live monitor
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
                    return_exceptions=True
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

    status = asyncio.run(run())

    if status == "completed":
        click.secho("✓ Workflow completed", fg="green")
    else:
        click.secho(f"✗ Workflow {status}", fg="red")
```

### New Command: inspect

One-shot inspection by default; use `--interactive` for REPL mode.
This design follows standard CLI conventions for composability.

```python
@main.command()
@click.argument("execution_id")
@click.option("--node", "-n", help="Specific node ID to inspect")
@click.option("--interactive", "-i", is_flag=True, help="Enter interactive inspection mode")
def inspect(execution_id: str, node: str | None, interactive: bool):
    """Inspect an execution's node details.

    By default, shows a summary of all nodes (one-shot).
    Use --node to inspect a specific node.
    Use --interactive for REPL-style exploration.
    """
    repo_path = get_repo_path()
    db_path = repo_path / ".supervisor" / "state.db"
    db = Database(db_path)

    # Get workflow
    try:
        with db._connect() as conn:
            row = conn.execute("""
                SELECT w.definition FROM graph_workflows w
                JOIN graph_executions e ON w.id = e.graph_id
                WHERE e.id = ?
            """, (execution_id,)).fetchone()
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
```

### Extend Existing: status

Update the existing `status` stub to show execution details.

```python
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
        with db._connect() as conn:
            exec_row = conn.execute(
                "SELECT status, started_at, completed_at, error FROM graph_executions WHERE id=?",
                (execution_id,)
            ).fetchone()

            if not exec_row:
                console.print(f"[red]Execution '{safe_exec_id}' not found[/]")
                return

            workflow_row = conn.execute("""
                SELECT w.definition FROM graph_workflows w
                JOIN graph_executions e ON w.id = e.graph_id WHERE e.id = ?
            """, (execution_id,)).fetchone()

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
            "cancelled": "yellow"
        }.get(exec_row[0], "white")

        # SECURITY: Escape error message which may contain user data
        safe_error = escape(str(exec_row[3])) if exec_row[3] else '-'

        console.print(Panel(
            f"[bold]Status:[/] [{status_color}]{exec_row[0]}[/]\n"
            f"[bold]Started:[/] {exec_row[1]}\n"
            f"[bold]Completed:[/] {exec_row[2] or '-'}\n"
            f"[bold]Error:[/] {safe_error}",
            title=f"Execution: {safe_exec_id[:8]}..."
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
    else:
        # List recent executions (existing functionality)
        console.print(
            Panel(
                f"Database: {db_path}\nWorkflow ID filter: {workflow_id or 'All'}",
                title="Supervisor Status",
            )
        )
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

### Inspect Execution (One-shot)

```bash
# Show summary of all nodes
$ supervisor inspect abc123
┌─────────┬─────────┬───────────────┐
│ ID      │ Type    │ Label         │
├─────────┼─────────┼───────────────┤
│ analyze │ task    │ Analyze Bug   │
│ fix     │ task    │ Apply Fix     │
│ test    │ gate    │ Run Tests     │
│ check   │ branch  │ Check Result  │
│ review  │ task    │ Final Review  │
└─────────┴─────────┴───────────────┘
  analyze: Analyze Bug [completed]
  fix: Apply Fix [completed]
  test: Run Tests [completed]
  check: Check Result [completed]
  review: Final Review [pending]

# Inspect specific node
$ supervisor inspect abc123 --node fix
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
```

### Inspect Execution (Interactive Mode)

```bash
$ supervisor inspect abc123 --interactive

Node Inspector (Interactive Mode)
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

inspect> graph
Workflow: Debug with Retry Loop (v1.0.0)
├── [✓] Analyze Bug
│   └── [✓] Apply Fix
│       └── [✓] Run Tests
│           └── [✓] Check Test Result
│               └── [ ] Final Review

inspect> quit
```

---

## 2.7 Dependencies

**NOTE:** The required dependencies are already in `pyproject.toml` from Phase 1:

```toml
[project.dependencies]
rich = ">=13.0"      # Already present
networkx = ">=3.0"   # Already present
```

No additional dependencies are required for Phase 2 core functionality.

Optional for advanced TUI (future enhancement):
```toml
[project.optional-dependencies]
tui = [
    "textual>=0.50.0",  # For full TUI application
]
```

---

## 2.8 Design Decisions

This section documents intentional design choices made during code review.

### render_graph() Simplified Visualization (DECLINED: Gemini)

**Issue:** The ASCII `render_graph()` shows topological levels but doesn't draw exact edge connections, which could be misleading.

**Decision:** DECLINED - This is intentional. Drawing crossing ASCII lines for arbitrary DAGs is complex and often produces unreadable output. The `render_as_tree()` method is the primary visualization tool and shows accurate structure including edge conditions. The docstring clarifies this distinction:

> "NOTE: render_graph() provides a simplified ASCII view that shows topological levels
> but does not visualize exact edge connections (would require complex ASCII line drawing).
> Use render_as_tree() for accurate structural visualization including edge conditions."

### Cyclic Graph Fallback (DECLINED: Gemini)

**Issue:** Cyclic graphs in `render_graph()` fall back to a single row display.

**Decision:** DECLINED - Cyclic graphs are invalid for DAG-based workflows (our target use case). The fallback is a graceful degradation rather than crashing. The `render_as_tree()` method properly handles back-edges with `visited` tracking and displays `↩ node_id (loop)` indicators.

### Fixed Output Truncation (DECLINED: Gemini)

**Issue:** Output is manually truncated to fixed lengths (40/100 chars) instead of using Rich's dynamic overflow handling.

**Decision:** DECLINED - Fixed truncation is intentional for `LiveExecutionMonitor` where Rich's automatic layout handling can cause layout instability in the `Live` context with frequent updates. Consistent column widths provide a more stable visual experience. The full output is available via the `inspect` command.

### Tree Rendering Visited Copy (DECLINED: Codex)

**Issue:** Tree rendering copies `visited` set for every edge, which is O(N) per edge.

**Decision:** DECLINED - This is the correct behavior for DAG visualization. When the same node is reachable via multiple paths (e.g., A→C and B→C), we intentionally render it in both subtrees to show the complete structure from each path. The `visited.copy()` ensures each path is tracked independently. For the typical workflow sizes we handle (<100 nodes), this is not a performance concern. The `max_depth` parameter provides a safety limit for pathological cases.

---

**End of Phase 2 Plan**
