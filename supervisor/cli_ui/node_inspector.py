"""Node inspection for workflow debugging.

Provides one-shot and interactive modes for inspecting node execution details.
"""

import json

from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from supervisor.cli_ui.graph_renderer import TerminalGraphRenderer
from supervisor.core.graph_schema import Node, WorkflowGraph
from supervisor.core.state import Database


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

    def _get_statuses(self, execution_id: str) -> dict[str, str]:
        """Get all node statuses for display."""
        try:
            with self.db._connect() as conn:
                rows = conn.execute(
                    "SELECT node_id, status FROM node_executions WHERE execution_id=?",
                    (execution_id,),
                ).fetchall()
                return {row[0]: row[1] for row in rows}
        except Exception:
            return {}

    def inspect_node(
        self,
        execution_id: str,
        workflow: WorkflowGraph,
        node_id: str | None = None,
    ):
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
                # SECURITY: Escape status from DB to prevent Rich markup injection
                safe_status = escape(str(status))
                self.console.print(f"  {safe_id}: {safe_label} [{safe_status}]")

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
                escape(node.label) if node.label else "-",
            )

        self.console.print(table)

    def _inspect_node(self, execution_id: str, node: Node):
        """Show detailed node information"""
        # SECURITY: Escape node ID and label
        safe_id = escape(node.id)
        safe_label = escape(node.label or "-")

        # Get execution data
        try:
            with self.db._connect() as conn:
                row = conn.execute(
                    "SELECT status, input_data, output_data, error FROM node_executions "
                    "WHERE execution_id=? AND node_id=?",
                    (execution_id, node.id),
                ).fetchone()
        except Exception as e:
            # SECURITY: Escape exception message to prevent Rich markup injection
            self.console.print(f"[red]DB error: {escape(str(e))}[/]")
            return

        if not row:
            self.console.print(f"[yellow]No execution data for {safe_id}[/]")
            return

        status, input_data, output_data, error = row

        # Node info panel - use Rich Group to preserve renderables
        # DO NOT stringify Rich objects - use Group to compose them
        # SECURITY: Escape status from DB to prevent Rich markup injection
        safe_status = escape(str(status))
        info_parts = [
            Text.from_markup(f"[bold]ID:[/] {safe_id}"),
            Text.from_markup(f"[bold]Type:[/] {node.type.value}"),
            Text.from_markup(f"[bold]Label:[/] {safe_label}"),
            Text.from_markup(f"[bold]Status:[/] {safe_status}"),
        ]

        # Add type-specific config
        config = (
            node.task_config
            or node.gate_config
            or node.branch_config
            or node.merge_config
            or node.parallel_config
            or node.subgraph_config
            or node.human_config
        )
        if config:
            info_parts.append(Text(""))  # Empty line
            info_parts.append(Text.from_markup("[bold]Configuration:[/]"))
            # NOTE: Use model_dump_json() instead of json.dumps(model_dump()) to
            # handle non-JSON-serializable types (datetime, UUID, Path, etc.)
            config_json = config.model_dump_json(indent=2)
            info_parts.append(Syntax(config_json, "json", theme="monokai"))

        # Use Group to compose multiple renderables without stringifying
        self.console.print(Panel(Group(*info_parts), title=f"Node: {safe_id}"))

        # Input data with JSON error handling
        # NOTE: Catch multiple error types - SQLite adapters may return bytes or other types
        # - JSONDecodeError: invalid JSON
        # - TypeError: json.loads(None) or non-string types
        # - ValueError: other value-related JSON errors
        # - UnicodeDecodeError: non-UTF-8 bytes from SQLite BLOB/adapters
        if input_data:
            try:
                input_json = json.loads(input_data)
                self.console.print(
                    Panel(
                        Syntax(json.dumps(input_json, indent=2), "json", theme="monokai"),
                        title="Input Data",
                    )
                )
            except (json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError):
                self.console.print(
                    Panel(
                        f"[dim]<Invalid JSON: {escape(str(input_data)[:100])}>[/]",
                        title="Input Data",
                    )
                )

        # Output data with JSON error handling
        # NOTE: Catch multiple error types - SQLite adapters may return bytes or other types
        # - JSONDecodeError: invalid JSON
        # - TypeError: json.loads(None) or non-string types
        # - ValueError: other value-related JSON errors
        # - UnicodeDecodeError: non-UTF-8 bytes from SQLite BLOB/adapters
        if output_data:
            try:
                output_json = json.loads(output_data)
                self.console.print(
                    Panel(
                        Syntax(json.dumps(output_json, indent=2), "json", theme="monokai"),
                        title="Output Data",
                    )
                )
            except (json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError):
                self.console.print(
                    Panel(
                        f"[dim]<Invalid JSON: {escape(str(output_data)[:100])}>[/]",
                        title="Output Data",
                    )
                )

        # Error - escape to prevent markup injection
        if error:
            self.console.print(Panel(f"[red]{escape(str(error))}[/]", title="Error", style="red"))
