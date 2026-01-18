"""Terminal graph rendering for workflow visualization.

Provides ASCII art and tree-based visualization of workflow graphs using Rich.
"""

from typing import Any

import networkx as nx
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.tree import Tree

from supervisor.core.graph_schema import Edge, Node, NodeStatus, NodeType, WorkflowGraph


class TerminalGraphRenderer:
    """
    Renders workflow graphs as ASCII art in the terminal.

    Features:
    - Topological layout (left-to-right flow)
    - Color-coded node types
    - Status indicators
    - Edge labels for conditions (render_as_tree only)

    NOTE: render_graph() provides a simplified ASCII view that shows topological levels
    but does not visualize exact edge connections (would require complex ASCII line drawing).
    Use render_as_tree() for accurate structural visualization including edge conditions.

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
    def _normalize_status(status: NodeStatus | str | None) -> str:
        """Normalize status to string for consistent lookup."""
        if isinstance(status, NodeStatus):
            return status.value
        return str(status) if status else "pending"

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def _build_node_map(self, workflow: WorkflowGraph) -> dict[str, Node]:
        """Build O(1) lookup map for nodes by ID."""
        return {n.id: n for n in workflow.nodes}

    def _build_edge_map(self, workflow: WorkflowGraph) -> dict[str, list[Edge]]:
        """Build O(1) lookup map for outgoing edges by source node ID.

        Performance: Precomputing this avoids O(N*E) traversal in tree rendering,
        reducing to O(N+E) total instead.
        """
        edge_map: dict[str, list[Edge]] = {n.id: [] for n in workflow.nodes}
        for edge in workflow.edges:
            if edge.source in edge_map:
                edge_map[edge.source].append(edge)
        return edge_map

    def render_graph(
        self,
        workflow: WorkflowGraph,
        statuses: dict[str, NodeStatus | str] | None = None,
    ) -> str:
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

            # Add connector arrows if not last level
            if level_idx < len(levels) - 1:
                lines.append("  " + "  |  " * len(level_nodes))
                lines.append("  " + "  v  " * len(level_nodes))

        return "\n".join(lines)

    def render_as_tree(
        self,
        workflow: WorkflowGraph,
        statuses: dict[str, NodeStatus | str] | None = None,
        max_depth: int = 50,
    ) -> Tree:
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
        # Build edge map for O(1) outgoing edge lookups (avoids O(N*E) traversal)
        edge_map = self._build_edge_map(workflow)

        # Find entry point with graceful fallback
        entry = node_map.get(workflow.entry_point)
        if not entry:
            tree.add("[red]Error: Entry point node not found[/]")
            return tree

        self._add_node_to_tree(
            tree,
            entry,
            statuses,
            node_map,
            edge_map,
            visited=set(),
            depth=0,
            max_depth=max_depth,
        )

        return tree

    def _add_node_to_tree(
        self,
        parent: Tree,
        node: Node,
        statuses: dict[str, str] | None,
        node_map: dict[str, Node],
        edge_map: dict[str, list[Edge]],
        visited: set,
        depth: int = 0,
        max_depth: int = 50,
    ):
        """Recursively add nodes to tree with depth limiting to prevent exponential blow-up.

        Performance: Uses precomputed edge_map for O(1) outgoing edge lookup instead of
        scanning all edges (O(E)) per node.
        """
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

        # Add children using precomputed edge_map for O(1) lookup
        outgoing = edge_map.get(node.id, [])
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
                    self._add_node_to_tree(
                        edge_branch,
                        child_node,
                        statuses,
                        node_map,
                        edge_map,
                        visited.copy(),
                        depth + 1,
                        max_depth,
                    )
                else:
                    self._add_node_to_tree(
                        branch,
                        child_node,
                        statuses,
                        node_map,
                        edge_map,
                        visited.copy(),
                        depth + 1,
                        max_depth,
                    )


class StatusTableRenderer:
    """Renders node execution status as a Rich table.

    SECURITY: All user-controlled strings (node labels, outputs, execution_id) are escaped
    to prevent Rich markup injection.
    """

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def render_status_table(
        self,
        workflow: WorkflowGraph,
        execution_id: str,
        statuses: dict[str, NodeStatus | str],
        outputs: dict[str, Any] | None = None,
    ) -> Table:
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
            # Normalize status to string for consistent comparison
            # (callers may pass NodeStatus enums or strings from DB)
            raw_status = statuses.get(node.id, "pending")
            status = TerminalGraphRenderer._normalize_status(raw_status)
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

            table.add_row(safe_label, node.type.value, status_text, output_str)

        return table
