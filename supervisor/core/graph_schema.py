"""Graph workflow schema definitions using Pydantic models.

This module defines the declarative graph workflow schema for Supervisor Studio Phase 1.
Workflows are defined as directed graphs with typed nodes (TASK, GATE, BRANCH, etc.)
and edges with optional conditions.

Security-first design:
- No arbitrary code execution in conditions (structured operators only)
- Loop protection via max_iterations
- Comprehensive validation before execution
"""

import re
from enum import Enum
from typing import Literal

import networkx as nx
from pydantic import BaseModel, Field, field_validator, model_validator


class NodeType(str, Enum):
    """Supported node types in workflow graphs"""

    TASK = "task"  # Execute a role (planner, implementer, reviewer, etc.)
    GATE = "gate"  # Verification gate (test, lint, security, etc.)
    BRANCH = "branch"  # Conditional branching with loop support
    MERGE = "merge"  # Synchronization point for parallel branches
    PARALLEL = "parallel"  # Split into parallel execution branches
    SUBGRAPH = "subgraph"  # Nested workflow execution
    HUMAN = "human"  # Human approval/intervention point


class NodeStatus(str, Enum):
    """Execution status for nodes"""

    PENDING = "pending"  # Not yet ready (dependencies not met)
    READY = "ready"  # Ready to execute (all dependencies satisfied)
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Execution failed
    SKIPPED = "skipped"  # Skipped due to branch conditions


class BranchOutcome(str, Enum):
    """Possible outcomes from BRANCH nodes"""

    TRUE = "on_true"
    FALSE = "on_false"
    MAX_ITERATIONS = "max_iterations_reached"


class TransitionCondition(BaseModel):
    """
    Safe, declarative condition evaluation for edges.
    NO arbitrary code execution - only structured operators.
    """

    field: str  # Supports dotted notation: "source_node.field_name"
    operator: Literal[
        "==", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "starts_with", "ends_with"
    ]
    value: str | int | float | bool | list[str | int | float | bool]

    @field_validator("field")
    @classmethod
    def validate_field(cls, v):
        """Ensure field names are safe dot-separated identifiers.

        Valid: "status", "node_1.result", "a.b.c"
        Invalid: "..", "a..b", ".foo", "foo.", "a-b"
        """
        # Pattern: identifier (letter/underscore + alphanumeric/underscore) optionally
        # followed by .identifier sequences
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid field name: {v}")
        return v

    @model_validator(mode="after")
    def check_value_type_for_operator(self) -> "TransitionCondition":
        """Ensure value type is compatible with the operator."""
        list_operators = {"in", "not_in"}
        is_list_op = self.operator in list_operators
        is_list_val = isinstance(self.value, list)

        if is_list_op and not is_list_val:
            raise ValueError(f"Operator '{self.operator}' requires value to be a list.")
        if not is_list_op and is_list_val:
            raise ValueError(f"Operator '{self.operator}' does not support list values.")
        return self


class LoopCondition(BaseModel):
    """
    Safe loop condition for BRANCH nodes.
    Replaces unsafe string evaluation with structured conditions.
    """

    field: str  # Field to evaluate from previous node output
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in", "not_in"]
    value: str | int | float | bool | list[str | int | float | bool]
    max_iterations: int = 10  # CRITICAL: Prevent infinite loops


class Edge(BaseModel):
    """Directed edge between nodes with optional conditions and data mapping"""

    id: str
    source: str  # Source node ID
    target: str  # Target node ID
    condition: TransitionCondition | None = None  # For conditional edges
    data_mapping: dict[str, str] | None = None  # Map outputs to inputs
    is_loop_edge: bool = False  # Marks edges that form loops (for re-execution)


class TaskNodeConfig(BaseModel):
    """Configuration for TASK nodes - execute a role"""

    role: str  # Role name (planner, implementer, reviewer, debugger, etc.)
    task_template: str = "{input}"  # Template for task description formatting
    timeout: int | None = None  # Execution timeout in seconds
    gates: list[str] = Field(default_factory=list)  # Gates to run after task (pre-apply safety)
    worktree_id: str | None = None  # Optional worktree for isolation


class GateNodeConfig(BaseModel):
    """Configuration for GATE nodes - verification gates"""

    gate_type: str  # Gate name from gates.yaml (test_gate, lint_gate, etc.)
    auto_approve: bool = False  # Skip human approval if gate passes
    requires_worktree: bool = True  # Whether gate needs worktree context


class BranchNodeConfig(BaseModel):
    """
    Configuration for BRANCH nodes.
    Uses declarative LoopCondition instead of unsafe string evaluation.
    """

    condition: LoopCondition  # Safe, structured condition
    on_true: str  # Target node ID if condition is true
    on_false: str  # Target node ID if condition is false


class MergeNodeConfig(BaseModel):
    """
    Configuration for MERGE nodes - synchronization points.
    """

    wait_for: Literal["all", "any"] = "all"  # Wait for all/any incoming edges
    merge_strategy: Literal["union", "intersection", "first"] = "union"
    # union: merge all inputs (later overwrites earlier)
    # intersection: only common keys
    # first: first completed input wins


class ParallelNodeConfig(BaseModel):
    """
    Configuration for PARALLEL nodes - fan-out execution.

    Design: PARALLEL acts as a FORK that completes immediately, allowing
    concurrent execution of downstream branches. Synchronization is handled
    by a downstream MERGE node, not by PARALLEL itself.

    The actual fan-out is determined by outgoing edges in the graph definition.
    """

    branches: list[str] = Field(default_factory=list)  # Reserved: explicit branch list
    wait_for: Literal["all", "any", "first"] = "all"  # Reserved: sync handled by MERGE


class SubgraphNodeConfig(BaseModel):
    """Configuration for SUBGRAPH nodes - nested workflows"""

    workflow_name: str  # Name of nested workflow to execute
    input_mapping: dict[str, str] = Field(default_factory=dict)  # Map parent inputs to child
    output_mapping: dict[str, str] = Field(default_factory=dict)  # Map child outputs to parent


class HumanNodeConfig(BaseModel):
    """Configuration for HUMAN nodes - approval points"""

    title: str  # Title shown in approval UI
    description: str | None = None  # Additional context


class Node(BaseModel):
    """Generic graph node with type-specific configuration"""

    id: str
    type: NodeType

    @field_validator("id")
    @classmethod
    def validate_node_id(cls, v):
        """Ensure node ID is a valid Python identifier.

        This validation ensures node IDs are clean, consistent keys that work
        reliably across the system. While format_map doesn't strictly require
        identifiers, this constraint:
        - Ensures compatibility with format(**kwargs) if ever needed
        - Prevents edge cases with special characters in dict keys
        - Maintains consistency with dot-notation field references
        """
        if not v.isidentifier():
            raise ValueError(f"Invalid node ID: '{v}'. Must be a valid Python identifier.")
        return v

    label: str | None = None
    description: str | None = None

    # Type-specific configuration (only one should be set based on type)
    task_config: TaskNodeConfig | None = None
    gate_config: GateNodeConfig | None = None
    branch_config: BranchNodeConfig | None = None
    merge_config: MergeNodeConfig | None = None
    parallel_config: ParallelNodeConfig | None = None
    subgraph_config: SubgraphNodeConfig | None = None
    human_config: HumanNodeConfig | None = None

    # UI metadata (position, styling) for visual editor
    ui_metadata: dict | None = None

    # Readiness policy for nodes with multiple incoming edges
    wait_for_incoming: Literal["all", "any"] = "any"

    @model_validator(mode="after")
    def validate_config_for_type(self) -> "Node":
        """Ensure the correct config is present for the node type."""
        config_map = {
            NodeType.TASK: ("task_config", self.task_config),
            NodeType.GATE: ("gate_config", self.gate_config),
            NodeType.BRANCH: ("branch_config", self.branch_config),
            NodeType.MERGE: ("merge_config", self.merge_config),
            NodeType.PARALLEL: ("parallel_config", self.parallel_config),
            NodeType.SUBGRAPH: ("subgraph_config", self.subgraph_config),
            NodeType.HUMAN: ("human_config", self.human_config),
        }

        expected_config_name, expected_config = config_map.get(self.type, (None, None))

        if expected_config_name and expected_config is None:
            raise ValueError(
                f"Node '{self.id}' of type '{self.type.value}' requires '{expected_config_name}'"
            )

        # Check that no other configs are set (only the matching one)
        for node_type, (config_name, config_value) in config_map.items():
            if node_type != self.type and config_value is not None:
                raise ValueError(
                    f"Node '{self.id}' of type '{self.type.value}' has unexpected "
                    f"'{config_name}' (should only have '{expected_config_name}')"
                )

        return self


class WorkflowGraph(BaseModel):
    """Complete workflow definition"""

    id: str
    name: str
    description: str | None = None
    version: str = "1.0.0"

    nodes: list[Node]
    edges: list[Edge]

    entry_point: str  # Starting node ID
    exit_points: list[str] = Field(default_factory=list)  # Terminal node IDs (auto-detected)

    # Global configuration
    config: dict = Field(
        default_factory=lambda: {
            "max_parallel_nodes": 4,  # Limit concurrent execution
            "fail_fast": True,  # Stop on first failure
        }
    )

    def validate_graph(self) -> list[str]:
        """
        Validate graph structure using NetworkX.
        Returns list of validation errors.
        """
        errors = []

        # Check for duplicate node IDs (critical - would corrupt execution state)
        seen_node_ids = set()
        for node in self.nodes:
            if node.id in seen_node_ids:
                errors.append(f"Duplicate node ID: '{node.id}'")
            seen_node_ids.add(node.id)
        node_ids = seen_node_ids

        # Check for duplicate edge IDs
        seen_edge_ids = set()
        for edge in self.edges:
            if edge.id in seen_edge_ids:
                errors.append(f"Duplicate edge ID: '{edge.id}'")
            seen_edge_ids.add(edge.id)

        # Check for duplicate edges (same source->target)
        seen_edge_pairs = set()
        for edge in self.edges:
            pair = (edge.source, edge.target)
            if pair in seen_edge_pairs:
                errors.append(f"Duplicate edge from '{edge.source}' to '{edge.target}'")
            seen_edge_pairs.add(pair)

        # Check entry point exists
        if self.entry_point not in node_ids:
            errors.append(f"Entry point '{self.entry_point}' not found")

        # Check all edge endpoints exist
        for edge in self.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge {edge.id}: source '{edge.source}' not found")
            if edge.target not in node_ids:
                errors.append(f"Edge {edge.id}: target '{edge.target}' not found")

        # Build NetworkX graph for advanced validation
        G = self._to_networkx()

        # Auto-detect exit points if not specified (compute but don't mutate self)
        exit_points = self.exit_points
        if not exit_points:
            exit_points = [n for n in node_ids if G.out_degree(n) == 0]
            if not exit_points:
                errors.append("No exit points found (no terminal nodes)")

        # Check for cycles without loop control
        # Limit cycle enumeration to prevent DoS on complex graphs
        MAX_CYCLES_TO_CHECK = 100
        MAX_NODES_FOR_FULL_CYCLE_CHECK = 50
        try:
            if len(self.nodes) > MAX_NODES_FOR_FULL_CYCLE_CHECK:
                # For large graphs, just check if any cycle exists
                try:
                    cycle = nx.find_cycle(G)
                    # Found a cycle - check if ANY branch node has loop control
                    has_any_loop_control = any(
                        n.type == NodeType.BRANCH
                        and n.branch_config
                        and n.branch_config.condition.max_iterations > 0
                        and any(e.is_loop_edge for e in self.edges if e.source == n.id)
                        for n in self.nodes
                    )
                    if not has_any_loop_control:
                        cycle_path = " -> ".join(edge[0] for edge in cycle)
                        errors.append(
                            f"Cycle detected without loop control: {cycle_path}... "
                            f"(graph too large for full cycle analysis)"
                        )
                except nx.NetworkXNoCycle:
                    pass  # No cycles - OK
            else:
                # For small graphs, enumerate all cycles
                for cycle_count, cycle in enumerate(nx.simple_cycles(G), start=1):
                    if cycle_count > MAX_CYCLES_TO_CHECK:
                        errors.append(
                            f"Too many cycles to validate (>{MAX_CYCLES_TO_CHECK}). "
                            f"Simplify graph structure."
                        )
                        break
                    # A cycle has proper loop control if:
                    # 1. There's a BRANCH node in the cycle
                    # 2. That BRANCH node has max_iterations > 0
                    # 3. The loop edge originates from that BRANCH node
                    has_loop_control = any(
                        n.type == NodeType.BRANCH
                        and n.branch_config
                        and n.branch_config.condition.max_iterations > 0
                        and any(
                            e.is_loop_edge and e.target in cycle
                            for e in self.edges
                            if e.source == n.id
                        )
                        for n in self.nodes
                        if n.id in cycle
                    )
                    if not has_loop_control:
                        errors.append(f"Cycle without loop control: {' -> '.join(cycle)}")
        except nx.NetworkXError as e:
            errors.append(f"Could not perform cycle detection: {e}")

        # Validate BRANCH node targets exist and have edges
        for node in self.nodes:
            if node.type == NodeType.BRANCH and node.branch_config:
                config = node.branch_config
                # Check on_true target exists
                if config.on_true not in node_ids:
                    errors.append(
                        f"BRANCH node '{node.id}': on_true target '{config.on_true}' not found"
                    )
                # Check on_false target exists
                if config.on_false not in node_ids:
                    errors.append(
                        f"BRANCH node '{node.id}': on_false target '{config.on_false}' not found"
                    )
                # Check edges exist to targets
                outgoing_targets = {e.target for e in self.edges if e.source == node.id}
                if config.on_true not in outgoing_targets:
                    errors.append(
                        f"BRANCH node '{node.id}': missing edge to on_true target '{config.on_true}'"
                    )
                if config.on_false not in outgoing_targets:
                    errors.append(
                        f"BRANCH node '{node.id}': missing edge to on_false target '{config.on_false}'"
                    )

        # Validate MERGE nodes have multiple incoming edges
        for node in self.nodes:
            if node.type == NodeType.MERGE:
                incoming = [e for e in self.edges if e.target == node.id]
                if len(incoming) < 2:
                    errors.append(f"MERGE node '{node.id}' should have at least 2 incoming edges")

        # Warn about SUBGRAPH nodes (not yet implemented)
        # This prevents users from authoring workflows that will fail at runtime
        for node in self.nodes:
            if node.type == NodeType.SUBGRAPH:
                errors.append(
                    f"SUBGRAPH node '{node.id}' is not yet implemented. "
                    f"Remove this node or use a different workflow pattern."
                )

        return errors

    def _to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph for analysis"""
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.id)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target)
        return G

    def get_terminal_nodes(self) -> set[str]:
        """Find nodes with no outgoing edges"""
        G = self._to_networkx()
        return {n for n in G.nodes() if G.out_degree(n) == 0}

    def analyze_parallelism(self) -> list[list[str]]:
        """Find nodes that can execute in parallel (topological levels)"""
        G = self._to_networkx()
        try:
            return [list(level) for level in nx.topological_generations(G)]
        except nx.NetworkXError:
            return []  # Has cycles

    def find_critical_path(self) -> list[str]:
        """Find longest path through graph"""
        G = self._to_networkx()
        try:
            return nx.dag_longest_path(G)
        except nx.NetworkXError:
            return []  # Has cycles
