"""Graph workflow schema definitions using Pydantic models.

This module defines the declarative graph workflow schema for Supervisor Studio Phase 1.
Workflows are defined as directed graphs with typed nodes (TASK, GATE, BRANCH, etc.)
and edges with optional conditions.

Security-first design:
- No arbitrary code execution in conditions (structured operators only)
- Loop protection via max_iterations
- Comprehensive validation before execution
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Literal, Union, Set
from enum import Enum
import networkx as nx


class NodeType(str, Enum):
    """Supported node types in workflow graphs"""
    TASK = "task"              # Execute a role (planner, implementer, reviewer, etc.)
    GATE = "gate"              # Verification gate (test, lint, security, etc.)
    BRANCH = "branch"          # Conditional branching with loop support
    MERGE = "merge"            # Synchronization point for parallel branches
    PARALLEL = "parallel"      # Split into parallel execution branches
    SUBGRAPH = "subgraph"      # Nested workflow execution
    HUMAN = "human"            # Human approval/intervention point


class NodeStatus(str, Enum):
    """Execution status for nodes"""
    PENDING = "pending"        # Not yet ready (dependencies not met)
    READY = "ready"            # Ready to execute (all dependencies satisfied)
    RUNNING = "running"        # Currently executing
    COMPLETED = "completed"    # Successfully completed
    FAILED = "failed"          # Execution failed
    SKIPPED = "skipped"        # Skipped due to branch conditions


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
    field: str                 # Supports dotted notation: "source_node.field_name"
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "starts_with", "ends_with"]
    value: Union[str, int, float, bool, List[Union[str, int, float, bool]]]

    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        """Ensure field names are safe dot-separated identifiers.

        Valid: "status", "node_1.result", "a.b.c"
        Invalid: "..", "a..b", ".foo", "foo.", "a-b"
        """
        import re
        # Pattern: identifier (letter/underscore + alphanumeric/underscore) optionally
        # followed by .identifier sequences
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid field name: {v}")
        return v


class LoopCondition(BaseModel):
    """
    Safe loop condition for BRANCH nodes.
    Replaces unsafe string evaluation with structured conditions.
    """
    field: str                 # Field to evaluate from previous node output
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in", "not_in"]
    value: Union[str, int, float, bool, List[Union[str, int, float, bool]]]
    max_iterations: int = 10   # CRITICAL: Prevent infinite loops


class Edge(BaseModel):
    """Directed edge between nodes with optional conditions and data mapping"""
    id: str
    source: str                # Source node ID
    target: str                # Target node ID
    condition: Optional[TransitionCondition] = None  # For conditional edges
    data_mapping: Optional[Dict[str, str]] = None    # Map outputs to inputs
    is_loop_edge: bool = False  # Marks edges that form loops (for re-execution)


class TaskNodeConfig(BaseModel):
    """Configuration for TASK nodes - execute a role"""
    role: str                  # Role name (planner, implementer, reviewer, debugger, etc.)
    task_template: str = "{input}"  # Template for task description formatting
    timeout: Optional[int] = None   # Execution timeout in seconds
    gates: List[str] = []      # Gates to run after task (pre-apply safety)
    worktree_id: Optional[str] = None  # Optional worktree for isolation


class GateNodeConfig(BaseModel):
    """Configuration for GATE nodes - verification gates"""
    gate_type: str             # Gate name from gates.yaml (test_gate, lint_gate, etc.)
    auto_approve: bool = False # Skip human approval if gate passes
    requires_worktree: bool = True  # Whether gate needs worktree context


class BranchNodeConfig(BaseModel):
    """
    Configuration for BRANCH nodes.
    Uses declarative LoopCondition instead of unsafe string evaluation.
    """
    condition: LoopCondition   # Safe, structured condition
    on_true: str               # Target node ID if condition is true
    on_false: str              # Target node ID if condition is false


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
    """Configuration for PARALLEL nodes - fan-out execution"""
    branches: List[str]        # List of node IDs to execute in parallel
    wait_for: Literal["all", "any", "first"] = "all"


class SubgraphNodeConfig(BaseModel):
    """Configuration for SUBGRAPH nodes - nested workflows"""
    workflow_name: str         # Name of nested workflow to execute
    input_mapping: Dict[str, str] = {}   # Map parent inputs to child inputs
    output_mapping: Dict[str, str] = {}  # Map child outputs to parent outputs


class HumanNodeConfig(BaseModel):
    """Configuration for HUMAN nodes - approval points"""
    title: str                 # Title shown in approval UI
    description: Optional[str] = None  # Additional context


class Node(BaseModel):
    """Generic graph node with type-specific configuration"""
    id: str
    type: NodeType
    label: Optional[str] = None
    description: Optional[str] = None

    # Type-specific configuration (only one should be set based on type)
    task_config: Optional[TaskNodeConfig] = None
    gate_config: Optional[GateNodeConfig] = None
    branch_config: Optional[BranchNodeConfig] = None
    merge_config: Optional[MergeNodeConfig] = None
    parallel_config: Optional[ParallelNodeConfig] = None
    subgraph_config: Optional[SubgraphNodeConfig] = None
    human_config: Optional[HumanNodeConfig] = None

    # UI metadata (position, styling) for visual editor
    ui_metadata: Optional[Dict] = None

    # Readiness policy for nodes with multiple incoming edges
    wait_for_incoming: Literal["all", "any"] = "any"

    @model_validator(mode='after')
    def validate_config_for_type(self) -> 'Node':
        """Ensure the correct config is present for the node type."""
        config_map = {
            NodeType.TASK: ('task_config', self.task_config),
            NodeType.GATE: ('gate_config', self.gate_config),
            NodeType.BRANCH: ('branch_config', self.branch_config),
            NodeType.MERGE: ('merge_config', self.merge_config),
            NodeType.PARALLEL: ('parallel_config', self.parallel_config),
            NodeType.SUBGRAPH: ('subgraph_config', self.subgraph_config),
            NodeType.HUMAN: ('human_config', self.human_config),
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
    description: Optional[str] = None
    version: str = "1.0.0"

    nodes: List[Node]
    edges: List[Edge]

    entry_point: str           # Starting node ID
    exit_points: List[str] = []  # Terminal node IDs (auto-detected if empty)

    # Global configuration
    config: Dict = Field(default_factory=lambda: {
        "max_parallel_nodes": 4,      # Limit concurrent execution
        "fail_fast": True             # Stop on first failure
    })

    def validate_graph(self) -> List[str]:
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

        # Auto-detect exit points if not specified
        if not self.exit_points:
            self.exit_points = [n for n in node_ids if G.out_degree(n) == 0]
            if not self.exit_points:
                errors.append("No exit points found (no terminal nodes)")

        # Check for cycles without loop control
        try:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                has_loop_control = any(
                    n.type == NodeType.BRANCH and n.branch_config and
                    n.branch_config.condition.max_iterations > 0 and
                    any(e.is_loop_edge for e in self.edges if e.source in cycle)
                    for n in self.nodes if n.id in cycle
                )
                if not has_loop_control:
                    errors.append(f"Cycle without loop control: {' -> '.join(cycle)}")
        except Exception:
            pass  # Graph may have issues preventing cycle detection

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

    def get_terminal_nodes(self) -> Set[str]:
        """Find nodes with no outgoing edges"""
        G = self._to_networkx()
        return {n for n in G.nodes() if G.out_degree(n) == 0}

    def analyze_parallelism(self) -> List[List[str]]:
        """Find nodes that can execute in parallel (topological levels)"""
        G = self._to_networkx()
        try:
            return [list(level) for level in nx.topological_generations(G)]
        except nx.NetworkXError:
            return []  # Has cycles

    def find_critical_path(self) -> List[str]:
        """Find longest path through graph"""
        G = self._to_networkx()
        try:
            return nx.dag_longest_path(G)
        except nx.NetworkXError:
            return []  # Has cycles
