# Graph Workflow Schema

**File:** `supervisor/core/graph_schema.py`

## Overview

Defines the declarative workflow graph schema used by Supervisor Studio and the `run-graph` CLI. Workflows are directed graphs with typed nodes, typed edges, and structured conditions (no arbitrary code execution).

## Key Types

### `WorkflowGraph`

Top-level graph model containing:
- `nodes`: list of `Node`
- `edges`: list of `Edge`
- `entry_point`: node id
- `exit_points`: optional end nodes
- Validation helpers (`validate_graph`, `_to_networkx`) for DAG correctness and reachability

### `NodeType`

Supported node types:
- `task`, `gate`, `branch`, `merge`, `parallel`, `subgraph`, `human`

### `NodeStatus`

Execution statuses: `pending`, `ready`, `running`, `completed`, `failed`, `skipped`.

### `Edge`

Directed edge between nodes:
- `source`, `target`
- Optional `condition` (`TransitionCondition`)
- Optional `data_mapping` for input/output wiring
- `is_loop_edge` for loop tracking

### `TransitionCondition` / `LoopCondition`

Structured conditions with safe operators:
- Operators: `==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `not_in`, `contains`, `starts_with`, `ends_with`
- Field names validated with safe dot-notation patterns
- Loop conditions include `max_iterations` for loop protection

## Security Notes

- No arbitrary expression evaluation; only structured operators
- Loop protection via `max_iterations`
- Field validation prevents traversal-like patterns
