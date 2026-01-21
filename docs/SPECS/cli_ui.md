# CLI UI Utilities

**Directory:** `supervisor/cli_ui/`

## Overview

Terminal UI helpers for graph visualization and inspection.

## Components

### `graph_renderer.py`

`TerminalGraphRenderer` renders workflow graphs using Rich:
- Tree view with edge conditions
- ASCII view with topological layout
- Color-coded node types and statuses

### `live_monitor.py`

`LiveExecutionMonitor` polls graph execution state and renders live updates
for `supervisor run-graph --live`.

### `node_inspector.py`

`NodeInspector` shows detailed node output and metadata, with an interactive
REPL-style mode for exploring executions.
