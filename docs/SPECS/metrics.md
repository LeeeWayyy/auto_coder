# Metrics

**Files:**
- `supervisor/metrics/aggregator.py`
- `supervisor/metrics/dashboard.py`
- `supervisor/core/state.py` (metrics table)

## Overview

Collects and aggregates execution performance metrics for roles and CLIs.
Provides a Rich-based dashboard for CLI display.

## Components

### Metrics Table

Stored in SQLite (`metrics`), capturing:
- role, cli, task_type, workflow_id
- success flag, duration, retry count
- optional token usage and error category

### `MetricsAggregator`

Queries and aggregates:
- Role performance
- CLI comparisons by task type
- Recent failures
- Best CLI selection for a task

### `MetricsDashboard`

Renders summary tables and trends in the terminal.
