# Graph Orchestrator

**File:** `supervisor/core/graph_engine.py`

## Overview

Implements a **stateless** workflow graph executor. All state lives in SQLite so executions can be resumed and multiple workers can cooperate safely.

## Key Class

### `GraphOrchestrator`

Responsibilities:
- Start a workflow execution and persist an immutable snapshot
- Claim and execute ready nodes transactionally
- Update node status and outputs in the database
- Enforce gate execution and approval nodes
- Broadcast node status changes via callbacks (used by Studio WebSockets)

Key behaviors:
- Stateless execution loops: each `execute_next_batch()` call is independent
- Supports horizontal scaling by polling DB for ready nodes
- Status callbacks include a version counter to avoid race conditions

## Persistence

Uses `Database` tables:
- `graph_workflows` (definitions)
- `graph_executions` (runs)
- `node_executions` (per-node state, versioned)
- `loop_counters` (loop protection)

## Studio Integration

`register_status_callback()` allows the Studio server to stream live updates
via WebSockets. Callbacks are invoked after DB commits and may run in worker
threads, so they must be thread-safe.
