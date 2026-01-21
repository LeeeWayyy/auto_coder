# Graph Workflow Worker

**File:** `supervisor/core/worker.py`

## Overview

Background worker that executes graph workflows by polling the database for
running executions and invoking `GraphOrchestrator.execute_next_batch()`.

## Key Methods

- `run_until_complete(execution_id)`: Run a single execution to completion
- `start_daemon()`: Poll and process all running executions (daemon mode)
- `stop()`: Stop daemon loop
