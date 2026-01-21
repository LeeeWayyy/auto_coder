# Workflow Coordinator

**File:** `supervisor/core/workflow.py`

## Overview

Coordinates the Feature → Phase → Component workflow lifecycle. Handles
planning output validation, dependency scheduling, approvals, and timeouts.

## Key Responsibilities

- Create feature records and persist state to SQLite
- Validate planner output with Pydantic schemas
- Schedule component execution using a DAG scheduler
- Support parallel execution and timeouts
- Integrate approval gates and interaction bridge
- Route model selection via adaptive routing

## Core Types

- `PlannerOutput`, `PhasePlan`, `ComponentPlan`: schemas for planner output
- `WorkflowCoordinator`: main coordinator
- `WorkflowError`: workflow-level exception wrapper

## Integration

Used by `supervisor workflow` and TUI (`supervisor workflow --tui`).
