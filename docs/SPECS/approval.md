# Approval & Interaction

**Files:**
- `supervisor/core/approval.py`
- `supervisor/core/interaction.py`

## Overview

Implements human-in-the-loop approvals for risky actions and provides a thread-safe bridge between workflow execution and the TUI.

## Key Types

### `ApprovalPolicy`

Simple policy controls:
- `auto_approve_low_risk`
- `risk_threshold`
- `require_approval_for` (operations that always require approval)

### `ApprovalGate`

- Assesses risk based on changed files and operation type
- Requests approval via `InteractionBridge` or CLI prompt
- Records approval events in the database

### `InteractionBridge`

Thread-safe queue + event mechanism:
- Workflow thread blocks on `request_approval()`
- TUI polls `get_pending_requests()` and submits decisions

## Integration

Used by `WorkflowCoordinator` and `SupervisorTUI` to coordinate approvals during workflow execution.
