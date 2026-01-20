# Terminal UI (TUI)

**File:** `supervisor/tui/app.py`

## Overview

Rich-based TUI for monitoring feature workflows and handling approval prompts.
Runs the workflow in a background thread and renders a live status view in the
main thread.

## Key Class

### `SupervisorTUI`

- Starts workflow function in a background thread
- Polls `InteractionBridge` for approval requests
- Displays feature/phase/component progress and recent events
- Provides a synchronous approval prompt when needed

## Integration

Used by `supervisor workflow --tui` with a shared `InteractionBridge` so
approvals flow between the coordinator and UI.
