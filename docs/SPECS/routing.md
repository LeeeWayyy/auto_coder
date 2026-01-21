# Adaptive Routing

**File:** `supervisor/core/routing.py`

## Overview

Implements task-aware model selection across multiple AI CLIs and model tiers
(e.g., `claude:opus`, `claude:sonnet`, `gemini:pro`). Routing can be static or
adaptive based on historical metrics.

## Key Concepts

- Role → task type inference (`_infer_task_type`)
- Task type → capability mapping
- `ModelProfile` registry with strengths, context limits, and quality scores
- `AdaptiveConfig` for epsilon-greedy exploration and lookback windows

## Integration

`WorkflowCoordinator` uses routing to select the best model for a role based on
capabilities and metrics captured in the SQLite `metrics` table.
