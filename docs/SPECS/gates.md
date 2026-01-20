# Gates System

**Files:**
- `supervisor/core/gate_models.py`
- `supervisor/core/gate_loader.py`
- `supervisor/core/gate_executor.py`
- `supervisor/core/gate_locks.py`

## Overview

Gates provide verification steps (lint, tests, security checks) that run in isolated worktrees and must pass before changes are applied.

## Key Concepts

### Gate Configuration

`GateConfig` defines:
- `command`, `timeout`, `severity`, dependencies
- cache settings and cache inputs
- allowed writes and shell safety options

Gate configs load from:
- `~/.supervisor/gates.yaml`
- package default `supervisor/config/gates.yaml`
- optional project `.supervisor/gates.yaml` (when enabled)

### Gate Execution

`GateExecutor` provides:
- Worktree baseline capture and integrity checks
- Cache key computation with safety limits
- Artifact storage with size/retention limits
- Safe env filtering and hardened git execution

### Concurrency & Locks

`gate_locks.py` enforces exclusive access to worktrees and artifacts during gate execution to prevent races.

## Security Notes

- Path traversal and shell invocation detection are enforced at load time
- Protected env vars are filtered from gate configs
- Worktree integrity checks prevent unintended writes
