# Git Worktree Isolation

**File:** `supervisor/core/workspace.py`

## Overview

Provides filesystem isolation for step execution using git worktrees. Each step executes in a clean worktree, and gates run in the worktree BEFORE changes are applied to the main tree. Failed steps leave no trace in the main tree.

## Key Classes

### `IsolatedWorkspace`

Main class for isolated step execution.

#### Constructor

```python
IsolatedWorkspace(
    repo_path: Path,
    executor: SandboxedExecutor,
    db: Database,
)
```

**Requirements:**
- `filelock` package must be installed (raises `FileLockRequiredError` otherwise)
- Path must be a valid git repository

#### Key Methods

##### `isolated_execution()`

Context manager for isolated step execution:

```python
with workspace.isolated_execution(step_id) as ctx:
    # ctx.worktree_path - Path to isolated worktree
    # ctx.step_id - The step identifier
    # ctx.original_head - HEAD SHA when worktree was created
```

##### `execute_step()`

Full step execution with worker function and gates:

```python
def execute_step(
    step: Step,
    worker_fn: Callable[[Step, Path], dict[str, Any]],
    gates: list[str] | None = None,
) -> dict[str, Any]
```

### `WorktreeContext`

Dataclass holding context for isolated execution:

```python
@dataclass
class WorktreeContext:
    worktree_path: Path
    step_id: str
    original_head: str
```

### `FileChange`

Represents a file change from git status:

```python
@dataclass
class FileChange:
    path: str
    status: str  # 'A', 'M', 'D', 'R', 'C'
    old_path: str | None = None  # For renames/copies
```

## Security Features

### Path Validation
- `_validate_path()`: Rejects absolute paths and path traversal (`..`)
- `_validate_parent_path()`: Ensures parent directories are not symlinks
- `_sanitize_step_id()`: Prevents path traversal in step IDs

### Symlink Protection
- `_validate_no_symlinks_in_source()`: Pre-validates all sources before any modifications
- `_reject_symlinks_ignore()`: Callback for `shutil.copytree` to reject symlinks
- Multiple symlink checks in `_copy_safe()` and `_remove_worktree()`

### TOCTOU Prevention
- Re-validates worktrees directory after mkdir
- Symlink checks immediately before operations
- Grace period (5 minutes) in cleanup to avoid race conditions

### Atomicity
- Staged copy: Writes to temp file first, then atomic rename
- File lock (`FileLock`) for apply operations
- HEAD conflict detection before applying changes

## Key Internal Methods

| Method | Purpose |
|--------|---------|
| `_create_worktree()` | Creates clean git worktree from HEAD |
| `_remove_worktree()` | Removes worktree with symlink protection |
| `_get_changed_files()` | Parses `git status --porcelain=v1 -z` |
| `_apply_changes()` | Two-pass apply: deletions then writes |
| `_copy_safe()` | Atomic copy with symlink rejection |
| `_run_gate()` | Executes gate in sandbox with `--` separator |
| `_cleanup_stale_worktrees()` | Removes orphaned worktrees on startup |

## Error Classes

- `WorktreeError`: General worktree operation failure
- `GateFailedError`: Gate verification failed (includes gate name and output)
- `FileLockRequiredError`: filelock package not installed

## Helper Functions

### `_truncate_output()`

Truncates output preserving both head and tail (errors often at end):

```python
def _truncate_output(output: str, max_length: int = 2000) -> str
```

Returns: First ~40% + separator + Last ~60%
