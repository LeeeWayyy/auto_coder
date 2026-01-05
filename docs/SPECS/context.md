# Context Packing

**File:** `supervisor/core/context.py`

## Overview

Context packing for AI workers. Assembles relevant files and context for each role based on configuration, using Repomix when available or falling back to simple file packing.

## Key Classes

### `ContextPacker`

Main class for packing context:

```python
packer = ContextPacker(repo_path=Path("/path/to/repo"))
prompt = packer.pack_context(
    role=role_config,
    task_description="Implement user authentication",
    target_files=["src/auth.py"],
    extra_context={"git_diff": diff_output}
)
```

## Context Assembly Order

1. **System prompt** - Role's base instructions (always first)
2. **Task description** - What the worker should do
3. **File context** - Relevant code files
4. **Extra context** - Git diff, test output, etc.

## File Packing Methods

### Repomix (Preferred)

When `repomix` is available via npx:

```bash
npx repomix --style xml --include "**/*.py" --ignore "**/test_*"
```

- Produces optimized XML output
- Better token efficiency
- Handles large codebases well

### Simple Packing (Fallback)

When Repomix is unavailable:

- Uses Python's `fnmatch` for glob patterns
- Reads files directly
- Wraps in markdown code blocks

## Token Budget Management

### Priority Order for Pruning

When context exceeds token budget, items are dropped in reverse priority:

```python
PRIORITY_ORDER = [
    "system_prompt",  # Never dropped (highest priority)
    "task",
    "target_file",
    "imports",
    "related",
    "tree",  # Dropped first (lowest priority)
]
```

### Budget Calculation

Uses simple character-based estimation:
- 4 characters ≈ 1 token
- `char_budget = token_budget * 4`

## Key Methods

### `pack_context()`

Main entry point:

```python
def pack_context(
    role: RoleConfig,
    task_description: str,
    target_files: list[str] | None = None,
    extra_context: dict[str, str] | None = None,
) -> str
```

### `render_prompt()`

Renders Jinja2 prompt template:

```python
def render_prompt(
    template_name: str,
    role: RoleConfig,
    task_description: str,
    **kwargs,
) -> str
```

### `get_git_diff()`

Gets git diff for context:

```python
diff = packer.get_git_diff(staged=True)  # --cached
diff = packer.get_git_diff(staged=False)  # unstaged
```

### `get_file_tree()`

Gets repository file tree:

```python
tree = packer.get_file_tree(max_depth=3)
```

## Pattern Resolution

Special pattern variables are resolved:

| Pattern | Resolution |
|---------|------------|
| `$TARGET` | Replaced with target_files |
| `$TARGET_IMPORTS` | (TODO) Import dependencies |

## Context Sections Format

Each section is formatted with markdown headers:

```markdown
## System Prompt

[Role's system prompt]

---

## Task

[Task description]

---

## Repository Context

### src/auth.py

```python
[file content]
```

---

## Git Diff

[diff output]
```

## Error Handling

### `ContextPackerError`

Raised on context packing failures.

### Fallback Behavior

- Repomix timeout → Fall back to simple packing
- Repomix error → Fall back to simple packing
- Template error → Fall back to simple context building
- Unreadable files → Skipped silently

## Template Support

Uses Jinja2 templates from `supervisor/prompts/` directory:

```python
packer = ContextPacker(repo_path)
if packer.jinja_env:
    prompt = packer.render_prompt(
        "implement.jinja2",
        role,
        task_description,
        files=target_files
    )
```
