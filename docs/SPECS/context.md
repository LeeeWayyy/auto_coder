# Context Packing

**File:** `supervisor/core/context.py`

## Overview

Context packing for AI workers. Assembles relevant files and context for each role based on configuration, using Repomix when available or falling back to simple file packing.

## Key Classes

### `ContextPacker`

Main class for packing context:

```python
packer = ContextPacker(repo_path=Path("/path/to/repo"))

# Phase 2: Preferred method - template-based prompt building
prompt = packer.build_full_prompt(
    "implement.j2",
    role=role_config,
    task_description="Implement user authentication",
    target_files=["src/auth.py"],
    extra_context={"test_output": test_results}
)

# Legacy method (for unknown roles)
prompt = packer.pack_context(
    role=role_config,
    task_description="Implement user authentication",
    target_files=["src/auth.py"],
    extra_context={"git_diff": diff_output}
)
```

## Template System (Phase 2)

### SandboxedEnvironment

Uses Jinja2's `SandboxedEnvironment` for security:

```python
self.jinja_env = SandboxedEnvironment(
    loader=FileSystemLoader(template_dir),
    undefined=StrictUndefined,  # Catch undefined variables
    autoescape=False,  # Not HTML, no XSS concern
)
```

### Template Allowlist

Only package-shipped templates are allowed:

```python
ALLOWED_TEMPLATES = frozenset([
    "_base.j2",
    "_output_schema.j2",
    "planning.j2",
    "implement.j2",
    "review_strict.j2",
])
```

Template names are validated before rendering.

### Custom Filters

```python
self.jinja_env.filters["truncate_lines"] = self._truncate_lines
self.jinja_env.filters["format_diff"] = self._format_diff
```

## Context Assembly

### Template Flow (Phase 2)

1. **`build_full_prompt()`** - Main entry point for known roles
2. **`pack_file_context()`** - Packs files with budget reservation
3. **`render_prompt()`** - Renders Jinja2 template with context

```python
def build_full_prompt(
    template_name: str,
    role: RoleConfig,
    task_description: str,
    target_files: list[str] | None = None,
    extra_context: dict[str, str] | None = None,
) -> str
```

### Legacy Flow

1. **`pack_context()`** - Assembles complete prompt (for unknown roles)
2. System prompt + task + files + extra context

## File Packing Methods

### Repomix (Preferred)

When `repomix` is available via npx:

```bash
npx repomix@0.2.20 --style xml --include "**/*.py"
```

- Version pinned for supply-chain safety
- Produces optimized XML output
- Better token efficiency

### Simple Packing (Fallback)

When Repomix is unavailable:

- Uses Python's `fnmatch` for glob patterns
- Reads files directly with size limits
- Wraps in markdown code blocks

## Token Budget Management

### Constants (Phase 2)

```python
TEMPLATE_OVERHEAD_CHARS = 4000  # ~1000 tokens for template boilerplate
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB per file
MAX_TOTAL_SIZE = 10 * 1024 * 1024  # 10MB total
MAX_ALWAYS_INCLUDE_SIZE = 2 * 1024 * 1024  # 2MB for always_include
```

### Priority Order for Pruning

**Phase 2 (`pack_file_context` via `_combine_file_context`):**

Uses role-mapped priority or falls back to `FILE_CONTEXT_PRIORITY`:

```python
FILE_CONTEXT_PRIORITY = [
    "target_file",     # High priority - the file being modified
    "git_diff",        # Changes in progress
    "changed_files",   # Full content of changed files
    "imports",         # Import dependencies
    "related",         # Related files
    "files",           # General file context
    "tree",            # Low priority (first to prune)
]
```

Priority selection:
1. Map `role.context.priority_order` via `PRIORITY_KEY_MAP`
2. If no overlap with file-context keys, use `FILE_CONTEXT_PRIORITY`

**Legacy (`pack_context` via `_pack_with_budget`):**

Uses `PROTECTED_KEYS` and `PRUNABLE_PRIORITY_ORDER`:

```python
PROTECTED_KEYS = frozenset(["system_prompt", "task", "always_include"])  # Never pruned

PRUNABLE_PRIORITY_ORDER = [
    "target_file",     # High priority (last to prune)
    "git_diff",
    "changed_files",
    "imports",
    "related",
    "files",
    "tree",            # Low priority (first to prune)
]
# Unlisted keys (extra_context, etc.) are pruned before listed keys
```

### Context Protection by Flow

**Phase 2 Template Flow (`pack_file_context`):**
- `always_include` is protected and never pruned
- `system_prompt` and `task` are handled by templates (not in file context)
- Pruning: drops keys in reverse of selected priority order (tree first when using default `FILE_CONTEXT_PRIORITY`)
- If still over budget: returns protected content only

**Legacy Flow (`pack_context`):**
- `PROTECTED_KEYS` (`system_prompt`, `task`, `always_include`) are never pruned
- Unlisted keys (extra_context, etc.) are pruned first (lowest priority)
- Then `PRUNABLE_PRIORITY_ORDER` keys: tree → files → related → imports → changed_files → git_diff → target_file
- If still over budget: truncates protected content only

### Budget Calculation

```python
# Template flow reserves overhead
file_budget = max(0, role.token_budget - (TEMPLATE_OVERHEAD_CHARS // 4))

# Character-based estimation
char_budget = budget * 4  # 4 chars ~= 1 token
```

## Key Methods

### `build_full_prompt()` (Phase 2)

Main entry point combining context packing and template rendering:

```python
def build_full_prompt(
    template_name: str,
    role: RoleConfig,
    task_description: str,
    target_files: list[str] | None = None,
    extra_context: dict[str, str] | None = None,
) -> str
```

### `pack_file_context()` (Phase 2)

Packs FILE context only (not system_prompt/task):

```python
def pack_file_context(
    role: RoleConfig,
    target_files: list[str] | None = None,
    extra_context: dict[str, str] | None = None,
) -> str
```

### `_pack_target_files()` (Phase 2)

Packs specific target files with high priority:

```python
def _pack_target_files(target_files: list[str]) -> str
```

### `_pack_always_include()` (Phase 2)

Packs protected files that are never pruned:

```python
def _pack_always_include(patterns: list[str]) -> str
```

### `pack_context()` (Legacy)

Full context packing for unknown roles:

```python
def pack_context(
    role: RoleConfig,
    task_description: str,
    target_files: list[str] | None = None,
    extra_context: dict[str, str] | None = None,
) -> str
```

### `render_prompt()`

Renders Jinja2 template with validation:

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

### `_get_changed_files()` (Phase 2)

Gets list of changed files from git:

```python
def _get_changed_files() -> list[str]
# Returns files from: git diff --name-only --cached
```

## Pattern Resolution

### Special Variables

| Pattern | Resolution |
|---------|------------|
| `$TARGET` | Replaced with target_files |
| `$TARGET_IMPORTS` | (TODO) Import dependencies |
| `$CHANGED_FILES` | Files from git diff --name-only --cached |

### Context Flags (Phase 2)

Boolean flags in `role.context`:

| Flag | Effect |
|------|--------|
| `git_diff: true` | Include staged git diff (`git diff --cached`) in context |

### Sentinels (Phase 2)

Non-file patterns handled separately:

```python
SENTINELS = frozenset(["git diff --cached", "$CHANGED_FILES"])
```

These trigger special handling, not file reads.

### skip_targets Parameter (Phase 2)

Prevents target file duplication:

```python
def _pack_files(role, target_files, skip_targets=True):
    # When skip_targets=True, $TARGET not expanded
    # (already packed via _pack_target_files)
```

## Template Structure

Templates in `supervisor/prompts/`:

### `_base.j2`

Base template with output requirements:

```jinja2
{% block content %}
{% endblock %}

---

{% block output_requirements %}
## Output Requirements
You MUST end your response with a fenced JSON code block...
{% include "_output_schema.j2" %}
{% endblock %}
```

### Role Templates

- `planning.j2` - Extends _base.j2 for planner
- `implement.j2` - Extends _base.j2 for implementer
- `review_strict.j2` - Extends _base.j2 for reviewer

## Error Handling

### `ContextPackerError`

Defined for context packing failures (currently unused - errors use fallback behavior instead).

### Fallback Behavior

- Repomix timeout -> Fall back to simple packing
- Repomix error -> Fall back to simple packing
- Template error -> Fall back to `pack_context(role, task, target_files, extra_context)` to ensure no context is lost
- Unreadable files -> Target files emit `[Could not read file]` placeholder; always_include and simple packing skip silently
- Unknown template -> Raises ValueError

## Usage Example

```python
from supervisor.core.context import ContextPacker
from supervisor.core.roles import RoleLoader

loader = RoleLoader()
role = loader.load_role("implementer")

packer = ContextPacker(Path("/path/to/repo"))

# Phase 2: Template-based (preferred)
prompt = packer.build_full_prompt(
    "implement.j2",
    role,
    "Add user authentication",
    target_files=["src/auth.py"],
)

# Legacy: For unknown roles
prompt = packer.pack_context(
    role,
    "Add user authentication",
    target_files=["src/auth.py"],
    extra_context={"git_diff": packer.get_git_diff()},
)
```
