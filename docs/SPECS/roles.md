# Role Configuration

**File:** `supervisor/core/roles.py`

## Overview

Role configuration loading with inheritance support. Roles define how AI workers behave, what context they receive, and suggested gates (informational - `role.gates` is metadata; actual gate execution depends on caller-supplied `gates` parameter).

## Design: Base + Overlay Model

- **Base roles**: `planner`, `implementer`, `reviewer` (shipped with supervisor)
- **Domain overlays**: Extend base roles with domain-specific knowledge
- **Merge semantics**:
  - Lists: Append (child extends parent)
  - Dicts: Deep merge
  - Scalars: Override (child replaces parent)

## Key Classes

### `RoleConfig`

Configuration for a worker role:

```python
@dataclass
class RoleConfig:
    name: str
    description: str
    cli: str              # "claude", "codex", or "gemini"
    flags: list[str]      # CLI flags
    system_prompt: str    # Base instructions
    context: dict         # Context configuration
    gates: list[str]      # e.g., ["test", "lint"]
    config: dict          # Runtime configuration
    extends: str | None   # Parent role name (stripped during merge; always None post-load)
    base_role: str | None # Root of extends chain (Phase 2) - use this for resolution
```

#### Properties

| Property | Default | Description |
|----------|---------|-------------|
| `token_budget` | 20000 | Max tokens for context |
| `include_patterns` | [] | File patterns to include |
| `exclude_patterns` | [] | File patterns to exclude |
| `max_retries` | 3 | Retry attempts (metadata only - actual retries controlled by caller's `RetryPolicy`) |
| `timeout` | 300 | Execution timeout in seconds (metadata only - actual timeout from `SandboxConfig`) |

#### `base_role` Field (Phase 2)

The `base_role` field tracks the **root** of the extends chain:

```python
# For base roles (planner, implementer, reviewer):
role.base_role = None  # They ARE the base

# For single-level overlay (python-implementer extends implementer):
role.base_role = "implementer"

# For multi-level overlay (my-impl extends python-impl extends implementer):
role.base_role = "implementer"  # Root of chain, not immediate parent
```

This enables template and schema resolution for overlay roles.

### `RoleLoader`

Loads and merges role definitions with JSON schema validation:

```python
loader = RoleLoader()
role = loader.load_role("implementer")
roles = loader.list_available_roles()
```

#### Search Paths (Priority Order)

1. `.supervisor/roles/` - Project-specific
2. `~/.supervisor/roles/` - User-global
3. Package built-in roles

#### Schema Validation (Phase 2)

Merged configurations are validated against `supervisor/config/role_schema.json`:

```python
# In RoleLoader.__post_init__
self._schema = self._load_schema()

# Validation runs AFTER inheritance merge
self._validate_merged_config(config)
```

This allows overlay roles to omit fields inherited from parents.

#### Pre-Merge Type Validation (Phase 2)

Before merging, lightweight type checks catch structural issues:

```python
type_checks = {
    "flags": list,
    "gates": list,
    "context": dict,
    "config": dict,
}
```

## Role Definition Format

YAML file structure:

```yaml
name: implementer
description: Implements code changes
cli: claude
flags:
  - "-p"
  - "--output-format"
  - "json"

system_prompt: |
  You are an expert software developer.
  Always write clean, tested code.

context:
  token_budget: 30000
  include:
    - "**/*.py"
    - "$TARGET"
  exclude:
    - "**/__pycache__/**"
    - "**/node_modules/**"
  always_include:
    - "docs/STANDARDS/coding.md"
  priority_order:
    - target_file
    - git_diff
    - imports
    - files

gates:
  - test
  - lint

config:
  max_retries: 3
  timeout: 300
```

## Inheritance

Child roles can extend parent roles:

```yaml
# python-implementer.yaml
name: python-implementer
extends: implementer

system_prompt_additions: |
  Additional Python-specific instructions here.

context:
  include:
    - "src/**/*.py"  # Appended to parent includes

gates:
  - mypy  # Appended to parent gates

config:
  timeout: 600  # Overrides parent
```

### Merge Rules

| Field | Merge Behavior |
|-------|----------------|
| `name`, `description`, `cli` | Override |
| `flags`, `gates` | Append |
| `context`, `config` | Deep merge |
| `system_prompt_additions` | Concatenate to parent prompt |

## Special Pattern Variables

| Variable | Description |
|----------|-------------|
| `$TARGET` | Target files passed to the role |
| `$TARGET_IMPORTS` | (TODO - not yet implemented) Import dependencies of target files |
| `$CHANGED_FILES` | Files from `git diff --name-only --cached` |
| `git diff --cached` | Sentinel for staged git diff (not a file) |

## Context Flags

| Flag | Description |
|------|-------------|
| `context.git_diff: true` | Include staged git diff in context |

## JSON Schema (Phase 2)

The `role_schema.json` validates merged configurations:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "required": ["name", "description", "cli", "system_prompt"],
  "additionalProperties": false,
  "properties": {
    "cli": {
      "enum": ["claude", "codex", "gemini"]
    },
    "context": {
      "properties": {
        "token_budget": {"type": "integer", "minimum": 1000}
      }
    }
  },
  "patternProperties": {
    "^x-": {}  // Extension fields allowed
  }
}
```

**Note:** `additionalProperties: false` means extra top-level fields are rejected unless `x-` prefixed (e.g., `x-custom-field`). Custom data should be placed under `context` or `config`.

## Error Classes

### `RoleNotFoundError`

Role configuration not found in any search path.

### `RoleValidationError`

Role configuration is invalid:
- Missing required fields
- Invalid field types
- Schema validation failure
- Empty YAML file
- Invalid YAML syntax

### `RoleCycleError`

Circular inheritance detected (e.g., A extends B extends A).

## Usage Example

```python
from supervisor.core.roles import RoleLoader

loader = RoleLoader()

# Load with inheritance resolution
role = loader.load_role("python-implementer")

# Access configuration
print(role.cli)  # "claude"
print(role.token_budget)  # 30000
print(role.gates)  # ["test", "lint", "mypy"]
print(role.base_role)  # "implementer" (root of extends chain)

# List all available roles
available = loader.list_available_roles()
# ["implementer", "planner", "reviewer", "python-implementer"]
```
