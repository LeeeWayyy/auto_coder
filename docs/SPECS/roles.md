# Role Configuration

**File:** `supervisor/core/roles.py`

## Overview

Role configuration loading with inheritance support. Roles define how AI workers behave, what context they receive, and what gates they must pass.

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
    cli: str              # e.g., "claude", "codex", "gemini"
    flags: list[str]      # CLI flags
    system_prompt: str    # Base instructions
    context: dict         # Context configuration
    gates: list[str]      # e.g., ["test", "lint"]
    config: dict          # Runtime configuration
    extends: str | None   # Parent role name
```

#### Properties

| Property | Default | Description |
|----------|---------|-------------|
| `token_budget` | 20000 | Max tokens for context |
| `include_patterns` | [] | File patterns to include |
| `exclude_patterns` | [] | File patterns to exclude |
| `max_retries` | 3 | Retry attempts |
| `timeout` | 300 | Execution timeout (seconds) |

### `RoleLoader`

Loads and merges role definitions:

```python
loader = RoleLoader()
role = loader.load_role("implementer")
roles = loader.list_available_roles()
```

#### Search Paths (Priority Order)

1. `.supervisor/roles/` - Project-specific
2. `~/.supervisor/roles/` - User-global
3. Package built-in roles

## Role Definition Format

YAML file structure:

```yaml
name: implementer
description: Implements code changes
cli: claude
flags:
  - "--max-turns"
  - "10"

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
# domain-implementer.yaml
name: domain-implementer
extends: implementer

system_prompt_additions: |
  Additional domain-specific instructions here.

context:
  include:
    - "src/domain/**/*.py"  # Appended to parent includes

gates:
  - integration-test  # Appended to parent gates

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
| `$TARGET_IMPORTS` | Import dependencies of target files |

## Error Classes

### `RoleNotFoundError`

Role configuration not found in any search path.

### `RoleValidationError`

Role configuration is invalid (missing required fields).

## Usage Example

```python
from supervisor.core.roles import RoleLoader

loader = RoleLoader()

# Load with inheritance resolution
role = loader.load_role("domain-implementer")

# Access configuration
print(role.cli)  # "claude"
print(role.token_budget)  # 30000
print(role.gates)  # ["test", "lint", "integration-test"]

# List all available roles
available = loader.list_available_roles()
# ["implementer", "planner", "reviewer", "domain-implementer"]
```
