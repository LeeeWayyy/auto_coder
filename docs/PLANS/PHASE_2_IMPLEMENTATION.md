# Phase 2 Implementation Plan: Core Workflow

**Status:** Draft - Revision 14
**Phase:** 2 of 5
**Prerequisites:** Phase 1 (Foundation) - COMPLETE

## Executive Summary

Phase 2 builds on the security-hardened foundation from Phase 1 to implement the core workflow capabilities. This phase transforms the supervisor from a secure sandbox into a functional AI orchestration system.

## Scope

From SUPERVISOR_ORCHESTRATOR.md Phase 2:
- [ ] Role configuration loading (YAML) with schema validation
- [ ] Jinja2 prompt templating
- [ ] Context packing with Repomix integration
- [ ] Structured output parsing (JSON-only, no marker fallback)

## Current State Analysis

### Already Implemented in Phase 1

| Component | File | Status |
|-----------|------|--------|
| Role YAML loading | `supervisor/core/roles.py` | Complete |
| Role inheritance | `supervisor/core/roles.py` | Complete |
| Context packing | `supervisor/core/context.py` | Partial |
| Repomix integration | `supervisor/core/context.py` | Partial |
| Schema validation | N/A | Not Started |
| Jinja2 templating | `supervisor/core/context.py` | Stub Only |
| Structured output parsing | N/A | Not Started |

### Gaps to Address

1. **Schema Validation**: No JSON Schema validation for role configs
2. **Jinja2 Templates**: Only stub implementation in context.py
3. **CLI-specific Adapters**: Need per-CLI output format handling (parser.py needs enhancement)

### Existing Code to Integrate With

**IMPORTANT:** The following files already exist and should be MODIFIED, not replaced:

| File | Contains | Phase 2 Action |
|------|----------|----------------|
| `supervisor/core/parser.py` | JSON extraction, role parsers | ADD CLI adapters |
| `supervisor/core/models.py` | `PlanOutput`, `ImplementationOutput`, `ReviewOutput` | KEEP existing schemas |
| `supervisor/core/engine.py` | `run_role()` orchestration | INTEGRATE templates |
| `supervisor/config/base_roles/*.yaml` | Base role configs with custom fields | SCHEMA must accommodate |

---

## Implementation Tasks

### Task 1: Role Schema Validation

**Goal:** Validate role YAML files against JSON Schema AFTER inheritance merge.

**Design Decision:** Keep `gates` as `list[str]` in Phase 2 to maintain compatibility with
existing merge logic and downstream consumers. Structured `GateConfig` will be introduced
in Phase 3 (Gates & Verification) with coordinated refactoring.

**Files to Create/Modify:**
- `supervisor/config/role_schema.json` (NEW) - Schema for MERGED configs
- `supervisor/core/roles.py` (MODIFY) - Add schema validation timing (no type changes)

**Implementation Details:**

```python
# Add to roles.py
import json
import jsonschema
import yaml

@dataclass
class RoleLoader:
    # Existing dataclass fields preserved
    search_paths: list[Path] = field(default_factory=list)
    package_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "config" / "base_roles"
    )

    # Schema loaded in __post_init__ to preserve dataclass init behavior
    _schema: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # EXISTING: Default search paths (preserved from current implementation)
        if not self.search_paths:
            self.search_paths = [
                Path(".supervisor/roles"),  # Project-specific
                Path.home() / ".supervisor/roles",  # User-global
                self.package_dir,  # Built-in
            ]

        # PHASE 2 ADDITION: Load JSON schema for validation
        self._schema = self._load_schema()

    def _load_schema(self) -> dict:
        """Load JSON schema for role validation.

        Raises RoleValidationError with actionable message on failure.
        """
        schema_path = Path(__file__).parent.parent / "config" / "role_schema.json"
        try:
            with open(schema_path) as f:
                return json.load(f)
        except FileNotFoundError:
            raise RoleValidationError(
                f"Role schema not found at {schema_path}. "
                f"Ensure supervisor package is properly installed."
            )
        except json.JSONDecodeError as e:
            raise RoleValidationError(
                f"Invalid JSON in role schema at {schema_path}: {e}"
            )

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load YAML file with proper error handling."""
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RoleValidationError(f"Invalid YAML in {path}: {e}")

        if config is None:
            raise RoleValidationError(f"Empty or invalid YAML file: {path}")

        if not isinstance(config, dict):
            raise RoleValidationError(
                f"Role config must be a dict, got {type(config).__name__} in {path}"
            )

        return config

    def load_role(self, name: str, _loading_chain: list[str] | None = None) -> RoleConfig:
        # ... existing loading logic ...

        role_file = self._find_role_file(name)
        config = self._load_yaml(role_file)  # Uses new method with error handling

        # Pre-merge type validation to catch structural issues before merge fails
        self._validate_pre_merge_types(config, role_file)

        # PHASE 2 ADDITION: Capture base_role before merge removes extends
        # base_role is the ROOT of the extends chain (handles multi-level overlays)
        base_role: str | None = None
        if "extends" in config:
            parent = self.load_role(config["extends"], _loading_chain)
            # Use parent's base_role if it has one (multi-level), otherwise parent's name
            base_role = parent.base_role if parent.base_role else parent.name
            config = self._merge_configs(parent, config)
        else:
            # Base roles have base_role = None (they ARE the base)
            base_role = None

        # CRITICAL: Full schema validation AFTER merge
        # This allows overlay roles to omit inherited fields like system_prompt
        self._validate_merged_config(config)

        # Pass base_role to RoleConfig
        return self._dict_to_role(config, base_role=base_role)

    def _validate_pre_merge_types(self, config: dict, path: Path) -> None:
        """Lightweight type checks before merge to catch structural issues early."""
        type_checks = {
            "flags": list,
            "gates": list,
            "context": dict,
            "config": dict,
        }
        for field, expected_type in type_checks.items():
            if field in config and not isinstance(config[field], expected_type):
                raise RoleValidationError(
                    f"Field '{field}' must be {expected_type.__name__}, "
                    f"got {type(config[field]).__name__} in {path}"
                )

    def _validate_merged_config(self, config: dict[str, Any]) -> None:
        """Validate MERGED role configuration against JSON Schema.

        IMPORTANT: This runs AFTER inheritance merge, so overlay roles
        that only define system_prompt_additions (not system_prompt)
        will have system_prompt populated from parent.
        """
        try:
            jsonschema.validate(config, self._schema)
        except jsonschema.ValidationError as e:
            raise RoleValidationError(f"Schema validation failed: {e.message}")
        except jsonschema.SchemaError as e:
            # Invalid schema definition (developer error, not user error)
            raise RoleValidationError(
                f"Invalid role schema definition (bug in role_schema.json): {e.message}"
            )

    # NOTE: gates stays as list[str] - Structured GateConfig will be introduced in Phase 3

    def _dict_to_role(self, config: dict[str, Any], base_role: str | None = None) -> RoleConfig:
        """Convert dict to RoleConfig with base_role for template/schema resolution."""
        return RoleConfig(
            name=config["name"],
            description=config["description"],
            cli=config["cli"],
            flags=config.get("flags", []),
            system_prompt=config["system_prompt"],
            context=config.get("context", {}),
            gates=config.get("gates", []),
            config=config.get("config", {}),
            extends=config.get("extends"),
            base_role=base_role,  # NEW: Root of extends chain for template/schema lookup
        )
```

**REQUIRED UPDATE to existing RoleConfig dataclass (roles.py):**

Add `base_role` field for template/schema resolution in overlays:

```python
@dataclass
class RoleConfig:
    """Configuration for a worker role."""

    name: str
    description: str
    cli: str
    flags: list[str]
    system_prompt: str
    context: dict[str, Any]
    gates: list[str]
    config: dict[str, Any]
    extends: str | None = None
    base_role: str | None = None  # NEW: Root of extends chain (planner/implementer/reviewer)
```

The `base_role` field:
- Is `None` for base roles (planner, implementer, reviewer) - they ARE the base
- Is set to the root of the extends chain for overlays (handles multi-level: A→B→reviewer sets base_role="reviewer")
- Used by `_get_template_for_role()` and `_get_schema_for_role()` to resolve template/schema

**Schema Definition (role_schema.json) - For MERGED configs:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Merged Role Configuration",
  "description": "Schema for role configs AFTER inheritance merge. All required fields must be present.",
  "type": "object",
  "required": ["name", "description", "cli", "system_prompt"],
  "additionalProperties": false,
  "patternProperties": {
    "^x-": { "description": "Extension fields (x-model, x-temperature, etc.) for custom role options" }
  },
  "properties": {
    "name": {
      "type": "string",
      "pattern": "^[a-zA-Z][a-zA-Z0-9_-]*$"
    },
    "description": { "type": "string" },
    "extends": { "type": "string" },
    "cli": {
      "type": "string",
      "enum": ["claude", "codex", "gemini"]
    },
    "flags": {
      "type": "array",
      "items": { "type": "string" }
    },
    "system_prompt": { "type": "string" },
    "context": {
      "type": "object",
      "additionalProperties": true,
      "description": "Allows role-specific fields: git_diff, priority_order, $TARGET, $TARGET_IMPORTS, etc.",
      "properties": {
        "include": { "type": "array", "items": { "type": "string" } },
        "exclude": { "type": "array", "items": { "type": "string" } },
        "always_include": { "type": "array", "items": { "type": "string" } },
        "token_budget": { "type": "integer", "minimum": 1000, "maximum": 100000 },
        "git_diff": { "type": "boolean" },
        "priority_order": { "type": "array", "items": { "type": "string" } }
      }
    },
    "gates": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Simple gate commands as strings. Structured GateConfig deferred to Phase 3."
    },
    "config": {
      "type": "object",
      "additionalProperties": true,
      "description": "Allows role-specific config: tdd, coverage_target, etc.",
      "properties": {
        "max_retries": { "type": "integer", "minimum": 1, "maximum": 10 },
        "timeout": { "type": "integer", "minimum": 30, "maximum": 3600 },
        "tdd": { "type": "boolean" },
        "coverage_target": { "type": "integer", "minimum": 0, "maximum": 100 }
      }
    }
  }
}
```

**Acceptance Criteria:**
- [ ] Invalid YAML fails with clear error message
- [ ] Missing required fields are caught (after merge)
- [ ] Invalid enum values are rejected
- [ ] Inheritance chain is validated
- [ ] Overlay roles without `system_prompt` work (inherit from parent)
- [ ] Unknown fields are rejected (`additionalProperties: false`)

---

### Task 2: Jinja2 Prompt Templating

**Goal:** Full Jinja2 template support for prompts with role context injection.

**Files to Create/Modify:**
- `supervisor/prompts/_base.j2` (NEW)
- `supervisor/prompts/_output_schema.j2` (NEW)
- `supervisor/prompts/planning.j2` (NEW)
- `supervisor/prompts/implement.j2` (NEW)
- `supervisor/prompts/review_strict.j2` (NEW)
- `supervisor/core/context.py` (MODIFY)

**Template Structure:**

```jinja2
{# _base.j2 - Base template with content and output requirements blocks #}
{# Child templates extend this and override content block #}

{% block content %}
{# Override in child templates #}
{% endblock %}

---

{% block output_requirements %}
## Output Requirements

You MUST end your response with a fenced JSON code block containing your structured output.

**CRITICAL:** Use exactly this format with triple backticks and "json" label:

```json
{ "status": "...", ... }
```

The JSON MUST be in a fenced code block (triple backticks with json label).
Bare JSON without fences will NOT be accepted.

{% include "_output_schema.j2" %}
{% endblock %}


{# planning.j2 - Example child template #}
{% extends "_base.j2" %}

{% block content %}
{{ role.system_prompt }}

## Context

{{ context }}

## Task

{{ task }}
{% endblock %}
```

**Implementation in context.py:**

```python
from jinja2 import FileSystemLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

# SECURITY: Allowlist of valid template names (shipped with package)
ALLOWED_TEMPLATES = frozenset([
    "_base.j2",
    "_output_schema.j2",
    "planning.j2",
    "implement.j2",
    "review_strict.j2",
])

class ContextPacker:
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path).absolute()

        # SECURITY: Template directory is package-internal, not user-controlled
        template_dir = Path(__file__).parent.parent / "prompts"

        # SECURITY: Use SandboxedEnvironment to prevent arbitrary code execution
        # StrictUndefined raises errors on undefined variables (catches typos)
        self.jinja_env = SandboxedEnvironment(
            loader=FileSystemLoader(template_dir),
            undefined=StrictUndefined,
            autoescape=False,  # Not HTML, no XSS concern
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        self.jinja_env.filters['truncate_lines'] = self._truncate_lines
        self.jinja_env.filters['format_diff'] = self._format_diff

    def build_full_prompt(
        self,
        template_name: str,
        role: RoleConfig,
        task_description: str,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
    ) -> str:
        """Build complete prompt: pack context + render template.

        This is the main entry point for prompt generation.
        It combines context packing and template rendering in one call.

        IMPORTANT: Template renders ALL content (system_prompt, task, context).
        pack_file_context() only returns file content, NOT system_prompt/task.
        This prevents duplication.

        Usage:
            packer = ContextPacker(repo_path)
            prompt = packer.build_full_prompt(
                "planning.j2",
                role=planner_role,
                task_description="Implement feature X",
                target_files=["src/main.py"],
            )
        """
        # Step 1: Pack FILE context only (not system_prompt/task - template handles those)
        file_context = self.pack_file_context(role, target_files, extra_context)

        # Step 2: Render template with role, task, and file context
        # Template handles: system_prompt, task, context (files), output requirements
        return self.render_prompt(template_name, role, task_description, file_context)

    # Reserve budget for system_prompt + task + output_requirements template overhead
    TEMPLATE_OVERHEAD_CHARS = 4000  # ~1000 tokens for template boilerplate

    def pack_file_context(
        self,
        role: RoleConfig,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
    ) -> str:
        """Pack FILE context only, with budget reservation for template overhead.

        Used when templates handle system_prompt/task rendering.
        Budget is reduced by TEMPLATE_OVERHEAD_CHARS to reserve space for
        system_prompt, task, and output_requirements in the final prompt.
        """
        context_parts: dict[str, str] = {}

        # Pack always_include files (protected)
        always_include = role.context.get("always_include", [])
        if always_include:
            always_content = self._pack_always_include(always_include)
            if always_content:
                context_parts["always_include"] = always_content

        # Pack target files
        if target_files:
            target_content = self._pack_target_files(target_files)
            if target_content:
                context_parts["target_file"] = target_content

        # Handle role-specific context directives
        # Check for git_diff boolean flag OR "git diff --cached" in include patterns
        include_patterns = role.context.get("include", [])
        if role.context.get("git_diff") or "git diff --cached" in include_patterns:
            git_diff = self.get_git_diff(staged=True)
            if git_diff:
                context_parts["git_diff"] = f"## Git Diff (Staged)\n\n```diff\n{git_diff}\n```"

        # Handle $CHANGED_FILES - resolve to list of files from git diff
        if "$CHANGED_FILES" in include_patterns:
            changed_files = self._get_changed_files()
            if changed_files:
                changed_content = self._pack_target_files(changed_files)
                if changed_content:
                    context_parts["changed_files"] = changed_content

        # Pack role-specific files
        # NOTE: Skip $TARGET patterns since target_files are already packed above
        # This prevents duplication when role includes have $TARGET
        file_context = self._pack_files(role, target_files, skip_targets=True)
        if file_context:
            context_parts["files"] = file_context

        # Add extra context (test output, etc.)
        if extra_context:
            for key, value in extra_context.items():
                context_parts[key] = f"## {key.replace('_', ' ').title()}\n\n{value}"

        # Budget for file context = total budget - template overhead
        # Clamp to prevent negative budget for small token_budget values
        file_budget = max(0, role.token_budget - (self.TEMPLATE_OVERHEAD_CHARS // 4))
        if file_budget == 0:
            # Warning: overhead exceeds budget - file context will be empty
            return "[WARNING: Template overhead exceeds token budget; no file context included]"
        return self._combine_file_context(context_parts, file_budget, role)

    # File-context specific priority order (maps from role-level keys to file-context keys)
    # Role configs may use keys like "task_description", "readme" for full prompts
    # File context uses: target_file, git_diff, changed_files, files, tree, always_include
    FILE_CONTEXT_PRIORITY = [
        "target_file",     # High priority - the file being modified
        "git_diff",        # Changes in progress
        "changed_files",   # Full content of changed files
        "imports",         # Import dependencies
        "related",         # Related files
        "files",           # General file context
        "tree",            # Directory tree (low priority - first to prune)
    ]

    # Mapping from role priority_order keys to file-context keys
    PRIORITY_KEY_MAP = {
        "target_file": "target_file",
        "git_diff": "git_diff",
        "changed_files": "changed_files",
        "imports": "imports",
        "related": "related",
        "files": "files",
        "tree": "tree",
        # Role-level keys that don't apply to file context (ignored)
        "system_prompt": None,
        "task_description": None,
        "task": None,
        "readme": None,
        "docs": None,
        "standards": None,
    }

    def _combine_file_context(
        self,
        context_parts: dict[str, str],
        budget: int,
        role: RoleConfig,
    ) -> str:
        """Combine file context parts within budget, using file-specific priority.

        Maps role.context.priority_order keys to file-context keys where applicable.
        Falls back to FILE_CONTEXT_PRIORITY when no overlap.
        Protected keys (always_include) are never pruned.
        """
        char_budget = budget * 4  # 4 chars ~= 1 token

        # Map role priority to file-context keys, filtering out non-applicable keys
        role_priority = role.context.get("priority_order", [])
        mapped_priority = []
        for key in role_priority:
            mapped = self.PRIORITY_KEY_MAP.get(key, key)  # Default: use key as-is
            if mapped is not None and mapped in context_parts:
                mapped_priority.append(mapped)

        # If no overlap with file context keys, use default file priority
        priority_order = mapped_priority if mapped_priority else self.FILE_CONTEXT_PRIORITY

        # Protected keys for file context (always_include is protected)
        protected = {"always_include"}
        protected_parts = {k: v for k, v in context_parts.items() if k in protected}
        prunable_parts = {k: v for k, v in context_parts.items() if k not in protected}

        # Try full context first
        full = "\n\n".join(context_parts.values())
        if len(full) <= char_budget:
            return full

        # Progressive pruning by role priority (reverse = low priority first)
        for drop_key in reversed(priority_order):
            if drop_key in prunable_parts:
                del prunable_parts[drop_key]
                current = "\n\n".join({**protected_parts, **prunable_parts}.values())
                if len(current) <= char_budget:
                    return current

        # Return protected content only
        return "\n\n".join(protected_parts.values())

    def _get_changed_files(self) -> list[str]:
        """Get list of changed files from git diff --name-only --cached.

        Used to resolve $CHANGED_FILES pattern in role include patterns.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--cached"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")
            return []
        except Exception:
            return []

    def render_prompt(
        self,
        template_name: str,
        role: RoleConfig,
        task_description: str,
        context: str,
        **kwargs: Any,
    ) -> str:
        """Render a template with provided context.

        SECURITY: Template name is validated against allowlist.

        NOTE: Prefer using build_full_prompt() which handles context packing.
        Use this directly only when you have pre-built context.
        """
        # SECURITY: Validate template name against allowlist
        if template_name not in ALLOWED_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Allowed: {sorted(ALLOWED_TEMPLATES)}"
            )

        template = self.jinja_env.get_template(template_name)
        return template.render(
            role=role,
            task=task_description,
            context=context,
            **kwargs,
        )

    def _truncate_lines(self, text: str, max_lines: int) -> str:
        """Truncate text to max_lines, adding notice if truncated."""
        lines = text.split('\n')
        if len(lines) <= max_lines:
            return text
        return '\n'.join(lines[:max_lines]) + f'\n\n[Truncated {len(lines) - max_lines} lines]'

    def _format_diff(self, diff: str) -> str:
        """Format git diff for prompt inclusion."""
        return f"```diff\n{diff}\n```"
```

**Acceptance Criteria:**
- [ ] Templates render without errors
- [ ] Role system_prompt is injected
- [ ] Context is properly included
- [ ] Output schema requirements are present
- [ ] Custom filters work correctly
- [ ] SandboxedEnvironment is used
- [ ] Template names are validated against allowlist
- [ ] Undefined variables raise errors (StrictUndefined)

---

### Task 3: Structured Output Parsing

**Goal:** Add CLI-specific adapters to existing parsing infrastructure. No marker fallback.

**Files to MODIFY (not create):**
- `supervisor/core/parser.py` (MODIFY) - Add CLI adapters
- `supervisor/core/models.py` (KEEP) - Existing schemas are correct

**IMPORTANT:** Output schemas already exist in `supervisor/core/models.py` with correct fields:

**NOTE on ReviewOutput redundancy:** The existing `ReviewOutput` has both `status` and `review_status`
fields with identical validation patterns. This redundancy exists in the base role prompts
(reviewer.yaml requires both fields). For Phase 2, we preserve this to maintain compatibility.
A future cleanup should consolidate to a single field.

```python
# EXISTING in models.py - DO NOT RECREATE
class PlanOutput(BaseModel):
    status: str = Field(..., pattern="^(COMPLETE|NEEDS_REFINEMENT|BLOCKED)$")
    phases: list[dict[str, Any]]
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    estimated_components: int
    risks: list[str] = Field(default_factory=list)
    next_step: str | None = None  # Required by prompts

class ImplementationOutput(BaseModel):
    status: str = Field(..., pattern="^(SUCCESS|PARTIAL|FAILED|BLOCKED)$")
    action_taken: str  # Required by prompts
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    tests_written: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_step: str | None = None  # Required by prompts

class ReviewOutput(BaseModel):
    status: str = Field(..., pattern="^(APPROVED|CHANGES_REQUESTED|REJECTED)$")
    review_status: str  # Required by prompts
    issues: list[ReviewIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    security_concerns: list[str] = Field(default_factory=list)
    next_step: str | None = None
```

**Parser Implementation:**

```python
# parser.py - ENHANCES existing parser.py, doesn't replace it
# The existing extract_json_block() and parse_worker_output() are KEPT.
# This adds CLI adapters that use the existing infrastructure.

import json
import re
from typing import TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

# EXISTING in parser.py - keep as-is
class ParsingError(Exception):
    """Failed to parse structured output."""
    pass

class InvalidOutputError(Exception):
    """Worker output doesn't match expected schema."""
    pass

# EXISTING - DO NOT MODIFY
def extract_json_block(raw_output: str) -> str | None:
    """Extract JSON block from worker output.

    STRICT MODE: Only accepts explicit ```json code blocks.
    No fallback to raw JSON detection - this prevents spoofing attacks
    where malicious content could be injected as "valid JSON".

    Workers MUST format their output as:
        ```json
        { "status": "...", ... }
        ```
    """
    # ONLY accept explicit ```json blocks - no fallbacks
    json_block_pattern = r"```json\s*([\s\S]*?)\s*```"
    matches = re.findall(json_block_pattern, raw_output)

    if not matches:
        return None

    # Return the last JSON block (most likely to be the structured output)
    return matches[-1].strip()

# EXISTING - DO NOT MODIFY
def parse_worker_output(raw_output: str, schema: type[T]) -> T:
    """Extract and validate structured output from worker response.

    STRICT MODE: No fallback to marker detection.
    """
    json_str = extract_json_block(raw_output)
    if not json_str:
        raise ParsingError(
            "No JSON block found in worker output. "
            "Worker must include output in ```json ... ``` block."
        )

    try:
        return schema.model_validate_json(json_str)
    except ValidationError as e:
        raise InvalidOutputError(f"Output doesn't match {schema.__name__} schema: {e}")
    except json.JSONDecodeError as e:
        raise ParsingError(f"Invalid JSON in output: {e}")
```

**IMPORTANT - Parser Contract and CLI Output Formats:**

The existing `parser.py` enforces **STRICT MODE** - only fenced ```json blocks are accepted.
NO fallback to raw JSON detection. This is a security feature to prevent spoofing attacks.

- Workers MUST use: ` ```json ... ``` `
- Workers MUST NOT rely on bare `{...}` JSON being detected
- This applies to ALL CLIs (Claude, Codex, Gemini)

**CLARIFICATION - CLI Output Formats and Adapter Extraction:**

The executor runs CLIs with JSON/JSONL output flags (see `supervisor/sandbox/executor.py:595-602`):
- `claude`: `["claude", "-p"]` - outputs markdown (no JSON wrapper)
- `codex`: `["codex", "exec", "--json", "--stdin"]` - outputs JSONL events
- `gemini`: `["gemini", "-o", "json"]` - outputs JSON envelope

**CRITICAL:** CLI adapters must extract the MODEL TEXT from these formats FIRST,
then run `parse_worker_output()` on that text to find fenced ` ```json ... ``` ` blocks.

| CLI | Output Format | Adapter Extraction |
|-----|---------------|-------------------|
| Claude | Raw markdown | Direct - text IS stdout |
| Codex | JSONL events | Extract `text` from `type: result` event |
| Gemini | JSON envelope | Extract `output`/`text` field from JSON |

**NOTE:** The `role.flags` in role configs are currently NOT passed to the executor
(see `supervisor/core/engine.py:572-573` - `client.execute` uses fixed args from CLIClient).
Phase 2 does NOT change this behavior. Role flags are for documentation/future use.
If per-role flags are needed, this requires extending CLIClient (deferred to Phase 4).

**REQUIRED FIX in existing parser.py:**

The current `parse_role_output()` function has an unsafe fallback that must be fixed.
The fix uses `GenericOutput` for unknown roles (not unvalidated `json.loads`):

```python
# CURRENT (UNSAFE) - in parser.py lines 115-126
def parse_role_output(role_name: str, raw_output: str) -> BaseModel:
    parser = ROLE_PARSERS.get(role_name)
    if parser:
        return parser(raw_output)

    # XXX: UNSAFE FALLBACK - accepts any JSON without schema validation
    json_str = extract_json_block(raw_output)
    if json_str:
        return json.loads(json_str)  # No schema enforcement!

    raise ParsingError(f"No parser for role '{role_name}' and no JSON found")

# FIXED VERSION - use GenericOutput for unknown roles (still validates via Pydantic)
def parse_role_output(role_name: str, raw_output: str) -> BaseModel:
    parser = ROLE_PARSERS.get(role_name)
    if parser:
        return parser(raw_output)

    # Unknown roles use GenericOutput - still validates fenced JSON and Pydantic model
    # This is SAFE because:
    # 1. extract_json_block() only accepts fenced ```json blocks
    # 2. GenericOutput.model_validate() runs Pydantic validation
    # 3. extra="allow" allows any fields, but they're still type-checked
    return parse_worker_output(raw_output, GenericOutput)
```

**NOTE on unknown-role parsing:** Unknown/overlay roles use `GenericOutput` (not `dict`).
This provides:
- Strict fenced JSON extraction (no raw JSON)
- Pydantic model validation
- Flexible schema (extra="allow") for custom role outputs
- A `status` field with default "UNKNOWN" for consistency

**CLI-Specific Adapters (NEW - add to parser.py):**

```python
# parser.py (additions to existing module)

class CLIAdapter:
    """Base adapter for CLI-specific output handling.

    IMPORTANT: Adapters must extract MODEL TEXT from CLI output format,
    then use parse_worker_output() to find fenced ```json blocks.
    """

    def parse_output(self, stdout: str, schema: type[T]) -> T:
        raise NotImplementedError

    def _extract_model_text(self, stdout: str) -> str:
        """Extract model text from CLI-specific output format.

        Override in subclasses for CLIs that wrap output in JSON/JSONL.
        """
        return stdout  # Default: stdout IS the model text

class ClaudeAdapter(CLIAdapter):
    """Adapter for Claude Code CLI output.

    Claude Code with -p outputs raw markdown (no JSON wrapper).
    Model text IS the stdout.
    """

    def parse_output(self, stdout: str, schema: type[T]) -> T:
        # Claude outputs raw markdown - use directly
        model_text = self._extract_model_text(stdout)
        return parse_worker_output(model_text, schema)

class CodexAdapter(CLIAdapter):
    """Adapter for Codex CLI output (JSONL events).

    Codex with --json emits JSONL events. The model text is in the
    'text' field of the 'result' event.

    SECURITY: No direct payload validation - extract text then parse fenced JSON.
    """

    def _extract_model_text(self, stdout: str) -> str:
        """Extract model text from Codex JSONL output."""
        lines = stdout.strip().split('\n')
        for line in reversed(lines):
            try:
                event = json.loads(line)
                if event.get('type') == 'result':
                    # Model text is in the 'text' field of the payload
                    # Codex JSONL: {"type": "result", "payload": {"text": "..."}}
                    payload = event.get('payload', {})
                    model_text = payload.get('text') or payload.get('output') or ''
                    if model_text:
                        return model_text
            except json.JSONDecodeError:
                continue

        raise ParsingError(
            "No 'result' event with text found in Codex JSONL output."
        )

    def parse_output(self, stdout: str, schema: type[T]) -> T:
        # Extract model text from JSONL, then parse fenced JSON
        model_text = self._extract_model_text(stdout)
        return parse_worker_output(model_text, schema)

class GeminiAdapter(CLIAdapter):
    """Adapter for Gemini CLI output (JSON envelope).

    Gemini with -o json outputs a JSON envelope. The model text is
    in the 'output' or 'text' field.
    """

    def _extract_model_text(self, stdout: str) -> str:
        """Extract model text from Gemini JSON envelope."""
        try:
            envelope = json.loads(stdout)
            # Gemini JSON: {"output": "...", ...} or {"text": "..."}
            model_text = envelope.get('output') or envelope.get('text') or ''
            if model_text:
                return model_text
        except json.JSONDecodeError:
            pass

        # If not valid JSON, assume raw markdown output (fallback)
        return stdout

    def parse_output(self, stdout: str, schema: type[T]) -> T:
        # Extract model text from JSON envelope, then parse fenced JSON
        model_text = self._extract_model_text(stdout)
        return parse_worker_output(model_text, schema)

def get_adapter(cli: str) -> CLIAdapter:
    """Get appropriate adapter for CLI type."""
    adapters = {
        'claude': ClaudeAdapter(),
        'codex': CodexAdapter(),
        'gemini': GeminiAdapter(),
    }
    return adapters.get(cli, ClaudeAdapter())
```

**Acceptance Criteria:**
- [ ] JSON blocks are extracted from markdown
- [ ] Last JSON block is used (final output)
- [ ] Schema validation catches missing/invalid fields
- [ ] Each CLI type has working adapter
- [ ] No marker/magic string fallback exists

---

### Task 4: Enhanced Context Packing

**Goal:** Complete the context packing implementation with token budget enforcement.

**Files to Modify:**
- `supervisor/core/context.py` (ENHANCE)

**NOTE:** Much of `_pack_with_budget` and `PRIORITY_ORDER` is already implemented in Phase 1.
This task focuses on:
1. Implementing missing `_pack_target_files` method
2. Adding `always_include` support
3. Verifying existing implementation works correctly

**CLARIFICATION - Two Context Packing Strategies:**

| Method | Use Case | What It Returns |
|--------|----------|-----------------|
| `pack_context()` | Legacy/direct use OR fallback for unknown roles | Complete prompt (system_prompt + task + files) |
| `pack_file_context()` | Template-based flow | FILE context only (templates add system_prompt + task) |
| `build_full_prompt()` | Main entry point for known roles | Calls pack_file_context + renders template |

For **known roles** (planner, implementer, reviewer): Use `build_full_prompt()` → templates.
For **unknown/overlay roles**: Fallback to `pack_context()` → direct prompt building.

**REQUIRED UPDATE to existing `_pack_files()` and `_resolve_patterns()` methods:**

Add `skip_targets` parameter to prevent target file duplication in template flow:

```python
def _pack_files(
    self,
    role: RoleConfig,
    target_files: list[str] | None = None,
    skip_targets: bool = False,  # NEW: Skip $TARGET when already packed
) -> str:
    """Pack files using repomix or fallback to simple packing."""
    include_patterns = role.include_patterns.copy()
    exclude_patterns = role.exclude_patterns.copy()

    # Add target files to includes (unless skip_targets=True)
    if target_files and not skip_targets:
        include_patterns.extend(target_files)

    # IMPORTANT: Pass skip_targets to _resolve_patterns to ignore $TARGET expansion
    include_patterns = self._resolve_patterns(include_patterns, target_files, skip_targets)

    # ... rest of existing implementation ...

def _resolve_patterns(
    self,
    patterns: list[str],
    target_files: list[str] | None,
    skip_targets: bool = False,  # NEW: Ignore $TARGET when already packed
) -> list[str]:
    """Resolve special pattern variables.

    Handles:
    - $TARGET: Expands to target_files (unless skip_targets=True)
    - $TARGET_IMPORTS: TODO
    - $CHANGED_FILES: Handled separately in pack_file_context (not a file pattern)
    - "git diff --cached": Sentinel for git diff, not a file (stripped here)
    """
    # Sentinel patterns that are NOT file patterns (handled elsewhere)
    SENTINELS = frozenset(["git diff --cached", "$CHANGED_FILES"])

    resolved = []
    for pattern in patterns:
        if pattern in SENTINELS:
            # Skip sentinels - they're handled separately, not file patterns
            continue
        elif pattern == "$TARGET":
            # Skip $TARGET expansion when already packed via _pack_target_files
            if not skip_targets and target_files:
                resolved.extend(target_files)
        elif pattern == "$TARGET_IMPORTS":
            # TODO: Implement import resolution
            pass
        elif pattern.startswith("$"):
            # Skip unresolved variables
            pass
        else:
            resolved.append(pattern)
    return resolved
```

This prevents duplication when `pack_file_context()` already includes target files via
`_pack_target_files()` and the role config also has `$TARGET` in its include patterns.

**Implementation:**

```python
class ContextPacker:
    # Rough estimation: 4 chars ~= 1 token (conservative)
    CHARS_PER_TOKEN = 4

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return len(text) // self.CHARS_PER_TOKEN

    def pack_context(
        self,
        role: RoleConfig,
        task_description: str,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
    ) -> str:
        """Pack context for a role within token budget."""
        context_parts: dict[str, str] = {}

        # Build context parts in priority order
        context_parts["system_prompt"] = role.system_prompt
        context_parts["task"] = f"## Task\n\n{task_description}"

        # Pack target files (high priority - files being modified)
        if target_files:
            target_content = self._pack_target_files(target_files)
            if target_content:
                context_parts["target_file"] = target_content

        # Pack always_include files (always present, even if over budget)
        always_include = role.context.get("always_include", [])
        if always_include:
            always_content = self._pack_always_include(always_include)
            if always_content:
                context_parts["always_include"] = always_content

        # Handle role-specific context directives (same as pack_file_context)
        # This ensures overlay/custom roles using sentinels get proper context
        include_patterns = role.context.get("include", [])
        if role.context.get("git_diff") or "git diff --cached" in include_patterns:
            git_diff = self.get_git_diff(staged=True)
            if git_diff:
                context_parts["git_diff"] = f"## Git Diff (Staged)\n\n```diff\n{git_diff}\n```"

        if "$CHANGED_FILES" in include_patterns:
            changed_files = self._get_changed_files()
            if changed_files:
                changed_content = self._pack_target_files(changed_files)
                if changed_content:
                    context_parts["changed_files"] = changed_content

        # Pack role-specific files (may be pruned if over budget)
        # NOTE: skip_targets=True since target_files already packed above via _pack_target_files
        file_context = self._pack_files(role, target_files, skip_targets=True)
        if file_context:
            context_parts["files"] = file_context

        if extra_context:
            for key, value in extra_context.items():
                context_parts[key] = f"## {key.replace('_', ' ').title()}\n\n{value}"

        return self._pack_with_budget(context_parts, role.token_budget)

    def _pack_target_files(self, target_files: list[str]) -> str:
        """Pack specific target files that are being modified.

        These get highest priority in context (after system_prompt and task).
        """
        parts = ["## Target Files\n"]

        for file_path in target_files:
            full_path = self.repo_path / file_path

            # Validate path stays within repo
            validated = self._validate_file_path(full_path)
            if validated is None:
                continue

            try:
                # Check file size
                if validated.stat().st_size > self.MAX_FILE_SIZE:
                    parts.append(f"### {file_path}\n\n[File exceeds size limit]\n")
                    continue

                content = validated.read_text()
                parts.append(f"### {file_path}\n\n```\n{content}\n```\n")
            except Exception:
                parts.append(f"### {file_path}\n\n[Could not read file]\n")

        return "\n".join(parts) if len(parts) > 1 else ""

    # Cap for always_include to prevent unbounded growth
    # Even protected content has a limit to avoid memory/timeout issues
    MAX_ALWAYS_INCLUDE_SIZE = 2 * 1024 * 1024  # 2MB total for always_include

    def _pack_always_include(self, patterns: list[str]) -> str:
        """Pack always_include files - protected but size-capped.

        Used for critical context like risk_limits.py, config files, etc.
        Never pruned by budget logic, but has its own hard cap.
        """
        parts = ["## Required Context\n"]
        total_size = 0

        for pattern in patterns:
            # Sanitize and resolve pattern
            safe_pattern = self._sanitize_pattern(pattern)
            if safe_pattern is None:
                continue

            # Handle glob patterns vs exact paths
            if "*" in safe_pattern or "?" in safe_pattern:
                for file_path in self.repo_path.glob(safe_pattern):
                    validated = self._validate_file_path(file_path)
                    if validated and validated.is_file():
                        added = self._append_file_content_with_cap(
                            parts, validated, total_size, self.MAX_ALWAYS_INCLUDE_SIZE
                        )
                        total_size += added
            else:
                file_path = self.repo_path / safe_pattern
                validated = self._validate_file_path(file_path)
                if validated and validated.is_file():
                    added = self._append_file_content_with_cap(
                        parts, validated, total_size, self.MAX_ALWAYS_INCLUDE_SIZE
                    )
                    total_size += added

        if total_size >= self.MAX_ALWAYS_INCLUDE_SIZE:
            parts.append(f"\n[WARNING: always_include capped at {self.MAX_ALWAYS_INCLUDE_SIZE // 1024}KB]")

        return "\n".join(parts) if len(parts) > 1 else ""

    def _append_file_content_with_cap(
        self, parts: list[str], file_path: Path, current_size: int, max_size: int
    ) -> int:
        """Append file content with size tracking. Returns bytes added."""
        try:
            file_size = file_path.stat().st_size

            # Check per-file limit
            if file_size > self.MAX_FILE_SIZE:
                rel_path = file_path.relative_to(self.repo_path)
                parts.append(f"### {rel_path}\n\n[File exceeds size limit]\n")
                return 0

            # Check total limit
            if current_size + file_size > max_size:
                rel_path = file_path.relative_to(self.repo_path)
                parts.append(f"### {rel_path}\n\n[Skipped: would exceed total size cap]\n")
                return 0

            content = file_path.read_text()
            rel_path = file_path.relative_to(self.repo_path)
            parts.append(f"### {rel_path}\n\n```\n{content}\n```\n")
            return len(content.encode("utf-8"))
        except Exception:
            return 0  # Skip unreadable files

    # Keys that are NEVER pruned, regardless of budget
    PROTECTED_KEYS = frozenset(["system_prompt", "task", "always_include"])

    # Priority order for pruning (only non-protected keys are pruned)
    PRUNABLE_PRIORITY_ORDER = [
        "target_file",     # High priority (last to prune)
        "imports",
        "related",
        "files",
        "tree",            # Low priority (first to prune)
    ]

    def _pack_with_budget(
        self,
        context_parts: dict[str, str],
        budget: int,
    ) -> str:
        """Pack context within token budget, with protected keys never pruned.

        Protected keys (system_prompt, task, always_include) are NEVER dropped.
        Only prunable keys are progressively dropped to fit budget.
        If still over budget after pruning all prunable keys, return with warning.
        """
        char_budget = budget * 4  # 4 chars ~= 1 token

        # Always include protected keys
        protected = {k: v for k, v in context_parts.items() if k in self.PROTECTED_KEYS}
        prunable = {k: v for k, v in context_parts.items() if k not in self.PROTECTED_KEYS}

        # Try full context first
        full = self._combine(context_parts)
        if len(full) <= char_budget:
            return full

        # Progressive pruning of ONLY prunable keys (reverse priority = low priority first)
        for drop_key in reversed(self.PRUNABLE_PRIORITY_ORDER):
            if drop_key in prunable:
                del prunable[drop_key]
                current = self._combine({**protected, **prunable})
                if len(current) <= char_budget:
                    return current

        # Still over budget with only protected keys - return with warning
        protected_only = self._combine(protected)
        if len(protected_only) > char_budget:
            # Even protected content exceeds budget - truncate with warning
            # NEVER drop protected content, but warn user
            return protected_only + "\n\n[WARNING: Context exceeds budget even with required content only]"

        return protected_only
```

**Acceptance Criteria:**
- [ ] Token estimation works correctly
- [ ] Context fits within budget
- [ ] Lower priority items are dropped first
- [ ] Truncation notice is added when needed
- [ ] `_pack_target_files` method works correctly
- [ ] `always_include` files are NEVER pruned (protected)
- [ ] `system_prompt` and `task` are NEVER pruned (protected)
- [ ] Existing `_pack_with_budget` logic verified working

---

### Task 5: Runtime Integration

**Goal:** Wire templates and parsing into the EXISTING `ExecutionEngine` path.

**Files to MODIFY:**
- `supervisor/core/engine.py` (MODIFY) - Update existing `ExecutionEngine.run_role()`
- `supervisor/core/parser.py` (MODIFY) - Add CLI adapters and GenericOutput

**IMPORTANT:** Do NOT create a new `WorkflowEngine` class. The existing `ExecutionEngine`
in `engine.py:231-574` already has:
- Circuit breaker, retry logic, event logging
- Isolated worktree execution
- Gate verification
- Context packing via `self.context_packer.pack_context()`

Phase 2 modifies this existing flow to add template support and CLI adapters.

**Changes to existing ExecutionEngine.run_role() (engine.py:294-450):**

```python
# EXISTING ExecutionEngine class - modify run_role() method

class ExecutionEngine:
    # ... existing __init__ ...

    def run_role(
        self,
        role_name: str,
        task_description: str,
        workflow_id: str,
        step_id: str | None = None,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
        retry_policy: RetryPolicy | None = None,
        gates: list[str] | None = None,
    ) -> BaseModel:
        """Run a role with full context packing, isolation, and error handling.

        PHASE 2 CHANGES:
        - Line ~348: Use template-based prompt building for known roles
        - Line ~389: Use CLI adapter instead of parse_role_output directly
        """
        # ... existing setup code (circuit breaker, step_id generation) ...

        # Load role configuration (NOW with schema validation)
        role = self.role_loader.load_role(role_name)

        # PHASE 2 CHANGE: Use template-based prompt building for known roles
        # NOTE: Pass role object (not role_name) to resolve overlay extends chain
        template_name = self._get_template_for_role(role)
        if template_name:
            # Template-based flow (planner, implementer, reviewer, AND overlays that extend them)
            prompt = self.context_packer.build_full_prompt(
                template_name,
                role,
                task_description,
                target_files,
                extra_context,
            )
        else:
            # Legacy fallback for truly unknown roles (no extends to known base)
            prompt = self.context_packer.pack_context(
                role=role,
                task_description=task_description,
                target_files=target_files,
                extra_context=extra_context,
            )

        # ... existing event recording ...

        for attempt in range(retry_policy.max_attempts):
            try:
                # ... existing feedback handling ...

                with self.workspace.isolated_execution(step_id) as ctx:
                    result = self._execute_cli(role, effective_prompt, ctx.worktree_path)

                    # ... existing error handling ...

                    # PHASE 2 CHANGE: Use CLI adapter for parsing
                    adapter = get_adapter(role.cli)
                    # NOTE: Use _get_schema_for_role to resolve overlay extends chain
                    schema = self._get_schema_for_role(role)
                    output = adapter.parse_output(result.stdout, schema)

                    # ... existing gate verification (unchanged) ...

    def _get_template_for_role(self, role: RoleConfig) -> str | None:
        """Map role to template, resolving overlays via base_role.

        For overlay roles (e.g., reviewer-python extends reviewer),
        uses the base role's template.

        base_role is computed by RoleLoader and handles multi-level overlays:
        - reviewer-python (extends reviewer) → base_role = "reviewer"
        - my-reviewer (extends reviewer-python extends reviewer) → base_role = "reviewer"

        Returns None only for truly unknown roles to use legacy pack_context.
        """
        templates = {
            "planner": "planning.j2",
            "implementer": "implement.j2",
            "reviewer": "review_strict.j2",
        }

        # First try exact role name (for base roles)
        if role.name in templates:
            return templates[role.name]

        # Then try base_role (for overlays - handles multi-level extends)
        if role.base_role and role.base_role in templates:
            return templates[role.base_role]

        # Unknown role - fallback to legacy pack_context
        return None

    def _get_schema_for_role(self, role: RoleConfig) -> type[BaseModel]:
        """Get output schema for role, resolving overlays via base_role.

        For overlay roles (e.g., implementer-python extends implementer),
        uses the base role's schema.

        base_role is computed by RoleLoader and handles multi-level overlays.
        """
        # First try exact role name (for base roles)
        if role.name in ROLE_SCHEMAS:
            return ROLE_SCHEMAS[role.name]

        # Then try base_role (for overlays - handles multi-level extends)
        if role.base_role and role.base_role in ROLE_SCHEMAS:
            return ROLE_SCHEMAS[role.base_role]

        # Unknown role - use GenericOutput
        return GenericOutput
```

**Additions to parser.py:**

```python
# Add these to supervisor/core/parser.py

# Generic output schema for unknown/overlay roles
class GenericOutput(BaseModel):
    """Generic output for overlay/custom roles.

    Accepts any JSON structure while maintaining Pydantic validation.
    All adapters require a Pydantic model (not dict) for model_validate*.
    """
    model_config = {"extra": "allow"}
    status: str = Field(default="UNKNOWN")

# Role to schema mapping (uses existing models from models.py)
ROLE_SCHEMAS: dict[str, type[BaseModel]] = {
    "planner": PlanOutput,
    "implementer": ImplementationOutput,
    "reviewer": ReviewOutput,
}

# CLI adapters (ClaudeAdapter, CodexAdapter, GeminiAdapter, get_adapter)
# ... as defined in Task 3 ...
```

**Acceptance Criteria:**
- [ ] `run_role()` uses templates for prompt generation
- [ ] Template selection works for all base roles
- [ ] CLI adapter selection works correctly
- [ ] Output parsing uses existing models from `models.py`

---

## Integration Points

### With Phase 1 Components

| Phase 2 Component | Integrates With | Integration Method |
|-------------------|-----------------|-------------------|
| Role schema validation | `RoleLoader` | Called during `load_role()` |
| Jinja2 templates | `ContextPacker` | `build_full_prompt()` in engine |
| Output parser | `run_role()` | CLI adapters post-execution |
| CLI adapters | `parser.py` | Added to existing module |

### New Dependencies

```toml
# Add to pyproject.toml
dependencies = [
    "jinja2>=3.1.0",
    "pydantic>=2.0.0",
    "jsonschema>=4.0.0",
]
```

---

## Testing Strategy

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_roles.py` | Schema validation, inheritance |
| `tests/test_parser.py` | JSON extraction, schema validation |
| `tests/test_context.py` | Token budget, priority pruning |
| `tests/test_templates.py` | Jinja2 rendering |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_role_to_prompt_flow` | Load role → pack context → render template |
| `test_cli_output_parsing` | Mock CLI output → parse → validate schema |
| `test_token_budget_enforcement` | Large context → prune → verify fits |

### Example Test Cases

```python
# tests/test_parser.py
def test_extract_json_from_markdown():
    """JSON block in markdown is extracted."""
    output = """
    Here's my analysis:

    ```json
    {"status": "SUCCESS", "files_created": ["foo.py"]}
    ```
    """
    result = parse_worker_output(output, ImplementationOutput)
    assert result.status == "SUCCESS"

def test_last_json_block_used():
    """When multiple JSON blocks, last one is used."""
    output = """
    First attempt (wrong):
    ```json
    {"status": "FAILED"}
    ```

    Fixed version:
    ```json
    {"status": "SUCCESS", "files_created": []}
    ```
    """
    result = parse_worker_output(output, ImplementationOutput)
    assert result.status == "SUCCESS"

def test_no_marker_fallback():
    """Magic strings like 'APPROVED' are NOT parsed."""
    output = "REVIEW_STATUS: APPROVED\n\nLooks good!"
    with pytest.raises(ParsingError, match="No JSON block found"):
        parse_worker_output(output, ReviewOutput)
```

---

## Security Considerations

### Jinja2 Template Safety

```python
from jinja2 import FileSystemLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

# SECURITY: Use SandboxedEnvironment to prevent arbitrary code execution
# StrictUndefined raises errors on undefined variables (catches typos)
self.jinja_env = SandboxedEnvironment(
    loader=FileSystemLoader(template_dir),
    undefined=StrictUndefined,
    autoescape=False,  # Not HTML, no XSS concern
    trim_blocks=True,
    lstrip_blocks=True,
)
# SECURITY: Template names validated against allowlist (ALLOWED_TEMPLATES)
```

### Parser Security

- **No eval()**: Only `json.loads()` for parsing
- **No marker detection**: Prevents spoofing via output manipulation
- **Schema validation**: Rejects unexpected fields/types
- **Last fenced block used**: When multiple ```json blocks exist, the last one is parsed
  (workers typically output the final structured result at the end)
- **Note**: No explicit trailing-text enforcement - parser extracts last fenced JSON block
  regardless of content after. This is intentional to handle CLI output that may include
  trailing logging or status messages.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Token estimation inaccurate | Medium | Low | Conservative 4 chars/token ratio |
| CLI output format changes | Low | Medium | Adapter pattern allows quick fixes |
| Template injection | Low | High | Templates are package files, not user input |
| Large context OOM | Medium | Medium | File size limits from Phase 1 |

---

## Completion Criteria

Phase 2 is complete when:

1. [ ] Role configs validate against JSON Schema
2. [ ] All base role templates exist and render correctly
3. [ ] Output parser extracts JSON from all CLI formats
4. [ ] Schema validation catches invalid outputs
5. [ ] Context packing respects token budgets
6. [ ] All unit tests pass
7. [ ] Integration tests pass
8. [ ] No marker/magic string fallbacks exist anywhere

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-01-04 | Claude | Initial Phase 2 plan |
| 2025-01-04 | Claude | Revision 2: Addressed Gemini + Codex feedback |
| 2025-01-04 | Claude | Revision 3: Keep gates as list[str], protect always_include from pruning, fix Jinja2 docs |
| 2025-01-04 | Claude | Revision 4: Fix template blocks, size-cap always_include, strict CodexAdapter, YAML error handling |
| 2025-01-04 | Claude | Revision 5: Integration with existing parser.py/models.py, loosen schema for role fields, runtime wiring, fix prompt duplication |
| 2025-01-04 | Claude | Revision 6: Budget reservation for templates, git_diff handling, pre-merge type validation, parser contract consistency (strict fenced-json only) |
| 2025-01-04 | Claude | Revision 7: Schema patternProperties for x- extensions, $CHANGED_FILES resolution, role-specific priority_order, _combine_file_context method, fix parse_role_output fallback, template fallback to pack_context, document ReviewOutput redundancy |
| 2025-01-04 | Claude | Revision 8: GenericOutput for unknown roles (not dict), file-specific priority list with key mapping, update parser docs (no trailing-text enforcement claim) |
| 2025-01-04 | Claude | Revision 9: Clarify CLI flags vs worker output format, integrate with existing ExecutionEngine (not new WorkflowEngine), note role.flags are not currently passed to executor |
| 2025-01-04 | Claude | Revision 10: CLI adapters extract model text from JSON/JSONL envelopes before parsing, RoleLoader uses __post_init__ for schema loading, resolve unknown-role parsing (use GenericOutput), fix target file duplication with skip_targets parameter |
| 2025-01-04 | Claude | Revision 11: Template explicitly requires fenced ```json blocks, complete $TARGET duplication fix in _resolve_patterns, schema load error handling with RoleValidationError |
| 2025-01-04 | Claude | Revision 12: Fix legacy pack_context duplication (skip_targets=True), strip git diff --cached sentinel from _resolve_patterns, catch jsonschema.SchemaError |
| 2025-01-04 | Claude | Revision 13: Overlay roles use base template/schema via extends chain, add git_diff/$CHANGED_FILES handling to legacy pack_context |
| 2025-01-04 | Claude | Revision 14: Add base_role field to RoleConfig (root of extends chain), RoleLoader computes base_role during load, handles multi-level overlays |
