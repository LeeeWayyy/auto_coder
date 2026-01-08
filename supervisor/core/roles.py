"""Role configuration loading with inheritance support.

Roles use a Base + Overlay model:
- Base roles: planner, implementer, reviewer (shipped with supervisor)
- Domain overlays: Extend base roles with domain knowledge
- Merge semantics: Lists append, dicts deep merge, scalars override
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from supervisor.core.gates import GateFailAction

# Valid role name pattern: alphanumeric, underscores, hyphens only
# Prevents path traversal attacks via names like "../../etc/passwd"
_VALID_ROLE_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


class RoleNotFoundError(Exception):
    """Role configuration not found."""

    pass


class RoleValidationError(Exception):
    """Role configuration is invalid."""

    pass


class RoleCycleError(Exception):
    """Circular dependency detected in role inheritance."""

    pass


@dataclass
class RoleGateConfig:
    """Gate configuration within a role.

    Supports both legacy format (list of strings) and new format (list of dicts).
    """

    name: str
    on_fail: GateFailAction | None = None
    required: bool = True


@dataclass
class RoleConfig:
    """Configuration for a worker role."""

    name: str
    description: str
    cli: str
    flags: list[str]
    system_prompt: str
    context: dict[str, Any]
    gates: list[RoleGateConfig]
    config: dict[str, Any]
    extends: str | None = None
    base_role: str | None = None  # Root of extends chain (planner/implementer/reviewer)

    @property
    def token_budget(self) -> int:
        return self.context.get("token_budget", 20000)

    @property
    def include_patterns(self) -> list[str]:
        return self.context.get("include", [])

    @property
    def exclude_patterns(self) -> list[str]:
        return self.context.get("exclude", [])

    @property
    def max_retries(self) -> int:
        return self.config.get("max_retries", 3)

    @property
    def timeout(self) -> int:
        return self.config.get("timeout", 300)

    @property
    def gate_names(self) -> list[str]:
        return [gate.name for gate in self.gates]

    @property
    def on_fail_overrides(self) -> dict[str, GateFailAction]:
        return {
            gate.name: gate.on_fail
            for gate in self.gates
            if gate.on_fail is not None
        }


@dataclass
class RoleLoader:
    """Load and merge role definitions with inheritance."""

    # Search paths in priority order (first found wins for each role)
    search_paths: list[Path] = field(default_factory=list)

    # Package directory for built-in roles
    package_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "config" / "base_roles"
    )

    # Schema loaded in __post_init__ to preserve dataclass init behavior
    _schema: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Default search paths
        if not self.search_paths:
            self.search_paths = [
                Path(".supervisor/roles"),  # Project-specific
                Path.home() / ".supervisor/roles",  # User-global
                self.package_dir,  # Built-in
            ]

        # PHASE 2: Load JSON schema for validation
        self._schema = self._load_schema()

    def _load_schema(self) -> dict:
        """Load JSON schema for role validation.

        Raises RoleValidationError with actionable message on failure.
        """
        schema_path = Path(__file__).parent.parent / "config" / "role_schema.json"
        try:
            with open(schema_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise RoleValidationError(
                f"Role schema not found at {schema_path}. "
                f"Ensure supervisor package is properly installed."
            )
        except json.JSONDecodeError as e:
            raise RoleValidationError(f"Invalid JSON in role schema at {schema_path}: {e}")

    def load_role(self, name: str, _loading_chain: list[str] | None = None) -> RoleConfig:
        """Load role with inheritance resolution.

        Args:
            name: Role name to load
            _loading_chain: Internal parameter for cycle detection (do not pass)

        Raises:
            RoleValidationError: If role name contains invalid characters
            RoleCycleError: If circular inheritance is detected
        """
        # SECURITY: Validate role name to prevent path traversal
        # Names like "../../etc/passwd" could escape search paths
        if not _VALID_ROLE_NAME.match(name):
            raise RoleValidationError(
                f"Invalid role name '{name}'. "
                f"Role names must start with a letter and contain only "
                f"alphanumeric characters, underscores, and hyphens."
            )

        # Cycle detection
        if _loading_chain is None:
            _loading_chain = []

        if name in _loading_chain:
            cycle = " -> ".join(_loading_chain + [name])
            raise RoleCycleError(f"Circular role inheritance detected: {cycle}")

        _loading_chain = _loading_chain + [name]  # Create new list to avoid mutation

        role_file = self._find_role_file(name)
        config = self._load_yaml(role_file)

        # PHASE 2: Pre-merge type validation to catch structural issues before merge fails
        self._validate_pre_merge_types(config, role_file)

        # PHASE 2: Capture base_role before merge removes extends
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

        # PHASE 2: Full schema validation AFTER merge
        # This allows overlay roles to omit inherited fields like system_prompt
        self._validate_merged_config(config)

        parsed_config = self._parse_role_config(config)
        return self._dict_to_role(parsed_config, base_role=base_role)

    def _find_role_file(self, name: str) -> Path:
        """Find role file in search paths."""
        for search_path in self.search_paths:
            role_file = search_path / f"{name}.yaml"
            if role_file.exists():
                return role_file

        raise RoleNotFoundError(
            f"Role '{name}' not found in search paths: {self.search_paths}"
        )

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load YAML file with proper error handling."""
        try:
            with open(path, encoding="utf-8") as f:
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

    def _validate_pre_merge_types(self, config: dict, path: Path) -> None:
        """Lightweight type checks before merge to catch structural issues early."""
        type_checks = {
            "flags": list,
            "gates": list,
            "context": dict,
            "config": dict,
        }
        for field_name, expected_type in type_checks.items():
            if field_name in config and not isinstance(config[field_name], expected_type):
                raise RoleValidationError(
                    f"Field '{field_name}' must be {expected_type.__name__}, "
                    f"got {type(config[field_name]).__name__} in {path}"
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

    def _parse_gates(self, gates_config: list) -> list[RoleGateConfig]:
        """Parse gates from role config, supporting string and dict formats."""
        parsed: list[RoleGateConfig] = []
        for idx, item in enumerate(gates_config):
            if isinstance(item, str):
                if not item:
                    raise RoleValidationError("Gate name cannot be empty")
                parsed.append(RoleGateConfig(name=item))
                continue

            if isinstance(item, dict):
                if "name" not in item or not isinstance(item["name"], str):
                    raise RoleValidationError(
                        f"Gate entry at index {idx} must include string 'name'"
                    )
                name = item["name"]
                required = item.get("required", True)
                if not isinstance(required, bool):
                    raise RoleValidationError(
                        f"Gate '{name}' field 'required' must be boolean"
                    )
                on_fail_value = item.get("on_fail")
                on_fail: GateFailAction | None = None
                if on_fail_value is not None:
                    if not isinstance(on_fail_value, str):
                        raise RoleValidationError(
                            f"Gate '{name}' field 'on_fail' must be string if set"
                        )
                    try:
                        on_fail = GateFailAction(on_fail_value.lower())
                    except ValueError as exc:
                        valid = ", ".join(a.value for a in GateFailAction)
                        raise RoleValidationError(
                            f"Gate '{name}' field 'on_fail' must be one of: {valid}"
                        ) from exc
                parsed.append(
                    RoleGateConfig(name=name, on_fail=on_fail, required=required)
                )
                continue

            raise RoleValidationError(
                f"Gate entry at index {idx} must be string or dict"
            )

        return parsed

    def _parse_role_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Normalize role config for runtime use."""
        parsed = config.copy()
        parsed["gates"] = self._parse_gates(parsed.get("gates", []))
        return parsed

    def _merge_configs(self, parent: RoleConfig, child: dict[str, Any]) -> dict[str, Any]:
        """Merge child config into parent with proper semantics.

        Merge rules:
        - Scalars (cli, timeout): Child overrides parent
        - Lists (include, gates, flags): Append child to parent
        - Dicts (context, config): Deep merge
        - system_prompt: Concatenate (parent + additions)
        """
        parent_dict = {
            "name": parent.name,
            "description": parent.description,
            "cli": parent.cli,
            "flags": parent.flags.copy(),
            "system_prompt": parent.system_prompt,
            "context": parent.context.copy(),
            "gates": self._serialize_gates(parent.gates),
            "config": parent.config.copy(),
        }

        merged = parent_dict.copy()

        # Handle each field with appropriate merge semantics
        for key, value in child.items():
            if key == "extends":
                continue  # Don't include extends in merged config

            if key == "system_prompt_additions":
                # Append to parent system prompt
                merged["system_prompt"] = (
                    parent_dict["system_prompt"] + "\n\n" + value
                )

            elif key in ("flags", "gates"):
                # Append lists
                merged[key] = parent_dict.get(key, []) + value

            elif key in ("context", "config"):
                # Deep merge dicts
                merged[key] = self._deep_merge(parent_dict.get(key, {}), value)

            else:
                # Override scalars
                merged[key] = value

        return merged

    def _deep_merge(self, parent: dict, child: dict) -> dict:
        """Deep merge two dictionaries."""
        result = parent.copy()
        for key, value in child.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value
        return result

    def _serialize_gates(self, gates: list[RoleGateConfig]) -> list[Any]:
        """Convert RoleGateConfig list back to config-friendly list."""
        serialized: list[Any] = []
        for gate in gates:
            if gate.on_fail is None and gate.required:
                serialized.append(gate.name)
            else:
                data: dict[str, Any] = {"name": gate.name, "required": gate.required}
                if gate.on_fail is not None:
                    data["on_fail"] = gate.on_fail.value
                serialized.append(data)
        return serialized

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
            base_role=base_role,
        )

    def list_available_roles(self) -> list[str]:
        """List all available roles across search paths."""
        roles = set()
        for search_path in self.search_paths:
            if search_path.exists():
                for role_file in search_path.glob("*.yaml"):
                    roles.add(role_file.stem)
        return sorted(roles)
