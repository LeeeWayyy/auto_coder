"""Gate configuration loading and validation."""

from __future__ import annotations

import heapq
import json
import re
from pathlib import Path

import yaml

from supervisor.core.gate_models import (
    PACKAGE_DIR,
    CircularDependencyError,
    GateConfig,
    GateConfigError,
    GateNotFoundError,
    GateSeverity,
)


class GateLoader:
    """Load gate configurations from YAML files with defined precedence."""

    STATIC_SEARCH_PATHS = [
        Path.home() / ".supervisor/gates.yaml",
        PACKAGE_DIR / "config/gates.yaml",
    ]

    # Protected env vars for security validation when scanning env wrappers.
    ENV_DENYLIST = frozenset(
        {
            "PATH",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "HOME",
            "USER",
        }
    )
    ENV_DENYLIST_PREFIXES = ("SUPERVISOR_",)

    # Shell binaries that require allow_shell=true
    SHELL_BINARY_NAMES = frozenset(
        {
            "bash",
            "sh",
            "zsh",
            "fish",
            "dash",
            "ksh",
            "csh",
            "tcsh",
            "cmd",
            "powershell",
            "pwsh",
        }
    )

    # Command wrappers that may wrap shell invocations
    COMMAND_WRAPPERS = frozenset(
        {
            "env",
            "command",
            "exec",
            "xargs",
            "nice",
            "nohup",
            "timeout",
            "stdbuf",
            "ionice",
            "chrt",
            "taskset",
            "numactl",
            "time",
            "chronic",
            "unbuffer",
            "sudo",
            "doas",
            "su",
            "runuser",
        }
    )

    # env flags that take a following argument (must be handled specially)
    ENV_FLAGS_WITH_ARGS = frozenset({"-C", "-u", "--chdir", "--unset"})

    # Dangerous env flags that must be blocked entirely
    ENV_BLOCKED_FLAGS = frozenset({"-S", "--split-string", "-i", "--ignore-environment"})

    def __init__(
        self,
        worktree_path: Path,
        schema_path: Path | None = None,
        allow_project_gates: bool = False,
    ) -> None:
        self._gates: dict[str, GateConfig] = {}
        self._loaded = False
        self._schema = self._load_schema(schema_path)
        self._allow_project_gates = allow_project_gates
        self._worktree_path = worktree_path.resolve()

        self._search_paths = self.STATIC_SEARCH_PATHS.copy()
        if allow_project_gates:
            self._search_paths.insert(0, self._worktree_path / ".supervisor/gates.yaml")

    def _load_schema(self, schema_path: Path | None) -> dict | None:
        path = schema_path or PACKAGE_DIR / "config/gates_schema.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _validate_config(self, config: dict, source_path: Path) -> None:
        if self._schema:
            try:
                import jsonschema

                jsonschema.validate(config, self._schema)
            except jsonschema.ValidationError as e:
                raise GateConfigError(
                    f"Gate config validation failed in {source_path}: {e.message}\n"
                    f"Path: {' -> '.join(str(p) for p in e.absolute_path)}"
                )
            except ImportError:
                pass

        self._basic_validate(config, source_path)

    # Static validation helpers (extracted for testability and reuse)
    _GATE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$")
    _ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _VALID_SEVERITIES = frozenset({"error", "warning", "info"})
    _BOOL_FIELDS = (
        "parallel_safe",
        "cache",
        "skip_on_dependency_failure",
        "allow_shell",
        "force_hash_large_cache_inputs",
    )

    @staticmethod
    def _has_path_traversal(pattern: str) -> bool:
        """Check if a pattern contains path traversal components."""
        normalized = pattern.replace("\\", "/")
        return ".." in normalized.split("/")

    @staticmethod
    def _get_basename(arg: str) -> str:
        """Extract lowercase basename from a command argument."""
        basename = arg.replace("\\", "/").split("/")[-1].lower()
        if basename.endswith(".exe"):
            basename = basename[:-4]
        return basename

    def _is_shell_binary(self, executable: str) -> bool:
        """Check if an executable is a shell binary."""
        return self._get_basename(executable) in self.SHELL_BINARY_NAMES

    @staticmethod
    def _is_env_assignment(arg: str) -> tuple[bool, str | None]:
        """Check if an argument is an environment variable assignment."""
        if "=" not in arg or arg.startswith("=") or arg.startswith("-"):
            return (False, None)
        key = arg.split("=", 1)[0]
        if not key or not (key[0].isalpha() or key[0] == "_"):
            return (False, None)
        if all(c.isalnum() or c == "_" for c in key):
            return (True, key)
        return (False, None)

    def _detect_shell_invocation(self, cmd_list: list[str]) -> str | None:
        """Detect if a command invokes a shell binary."""
        if not cmd_list:
            return None

        found_double_dash = False
        for arg in cmd_list:
            if arg == "--":
                found_double_dash = True
                continue
            if found_double_dash:
                if self._is_shell_binary(arg):
                    return arg
                break
            if arg.startswith("-"):
                continue
            is_assign, _ = self._is_env_assignment(arg)
            if is_assign:
                continue
            if self._is_shell_binary(arg):
                return arg
            if self._get_basename(arg) in self.COMMAND_WRAPPERS:
                continue
            break
        return None

    def _check_env_blocked_flags(self, cmd_list: list[str]) -> str | None:
        """Check if any blocked env flags are used with an 'env' wrapper.

        SECURITY: Only check for blocked flags (-i, -S, etc.) within env's arguments.
        Commands like 'grep -i' or 'sed -i' are legitimate and should not be blocked.
        """
        # First, find if there's an env wrapper
        env_index = self._find_env_wrapper_index(cmd_list)
        if env_index is None:
            return None  # No env wrapper, no need to check for env-specific flags

        # Only scan arguments between env and the actual command
        env_args = cmd_list[env_index + 1 :]
        for arg in env_args:
            if arg == "--":
                break  # Options end at --
            if arg in self.ENV_BLOCKED_FLAGS:
                return arg
            for blocked in self.ENV_BLOCKED_FLAGS:
                if arg.startswith(blocked + "="):
                    return blocked
            # Check for combined short flags like -iS or -Si
            if arg.startswith("-") and not arg.startswith("--") and len(arg) > 1:
                for char in arg[1:]:
                    if f"-{char}" in self.ENV_BLOCKED_FLAGS:
                        return f"-{char}"
            # If this arg is not a flag and not an assignment, it's the command - stop
            if not arg.startswith("-"):
                is_assign, _ = self._is_env_assignment(arg)
                if not is_assign:
                    break
        return None

    def _check_env_denylist_bypass(self, cmd_list: list[str]) -> str | None:
        """Check if command attempts to bypass env denylist via wrappers."""
        if not cmd_list or len(cmd_list) < 2:
            return None

        env_index = self._find_env_wrapper_index(cmd_list)
        if env_index is None:
            return None

        return self._scan_env_assignments(cmd_list[env_index + 1 :])

    def _find_env_wrapper_index(self, cmd_list: list[str]) -> int | None:
        """Find the index of an 'env' wrapper in the command."""
        found_double_dash = False
        skip_next = False
        for idx, arg in enumerate(cmd_list):
            if skip_next:
                skip_next = False
                continue
            if arg == "--":
                found_double_dash = True
                continue
            if found_double_dash:
                if self._get_basename(arg) == "env":
                    return idx
                break
            if arg.startswith("-"):
                if (
                    arg in self.ENV_FLAGS_WITH_ARGS
                    or any(arg.startswith(f + "=") for f in self.ENV_FLAGS_WITH_ARGS)
                ) and "=" not in arg:
                    skip_next = True
                continue
            is_assign, _ = self._is_env_assignment(arg)
            if is_assign:
                continue
            basename = self._get_basename(arg)
            if basename == "env":
                return idx
            if basename in self.COMMAND_WRAPPERS:
                continue
            break
        return None

    def _scan_env_assignments(self, args: list[str]) -> str | None:
        """Scan arguments for denylisted environment variable assignments.

        SECURITY: env syntax is `env [OPTION]... [-] [NAME=VALUE]... [COMMAND [ARG]...]`
        NAME=VALUE assignments can appear AFTER `--`, so we must NOT stop scanning at `--`.
        The `--` only ends option parsing, not the assignment section.
        """
        skip_next = False
        past_options = False  # Track whether we've passed the options section
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg == "--":
                # `--` ends options, but assignments follow - continue scanning
                past_options = True
                continue
            if not past_options and arg.startswith("-"):
                if (
                    arg in self.ENV_FLAGS_WITH_ARGS
                    or any(arg.startswith(f + "=") for f in self.ENV_FLAGS_WITH_ARGS)
                ) and "=" not in arg:
                    skip_next = True
                continue
            is_assign, key = self._is_env_assignment(arg)
            if is_assign and key:
                if key.upper() in self.ENV_DENYLIST:
                    return key
                for prefix in self.ENV_DENYLIST_PREFIXES:
                    if key.upper().startswith(prefix):
                        return key
            elif not is_assign:
                # Non-assignment, non-option = start of COMMAND, stop scanning
                break
        return None

    def _validate_pattern_list(
        self, patterns: list, field_name: str, gate_name: str, source_path: Path
    ) -> None:
        """Validate a list of path patterns (allowed_writes or cache_inputs)."""
        if not isinstance(patterns, list):
            raise GateConfigError(f"Gate '{gate_name}' {field_name} must be list in {source_path}")
        for i, pattern in enumerate(patterns):
            if not isinstance(pattern, str):
                raise GateConfigError(
                    f"Gate '{gate_name}' {field_name}[{i}] must be string in {source_path}"
                )
            if not pattern.strip():
                raise GateConfigError(
                    f"Gate '{gate_name}' {field_name}[{i}] cannot be empty in {source_path}"
                )
            if pattern.startswith("/"):
                raise GateConfigError(
                    f"Gate '{gate_name}' {field_name}[{i}]: absolute paths not allowed "
                    f"(got '{pattern}') in {source_path}"
                )
            if len(pattern) >= 2 and pattern[1] == ":" and pattern[0].isalpha():
                raise GateConfigError(
                    f"Gate '{gate_name}' {field_name}[{i}]: Windows absolute paths "
                    f"not allowed (got '{pattern}') in {source_path}"
                )
            if self._has_path_traversal(pattern):
                raise GateConfigError(
                    f"Gate '{gate_name}' {field_name}[{i}]: path traversal not allowed "
                    f"(got '{pattern}') in {source_path}"
                )
            if "\\" in pattern:
                raise GateConfigError(
                    f"Gate '{gate_name}' {field_name}[{i}]: backslash not allowed "
                    f"(got '{pattern}') in {source_path}. Use forward slashes."
                )

    def _validate_command(
        self, cmd: object, gate_name: str, gate_config: dict, source_path: Path
    ) -> None:
        """Validate gate command configuration."""
        if isinstance(cmd, str):
            raise GateConfigError(
                f"Gate '{gate_name}' command must be a list, not string "
                f"(got '{cmd[:50]}...') in {source_path}. "
                "SECURITY: String commands enable shell injection. "
                'Use: command: ["pytest", "-v"] instead of command: "pytest -v"'
            )
        if not isinstance(cmd, list):
            raise GateConfigError(f"Gate '{gate_name}' command must be a list in {source_path}")
        if len(cmd) == 0:
            raise GateConfigError(f"Gate '{gate_name}' command cannot be empty in {source_path}")
        if not all(isinstance(c, str) for c in cmd):
            raise GateConfigError(
                f"Gate '{gate_name}' command items must all be strings in {source_path}"
            )

        # Security checks
        shell_binary = self._detect_shell_invocation(cmd)
        if shell_binary and not gate_config.get("allow_shell", False):
            raise GateConfigError(
                f"Gate '{gate_name}' uses shell binary '{shell_binary}' but "
                "allow_shell is not set. SECURITY: Shell commands bypass "
                "injection protection. If needed, set allow_shell: true."
            )

        blocked_flag = self._check_env_blocked_flags(cmd)
        if blocked_flag:
            raise GateConfigError(
                f"Gate '{gate_name}' uses blocked env flag '{blocked_flag}'. "
                "SECURITY: -S/--split-string and -i/--ignore-environment are blocked "
                "because they can bypass environment variable security checks."
            )

        denylisted_env = self._check_env_denylist_bypass(cmd)
        if denylisted_env:
            raise GateConfigError(
                f"Gate '{gate_name}' attempts to set denylisted environment variable "
                f"'{denylisted_env}' via env wrapper. SECURITY: Denylisted variables "
                "cannot be overridden, including via wrappers."
            )

    def _validate_env(self, env: object, gate_name: str, source_path: Path) -> None:
        """Validate gate environment configuration."""
        if not isinstance(env, dict):
            raise GateConfigError(f"Gate '{gate_name}' env must be dict in {source_path}")
        for k, v in env.items():
            if not isinstance(k, str):
                raise GateConfigError(
                    f"Gate '{gate_name}' env key must be string, got {type(k).__name__} "
                    f"in {source_path}"
                )
            if not isinstance(v, str):
                raise GateConfigError(
                    f"Gate '{gate_name}' env['{k}'] value must be string, "
                    f"got {type(v).__name__} in {source_path}"
                )
            if not self._ENV_NAME_PATTERN.match(k):
                raise GateConfigError(
                    f"Gate '{gate_name}' env key '{k}' is invalid. "
                    f"Must be a valid POSIX identifier in {source_path}"
                )
            if "\0" in k or "=" in k:
                raise GateConfigError(
                    f"Gate '{gate_name}' env key '{k}' contains invalid characters in {source_path}"
                )

    def _validate_single_gate(self, gate_name: str, gate_config: dict, source_path: Path) -> None:
        """Validate a single gate configuration."""
        if not isinstance(gate_name, str):
            raise GateConfigError(f"Invalid gate name in {source_path}: {gate_name}")

        if not self._GATE_NAME_PATTERN.match(gate_name):
            raise GateConfigError(
                f"Invalid gate name '{gate_name}' in {source_path}. "
                "Gate names must match pattern "
                "[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9] "
                "(no leading/trailing dots, no path separators)"
            )

        if not isinstance(gate_config, dict):
            raise GateConfigError(f"Invalid config for gate '{gate_name}' in {source_path}")

        if "command" not in gate_config:
            raise GateConfigError(f"Gate '{gate_name}' missing required 'command' in {source_path}")

        self._validate_command(gate_config["command"], gate_name, gate_config, source_path)

        if "timeout" in gate_config:
            timeout = gate_config["timeout"]
            if not isinstance(timeout, int):
                raise GateConfigError(f"Gate '{gate_name}' timeout must be int in {source_path}")
            if timeout <= 0:
                raise GateConfigError(f"Gate '{gate_name}' timeout must be > 0 in {source_path}")
            if timeout > 3600:
                raise GateConfigError(
                    f"Gate '{gate_name}' timeout must be <= 3600 (1 hour) in {source_path}"
                )

        if "depends_on" in gate_config:
            deps = gate_config["depends_on"]
            if not isinstance(deps, list):
                raise GateConfigError(
                    f"Gate '{gate_name}' depends_on must be list in {source_path}"
                )
            if not all(isinstance(d, str) for d in deps):
                raise GateConfigError(
                    f"Gate '{gate_name}' depends_on items must all be strings in {source_path}"
                )

        if "severity" in gate_config:
            sev = gate_config["severity"]
            if not isinstance(sev, str) or sev.lower() not in self._VALID_SEVERITIES:
                raise GateConfigError(
                    f"Gate '{gate_name}' severity must be one of {self._VALID_SEVERITIES} "
                    f"in {source_path}"
                )

        for field in self._BOOL_FIELDS:
            if field in gate_config and not isinstance(gate_config[field], bool):
                raise GateConfigError(
                    f"Gate '{gate_name}' {field} must be boolean in {source_path}"
                )

        if "working_dir" in gate_config and not isinstance(gate_config["working_dir"], str):
            raise GateConfigError(f"Gate '{gate_name}' working_dir must be string in {source_path}")

        if "env" in gate_config:
            self._validate_env(gate_config["env"], gate_name, source_path)

        if "allowed_writes" in gate_config:
            self._validate_pattern_list(
                gate_config["allowed_writes"], "allowed_writes", gate_name, source_path
            )

        if "cache_inputs" in gate_config:
            self._validate_pattern_list(
                gate_config["cache_inputs"], "cache_inputs", gate_name, source_path
            )

    def _basic_validate(self, config: dict, source_path: Path) -> None:
        """Validate gate configuration structure and security constraints."""
        if not isinstance(config, dict):
            raise GateConfigError(
                f"Invalid config in {source_path}: expected dict, got {type(config)}"
            )

        if "gates" not in config:
            return

        if not isinstance(config["gates"], dict):
            raise GateConfigError(f"Invalid 'gates' in {source_path}: expected dict")

        for gate_name, gate_config in config["gates"].items():
            self._validate_single_gate(gate_name, gate_config, source_path)

    def load_gates(self) -> dict[str, GateConfig]:
        if self._loaded:
            return self._gates

        merged: dict[str, GateConfig] = {}
        for path in reversed(self._search_paths):
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                if config:
                    self._validate_config(config, path)

                if config and "gates" in config:
                    for name, gate_dict in config["gates"].items():
                        reserved_keys = {"name"}
                        filtered_dict = {
                            k: v for k, v in gate_dict.items() if k not in reserved_keys
                        }
                        if "severity" in filtered_dict:
                            severity_str = filtered_dict["severity"]
                            try:
                                filtered_dict["severity"] = GateSeverity(severity_str.lower())
                            except ValueError:
                                raise GateConfigError(
                                    f"Invalid severity '{severity_str}' for gate '{name}' in {path}. "
                                    "Must be one of: error, warning, info"
                                )

                        merged[name] = GateConfig(name=name, **filtered_dict)
            except (yaml.YAMLError, TypeError, KeyError) as e:
                raise GateConfigError(f"Invalid gate config in {path}: {e}")

        self._gates = merged
        self._loaded = True
        return merged

    def get_gate(self, name: str) -> GateConfig:
        if not self._loaded:
            self.load_gates()
        if name not in self._gates:
            raise GateNotFoundError(
                f"Gate '{name}' not found. Define it in .supervisor/gates.yaml "
                f"or use a built-in gate: {list(self._gates.keys())}"
            )
        return self._gates[name]

    def resolve_execution_order(self, gates: list[str]) -> list[str]:
        """Resolve gate dependencies to execution order (topological sort)."""
        all_gates: set[str] = set()
        to_process: list[str] = list(gates)

        while to_process:
            gate_name = to_process.pop()
            if gate_name in all_gates:
                continue
            all_gates.add(gate_name)
            config = self.get_gate(gate_name)
            for dep in config.depends_on:
                if dep not in all_gates:
                    to_process.append(dep)

        in_degree: dict[str, int] = dict.fromkeys(all_gates, 0)
        dependents: dict[str, list[str]] = {g: [] for g in all_gates}

        for gate_name in all_gates:
            config = self.get_gate(gate_name)
            for dep in config.depends_on:
                dependents[dep].append(gate_name)
                in_degree[gate_name] += 1

        heap: list[str] = [g for g in all_gates if in_degree[g] == 0]
        heapq.heapify(heap)
        result: list[str] = []

        while heap:
            gate_name = heapq.heappop(heap)
            result.append(gate_name)
            for dependent in dependents[gate_name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    heapq.heappush(heap, dependent)

        if len(result) != len(all_gates):
            cycle_gates = [g for g in all_gates if in_degree[g] > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among gates: {sorted(cycle_gates)}. "
                "Check depends_on configuration for these gates."
            )

        return result
