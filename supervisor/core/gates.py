"""Gate configuration models, results, and core exceptions."""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import heapq
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError:
    FileLock = None  # type: ignore[misc, assignment]
    FileLockTimeout = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent.parent

if TYPE_CHECKING:
    from supervisor.core.state import Database
    from supervisor.sandbox.executor import SandboxedExecutor


class GateSeverity(str, Enum):
    """Severity level for a gate."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class GateStatus(str, Enum):
    """Execution status of a gate."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateFailAction(str, Enum):
    """Action to take when a gate fails."""

    BLOCK = "block"
    RETRY_WITH_FEEDBACK = "retry_with_feedback"
    WARN = "warn"


class GateConfigError(Exception):
    """Invalid gate configuration."""

    pass


class ConcurrentGateExecutionError(Exception):
    """Cannot acquire worktree lock for gate execution."""

    pass


class CacheInputLimitExceeded(Exception):
    """Cache inputs exceed file limit and cannot be hashed safely."""

    pass


class GateNotFoundError(Exception):
    """Gate is not defined in any loaded configuration."""

    pass


class CircularDependencyError(Exception):
    """Circular dependency detected between gates."""

    pass


@dataclass
class WorktreeBaseline:
    """Snapshot of tracked and untracked files prior to gate execution.

    Attributes:
        files: Mapping of relative file paths to (mtime, size, content_hash) tuples.
               content_hash may be None for very large files.
        pre_tracked_clean: True if tracked files were clean before gate execution.
    """

    files: dict[str, tuple[int, int, str | None]]
    pre_tracked_clean: bool


@dataclass
class GateConfig:
    """Configuration for a verification gate."""

    name: str
    command: list[str]
    description: str = ""
    timeout: int = 300
    depends_on: list[str] = field(default_factory=list)
    severity: GateSeverity = GateSeverity.ERROR
    env: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None
    parallel_safe: bool = False
    cache: bool = True
    cache_inputs: list[str] = field(default_factory=list)
    force_hash_large_cache_inputs: bool = False
    skip_on_dependency_failure: bool = True
    allowed_writes: list[str] = field(default_factory=list)
    allow_shell: bool = False


@dataclass
class GateResult:
    """Result of gate execution."""

    gate_name: str
    status: GateStatus
    output: str
    duration_seconds: float
    returncode: int | None = None
    timed_out: bool = False
    retry_count: int = 0
    cached: bool = False
    cache_key: str | None = None
    artifact_path: str | None = None
    integrity_violation: bool = False

    OUTPUT_MAX_CHARS = 10000
    EVENT_OUTPUT_MAX_CHARS = 2000

    ARTIFACT_MAX_SIZE = 10 * 1024 * 1024
    ARTIFACT_RETENTION_DAYS = 7
    ARTIFACT_MAX_COUNT_PER_WORKFLOW = 100
    ARTIFACT_MAX_TOTAL_SIZE = 1024 * 1024 * 1024

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASSED

    @property
    def skipped(self) -> bool:
        return self.status == GateStatus.SKIPPED

    @property
    def failed(self) -> bool:
        return self.status == GateStatus.FAILED


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
    SHELL_BINARY_NAMES = frozenset({
        "bash", "sh", "zsh", "fish", "dash", "ksh", "csh", "tcsh",
        "cmd", "powershell", "pwsh",
    })

    # Command wrappers that may wrap shell invocations
    COMMAND_WRAPPERS = frozenset({
        "env", "command", "exec", "xargs", "nice", "nohup", "timeout",
        "stdbuf", "ionice", "chrt", "taskset", "numactl", "time",
        "chronic", "unbuffer", "sudo", "doas", "su", "runuser",
    })

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
    _GATE_NAME_PATTERN = re.compile(
        r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$"
    )
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
        """Check if any blocked env flags are used in the command."""
        for arg in cmd_list:
            if arg == "--":
                break
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
                if arg in self.ENV_FLAGS_WITH_ARGS or any(
                    arg.startswith(f + "=") for f in self.ENV_FLAGS_WITH_ARGS
                ):
                    if "=" not in arg:
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
        """Scan arguments for denylisted environment variable assignments."""
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg == "--":
                break
            if arg.startswith("-"):
                if arg in self.ENV_FLAGS_WITH_ARGS or any(
                    arg.startswith(f + "=") for f in self.ENV_FLAGS_WITH_ARGS
                ):
                    if "=" not in arg:
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
                break
        return None

    def _validate_pattern_list(
        self, patterns: list, field_name: str, gate_name: str, source_path: Path
    ) -> None:
        """Validate a list of path patterns (allowed_writes or cache_inputs)."""
        if not isinstance(patterns, list):
            raise GateConfigError(
                f"Gate '{gate_name}' {field_name} must be list in {source_path}"
            )
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
            raise GateConfigError(
                f"Gate '{gate_name}' command must be a list in {source_path}"
            )
        if len(cmd) == 0:
            raise GateConfigError(
                f"Gate '{gate_name}' command cannot be empty in {source_path}"
            )
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

    def _validate_env(
        self, env: object, gate_name: str, source_path: Path
    ) -> None:
        """Validate gate environment configuration."""
        if not isinstance(env, dict):
            raise GateConfigError(
                f"Gate '{gate_name}' env must be dict in {source_path}"
            )
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
                    f"Gate '{gate_name}' env key '{k}' contains invalid characters "
                    f"in {source_path}"
                )

    def _validate_single_gate(
        self, gate_name: str, gate_config: dict, source_path: Path
    ) -> None:
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
            raise GateConfigError(
                f"Invalid config for gate '{gate_name}' in {source_path}"
            )

        if "command" not in gate_config:
            raise GateConfigError(
                f"Gate '{gate_name}' missing required 'command' in {source_path}"
            )

        self._validate_command(
            gate_config["command"], gate_name, gate_config, source_path
        )

        if "timeout" in gate_config:
            timeout = gate_config["timeout"]
            if not isinstance(timeout, int):
                raise GateConfigError(
                    f"Gate '{gate_name}' timeout must be int in {source_path}"
                )
            if timeout <= 0:
                raise GateConfigError(
                    f"Gate '{gate_name}' timeout must be > 0 in {source_path}"
                )
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

        if "working_dir" in gate_config and not isinstance(
            gate_config["working_dir"], str
        ):
            raise GateConfigError(
                f"Gate '{gate_name}' working_dir must be string in {source_path}"
            )

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
                                filtered_dict["severity"] = GateSeverity(
                                    severity_str.lower()
                                )
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

        in_degree: dict[str, int] = {g: 0 for g in all_gates}
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


class GateExecutor:
    """Execute verification gates in sandboxed containers (basic structure)."""

    # Cache settings
    CACHE_MAX_SIZE = 100
    CACHE_TTL_SECONDS = 3600

    # Cache key computation limits
    CACHE_KEY_TIMEOUT = 30
    CACHE_KEY_MAX_UNTRACKED_FILES = 500
    CACHE_KEY_MAX_FILE_SIZE = 1_000_000  # 1MB
    CACHE_KEY_SKIP_PATTERNS = frozenset(
        [
            "node_modules/",
            "node_modules\\",
            ".venv/",
            ".venv\\",
            "venv/",
            "venv\\",
            "__pycache__/",
            "__pycache__\\",
            "build/",
            "build\\",
            "dist/",
            "dist\\",
            ".git/",
            ".git\\",
            "vendor/",
            "vendor\\",
            "target/",
            "target\\",
            ".next/",
            ".next\\",
        ]
    )

    # Protected env vars that cannot be overridden by gate config (security)
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

    # Git config overrides to disable repo-configured helpers
    GIT_SAFE_CONFIG_BASE = [
        "-c",
        "core.fsmonitor=false",
        "-c",
        "diff.external=",
        "-c",
        "diff.textconv=",
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.clean=",
    ]

    _safe_hooks_dir: Path | None = None

    class GateTimeout(Exception):
        """Cache key computation timed out."""

        pass

    def __init__(
        self,
        executor: "SandboxedExecutor",
        gate_loader: GateLoader,
        db: "Database",
    ) -> None:
        self.executor = executor
        self.gate_loader = gate_loader
        self.db = db
        self._cache: dict[str, tuple[GateResult, float]] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def _get_safe_hooks_dir(cls) -> Path:
        if cls._safe_hooks_dir is None:
            import tempfile

            cls._safe_hooks_dir = Path(
                tempfile.mkdtemp(prefix="supervisor_safe_hooks_")
            )
        return cls._safe_hooks_dir

    def _filter_env(self, env: dict[str, str]) -> dict[str, str]:
        """Filter out protected env vars from gate config."""
        filtered: dict[str, str] = {}
        for key, value in env.items():
            key_upper = key.upper()
            if key_upper in self.ENV_DENYLIST:
                continue
            if any(key_upper.startswith(prefix) for prefix in self.ENV_DENYLIST_PREFIXES):
                continue
            filtered[key] = value
        return filtered

    def _get_safe_git_env(self) -> tuple[dict[str, str], list[str]]:
        """Get sanitized environment and safe config for git subprocess calls."""
        import os
        import sys

        git_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        git_env["GIT_CONFIG_NOSYSTEM"] = "1"
        git_env["GIT_CONFIG_GLOBAL"] = "NUL" if sys.platform == "win32" else "/dev/null"

        safe_hooks_dir = self._get_safe_hooks_dir()
        git_safe_config = list(self.GIT_SAFE_CONFIG_BASE) + [
            "-c",
            f"core.hooksPath={safe_hooks_dir}",
        ]
        return git_env, git_safe_config

    def _evict_cache(self) -> None:
        """Evict oldest cache entry to maintain max size."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]

    def _get_cached_result(self, key: str) -> GateResult | None:
        """Fetch a cached result if present and not expired."""
        with self._cache_lock:
            if key not in self._cache:
                return None
            result, timestamp = self._cache[key]
            if time.time() - timestamp > self.CACHE_TTL_SECONDS:
                del self._cache[key]
                return None
            return result

    def _cache_result(self, key: str, result: GateResult) -> None:
        """Store a result in the cache with FIFO eviction."""
        with self._cache_lock:
            if len(self._cache) >= self.CACHE_MAX_SIZE:
                self._evict_cache()
            self._cache[key] = (result, time.time())

    def _compute_cache_key(self, worktree_path: Path, config: GateConfig) -> str | None:
        """Compute deterministic cache key for gate result."""
        if not config.cache:
            return None

        deadline = time.monotonic() + self.CACHE_KEY_TIMEOUT

        def _check_deadline() -> None:
            if time.monotonic() > deadline:
                raise self.GateTimeout("Cache key computation timed out")

        try:
            _check_deadline()
            git_env, git_safe_config = self._get_safe_git_env()

            def _run_git(args: list[str], timeout: int = 5) -> subprocess.CompletedProcess:
                _check_deadline()
                return subprocess.run(
                    ["git"] + git_safe_config + args,
                    cwd=worktree_path,
                    capture_output=True,
                    timeout=timeout,
                    env=git_env,
                )

            head_result = _run_git(["rev-parse", "HEAD"], timeout=5)
            if head_result.returncode != 0:
                return None
            head_commit = head_result.stdout.decode(
                "utf-8", errors="surrogateescape"
            ).strip()
            _check_deadline()

            staged_result = _run_git(["diff", "--cached"], timeout=10)
            if staged_result.returncode != 0:
                return None
            _check_deadline()

            unstaged_result = _run_git(["diff"], timeout=10)
            if unstaged_result.returncode != 0:
                return None
            _check_deadline()

            untracked_result = _run_git(
                ["ls-files", "--others", "--exclude-standard", "-z"], timeout=5
            )
            if untracked_result.returncode != 0:
                return None
            _check_deadline()

            content_hasher = hashlib.sha256()

            # Include deterministic config inputs first
            filtered_env = self._filter_env(config.env or {})
            sorted_env = sorted(filtered_env.items())
            config_blob = json.dumps(
                {
                    "gate": config.name,
                    "command": config.command,
                    "env": sorted_env,
                    "working_dir": config.working_dir or "",
                    "timeout": config.timeout,
                    # Include integrity rules so tightening allowed_writes invalidates cache
                    "allowed_writes": sorted(config.allowed_writes or []),
                },
                separators=(",", ":"),
                ensure_ascii=True,
            )
            content_hasher.update(config_blob.encode("utf-8"))
            content_hasher.update(b"\0")
            content_hasher.update(head_commit.encode("utf-8"))
            content_hasher.update(b"\0")
            content_hasher.update(staged_result.stdout)
            content_hasher.update(b"\0")
            content_hasher.update(unstaged_result.stdout)
            content_hasher.update(b"\0")

            # Hash untracked file contents
            files_processed = 0
            if untracked_result.stdout:
                for filename_bytes in untracked_result.stdout.split(b"\0"):
                    if not filename_bytes:
                        continue
                    _check_deadline()
                    filename = filename_bytes.decode("utf-8", errors="surrogateescape")
                    normalized = filename.replace("\\", "/")

                    skip = False
                    for pattern in self.CACHE_KEY_SKIP_PATTERNS:
                        pattern_norm = pattern.replace("\\", "/").strip("/")
                        # Check if pattern matches as a path component (gitignore-style)
                        if normalized.startswith(pattern_norm + "/") or pattern_norm in normalized.split("/"):
                            skip = True
                            break
                    if skip:
                        continue

                    files_processed += 1
                    if files_processed > self.CACHE_KEY_MAX_UNTRACKED_FILES:
                        return None

                    file_path = worktree_path / filename
                    if not file_path.is_file():
                        continue

                    # SECURITY: Skip symlinks to avoid escaping worktree
                    if file_path.is_symlink():
                        continue

                    stat = file_path.stat()
                    if stat.st_size > self.CACHE_KEY_MAX_FILE_SIZE:
                        return None

                    file_hasher = hashlib.sha256()
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            file_hasher.update(chunk)
                    file_hash = file_hasher.hexdigest()[:32]
                    content_hasher.update(f"{filename}:{file_hash}\n".encode("utf-8"))

            # Include cache_inputs: additional files/patterns that affect the gate
            if config.cache_inputs:
                _check_deadline()
                cache_input_files: list[str] = []
                for pattern in config.cache_inputs:
                    _check_deadline()
                    # Expand glob patterns
                    pattern_norm = pattern.replace("\\", "/")
                    for root, _dirs, files in os.walk(worktree_path):
                        _check_deadline()
                        rel_root = Path(root).relative_to(worktree_path)
                        for fname in files:
                            rel_path = str(rel_root / fname).replace("\\", "/")
                            if rel_path.startswith("./"):
                                rel_path = rel_path[2:]
                            if fnmatch.fnmatch(rel_path, pattern_norm):
                                cache_input_files.append(rel_path)

                # Sort for determinism and limit count
                cache_input_files = sorted(set(cache_input_files))
                if len(cache_input_files) > self.CACHE_KEY_MAX_UNTRACKED_FILES:
                    if config.force_hash_large_cache_inputs:
                        cache_input_files = cache_input_files[
                            : self.CACHE_KEY_MAX_UNTRACKED_FILES
                        ]
                    else:
                        return None  # Too many files to hash

                for rel_path in cache_input_files:
                    _check_deadline()
                    file_path = worktree_path / rel_path
                    if not file_path.is_file():
                        continue
                    # Skip symlinks for cache key (security)
                    if file_path.is_symlink():
                        continue
                    stat = file_path.stat()
                    if stat.st_size > self.CACHE_KEY_MAX_FILE_SIZE:
                        if not config.force_hash_large_cache_inputs:
                            return None  # File too large
                        # force_hash_large_cache_inputs=True: hash the full file
                        # even though it's large (may be slow but ensures correctness)

                    # Hash the file contents
                    file_hasher = hashlib.sha256()
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            file_hasher.update(chunk)
                    file_hash = file_hasher.hexdigest()[:32]
                    content_hasher.update(
                        f"cache_input:{rel_path}:{file_hash}\n".encode("utf-8")
                    )

            executor_image_id = getattr(self.executor, "image_id", None)
            if executor_image_id:
                content_hasher.update(f"image:{executor_image_id}".encode("utf-8"))

            return content_hasher.hexdigest()[:32]
        except self.GateTimeout:
            return None
        except Exception:
            return None

    # ReDoS protection constants
    _MAX_GLOB_PATTERN_LENGTH = 1024
    _MAX_CONSECUTIVE_WILDCARDS = 10
    _MAX_CHARACTER_CLASS_LENGTH = 64

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _glob_to_regex(pattern: str) -> re.Pattern[str] | None:
        """Convert a glob pattern to a compiled regex (cached).

        Includes ReDoS protection:
        - Pattern length limit
        - Consecutive wildcard limit
        - Character class length limit
        """
        # SECURITY: ReDoS protection - limit pattern length
        if len(pattern) > GateExecutor._MAX_GLOB_PATTERN_LENGTH:
            return None

        # SECURITY: ReDoS protection - detect consecutive wildcards
        wildcard_count = 0
        for char in pattern:
            if char in "*?":
                wildcard_count += 1
                if wildcard_count > GateExecutor._MAX_CONSECUTIVE_WILDCARDS:
                    return None
            else:
                wildcard_count = 0

        regex_parts: list[str] = []
        i = 0
        while i < len(pattern):
            if i + 1 < len(pattern) and pattern[i : i + 2] == "**":
                if i + 2 < len(pattern) and pattern[i + 2] == "/":
                    # **/ matches zero or more directory components
                    regex_parts.append("(?:.*/)?")
                    i += 3
                elif i + 2 == len(pattern):
                    # ** at end of pattern matches everything remaining
                    regex_parts.append(".*")
                    i += 2
                else:
                    # SECURITY: ** not followed by / and not at end is ambiguous.
                    # e.g., "build**bar" could match "build_evil_bar" unexpectedly.
                    # Treat as literal "**" to prevent overly broad matching.
                    regex_parts.append(re.escape("**"))
                    i += 2
            elif pattern[i] == "*":
                regex_parts.append("[^/]*")
                i += 1
            elif pattern[i] == "?":
                regex_parts.append("[^/]")
                i += 1
            elif pattern[i] == "[":
                j = i + 1
                if j < len(pattern) and pattern[j] in "!^":
                    j += 1
                if j < len(pattern) and pattern[j] == "]":
                    j += 1
                while j < len(pattern) and pattern[j] != "]":
                    j += 1
                if j < len(pattern):
                    # SECURITY: ReDoS protection - limit character class length
                    char_class = pattern[i : j + 1]
                    if len(char_class) > GateExecutor._MAX_CHARACTER_CLASS_LENGTH:
                        return None
                    regex_parts.append(char_class)
                    i = j + 1
                else:
                    regex_parts.append(re.escape(pattern[i]))
                    i += 1
            else:
                regex_parts.append(re.escape(pattern[i]))
                i += 1

        regex_str = "^" + "".join(regex_parts) + "$"
        try:
            return re.compile(regex_str)
        except re.error:
            return None

    @staticmethod
    def _path_match(path: str, pattern: str) -> bool:
        """Git-style glob matching for allowed_writes patterns."""

        def _normalize(value: str) -> str:
            value = value.replace("\\", "/")
            while value.startswith("./"):
                value = value[2:]
            while "//" in value:
                value = value.replace("//", "/")
            return value

        path = _normalize(path)
        pattern = _normalize(pattern)

        compiled = GateExecutor._glob_to_regex(pattern)
        if compiled is None:
            return path == pattern
        return bool(compiled.match(path))

    # Baseline capture size limits (prevents memory exhaustion)
    _MAX_BASELINE_FILES = 10000  # Maximum files to track in baseline
    _MAX_BASELINE_FILE_SIZE = 50 * 1024 * 1024  # 50MB - skip hash for larger files
    _MAX_GIT_OUTPUT_SIZE = 10 * 1024 * 1024  # 10MB git status output limit

    def _capture_worktree_baseline(self, worktree_path: Path) -> WorktreeBaseline | None:
        """Capture file states before gate execution.

        Size limits are enforced to prevent memory exhaustion:
        - Max files: _MAX_BASELINE_FILES
        - Max file size for hashing: _MAX_BASELINE_FILE_SIZE
        - Max git output: _MAX_GIT_OUTPUT_SIZE
        """
        try:
            git_env, git_safe_config = self._get_safe_git_env()
            result = subprocess.run(
                ["git"] + git_safe_config + ["status", "--porcelain", "-z", "-uall"],
                cwd=worktree_path,
                capture_output=True,
                timeout=10,
                env=git_env,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="surrogateescape")
                logger.warning(f"Baseline capture failed (git status): {stderr}")
                return None

            # SECURITY: Limit git output size to prevent memory exhaustion
            if len(result.stdout) > self._MAX_GIT_OUTPUT_SIZE:
                logger.warning(
                    f"Baseline capture aborted: git status output too large "
                    f"({len(result.stdout)} > {self._MAX_GIT_OUTPUT_SIZE} bytes)"
                )
                return None

            baseline: dict[str, tuple[int, int, str | None]] = {}
            has_tracked_changes = False
            entries = result.stdout.split(b"\0")
            i = 0
            file_count = 0
            while i < len(entries):
                # SECURITY: Limit number of files to prevent memory exhaustion
                if file_count >= self._MAX_BASELINE_FILES:
                    logger.warning(
                        f"Baseline capture truncated: too many files "
                        f"({file_count} >= {self._MAX_BASELINE_FILES})"
                    )
                    break

                entry = entries[i]
                if not entry or len(entry) < 3:
                    i += 1
                    continue

                entry_str = entry.decode("utf-8", errors="surrogateescape")
                space_idx = entry_str.find(" ")
                if space_idx < 2:
                    i += 1
                    continue

                status_token = entry_str[:space_idx]
                status = status_token[:2]
                path = entry_str[space_idx + 1 :]

                is_rename_copy = status[0] in ("R", "C") or status[1] in ("R", "C")
                if is_rename_copy and (i + 1) < len(entries):
                    new_path_bytes = entries[i + 1]
                    path = (
                        new_path_bytes.decode("utf-8", errors="surrogateescape")
                        if new_path_bytes
                        else path
                    )
                    i += 2
                else:
                    i += 1

                if status.strip() and not status.startswith("??"):
                    has_tracked_changes = True

                file_path = worktree_path / path
                file_count += 1
                try:
                    # SECURITY: Track symlinks separately to detect symlink attacks
                    # Use size=-2 to indicate symlink (vs -1 for non-file/error)
                    if file_path.is_symlink():
                        # Record symlink target for change detection
                        try:
                            symlink_target = str(file_path.readlink())
                            # Hash the symlink target for change detection
                            target_hash = hashlib.sha256(symlink_target.encode()).hexdigest()[:32]
                            baseline[path] = (0, -2, target_hash)  # -2 indicates symlink
                        except OSError:
                            baseline[path] = (0, -2, None)
                    elif file_path.is_file():
                        stat = file_path.stat()
                        mtime = int(stat.st_mtime * 1000)
                        size = stat.st_size

                        # SECURITY: Skip hashing for very large files (use mtime+size only)
                        if size > self._MAX_BASELINE_FILE_SIZE:
                            baseline[path] = (mtime, size, None)
                        else:
                            file_hasher = hashlib.sha256()
                            with open(file_path, "rb") as f:
                                for chunk in iter(lambda: f.read(8192), b""):
                                    file_hasher.update(chunk)
                            content_hash = file_hasher.hexdigest()[:32]
                            baseline[path] = (mtime, size, content_hash)
                    else:
                        baseline[path] = (0, -1, None)
                except OSError:
                    baseline[path] = (0, -1, None)

            return WorktreeBaseline(
                files=baseline, pre_tracked_clean=not has_tracked_changes
            )
        except Exception as e:
            logger.warning(f"Baseline capture failed: {e}")
            return None

    def _verify_worktree_integrity(
        self,
        worktree_path: Path,
        baseline: WorktreeBaseline,
        config: GateConfig,
    ) -> tuple[bool, list[str]]:
        """Check if gate modified files it shouldn't have."""
        violations: list[str] = []
        allowed = config.allowed_writes or []

        def _is_allowed(path: str) -> bool:
            if not allowed:
                return False
            normalized = path.replace("\\", "/")
            return any(self._path_match(normalized, pattern) for pattern in allowed)

        try:
            git_env, git_safe_config = self._get_safe_git_env()

            # SECURITY: Get list of tracked files to prevent allowed_writes from
            # exempting modifications to clean tracked files (P1 security fix)
            tracked_result = subprocess.run(
                ["git"] + git_safe_config + ["ls-files", "-z"],
                cwd=worktree_path,
                capture_output=True,
                timeout=10,
                env=git_env,
            )
            tracked_files: set[str] = set()
            if tracked_result.returncode == 0:
                for entry in tracked_result.stdout.split(b"\0"):
                    if entry:
                        tracked_files.add(entry.decode("utf-8", errors="surrogateescape"))

            result = subprocess.run(
                ["git"] + git_safe_config + ["status", "--porcelain", "-z", "-uall"],
                cwd=worktree_path,
                capture_output=True,
                timeout=10,
                env=git_env,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="surrogateescape")
                logger.warning(f"Integrity check failed (git status): {stderr}")
                return (False, ["<unknown - git status failed>"])

            entries = result.stdout.split(b"\0")
            current_paths: set[str] = set()
            i = 0
            while i < len(entries):
                entry = entries[i]
                if not entry or len(entry) < 3:
                    i += 1
                    continue

                entry_str = entry.decode("utf-8", errors="surrogateescape")
                space_idx = entry_str.find(" ")
                if space_idx < 2:
                    i += 1
                    continue

                status_token = entry_str[:space_idx]
                status = status_token[:2]
                path = entry_str[space_idx + 1 :]

                is_rename_copy = status[0] in ("R", "C") or status[1] in ("R", "C")
                if is_rename_copy and (i + 1) < len(entries):
                    new_path_bytes = entries[i + 1]
                    path = (
                        new_path_bytes.decode("utf-8", errors="surrogateescape")
                        if new_path_bytes
                        else path
                    )
                    i += 2
                else:
                    i += 1

                if status.strip():
                    current_paths.add(path)

            resolved_worktree = worktree_path.resolve()

            for path in sorted(current_paths):
                if path not in baseline.files:
                    # SECURITY (P1): Check if file is tracked. Tracked files that were
                    # clean before gate execution are NOT exempt from integrity checks,
                    # even if they match allowed_writes. Only truly NEW untracked files
                    # can be exempted by allowed_writes patterns.
                    is_tracked = path in tracked_files
                    if not is_tracked and _is_allowed(path):
                        # Truly new untracked file matching allowed_writes - OK
                        continue
                    # Check if it's a symlink escaping worktree
                    file_path = worktree_path / path
                    if file_path.is_symlink():
                        try:
                            resolved_target = file_path.resolve()
                            resolved_target.relative_to(resolved_worktree)
                        except (ValueError, OSError):
                            # SECURITY: New symlink escapes worktree
                            violations.append(f"{path} (new symlink escapes worktree)")
                            continue
                    if is_tracked:
                        violations.append(f"{path} (tracked file modified)")
                    else:
                        violations.append(path)
                    continue

                # SECURITY: Tracked files in baseline are ALWAYS checked.
                # Untracked files in baseline that match allowed_writes are exempt
                # (for idempotency - gates can update their own artifacts).
                is_tracked_file = path in tracked_files
                is_exempt_untracked = not is_tracked_file and _is_allowed(path)

                pre_mtime, pre_size, pre_hash = baseline.files[path]
                file_path = worktree_path / path
                if not file_path.exists():
                    # File was deleted - only allow if it's an exempt untracked file
                    if pre_size not in (-1,) and not is_exempt_untracked:
                        violations.append(path)
                    continue

                # SECURITY: Check if file type changed (file <-> symlink)
                is_symlink = file_path.is_symlink()
                was_symlink = pre_size == -2

                if is_symlink != was_symlink:
                    # Type changed - violation unless exempt untracked
                    if not is_exempt_untracked:
                        violations.append(path)
                    continue

                if is_symlink:
                    # SECURITY: Symlink handling - check stays in worktree FIRST (before hash)
                    # This ordering mitigates TOCTOU: security check happens atomically with resolve()
                    try:
                        # SECURITY: Verify symlink doesn't escape worktree (do this FIRST)
                        resolved_target = file_path.resolve()
                        try:
                            resolved_target.relative_to(resolved_worktree)
                        except ValueError:
                            # Symlink points outside worktree - always a violation (security)
                            violations.append(f"{path} (symlink escapes worktree)")
                            continue

                        # Change detection: hash the symlink target
                        symlink_target = str(file_path.readlink())
                        cur_target_hash = hashlib.sha256(symlink_target.encode()).hexdigest()[:32]

                        # Check if symlink target changed (unless exempt untracked)
                        if pre_hash is not None and cur_target_hash != pre_hash:
                            if not is_exempt_untracked:
                                violations.append(path)
                            continue
                    except OSError:
                        # Symlink is broken (can't resolve/readlink)
                        # Only flag as violation if it wasn't already broken in baseline
                        # pre_hash is None for broken symlinks in baseline (size=-2)
                        if pre_hash is not None and not is_exempt_untracked:
                            # Was valid before, now broken - violation
                            violations.append(path)
                elif file_path.is_file():
                    try:
                        stat = file_path.stat()
                        cur_mtime = int(stat.st_mtime * 1000)
                        cur_size = stat.st_size
                        if cur_mtime != pre_mtime or cur_size != pre_size:
                            if not is_exempt_untracked:
                                violations.append(path)
                            continue

                        if pre_hash is not None:
                            file_hasher = hashlib.sha256()
                            with open(file_path, "rb") as f:
                                for chunk in iter(lambda: f.read(8192), b""):
                                    file_hasher.update(chunk)
                            cur_hash = file_hasher.hexdigest()[:32]
                            if cur_hash != pre_hash and not is_exempt_untracked:
                                violations.append(path)
                    except OSError:
                        if not is_exempt_untracked:
                            violations.append(path)
                else:
                    if pre_size >= 0 and not is_exempt_untracked:  # Was a regular file
                        violations.append(path)

            # Check for deleted files - files in baseline that are no longer present
            # SECURITY: Tracked files are ALWAYS checked. Untracked files matching
            # allowed_writes can be deleted (for idempotency).
            for path, (pre_mtime, pre_size, pre_hash) in baseline.files.items():
                if path in current_paths:
                    continue
                file_path = worktree_path / path
                if not file_path.exists() and pre_size != -1:
                    # Only flag deletion if it's a tracked file or doesn't match allowed_writes
                    is_tracked_file = path in tracked_files
                    is_exempt_untracked = not is_tracked_file and _is_allowed(path)
                    if not is_exempt_untracked:
                        violations.append(path)

            return (len(violations) == 0, violations)
        except Exception as e:
            logger.warning(f"Integrity check failed: {e}")
            return (False, ["<unknown - integrity check failed>"])

    # Common secret patterns to redact from gate output/artifacts.
    SECRET_PATTERNS = [
        (re.compile(r"ghp_[a-zA-Z0-9]{36,}"), "[REDACTED:github_pat]"),
        (re.compile(r"github_pat_[a-zA-Z0-9_]+"), "[REDACTED:github_pat_v2]"),
        (re.compile(r"gho_[a-zA-Z0-9]{36,}"), "[REDACTED:github_oauth]"),
        (re.compile(r"sk-[a-zA-Z0-9]{48,}"), "[REDACTED:openai_key]"),
        (re.compile(r"sk-proj-[a-zA-Z0-9\-_]+"), "[REDACTED:openai_project]"),
        (re.compile(r"sk-ant-[a-zA-Z0-9\-_]+"), "[REDACTED:anthropic_key]"),
        (re.compile(r"AKIA[A-Z0-9]{16}"), "[REDACTED:aws_access_key]"),
        (re.compile(r"xox[baprs]-[a-zA-Z0-9\-]+"), "[REDACTED:slack_token]"),
        (re.compile(r"(?i)(api[_-]?key|apikey)[\"']?\s*[:=]\s*[\"']?[\w\-]+"), r"\1=[REDACTED]"),
        (re.compile(r"(?i)(token|secret|password|passwd|pwd)[\"']?\s*[:=]\s*[\"']?[\w\-]+"), r"\1=[REDACTED]"),
        (re.compile(r"(?i)(authorization|auth)[\"']?\s*[:=]\s*[\"']?bearer\s+[\w\-\.]+"), r"\1=Bearer [REDACTED]"),
    ]

    def _redact_secrets(self, text: str) -> str:
        """Redact common secret patterns from output."""
        for pattern, replacement in self.SECRET_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def _store_artifact(
        self,
        workflow_id: str,
        gate_name: str,
        output: str,
        worktree_path: Path,
    ) -> str | None:
        """Store full gate output as artifact file.

        Path: {worktree}/.supervisor/artifacts/gates/{hashed_workflow_id}/{gate_name}-{timestamp}.log
        """
        try:
            resolved_worktree = worktree_path.resolve()
            safe_workflow_id = hashlib.sha256(workflow_id.encode("utf-8")).hexdigest()[:32]

            # Validate existing path components for symlink safety.
            path_parts = [".supervisor", ".supervisor/artifacts", ".supervisor/artifacts/gates"]
            for part in path_parts:
                check_path = resolved_worktree / part
                if check_path.exists():
                    if check_path.is_symlink():
                        logger.warning(
                            f"SECURITY: {part} is a symlink. Artifact storage disabled."
                        )
                        return None
                    try:
                        check_path.resolve().relative_to(resolved_worktree)
                    except ValueError:
                        logger.warning(
                            f"SECURITY: {part} resolves outside worktree. Artifact storage disabled."
                        )
                        return None

            artifacts_dir = (
                resolved_worktree
                / ".supervisor"
                / "artifacts"
                / "gates"
                / safe_workflow_id
            )

            # Concurrency protection.
            with ArtifactLock(worktree_path, exclusive=True) as lock:
                if not lock.acquired:
                    logger.warning(
                        f"Artifact storage skipped for gate '{gate_name}': lock unavailable."
                    )
                    return None

                artifacts_dir.mkdir(parents=True, exist_ok=True)
                try:
                    artifacts_dir.resolve().relative_to(resolved_worktree)
                except ValueError:
                    logger.warning(
                        "SECURITY: Artifact directory escapes worktree. Storage disabled."
                    )
                    return None

                # Enforce per-workflow artifact count (LRU by mtime).
                try:
                    existing = sorted(
                        artifacts_dir.glob("*.log"),
                        key=lambda p: p.stat().st_mtime,
                    )
                    while len(existing) >= GateResult.ARTIFACT_MAX_COUNT_PER_WORKFLOW:
                        oldest = existing.pop(0)
                        if oldest.is_symlink():
                            continue
                        try:
                            oldest.resolve().relative_to(resolved_worktree)
                        except ValueError:
                            continue
                        try:
                            oldest.unlink()
                        except OSError:
                            pass
                except OSError:
                    pass

                # Redact secrets and enforce size cap.
                redacted = self._redact_secrets(output)
                output_bytes = redacted.encode("utf-8", errors="replace")
                max_bytes = GateResult.ARTIFACT_MAX_SIZE
                if len(output_bytes) > max_bytes:
                    marker = (
                        f"...[artifact truncated to {GateResult.ARTIFACT_MAX_SIZE} bytes]...\n"
                    ).encode("utf-8")
                    tail_budget = max_bytes - len(marker)
                    tail = output_bytes[-tail_budget:] if tail_budget > 0 else b""
                    redacted = marker.decode("utf-8") + tail.decode(
                        "utf-8", errors="replace"
                    )

                # Unique filename to avoid collisions.
                timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
                unique_suffix = uuid.uuid4().hex[:8]
                artifact_filename = f"{gate_name}-{timestamp}-{unique_suffix}.log"
                artifact_file = artifacts_dir / artifact_filename

                # Atomic write via temp file + replace.
                fd, temp_path = tempfile.mkstemp(
                    prefix=".artifact_",
                    suffix=".tmp",
                    dir=str(artifacts_dir),
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(redacted)

                    final_resolved = artifact_file.resolve()
                    try:
                        final_resolved.relative_to(resolved_worktree)
                    except ValueError:
                        os.unlink(temp_path)
                        logger.warning(
                            "SECURITY: Final artifact path escapes worktree. Storage disabled."
                        )
                        return None

                    os.replace(temp_path, str(artifact_file))
                    return str(artifact_file)
                except Exception:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    raise
        except Exception as e:
            logger.warning(f"Failed to store artifact for gate '{gate_name}': {e}")
            return None

    @staticmethod
    def cleanup_artifacts(
        worktree_path: Path,
        retention_days: int = GateResult.ARTIFACT_RETENTION_DAYS,
    ) -> int:
        """Remove expired artifacts and enforce global size cap."""
        from datetime import datetime, timedelta

        resolved_worktree = worktree_path.resolve()
        supervisor_dir = resolved_worktree / ".supervisor"
        if supervisor_dir.exists() and supervisor_dir.is_symlink():
            try:
                supervisor_dir.resolve().relative_to(resolved_worktree)
            except ValueError:
                logger.warning(
                    "SECURITY: .supervisor is a symlink outside worktree. Cleanup disabled."
                )
                return 0

        artifacts_dir = resolved_worktree / ".supervisor" / "artifacts" / "gates"
        if not artifacts_dir.exists():
            return 0
        try:
            artifacts_dir.resolve().relative_to(resolved_worktree)
        except ValueError:
            logger.warning("SECURITY: Artifacts dir escapes worktree. Cleanup disabled.")
            return 0

        deleted = 0
        cutoff = datetime.now() - timedelta(days=retention_days)

        with ArtifactLock(worktree_path, exclusive=True) as lock:
            if not lock.acquired:
                logger.info("Artifact cleanup skipped: lock unavailable.")
                return 0

            for workflow_dir in artifacts_dir.iterdir():
                if workflow_dir.is_symlink() or not workflow_dir.is_dir():
                    continue
                try:
                    workflow_dir.resolve().relative_to(resolved_worktree)
                except ValueError:
                    continue

                for artifact in workflow_dir.iterdir():
                    if artifact.is_symlink():
                        continue
                    try:
                        artifact.resolve().relative_to(resolved_worktree)
                    except ValueError:
                        continue
                    try:
                        mtime = datetime.fromtimestamp(artifact.lstat().st_mtime)
                        if mtime < cutoff:
                            artifact.unlink()
                            deleted += 1
                    except OSError:
                        pass

                try:
                    if not any(workflow_dir.iterdir()):
                        workflow_dir.rmdir()
                except OSError:
                    pass

            # Enforce global size cap (LRU eviction across workflows).
            try:
                all_artifacts: list[tuple[Path, float, int]] = []
                for wf_dir in artifacts_dir.iterdir():
                    if wf_dir.is_symlink() or not wf_dir.is_dir():
                        continue
                    for artifact in wf_dir.iterdir():
                        if artifact.is_symlink():
                            continue
                        try:
                            stat = artifact.lstat()
                            all_artifacts.append((artifact, stat.st_mtime, stat.st_size))
                        except OSError:
                            pass

                total_size = sum(a[2] for a in all_artifacts)
                if total_size > GateResult.ARTIFACT_MAX_TOTAL_SIZE:
                    all_artifacts.sort(key=lambda x: x[1])  # oldest first
                    while total_size > GateResult.ARTIFACT_MAX_TOTAL_SIZE and all_artifacts:
                        artifact_path, _, size = all_artifacts.pop(0)
                        try:
                            artifact_path.resolve().relative_to(resolved_worktree)
                        except ValueError:
                            continue
                        try:
                            artifact_path.unlink()
                            total_size -= size
                            deleted += 1
                        except OSError:
                            pass
            except Exception as e:
                logger.warning(f"Artifact size cleanup failed: {e}")

        return deleted

    def run_gate(
        self,
        config: GateConfig,
        worktree_path: Path,
        workflow_id: str,
        step_id: str,
    ) -> GateResult:
        def _truncate(text: str, max_chars: int) -> str:
            if len(text) <= max_chars:
                return text
            return f"...[truncated {len(text) - max_chars} chars]...\n{text[-max_chars:]}"

        # Resolve and validate working directory.
        try:
            if config.working_dir:
                workdir = (worktree_path / config.working_dir).resolve()
                worktree_resolved = worktree_path.resolve()
                workdir.relative_to(worktree_resolved)
                if not workdir.exists() or not workdir.is_dir():
                    raise GateConfigError(
                        f"working_dir '{config.working_dir}' does not exist or is not a directory"
                    )
            else:
                workdir = worktree_path.resolve()
        except (OSError, ValueError, GateConfigError) as e:
            error_output = f"Invalid gate working_dir: {e}"
            gate_result = GateResult(
                gate_name=config.name,
                status=GateStatus.FAILED,
                output=self._redact_secrets(error_output),
                duration_seconds=0,
                returncode=None,
                timed_out=False,
                retry_count=0,
                cached=False,
                cache_key=None,
                artifact_path=None,
                integrity_violation=False,
            )
            from supervisor.core.state import Event, EventType

            self.db.append_event(
                Event(
                    workflow_id=workflow_id,
                    event_type=EventType.GATE_FAILED,
                    step_id=step_id,
                    payload={
                        "gate": gate_result.gate_name,
                        "output": _truncate(
                            gate_result.output, GateResult.EVENT_OUTPUT_MAX_CHARS
                        ),
                        "duration": gate_result.duration_seconds,
                        "returncode": gate_result.returncode,
                        "timed_out": gate_result.timed_out,
                        "cached": gate_result.cached,
                        "artifact_path": gate_result.artifact_path,
                    },
                )
            )
            return gate_result

        cache_key = self._compute_cache_key(worktree_path, config)
        if cache_key is not None:
            cached = self._get_cached_result(cache_key)
            if cached:
                result = copy(cached)
                result.cached = True
                result.cache_key = cache_key
                # Clear artifact_path - it's workflow-specific and may not exist
                # in the current workflow's artifact directory
                result.artifact_path = None
                from supervisor.core.state import Event, EventType

                self.db.append_event(
                    Event(
                        workflow_id=workflow_id,
                        event_type=EventType.GATE_PASSED
                        if result.status == GateStatus.PASSED
                        else EventType.GATE_FAILED,
                        step_id=step_id,
                        payload={
                            "gate": result.gate_name,
                            "output": _truncate(
                                result.output, GateResult.EVENT_OUTPUT_MAX_CHARS
                            ),
                            "duration": result.duration_seconds,
                            "returncode": result.returncode,
                            "timed_out": result.timed_out,
                            "cached": result.cached,
                            "artifact_path": result.artifact_path,
                        },
                    )
                )
                return result

        baseline = self._capture_worktree_baseline(worktree_path)
        if baseline is None:
            baseline = WorktreeBaseline(files={}, pre_tracked_clean=False)

        filtered_env = self._filter_env(config.env)
        start_time = time.time()
        exec_result = None
        error_output = None
        timed_out = False

        try:
            exec_result = self.executor.run(
                command=config.command,
                workdir=workdir,
                timeout=config.timeout,
                env=filtered_env,
            )
        except Exception as e:
            timed_out = isinstance(e, TimeoutError)
            error_output = f"Gate executor error: {type(e).__name__}: {e}"

        duration = time.time() - start_time

        if exec_result is not None:
            raw_output = f"{exec_result.stdout}\n{exec_result.stderr}".strip()
            returncode = exec_result.returncode
            timed_out = getattr(exec_result, "timed_out", False)
        else:
            raw_output = error_output or "Gate executor error"
            returncode = None

        integrity_ok, violations = self._verify_worktree_integrity(
            worktree_path, baseline, config
        )
        integrity_violation = not integrity_ok
        if violations:
            raw_output += (
                "\n\nWORKTREE INTEGRITY VIOLATION: "
                + ", ".join(sorted(set(violations)))
            )

        redacted_output = self._redact_secrets(raw_output)
        artifact_path = None
        if len(redacted_output) > GateResult.OUTPUT_MAX_CHARS:
            artifact_path = self._store_artifact(
                workflow_id, config.name, raw_output, worktree_path
            )

        output = _truncate(redacted_output, GateResult.OUTPUT_MAX_CHARS)
        status = (
            GateStatus.PASSED
            if exec_result is not None and returncode == 0 and integrity_ok
            else GateStatus.FAILED
        )

        gate_result = GateResult(
            gate_name=config.name,
            status=status,
            output=output,
            duration_seconds=duration,
            returncode=returncode,
            timed_out=timed_out,
            retry_count=0,
            cached=False,
            cache_key=cache_key,
            artifact_path=artifact_path,
            integrity_violation=integrity_violation,
        )

        if gate_result.passed and cache_key is not None:
            self._cache_result(cache_key, gate_result)

        from supervisor.core.state import Event, EventType

        self.db.append_event(
            Event(
                workflow_id=workflow_id,
                event_type=EventType.GATE_PASSED
                if gate_result.status == GateStatus.PASSED
                else EventType.GATE_FAILED,
                step_id=step_id,
                payload={
                    "gate": gate_result.gate_name,
                    "output": _truncate(
                        gate_result.output, GateResult.EVENT_OUTPUT_MAX_CHARS
                    ),
                    "duration": gate_result.duration_seconds,
                    "returncode": gate_result.returncode,
                    "timed_out": gate_result.timed_out,
                    "cached": gate_result.cached,
                    "artifact_path": gate_result.artifact_path,
                },
            )
        )

        return gate_result

    def _get_on_fail_action(
        self,
        config: GateConfig,
        on_fail_overrides: dict[str, GateFailAction] | None = None,
    ) -> GateFailAction:
        """Resolve on-fail action for a gate with overrides and severity defaults."""
        if on_fail_overrides and config.name in on_fail_overrides:
            return on_fail_overrides[config.name]
        if config.severity in (GateSeverity.WARNING, GateSeverity.INFO):
            return GateFailAction.WARN
        return GateFailAction.BLOCK

    def run_gates(
        self,
        gate_names: list[str],
        worktree_path: Path,
        workflow_id: str,
        step_id: str,
        on_fail_overrides: dict[str, GateFailAction] | None = None,
    ) -> list[GateResult]:
        """Run multiple gates in dependency order with skip-on-blocking behavior."""
        def _truncate(text: str, max_chars: int) -> str:
            if len(text) <= max_chars:
                return text
            return f"...[truncated {len(text) - max_chars} chars]...\n{text[-max_chars:]}"

        with WorktreeLock(worktree_path) as lock:
            if not lock.acquired:
                raise ConcurrentGateExecutionError(
                    f"Cannot acquire worktree lock for {worktree_path}."
                )

            ordered = self.gate_loader.resolve_execution_order(gate_names)
            results: list[GateResult] = []
            blocking_failures: set[str] = set()

            for gate_name in ordered:
                config = self.gate_loader.get_gate(gate_name)

                if config.skip_on_dependency_failure:
                    blocking_deps = [
                        dep for dep in config.depends_on if dep in blocking_failures
                    ]
                    if blocking_deps:
                        skip_result = GateResult(
                            gate_name=gate_name,
                            status=GateStatus.SKIPPED,
                            output=(
                                "SKIPPED: Dependencies had blocking failures: "
                                f"{blocking_deps}"
                            ),
                            duration_seconds=0,
                            cached=False,
                            cache_key=None,
                        )
                        results.append(skip_result)

                        from supervisor.core.state import Event, EventType

                        self.db.append_event(
                            Event(
                                workflow_id=workflow_id,
                                event_type=EventType.GATE_SKIPPED,
                                step_id=step_id,
                                payload={
                                    "gate": skip_result.gate_name,
                                    "output": _truncate(
                                        skip_result.output,
                                        GateResult.EVENT_OUTPUT_MAX_CHARS,
                                    ),
                                    "duration": skip_result.duration_seconds,
                                    "returncode": skip_result.returncode,
                                    "timed_out": skip_result.timed_out,
                                    "cached": skip_result.cached,
                                    "artifact_path": skip_result.artifact_path,
                                },
                            )
                        )
                        blocking_failures.add(gate_name)
                        continue

                result = self.run_gate(config, worktree_path, workflow_id, step_id)
                results.append(result)

                if result.integrity_violation:
                    blocking_failures.add(gate_name)
                    return results

                if result.status == GateStatus.FAILED:
                    action = self._get_on_fail_action(
                        config, on_fail_overrides=on_fail_overrides
                    )
                    if action in (
                        GateFailAction.BLOCK,
                        GateFailAction.RETRY_WITH_FEEDBACK,
                    ):
                        blocking_failures.add(gate_name)
                        return results

            return results


class BaseFileLock:
    """Base class for file-based locks using filelock package.

    Uses the robust, well-tested filelock library for cross-platform locking.
    Provides both inter-process (file-based) and intra-process (threading) synchronization.
    """

    LOCK_TIMEOUT: int = 30
    LOCK_FILENAME: str = ".lock"

    def __init__(self, worktree_path: Path):
        if FileLock is None:
            raise RuntimeError(
                "The 'filelock' package is required for locking. "
                "Install with: pip install filelock"
            )
        self.worktree_path = worktree_path.resolve()
        self._filelock: FileLock | None = None
        self.acquired = False

    def __enter__(self) -> "BaseFileLock":
        supervisor_dir = self.worktree_path / ".supervisor"

        # SECURITY: Check for symlink attacks
        if supervisor_dir.exists() and supervisor_dir.is_symlink():
            logger.warning("SECURITY: .supervisor is a symlink; lock disabled.")
            return self

        try:
            supervisor_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning(f"Cannot create .supervisor directory for {self.LOCK_FILENAME}")
            return self

        # SECURITY: Verify directory doesn't escape worktree
        try:
            supervisor_dir.resolve().relative_to(self.worktree_path)
        except ValueError:
            logger.warning("SECURITY: .supervisor escapes worktree; lock disabled.")
            return self

        lock_path = supervisor_dir / self.LOCK_FILENAME

        # SECURITY: Check lock file isn't a symlink
        if lock_path.exists() and lock_path.is_symlink():
            logger.warning(f"SECURITY: {self.LOCK_FILENAME} is a symlink; lock disabled.")
            return self

        try:
            self._filelock = FileLock(str(lock_path), timeout=self.LOCK_TIMEOUT)
            self._filelock.acquire()
            self.acquired = True
        except FileLockTimeout:
            logger.warning(f"{self.__class__.__name__} timeout after {self.LOCK_TIMEOUT}s")
        except Exception as e:
            logger.warning(f"Failed to acquire {self.__class__.__name__}: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._filelock is not None and self.acquired:
            try:
                self._filelock.release()
            except Exception:
                pass
        return False


class ArtifactLock(BaseFileLock):
    """File-based lock for artifact operations.

    Uses filelock for robust cross-platform locking.
    """

    LOCK_TIMEOUT = 30
    LOCK_FILENAME = ".artifact_lock"


class WorktreeLock(BaseFileLock):
    """Lock for exclusive gate execution in a worktree.

    Uses filelock for robust cross-platform locking.
    """

    LOCK_TIMEOUT = 60
    LOCK_FILENAME = ".worktree_lock"
