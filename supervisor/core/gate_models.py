"""Gate configuration models, results, and core exceptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.parent


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
