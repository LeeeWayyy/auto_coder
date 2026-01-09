"""Gate configuration, execution, and verification.

This module is a facade that re-exports all gate-related classes from their
respective submodules for backward compatibility.

Submodules:
- gate_models: Data classes, enums, and exceptions
- gate_loader: GateLoader for loading gate configurations
- gate_executor: GateExecutor for running gates
- gate_locks: File-based locking for concurrent access
"""

from __future__ import annotations

# Re-export all public symbols for backward compatibility
from supervisor.core.gate_executor import GateExecutor
from supervisor.core.gate_loader import GateLoader
from supervisor.core.gate_locks import ArtifactLock, BaseFileLock, WorktreeLock
from supervisor.core.gate_models import (
    PACKAGE_DIR,
    CacheInputLimitExceeded,
    CircularDependencyError,
    ConcurrentGateExecutionError,
    GateConfig,
    GateConfigError,
    GateFailAction,
    GateNotFoundError,
    GateResult,
    GateSeverity,
    GateStatus,
    WorktreeBaseline,
)

__all__ = [
    # Models and data classes
    "PACKAGE_DIR",
    "GateSeverity",
    "GateStatus",
    "GateFailAction",
    "GateConfigError",
    "ConcurrentGateExecutionError",
    "CacheInputLimitExceeded",
    "GateNotFoundError",
    "CircularDependencyError",
    "WorktreeBaseline",
    "GateConfig",
    "GateResult",
    # Loader
    "GateLoader",
    # Executor
    "GateExecutor",
    # Locks
    "BaseFileLock",
    "ArtifactLock",
    "WorktreeLock",
]
