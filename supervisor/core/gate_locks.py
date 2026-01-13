"""File-based locking for gate operations using filelock."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

try:
    from filelock import FileLock
    from filelock import Timeout as FileLockTimeout
except ImportError:
    FileLock = None  # type: ignore[misc, assignment]
    FileLockTimeout = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


class BaseFileLock:
    """Base class for file-based locks using filelock package.

    Uses the robust, well-tested filelock library for cross-platform locking.
    Provides both inter-process (file-based) and intra-process (threading) synchronization.
    """

    LOCK_TIMEOUT: int = 30
    LOCK_FILENAME: str = ".lock"

    def __init__(self, worktree_path: Path, exclusive: bool = True):
        if FileLock is None:
            raise RuntimeError(
                "The 'filelock' package is required for locking. Install with: pip install filelock"
            )
        self.worktree_path = worktree_path.resolve()
        self._filelock: FileLock | None = None
        self.acquired = False
        self._exclusive = exclusive

    def __enter__(self) -> BaseFileLock:
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
            with contextlib.suppress(Exception):
                self._filelock.release()
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
