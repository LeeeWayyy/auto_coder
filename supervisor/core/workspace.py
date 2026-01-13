"""Git worktree isolation for safe step execution.

Each step executes in a clean git worktree. Gates run in the worktree
BEFORE changes are applied to the main tree. Failed steps leave no
trace in the main tree.
"""

import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from filelock import FileLock

    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False
    FileLock = None  # type: ignore


class FileLockRequiredError(Exception):
    """Raised when filelock package is required but not installed."""

    pass


from supervisor.core.models import Step
from supervisor.core.state import Database, Event, EventType
from supervisor.sandbox.executor import SandboxedExecutor


class WorktreeError(Exception):
    """Error during worktree operations."""

    pass


class GateFailedError(Exception):
    """A verification gate failed."""

    def __init__(self, gate_name: str, output: str):
        self.gate_name = gate_name
        self.output = output
        super().__init__(f"Gate '{gate_name}' failed: {output}")


class ApplyError(Exception):
    """Error during apply to main repository.

    CRITICAL: Apply errors are FATAL and should NOT be retried.
    A partial apply may have left the repository in an inconsistent state.
    Retrying would compound the corruption.
    """

    pass


@dataclass
class WorktreeContext:
    """Context for an isolated worktree execution."""

    worktree_path: Path
    step_id: str
    original_head: str


@dataclass
class FileChange:
    """Represents a file change from git status."""

    path: str
    status: str  # 'A' (added), 'M' (modified), 'D' (deleted), 'R' (renamed), 'C' (copied)
    old_path: str | None = None  # For renames/copies


def _reject_symlinks_ignore(src: str, names: list[str]) -> list[str]:
    """Ignore function for shutil.copytree that rejects symlinks.

    Raises WorktreeError if any symlink is found in the directory.
    Returns empty list otherwise (nothing to ignore).
    """
    import os

    for name in names:
        path = os.path.join(src, name)
        if os.path.islink(path):
            raise WorktreeError(f"Symlinks not allowed in worktree changes: {path}")
    return []  # Don't ignore anything, just check for symlinks


def _truncate_output(output: str, max_length: int = 2000) -> str:
    """Truncate output preserving both head and tail.

    Error logs often have the actual failure at the END (e.g., test summaries,
    stack traces), so capturing only the head loses critical information.

    Returns: First ~40% + "..." + Last ~60% if truncation needed.
    """
    if len(output) <= max_length:
        return output

    # Handle very small max_length - just truncate with simple ellipsis
    MIN_FOR_SPLIT = 60  # Minimum length to do head/tail split
    if max_length < MIN_FOR_SPLIT:
        if max_length <= 3:
            return output[:max_length]
        return output[: max_length - 3] + "..."

    # Calculate actual separator length dynamically
    truncated_chars = len(output) - max_length
    separator = f"\n\n... [{truncated_chars} chars truncated] ...\n\n"
    separator_len = len(separator)

    # Ensure we have room for content after the separator
    available = max_length - separator_len
    if available < 20:
        # Not enough room for meaningful split, just truncate
        return output[: max_length - 3] + "..."

    # Prioritize tail (where errors usually are) over head
    head_size = int(available * 0.4)
    tail_size = available - head_size

    head = output[:head_size]
    tail = output[-tail_size:]

    return f"{head}{separator}{tail}"


def _sanitize_step_id(step_id: str) -> str:
    """Sanitize step_id for use as a directory name.

    Prevents path traversal attacks.
    """
    # Replace path separators and null bytes
    sanitized = re.sub(r"[/\\\x00]", "-", step_id)
    # Remove leading dots (hidden files / parent traversal)
    sanitized = sanitized.lstrip(".")
    # Only allow alphanumeric, dash, underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", sanitized)
    # Limit length
    sanitized = sanitized[:64]
    # If empty after sanitization, use UUID
    if not sanitized:
        sanitized = uuid.uuid4().hex[:16]
    return sanitized


class IsolatedWorkspace:
    """Execute steps in isolated git worktrees.

    Workflow:
    1. Create clean worktree from HEAD
    2. Execute worker in worktree
    3. Run gates IN the worktree (not main tree)
    4. ONLY after gates pass: apply changes + update state atomically
    5. On failure: discard worktree, main tree unchanged
    """

    WORKTREES_DIR = ".worktrees"
    # Timeout for git subprocess calls (seconds)
    # Local git operations should complete quickly, but can hang on
    # corrupted repos or busy filesystems
    GIT_TIMEOUT = 30

    def __init__(
        self,
        repo_path: Path,
        executor: SandboxedExecutor,
        db: Database,
    ):
        # SECURITY: Require filelock for concurrency safety
        # Without it, parallel step execution can corrupt the main tree
        if not _FILELOCK_AVAILABLE:
            raise FileLockRequiredError(
                "The 'filelock' package is required for safe concurrent execution. "
                "Install with: pip install filelock"
            )

        self.repo_path = Path(repo_path).absolute()
        self.executor = executor
        self.db = db

        # Ensure .supervisor directory exists for lock file
        # This allows IsolatedWorkspace to be used independently of Database
        (self.repo_path / ".supervisor").mkdir(parents=True, exist_ok=True)

        # File lock for atomic apply operations
        # This prevents race conditions when multiple steps apply changes
        self._apply_lock = FileLock(self.repo_path / ".supervisor" / ".apply.lock")
        self._validate_git_repo()
        self._cleanup_stale_worktrees()

    def _cleanup_stale_worktrees(self) -> None:
        """Clean up stale worktrees from previous runs.

        Called on startup to remove worktrees that are broken or not
        registered with git. Does NOT remove active worktrees to avoid
        interfering with other running instances.

        Uses a global lock to prevent race conditions with other supervisor
        instances running cleanup simultaneously.
        """
        worktrees_dir = self.repo_path / self.WORKTREES_DIR
        if not worktrees_dir.exists():
            return

        # SECURITY: Validate worktrees_dir first
        if worktrees_dir.is_symlink():
            return  # Don't touch symlinks

        try:
            worktrees_dir.resolve().relative_to(self.repo_path.resolve())
        except ValueError:
            return  # Don't touch paths outside repo

        # RACE CONDITION FIX: Use global lock to prevent concurrent cleanup
        # This prevents multiple supervisor instances from racing to delete
        # worktrees that another instance is about to register
        if not _FILELOCK_AVAILABLE:
            return  # Cannot safely clean up without locking

        cleanup_lock_path = self.repo_path / self.WORKTREES_DIR / ".cleanup.lock"
        try:
            cleanup_lock_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return  # Can't create lock directory, skip cleanup

        cleanup_lock = FileLock(str(cleanup_lock_path), timeout=5)
        try:
            cleanup_lock.acquire()
        except Exception:
            return  # Another instance is cleaning up, skip

        try:
            self._do_cleanup_stale_worktrees(worktrees_dir)
        finally:
            cleanup_lock.release()

    def _do_cleanup_stale_worktrees(self, worktrees_dir: Path) -> None:
        """Perform actual cleanup with lock held.

        Separated from _cleanup_stale_worktrees for cleaner lock handling.
        """
        # Get list of active worktrees from git
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return  # Abort cleanup on timeout to prevent data loss

        # CRITICAL: Abort cleanup if git command fails
        # If we can't get the active worktree list, we risk deleting
        # worktrees that are actually in use by other running instances
        if result.returncode != 0:
            return  # Abort cleanup to prevent accidental data loss

        # Parse active worktree paths
        active_worktrees: set[Path] = set()
        for line in result.stdout.split("\n"):
            if line.startswith("worktree "):
                wt_path = Path(line[9:].strip())
                active_worktrees.add(wt_path.resolve())

        # Use git worktree prune to clean up stale worktree metadata
        try:
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=self.repo_path,
                capture_output=True,
                timeout=self.GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            pass  # Best effort - continue with directory cleanup

        # Collect candidate entries for removal first
        candidates: list[Path] = []
        for entry in worktrees_dir.iterdir():
            if entry.is_symlink():
                continue  # Skip symlinks for safety

            if not entry.is_dir():
                continue

            # Skip if this is an active worktree (could be from another instance)
            if entry.resolve() in active_worktrees:
                continue

            # GRACE PERIOD: Skip recently modified directories
            # This prevents deleting worktrees that are currently being created
            # (between mkdir and git worktree add registration)
            try:
                mtime = entry.stat().st_mtime
                age_seconds = time.time() - mtime
                if age_seconds < 300:  # 5 minute grace period
                    continue
            except OSError:
                continue  # Can't stat, skip to be safe

            candidates.append(entry)

        if not candidates:
            return

        # RACE CONDITION FIX: Take a second snapshot once (not per-entry)
        # This prevents O(n²) git subprocess calls while still detecting
        # worktrees that were created after our initial snapshot
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.GIT_TIMEOUT,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("worktree "):
                        wt_path = Path(line[9:].strip()).resolve()
                        active_worktrees.add(wt_path)
        except subprocess.TimeoutExpired:
            return  # Abort cleanup on timeout

        # Now remove candidates that are still not in active worktrees
        for entry in candidates:
            # Re-check against updated snapshot
            if entry.resolve() in active_worktrees:
                continue

            try:
                # This worktree is not registered with git, safe to remove
                result = subprocess.run(
                    ["git", "worktree", "remove", str(entry), "--force"],
                    cwd=self.repo_path,
                    capture_output=True,
                    timeout=self.GIT_TIMEOUT,
                )
                # If git command failed, use _remove_safe as fallback
                # _remove_safe handles symlinks, path traversal, and ancestor checks
                if result.returncode != 0 and (entry.exists() or entry.is_symlink()):
                    self._remove_safe(entry)
            except (subprocess.TimeoutExpired, Exception):
                pass  # Best effort cleanup

    def _validate_git_repo(self) -> None:
        """Ensure we're in a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                timeout=self.GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise WorktreeError(
                f"Git validation timed out after {self.GIT_TIMEOUT}s: {self.repo_path}"
            )
        if result.returncode != 0:
            raise WorktreeError(f"Not a git repository: {self.repo_path}")

    def _get_head_sha(self) -> str:
        """Get current HEAD commit SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise WorktreeError(f"Getting HEAD timed out after {self.GIT_TIMEOUT}s")
        if result.returncode != 0:
            raise WorktreeError(f"Failed to get HEAD: {result.stderr}")
        return result.stdout.strip()

    def _validate_worktrees_dir(self, worktrees_dir: Path) -> None:
        """Validate the worktrees directory is safe to use.

        SECURITY: Prevents symlink attacks where an attacker creates
        .worktrees as a symlink to redirect worktree operations outside repo.
        """
        if worktrees_dir.is_symlink():
            raise WorktreeError(
                f"SECURITY: {worktrees_dir} is a symlink. "
                "This could be an attack attempt. Remove it manually."
            )

        if worktrees_dir.exists():
            # Verify it resolves to within the repo
            try:
                worktrees_dir.resolve().relative_to(self.repo_path.resolve())
            except ValueError:
                raise WorktreeError(
                    f"SECURITY: {worktrees_dir} resolves outside repo. "
                    "This could be an attack attempt."
                )

    def _is_active_worktree(self, worktree_path: Path) -> bool:
        """Check if a worktree is active (registered with git).

        Returns True if the worktree is in git's worktree list,
        indicating it may be in use by another process.
        """
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return False  # Can't verify, treat as inactive

        if result.returncode != 0:
            return False

        resolved_path = worktree_path.resolve()
        for line in result.stdout.split("\n"):
            if line.startswith("worktree "):
                wt_path = Path(line[9:].strip()).resolve()
                if wt_path == resolved_path:
                    return True
        return False

    def _create_worktree(self, step_id: str) -> Path:
        """Create a clean git worktree for a step.

        If a worktree with the same step_id exists and is active (in use),
        creates a new unique worktree with a UUID suffix to avoid collision.
        """
        worktrees_dir = self.repo_path / self.WORKTREES_DIR

        # SECURITY: Validate worktrees_dir before any operations
        self._validate_worktrees_dir(worktrees_dir)

        worktrees_dir.mkdir(exist_ok=True)

        # SECURITY: Re-validate AFTER mkdir to prevent TOCTOU attack
        # An attacker could create a symlink between validation and mkdir
        self._validate_worktrees_dir(worktrees_dir)

        # Sanitize step_id to prevent path traversal
        safe_step_id = _sanitize_step_id(step_id)
        worktree_path = worktrees_dir / safe_step_id

        # SECURITY: Also validate the specific worktree path
        if worktree_path.is_symlink():
            raise WorktreeError(
                f"SECURITY: {worktree_path} is a symlink. This could be an attack attempt."
            )

        # Check if worktree exists
        if worktree_path.exists():
            # Check if it's an active worktree (possibly used by another process)
            if self._is_active_worktree(worktree_path):
                # Active worktree - create a unique name with UUID suffix
                # This prevents deleting a worktree that's in use
                unique_suffix = uuid.uuid4().hex[:8]
                safe_step_id = f"{safe_step_id}-{unique_suffix}"
                worktree_path = worktrees_dir / safe_step_id
            else:
                # Inactive/stale worktree - safe to remove
                self._remove_worktree(worktree_path)

        try:
            result = subprocess.run(
                ["git", "worktree", "add", str(worktree_path), "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise WorktreeError(f"Creating worktree timed out after {self.GIT_TIMEOUT}s")
        if result.returncode != 0:
            raise WorktreeError(f"Failed to create worktree: {result.stderr}")

        return worktree_path

    def _remove_worktree(self, worktree_path: Path) -> None:
        """Remove a git worktree.

        SECURITY: Validates worktree path before removal to prevent TOCTOU
        attack where .worktrees or ancestors are swapped to symlinks.
        """
        # TOCTOU FIX: Validate path BEFORE calling git worktree remove
        # Check 1: Is the path itself a symlink?
        if worktree_path.is_symlink():
            # Symlink - just unlink it, don't call git
            worktree_path.unlink()
            return

        # Check 2: Does resolved path stay under repo?
        try:
            resolved = worktree_path.resolve()
            resolved.relative_to(self.repo_path.resolve())
        except ValueError:
            # Path escapes repo - possible attack, refuse to delete
            import sys

            print(
                f"SECURITY WARNING: Worktree path {worktree_path} resolves outside repo. "
                f"Refusing to delete. Manual cleanup may be required.",
                file=sys.stderr,
            )
            return

        # Check 3: Are all ancestors (up to repo_path) non-symlinks?
        current = worktree_path.parent
        while current != self.repo_path and current != current.parent:
            if current.is_symlink():
                import sys

                print(
                    f"SECURITY WARNING: Ancestor {current} is a symlink. "
                    f"Refusing to delete worktree. Manual cleanup may be required.",
                    file=sys.stderr,
                )
                return
            current = current.parent

        # Path validated - now safe to call git worktree remove
        try:
            subprocess.run(
                ["git", "worktree", "remove", str(worktree_path), "--force"],
                cwd=self.repo_path,
                capture_output=True,
                timeout=self.GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            pass  # Continue with fallback removal
        # Also remove directory if git didn't clean it up
        # SECURITY: Re-validate path before fallback removal
        if worktree_path.exists() or worktree_path.is_symlink():
            # Check 1: Is the path itself a symlink?
            if worktree_path.is_symlink():
                # Symlink - just unlink it, don't follow
                worktree_path.unlink()
                return

            # Check 2: Does resolved path stay under repo?
            try:
                resolved = worktree_path.resolve()
                resolved.relative_to(self.repo_path.resolve())
            except ValueError:
                # Path escapes repo - possible attack, refuse to delete
                import sys

                print(
                    f"SECURITY WARNING: Worktree path {worktree_path} resolves outside repo. "
                    f"Refusing to delete. Manual cleanup may be required.",
                    file=sys.stderr,
                )
                return

            # Check 3: Are all ancestors (up to repo_path) non-symlinks?
            current = worktree_path.parent
            while current != self.repo_path and current != current.parent:
                if current.is_symlink():
                    import sys

                    print(
                        f"SECURITY WARNING: Ancestor {current} is a symlink. "
                        f"Refusing to delete worktree. Manual cleanup may be required.",
                        file=sys.stderr,
                    )
                    return
                current = current.parent

            # Safe to remove
            shutil.rmtree(worktree_path, ignore_errors=True)

    def _get_changed_files(self, worktree_path: Path) -> list[FileChange]:
        """Get list of files changed in worktree using NUL-terminated output.

        Uses git status -z for robust parsing of filenames with spaces/special chars.

        Porcelain v1 format:
        - Standard: "XY path" where XY is 2 chars (e.g., " M", "A ", "??")
        - Renames with -M: "R100 old_path\0new_path\0" where R100 can be 2-4 chars

        The first space after position 1 separates status from path.

        Raises:
            WorktreeError: If git status fails (prevents silent data loss)
            WorktreeError: If unmerged/conflict files are detected
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain=v1", "-z"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=self.GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise WorktreeError(
                f"git status timed out after {self.GIT_TIMEOUT}s in worktree {worktree_path}"
            )
        if result.returncode != 0:
            # Fail loudly instead of returning empty - prevents silent data loss
            raise WorktreeError(f"git status failed in worktree {worktree_path}: {result.stderr}")

        # CONFLICT DETECTION: Unmerged status codes in git porcelain v1
        # These indicate merge conflicts that must be resolved before apply
        UNMERGED_STATUSES = {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}

        changes = []
        # Split by NUL, filter empty strings
        entries = [e for e in result.stdout.split("\x00") if e]

        i = 0
        while i < len(entries):
            entry = entries[i]
            if len(entry) < 4:  # Minimum: "XY " + at least 1 char path
                i += 1
                continue

            # Porcelain v1 format parsing:
            # - Status is at least 2 chars (XY), but can be longer for renames (R100)
            # - Find the space separator dynamically (first space after position 1)
            # - Position 0-1 always exist, space separator may be at 2 or later
            sep_pos = entry.find(" ", 2)
            if sep_pos == -1:
                # Fallback: assume standard 2-char status
                sep_pos = 2

            status = entry[:sep_pos]
            first_path = entry[sep_pos + 1 :]

            # SECURITY: Detect unmerged/conflict statuses and fail fast
            # These indicate merge conflicts that should not be applied
            status_xy = status[:2] if len(status) >= 2 else status
            if status_xy in UNMERGED_STATUSES or "U" in status_xy:
                raise WorktreeError(
                    f"Unmerged/conflicted file detected: {first_path} (status: {status_xy}). "
                    f"Resolve conflicts before applying changes."
                )

            # Check if it's a rename (R) or copy (C)
            # Status can be "R ", "R100", "C ", "C050", etc.
            is_rename = status.startswith("R") or (len(status) >= 2 and status[1] == "R")
            is_copy = status.startswith("C") or (len(status) >= 2 and status[1] == "C")

            if is_rename or is_copy:
                # Rename/Copy format in porcelain v1 -z: "XY old_path\0new_path\0"
                # first_path is old_path, entries[i+1] is new_path
                if i + 1 < len(entries):
                    new_path = entries[i + 1]
                    s_type = "R" if is_rename else "C"
                    # path = destination (new), old_path = source (old)
                    changes.append(FileChange(path=new_path, status=s_type, old_path=first_path))
                    i += 2
                    continue
            elif "?" in status or "A" in status:
                changes.append(FileChange(path=first_path, status="A"))
            elif "D" in status:
                changes.append(FileChange(path=first_path, status="D"))
            else:
                # Modified or other status (includes " M", "M ", "MM", etc.)
                changes.append(FileChange(path=first_path, status="M"))

            i += 1

        return changes

    def _remove_safe(self, path: Path) -> None:
        """Safely remove a file or directory, handling symlinks correctly.

        IMPORTANT: Must check is_symlink() BEFORE is_dir() because
        is_dir() returns True for symlinks pointing to directories,
        but shutil.rmtree() crashes on symlinks.

        SECURITY: Handles symlinks first (without resolving) to prevent DoS
        where a malicious symlink pointing outside repo blocks cleanup.
        For non-symlink paths, re-validates that path remains under repo.
        """
        if not path.exists() and not path.is_symlink():
            return

        # SECURITY: Handle symlinks FIRST without resolving
        # This prevents DoS where symlink pointing outside repo blocks cleanup
        if path.is_symlink():
            # Symlinks (even to directories) should be unlinked, not rmtree'd
            path.unlink()
            return

        # For non-symlinks: Re-validate path is still under repo right before deletion
        # This prevents TOCTOU attack where parent is swapped to symlink
        try:
            resolved = path.resolve()
            resolved.relative_to(self.repo_path.resolve())
        except ValueError:
            raise WorktreeError(
                f"SECURITY: Path escapes repo at deletion time: {path}. "
                "Possible TOCTOU attack - parent may have been swapped to symlink."
            )

        # Also verify all parents are not symlinks
        current = path.parent
        while current != self.repo_path and current != current.parent:
            if current.is_symlink():
                raise WorktreeError(
                    f"SECURITY: Parent became symlink at deletion time: {current}. "
                    "Possible TOCTOU attack."
                )
            current = current.parent

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def _validate_path(self, path: str) -> None:
        """Validate a path is safe to use.

        SECURITY: Prevents path traversal attacks via malformed paths.
        """
        # Reject absolute paths
        if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
            raise WorktreeError(f"SECURITY: Absolute path not allowed: {path}")

        # Reject path traversal
        if ".." in path.split("/") or ".." in path.split("\\"):
            raise WorktreeError(f"SECURITY: Path traversal not allowed: {path}")

        # Validate resolved path stays within repo
        resolved = (self.repo_path / path).resolve()
        try:
            resolved.relative_to(self.repo_path.resolve())
        except ValueError:
            raise WorktreeError(f"SECURITY: Path escapes repo: {path}")

    def _validate_no_symlinks_in_source(
        self, worktree_path: Path, changes: list[FileChange]
    ) -> None:
        """Pre-validate all source files for symlinks BEFORE any modifications.

        SECURITY: This ensures atomicity - if any symlink is found, we abort
        before making any changes to the main tree.
        """
        for change in changes:
            if change.status == "D":
                continue  # Deletions don't have source files

            src = worktree_path / change.path
            if not src.exists():
                continue

            if src.is_symlink():
                raise WorktreeError(f"SECURITY: Symlink in worktree not allowed: {change.path}")

            # Check for symlinks inside directories
            if src.is_dir():
                for root, dirs, files in os.walk(src):
                    root_path = Path(root)
                    for name in dirs + files:
                        if (root_path / name).is_symlink():
                            raise WorktreeError(
                                f"SECURITY: Symlink in worktree not allowed: {root_path / name}"
                            )

    def _verify_head_unchanged(self, original_head: str) -> None:
        """Verify HEAD hasn't changed since worktree was created.

        CONFLICT DETECTION: If HEAD has advanced (e.g., another step completed),
        applying changes could silently overwrite concurrent work.

        Args:
            original_head: The HEAD SHA when the worktree was created

        Raises:
            WorktreeError: If HEAD has changed, indicating a conflict
        """
        current_head = self._get_head_sha()
        if current_head != original_head:
            raise WorktreeError(
                f"HEAD has changed during step execution. "
                f"Original: {original_head[:8]}, Current: {current_head[:8]}. "
                f"Another step may have completed concurrently. "
                f"Retry the step to work from the updated base."
            )

    def _apply_changes(self, worktree_path: Path, original_head: str | None = None) -> list[str]:
        """Apply changes from worktree to main tree.

        Handles renames, copies, deletions, and directory creation using a
        two-pass approach to safely handle rename chains and swaps.

        SECURITY: Pre-validates all changes before any modifications.
        SECURITY: Rejects symlinks to prevent host file exfiltration.
        SECURITY: Validates paths to prevent traversal attacks.
        CONFLICT: If original_head provided, verifies HEAD unchanged.
        """
        # CONFLICT DETECTION: Verify HEAD unchanged before applying
        if original_head is not None:
            self._verify_head_unchanged(original_head)
        changes = self._get_changed_files(worktree_path)
        applied_files = []

        # SECURITY: Validate all paths before any modifications
        for change in changes:
            self._validate_path(change.path)
            if change.old_path:
                self._validate_path(change.old_path)

        # SECURITY: Pre-validate all sources for symlinks BEFORE modifications
        # This ensures atomicity - we abort before making any changes
        self._validate_no_symlinks_in_source(worktree_path, changes)

        # INTEGRITY: Pre-validate all source files exist BEFORE any modifications
        # This prevents partial updates where we delete files but then can't copy
        for change in changes:
            if change.status == "D":
                continue  # Deletions don't need source files
            src = worktree_path / change.path
            if not src.exists():
                raise WorktreeError(
                    f"Source file missing in worktree: {change.path}. "
                    f"Cannot apply changes - worktree may be corrupted."
                )

        # Pass 1: Deletions
        # Remove all paths that will be deleted or moved (source of rename)
        # This ensures that Pass 2 won't accidentally delete files it just wrote
        for change in changes:
            if change.status == "D":
                self._remove_safe(self.repo_path / change.path)
            elif change.status == "R" and change.old_path:
                self._remove_safe(self.repo_path / change.old_path)

        # Pass 2: Writes
        # Apply all additions, modifications, copies, and rename destinations
        for change in changes:
            src = worktree_path / change.path
            dst = self.repo_path / change.path

            if change.status == "D":
                applied_files.append(f"-{change.path}")
            elif change.status == "R" and change.old_path:
                if src.exists():
                    self._copy_safe(src, dst, change.path)
                applied_files.append(f"{change.old_path} -> {change.path}")
            elif change.status == "C" and change.old_path:
                if src.exists():
                    self._copy_safe(src, dst, change.path)
                applied_files.append(f"{change.old_path} +> {change.path}")
            else:
                # Add or Modify
                if src.exists():
                    self._copy_safe(src, dst, change.path)
                    applied_files.append(change.path)

        return applied_files

    def _validate_parent_path(self, dst: Path) -> None:
        """Validate all parent directories of dst are not symlinks.

        SECURITY: Prevents writing outside repo via symlinked parent directories.
        """
        # Walk up from dst.parent to repo_path, checking for symlinks
        current = dst.parent
        repo_resolved = self.repo_path.resolve()

        while current != self.repo_path and current != current.parent:
            if current.is_symlink():
                raise WorktreeError(f"SECURITY: Parent directory is symlink: {current}")
            # Verify we're still within repo
            try:
                current.resolve().relative_to(repo_resolved)
            except ValueError:
                raise WorktreeError(f"SECURITY: Path escapes repo via parent: {current}")
            current = current.parent

    def _copy_safe(self, src: Path, dst: Path, display_path: str) -> None:
        """Copy file/directory from src to dst, rejecting symlinks.

        SECURITY: Rejects any symlinks from source to prevent host file exfiltration.
        SECURITY: Validates parent directories are not symlinks.
        ATOMICITY: Uses staged copy to prevent data loss on copy failure.
                   Original file is preserved until new content is fully written.
        """
        # SECURITY: Reject top-level symlinks in source
        if src.is_symlink():
            raise WorktreeError(f"Symlinks not allowed in worktree changes: {display_path}")

        # SECURITY: Validate parent directories before any write operations
        # This prevents writing outside repo via symlinked parent dirs
        self._validate_parent_path(dst)

        dst.parent.mkdir(parents=True, exist_ok=True)

        # ATOMICITY: Use staged copy to temp location first
        # If copy fails, original dst is preserved (no data loss)
        temp_dst = dst.with_suffix(dst.suffix + f".tmp.{uuid.uuid4().hex[:8]}")

        try:
            if src.is_dir():
                # Use ignore function to reject any symlinks inside the directory
                # Note: symlinks=True would copy links as links (safe), but we
                # want to reject them entirely as they shouldn't exist in worktree
                shutil.copytree(
                    src,
                    temp_dst,
                    dirs_exist_ok=False,
                    ignore=_reject_symlinks_ignore,
                )
            else:
                shutil.copy2(src, temp_dst)

            # SECURITY: Remove existing destination (may be symlink to outside repo)
            # Only done AFTER successful copy to temp, ensuring we have valid new content
            self._remove_safe(dst)

            # Atomic move from temp to destination
            # os.replace is atomic on same filesystem (which it is, same repo)
            if src.is_dir():
                # os.replace doesn't work for directories, use rename
                temp_dst.rename(dst)
            else:
                os.replace(temp_dst, dst)

        except Exception:
            # Cleanup temp file/dir on any failure
            self._remove_safe(temp_dst)
            raise

    def _run_gate(
        self,
        gate_name: str,
        worktree_path: Path,
    ) -> tuple[bool, str]:
        """Run a verification gate in the worktree.

        SECURITY: Uses "--" separator to prevent gate_name from being
        interpreted as make options (e.g., "-j8" or "--file=malicious").
        """
        # Gate commands are run inside sandbox
        # SECURITY: "--" prevents gate_name from being parsed as options
        result = self.executor.run(
            command=["make", "--", gate_name],
            workdir=worktree_path,
        )

        passed = result.returncode == 0
        output = result.stdout if passed else f"{result.stdout}\n{result.stderr}"

        return passed, output

    def _get_worktree_head(self, worktree_path: Path) -> str:
        """Get the HEAD commit SHA of a worktree.

        Uses the worktree's actual HEAD, not the main repo's HEAD.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=self.GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise WorktreeError(f"Getting worktree HEAD timed out after {self.GIT_TIMEOUT}s")
        if result.returncode != 0:
            raise WorktreeError(f"Failed to get worktree HEAD: {result.stderr}")
        return result.stdout.strip()

    @contextmanager
    def isolated_execution(self, step_id: str) -> Generator[WorktreeContext, None, None]:
        """Context manager for isolated step execution.

        Usage:
            with workspace.isolated_execution(step.id) as ctx:
                # Execute worker in ctx.worktree_path
                # Run gates
                # If all passes, changes are applied on context exit

        NOTE: original_head is captured AFTER worktree creation to prevent
        false conflict detection if HEAD advances between capture and create.
        """
        worktree_path = self._create_worktree(step_id)

        # Capture HEAD AFTER worktree creation to get the actual base commit
        # This prevents false conflicts if HEAD advances between this call and create
        head_sha = self._get_worktree_head(worktree_path)

        ctx = WorktreeContext(
            worktree_path=worktree_path,
            step_id=step_id,
            original_head=head_sha,
        )

        try:
            yield ctx
        finally:
            self._remove_worktree(worktree_path)

    def execute_step(
        self,
        step: Step,
        worker_fn: Callable[[Step, Path], dict[str, Any]],
        gates: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute a step with full isolation and gate verification.

        Args:
            step: The step to execute
            worker_fn: Function that takes (step, worktree_path) and returns output dict
            gates: List of gate names to run (e.g., ["test", "lint"])

        Returns:
            Worker output dictionary

        Raises:
            GateFailedError: If any gate fails
            WorktreeError: If worktree operations fail
        """
        # Use step.gates if gates is None, but allow explicitly empty list []
        gates = step.gates if gates is None else gates

        # Record step started
        self.db.append_event(
            Event(
                workflow_id=step.workflow_id,
                event_type=EventType.STEP_STARTED,
                step_id=step.id,
                role=step.role,
                payload={"gates": gates, "context": step.context},
            )
        )

        # Create worktree - log STEP_FAILED if creation fails
        try:
            worktree_path = self._create_worktree(step.id)
        except Exception as e:
            # Event sourcing: append_event updates projection via _update_projections
            self.db.append_event(
                Event(
                    workflow_id=step.workflow_id,
                    event_type=EventType.STEP_FAILED,
                    step_id=step.id,
                    payload={"error": f"Worktree creation failed: {e}"},
                )
            )
            raise

        # Capture HEAD AFTER worktree creation to get the actual base commit
        # This prevents false conflicts if HEAD advances between event and worktree creation
        original_head = self._get_worktree_head(worktree_path)

        try:
            # 1. Execute worker in isolation
            output = worker_fn(step, worktree_path)

            # 2. Run gates IN THE WORKTREE (not main tree)
            for gate_name in gates:
                passed, gate_output = self._run_gate(gate_name, worktree_path)

                if passed:
                    self.db.append_event(
                        Event(
                            workflow_id=step.workflow_id,
                            event_type=EventType.GATE_PASSED,
                            step_id=step.id,
                            payload={
                                "gate": gate_name,
                                "output": _truncate_output(gate_output, 1000),
                            },
                        )
                    )
                else:
                    # Emit gate failed AND step failed events
                    self.db.append_event(
                        Event(
                            workflow_id=step.workflow_id,
                            event_type=EventType.GATE_FAILED,
                            step_id=step.id,
                            payload={
                                "gate": gate_name,
                                "output": _truncate_output(gate_output, 1000),
                            },
                        )
                    )
                    # Event sourcing: append_event updates projection via _update_projections
                    self.db.append_event(
                        Event(
                            workflow_id=step.workflow_id,
                            event_type=EventType.STEP_FAILED,
                            step_id=step.id,
                            payload={
                                "error": f"Gate '{gate_name}' failed",
                                "gate_output": _truncate_output(gate_output, 2000),
                            },
                        )
                    )
                    raise GateFailedError(gate_name, gate_output)

            # 3. ONLY after gates pass: apply to main tree + update state
            # Use file lock to prevent race conditions with parallel step execution
            # Pass original_head for conflict detection

            # CRASH RECOVERY: Record applying event BEFORE modifying repo
            # On recovery, if STEP_APPLYING exists without STEP_COMPLETED/FAILED,
            # it indicates a crash during apply that needs investigation
            self.db.append_event(
                Event(
                    workflow_id=step.workflow_id,
                    event_type=EventType.STEP_APPLYING,
                    step_id=step.id,
                    payload={"original_head": original_head},
                )
            )

            # CRITICAL: Apply errors are FATAL - do not retry
            # A partial apply may corrupt the repository
            try:
                with self._apply_lock:
                    changed_files = self._apply_changes(worktree_path, original_head)
            except Exception as apply_err:
                # Wrap in ApplyError to signal non-retriable failure
                raise ApplyError(
                    f"Apply failed (repository may be in inconsistent state): {apply_err}"
                ) from apply_err

            # Event sourcing: append_event updates projection atomically
            # If this fails, repo is modified but DB is stale - we log the inconsistent state
            try:
                self.db.append_event(
                    Event(
                        workflow_id=step.workflow_id,
                        event_type=EventType.STEP_COMPLETED,
                        step_id=step.id,
                        payload={"output": output, "files_changed": changed_files},
                    )
                )
            except Exception as db_error:
                # CRITICAL: Git repo was modified but DB update failed
                # Log this inconsistent state so it can be detected and remediated
                import sys

                print(
                    f"CRITICAL: Step {step.id} applied changes to repo but DB update failed: {db_error}. "
                    f"Changed files: {changed_files}. Manual remediation may be required.",
                    file=sys.stderr,
                )
                # Try to record the failure in DB (best effort)
                try:
                    self.db.append_event(
                        Event(
                            workflow_id=step.workflow_id,
                            event_type=EventType.STEP_FAILED,
                            step_id=step.id,
                            payload={
                                "error": f"DB update failed after apply: {db_error}",
                                "files_changed": changed_files,
                                "inconsistent_state": True,
                            },
                        )
                    )
                except Exception:
                    pass  # Best effort - original error is more important
                raise

            return output

        except GateFailedError:
            # Already logged above
            raise

        except Exception as e:
            # Worker failed - event sourcing updates projection via _update_projections
            self.db.append_event(
                Event(
                    workflow_id=step.workflow_id,
                    event_type=EventType.STEP_FAILED,
                    step_id=step.id,
                    payload={"error": str(e)},
                )
            )
            raise

        finally:
            self._remove_worktree(worktree_path)
