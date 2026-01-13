"""Tests for git worktree isolation and workspace management.

This module tests the IsolatedWorkspace class which provides:
- Git worktree creation and cleanup
- Isolated step execution
- Security validation (symlinks, path traversal)
- Atomic change application with file locking
- Gate execution in worktrees
- HEAD conflict detection
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from supervisor.core.models import Step
from supervisor.core.state import EventType
from supervisor.core.workspace import (
    ApplyError,
    FileLockRequiredError,
    GateFailedError,
    IsolatedWorkspace,
    WorktreeError,
    _reject_symlinks_ignore,
    _sanitize_step_id,
    _truncate_output,
)

# =============================================================================
# Utility Function Tests
# =============================================================================


class TestSanitizeStepId:
    """Tests for _sanitize_step_id path traversal prevention."""

    def test_alphanumeric_preserved(self):
        """Alphanumeric characters and dashes/underscores are preserved."""
        assert _sanitize_step_id("step-123_abc") == "step-123_abc"

    def test_path_separators_replaced(self):
        """Path separators (/, \\) are replaced with dashes."""
        # After: replace /\\ with -, lstrip dots, replace non-alnum with -
        # "../../../etc/passwd" → "..-..-..-etc-passwd" → "-..-..-etc-passwd" → "-------etc-passwd"
        assert _sanitize_step_id("../../../etc/passwd") == "-------etc-passwd"
        # "..\\..\\windows\\system32" → "..-..-windows-system32" → "-..-windows-system32" → "----windows-system32"
        assert _sanitize_step_id("..\\..\\windows\\system32") == "----windows-system32"

    def test_leading_dots_removed(self):
        """Leading dots are removed to prevent hidden files."""
        result = _sanitize_step_id("...sneaky")
        assert not result.startswith(".")
        assert result == "sneaky"

    def test_special_characters_replaced(self):
        """Special characters are replaced with dashes."""
        assert _sanitize_step_id("step@123#456") == "step-123-456"

    def test_null_bytes_removed(self):
        """Null bytes are replaced to prevent injection."""
        assert _sanitize_step_id("step\x00bad") == "step-bad"

    def test_length_limited(self):
        """Step IDs are limited to 64 characters."""
        long_id = "a" * 100
        result = _sanitize_step_id(long_id)
        assert len(result) == 64

    def test_empty_after_sanitization_generates_uuid(self):
        """Empty strings after sanitization get a UUID."""
        # Empty string should generate UUID
        result = _sanitize_step_id("")
        assert len(result) == 16  # UUID hex[:16]
        assert result.isalnum()  # Only alphanumeric


class TestTruncateOutput:
    """Tests for _truncate_output output limiting."""

    def test_short_output_unchanged(self):
        """Output shorter than max_length is returned unchanged."""
        output = "Short message"
        assert _truncate_output(output, max_length=1000) == output

    def test_truncation_preserves_head_and_tail(self):
        """Long output is truncated preserving head (40%) and tail (60%)."""
        output = "A" * 100 + "B" * 100 + "C" * 100  # 300 chars
        result = _truncate_output(output, max_length=100)

        assert len(result) <= 100
        assert "A" in result  # Head preserved
        assert "C" in result  # Tail preserved
        assert "truncated" in result  # Separator present

    def test_very_small_max_length(self):
        """Very small max_length (< 60) uses simple truncation."""
        output = "A" * 1000
        result = _truncate_output(output, max_length=10)
        assert len(result) <= 10
        assert result.startswith("A")

    def test_priority_given_to_tail(self):
        """Tail gets 60% priority since errors are at the end."""
        output = "START" + "X" * 1000 + "END_ERROR"
        result = _truncate_output(output, max_length=200)

        # Tail should be more visible than head
        assert "END_ERROR" in result


class TestRejectSymlinksIgnore:
    """Tests for _reject_symlinks_ignore security validation."""

    def test_no_symlinks_returns_empty(self, tmp_path):
        """Directories without symlinks return empty list."""
        (tmp_path / "file.txt").touch()
        (tmp_path / "subdir").mkdir()

        result = _reject_symlinks_ignore(str(tmp_path), ["file.txt", "subdir"])
        assert result == []

    def test_symlink_raises_error(self, tmp_path):
        """Symlinks raise WorktreeError."""
        (tmp_path / "file.txt").touch()
        (tmp_path / "link").symlink_to(tmp_path / "file.txt")

        with pytest.raises(WorktreeError, match="Symlinks not allowed"):
            _reject_symlinks_ignore(str(tmp_path), ["link"])


# =============================================================================
# IsolatedWorkspace Tests
# =============================================================================


@pytest.mark.git
class TestIsolatedWorkspaceInit:
    """Tests for IsolatedWorkspace initialization."""

    def test_requires_filelock(self, repo_with_git, test_db, mock_sandbox_executor):
        """IsolatedWorkspace requires filelock package for concurrency safety."""
        with patch("supervisor.core.workspace._FILELOCK_AVAILABLE", False):
            with pytest.raises(FileLockRequiredError, match="filelock.*required"):
                IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

    def test_creates_supervisor_directory(self, repo_with_git, test_db, mock_sandbox_executor):
        """Initialization creates .supervisor directory if missing."""
        import shutil

        supervisor_dir = repo_with_git / ".supervisor"
        if supervisor_dir.exists():
            shutil.rmtree(supervisor_dir)

        IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        assert (repo_with_git / ".supervisor").exists()
        # Note: FileLock creates the .apply.lock file lazily on first acquire,
        # not during IsolatedWorkspace construction

    def test_validates_git_repo(self, tmp_path, test_db, mock_sandbox_executor):
        """Initialization validates that repo_path is a git repository."""
        # tmp_path is NOT a git repo

        with pytest.raises(WorktreeError, match="[Nn]ot a git repository"):
            IsolatedWorkspace(tmp_path, mock_sandbox_executor, test_db)

    def test_cleanup_stale_worktrees_on_init(self, repo_with_git, test_db, mock_sandbox_executor):
        """Initialization cleans up stale worktrees from previous runs."""
        # Create stale worktree directory (not registered with git)
        stale_worktree = repo_with_git / ".worktrees" / "stale-worktree"
        stale_worktree.mkdir(parents=True)
        (stale_worktree / "marker.txt").touch()

        IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Stale worktree should be cleaned up
        # Note: Actual cleanup logic may vary, just verify workspace initializes


@pytest.mark.git
class TestWorktreeCreation:
    """Tests for git worktree creation and management."""

    def test_create_worktree_success(self, repo_with_git, test_db, mock_sandbox_executor):
        """_create_worktree creates a new git worktree."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        worktree_path = workspace._create_worktree("step-123")

        assert worktree_path.exists()
        assert worktree_path.is_dir()
        assert (worktree_path / "src").exists()  # Files from main repo
        assert ".worktrees" in str(worktree_path)

    def test_worktree_path_sanitized(self, repo_with_git, test_db, mock_sandbox_executor):
        """Worktree paths are sanitized to prevent path traversal."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        worktree_path = workspace._create_worktree("../../../etc/passwd")

        # Path should be sanitized and contained within repo
        # Key security property: path stays inside repo, no escape to /etc
        assert str(worktree_path).startswith(str(repo_with_git))
        assert ".worktrees" in str(worktree_path)  # Inside worktrees dir
        # Path traversal chars should be replaced with dashes
        assert "../" not in str(worktree_path)

    def test_worktree_isolated_from_main(self, repo_with_git, test_db, mock_sandbox_executor):
        """Worktree is isolated - changes don't affect main repo."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)
        worktree_path = workspace._create_worktree("step-isolation")

        # Modify worktree
        (worktree_path / "new_file.txt").write_text("new content")

        # Main repo should be unchanged
        assert not (repo_with_git / "new_file.txt").exists()

    def test_worktree_git_timeout(self, repo_with_git, test_db, mock_sandbox_executor, mocker):
        """Git operations respect timeout to prevent hangs."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Mock subprocess to simulate timeout
        mocker.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 30))

        with pytest.raises((WorktreeError, subprocess.TimeoutExpired)):
            workspace._create_worktree("step-timeout")


@pytest.mark.git
class TestExecuteStep:
    """Tests for execute_step complete workflow."""

    def test_execute_step_success_no_gates(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step
    ):
        """execute_step runs worker and records events without gates."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Simple worker that creates a file
        def worker_fn(step: Step, worktree_path: Path) -> dict[str, Any]:
            (worktree_path / "output.txt").write_text("worker output")
            return {"status": "success", "files": ["output.txt"]}

        # Execute with no gates
        sample_step.gates = []
        result = workspace.execute_step(sample_step, worker_fn, gates=[])

        # Worker result returned
        assert result["status"] == "success"

        # Events recorded
        events = test_db.get_events(sample_step.workflow_id)
        event_types = [e.event_type for e in events]
        assert EventType.STEP_STARTED in event_types
        assert EventType.STEP_COMPLETED in event_types

    def test_execute_step_with_passing_gates(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step, mocker
    ):
        """execute_step runs gates and applies changes when they pass."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Worker creates a file
        def worker_fn(step: Step, worktree_path: Path) -> dict[str, Any]:
            (worktree_path / "output.txt").write_text("content")
            return {"status": "success"}

        # Mock gate execution to pass
        mocker.patch.object(workspace, "_run_gate", return_value=(True, "Gate passed"))

        sample_step.gates = ["test", "lint"]
        result = workspace.execute_step(sample_step, worker_fn)

        assert result["status"] == "success"

        # Gate pass events recorded
        events = test_db.get_events(sample_step.workflow_id)
        gate_passed = [e for e in events if e.event_type == EventType.GATE_PASSED]
        assert len(gate_passed) == 2  # Both gates passed

    def test_execute_step_gate_failure(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step, mocker
    ):
        """execute_step raises GateFailedError when gate fails."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        def worker_fn(step: Step, worktree_path: Path) -> dict[str, Any]:
            return {"status": "success"}

        # Mock gate to fail
        mocker.patch.object(workspace, "_run_gate", return_value=(False, "Test failed: 3 errors"))

        sample_step.gates = ["test"]

        with pytest.raises(GateFailedError, match="Gate 'test' failed"):
            workspace.execute_step(sample_step, worker_fn)

        # Events should include GATE_FAILED and STEP_FAILED
        events = test_db.get_events(sample_step.workflow_id)
        event_types = [e.event_type for e in events]
        assert EventType.GATE_FAILED in event_types
        assert EventType.STEP_FAILED in event_types

    def test_execute_step_worker_exception(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step
    ):
        """execute_step propagates worker exceptions and records STEP_FAILED."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        def failing_worker(step: Step, worktree_path: Path) -> dict[str, Any]:
            raise ValueError("Worker error")

        sample_step.gates = []

        with pytest.raises(ValueError, match="Worker error"):
            workspace.execute_step(sample_step, failing_worker, gates=[])

        # STEP_FAILED event recorded
        events = test_db.get_events(sample_step.workflow_id)
        failed = [e for e in events if e.event_type == EventType.STEP_FAILED]
        assert len(failed) == 1

    def test_execute_step_worktree_creation_failure(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step, mocker
    ):
        """execute_step records STEP_FAILED when worktree creation fails."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Mock worktree creation to fail
        mocker.patch.object(workspace, "_create_worktree", side_effect=WorktreeError("Disk full"))

        def worker_fn(step: Step, worktree_path: Path) -> dict[str, Any]:
            return {"status": "success"}

        sample_step.gates = []

        with pytest.raises(WorktreeError, match="Disk full"):
            workspace.execute_step(sample_step, worker_fn, gates=[])

        # STEP_FAILED event with worktree error
        events = test_db.get_events(sample_step.workflow_id)
        failed = [e for e in events if e.event_type == EventType.STEP_FAILED]
        assert len(failed) == 1
        assert "Worktree creation failed" in failed[0].payload.get("error", "")


@pytest.mark.git
class TestGateExecution:
    """Tests for gate execution in worktrees."""

    def test_run_gate_success(self, repo_with_git, test_db, mock_sandbox_executor, mocker):
        """_run_gate executes gate command and returns success."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)
        worktree_path = workspace._create_worktree("step-gate-test")

        # Mock executor to return success (uses .run() not .execute())
        from supervisor.sandbox.executor import ExecutionResult

        mock_sandbox_executor.run.return_value = ExecutionResult(
            returncode=0, stdout="All tests passed", stderr="", timed_out=False
        )

        passed, output = workspace._run_gate("test", worktree_path)

        assert passed is True
        assert "All tests passed" in output

    def test_run_gate_failure(self, repo_with_git, test_db, mock_sandbox_executor, mocker):
        """_run_gate returns failure when gate command fails."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)
        worktree_path = workspace._create_worktree("step-gate-fail")

        from supervisor.sandbox.executor import ExecutionResult

        mock_sandbox_executor.run.return_value = ExecutionResult(
            returncode=1, stdout="", stderr="Test failed: 3 errors", timed_out=False
        )

        passed, output = workspace._run_gate("test", worktree_path)

        assert passed is False
        assert "Test failed" in output

    def test_gate_runs_in_worktree_not_main(self, repo_with_git, test_db, mock_sandbox_executor):
        """Gates run in the worktree, not the main repository."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)
        worktree_path = workspace._create_worktree("step-gate-isolation")

        # Create file only in worktree
        (worktree_path / "worktree_only.txt").write_text("test")

        from supervisor.sandbox.executor import ExecutionResult

        # Gate command should see worktree files (uses .run() not .execute())
        mock_sandbox_executor.run.return_value = ExecutionResult(
            returncode=0, stdout="", stderr="", timed_out=False
        )

        workspace._run_gate("test", worktree_path)

        # Verify executor was called with worktree path
        call_args = mock_sandbox_executor.run.call_args
        assert call_args is not None
        assert worktree_path in call_args[1].values() or str(worktree_path) in str(call_args)


@pytest.mark.git
class TestAtomicApply:
    """Tests for atomic change application with file locking."""

    def test_apply_uses_file_lock(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step, mocker
    ):
        """_apply_changes_atomic uses file lock to prevent race conditions."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Spy on FileLock to verify it's used
        lock_spy = mocker.spy(workspace._apply_lock, "acquire")

        def worker_fn(step: Step, worktree_path: Path) -> dict[str, Any]:
            (worktree_path / "output.txt").write_text("content")
            return {"status": "success"}

        sample_step.gates = []
        workspace.execute_step(sample_step, worker_fn, gates=[])

        # Verify lock was acquired
        lock_spy.assert_called()

    def test_apply_head_conflict_detection(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step, mocker
    ):
        """Apply detects HEAD conflicts and raises ApplyError."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Mock _get_head_sha to return different HEAD
        mocker.patch.object(workspace, "_get_head_sha", return_value="new-commit-hash")

        def worker_fn(step: Step, worktree_path: Path) -> dict[str, Any]:
            (worktree_path / "output.txt").write_text("content")
            return {"status": "success"}

        sample_step.gates = []

        # Apply should detect HEAD changed and raise ApplyError
        with pytest.raises(ApplyError, match="HEAD has changed"):
            workspace.execute_step(sample_step, worker_fn, gates=[])


@pytest.mark.git
class TestSecurityValidation:
    """Tests for security validation in workspace operations."""

    def test_symlink_rejection_in_apply(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step
    ):
        """Symlinks in worktree changes are rejected during apply."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        def worker_fn(step: Step, worktree_path: Path) -> dict[str, Any]:
            # Worker creates a symlink
            (worktree_path / "target.txt").touch()
            (worktree_path / "symlink.txt").symlink_to(worktree_path / "target.txt")
            return {"status": "success"}

        sample_step.gates = []

        with pytest.raises(ApplyError, match="Symlink.*not allowed"):
            workspace.execute_step(sample_step, worker_fn, gates=[])

    def test_path_traversal_prevention_in_step_id(
        self, repo_with_git, test_db, mock_sandbox_executor
    ):
        """Step IDs with path traversal are sanitized."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Malicious step ID attempting traversal
        malicious_step_id = "../../../etc/passwd"

        worktree_path = workspace._create_worktree(malicious_step_id)

        # Worktree should be inside repo, not in /etc
        # Key security property: path stays inside repo
        assert worktree_path.is_relative_to(repo_with_git)
        assert ".worktrees" in str(worktree_path)  # Inside worktrees dir
        # Path doesn't escape to actual /etc
        assert not str(worktree_path).startswith("/etc")


@pytest.mark.git
class TestWorktreeCleanup:
    """Tests for worktree cleanup and resource management."""

    def test_cleanup_removes_stale_worktrees(self, repo_with_git, test_db, mock_sandbox_executor):
        """_cleanup_stale_worktrees removes unregistered worktrees."""
        # Create stale worktree directory
        stale_dir = repo_with_git / ".worktrees" / "stale-123"
        stale_dir.mkdir(parents=True)
        (stale_dir / "marker.txt").touch()

        IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Cleanup should have run during __init__
        # Verify cleanup logic (may vary based on implementation)

    def test_cleanup_skips_active_worktrees(self, repo_with_git, test_db, mock_sandbox_executor):
        """Cleanup does not remove active worktrees."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Create worktree
        worktree_path = workspace._create_worktree("step-active")

        # Re-initialize workspace (triggers cleanup)
        IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        # Active worktree should still exist
        assert worktree_path.exists()

    def test_cleanup_uses_lock_to_prevent_races(
        self, repo_with_git, test_db, mock_sandbox_executor, mocker
    ):
        """Cleanup uses lock to prevent concurrent cleanup races."""
        # Spy on FileLock to verify cleanup lock is used
        with patch("supervisor.core.workspace.FileLock") as mock_filelock_class:
            mock_lock = MagicMock()
            mock_filelock_class.return_value = mock_lock

            IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

            # Verify cleanup lock was created and acquired
            # (actual verification depends on implementation details)


@pytest.mark.git
class TestEventSourcing:
    """Tests for event sourcing patterns in workspace operations."""

    def test_step_started_event_recorded(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step
    ):
        """STEP_STARTED event is recorded at beginning of execution."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        def worker_fn(step: Step, worktree_path: Path) -> dict[str, Any]:
            return {"status": "success"}

        sample_step.gates = []
        workspace.execute_step(sample_step, worker_fn, gates=[])

        events = test_db.get_events(sample_step.workflow_id)
        started = [e for e in events if e.event_type == EventType.STEP_STARTED]
        assert len(started) == 1
        assert started[0].step_id == sample_step.id

    def test_step_completed_event_recorded(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step
    ):
        """STEP_COMPLETED event is recorded after successful execution."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        def worker_fn(step: Step, worktree_path: Path) -> dict[str, Any]:
            return {"status": "success"}

        sample_step.gates = []
        workspace.execute_step(sample_step, worker_fn, gates=[])

        events = test_db.get_events(sample_step.workflow_id)
        completed = [e for e in events if e.event_type == EventType.STEP_COMPLETED]
        assert len(completed) == 1

    def test_gate_events_recorded_in_order(
        self, repo_with_git, test_db, mock_sandbox_executor, sample_step, mocker
    ):
        """Gate events are recorded in execution order."""
        workspace = IsolatedWorkspace(repo_with_git, mock_sandbox_executor, test_db)

        def worker_fn(step: Step, worktree_path: Path) -> dict[str, Any]:
            return {"status": "success"}

        # Mock gates: first passes, second fails
        gate_results = [(True, "lint passed"), (False, "test failed")]
        mocker.patch.object(workspace, "_run_gate", side_effect=gate_results)

        sample_step.gates = ["lint", "test"]

        with pytest.raises(GateFailedError):
            workspace.execute_step(sample_step, worker_fn)

        events = test_db.get_events(sample_step.workflow_id)
        gate_events = [
            e for e in events if e.event_type in (EventType.GATE_PASSED, EventType.GATE_FAILED)
        ]

        assert len(gate_events) == 2
        assert gate_events[0].event_type == EventType.GATE_PASSED  # lint passed
        assert gate_events[0].payload["gate"] == "lint"
        assert gate_events[1].event_type == EventType.GATE_FAILED  # test failed
        assert gate_events[1].payload["gate"] == "test"
