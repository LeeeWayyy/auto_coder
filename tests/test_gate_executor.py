"""Tests for gate execution orchestration with caching and security.

This module tests the GateExecutor class which provides:
- Gate command execution in sandboxed containers
- Result caching with cache key computation
- Dependency resolution
- Environment variable filtering for security
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from supervisor.core.gate_executor import GateExecutor
from supervisor.core.gate_models import (
    GateConfig,
    GateResult,
    GateStatus,
    GateSeverity,
    GateFailAction,
)
from supervisor.sandbox.executor import ExecutionResult


# =============================================================================
# Environment Filtering Tests (Security)
# =============================================================================


class TestEnvironmentFiltering:
    """Tests for security filtering of environment variables."""

    @pytest.fixture
    def executor(self, mock_sandbox_executor, mock_gate_loader, test_db):
        """Create a GateExecutor instance for testing."""
        return GateExecutor(mock_sandbox_executor, mock_gate_loader, test_db)

    def test_filter_env_removes_protected_vars(self, executor):
        """Protected env vars like PATH, HOME are not passed to gates."""
        gate_env = {
            "PATH": "/malicious/bin",
            "HOME": "/tmp/fake",
            "CUSTOM_VAR": "allowed",
        }

        filtered = executor._filter_env(gate_env)

        assert "PATH" not in filtered
        assert "HOME" not in filtered
        assert filtered["CUSTOM_VAR"] == "allowed"

    def test_filter_env_removes_supervisor_prefix(self, executor):
        """Env vars with SUPERVISOR_ prefix are filtered."""
        gate_env = {
            "SUPERVISOR_INTERNAL": "value",
            "SUPERVISOR_DB_PATH": "/path",
            "NORMAL_VAR": "allowed",
        }

        filtered = executor._filter_env(gate_env)

        assert "SUPERVISOR_INTERNAL" not in filtered
        assert "SUPERVISOR_DB_PATH" not in filtered
        assert filtered["NORMAL_VAR"] == "allowed"

    def test_filter_env_case_insensitive(self, executor):
        """Environment filtering is case-insensitive for denylist."""
        gate_env = {
            "path": "/bin",  # lowercase - should still be filtered
            "Path": "/usr/bin",  # mixed case - should still be filtered
            "CUSTOM": "allowed",
        }

        filtered = executor._filter_env(gate_env)

        # Implementation uses key.upper() comparison, so all case variants are filtered
        assert "path" not in filtered, "lowercase 'path' should be filtered"
        assert "Path" not in filtered, "mixed-case 'Path' should be filtered"
        assert filtered["CUSTOM"] == "allowed"

    def test_filter_env_removes_ld_preload(self, executor):
        """LD_PRELOAD and LD_LIBRARY_PATH are filtered for security."""
        gate_env = {
            "LD_PRELOAD": "/malicious/lib.so",
            "LD_LIBRARY_PATH": "/tmp/libs",
            "SAFE_VAR": "value",
        }

        filtered = executor._filter_env(gate_env)

        assert "LD_PRELOAD" not in filtered
        assert "LD_LIBRARY_PATH" not in filtered
        assert filtered["SAFE_VAR"] == "value"


# =============================================================================
# Cache Key Computation Tests
# =============================================================================


class TestCacheKeyComputation:
    """Tests for cache key computation based on worktree state."""

    @pytest.fixture
    def executor(self, mock_sandbox_executor, mock_gate_loader, test_db):
        """Create a GateExecutor instance for testing."""
        return GateExecutor(mock_sandbox_executor, mock_gate_loader, test_db)

    def test_compute_cache_key_none_when_disabled(self, executor, temp_repo):
        """Cache key is None when caching is disabled."""
        gate_config = GateConfig(
            name="test",
            command=["pytest"],
            timeout=60,
            cache=False,  # Disabled
        )

        cache_key = executor._compute_cache_key(temp_repo, gate_config)

        assert cache_key is None

    def test_compute_cache_key_skips_ignored_directories(self, executor, repo_with_git):
        """Cache key computation skips ignored directories like node_modules."""
        gate_config = GateConfig(name="test", command=["pytest"], timeout=60, cache=True)

        # Compute initial key
        key1 = executor._compute_cache_key(repo_with_git, gate_config)

        # Create node_modules directory (should be ignored)
        node_modules = repo_with_git / "node_modules"
        node_modules.mkdir()
        (node_modules / "package.json").write_text('{"name": "test"}')

        # Compute new key - should be same (node_modules ignored)
        key2 = executor._compute_cache_key(repo_with_git, gate_config)

        # If keys are computed, they should be same (node_modules ignored)
        # If caching can't compute key in temp repo, both will be None
        assert key1 == key2


# =============================================================================
# Gate Configuration Tests
# =============================================================================


class TestGateConfig:
    """Tests for GateConfig model."""

    def test_gate_config_defaults(self):
        """GateConfig has sensible defaults."""
        config = GateConfig(
            name="test",
            command=["pytest"],
            timeout=60,
        )

        assert config.name == "test"
        assert config.command == ["pytest"]
        assert config.timeout == 60
        assert config.cache is True  # Default is True
        assert config.severity == GateSeverity.ERROR  # Default

    def test_gate_config_with_all_options(self):
        """GateConfig accepts all options."""
        config = GateConfig(
            name="lint",
            command=["ruff", "check", "."],
            timeout=120,
            description="Run linter",
            cache=False,
            severity=GateSeverity.WARNING,
            working_dir="src",
            depends_on=["test"],
        )

        assert config.name == "lint"
        assert config.cache is False
        assert config.severity == GateSeverity.WARNING
        assert config.working_dir == "src"
        assert config.depends_on == ["test"]


# =============================================================================
# Gate Result Tests
# =============================================================================


class TestGateResult:
    """Tests for GateResult model."""

    def test_gate_result_passed(self):
        """GateResult represents passed gate."""
        result = GateResult(
            gate_name="test",
            status=GateStatus.PASSED,
            output="All tests passed",
            duration_seconds=1.5,
            returncode=0,
        )

        assert result.gate_name == "test"
        assert result.status == GateStatus.PASSED
        assert result.returncode == 0
        assert result.passed is True

    def test_gate_result_failed(self):
        """GateResult represents failed gate."""
        result = GateResult(
            gate_name="test",
            status=GateStatus.FAILED,
            output="Test failed",
            duration_seconds=0.5,
            returncode=1,
        )

        assert result.status == GateStatus.FAILED
        assert result.returncode == 1
        assert result.failed is True

    def test_gate_result_skipped(self):
        """GateResult represents skipped gate."""
        result = GateResult(
            gate_name="test",
            status=GateStatus.SKIPPED,
            output="",
            duration_seconds=0.0,
        )

        assert result.status == GateStatus.SKIPPED
        assert result.skipped is True

    def test_gate_result_cached(self):
        """GateResult represents cached result."""
        result = GateResult(
            gate_name="test",
            status=GateStatus.PASSED,
            output="Cached result",
            duration_seconds=0.0,
            cached=True,
        )

        assert result.cached is True


# =============================================================================
# Gate Execution Basic Tests
# =============================================================================


class TestGateExecution:
    """Basic tests for gate execution."""

    def test_run_gate_invalid_working_dir(self, mock_sandbox_executor, mock_gate_loader, test_db, temp_repo):
        """Gate execution fails for invalid working directory."""
        executor = GateExecutor(mock_sandbox_executor, mock_gate_loader, test_db)

        config = GateConfig(
            name="test",
            command=["pytest"],
            timeout=60,
            working_dir="/absolute/path",  # Absolute paths should be rejected
        )

        # Running with absolute working_dir should fail validation
        result = executor.run_gate(config, temp_repo, "wf-1", "step-1")

        # Should fail due to path traversal prevention
        assert result.status == GateStatus.FAILED

    def test_run_gate_path_traversal_prevention(self, mock_sandbox_executor, mock_gate_loader, test_db, temp_repo):
        """Gate execution prevents path traversal."""
        executor = GateExecutor(mock_sandbox_executor, mock_gate_loader, test_db)

        config = GateConfig(
            name="test",
            command=["pytest"],
            timeout=60,
            working_dir="../../../etc",  # Path traversal attempt
        )

        result = executor.run_gate(config, temp_repo, "wf-1", "step-1")

        # Should fail due to path traversal prevention
        assert result.status == GateStatus.FAILED


# =============================================================================
# Gate Severity Tests
# =============================================================================


class TestGateSeverity:
    """Tests for gate severity levels."""

    def test_severity_levels(self):
        """Gate severity levels are defined correctly."""
        assert GateSeverity.ERROR.value == "error"
        assert GateSeverity.WARNING.value == "warning"
        assert GateSeverity.INFO.value == "info"

    def test_fail_actions(self):
        """Gate fail actions are defined correctly."""
        assert GateFailAction.BLOCK.value == "block"
        assert GateFailAction.WARN.value == "warn"
        assert GateFailAction.RETRY_WITH_FEEDBACK.value == "retry_with_feedback"
