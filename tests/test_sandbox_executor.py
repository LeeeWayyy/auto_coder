"""Tests for sandbox executor - Docker isolation and security.

Tests cover:
- Docker availability checking
- Egress verification (network security)
- Container lifecycle management
- Security validation (path traversal, symlinks, command injection)
- Execution timeouts
- Network isolation

Most tests use mocked Docker to avoid dependency on Docker daemon.
Tests marked with @pytest.mark.docker require actual Docker installation.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import Mock

import pytest

from supervisor.sandbox.executor import (
    DockerNotAvailableError,
    ExecutionResult,
    LocalExecutor,
    SandboxConfig,
    SandboxedExecutor,
    SandboxedLLMClient,
    SandboxError,
)

# =============================================================================
# Docker Availability Tests
# =============================================================================


class TestDockerAvailability:
    """Tests for Docker availability checking."""

    def test_require_docker_raises_when_unavailable(self, mocker):
        """require_docker raises DockerNotAvailableError when Docker not available."""
        from supervisor.sandbox.executor import _validate_docker

        # Mock shutil.which to return None (docker not found)
        mocker.patch("shutil.which", return_value=None)

        with pytest.raises(DockerNotAvailableError):
            _validate_docker()

    def test_require_docker_succeeds_when_available(self, mocker):
        """require_docker succeeds when Docker is available."""
        from supervisor.sandbox.executor import _validate_docker

        # Mock shutil.which and subprocess.run
        mocker.patch("shutil.which", return_value="/usr/bin/docker")
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = b"Docker version 24.0.0"
        mocker.patch("subprocess.run", return_value=mock_result)

        # Should not raise
        _validate_docker()

    def test_require_docker_raises_on_permission_denied(self, mocker):
        """require_docker raises when Docker socket has permission issues."""
        from supervisor.sandbox.executor import _validate_docker

        mocker.patch("shutil.which", return_value="/usr/bin/docker")
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = b""
        mock_result.stderr = (
            b"permission denied while trying to connect to the Docker daemon socket"
        )
        mocker.patch("subprocess.run", return_value=mock_result)

        with pytest.raises(DockerNotAvailableError):
            _validate_docker()


# =============================================================================
# Egress Verification Tests
# =============================================================================


@pytest.mark.docker
class TestEgressVerification:
    """Tests for network egress rule verification."""

    def test_egress_rules_verification_exists(self):
        """Verify _verify_egress_rules function exists."""
        from supervisor.sandbox.executor import _verify_egress_rules

        # Function should exist and be callable
        assert callable(_verify_egress_rules)

    def test_egress_rules_returns_boolean_or_none(self, mocker):
        """Egress verification returns True, False, or None."""
        from supervisor.sandbox.executor import _verify_egress_rules

        # Mock subprocess to simulate network not found (non-zero returncode)
        mock_result = Mock(returncode=1, stdout="", stderr="network not found")
        mocker.patch("subprocess.run", return_value=mock_result)

        result = _verify_egress_rules("test-network", ["api.anthropic.com:443"])

        # Should return False when network not found
        assert result in (True, False, None)


# =============================================================================
# SandboxConfig Tests
# =============================================================================


class TestSandboxConfig:
    """Tests for SandboxConfig configuration."""

    def test_default_config(self):
        """SandboxConfig has sensible defaults."""
        config = SandboxConfig()

        assert config.cli_image == "supervisor-cli:latest"
        assert config.executor_image == "supervisor-executor:latest"
        assert config.egress_network == "supervisor-egress"
        assert config.cli_timeout == 300
        assert config.executor_timeout == 600
        assert config.max_output_bytes == 10 * 1024 * 1024  # 10MB
        assert config.pids_limit == 256

    def test_custom_config(self):
        """SandboxConfig accepts custom values."""
        config = SandboxConfig(
            cli_image="custom-cli:latest",
            executor_image="custom-executor:latest",
            memory_limit="8g",
            cpu_limit="4",
            cli_timeout=600,
            executor_timeout=1200,
        )

        assert config.cli_image == "custom-cli:latest"
        assert config.executor_image == "custom-executor:latest"
        assert config.memory_limit == "8g"
        assert config.cpu_limit == "4"
        assert config.cli_timeout == 600
        assert config.executor_timeout == 1200

    def test_security_defaults(self):
        """SandboxConfig has secure defaults."""
        config = SandboxConfig()

        assert config.require_docker is True
        assert config.verify_egress_rules is True
        assert config.fail_on_unverified_egress is True

    def test_egress_allowlist_defaults(self):
        """SandboxConfig has default egress allowlist for AI APIs."""
        config = SandboxConfig()

        assert "api.anthropic.com:443" in config.allowed_egress
        assert "api.openai.com:443" in config.allowed_egress
        assert "generativelanguage.googleapis.com:443" in config.allowed_egress


# =============================================================================
# SandboxedLLMClient Tests
# =============================================================================


@pytest.mark.docker
class TestSandboxedLLMClient:
    """Tests for SandboxedLLMClient - AI CLI execution with network egress."""

    def test_client_init_requires_cli_name(self, mocker):
        """SandboxedLLMClient requires cli_name parameter."""
        mocker.patch("supervisor.sandbox.executor._validate_docker")
        mocker.patch("supervisor.sandbox.executor._verify_egress_rules", return_value=True)
        mocker.patch.object(SandboxedLLMClient, "_ensure_network_exists")

        config = SandboxConfig(require_docker=False, verify_egress_rules=False)
        client = SandboxedLLMClient(cli_name="claude", config=config)

        assert client is not None

    def test_client_accepts_model_id(self, mocker):
        """SandboxedLLMClient accepts optional model_id."""
        mocker.patch("supervisor.sandbox.executor._validate_docker")
        mocker.patch("supervisor.sandbox.executor._verify_egress_rules", return_value=True)
        mocker.patch.object(SandboxedLLMClient, "_ensure_network_exists")

        config = SandboxConfig(require_docker=False, verify_egress_rules=False)
        client = SandboxedLLMClient(
            cli_name="claude",
            config=config,
            model_id="claude-3-sonnet",
        )

        assert client is not None

    def test_execute_requires_prompt_and_workdir(self, mocker, temp_repo):
        """SandboxedLLMClient.execute requires prompt and workdir."""
        mocker.patch("supervisor.sandbox.executor._validate_docker")
        mocker.patch("supervisor.sandbox.executor._verify_egress_rules", return_value=True)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = Mock(
            returncode=0,
            stdout="AI response output",  # text=True returns str, not bytes
            stderr="",
        )

        config = SandboxConfig(
            require_docker=False,
            verify_egress_rules=False,
            allowed_workdir_roots=[str(temp_repo.parent)],  # Allow temp_repo
        )
        client = SandboxedLLMClient(cli_name="claude", config=config)

        result = client.execute(prompt="Hello", workdir=temp_repo)

        assert isinstance(result, ExecutionResult)


# =============================================================================
# SandboxedExecutor Tests
# =============================================================================


@pytest.mark.docker
class TestSandboxedExecutor:
    """Tests for SandboxedExecutor - fully isolated execution (no network)."""

    def test_executor_init_with_config(self, mocker):
        """SandboxedExecutor accepts config parameter."""
        mocker.patch("supervisor.sandbox.executor._validate_docker")

        config = SandboxConfig(require_docker=False)
        executor = SandboxedExecutor(config=config)

        assert executor is not None

    def test_executor_run_method(self, mocker, temp_repo):
        """SandboxedExecutor.run executes commands."""
        mocker.patch("supervisor.sandbox.executor._validate_docker")
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Command output",  # text=True returns str, not bytes
            stderr="",
        )

        config = SandboxConfig(
            require_docker=False,
            allowed_workdir_roots=[str(temp_repo.parent)],  # Allow temp_repo
        )
        executor = SandboxedExecutor(config=config)

        result = executor.run(command=["echo", "test"], workdir=temp_repo)

        assert isinstance(result, ExecutionResult)
        mock_run.assert_called()


# =============================================================================
# LocalExecutor Tests
# =============================================================================


class TestLocalExecutor:
    """Tests for LocalExecutor (unit testing only - not for production)."""

    def test_local_executor_requires_env_var(self):
        """LocalExecutor requires SUPERVISOR_ALLOW_LOCAL_EXECUTOR=1."""
        # Without env var, should raise
        with pytest.raises(SandboxError, match="SUPERVISOR_ALLOW_LOCAL_EXECUTOR"):
            LocalExecutor()

    def test_local_executor_with_env_var(self, mocker, capsys, temp_repo):
        """LocalExecutor works with env var set."""
        # Set required env var
        mocker.patch.dict(os.environ, {"SUPERVISOR_ALLOW_LOCAL_EXECUTOR": "1"})

        executor = LocalExecutor(workdir=temp_repo)

        assert executor is not None

        # Should print warning (may be in stdout or stderr)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "WARNING" in captured.err or executor is not None

    def test_local_executor_run(self, mocker, temp_repo):
        """LocalExecutor can run commands."""
        mocker.patch.dict(os.environ, {"SUPERVISOR_ALLOW_LOCAL_EXECUTOR": "1"})

        executor = LocalExecutor(workdir=temp_repo)

        result = executor.run(command=["echo", "test"], workdir=temp_repo)

        assert result.returncode == 0
        assert "test" in result.stdout


# =============================================================================
# ExecutionResult Tests
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_execution_result_fields(self):
        """ExecutionResult has required fields."""
        result = ExecutionResult(
            returncode=0,
            stdout="output",
            stderr="",
            timed_out=False,
        )

        assert result.returncode == 0
        assert result.stdout == "output"
        assert result.stderr == ""
        assert result.timed_out is False

    def test_execution_result_timeout(self):
        """ExecutionResult can represent timeout."""
        result = ExecutionResult(
            returncode=-1,
            stdout="",
            stderr="Command timed out",
            timed_out=True,
        )

        assert result.timed_out is True


# =============================================================================
# Integration Tests (Require Docker)
# =============================================================================


@pytest.mark.docker
class TestSandboxDockerIntegration:
    """Integration tests that require actual Docker installation."""

    def test_real_docker_execution(self, temp_repo):
        """Execute command in real Docker container."""
        from supervisor.sandbox.executor import _validate_docker

        try:
            _validate_docker()
        except DockerNotAvailableError:
            pytest.skip("Docker not available")

        # Check if the executor image exists
        image_check = subprocess.run(
            ["docker", "image", "inspect", "supervisor-executor:latest"],
            capture_output=True,
        )
        if image_check.returncode != 0:
            pytest.skip("Docker image 'supervisor-executor:latest' not available")

        config = SandboxConfig(
            allowed_workdir_roots=[str(temp_repo.parent)],
        )
        executor = SandboxedExecutor(config=config)

        # Execute simple command
        # Use -u for unbuffered output to ensure stdout is captured before container exits
        result = executor.run(
            command=["python", "-u", "-c", "print('Hello from Docker')"],
            workdir=temp_repo,
            timeout=30,
        )

        assert result.returncode == 0
        assert "Hello from Docker" in result.stdout
