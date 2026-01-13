# conftest.py - Shared pytest fixtures for all tests
"""Shared pytest fixtures for Supervisor test suite.

This module provides foundational fixtures used across all test modules:
- Temporary repositories and file structures
- Test databases with event sourcing
- Sample configurations (roles, gates, sandbox)
- Mock objects for external dependencies

Usage:
    Import fixtures implicitly via pytest's fixture discovery.
    All fixtures in this file are automatically available in test modules.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, Mock

import pytest
import yaml

from supervisor.core.roles import RoleConfig
from supervisor.core.state import Database


# =============================================================================
# Repository and File System Fixtures
# =============================================================================


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with a basic file structure.

    Creates:
        - src/ directory with sample Python files
        - tests/ directory
        - README.md
        - pyproject.toml
        - .git/ directory (for git operations)

    Returns:
        Path to the temporary repository root.

    Example:
        def test_something(temp_repo):
            assert (temp_repo / "src" / "main.py").exists()
            # Add test files as needed
            (temp_repo / "src" / "feature.py").write_text("...")
    """
    # Create source directory
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("")
    (src_dir / "main.py").write_text(
        '"""Main module."""\n\ndef main():\n    """Entry point."""\n    pass\n'
    )
    (src_dir / "utils.py").write_text(
        '"""Utilities."""\n\ndef helper(x: int) -> int:\n    """Helper function."""\n    return x * 2\n'
    )

    # Create tests directory
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "test_main.py").write_text(
        "import pytest\n\ndef test_placeholder():\n    assert True\n"
    )

    # Create project files
    (tmp_path / "README.md").write_text("# Test Project\n\nA test project for Supervisor.\n")

    (tmp_path / "pyproject.toml").write_text(
        """[project]
name = "test-project"
version = "0.1.0"

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
    )

    # Initialize basic git repository structure
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n")

    return tmp_path


@pytest.fixture
def empty_repo(tmp_path: Path) -> Path:
    """Create an empty temporary directory (no files).

    Useful for tests that need to build file structure from scratch.

    Returns:
        Path to empty temporary directory.
    """
    return tmp_path


@pytest.fixture
def repo_with_git(temp_repo: Path) -> Path:
    """Create a temporary repository with actual git initialization.

    WARNING: Runs actual git commands. Slower than temp_repo.
    Only use when you need real git operations (worktrees, commits, etc.).

    Returns:
        Path to git-initialized repository.

    Example:
        def test_git_operation(repo_with_git):
            # Can run real git commands
            subprocess.run(
                ["git", "add", "."],
                cwd=repo_with_git,
                check=True
            )
    """
    # Remove fake .git directory first
    import shutil

    fake_git = temp_repo / ".git"
    if fake_git.exists():
        shutil.rmtree(fake_git)

    try:
        subprocess.run(
            ["git", "init"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
        )
        # Initial commit to have a valid HEAD
        subprocess.run(
            ["git", "add", "."],
            cwd=temp_repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
        )
        return temp_repo
    except subprocess.CalledProcessError:
        pytest.skip("Git not available")


# =============================================================================
# Database and State Management Fixtures
# =============================================================================


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    """Create a temporary test database with event sourcing.

    Creates a fresh SQLite database in a temporary directory.
    Database is automatically cleaned up after the test.

    Returns:
        Initialized Database instance.

    Example:
        def test_event_logging(test_db):
            test_db.log_event(Event(
                workflow_id="test-123",
                event_type=EventType.WORKFLOW_STARTED,
                payload={"description": "Test workflow"}
            ))
            events = test_db.get_events("test-123")
            assert len(events) == 1
    """
    db_path = tmp_path / "test.db"
    return Database(db_path)


@pytest.fixture
def populated_db(test_db: Database) -> tuple[Database, str, str, str]:
    """Create a database populated with sample feature/phase/component hierarchy.

    Creates a realistic workflow structure:
    - 1 Feature with 2 phases
    - Phase 1 has 2 components
    - Phase 2 has 1 component

    Returns:
        Tuple of (database, feature_id, phase1_id, phase2_id).

    Example:
        def test_component_queries(populated_db):
            db, feature_id, phase1_id, _ = populated_db
            components = db.get_components_for_phase(phase1_id)
            assert len(components) == 2
    """
    feature_id = "F-TEST001"

    # Create feature using correct API
    test_db.create_feature(
        feature_id=feature_id,
        title="User Authentication",
        description="Test feature: User authentication",
    )

    # Create phases
    phase1_id = f"{feature_id}-PH1"
    phase2_id = f"{feature_id}-PH2"

    test_db.create_phase(
        phase_id=phase1_id,
        feature_id=feature_id,
        title="Phase 1: Core Implementation",
        sequence=1,
    )
    test_db.create_phase(
        phase_id=phase2_id,
        feature_id=feature_id,
        title="Phase 2: Integration",
        sequence=2,
    )

    # Create components in phase 1
    comp1_id = f"{phase1_id}-C1"
    comp2_id = f"{phase1_id}-C2"

    test_db.create_component(
        component_id=comp1_id,
        phase_id=phase1_id,
        title="login_endpoint",
        description="Implement /api/login endpoint",
        files=["src/api/auth.py"],
        depends_on=[],
    )
    test_db.create_component(
        component_id=comp2_id,
        phase_id=phase1_id,
        title="password_hashing",
        description="Add password hashing utilities",
        files=["src/utils/crypto.py"],
        depends_on=[],
    )

    # Create component in phase 2
    comp3_id = f"{phase2_id}-C1"

    test_db.create_component(
        component_id=comp3_id,
        phase_id=phase2_id,
        title="integration_tests",
        description="Add integration tests",
        files=["tests/integration/test_auth.py"],
        depends_on=[comp1_id, comp2_id],  # Depends on phase 1 components
    )

    return test_db, feature_id, phase1_id, phase2_id


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_role_config() -> RoleConfig:
    """Create a sample role configuration for testing.

    Returns:
        Basic RoleConfig instance with common defaults.

    Example:
        def test_role_loading(sample_role_config):
            assert sample_role_config.name == "tester"
            assert sample_role_config.cli == "claude"
    """
    return RoleConfig(
        name="tester",
        description="Test role for unit tests",
        cli="claude:sonnet",
        flags=["-p", "You are a helpful test assistant."],
        system_prompt="You are a helpful test assistant. Follow instructions carefully.",
        context={"token_budget": 25000, "include": ["src/**/*.py"]},
        gates=["test", "lint"],
        config={"max_retries": 3, "timeout": 300},
    )


@pytest.fixture
def minimal_role_config() -> RoleConfig:
    """Create a minimal role configuration (only required fields).

    Returns:
        Minimal RoleConfig instance.
    """
    return RoleConfig(
        name="minimal",
        description="Minimal test role",
        cli="claude:sonnet",
        flags=[],
        system_prompt="Test prompt",
        context={},
        gates=[],
        config={},
    )


@pytest.fixture
def sample_gates_config(tmp_path: Path) -> Path:
    """Create a sample gates.yaml configuration file.

    Returns:
        Path to the gates.yaml file.

    Example:
        def test_gate_loading(sample_gates_config):
            loader = GateLoader(sample_gates_config)
            gates = loader.load_all()
            assert "test" in gates
    """
    gates_config = {
        "gates": {
            "test": {
                "command": ["pytest", "-q"],
                "timeout": 300,
                "description": "Run test suite",
                "severity": "error",
            },
            "lint": {
                "command": ["ruff", "check", "."],
                "timeout": 60,
                "description": "Run linter",
                "depends_on": [],
            },
            "type_check": {
                "command": ["mypy", "."],
                "timeout": 120,
                "description": "Type checking",
                "depends_on": ["lint"],
            },
        }
    }

    config_path = tmp_path / "gates.yaml"
    config_path.write_text(yaml.safe_dump(gates_config))
    return config_path


# =============================================================================
# Mock Fixtures for External Dependencies
# =============================================================================


@pytest.fixture
def mock_docker_client() -> Mock:
    """Create a mock Docker client for testing sandbox execution.

    Returns:
        Mock Docker client with common operations stubbed.

    Example:
        def test_sandbox_execution(mock_docker_client, mocker):
            mocker.patch('docker.from_env', return_value=mock_docker_client)
            executor = SandboxedExecutor(...)
            # Test without actual Docker
    """
    mock_client = Mock()
    mock_container = Mock()

    # Mock container operations
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = b"Command output\n"
    mock_container.remove.return_value = None

    # Mock client operations
    mock_client.containers.run.return_value = mock_container
    mock_client.containers.get.return_value = mock_container
    mock_client.ping.return_value = True
    mock_client.version.return_value = {"Version": "24.0.0"}

    return mock_client


@pytest.fixture
def mock_subprocess(mocker) -> Mock:
    """Create a mock for subprocess.run operations.

    Returns:
        Mock subprocess.run function.

    Example:
        def test_git_command(mock_subprocess):
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "git output"
            # Test git operations without actual git commands
    """
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = ""
    mock_run.return_value.stderr = ""
    return mock_run


@pytest.fixture
def mock_execution_engine(mocker) -> Mock:
    """Create a mock ExecutionEngine for testing CLI commands.

    Returns:
        Mock ExecutionEngine instance.

    Example:
        def test_cli_plan_command(mock_execution_engine):
            mock_execution_engine.run_plan.return_value = "feature-123"
            # Test CLI without actual engine execution
    """
    mock_engine = Mock()
    mock_engine.run_plan.return_value = "feature-123"
    mock_engine.run_role.return_value = {"status": "success", "step_id": "step-123"}
    return mock_engine


@pytest.fixture
def mock_sandbox_executor() -> Mock:
    """Create a mock SandboxedExecutor for testing gate execution.

    Returns:
        Mock SandboxedExecutor instance.
    """
    from supervisor.sandbox.executor import ExecutionResult

    mock_executor = Mock()
    mock_executor.execute.return_value = ExecutionResult(
        returncode=0,
        stdout="Success",
        stderr="",
        timed_out=False,
    )
    return mock_executor


@pytest.fixture
def mock_gate_loader() -> Mock:
    """Create a mock GateLoader for testing gate execution.

    Returns:
        Mock GateLoader instance.
    """
    from supervisor.core.gate_models import GateConfig, GateSeverity

    mock_loader = Mock()
    mock_loader.load_gate.return_value = GateConfig(
        name="test",
        command=["pytest", "-q"],
        timeout=300,
        description="Run tests",
        severity=GateSeverity.ERROR,
    )
    return mock_loader


@pytest.fixture
def sample_step() -> Mock:
    """Create a sample Step mock for testing.

    Returns:
        Mock Step instance.
    """
    from supervisor.core.models import Step, StepStatus

    mock_step = Mock(spec=Step)
    mock_step.id = "step-001"
    mock_step.workflow_id = "wf-001"
    mock_step.role = "implementer"
    mock_step.status = StepStatus.PENDING
    mock_step.gates = ["test", "lint"]
    mock_step.context = {}
    return mock_step


# =============================================================================
# Sandbox Configuration Fixtures
# =============================================================================


@pytest.fixture
def mock_sandbox_config() -> dict[str, Any]:
    """Create a mock sandbox configuration.

    Returns:
        Dictionary with sandbox settings.

    Example:
        def test_sandbox_validation(mock_sandbox_config):
            assert mock_sandbox_config["cli_image"] is not None
    """
    return {
        "cli_image": "supervisor-cli:latest",
        "executor_image": "supervisor-executor:latest",
        "egress_network": "supervisor-egress",
        "memory_limit": "4g",
        "cpu_limit": "2",
        "cli_timeout": 300,
        "executor_timeout": 600,
        "max_output_bytes": 10 * 1024 * 1024,
        "pids_limit": 256,
    }


# =============================================================================
# Event Sourcing Test Helpers
# =============================================================================


@pytest.fixture
def sample_event_sequence() -> list[dict[str, Any]]:
    """Create a sample sequence of events for event sourcing tests.

    Returns:
        List of event dictionaries representing a typical workflow.

    Example:
        def test_event_replay(test_db, sample_event_sequence):
            for event_data in sample_event_sequence:
                test_db.log_event(Event(**event_data))
            # Verify projection state
    """
    workflow_id = "test-workflow-001"
    return [
        {
            "workflow_id": workflow_id,
            "event_type": "workflow_started",
            "role": None,
            "payload": {"description": "Test workflow", "started_by": "test_user"},
        },
        {
            "workflow_id": workflow_id,
            "event_type": "step_started",
            "role": "planner",
            "step_id": "step-001",
            "payload": {"task": "Plan feature"},
        },
        {
            "workflow_id": workflow_id,
            "event_type": "step_completed",
            "role": "planner",
            "step_id": "step-001",
            "status": "success",
            "payload": {"result": "Planning complete"},
        },
        {
            "workflow_id": workflow_id,
            "event_type": "gate_passed",
            "step_id": "step-001",
            "payload": {"gate": "lint", "duration": 1.2},
        },
        {
            "workflow_id": workflow_id,
            "event_type": "workflow_completed",
            "status": "success",
            "payload": {"total_steps": 1},
        },
    ]


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "docker: marks tests requiring Docker")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "git: marks tests requiring git")
