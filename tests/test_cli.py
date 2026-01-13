"""Tests for CLI commands.

Tests all supervisor CLI commands using Click's CliRunner:
- init: Initialize project
- plan: Run planner on a task
- run: Execute specific role
- workflow: Run feature workflow
- metrics: View performance metrics
- roles: List available roles
- status: Show workflow status
- version: Show version information
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from supervisor.cli import main
from supervisor.core.state import Database


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def isolated_fs(cli_runner):
    """Create an isolated filesystem for CLI tests."""
    with cli_runner.isolated_filesystem():
        yield Path.cwd()


class TestInitCommand:
    """Tests for 'supervisor init' command."""

    def test_init_creates_directory_structure(self, cli_runner):
        """Init creates .supervisor directory with proper structure."""
        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["init"])

            assert result.exit_code == 0
            assert "Project initialized!" in result.output

            # Check directory structure
            assert (Path(".supervisor")).exists()
            assert (Path(".supervisor/roles")).exists()
            assert (Path(".supervisor/templates")).exists()

    def test_init_creates_config_files(self, cli_runner):
        """Init creates all required configuration files."""
        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["init"])
            assert result.exit_code == 0

            # Check config files
            config_path = Path(".supervisor/config.yaml")
            assert config_path.exists()
            config = yaml.safe_load(config_path.read_text())
            assert "default_cli" in config
            assert config["default_cli"] == "claude"

            # Check limits.yaml
            limits_path = Path(".supervisor/limits.yaml")
            assert limits_path.exists()
            limits = yaml.safe_load(limits_path.read_text())
            assert "workflow_timeout" in limits
            assert limits["workflow_timeout"] == 3600

            # Check adaptive.yaml
            adaptive_path = Path(".supervisor/adaptive.yaml")
            assert adaptive_path.exists()

            # Check approval.yaml
            approval_path = Path(".supervisor/approval.yaml")
            assert approval_path.exists()

    def test_init_creates_database(self, cli_runner):
        """Init creates SQLite database."""
        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["init"])
            assert result.exit_code == 0

            db_path = Path(".supervisor/state.db")
            assert db_path.exists()

            # Verify database is valid
            db = Database(db_path)
            assert db is not None

    def test_init_already_initialized(self, cli_runner):
        """Init on already initialized project shows warning."""
        with cli_runner.isolated_filesystem():
            # First init
            result1 = cli_runner.invoke(main, ["init"])
            assert result1.exit_code == 0

            # Second init
            result2 = cli_runner.invoke(main, ["init"])
            assert result2.exit_code == 0
            assert "already initialized" in result2.output.lower()


class TestPlanCommand:
    """Tests for 'supervisor plan' command."""

    def test_plan_command_basic(self, cli_runner, mocker):
        """Plan command executes with basic task."""
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")
        mock_instance = mock_engine.return_value
        mock_instance.run_plan.return_value = Mock(phases=[], risks=[])

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["plan", "Add user authentication"])
            assert result.exit_code == 0
            assert "Planning task:" in result.output
            assert "Add user authentication" in result.output

            # Verify engine was called
            mock_instance.run_plan.assert_called_once()
            call_args = mock_instance.run_plan.call_args
            assert "Add user authentication" in call_args[0]

    def test_plan_command_with_workflow_id(self, cli_runner, mocker):
        """Plan command accepts custom workflow ID."""
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")
        mock_instance = mock_engine.return_value
        mock_instance.run_plan.return_value = Mock(phases=[], risks=[])

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(
                main, ["plan", "Test task", "--workflow-id", "custom-wf-123"]
            )
            assert result.exit_code == 0
            assert "custom-wf-123" in result.output

    def test_plan_command_dry_run(self, cli_runner, mocker):
        """Plan command dry-run mode doesn't execute engine."""
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["plan", "Test task", "--dry-run"])
            assert result.exit_code == 0
            assert "Dry run" in result.output

            # Engine should not be called in dry run
            mock_engine.return_value.run_plan.assert_not_called()

    def test_plan_command_engine_error(self, cli_runner, mocker):
        """Plan command handles engine errors gracefully."""
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")
        mock_instance = mock_engine.return_value
        mock_instance.run_plan.side_effect = Exception("Engine error")

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["plan", "Test task"])
            assert result.exit_code == 1
            assert "Error:" in result.output


class TestRunCommand:
    """Tests for 'supervisor run' command."""

    def test_run_command_basic(self, cli_runner, mocker):
        """Run command executes role with task."""
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")
        mock_instance = mock_engine.return_value
        mock_instance.run_role.return_value = Mock(
            status="success", files_modified=[], files_created=[]
        )

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["run", "implementer", "Add function"])
            assert result.exit_code == 0
            assert "Running role:" in result.output
            assert "implementer" in result.output

    def test_run_command_with_targets(self, cli_runner, mocker):
        """Run command accepts target files."""
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")
        mock_instance = mock_engine.return_value
        mock_instance.run_role.return_value = Mock(
            status="success", files_modified=[], files_created=[]
        )

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(
                main,
                ["run", "implementer", "Fix bug", "-t", "src/main.py", "-t", "src/utils.py"],
            )
            assert result.exit_code == 0

            # Verify target files were passed
            call_args = mock_instance.run_role.call_args
            assert call_args[1]["target_files"] == ["src/main.py", "src/utils.py"]

    def test_run_command_with_workflow_id(self, cli_runner, mocker):
        """Run command accepts custom workflow ID."""
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")
        mock_instance = mock_engine.return_value
        mock_instance.run_role.return_value = Mock(
            status="success", files_modified=[], files_created=[]
        )

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(
                main,
                ["run", "implementer", "Task", "--workflow-id", "wf-custom"],
            )
            assert result.exit_code == 0
            assert "wf-custom" in result.output

    def test_run_command_shows_file_modifications(self, cli_runner, mocker):
        """Run command displays modified and created files."""
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")
        mock_instance = mock_engine.return_value
        mock_instance.run_role.return_value = Mock(
            status="success",
            files_modified=["src/main.py", "src/utils.py"],
            files_created=["src/new_module.py"],
        )

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["run", "implementer", "Task"])
            assert result.exit_code == 0
            assert "Files modified:" in result.output
            assert "src/main.py" in result.output
            assert "Files created:" in result.output
            assert "src/new_module.py" in result.output

    def test_run_command_handles_errors(self, cli_runner, mocker):
        """Run command handles execution errors."""
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")
        mock_instance = mock_engine.return_value
        mock_instance.run_role.side_effect = ValueError("Invalid role")

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["run", "invalid_role", "Task"])
            assert result.exit_code == 1
            assert "Error:" in result.output


class TestWorkflowCommand:
    """Tests for 'supervisor workflow' command."""

    def test_workflow_command_basic(self, cli_runner, mocker):
        """Workflow command executes feature workflow."""
        # Mock all required components
        mocker.patch("supervisor.cli.Database")
        mocker.patch("supervisor.cli._load_limits_config", return_value=({}, 300.0, 3600.0))
        mocker.patch("supervisor.cli._load_approval_config", return_value=None)

        # Mock ExecutionEngine to avoid git repo requirement
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")
        mock_engine_instance = mock_engine.return_value

        mock_coordinator = mocker.patch("supervisor.core.workflow.WorkflowCoordinator")
        mock_instance = mock_coordinator.return_value
        mock_instance.run_implementation.return_value = None

        with cli_runner.isolated_filesystem():
            # Create .supervisor directory for database
            Path(".supervisor").mkdir()
            Path(".supervisor/state.db").touch()

            result = cli_runner.invoke(main, ["workflow", "feat-123"])
            assert result.exit_code == 0
            assert "Workflow complete!" in result.output

    def test_workflow_command_with_tui(self, cli_runner, mocker):
        """Workflow command can run with TUI mode."""
        mocker.patch("supervisor.cli.Database")
        mocker.patch("supervisor.cli._load_limits_config", return_value=({}, 300.0, 3600.0))
        mocker.patch("supervisor.cli._load_approval_config", return_value=None)
        mocker.patch("supervisor.cli.ExecutionEngine")
        mocker.patch("supervisor.core.interaction.InteractionBridge")

        mock_tui = mocker.patch("supervisor.tui.app.SupervisorTUI")
        mock_tui_instance = mock_tui.return_value
        mock_tui_instance.run_with_workflow.return_value = None

        with cli_runner.isolated_filesystem():
            Path(".supervisor").mkdir()
            Path(".supervisor/state.db").touch()

            result = cli_runner.invoke(main, ["workflow", "feat-123", "--tui"])
            assert result.exit_code == 0

            # Verify TUI was created
            mock_tui.assert_called_once()

    def test_workflow_command_parallel_flag(self, cli_runner, mocker):
        """Workflow command respects parallel/sequential flag."""
        mocker.patch("supervisor.cli.Database")
        mocker.patch("supervisor.cli._load_limits_config", return_value=({}, 300.0, 3600.0))
        mocker.patch("supervisor.cli._load_approval_config", return_value=None)
        mocker.patch("supervisor.cli.ExecutionEngine")

        mock_coordinator = mocker.patch("supervisor.core.workflow.WorkflowCoordinator")
        mock_instance = mock_coordinator.return_value
        mock_instance.run_implementation.return_value = None

        with cli_runner.isolated_filesystem():
            Path(".supervisor").mkdir()
            Path(".supervisor/state.db").touch()

            # Test parallel (default)
            result = cli_runner.invoke(main, ["workflow", "feat-123", "--parallel"])
            assert result.exit_code == 0
            mock_instance.run_implementation.assert_called_with("feat-123", parallel=True)

            # Test sequential
            result = cli_runner.invoke(main, ["workflow", "feat-123", "--sequential"])
            assert result.exit_code == 0
            mock_instance.run_implementation.assert_called_with("feat-123", parallel=False)


class TestMetricsCommand:
    """Tests for 'supervisor metrics' command."""

    def test_metrics_command_basic(self, cli_runner, mocker):
        """Metrics command displays performance data."""
        mock_db = mocker.patch("supervisor.cli.Database")
        mock_aggregator = mocker.patch("supervisor.cli.MetricsAggregator")
        mock_dashboard = mocker.patch("supervisor.cli.MetricsDashboard")
        mock_dashboard_instance = mock_dashboard.return_value

        with cli_runner.isolated_filesystem():
            Path(".supervisor").mkdir()
            Path(".supervisor/state.db").touch()

            result = cli_runner.invoke(main, ["metrics"])
            assert result.exit_code == 0

            # Verify dashboard was shown
            mock_dashboard_instance.show.assert_called_once()

    def test_metrics_command_with_days(self, cli_runner, mocker):
        """Metrics command accepts days parameter."""
        mocker.patch("supervisor.cli.Database")
        mocker.patch("supervisor.cli.MetricsAggregator")
        mock_dashboard = mocker.patch("supervisor.cli.MetricsDashboard")
        mock_dashboard_instance = mock_dashboard.return_value

        with cli_runner.isolated_filesystem():
            Path(".supervisor").mkdir()
            Path(".supervisor/state.db").touch()

            result = cli_runner.invoke(main, ["metrics", "--days", "7"])
            assert result.exit_code == 0

            # Verify days parameter was passed
            mock_dashboard_instance.show.assert_called_once_with(days=7)

    def test_metrics_command_no_database(self, cli_runner):
        """Metrics command handles missing database gracefully."""
        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["metrics"])
            assert result.exit_code == 0
            assert "No supervisor database found" in result.output

    def test_metrics_command_live_mode_not_implemented(self, cli_runner, mocker):
        """Metrics command shows message for unimplemented live mode."""
        mocker.patch("supervisor.cli.Database")
        mocker.patch("supervisor.cli.MetricsAggregator")
        mock_dashboard = mocker.patch("supervisor.cli.MetricsDashboard")
        mock_dashboard_instance = mock_dashboard.return_value

        with cli_runner.isolated_filesystem():
            Path(".supervisor").mkdir()
            Path(".supervisor/state.db").touch()

            result = cli_runner.invoke(main, ["metrics", "--live"])
            assert result.exit_code == 0
            assert "not yet implemented" in result.output.lower()


class TestRolesCommand:
    """Tests for 'supervisor roles' command."""

    def test_roles_command_lists_available_roles(self, cli_runner, mocker):
        """Roles command lists all available roles."""
        mock_loader = mocker.patch("supervisor.cli.RoleLoader")
        mock_loader_instance = mock_loader.return_value

        # Mock available roles
        mock_loader_instance.list_available_roles.return_value = ["planner", "implementer", "reviewer"]

        # Mock load_role to return role configs
        def mock_load_role(name):
            return Mock(
                name=name,
                description=f"{name.capitalize()} role",
                cli="claude",
            )

        mock_loader_instance.load_role.side_effect = mock_load_role

        result = cli_runner.invoke(main, ["roles"])
        assert result.exit_code == 0
        assert "planner" in result.output.lower()
        assert "implementer" in result.output.lower()
        assert "reviewer" in result.output.lower()

    def test_roles_command_handles_load_errors(self, cli_runner, mocker):
        """Roles command handles individual role loading errors."""
        mock_loader = mocker.patch("supervisor.cli.RoleLoader")
        mock_loader_instance = mock_loader.return_value

        mock_loader_instance.list_available_roles.return_value = ["planner", "broken_role"]

        def mock_load_role(name):
            if name == "broken_role":
                raise Exception("Failed to load")
            return Mock(name=name, description="Test role", cli="claude")

        mock_loader_instance.load_role.side_effect = mock_load_role

        result = cli_runner.invoke(main, ["roles"])
        assert result.exit_code == 0
        assert "planner" in result.output.lower()
        assert "Error loading" in result.output


class TestStatusCommand:
    """Tests for 'supervisor status' command."""

    def test_status_command_basic(self, cli_runner, mocker):
        """Status command shows database information."""
        mocker.patch("supervisor.cli.Database")

        with cli_runner.isolated_filesystem():
            Path(".supervisor").mkdir()
            Path(".supervisor/state.db").touch()

            result = cli_runner.invoke(main, ["status"])
            assert result.exit_code == 0
            assert "Supervisor Status" in result.output
            assert "Database:" in result.output

    def test_status_command_with_workflow_id(self, cli_runner, mocker):
        """Status command accepts workflow ID filter."""
        mocker.patch("supervisor.cli.Database")

        with cli_runner.isolated_filesystem():
            Path(".supervisor").mkdir()
            Path(".supervisor/state.db").touch()

            result = cli_runner.invoke(main, ["status", "--workflow-id", "wf-123"])
            assert result.exit_code == 0
            assert "wf-123" in result.output

    def test_status_command_no_database(self, cli_runner):
        """Status command handles missing database."""
        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(main, ["status"])
            assert result.exit_code == 0
            assert "No supervisor database found" in result.output


class TestVersionCommand:
    """Tests for 'supervisor version' command."""

    def test_version_command(self, cli_runner, mocker):
        """Version command displays version information."""
        mocker.patch("supervisor.__version__", "0.1.0")

        result = cli_runner.invoke(main, ["version"])
        assert result.exit_code == 0
        assert "Supervisor" in result.output
        assert "0.1.0" in result.output


class TestConfigLoading:
    """Tests for configuration loading helper functions."""

    def test_load_approval_config_exists(self, cli_runner):
        """Loading approval config when file exists."""
        from supervisor.cli import _load_approval_config

        with cli_runner.isolated_filesystem():
            Path(".supervisor").mkdir()
            approval_config = {
                "approval": {
                    "auto_approve_low_risk": False,
                    "risk_threshold": "high",
                    "require_approval_for": ["deploy"],
                }
            }
            Path(".supervisor/approval.yaml").write_text(yaml.safe_dump(approval_config))

            policy = _load_approval_config(Path("."))
            assert policy is not None
            assert policy.auto_approve_low_risk is False
            assert policy.risk_threshold == "high"
            assert policy.require_approval_for == ["deploy"]

    def test_load_approval_config_missing(self, cli_runner):
        """Loading approval config when file doesn't exist."""
        from supervisor.cli import _load_approval_config

        with cli_runner.isolated_filesystem():
            policy = _load_approval_config(Path("."))
            assert policy is None

    def test_load_limits_config_with_values(self, cli_runner):
        """Loading limits config with custom values."""
        from supervisor.cli import _load_limits_config

        with cli_runner.isolated_filesystem():
            Path(".supervisor").mkdir()
            limits_config = {
                "role_timeouts": {"planner": 900, "implementer": 450},
                "component_timeout": 600,
                "workflow_timeout": 7200,
            }
            Path(".supervisor/limits.yaml").write_text(yaml.safe_dump(limits_config))

            role_timeouts, component_timeout, workflow_timeout = _load_limits_config(Path("."))
            assert role_timeouts == {"planner": 900, "implementer": 450}
            assert component_timeout == 600
            assert workflow_timeout == 7200

    def test_load_limits_config_defaults(self, cli_runner):
        """Loading limits config returns defaults when file missing."""
        from supervisor.cli import _load_limits_config

        with cli_runner.isolated_filesystem():
            role_timeouts, component_timeout, workflow_timeout = _load_limits_config(Path("."))
            assert role_timeouts == {}
            assert component_timeout == 300.0
            assert workflow_timeout == 3600.0


class TestCLIIntegration:
    """Integration tests for CLI command combinations."""

    def test_full_workflow_init_and_plan(self, cli_runner, mocker):
        """Integration: init followed by plan."""
        mock_engine = mocker.patch("supervisor.cli.ExecutionEngine")
        mock_instance = mock_engine.return_value
        mock_instance.run_plan.return_value = Mock(phases=[], risks=[])

        with cli_runner.isolated_filesystem():
            # Initialize
            result_init = cli_runner.invoke(main, ["init"])
            assert result_init.exit_code == 0

            # Verify .supervisor exists
            assert Path(".supervisor").exists()

            # Run plan
            result_plan = cli_runner.invoke(main, ["plan", "Test feature"])
            assert result_plan.exit_code == 0

    def test_help_text_for_all_commands(self, cli_runner):
        """All commands have help text."""
        commands = ["init", "plan", "run", "workflow", "metrics", "roles", "status", "version"]

        for cmd in commands:
            result = cli_runner.invoke(main, [cmd, "--help"])
            assert result.exit_code == 0
            assert "Usage:" in result.output or "Show" in result.output
