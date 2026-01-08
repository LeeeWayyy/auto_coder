"""Tests for gate configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from supervisor.core.gates import (
    CircularDependencyError,
    GateConfig,
    GateConfigError,
    GateLoader,
    GateNotFoundError,
    GateResult,
    GateSeverity,
    GateStatus,
    PACKAGE_DIR,
)


@pytest.fixture
def temp_gate_file(tmp_path: Path) -> Path:
    """Create a temp gates.yaml path."""
    return tmp_path / "gates.yaml"


def write_gates(path: Path, gates: dict) -> None:
    """Write gate config to a YAML file."""
    path.write_text(yaml.safe_dump({"gates": gates}))


class TestGateConfig:
    """Tests for GateConfig and GateResult dataclasses."""

    def test_gate_config_defaults(self):
        """Default values are correct."""
        config = GateConfig(name="test", command=["pytest", "-q"])
        assert config.description == ""
        assert config.timeout == 300
        assert config.depends_on == []
        assert config.severity == GateSeverity.ERROR
        assert config.env == {}
        assert config.working_dir is None
        assert config.parallel_safe is False
        assert config.cache is True
        assert config.cache_inputs == []
        assert config.force_hash_large_cache_inputs is False
        assert config.skip_on_dependency_failure is True
        assert config.allowed_writes == []
        assert config.allow_shell is False

    def test_gate_config_all_fields(self):
        """All fields can be set explicitly."""
        config = GateConfig(
            name="lint",
            command=["ruff", "check", "."],
            description="Lint",
            timeout=120,
            depends_on=["format"],
            severity=GateSeverity.WARNING,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
            working_dir="src",
            parallel_safe=True,
            cache=False,
            cache_inputs=["build/**"],
            force_hash_large_cache_inputs=True,
            skip_on_dependency_failure=False,
            allowed_writes=[".ruff_cache/**"],
            allow_shell=True,
        )
        assert config.name == "lint"
        assert config.command == ["ruff", "check", "."]
        assert config.description == "Lint"
        assert config.timeout == 120
        assert config.depends_on == ["format"]
        assert config.severity == GateSeverity.WARNING
        assert config.env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert config.working_dir == "src"
        assert config.parallel_safe is True
        assert config.cache is False
        assert config.cache_inputs == ["build/**"]
        assert config.force_hash_large_cache_inputs is True
        assert config.skip_on_dependency_failure is False
        assert config.allowed_writes == [".ruff_cache/**"]
        assert config.allow_shell is True

    def test_gate_result_properties(self):
        """GateResult status properties reflect status."""
        passed = GateResult(
            gate_name="test",
            status=GateStatus.PASSED,
            output="ok",
            duration_seconds=1.0,
        )
        failed = GateResult(
            gate_name="test",
            status=GateStatus.FAILED,
            output="fail",
            duration_seconds=1.0,
        )
        skipped = GateResult(
            gate_name="test",
            status=GateStatus.SKIPPED,
            output="skip",
            duration_seconds=0.0,
        )
        assert passed.passed is True
        assert passed.failed is False
        assert passed.skipped is False
        assert failed.passed is False
        assert failed.failed is True
        assert failed.skipped is False
        assert skipped.passed is False
        assert skipped.failed is False
        assert skipped.skipped is True


class TestGateLoader:
    """Tests for GateLoader."""

    def test_load_default_config(self, tmp_path, monkeypatch):
        """Loading from default config works."""
        default_path = PACKAGE_DIR / "config" / "gates.yaml"
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [default_path])

        loader = GateLoader(worktree_path=tmp_path)
        gates = loader.load_gates()
        assert "test" in gates
        assert gates["test"].command[0] == "pytest"

    def test_load_temp_yaml(self, tmp_path, temp_gate_file, monkeypatch):
        """Loading from temp YAML files works."""
        write_gates(temp_gate_file, {"lint": {"command": ["ruff", "check", "."]}})
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        gates = loader.load_gates()
        assert gates["lint"].command == ["ruff", "check", "."]

    def test_merge_precedence(self, tmp_path, monkeypatch):
        """Higher-precedence configs override lower-precedence ones."""
        package_path = tmp_path / "package.yaml"
        user_path = tmp_path / "user.yaml"
        project_path = tmp_path / ".supervisor" / "gates.yaml"
        project_path.parent.mkdir(parents=True)

        write_gates(package_path, {"lint": {"command": ["ruff", "check", "."]}})
        write_gates(user_path, {"lint": {"command": ["flake8", "."]}})
        write_gates(project_path, {"lint": {"command": ["pylint", "src"]}})

        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [user_path, package_path])

        loader = GateLoader(worktree_path=tmp_path, allow_project_gates=True)
        gates = loader.load_gates()
        assert gates["lint"].command == ["pylint", "src"]

    def test_get_gate(self, tmp_path, temp_gate_file, monkeypatch):
        """get_gate returns the expected config."""
        write_gates(temp_gate_file, {"test": {"command": ["pytest"]}})
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        gate = loader.get_gate("test")
        assert gate.name == "test"
        assert gate.command == ["pytest"]

    def test_get_gate_unknown(self, tmp_path, temp_gate_file, monkeypatch):
        """get_gate raises GateNotFoundError for unknown gate."""
        write_gates(temp_gate_file, {"test": {"command": ["pytest"]}})
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(GateNotFoundError):
            loader.get_gate("missing")

    def test_invalid_string_command_rejected(self, tmp_path, temp_gate_file, monkeypatch):
        """String commands are rejected."""
        temp_gate_file.write_text(
            yaml.safe_dump({"gates": {"bad": {"command": "pytest -q"}}})
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(GateConfigError, match="command must be a list"):
            loader.load_gates()

    def test_invalid_empty_command_rejected(self, tmp_path, temp_gate_file, monkeypatch):
        """Empty command list is rejected."""
        write_gates(temp_gate_file, {"bad": {"command": []}})
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(GateConfigError, match="command cannot be empty"):
            loader.load_gates()

    def test_shell_binary_requires_allow_shell(self, tmp_path, temp_gate_file, monkeypatch):
        """Shell binaries are rejected unless allow_shell=true."""
        write_gates(
            temp_gate_file,
            {"bad": {"command": ["bash", "-c", "echo hi"]}},
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(GateConfigError, match="allow_shell is not set"):
            loader.load_gates()

    def test_path_traversal_in_allowed_writes_rejected(
        self, tmp_path, temp_gate_file, monkeypatch
    ):
        """Path traversal patterns in allowed_writes are rejected."""
        write_gates(
            temp_gate_file,
            {"bad": {"command": ["pytest"], "allowed_writes": ["../oops"]}},
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(GateConfigError, match="path traversal"):
            loader.load_gates()

    def test_invalid_gate_name_rejected(self, tmp_path, temp_gate_file, monkeypatch):
        """Invalid gate names are rejected."""
        write_gates(temp_gate_file, {"../bad": {"command": ["pytest"]}})
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(GateConfigError, match="Invalid gate name"):
            loader.load_gates()


class TestDependencyResolution:
    """Tests for dependency resolution ordering."""

    def test_resolve_execution_order(self, tmp_path, temp_gate_file, monkeypatch):
        """Dependencies resolve to correct order."""
        write_gates(
            temp_gate_file,
            {
                "lint": {"command": ["ruff", "check", "."]},
                "type_check": {"command": ["mypy", "."], "depends_on": ["lint"]},
            },
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        order = loader.resolve_execution_order(["type_check"])
        assert order == ["lint", "type_check"]

    def test_transitive_dependencies_included(self, tmp_path, temp_gate_file, monkeypatch):
        """Transitive dependencies are included."""
        write_gates(
            temp_gate_file,
            {
                "a": {"command": ["echo", "a"]},
                "b": {"command": ["echo", "b"], "depends_on": ["a"]},
                "c": {"command": ["echo", "c"], "depends_on": ["b"]},
            },
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        order = loader.resolve_execution_order(["c"])
        assert order == ["a", "b", "c"]

    def test_circular_dependency_raises(self, tmp_path, temp_gate_file, monkeypatch):
        """Circular dependencies raise CircularDependencyError."""
        write_gates(
            temp_gate_file,
            {
                "a": {"command": ["echo", "a"], "depends_on": ["b"]},
                "b": {"command": ["echo", "b"], "depends_on": ["a"]},
            },
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(CircularDependencyError):
            loader.resolve_execution_order(["a"])

    def test_deterministic_order(self, tmp_path, temp_gate_file, monkeypatch):
        """Execution order is deterministic for the same input."""
        write_gates(
            temp_gate_file,
            {
                "a": {"command": ["echo", "a"]},
                "b": {"command": ["echo", "b"]},
                "c": {"command": ["echo", "c"], "depends_on": ["a", "b"]},
            },
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        order1 = loader.resolve_execution_order(["c"])
        order2 = loader.resolve_execution_order(["c"])
        assert order1 == order2


class TestSecurityValidation:
    """Tests for security validation helpers."""

    def test_shell_detection_through_wrappers(self, tmp_path, temp_gate_file, monkeypatch):
        """Shell detection works through wrapper chains."""
        # Test with env wrapper (env bash -c "...")
        write_gates(
            temp_gate_file,
            {"bad": {"command": ["env", "bash", "-c", "echo hi"]}},
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(GateConfigError, match="allow_shell is not set"):
            loader.load_gates()

    def test_shell_detection_direct(self, tmp_path, temp_gate_file, monkeypatch):
        """Direct shell binary is detected."""
        write_gates(
            temp_gate_file,
            {"bad": {"command": ["bash", "-c", "echo hi"]}},
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(GateConfigError, match="allow_shell is not set"):
            loader.load_gates()

    def test_env_denylist_bypass_detected(self, tmp_path, temp_gate_file, monkeypatch):
        """Env denylist bypass is detected."""
        write_gates(
            temp_gate_file,
            {
                "bad": {
                    "command": ["command", "env", "PATH=/tmp", "echo", "hi"]
                }
            },
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(GateConfigError, match="denylisted"):
            loader.load_gates()

    def test_path_traversal_in_cache_inputs_rejected(
        self, tmp_path, temp_gate_file, monkeypatch
    ):
        """Path traversal patterns in cache_inputs are rejected."""
        write_gates(
            temp_gate_file,
            {"bad": {"command": ["pytest"], "cache_inputs": ["..\\oops"]}},
        )
        monkeypatch.setattr(GateLoader, "STATIC_SEARCH_PATHS", [temp_gate_file])

        loader = GateLoader(worktree_path=tmp_path)
        with pytest.raises(GateConfigError, match="path traversal"):
            loader.load_gates()
