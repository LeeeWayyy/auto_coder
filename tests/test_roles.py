"""Tests for role configuration loading and validation."""

import pytest
import yaml

from supervisor.core.roles import (
    RoleConfig,
    RoleCycleError,
    RoleLoader,
    RoleNotFoundError,
    RoleValidationError,
)


class TestRoleConfig:
    """Tests for RoleConfig dataclass."""

    def test_token_budget_default(self):
        """Token budget defaults to 20000."""
        role = RoleConfig(
            name="test",
            description="Test role",
            cli="claude:sonnet",
            flags=[],
            system_prompt="Test prompt",
            context={},
            gates=[],
            config={},
        )
        assert role.token_budget == 20000

    def test_token_budget_custom(self):
        """Token budget can be customized via context."""
        role = RoleConfig(
            name="test",
            description="Test role",
            cli="claude:sonnet",
            flags=[],
            system_prompt="Test prompt",
            context={"token_budget": 10000},
            gates=[],
            config={},
        )
        assert role.token_budget == 10000

    def test_base_role_none_for_base_roles(self):
        """Base roles have base_role=None."""
        role = RoleConfig(
            name="planner",
            description="Planning role",
            cli="claude:sonnet",
            flags=[],
            system_prompt="Plan tasks",
            context={},
            gates=[],
            config={},
            base_role=None,
        )
        assert role.base_role is None

    def test_base_role_set_for_overlays(self):
        """Overlay roles have base_role set to root of extends chain."""
        role = RoleConfig(
            name="python-planner",
            description="Python planning role",
            cli="claude:sonnet",
            flags=[],
            system_prompt="Plan Python tasks",
            context={},
            gates=[],
            config={},
            extends="planner",
            base_role="planner",
        )
        assert role.base_role == "planner"


class TestRoleLoader:
    """Tests for RoleLoader."""

    @pytest.fixture
    def temp_roles_dir(self, tmp_path):
        """Create temporary roles directory with test roles."""
        roles_dir = tmp_path / "roles"
        roles_dir.mkdir()
        return roles_dir

    @pytest.fixture
    def loader_with_temp_roles(self, temp_roles_dir):
        """Create RoleLoader with temp roles directory."""
        return RoleLoader(search_paths=[temp_roles_dir])

    def test_load_valid_role(self, temp_roles_dir, loader_with_temp_roles):
        """Valid role loads successfully."""
        role_config = {
            "name": "testrole",
            "description": "A test role",
            "cli": "claude:sonnet",
            "system_prompt": "You are a test assistant.",
        }
        (temp_roles_dir / "testrole.yaml").write_text(yaml.dump(role_config))

        role = loader_with_temp_roles.load_role("testrole")
        assert role.name == "testrole"
        assert role.cli == "claude:sonnet"
        assert role.system_prompt == "You are a test assistant."

    def test_load_role_with_inheritance(self, temp_roles_dir, loader_with_temp_roles):
        """Role with extends inherits from parent."""
        # Create base role
        base_config = {
            "name": "base",
            "description": "Base role",
            "cli": "claude:sonnet",
            "system_prompt": "Base system prompt.",
            "flags": ["-p"],
        }
        (temp_roles_dir / "base.yaml").write_text(yaml.dump(base_config))

        # Create child role
        child_config = {
            "name": "child",
            "description": "Child role",
            "extends": "base",
            "system_prompt_additions": "\n\nAdditional instructions.",
        }
        (temp_roles_dir / "child.yaml").write_text(yaml.dump(child_config))

        role = loader_with_temp_roles.load_role("child")
        assert role.name == "child"
        assert role.cli == "claude:sonnet"  # Inherited
        assert "Base system prompt." in role.system_prompt
        assert "Additional instructions." in role.system_prompt
        assert role.flags == ["-p"]  # Inherited
        assert role.base_role == "base"  # Root of extends chain

    def test_multi_level_inheritance_base_role(self, temp_roles_dir, loader_with_temp_roles):
        """Multi-level inheritance sets base_role to root."""
        # Create three-level hierarchy: grandparent -> parent -> child
        grandparent = {
            "name": "grandparent",
            "description": "Grandparent role",
            "cli": "claude:sonnet",
            "system_prompt": "Grandparent prompt.",
        }
        (temp_roles_dir / "grandparent.yaml").write_text(yaml.dump(grandparent))

        parent = {
            "name": "parent",
            "description": "Parent role",
            "extends": "grandparent",
            "system_prompt_additions": "\n\nParent additions.",
        }
        (temp_roles_dir / "parent.yaml").write_text(yaml.dump(parent))

        child = {
            "name": "child",
            "description": "Child role",
            "extends": "parent",
            "system_prompt_additions": "\n\nChild additions.",
        }
        (temp_roles_dir / "child.yaml").write_text(yaml.dump(child))

        role = loader_with_temp_roles.load_role("child")
        # base_role should be grandparent (root of chain), not parent
        assert role.base_role == "grandparent"

    def test_invalid_role_name_rejected(self, loader_with_temp_roles):
        """Invalid role names are rejected."""
        with pytest.raises(RoleValidationError, match="Invalid role name"):
            loader_with_temp_roles.load_role("../etc/passwd")

    def test_cycle_detection(self, temp_roles_dir, loader_with_temp_roles):
        """Circular inheritance is detected."""
        # Create roles that form a cycle
        role_a = {
            "name": "role_a",
            "description": "Role A",
            "extends": "role_b",
            "cli": "claude:sonnet",
            "system_prompt": "A",
        }
        role_b = {
            "name": "role_b",
            "description": "Role B",
            "extends": "role_a",
            "cli": "claude:sonnet",
            "system_prompt": "B",
        }
        (temp_roles_dir / "role_a.yaml").write_text(yaml.dump(role_a))
        (temp_roles_dir / "role_b.yaml").write_text(yaml.dump(role_b))

        with pytest.raises(RoleCycleError, match="Circular"):
            loader_with_temp_roles.load_role("role_a")

    def test_role_not_found(self, loader_with_temp_roles):
        """Missing role raises RoleNotFoundError."""
        with pytest.raises(RoleNotFoundError):
            loader_with_temp_roles.load_role("nonexistent")

    def test_schema_validation_invalid_cli(self, temp_roles_dir, loader_with_temp_roles):
        """Invalid CLI value is rejected by schema."""
        role_config = {
            "name": "invalid",
            "description": "Invalid role",
            "cli": "invalid_cli",  # Not in enum
            "system_prompt": "Test",
        }
        (temp_roles_dir / "invalid.yaml").write_text(yaml.dump(role_config))

        with pytest.raises(RoleValidationError, match="Schema validation failed"):
            loader_with_temp_roles.load_role("invalid")

    def test_pre_merge_type_validation(self, temp_roles_dir, loader_with_temp_roles):
        """Wrong types are caught before merge."""
        role_config = {
            "name": "badtype",
            "description": "Bad type role",
            "cli": "claude:sonnet",
            "system_prompt": "Test",
            "flags": "not-a-list",  # Should be list
        }
        (temp_roles_dir / "badtype.yaml").write_text(yaml.dump(role_config))

        with pytest.raises(RoleValidationError, match="must be list"):
            loader_with_temp_roles.load_role("badtype")

    def test_empty_yaml_rejected(self, temp_roles_dir, loader_with_temp_roles):
        """Empty YAML file is rejected."""
        (temp_roles_dir / "empty.yaml").write_text("")

        with pytest.raises(RoleValidationError, match="Empty"):
            loader_with_temp_roles.load_role("empty")

    def test_invalid_yaml_rejected(self, temp_roles_dir, loader_with_temp_roles):
        """Invalid YAML syntax is rejected."""
        (temp_roles_dir / "invalid.yaml").write_text("{ invalid yaml: [")

        with pytest.raises(RoleValidationError, match="Invalid YAML"):
            loader_with_temp_roles.load_role("invalid")
