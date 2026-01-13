"""Tests for context packing and template rendering."""


import pytest

from supervisor.core.context import ALLOWED_TEMPLATES, ContextPacker
from supervisor.core.roles import RoleConfig


@pytest.fixture
def temp_repo(tmp_path):
    """Create temporary repository with test files."""
    # Create some test files
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    pass\n")
    (tmp_path / "src" / "utils.py").write_text("def helper():\n    return True\n")
    (tmp_path / "README.md").write_text("# Test Project\n")

    # Initialize git repo for git_diff tests
    (tmp_path / ".git").mkdir()

    return tmp_path


@pytest.fixture
def packer(temp_repo):
    """Create ContextPacker with temp repo."""
    return ContextPacker(temp_repo)


@pytest.fixture
def basic_role():
    """Create basic role config."""
    return RoleConfig(
        name="tester",
        description="Test role",
        cli="claude",
        flags=["-p"],
        system_prompt="You are a helpful tester.",
        context={"token_budget": 10000},
        gates=[],
        config={},
    )


class TestContextPacker:
    """Tests for ContextPacker core functionality."""

    def test_pack_context_includes_system_prompt(self, packer, basic_role):
        """System prompt is included in packed context."""
        result = packer.pack_context(basic_role, "Test task")
        assert "You are a helpful tester" in result

    def test_pack_context_includes_task(self, packer, basic_role):
        """Task description is included in packed context."""
        result = packer.pack_context(basic_role, "Implement feature X")
        assert "Implement feature X" in result

    def test_pack_target_files(self, packer, temp_repo, basic_role):
        """Target files are packed with high priority."""
        target = ["src/main.py"]
        result = packer.pack_context(basic_role, "Test task", target_files=target)
        assert "def main()" in result
        assert "Target Files" in result

    def test_path_traversal_rejected(self, packer, basic_role):
        """Path traversal attempts are rejected."""
        malicious_files = ["../etc/passwd"]
        result = packer.pack_context(basic_role, "Test", target_files=malicious_files)
        # Should not contain system files
        assert "/etc/passwd" not in result


class TestTokenBudget:
    """Tests for token budget enforcement."""

    def test_context_within_budget(self, packer, basic_role):
        """Context within budget is not truncated."""
        result = packer.pack_context(basic_role, "Short task")
        assert "[Context truncated" not in result

    def test_context_exceeds_budget_truncated(self, packer, temp_repo):
        """Context exceeding budget is truncated."""
        # Create a large file
        large_content = "x" * 50000
        (temp_repo / "large.txt").write_text(large_content)

        role = RoleConfig(
            name="test",
            description="Test",
            cli="claude",
            flags=[],
            system_prompt="Test",
            context={
                "token_budget": 100,  # Very small budget
                "include": ["large.txt"],
            },
            gates=[],
            config={},
        )

        result = packer.pack_context(role, "Task")
        # Either truncated or pruned
        assert len(result) < 50000


class TestTemplateRendering:
    """Tests for Jinja2 template rendering."""

    def test_allowed_template_renders(self, packer, basic_role):
        """Allowed templates render successfully."""
        # Render planning template
        result = packer.render_prompt(
            "planning.j2",
            basic_role,
            "Create a plan",
            context="Test context",
        )
        assert "You are a helpful tester" in result
        assert "Create a plan" in result

    def test_unknown_template_rejected(self, packer, basic_role):
        """Unknown templates are rejected."""
        with pytest.raises(ValueError, match="Unknown template"):
            packer.render_prompt(
                "malicious.j2",
                basic_role,
                "Task",
                context="",
            )

    def test_build_full_prompt(self, packer, basic_role):
        """build_full_prompt combines context and template."""
        result = packer.build_full_prompt(
            "planning.j2",
            basic_role,
            "Build feature X",
        )
        assert "You are a helpful tester" in result
        assert "Build feature X" in result


class TestSentinelHandling:
    """Tests for special sentinel pattern handling."""

    def test_git_diff_sentinel_not_file(self, packer, temp_repo):
        """git diff --cached is handled as sentinel, not file."""
        role = RoleConfig(
            name="reviewer",
            description="Review",
            cli="claude",
            flags=[],
            system_prompt="Review code",
            context={
                "include": ["git diff --cached", "$CHANGED_FILES"],
            },
            gates=[],
            config={},
        )

        # Should not try to read "git diff --cached" as a file
        result = packer.pack_context(role, "Review changes")
        # No error should occur, and no file named "git diff --cached" in output
        assert "Error reading file 'git diff --cached'" not in result


class TestAlwaysInclude:
    """Tests for always_include protected files."""

    def test_always_include_packed(self, packer, temp_repo, basic_role):
        """always_include files are included."""
        (temp_repo / "CRITICAL.md").write_text("# Critical info")

        role = RoleConfig(
            name="test",
            description="Test",
            cli="claude",
            flags=[],
            system_prompt="Test",
            context={
                "always_include": ["CRITICAL.md"],
            },
            gates=[],
            config={},
        )

        result = packer.pack_file_context(role)
        assert "Critical info" in result

    def test_always_include_never_pruned(self, packer, temp_repo):
        """always_include files survive budget pruning."""
        (temp_repo / "CRITICAL.md").write_text("# MUST KEEP")

        role = RoleConfig(
            name="test",
            description="Test",
            cli="claude",
            flags=[],
            system_prompt="Test",
            context={
                "always_include": ["CRITICAL.md"],
                "token_budget": 100,  # Very small
            },
            gates=[],
            config={},
        )

        result = packer.pack_file_context(role)
        # Should still contain critical file
        assert "MUST KEEP" in result


class TestSkipTargets:
    """Tests for skip_targets deduplication."""

    def test_target_files_not_duplicated(self, packer, temp_repo):
        """Target files packed once even if in include patterns."""
        role = RoleConfig(
            name="test",
            description="Test",
            cli="claude",
            flags=[],
            system_prompt="Test",
            context={
                "include": ["$TARGET"],  # Would duplicate target_files
            },
            gates=[],
            config={},
        )

        target_files = ["src/main.py"]
        result = packer.pack_file_context(role, target_files)

        # Count occurrences of "def main()"
        count = result.count("def main()")
        assert count == 1  # Should appear exactly once


class TestFilters:
    """Tests for Jinja2 filters."""

    def test_truncate_lines_filter(self, packer):
        """truncate_lines filter limits line count."""
        text = "line1\nline2\nline3\nline4\nline5"
        result = packer._truncate_lines(text, 3)
        assert "line1" in result
        assert "line3" in result
        assert "line4" not in result
        assert "Truncated 2 lines" in result

    def test_format_diff_filter(self, packer):
        """format_diff filter wraps in code block."""
        diff = "+added line\n-removed line"
        result = packer._format_diff(diff)
        assert "```diff" in result
        assert "+added line" in result


class TestAllowedTemplates:
    """Tests for template allowlist."""

    def test_base_template_allowed(self):
        """Base template is in allowlist."""
        assert "_base.j2" in ALLOWED_TEMPLATES

    def test_planning_template_allowed(self):
        """Planning template is in allowlist."""
        assert "planning.j2" in ALLOWED_TEMPLATES

    def test_implement_template_allowed(self):
        """Implement template is in allowlist."""
        assert "implement.j2" in ALLOWED_TEMPLATES

    def test_review_template_allowed(self):
        """Review template is in allowlist."""
        assert "review_strict.j2" in ALLOWED_TEMPLATES
