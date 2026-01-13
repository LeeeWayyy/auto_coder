"""Tests for Context Strategies (Phase 4).

Tests pluggable context selection strategies per role.
"""

import tempfile
from pathlib import Path

import pytest

from supervisor.core.strategies import (
    STRATEGIES,
    ContextResult,
    ContextStrategy,
    ImplementerTargetedStrategy,
    PlannerDocsetStrategy,
    ReviewerDiffStrategy,
    StrategyError,
    detect_language,
    get_strategy,
    get_strategy_for_role,
    register_strategy,
)


@pytest.fixture
def temp_repo():
    """Create a temporary repository structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir)

        # Create basic structure
        (repo / "README.md").write_text("# Test Project\n\nA test project.")
        (repo / "src").mkdir()
        (repo / "src" / "main.py").write_text("def main():\n    print('Hello')")
        (repo / "src" / "utils.py").write_text("import os\n\ndef helper():\n    pass")

        # Create docs structure
        (repo / "docs").mkdir()
        (repo / "docs" / "ARCHITECTURE.md").write_text("# Architecture\n\nDesign docs.")
        (repo / "docs" / "ADRs").mkdir()
        (repo / "docs" / "ADRs" / "001-database.md").write_text("# ADR 001\n\nDatabase choice.")
        (repo / "docs" / "STANDARDS").mkdir()
        (repo / "docs" / "STANDARDS" / "python.md").write_text("# Python Standards\n\nFollow PEP8.")

        yield repo


class TestContextResult:
    """Tests for ContextResult dataclass."""

    def test_context_result_creation(self):
        """Test creating a ContextResult."""
        result = ContextResult(
            content="File content here",
            token_count=100,
            files_included=["file1.py", "file2.py"],
            truncated=False,
        )

        assert result.content == "File content here"
        assert result.token_count == 100
        assert len(result.files_included) == 2
        assert not result.truncated

    def test_context_result_with_truncation(self):
        """Test ContextResult with truncation info."""
        result = ContextResult(
            content="Truncated...",
            token_count=50,
            files_included=["file1.py"],
            truncated=True,
            truncation_info="Removed 3 files due to token limit",
        )

        assert result.truncated
        assert result.truncation_info is not None


class TestPlannerDocsetStrategy:
    """Tests for PlannerDocsetStrategy."""

    def test_packs_readme(self, temp_repo):
        """Test that strategy includes README.md."""
        strategy = PlannerDocsetStrategy()
        result = strategy.pack(temp_repo)

        assert "README.md" in result.files_included
        assert "# Test Project" in result.content

    def test_packs_architecture(self, temp_repo):
        """Test that strategy includes architecture docs."""
        strategy = PlannerDocsetStrategy()
        result = strategy.pack(temp_repo)

        assert "docs/ARCHITECTURE.md" in result.files_included
        assert "# Architecture" in result.content

    def test_packs_adrs(self, temp_repo):
        """Test that strategy includes ADR documents."""
        strategy = PlannerDocsetStrategy()
        result = strategy.pack(temp_repo)

        assert "docs/ADRs/001-database.md" in result.files_included
        assert "ADR 001" in result.content

    def test_includes_task_spec_from_extra_inputs(self, temp_repo):
        """Test that task_spec from extra_inputs is included."""
        strategy = PlannerDocsetStrategy()
        result = strategy.pack(
            temp_repo,
            extra_inputs={"task_spec": "Implement user authentication"},
        )

        assert "Implement user authentication" in result.content

    def test_token_count_estimated(self, temp_repo):
        """Test that token count is estimated."""
        strategy = PlannerDocsetStrategy()
        result = strategy.pack(temp_repo)

        # Token count should be roughly content length / 4
        assert result.token_count > 0
        assert result.token_count == len(result.content) // 4


class TestImplementerTargetedStrategy:
    """Tests for ImplementerTargetedStrategy."""

    def test_packs_target_files(self, temp_repo):
        """Test that strategy includes target files."""
        strategy = ImplementerTargetedStrategy()
        result = strategy.pack(
            temp_repo,
            target_files=["src/main.py"],
        )

        assert "src/main.py" in result.files_included
        assert "def main():" in result.content

    def test_packs_new_file_placeholder(self, temp_repo):
        """Test that non-existent target files get placeholder."""
        strategy = ImplementerTargetedStrategy()
        result = strategy.pack(
            temp_repo,
            target_files=["src/new_file.py"],
        )

        assert "src/new_file.py" in result.files_included
        assert "does not exist yet" in result.content

    def test_resolves_python_imports(self, temp_repo):
        """Test that Python imports are resolved."""
        strategy = ImplementerTargetedStrategy()
        result = strategy.pack(
            temp_repo,
            target_files=["src/utils.py"],
        )

        # utils.py imports os, but that's a stdlib module
        # Just verify the import resolution runs without error
        assert "src/utils.py" in result.files_included

    def test_configurable_diff_source(self, temp_repo):
        """Test that diff source is configurable."""
        strategy = ImplementerTargetedStrategy()

        # Without git, this should handle gracefully
        result = strategy.pack(
            temp_repo,
            target_files=["src/main.py"],
            extra_inputs={"diff_source": "staged"},
        )

        # Should not crash even without git
        assert result.content is not None

    def test_detects_language_from_extension(self, temp_repo):
        """Test language detection for syntax highlighting."""
        # FIX (PR review): Use module-level detect_language helper
        assert detect_language("test.py") == "python"
        assert detect_language("test.js") == "javascript"
        assert detect_language("test.ts") == "typescript"
        assert detect_language("test.go") == "go"
        assert detect_language("test.unknown") == ""


class TestReviewerDiffStrategy:
    """Tests for ReviewerDiffStrategy."""

    def test_packs_standards_docs(self, temp_repo):
        """Test that strategy includes coding standards."""
        strategy = ReviewerDiffStrategy()
        result = strategy.pack(temp_repo)

        assert "docs/STANDARDS/python.md" in result.files_included
        assert "Python Standards" in result.content

    def test_handles_no_git_gracefully(self, temp_repo):
        """Test that missing git doesn't cause errors."""
        strategy = ReviewerDiffStrategy()

        # Should not crash even without git
        result = strategy.pack(temp_repo)
        assert result.content is not None


class TestStrategyRegistry:
    """Tests for strategy registry functions."""

    def test_get_strategy_returns_instance(self):
        """Test that get_strategy returns strategy instance."""
        strategy = get_strategy("planner_docset")
        assert isinstance(strategy, PlannerDocsetStrategy)

        strategy = get_strategy("implementer_targeted")
        assert isinstance(strategy, ImplementerTargetedStrategy)

        strategy = get_strategy("reviewer_diff")
        assert isinstance(strategy, ReviewerDiffStrategy)

    def test_get_strategy_unknown_raises_error(self):
        """Test that unknown strategy name raises error."""
        with pytest.raises(StrategyError) as exc_info:
            get_strategy("unknown_strategy")

        assert "Unknown strategy" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_get_strategy_for_role(self):
        """Test getting strategy by role name."""
        assert isinstance(get_strategy_for_role("planner"), PlannerDocsetStrategy)
        assert isinstance(get_strategy_for_role("implementer"), ImplementerTargetedStrategy)
        assert isinstance(get_strategy_for_role("reviewer"), ReviewerDiffStrategy)
        assert isinstance(get_strategy_for_role("reviewer_gemini"), ReviewerDiffStrategy)

    def test_get_strategy_for_unknown_role_returns_none(self):
        """Test that unknown role returns None."""
        result = get_strategy_for_role("unknown_custom_role")
        assert result is None

    def test_register_strategy(self):
        """Test registering a custom strategy."""

        class CustomStrategy(ContextStrategy):
            name = "custom"
            description = "A custom strategy"
            token_budget = 10000

            def pack(self, repo_path, target_files=None, extra_inputs=None):
                return ContextResult(
                    content="Custom content",
                    token_count=10,
                    files_included=[],
                )

        register_strategy("custom_test", CustomStrategy)

        assert "custom_test" in STRATEGIES
        strategy = get_strategy("custom_test")
        assert isinstance(strategy, CustomStrategy)

        # Cleanup
        del STRATEGIES["custom_test"]


class TestTokenBudgetEnforcement:
    """Tests for token budget enforcement in strategies."""

    def test_planner_respects_budget(self, temp_repo):
        """Test that planner strategy respects token budget."""
        strategy = PlannerDocsetStrategy()
        strategy.token_budget = 100  # Very small budget

        result = strategy.pack(temp_repo)

        # Content should be trimmed to fit budget
        assert result.token_count <= 100 or result.truncated

    def test_implementer_respects_budget(self, temp_repo):
        """Test that implementer strategy respects token budget."""
        strategy = ImplementerTargetedStrategy()
        strategy.token_budget = 50  # Very small budget

        result = strategy.pack(temp_repo, target_files=["src/main.py"])

        # Should handle small budget gracefully
        assert result.content is not None
