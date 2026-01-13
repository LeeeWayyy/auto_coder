"""Tests for Parallel Review execution (Phase 4).

Tests multi-model parallel review and result aggregation.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from pydantic import BaseModel

from supervisor.core.parallel import (
    AggregatedReviewResult,
    ParallelExecutionError,
    ParallelReviewer,
    ReviewResult,
)


class MockReviewOutput(BaseModel):
    """Mock review output for testing."""

    status: str = "APPROVED"
    review_status: str = "APPROVED"
    issues: list[str] = []


class MockRoleConfig:
    """Mock role config for testing."""

    def __init__(self, cli: str = "claude"):
        self.cli = cli


class TestReviewResult:
    """Tests for ReviewResult dataclass."""

    def test_success_result(self):
        """Test successful review result."""
        result = ReviewResult(
            role_name="reviewer_gemini",
            cli="gemini",
            output=MockReviewOutput(status="APPROVED"),
            duration_seconds=5.0,
            success=True,
        )

        assert result.success
        assert result.error is None
        assert result.output is not None

    def test_failure_result(self):
        """Test failed review result."""
        result = ReviewResult(
            role_name="reviewer_codex",
            cli="codex",
            output=None,
            duration_seconds=2.0,
            success=False,
            error="Connection timeout",
        )

        assert not result.success
        assert result.error == "Connection timeout"
        assert result.output is None


class TestAggregatedReviewResult:
    """Tests for AggregatedReviewResult."""

    def test_all_approved(self):
        """Test all_approved property."""
        results = [
            ReviewResult("r1", "claude", MockReviewOutput(), 1.0, True),
            ReviewResult("r2", "gemini", MockReviewOutput(), 1.0, True),
        ]

        agg = AggregatedReviewResult(
            results=results,
            approved=True,
            approval_policy="ALL_APPROVED",
            summary="All approved",
        )

        assert agg.all_approved
        assert agg.any_approved
        assert agg.majority_approved

    def test_any_approved(self):
        """Test any_approved property."""
        results = [
            ReviewResult("r1", "claude", MockReviewOutput(), 1.0, True),
            ReviewResult("r2", "gemini", None, 1.0, False, "Failed"),
        ]

        agg = AggregatedReviewResult(
            results=results,
            approved=True,
            approval_policy="ANY_APPROVED",
            summary="One approved",
        )

        assert not agg.all_approved
        assert agg.any_approved
        # 1 of 2 = 0.5, which is NOT > 0.5, so majority_approved is False
        assert not agg.majority_approved

    def test_none_approved(self):
        """Test when no reviewers approve."""
        results = [
            ReviewResult("r1", "claude", None, 1.0, False, "Error 1"),
            ReviewResult("r2", "gemini", None, 1.0, False, "Error 2"),
        ]

        agg = AggregatedReviewResult(
            results=results,
            approved=False,
            approval_policy="ALL_APPROVED",
            summary="None approved",
        )

        assert not agg.all_approved
        assert not agg.any_approved
        assert not agg.majority_approved

    def test_get_rejections(self):
        """Test getting list of rejections."""
        results = [
            ReviewResult("r1", "claude", MockReviewOutput(), 1.0, True),
            ReviewResult("r2", "gemini", None, 1.0, False, "Failed"),
            ReviewResult("r3", "codex", None, 1.0, False, "Timeout"),
        ]

        agg = AggregatedReviewResult(
            results=results,
            approved=False,
            approval_policy="ALL_APPROVED",
            summary="Mixed",
        )

        rejections = agg.get_rejections()
        assert len(rejections) == 2
        assert all(not r.success for r in rejections)

    def test_get_issues_from_output(self):
        """Test collecting issues from review outputs."""

        class OutputWithIssues(BaseModel):
            issues: list[str] = ["Issue 1", "Issue 2"]

        results = [
            ReviewResult("r1", "claude", OutputWithIssues(), 1.0, False),
        ]

        agg = AggregatedReviewResult(
            results=results,
            approved=False,
            approval_policy="ALL_APPROVED",
            summary="Has issues",
        )

        issues = agg.get_issues()
        assert len(issues) == 2
        assert "[r1] Issue 1" in issues
        assert "[r1] Issue 2" in issues

    def test_get_issues_handles_none_output(self):
        """Test that get_issues handles None output (error case)."""
        results = [
            ReviewResult("r1", "claude", None, 1.0, False, "Connection error"),
        ]

        agg = AggregatedReviewResult(
            results=results,
            approved=False,
            approval_policy="ALL_APPROVED",
            summary="Error",
        )

        issues = agg.get_issues()
        assert len(issues) == 1
        assert "[r1] Error: Connection error" in issues


class TestParallelReviewer:
    """Tests for ParallelReviewer."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock execution engine."""
        engine = MagicMock()

        # Mock role_loader.load_role
        engine.role_loader.load_role.return_value = MockRoleConfig("claude")

        # Mock run_role to return approved output
        engine.run_role.return_value = MockReviewOutput(status="APPROVED", review_status="APPROVED")

        return engine

    def test_empty_roles_raises_error(self, mock_engine):
        """Test that empty roles list raises ValueError."""
        reviewer = ParallelReviewer(mock_engine)

        with pytest.raises(ValueError) as exc_info:
            reviewer.run_parallel_review(
                roles=[],
                task_description="Review code",
                workflow_id="W-001",
            )

        assert "At least one reviewer role" in str(exc_info.value)

    def test_invalid_approval_policy_raises_error(self, mock_engine):
        """Test that invalid approval policy raises ValueError."""
        reviewer = ParallelReviewer(mock_engine)

        with pytest.raises(ValueError) as exc_info:
            reviewer.run_parallel_review(
                roles=["reviewer"],
                task_description="Review code",
                workflow_id="W-001",
                approval_policy="INVALID_POLICY",
            )

        assert "Invalid approval_policy" in str(exc_info.value)

    def test_all_approved_policy(self, mock_engine):
        """Test ALL_APPROVED policy."""
        reviewer = ParallelReviewer(mock_engine)

        result = reviewer.run_parallel_review(
            roles=["reviewer_gemini", "reviewer_codex"],
            task_description="Review code",
            workflow_id="W-001",
            approval_policy="ALL_APPROVED",
        )

        assert result.approved
        assert result.approval_policy == "ALL_APPROVED"
        assert len(result.results) == 2

    def test_any_approved_policy_partial_success(self, mock_engine):
        """Test ANY_APPROVED policy with one failure."""
        # First call succeeds, second fails
        mock_engine.run_role.side_effect = [
            MockReviewOutput(status="APPROVED"),
            Exception("Connection timeout"),
        ]

        reviewer = ParallelReviewer(mock_engine)

        result = reviewer.run_parallel_review(
            roles=["reviewer_gemini", "reviewer_codex"],
            task_description="Review code",
            workflow_id="W-001",
            approval_policy="ANY_APPROVED",
        )

        # Should still be approved with ANY_APPROVED policy
        assert result.approved
        assert len(result.results) == 2

    def test_majority_approved_policy(self, mock_engine):
        """Test MAJORITY_APPROVED policy."""
        # Two approve, one fails
        mock_engine.run_role.side_effect = [
            MockReviewOutput(status="APPROVED"),
            MockReviewOutput(status="APPROVED"),
            Exception("Timeout"),
        ]

        reviewer = ParallelReviewer(mock_engine)

        result = reviewer.run_parallel_review(
            roles=["r1", "r2", "r3"],
            task_description="Review code",
            workflow_id="W-001",
            approval_policy="MAJORITY_APPROVED",
        )

        # 2/3 approved = > 50%, should pass
        assert result.approved

    def test_handles_exception_in_reviewer(self, mock_engine):
        """Test that exceptions in reviewers are handled gracefully."""
        mock_engine.run_role.side_effect = Exception("Unexpected error")

        reviewer = ParallelReviewer(mock_engine)

        result = reviewer.run_parallel_review(
            roles=["reviewer"],
            task_description="Review code",
            workflow_id="W-001",
        )

        assert not result.approved
        assert len(result.results) == 1
        assert result.results[0].error == "Unexpected error"

    def test_summary_includes_approvers_and_rejecters(self, mock_engine):
        """Test that summary lists approved and rejected reviewers."""
        mock_engine.run_role.side_effect = [
            MockReviewOutput(status="APPROVED"),
            Exception("Failed"),
        ]

        reviewer = ParallelReviewer(mock_engine)

        result = reviewer.run_parallel_review(
            roles=["reviewer_pass", "reviewer_fail"],
            task_description="Review code",
            workflow_id="W-001",
            approval_policy="ANY_APPROVED",
        )

        assert "reviewer_pass" in result.summary
        assert "reviewer_fail" in result.summary
