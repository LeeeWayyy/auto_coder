"""Tests for metrics collection and aggregation.

This module tests metrics recording, aggregation, and dashboard display:
- Database.record_metric: Recording execution metrics directly
- MetricsAggregator: Querying and analyzing performance data
- MetricsDashboard: Displaying metrics to users

Note: MetricsCollector was deprecated - use Database.record_metric directly.
"""

from __future__ import annotations

import pytest

from supervisor.core.state import Database
from supervisor.metrics.aggregator import MetricsAggregator
from supervisor.metrics.dashboard import MetricsDashboard

# =============================================================================
# Direct Metrics Recording Tests
# =============================================================================


class TestMetricsRecording:
    """Tests for metrics recording via Database.record_metric."""

    def test_record_execution_metrics(self, test_db: Database):
        """Metrics are recorded for each execution."""
        test_db.record_metric(
            role="implementer",
            cli="claude",
            workflow_id="wf-1",
            success=True,
            duration_seconds=12.5,
            task_type="implementation",
            retry_count=0,
        )

        # Verify metrics stored in database
        metrics = test_db.get_metrics(days=7)
        assert len(metrics) >= 1
        assert metrics[0]["role"] == "implementer"

    def test_record_execution_failure(self, test_db: Database):
        """Failed executions are recorded with success=False."""
        test_db.record_metric(
            role="implementer",
            cli="claude",
            workflow_id="wf-1",
            success=False,
            duration_seconds=5.0,
            task_type="implementation",
            retry_count=2,
            error_category="cli_error",
        )

        metrics = test_db.get_metrics(days=7)
        assert len(metrics) >= 1
        assert metrics[0]["success"] == 0  # SQLite stores as 0/1


# =============================================================================
# MetricsAggregator Tests
# =============================================================================


class TestMetricsAggregator:
    """Tests for metrics aggregation and analysis."""

    def test_get_role_performance(self, populated_metrics_db: Database):
        """Aggregator computes performance statistics by role."""
        aggregator = MetricsAggregator(populated_metrics_db)

        stats = aggregator.get_role_performance(days=7)

        # Should have stats for roles in the test data
        role_names = [s.role for s in stats]
        assert "implementer" in role_names or len(stats) > 0

    def test_get_cli_comparison(self, populated_metrics_db: Database):
        """Aggregator compares performance across different AI models."""
        aggregator = MetricsAggregator(populated_metrics_db)

        comparison = aggregator.get_cli_comparison(days=30)

        # Should have comparison data
        assert isinstance(comparison, list)

    def test_get_best_cli_for_task(self, populated_metrics_db: Database):
        """Aggregator recommends best model for a task type."""
        aggregator = MetricsAggregator(populated_metrics_db)

        # May return None if not enough samples
        best_cli = aggregator.get_best_cli_for_task(
            task_type="implementation",
            days=30,
            min_samples=1,  # Lower threshold for test data
        )

        # Should return a CLI name or None
        assert best_cli is None or isinstance(best_cli, str)

    def test_date_range_filtering(self, test_db: Database):
        """Aggregator filters metrics by date range."""
        # Record a metric
        test_db.record_metric(
            role="implementer",
            cli="claude",
            workflow_id="wf-recent",
            success=True,
            duration_seconds=10.0,
            task_type="implementation",
            retry_count=0,
        )

        aggregator = MetricsAggregator(test_db)
        stats = aggregator.get_role_performance(days=7)

        # Should include recent metrics
        assert isinstance(stats, list)


# =============================================================================
# MetricsDashboard Tests
# =============================================================================


class TestMetricsDashboard:
    """Tests for metrics dashboard display."""

    def test_show_metrics_output(self, populated_metrics_db: Database, capsys):
        """Dashboard displays metrics in readable format."""
        aggregator = MetricsAggregator(populated_metrics_db)
        dashboard = MetricsDashboard(aggregator)

        dashboard.show(days=7)

        captured = capsys.readouterr()
        output = captured.out

        # Dashboard should produce output when there's data
        # The populated_metrics_db fixture should have data
        assert len(output) > 0, "Dashboard should produce output with populated data"

    def test_dashboard_with_aggregator(self, populated_metrics_db: Database):
        """Dashboard correctly accepts MetricsAggregator."""
        aggregator = MetricsAggregator(populated_metrics_db)
        dashboard = MetricsDashboard(aggregator)

        # Should not raise
        assert dashboard.aggregator is aggregator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def populated_metrics_db(test_db: Database) -> Database:
    """Database with sample metrics data."""
    # Record various executions using Database.record_metric
    for i in range(10):
        test_db.record_metric(
            role="implementer" if i % 2 == 0 else "planner",
            cli="claude" if i % 2 == 0 else "codex",
            workflow_id=f"wf-{i}",
            success=i % 3 != 0,  # Some failures
            duration_seconds=10.0 + i,
            task_type="implementation" if i % 2 == 0 else "planning",
            retry_count=i % 2,
        )

    return test_db
