"""Metrics aggregation and analysis.

FIX (Codex v3): Uses db._connect() pattern like other Database methods.
Cannot use db.execute() as it doesn't exist.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from supervisor.core.state import Database

logger = logging.getLogger(__name__)


@dataclass
class RolePerformance:
    """Aggregated performance for a role."""

    role: str
    success_rate: float
    avg_duration_seconds: float
    avg_retries: float
    total_executions: int

    @property
    def formatted_success_rate(self) -> str:
        return f"{self.success_rate * 100:.1f}%"


@dataclass
class CLIPerformance:
    """Aggregated performance for a CLI by task type."""

    cli: str
    task_type: str
    success_rate: float
    avg_duration_seconds: float
    total_executions: int


class MetricsAggregator:
    """Aggregate and analyze metrics.

    FIX (Codex v3): Uses db._connect() pattern to access SQLite.

    USAGE:
        aggregator = MetricsAggregator(db)
        roles = aggregator.get_role_performance(days=30)
        cli_stats = aggregator.get_cli_comparison(days=30)
    """

    def __init__(self, db: Database):
        self.db = db

    def get_role_performance(
        self,
        days: int = 30,
    ) -> list[RolePerformance]:
        """Get performance metrics by role.

        FIX (Codex v3): Uses db._connect() pattern.
        FIX (v14): Use datetime() for format-agnostic timestamp comparison.
        """
        # FIX: Use _connect() pattern like other Database methods
        with self.db._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    role,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_seconds) as avg_duration,
                    AVG(retry_count) as avg_retries,
                    COUNT(*) as total
                FROM metrics
                WHERE timestamp > datetime('now', '-{days} days')
                GROUP BY role
                ORDER BY success_rate DESC
                """
            ).fetchall()

        return [
            RolePerformance(
                role=row[0],
                success_rate=row[1] or 0.0,
                avg_duration_seconds=row[2] or 0.0,
                avg_retries=row[3] or 0.0,
                total_executions=row[4],
            )
            for row in rows
        ]

    def get_cli_comparison(
        self,
        days: int = 30,
    ) -> list[CLIPerformance]:
        """Get performance comparison across CLIs.

        FIX (Codex v3): Uses db._connect() pattern.
        FIX (v14): Use datetime() for format-agnostic timestamp comparison.
        """
        with self.db._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    cli,
                    task_type,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_seconds) as avg_duration,
                    COUNT(*) as total
                FROM metrics
                WHERE timestamp > datetime('now', '-{days} days')
                GROUP BY cli, task_type
                ORDER BY task_type, success_rate DESC
                """
            ).fetchall()

        return [
            CLIPerformance(
                cli=row[0],
                task_type=row[1],
                success_rate=row[2] or 0.0,
                avg_duration_seconds=row[3] or 0.0,
                total_executions=row[4],
            )
            for row in rows
        ]

    def get_best_cli_for_task(
        self,
        task_type: str,
        days: int = 30,
        min_samples: int = 10,
    ) -> str | None:
        """Get the best performing CLI for a task type.

        FIX (Codex v3): Uses db._connect() pattern.
        FIX (v14): Use datetime() for format-agnostic timestamp comparison.
        """
        with self.db._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    cli,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_seconds) as avg_duration,
                    COUNT(*) as total
                FROM metrics
                WHERE timestamp > datetime('now', '-{days} days') AND task_type = ?
                GROUP BY cli
                HAVING COUNT(*) >= ?
                ORDER BY success_rate DESC
                LIMIT 1
                """,
                (task_type, min_samples),
            ).fetchone()

        return rows[0] if rows else None

    def get_recent_failures(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent execution failures for debugging.

        FIX (Codex v3): Uses db._connect() pattern.
        """
        with self.db._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    timestamp,
                    role,
                    cli,
                    workflow_id,
                    error_category,
                    duration_seconds
                FROM metrics
                WHERE success = 0
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "timestamp": row[0],
                "role": row[1],
                "cli": row[2],
                "workflow_id": row[3],
                "error_category": row[4],
                "duration_seconds": row[5],
            }
            for row in rows
        ]

    def get_summary_stats(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get high-level summary statistics.

        FIX (Codex v3): Uses db._connect() pattern.
        FIX (v14): Use datetime() for format-agnostic timestamp comparison.
        """
        with self.db._connect() as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_executions,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as overall_success_rate,
                    AVG(duration_seconds) as avg_duration,
                    SUM(retry_count) as total_retries,
                    COUNT(DISTINCT workflow_id) as unique_workflows
                FROM metrics
                WHERE timestamp > datetime('now', '-{days} days')
                """
            ).fetchone()

        return {
            "total_executions": row[0],
            "overall_success_rate": f"{(row[1] or 0) * 100:.1f}%",
            "avg_duration_seconds": row[2] or 0,
            "total_retries": row[3] or 0,
            "unique_workflows": row[4] or 0,
            "period_days": days,
        }
