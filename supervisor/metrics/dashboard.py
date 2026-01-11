"""Rich-based metrics dashboard."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

from supervisor.metrics.aggregator import MetricsAggregator


class MetricsDashboard:
    """Terminal dashboard for metrics visualization.

    USAGE:
        dashboard = MetricsDashboard(aggregator)
        dashboard.show()  # One-time display
        dashboard.live()  # Live updating display
    """

    def __init__(self, aggregator: MetricsAggregator):
        self.aggregator = aggregator
        self.console = Console()

    def show(self, days: int = 30) -> None:
        """Display metrics dashboard once."""
        self.console.print()
        self.console.rule(f"[bold blue]Supervisor Metrics - Last {days} Days[/bold blue]")

        # Summary stats
        self._show_summary(days)
        self.console.print()

        # Role performance
        self._show_role_performance(days)
        self.console.print()

        # CLI comparison
        self._show_cli_comparison(days)
        self.console.print()

        # Recent failures
        self._show_recent_failures()

    def _show_summary(self, days: int) -> None:
        """Show summary statistics."""
        stats = self.aggregator.get_summary_stats(days)

        table = Table(title="Summary", show_header=False, box=None)
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Total Executions", str(stats["total_executions"]))
        table.add_row("Overall Success Rate", stats["overall_success_rate"])
        table.add_row("Avg Duration", f"{stats['avg_duration_seconds']:.1f}s")
        table.add_row("Total Retries", str(stats["total_retries"]))
        table.add_row("Unique Workflows", str(stats["unique_workflows"]))

        self.console.print(Panel(table))

    def _show_role_performance(self, days: int) -> None:
        """Show role performance table."""
        roles = self.aggregator.get_role_performance(days)

        table = Table(title="Role Performance")
        table.add_column("Role", style="cyan")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg Time", justify="right")
        table.add_column("Avg Retries", justify="right")
        table.add_column("Executions", justify="right")

        for role in roles:
            success_style = "green" if role.success_rate > 0.9 else "yellow" if role.success_rate > 0.7 else "red"
            table.add_row(
                role.role,
                f"[{success_style}]{role.formatted_success_rate}[/]",
                f"{role.avg_duration_seconds:.1f}s",
                f"{role.avg_retries:.1f}",
                str(role.total_executions),
            )

        self.console.print(table)

    def _show_cli_comparison(self, days: int) -> None:
        """Show CLI comparison by task type."""
        stats = self.aggregator.get_cli_comparison(days)

        table = Table(title="CLI Performance by Task Type")
        table.add_column("Task Type", style="bold")
        table.add_column("CLI", style="cyan")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg Time", justify="right")
        table.add_column("Total", justify="right")

        current_task = None
        for stat in stats:
            # Add separator between task types
            if current_task and current_task != stat.task_type:
                table.add_row("", "", "", "", "")
            current_task = stat.task_type

            # Mark best performer
            is_best = stat.success_rate == max(
                s.success_rate for s in stats if s.task_type == stat.task_type
            )
            best_marker = " ✓" if is_best else ""

            success_style = "green" if stat.success_rate > 0.9 else "yellow" if stat.success_rate > 0.7 else "red"

            table.add_row(
                stat.task_type,
                f"{stat.cli}{best_marker}",
                f"[{success_style}]{stat.success_rate * 100:.1f}%[/]",
                f"{stat.avg_duration_seconds:.1f}s",
                str(stat.total_executions),
            )

        self.console.print(table)

    def _show_recent_failures(self, limit: int = 5) -> None:
        """Show recent failures."""
        failures = self.aggregator.get_recent_failures(limit)

        if not failures:
            self.console.print("[dim]No recent failures[/dim]")
            return

        table = Table(title="Recent Failures")
        table.add_column("Time", style="dim")
        table.add_column("Role")
        table.add_column("CLI")
        table.add_column("Workflow")
        table.add_column("Error")

        for f in failures:
            table.add_row(
                f["timestamp"][:19] if f["timestamp"] else "",
                f["role"],
                f["cli"],
                f["workflow_id"][:12] if f["workflow_id"] else "",
                f["error_category"] or "unknown",
            )

        self.console.print(table)
