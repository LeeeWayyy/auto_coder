"""Metrics collection - DEPRECATED.

FIX (v27 - Gemini PR review): This module is deprecated.
Primary metrics collection is handled via Engine.run_role() instrumentation
which records metrics directly to the database.

This file is kept for backwards compatibility but the MetricsCollector class
has been removed as it was unused and caused confusion about the intended
metrics collection path.

For metrics, use:
- Engine.run_role() which automatically records metrics
- MetricsAggregator for querying collected metrics
- MetricsDashboard for displaying metrics
"""

# Deprecated - no exports
__all__: list[str] = []
