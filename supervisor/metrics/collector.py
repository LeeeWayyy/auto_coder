"""Optional collector wrapper - can derive metrics from events instead.

FIX (Codex review): The primary collection path is via Engine instrumentation.
This collector can optionally derive metrics from events for existing data.

FIX (v13 - Codex): Added class wrapper and proper imports.
"""

import logging
import time
from supervisor.core.routing import _infer_task_type
from supervisor.core.state import Database

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Optional helper for metrics timing.

    FIX (v13 - Codex): This class was missing - methods were orphaned.
    Primary metrics collection is via Engine.run_role() instrumentation.
    This class provides optional timing helpers.
    """

    def __init__(self, db: Database):
        self.db = db
        self._step_start_times: dict[str, float] = {}

    # FIX (v27 - Gemini PR review): Removed duplicate _infer_task_type
    # Now using shared function from supervisor.core.routing

    def start_step(self, step_id: str) -> None:
        """Record step start time for duration calculation."""
        self._step_start_times[step_id] = time.time()

    def end_step(self, step_id: str) -> float:
        """Get duration and cleanup step tracking."""
        start = self._step_start_times.pop(step_id, None)
        if start:
            return time.time() - start
        return 0.0
