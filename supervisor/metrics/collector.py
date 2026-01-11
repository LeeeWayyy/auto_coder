"""Optional collector wrapper - can derive metrics from events instead.

FIX (Codex review): The primary collection path is via Engine instrumentation.
This collector can optionally derive metrics from events for existing data.

FIX (v13 - Codex): Added class wrapper and proper imports.
"""

import logging
import time
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

    def _infer_task_type(self, role: str) -> str:
        """Infer task type from role name.

        FIX (v11): Standardized task_type values across schema, engine, router.
        """
        role_lower = role.lower()
        if "plan" in role_lower:
            return "plan"
        elif "review" in role_lower:
            return "review"
        elif "implement" in role_lower:
            return "implement"
        elif "test" in role_lower:
            return "test"
        elif "investigat" in role_lower:
            return "investigate"
        elif "doc" in role_lower:
            return "document"
        else:
            return "other"

    def start_step(self, step_id: str) -> None:
        """Record step start time for duration calculation."""
        self._step_start_times[step_id] = time.time()

    def end_step(self, step_id: str) -> float:
        """Get duration and cleanup step tracking."""
        start = self._step_start_times.pop(step_id, None)
        if start:
            return time.time() - start
        return 0.0
