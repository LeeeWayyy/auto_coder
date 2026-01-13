"""Thread-safe bridge for workflow-TUI communication.

FIX (Gemini review): Provides blocking sync API for workflow thread
and polling API for TUI main thread.

FIX (v13 - Gemini): Placed in CORE layer, not TUI layer, to ensure
WorkflowCoordinator can import without depending on presentation layer.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ApprovalDecision(str, Enum):
    """User decision for approval gate.

    FIX (v22 - Codex): Added is_proceed() helper for cleaner decision handling.
    """

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"  # Reserved for future implementation
    SKIP = "skip"

    def is_proceed(self) -> bool:
        """Return True if workflow should proceed (APPROVE or SKIP)."""
        return self in (ApprovalDecision.APPROVE, ApprovalDecision.SKIP)


@dataclass
class ApprovalRequest:
    """Request for human approval.

    FIX (v25 - Codex): Added diff_lines and expiry_time for post-execution
    approval flow showing actual git diff.
    """

    gate_id: str
    feature_id: str
    component_id: str | None
    title: str
    description: str
    risk_level: str
    changes: list[str]  # File paths (for counts and risk assessment)
    review_summary: str
    diff_lines: list[str] | None = None  # Actual git diff for display
    expiry_time: float | None = None  # Unix timestamp when request expires


class InteractionBridge:
    """Thread-safe bridge between sync workflow and TUI.

    USAGE (from workflow thread - BLOCKING):
        decision = bridge.request_approval(request, timeout=300)

    USAGE (from TUI main thread - NON-BLOCKING):
        pending = bridge.get_pending_requests()
        for req in pending:
            # Show to user, get decision
            bridge.submit_decision(req.gate_id, decision)
    """

    def __init__(self):
        # Queue for workflow -> TUI requests
        self._pending_requests: queue.Queue[ApprovalRequest] = queue.Queue()
        # Map of gate_id -> (Event, decision) for TUI -> workflow responses
        self._decisions: dict[str, tuple[threading.Event, ApprovalDecision | None]] = {}
        self._lock = threading.Lock()

    def request_approval(
        self,
        request: ApprovalRequest,
        timeout: float = 300.0,
    ) -> ApprovalDecision:
        """Request approval from TUI. BLOCKS until decision received.

        Called from workflow thread.
        """
        # Create event for this request
        event = threading.Event()
        with self._lock:
            self._decisions[request.gate_id] = (event, None)

        # Queue request for TUI
        self._pending_requests.put(request)

        # Block until TUI responds or timeout
        if event.wait(timeout=timeout):
            with self._lock:
                if request.gate_id in self._decisions:
                    _, decision = self._decisions.pop(request.gate_id)
                    return decision if decision else ApprovalDecision.SKIP
                return ApprovalDecision.SKIP
        else:
            # Timeout - clean up and return skip
            with self._lock:
                self._decisions.pop(request.gate_id, None)
            logger.warning(f"Approval request '{request.gate_id}' timed out")
            return ApprovalDecision.SKIP

    def get_pending_requests(self) -> list[ApprovalRequest]:
        """Get all pending approval requests. NON-BLOCKING.

        Called from TUI main thread.
        """
        requests = []
        while True:
            try:
                requests.append(self._pending_requests.get_nowait())
            except queue.Empty:
                break
        return requests

    def submit_decision(self, gate_id: str, decision: ApprovalDecision) -> bool:
        """Submit decision for a pending request.

        Called from TUI main thread. Returns True if request was pending.
        """
        with self._lock:
            if gate_id not in self._decisions:
                return False
            event, _ = self._decisions[gate_id]
            self._decisions[gate_id] = (event, decision)
            event.set()  # Unblock workflow thread
        return True
