"""Approval gate integration for human-in-the-loop workflows."""

import logging
from dataclasses import dataclass, field
from typing import Any

from supervisor.core.state import Database, Event, EventType
from supervisor.core.interaction import ApprovalRequest, ApprovalDecision, InteractionBridge

logger = logging.getLogger(__name__)


@dataclass
class ApprovalPolicy:
    """Policy for determining when approval is required.

    FIX (v14 - Codex): Simplified to match config loader and CLI construction.
    The original *_conditions schema from SUPERVISOR_ORCHESTRATOR.md is complex;
    this simpler schema suffices for MVP approval gates.
    """
    # Auto-approve low-risk changes without prompting
    auto_approve_low_risk: bool = True

    # Minimum risk level that requires approval ("low", "medium", "high", "critical")
    risk_threshold: str = "medium"

    # Operations that always require approval regardless of risk
    require_approval_for: list[str] = field(default_factory=lambda: ["deploy", "commit"])


class ApprovalGate:
    """Human approval gate for workflow steps.

    USAGE (v10 - SYNC, no await):
        gate = ApprovalGate(db, bridge, policy)

        # Check if approval needed
        if gate.requires_approval(context):
            # FIX (v10): SYNC call - blocks until user responds via InteractionBridge
            decision = gate.request_approval(
                feature_id=feature_id,
                title="Deploy to production",
                changes=["api/auth.py", "middleware/auth.py"],
                review_summary="All reviewers approved",
                bridge=bridge,  # InteractionBridge for TUI communication
            )

            if decision != ApprovalDecision.APPROVE:
                raise ApprovalRejected(decision)
    """

    def __init__(
        self,
        db: Database,
        policy: ApprovalPolicy | None = None,
    ):
        """Initialize ApprovalGate.

        FIX (v16): Removed tui parameter - core should not depend on TUI.
        Approval flow uses InteractionBridge passed via request_approval().
        """
        self.db = db
        self.policy = policy or self._default_policy()

    def _default_policy(self) -> ApprovalPolicy:
        """Default approval policy.

        FIX (v14): Simplified to match new ApprovalPolicy schema.
        """
        return ApprovalPolicy(
            auto_approve_low_risk=True,
            risk_threshold="medium",
            require_approval_for=["deploy", "commit"],
        )

    def assess_risk_level(self, context: dict[str, Any]) -> str:
        """Assess risk level based on context.

        Returns: "low", "medium", "high", or "critical"
        """
        changes = context.get("changes", [])
        file_count = len(changes)

        # Check critical conditions
        critical_patterns = ["encrypt", "key", "secret", "credential", "production"]
        for change in changes:
            for pattern in critical_patterns:
                if pattern in change.lower():
                    return "critical"

        # Check high risk conditions
        high_risk_patterns = ["auth", "payment", "api/", "security"]
        for change in changes:
            for pattern in high_risk_patterns:
                if pattern in change.lower():
                    return "high"

        # Check file count
        # FIX (v27 - Gemini PR review): Simplified redundant conditions
        if file_count > 20:
            return "high"
        elif file_count > 3:
            return "medium"

        return "low"

    def requires_approval(self, context: dict[str, Any]) -> bool:
        """Check if context requires human approval.

        FIX (v14): Updated to use simplified policy fields.
        """
        # Check if operation type requires approval regardless of risk
        operation = context.get("operation", "")
        if operation in self.policy.require_approval_for:
            return True

        # Assess risk level
        risk = self.assess_risk_level(context)

        # Auto-approve low risk if policy allows
        if risk == "low" and self.policy.auto_approve_low_risk:
            return False

        # Compare risk level against threshold
        risk_levels = ["low", "medium", "high", "critical"]
        risk_idx = risk_levels.index(risk)
        threshold_idx = risk_levels.index(self.policy.risk_threshold)

        return risk_idx >= threshold_idx

    def request_approval(
        self,
        feature_id: str,
        title: str,
        changes: list[str],
        review_summary: str,
        component_id: str | None = None,
        bridge: "InteractionBridge | None" = None,  # FIX: Use bridge for sync
        diff_lines: list[str] | None = None,  # FIX (v25): Actual diff for display
    ) -> ApprovalDecision:
        """Request human approval.

        Uses TUI if available, falls back to CLI prompt.

        FIX (v25 - Codex): Added diff_lines parameter to display actual git diff
        in approval UI, separate from changes (file paths) used for risk scoring.
        """
        import time
        import uuid

        gate_id = f"gate-{uuid.uuid4().hex[:8]}"
        risk_level = self.assess_risk_level({"changes": changes})

        request = ApprovalRequest(
            gate_id=gate_id,
            feature_id=feature_id,
            component_id=component_id,
            title=title,
            description="",
            risk_level=risk_level,
            changes=changes,
            review_summary=review_summary,
            diff_lines=diff_lines,  # FIX (v25): Pass diff for display
            expiry_time=time.time() + 300.0,  # 5 minute expiry
        )

        # Log approval request using EXISTING EventType
        self.db.append_event(
            Event(
                workflow_id=feature_id,
                event_type=EventType.APPROVAL_REQUESTED,  # USE EXISTING
                payload={
                    "gate_id": gate_id,
                    "risk_level": risk_level,
                    "changes": changes,
                },
            )
        )

        # Request approval using bridge (SYNC) or CLI fallback
        # FIX (Gemini review): Use InteractionBridge for sync workflow-TUI communication
        if bridge:
            # Called from workflow thread - blocks until TUI responds
            decision = bridge.request_approval(request, timeout=300.0)
        else:
            # CLI fallback (no TUI running)
            decision = self._cli_approval_sync(request)

        # FIX (v26 - Codex): Log correct event type based on decision
        # SKIP should be APPROVAL_SKIPPED, not APPROVAL_DENIED
        if decision == ApprovalDecision.APPROVE:
            event_type = EventType.APPROVAL_GRANTED
        elif decision == ApprovalDecision.SKIP:
            event_type = EventType.APPROVAL_SKIPPED
        else:
            event_type = EventType.APPROVAL_DENIED
        self.db.append_event(
            Event(
                workflow_id=feature_id,
                event_type=event_type,
                payload={
                    "gate_id": gate_id,
                    "decision": decision.value,
                },
            )
        )

        return decision

    def _cli_approval_sync(self, request: ApprovalRequest) -> ApprovalDecision:
        """Fallback CLI-based approval. SYNCHRONOUS.

        FIX (Gemini review): Changed from async to sync.
        """
        from rich.console import Console
        from rich.prompt import Confirm

        console = Console()
        console.print(f"\n[bold red]Approval Required[/bold red]: {request.title}")
        console.print(f"Risk Level: [{request.risk_level}]")
        console.print(f"Changes: {len(request.changes)} files")

        for change in request.changes[:5]:
            console.print(f"  • {change}")
        if len(request.changes) > 5:
            console.print(f"  ... and {len(request.changes) - 5} more")

        approved = Confirm.ask("Approve?", default=True)
        return ApprovalDecision.APPROVE if approved else ApprovalDecision.REJECT
