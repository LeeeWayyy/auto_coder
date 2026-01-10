"""Data models for the Supervisor orchestrator.

Uses Pydantic for schema-enforced structured outputs.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    AWAITING_APPROVAL = "awaiting_approval"


class ComponentStatus(str, Enum):
    """Status of a component in hierarchical workflow."""

    PENDING = "pending"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"


class PhaseStatus(str, Enum):
    """Status of a phase (rolled up from components)."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class FeatureStatus(str, Enum):
    """Status of a feature (rolled up from phases)."""

    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Workflow State Models ---


class Step(BaseModel):
    """A single step in a workflow."""

    id: str
    workflow_id: str
    role: str
    status: StepStatus = StepStatus.PENDING
    gates: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowState(BaseModel):
    """Current state of a workflow execution."""

    id: str
    current_step: str
    status: StepStatus = StepStatus.PENDING
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# --- Hierarchical Workflow Models ---


class Component(BaseModel):
    """Atomic unit of work in hierarchical workflow."""

    id: str  # e.g., "P1T5-PH1-C1"
    phase_id: str
    title: str
    # FIX (PR review): Add description field (was causing AttributeError)
    description: str = ""
    files: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    status: ComponentStatus = ComponentStatus.PENDING
    assigned_role: str | None = None
    output: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Interface(BaseModel):
    """Interface definition locked before parallel implementation."""

    name: str
    type: str  # "typescript", "openapi", "python_protocol"
    definition: str
    locked: bool = False


class Phase(BaseModel):
    """Groups related components in a feature."""

    id: str  # e.g., "P1T5-PH1"
    feature_id: str
    title: str
    sequence: int  # Execution order
    status: PhaseStatus = PhaseStatus.PENDING
    interfaces: list[Interface] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Feature(BaseModel):
    """Top-level task in hierarchical workflow."""

    id: str  # e.g., "P1T5"
    title: str
    description: str = ""
    status: FeatureStatus = FeatureStatus.PLANNING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


# --- Worker Output Models (Schema-Enforced) ---


class PlanOutput(BaseModel):
    """Schema for Planner role output - enforced, not optional."""

    status: str = Field(..., pattern="^(COMPLETE|NEEDS_REFINEMENT|BLOCKED)$")
    phases: list[dict[str, Any]]
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    estimated_components: int
    risks: list[str] = Field(default_factory=list)
    next_step: str | None = None


class ImplementationOutput(BaseModel):
    """Schema for Implementer role output."""

    status: str = Field(..., pattern="^(SUCCESS|PARTIAL|FAILED|BLOCKED)$")
    action_taken: str
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    tests_written: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_step: str | None = None


class ReviewIssue(BaseModel):
    """A single issue found during review."""

    severity: str = Field(..., pattern="^(critical|high|medium|low)$")
    file: str
    line: int | None = None
    description: str
    suggestion: str | None = None


class ReviewOutput(BaseModel):
    """Schema for Reviewer role output."""

    status: str = Field(..., pattern="^(APPROVED|CHANGES_REQUESTED|REJECTED)$")
    review_status: str = Field(..., pattern="^(APPROVED|CHANGES_REQUESTED|REJECTED)$")
    issues: list[ReviewIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    security_concerns: list[str] = Field(default_factory=list)
    next_step: str | None = None


# --- Error Models ---


class ErrorCategory(str, Enum):
    """Six categories of errors in CLI orchestration."""

    NETWORK = "network"  # Timeouts, DNS, connection failures
    CLI_ERROR = "cli_error"  # Non-zero exit codes, CLI crashes
    VALIDATION = "validation"  # Output doesn't match schema
    PARSING = "parsing"  # Can't extract structured data from output
    LOGIC = "logic"  # Worker produced wrong/incomplete result
    RESOURCE = "resource"  # Out of memory, disk full, etc.


class ErrorAction(str, Enum):
    """What to do when an error occurs."""

    RETRY_SAME = "retry_same"  # Retry with same context
    RETRY_WITH_FEEDBACK = "retry_with_feedback"  # Retry with error appended
    RETRY_ONCE = "retry_once"  # Single retry then escalate
    ESCALATE = "escalate"  # Immediate human intervention
