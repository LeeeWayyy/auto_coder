"""Tests for core data models and enums.

This module tests the Pydantic models and enums used throughout Supervisor:
- Step, StepStatus
- Feature, Phase, Component (hierarchical workflow)
- Model validation and serialization
- Status enums
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from supervisor.core.models import (
    Component,
    ComponentStatus,
    Feature,
    FeatureStatus,
    Phase,
    PhaseStatus,
    Step,
    StepStatus,
)

# =============================================================================
# Step Model Tests
# =============================================================================


class TestStepModel:
    """Tests for Step data model."""

    def test_step_creation(self):
        """Step can be created with required fields."""
        step = Step(
            id="step-123",
            workflow_id="wf-1",
            role="implementer",
            status=StepStatus.PENDING,
        )

        assert step.id == "step-123"
        assert step.workflow_id == "wf-1"
        assert step.role == "implementer"
        assert step.status == StepStatus.PENDING

    def test_step_with_optional_fields(self):
        """Step can include optional fields like gates, context."""
        step = Step(
            id="step-123",
            workflow_id="wf-1",
            role="implementer",
            status=StepStatus.PENDING,
            gates=["test", "lint"],
            context={"target_files": ["api/routes.py"]},
        )

        assert step.gates == ["test", "lint"]
        assert step.context["target_files"] == ["api/routes.py"]

    def test_step_status_enum(self):
        """Step status uses StepStatus enum."""
        step = Step(
            id="step-1",
            workflow_id="wf-1",
            role="reviewer",
            status=StepStatus.COMPLETED,
        )

        assert step.status == StepStatus.COMPLETED
        assert step.status.value == "completed"

    def test_step_validation_requires_fields(self):
        """Step validation requires mandatory fields."""
        with pytest.raises(ValidationError):
            Step(id="step-1")  # Missing workflow_id, role, status


# =============================================================================
# Feature/Phase/Component Model Tests
# =============================================================================


class TestHierarchicalModels:
    """Tests for Feature, Phase, Component models."""

    def test_feature_creation(self):
        """Feature can be created with title and description."""
        feature = Feature(
            id="feat-123",
            title="User Authentication",
            description="Add user authentication",
        )

        assert feature.id == "feat-123"
        assert feature.title == "User Authentication"
        assert feature.description == "Add user authentication"
        assert feature.status == FeatureStatus.PLANNING

    def test_phase_creation(self):
        """Phase can be created with feature reference."""
        phase = Phase(
            id="phase-1",
            feature_id="feat-123",
            title="Implementation",
            sequence=1,
        )

        assert phase.id == "phase-1"
        assert phase.feature_id == "feat-123"
        assert phase.title == "Implementation"
        assert phase.sequence == 1
        assert phase.status == PhaseStatus.PENDING

    def test_component_creation(self):
        """Component can be created with phase reference."""
        component = Component(
            id="comp-1",
            phase_id="phase-1",
            title="auth_middleware",
            description="Implement auth middleware",
            depends_on=[],
        )

        assert component.id == "comp-1"
        assert component.phase_id == "phase-1"
        assert component.title == "auth_middleware"
        assert component.status == ComponentStatus.PENDING

    def test_component_with_dependencies(self):
        """Component can have dependencies on other components."""
        component = Component(
            id="comp-2",
            phase_id="phase-1",
            title="auth_routes",
            description="Auth API routes",
            depends_on=["comp-1"],  # Depends on auth_middleware
        )

        assert component.depends_on == ["comp-1"]


# =============================================================================
# Status Enum Tests
# =============================================================================


class TestStatusEnums:
    """Tests for status enumerations."""

    def test_step_status_values(self):
        """StepStatus enum has expected values."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"

    def test_feature_status_values(self):
        """FeatureStatus enum has expected values."""
        assert FeatureStatus.PLANNING.value == "planning"
        assert FeatureStatus.IN_PROGRESS.value == "in_progress"
        assert FeatureStatus.COMPLETED.value == "completed"
        assert FeatureStatus.FAILED.value == "failed"

    def test_status_comparison(self):
        """Status enums can be compared."""
        assert StepStatus.PENDING == StepStatus.PENDING
        assert StepStatus.PENDING != StepStatus.COMPLETED


# =============================================================================
# Serialization Tests
# =============================================================================


class TestModelSerialization:
    """Tests for model serialization and deserialization."""

    def test_step_to_dict(self):
        """Step can be serialized to dictionary."""
        step = Step(
            id="step-1",
            workflow_id="wf-1",
            role="implementer",
            status=StepStatus.PENDING,
        )

        data = step.model_dump()

        assert data["id"] == "step-1"
        assert data["workflow_id"] == "wf-1"
        assert data["status"] == "pending"

    def test_step_from_dict(self):
        """Step can be deserialized from dictionary."""
        data = {
            "id": "step-1",
            "workflow_id": "wf-1",
            "role": "implementer",
            "status": "pending",
        }

        step = Step(**data)

        assert step.id == "step-1"
        assert step.status == StepStatus.PENDING

    def test_step_json_serialization(self):
        """Step can be serialized to JSON."""
        step = Step(
            id="step-1",
            workflow_id="wf-1",
            role="implementer",
            status=StepStatus.PENDING,
        )

        json_str = step.model_dump_json()

        assert '"id":"step-1"' in json_str or '"id": "step-1"' in json_str
        assert '"status":"pending"' in json_str or '"status": "pending"' in json_str
