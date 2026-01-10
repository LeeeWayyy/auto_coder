"""Tests for Workflow Coordinator (Phase 4).

Tests Feature->Phase->Component workflow orchestration.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from supervisor.core.models import (
    ComponentStatus,
    FeatureStatus,
    PhaseStatus,
)
from supervisor.core.state import Database
from supervisor.core.workflow import (
    ComponentPlan,
    PhasePlan,
    PlannerOutput,
    WorkflowCoordinator,
    WorkflowError,
)


class MockPlannerOutput(BaseModel):
    """Mock planner output for testing."""

    phases: list[dict]
    summary: str = "Test plan"


@pytest.fixture
def temp_db():
    """Create temporary database for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        yield db


@pytest.fixture
def mock_engine():
    """Create mock execution engine."""
    engine = MagicMock()
    engine.repo_path = Path("/test/repo")
    engine.db = None  # Will be set per test

    # Mock role_loader
    engine.role_loader.load_role.return_value = MagicMock(cli="claude")

    return engine


class TestPlannerOutputSchema:
    """Tests for planner output Pydantic schemas."""

    def test_component_plan_minimal(self):
        """Test ComponentPlan with minimal fields."""
        plan = ComponentPlan(title="Test Component")
        assert plan.title == "Test Component"
        assert plan.files == []
        assert plan.depends_on == []
        assert plan.role == "implementer"

    def test_component_plan_full(self):
        """Test ComponentPlan with all fields."""
        plan = ComponentPlan(
            title="Auth Service",
            symbolic_id="auth_service",
            files=["services/auth.py"],
            depends_on=["user_model"],
            role="implementer",
            description="Implements authentication logic",
        )
        assert plan.symbolic_id == "auth_service"
        assert len(plan.files) == 1
        assert len(plan.depends_on) == 1

    def test_phase_plan_requires_components(self):
        """Test that PhasePlan requires at least one component."""
        with pytest.raises(ValueError):
            PhasePlan(title="Empty Phase", components=[])

    def test_planner_output_requires_phases(self):
        """Test that PlannerOutput requires at least one phase."""
        with pytest.raises(ValueError):
            PlannerOutput(phases=[])

    def test_planner_output_valid(self):
        """Test valid PlannerOutput."""
        output = PlannerOutput(
            phases=[
                PhasePlan(
                    title="Phase 1",
                    components=[ComponentPlan(title="Component A")],
                )
            ],
            summary="Implementation plan",
        )
        assert len(output.phases) == 1
        assert output.summary == "Implementation plan"


class TestWorkflowCoordinatorCreate:
    """Tests for feature creation."""

    def test_create_feature(self, temp_db, mock_engine):
        """Test creating a new feature."""
        mock_engine.db = temp_db
        coordinator = WorkflowCoordinator(mock_engine, temp_db)

        feature = coordinator.create_feature(
            title="User Authentication",
            description="Implement JWT-based authentication",
        )

        assert feature is not None
        assert feature.title == "User Authentication"
        assert feature.status == FeatureStatus.PLANNING

    def test_feature_id_format(self, temp_db, mock_engine):
        """Test that feature ID follows expected format."""
        mock_engine.db = temp_db
        coordinator = WorkflowCoordinator(mock_engine, temp_db)

        feature = coordinator.create_feature(title="Test Feature")

        assert feature.id.startswith("F-")
        assert len(feature.id) == 10  # F- + 8 hex chars


class TestWorkflowCoordinatorPlanning:
    """Tests for planning phase."""

    def test_run_planning_creates_phases(self, temp_db, mock_engine):
        """Test that planning creates phases from planner output."""
        mock_engine.db = temp_db

        # Mock planner output
        mock_engine.run_role.return_value = MockPlannerOutput(
            phases=[
                {
                    "title": "Backend",
                    "components": [
                        {"title": "User Model", "files": ["models/user.py"]},
                        {"title": "Auth Service", "files": ["services/auth.py"]},
                    ],
                },
                {
                    "title": "API",
                    "components": [
                        {"title": "Endpoints", "files": ["api/auth.py"]},
                    ],
                },
            ]
        )

        coordinator = WorkflowCoordinator(mock_engine, temp_db)
        feature = coordinator.create_feature("Test Feature")

        phases = coordinator.run_planning(feature.id)

        assert len(phases) == 2
        assert phases[0].title == "Backend"
        assert phases[1].title == "API"

    def test_run_planning_creates_components(self, temp_db, mock_engine):
        """Test that planning creates components."""
        mock_engine.db = temp_db

        mock_engine.run_role.return_value = MockPlannerOutput(
            phases=[
                {
                    "title": "Phase 1",
                    "components": [
                        {"title": "Component A", "files": ["a.py"]},
                        {"title": "Component B", "files": ["b.py"]},
                    ],
                },
            ]
        )

        coordinator = WorkflowCoordinator(mock_engine, temp_db)
        feature = coordinator.create_feature("Test Feature")
        coordinator.run_planning(feature.id)

        components = temp_db.get_components(feature.id)
        assert len(components) == 2

    def test_run_planning_maps_dependencies(self, temp_db, mock_engine):
        """Test that symbolic dependencies are mapped to generated IDs."""
        mock_engine.db = temp_db

        mock_engine.run_role.return_value = MockPlannerOutput(
            phases=[
                {
                    "title": "Phase 1",
                    "components": [
                        {"title": "Component A", "symbolic_id": "comp_a", "files": ["a.py"]},
                        {
                            "title": "Component B",
                            "symbolic_id": "comp_b",
                            "files": ["b.py"],
                            "depends_on": ["comp_a"],  # Symbolic reference
                        },
                    ],
                },
            ]
        )

        coordinator = WorkflowCoordinator(mock_engine, temp_db)
        feature = coordinator.create_feature("Test Feature")
        coordinator.run_planning(feature.id)

        components = temp_db.get_components(feature.id)
        comp_b = next(c for c in components if c.title == "Component B")

        # depends_on should now reference the generated ID, not symbolic
        assert len(comp_b.depends_on) == 1
        assert comp_b.depends_on[0].startswith(feature.id)  # Generated ID format

    def test_run_planning_invalid_output_raises_error(self, temp_db, mock_engine):
        """Test that invalid planner output raises WorkflowError."""
        mock_engine.db = temp_db

        # Return invalid output (no phases)
        mock_engine.run_role.return_value = {"invalid": "output"}

        coordinator = WorkflowCoordinator(mock_engine, temp_db)
        feature = coordinator.create_feature("Test Feature")

        with pytest.raises(WorkflowError) as exc_info:
            coordinator.run_planning(feature.id)

        assert "validation failed" in str(exc_info.value).lower()


class TestWorkflowCoordinatorStatus:
    """Tests for feature status retrieval."""

    def test_get_feature_status(self, temp_db, mock_engine):
        """Test getting detailed feature status."""
        mock_engine.db = temp_db

        mock_engine.run_role.return_value = MockPlannerOutput(
            phases=[
                {
                    "title": "Phase 1",
                    "components": [
                        {"title": "Component A", "files": ["a.py"]},
                        {"title": "Component B", "files": ["b.py"]},
                    ],
                },
            ]
        )

        coordinator = WorkflowCoordinator(mock_engine, temp_db)
        feature = coordinator.create_feature("Test Feature")
        coordinator.run_planning(feature.id)

        status = coordinator.get_feature_status(feature.id)

        assert status["feature"]["id"] == feature.id
        assert len(status["phases"]) == 1
        assert status["summary"]["total_components"] == 2
        assert status["summary"]["pending_components"] == 2

    def test_get_feature_status_not_found(self, temp_db, mock_engine):
        """Test that getting status of non-existent feature raises error."""
        mock_engine.db = temp_db
        coordinator = WorkflowCoordinator(mock_engine, temp_db)

        with pytest.raises(WorkflowError) as exc_info:
            coordinator.get_feature_status("F-NOTFOUND")

        assert "not found" in str(exc_info.value)


class TestWorkflowCoordinatorDuplicateSymbolicIds:
    """Tests for duplicate symbolic ID detection."""

    def test_duplicate_symbolic_id_raises_error(self, temp_db, mock_engine):
        """Test that duplicate symbolic IDs in same phase raise error."""
        mock_engine.db = temp_db

        mock_engine.run_role.return_value = MockPlannerOutput(
            phases=[
                {
                    "title": "Phase 1",
                    "components": [
                        {"title": "Component A", "symbolic_id": "duplicate"},
                        {"title": "Component B", "symbolic_id": "duplicate"},  # Duplicate!
                    ],
                },
            ]
        )

        coordinator = WorkflowCoordinator(mock_engine, temp_db)
        feature = coordinator.create_feature("Test Feature")

        with pytest.raises(WorkflowError) as exc_info:
            coordinator.run_planning(feature.id)

        assert "Duplicate symbolic_id" in str(exc_info.value)

    def test_same_title_different_phases_allowed(self, temp_db, mock_engine):
        """Test that same title in different phases is allowed (phase-prefixed)."""
        mock_engine.db = temp_db

        mock_engine.run_role.return_value = MockPlannerOutput(
            phases=[
                {
                    "title": "Phase 1",
                    "components": [{"title": "Setup", "files": ["setup1.py"]}],
                },
                {
                    "title": "Phase 2",
                    "components": [{"title": "Setup", "files": ["setup2.py"]}],
                },
            ]
        )

        coordinator = WorkflowCoordinator(mock_engine, temp_db)
        feature = coordinator.create_feature("Test Feature")

        # Should not raise - different phases have different prefixes
        phases = coordinator.run_planning(feature.id)
        assert len(phases) == 2
