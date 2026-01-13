"""Tests for DAG Scheduler (Phase 4).

Tests dependency-aware scheduling with phase sequencing.
"""

import tempfile
from pathlib import Path

import pytest

from supervisor.core.models import (
    ComponentStatus,
)
from supervisor.core.scheduler import (
    CyclicDependencyError,
    DAGScheduler,
    DependencyNotFoundError,
)
from supervisor.core.state import Database


@pytest.fixture
def temp_db():
    """Create temporary database for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        yield db


@pytest.fixture
def simple_feature(temp_db):
    """Create a simple feature with one phase and two components."""
    feature_id = "F-TEST001"
    temp_db.create_feature(feature_id, "Test Feature", "A test feature")
    temp_db.create_phase("F-TEST001-PH1", feature_id, "Phase 1", 1)
    temp_db.create_component(
        "F-TEST001-PH1-C1",
        "F-TEST001-PH1",
        "Component A",
        files=["a.py"],
        depends_on=[],
    )
    temp_db.create_component(
        "F-TEST001-PH1-C2",
        "F-TEST001-PH1",
        "Component B",
        files=["b.py"],
        depends_on=["F-TEST001-PH1-C1"],
    )
    return feature_id


@pytest.fixture
def multi_phase_feature(temp_db):
    """Create a feature with multiple phases."""
    feature_id = "F-TEST002"
    temp_db.create_feature(feature_id, "Multi-Phase Feature", "Feature with 3 phases")

    # Phase 1: Backend
    temp_db.create_phase("F-TEST002-PH1", feature_id, "Backend", 1)
    temp_db.create_component(
        "F-TEST002-PH1-C1",
        "F-TEST002-PH1",
        "User Model",
        files=["models/user.py"],
    )
    temp_db.create_component(
        "F-TEST002-PH1-C2",
        "F-TEST002-PH1",
        "Auth Service",
        files=["services/auth.py"],
        depends_on=["F-TEST002-PH1-C1"],
    )

    # Phase 2: API (depends on Phase 1)
    temp_db.create_phase("F-TEST002-PH2", feature_id, "API", 2)
    temp_db.create_component(
        "F-TEST002-PH2-C1",
        "F-TEST002-PH2",
        "API Endpoints",
        files=["api/auth.py"],
    )

    # Phase 3: Frontend (depends on Phase 2)
    temp_db.create_phase("F-TEST002-PH3", feature_id, "Frontend", 3)
    temp_db.create_component(
        "F-TEST002-PH3-C1",
        "F-TEST002-PH3",
        "Login Component",
        files=["components/login.tsx"],
    )

    return feature_id


class TestDAGSchedulerBuildGraph:
    """Tests for DAG graph building."""

    def test_build_graph_simple(self, temp_db, simple_feature):
        """Test building graph for simple feature."""
        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(simple_feature)

        assert scheduler._built
        assert len(scheduler._components) == 2
        assert "F-TEST001-PH1-C1" in scheduler._components
        assert "F-TEST001-PH1-C2" in scheduler._components

    def test_build_graph_multi_phase(self, temp_db, multi_phase_feature):
        """Test building graph with multiple phases."""
        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(multi_phase_feature)

        assert scheduler._built
        assert len(scheduler._components) == 4

        # Phase 2 components should depend on all Phase 1 components
        phase2_deps = scheduler._dependencies.get("F-TEST002-PH2-C1", [])
        assert "F-TEST002-PH1-C1" in phase2_deps
        assert "F-TEST002-PH1-C2" in phase2_deps

        # Phase 3 components should depend on all Phase 1 and 2 components
        phase3_deps = scheduler._dependencies.get("F-TEST002-PH3-C1", [])
        assert len(phase3_deps) >= 3  # At least all from phases 1 and 2

    def test_build_graph_explicit_dependency(self, temp_db, simple_feature):
        """Test that explicit dependencies are captured."""
        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(simple_feature)

        # C2 depends on C1 explicitly
        c2_deps = scheduler._dependencies.get("F-TEST001-PH1-C2", [])
        assert "F-TEST001-PH1-C1" in c2_deps


class TestDAGSchedulerCycleDetection:
    """Tests for cycle detection."""

    def test_detects_simple_cycle(self, temp_db):
        """Test detection of A->B->A cycle."""
        feature_id = "F-CYCLE"
        temp_db.create_feature(feature_id, "Cyclic Feature", "Has a cycle")
        temp_db.create_phase("F-CYCLE-PH1", feature_id, "Phase 1", 1)
        temp_db.create_component(
            "F-CYCLE-PH1-C1",
            "F-CYCLE-PH1",
            "Component A",
            depends_on=["F-CYCLE-PH1-C2"],
        )
        temp_db.create_component(
            "F-CYCLE-PH1-C2",
            "F-CYCLE-PH1",
            "Component B",
            depends_on=["F-CYCLE-PH1-C1"],
        )

        scheduler = DAGScheduler(temp_db)
        with pytest.raises(CyclicDependencyError) as exc_info:
            scheduler.build_graph(feature_id)

        assert "Circular dependency" in str(exc_info.value)
        assert "F-CYCLE-PH1-C1" in str(exc_info.value)
        assert "F-CYCLE-PH1-C2" in str(exc_info.value)

    def test_detects_complex_cycle(self, temp_db):
        """Test detection of A->B->C->A cycle."""
        feature_id = "F-CYCLE3"
        temp_db.create_feature(feature_id, "Complex Cycle", "Three-way cycle")
        temp_db.create_phase("F-CYCLE3-PH1", feature_id, "Phase 1", 1)
        temp_db.create_component(
            "F-CYCLE3-PH1-C1",
            "F-CYCLE3-PH1",
            "A",
            depends_on=["F-CYCLE3-PH1-C3"],
        )
        temp_db.create_component(
            "F-CYCLE3-PH1-C2",
            "F-CYCLE3-PH1",
            "B",
            depends_on=["F-CYCLE3-PH1-C1"],
        )
        temp_db.create_component(
            "F-CYCLE3-PH1-C3",
            "F-CYCLE3-PH1",
            "C",
            depends_on=["F-CYCLE3-PH1-C2"],
        )

        scheduler = DAGScheduler(temp_db)
        with pytest.raises(CyclicDependencyError):
            scheduler.build_graph(feature_id)


class TestDAGSchedulerDependencyNotFound:
    """Tests for missing dependency detection."""

    def test_raises_on_missing_dependency(self, temp_db):
        """Test that missing dependency raises error."""
        feature_id = "F-MISSING"
        temp_db.create_feature(feature_id, "Missing Dep Feature", "Has missing dep")
        temp_db.create_phase("F-MISSING-PH1", feature_id, "Phase 1", 1)
        temp_db.create_component(
            "F-MISSING-PH1-C1",
            "F-MISSING-PH1",
            "Component A",
            depends_on=["F-MISSING-PH1-C999"],  # Does not exist
        )

        scheduler = DAGScheduler(temp_db)
        with pytest.raises(DependencyNotFoundError) as exc_info:
            scheduler.build_graph(feature_id)

        assert "F-MISSING-PH1-C999" in str(exc_info.value)
        assert "doesn't exist" in str(exc_info.value)


class TestDAGSchedulerReadyComponents:
    """Tests for get_ready_components()."""

    def test_get_ready_components_initial(self, temp_db, simple_feature):
        """Test that components with no deps are initially ready."""
        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(simple_feature)

        ready = scheduler.get_ready_components()
        assert len(ready) == 1
        assert ready[0].id == "F-TEST001-PH1-C1"

    def test_get_ready_after_completion(self, temp_db, simple_feature):
        """Test that dependent component becomes ready after dependency completes."""
        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(simple_feature)

        # Initially only C1 is ready
        ready = scheduler.get_ready_components()
        assert len(ready) == 1
        assert ready[0].id == "F-TEST001-PH1-C1"

        # Mark C1 as complete
        scheduler.update_component_status(
            "F-TEST001-PH1-C1",
            ComponentStatus.COMPLETED,
            workflow_id=simple_feature,
        )

        # Now C2 should be ready
        ready = scheduler.get_ready_components()
        assert len(ready) == 1
        assert ready[0].id == "F-TEST001-PH1-C2"

    def test_empty_before_build(self, temp_db):
        """Test that get_ready_components returns empty before build."""
        scheduler = DAGScheduler(temp_db)
        assert scheduler.get_ready_components() == []


class TestDAGSchedulerParallelBatches:
    """Tests for get_parallel_batches()."""

    def test_parallel_batches_no_conflict(self, temp_db):
        """Test batching components with no file conflicts."""
        feature_id = "F-PARALLEL"
        temp_db.create_feature(feature_id, "Parallel Feature", "No conflicts")
        temp_db.create_phase("F-PARALLEL-PH1", feature_id, "Phase 1", 1)
        temp_db.create_component(
            "F-PARALLEL-PH1-C1",
            "F-PARALLEL-PH1",
            "A",
            files=["a.py"],
        )
        temp_db.create_component(
            "F-PARALLEL-PH1-C2",
            "F-PARALLEL-PH1",
            "B",
            files=["b.py"],
        )
        temp_db.create_component(
            "F-PARALLEL-PH1-C3",
            "F-PARALLEL-PH1",
            "C",
            files=["c.py"],
        )

        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(feature_id)

        ready = scheduler.get_ready_components()
        batches = scheduler.get_parallel_batches(ready)

        # All three can run in parallel - single batch
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_parallel_batches_with_conflict(self, temp_db):
        """Test batching components with file conflicts."""
        feature_id = "F-CONFLICT"
        temp_db.create_feature(feature_id, "Conflict Feature", "File conflicts")
        temp_db.create_phase("F-CONFLICT-PH1", feature_id, "Phase 1", 1)
        temp_db.create_component(
            "F-CONFLICT-PH1-C1",
            "F-CONFLICT-PH1",
            "A",
            files=["shared.py", "a.py"],
        )
        temp_db.create_component(
            "F-CONFLICT-PH1-C2",
            "F-CONFLICT-PH1",
            "B",
            files=["b.py"],
        )
        temp_db.create_component(
            "F-CONFLICT-PH1-C3",
            "F-CONFLICT-PH1",
            "C",
            files=["shared.py", "c.py"],  # Conflicts with A
        )

        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(feature_id)

        ready = scheduler.get_ready_components()
        batches = scheduler.get_parallel_batches(ready)

        # A and B in first batch, C in second (conflicts with A)
        assert len(batches) == 2
        batch1_ids = {c.id for c in batches[0]}
        batch2_ids = {c.id for c in batches[1]}

        # C1 and C3 should be in different batches
        if "F-CONFLICT-PH1-C1" in batch1_ids:
            assert "F-CONFLICT-PH1-C3" in batch2_ids
        else:
            assert "F-CONFLICT-PH1-C1" in batch2_ids


class TestDAGSchedulerPhaseSequencing:
    """Tests for phase sequencing."""

    def test_phase_sequencing(self, temp_db, multi_phase_feature):
        """Test that Phase 2 components wait for Phase 1."""
        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(multi_phase_feature)

        # Initially only Phase 1 components with no deps are ready
        ready = scheduler.get_ready_components()
        ready_ids = {c.id for c in ready}
        assert "F-TEST002-PH1-C1" in ready_ids  # No deps
        assert "F-TEST002-PH2-C1" not in ready_ids  # Waits for Phase 1
        assert "F-TEST002-PH3-C1" not in ready_ids  # Waits for Phase 2

    def test_empty_phase_handling(self, temp_db):
        """Test that empty intermediate phase doesn't break sequencing."""
        feature_id = "F-EMPTY"
        temp_db.create_feature(feature_id, "Empty Phase Feature", "Has empty phase")

        # Phase 1 with component
        temp_db.create_phase("F-EMPTY-PH1", feature_id, "Phase 1", 1)
        temp_db.create_component(
            "F-EMPTY-PH1-C1",
            "F-EMPTY-PH1",
            "Component A",
            files=["a.py"],
        )

        # Phase 2 - EMPTY (no components)
        temp_db.create_phase("F-EMPTY-PH2", feature_id, "Phase 2 (Empty)", 2)

        # Phase 3 with component
        temp_db.create_phase("F-EMPTY-PH3", feature_id, "Phase 3", 3)
        temp_db.create_component(
            "F-EMPTY-PH3-C1",
            "F-EMPTY-PH3",
            "Component B",
            files=["b.py"],
        )

        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(feature_id)

        # Phase 3 component should still depend on Phase 1 component
        # even with empty Phase 2 in between
        ready = scheduler.get_ready_components()
        ready_ids = {c.id for c in ready}
        assert "F-EMPTY-PH1-C1" in ready_ids
        assert "F-EMPTY-PH3-C1" not in ready_ids

        # After completing Phase 1, Phase 3 should be ready
        scheduler.update_component_status(
            "F-EMPTY-PH1-C1",
            ComponentStatus.COMPLETED,
            workflow_id=feature_id,
        )

        ready = scheduler.get_ready_components()
        ready_ids = {c.id for c in ready}
        assert "F-EMPTY-PH3-C1" in ready_ids


class TestDAGSchedulerFeatureCompletion:
    """Tests for feature completion detection."""

    def test_is_feature_complete(self, temp_db, simple_feature):
        """Test feature completion detection."""
        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(simple_feature)

        assert not scheduler.is_feature_complete()

        # Complete all components
        scheduler.update_component_status(
            "F-TEST001-PH1-C1",
            ComponentStatus.COMPLETED,
            workflow_id=simple_feature,
        )
        scheduler.update_component_status(
            "F-TEST001-PH1-C2",
            ComponentStatus.COMPLETED,
            workflow_id=simple_feature,
        )

        assert scheduler.is_feature_complete()

    def test_is_feature_blocked(self, temp_db, simple_feature):
        """Test blocked feature detection."""
        scheduler = DAGScheduler(temp_db)
        scheduler.build_graph(simple_feature)

        # Not blocked initially (C1 is ready)
        assert not scheduler.is_feature_blocked()

        # Mark C1 as failed
        scheduler.update_component_status(
            "F-TEST001-PH1-C1",
            ComponentStatus.FAILED,
            error="Test failure",
            workflow_id=simple_feature,
        )

        # Now blocked (C2 depends on failed C1)
        assert scheduler.is_feature_blocked()


class TestDAGSchedulerPathNormalization:
    """Tests for file path normalization."""

    def test_normalize_path_relative(self, temp_db):
        """Test normalizing relative paths."""
        scheduler = DAGScheduler(temp_db, repo_path="/test/repo")

        assert scheduler._normalize_path("a.py") == "a.py"
        assert scheduler._normalize_path("./a.py") == "a.py"
        assert scheduler._normalize_path("src/a.py") == "src/a.py"
        assert scheduler._normalize_path("./src/../src/a.py") == "src/a.py"

    def test_normalize_path_absolute(self, temp_db):
        """Test normalizing absolute paths within repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = DAGScheduler(temp_db, repo_path=tmpdir)
            (Path(tmpdir) / "test.py").touch()

            result = scheduler._normalize_path(f"{tmpdir}/test.py")
            assert result == "test.py"

    def test_normalize_path_outside_repo_fails(self, temp_db):
        """Test that paths outside repo raise error."""
        scheduler = DAGScheduler(temp_db, repo_path="/test/repo")

        with pytest.raises(ValueError) as exc_info:
            scheduler._normalize_path("/other/path/file.py")

        assert "outside repo root" in str(exc_info.value)
