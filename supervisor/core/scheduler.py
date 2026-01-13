"""DAG Scheduler for component execution with phase sequencing.

Phase 4 deliverable 4.1: Dependency-aware scheduling of components.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from supervisor.core.state import Database, Event, EventType
from supervisor.core.utils import normalize_repo_path

if TYPE_CHECKING:
    from supervisor.core.models import Component, ComponentStatus

logger = logging.getLogger(__name__)


class SchedulerError(Exception):
    """Error in DAG scheduler."""

    pass


class CyclicDependencyError(SchedulerError):
    """Circular dependency detected in component graph."""

    pass


class WorkflowBlockedError(SchedulerError):
    """Workflow is blocked - no ready components but not all complete."""

    pass


class DependencyNotFoundError(SchedulerError):
    """A declared dependency does not exist."""

    pass


@dataclass
class DependencyEdge:
    """Represents a dependency relationship."""

    # FIX (PR review): Corrected comments - they were swapped
    from_id: str  # The dependency (must complete first)
    to_id: str  # The component that depends on from_id
    edge_type: str  # "explicit" (declared) or "phase" (implicit)


class DAGScheduler:
    """Execute components respecting dependencies AND phase ordering.

    DEPENDENCY SEMANTICS:
    1. Explicit dependencies: Declared in component's depends_on list
    2. Phase dependencies: Components in Phase N+1 depend on ALL Phase N components

    PHASE SEQUENCING:
    - Components within a phase can run in parallel (if no explicit dependencies)
    - Components in later phases wait for ALL earlier phase components

    PARALLEL EXECUTION:
    - get_ready_components() returns all components whose dependencies are met
    - Caller can execute them in parallel (ThreadPoolExecutor)
    - File conflict detection prevents concurrent writes to same file

    Example:
        Feature: Auth System
        ├── Phase 1 (Backend)
        │   ├── Component A: User Model (no deps)
        │   └── Component B: Auth Service (depends on A)
        ├── Phase 2 (API)
        │   └── Component C: Endpoints (depends on Phase 1)

        Execution order:
        1. Component A runs (no deps)
        2. Component B runs (A complete)
        3. Component C runs (Phase 1 complete)
    """

    def __init__(self, db: Database, repo_path: str | Path | None = None):
        self.db = db
        # FIX (Codex v4): Store repo path for path normalization
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        # Adjacency list: node -> list of nodes that depend on it
        self._dependents: dict[str, list[str]] = {}
        # Reverse: node -> list of its dependencies
        self._dependencies: dict[str, list[str]] = {}
        # Component data cache
        self._components: dict[str, Component] = {}
        # Files assigned to each component (for conflict detection) - normalized paths
        self._component_files: dict[str, set[str]] = {}
        # Build state
        self._built = False
        # Thread-safety: Lock for status updates (FIX: Codex review - thread-safety gaps)
        # FIX (Codex fresh review): Use RLock for reentrant locking in nested calls
        self._status_lock = threading.RLock()

    def _normalize_path(self, p: str, fail_closed: bool = True) -> str:
        """Normalize a file path to canonical repo-relative form.

        FIX (PR review): Delegates to shared utility to avoid code duplication.
        Scheduler uses fail_closed=True by default for strict conflict detection.

        Args:
            p: File path (relative or absolute)
            fail_closed: If True, raise on out-of-repo paths (default True)

        Returns:
            Normalized repo-relative path string

        Raises:
            ValueError: If path resolves outside repo root and fail_closed=True
        """
        return normalize_repo_path(p, self.repo_path, fail_closed=fail_closed)

    def build_graph(self, feature_id: str) -> None:
        """Build dependency graph from feature's components.

        ALGORITHM:
        1. Load all phases and components for feature
        2. Add explicit dependencies from component.depends_on
        3. Add implicit phase dependencies (Phase N → Phase N+1)
        4. Validate: no cycles, all dependencies exist

        Args:
            feature_id: Feature to build graph for

        Raises:
            CyclicDependencyError: If circular dependencies detected
            DependencyNotFoundError: If a dependency references non-existent component
            ValueError: If any component file path resolves outside repo root
        """

        # Load data
        phases = self.db.get_phases(feature_id)
        all_components = self.db.get_components(feature_id)

        # Initialize graph structures
        self._dependents = {c.id: [] for c in all_components}
        self._dependencies = {c.id: [] for c in all_components}
        self._components = {c.id: c for c in all_components}
        # FIX (Codex v4): Normalize file paths for conflict detection
        # Ensures ./a.py and a.py are detected as the same file
        self._component_files = {
            c.id: {self._normalize_path(f) for f in c.files} for c in all_components
        }

        # Build component-to-phase mapping
        # FIX (Codex review): Validate phase_id exists before mapping
        component_to_phase: dict[str, str] = {}
        phase_components: dict[str, list[str]] = {p.id: [] for p in phases}
        phase_ids = set(phase_components.keys())
        for comp in all_components:
            if comp.phase_id not in phase_ids:
                raise DependencyNotFoundError(
                    f"Component '{comp.id}' references phase '{comp.phase_id}' which doesn't exist. "
                    f"Available phases: {sorted(phase_ids)}"
                )
            component_to_phase[comp.id] = comp.phase_id
            phase_components[comp.phase_id].append(comp.id)

        # Step 1: Add explicit dependencies
        all_component_ids = set(self._components.keys())
        for comp in all_components:
            for dep_id in comp.depends_on or []:
                if dep_id not in all_component_ids:
                    raise DependencyNotFoundError(
                        f"Component '{comp.id}' depends on '{dep_id}' which doesn't exist. "
                        f"Available components: {sorted(all_component_ids)}"
                    )
                self._add_edge(dep_id, comp.id, "explicit")

        # Step 2: Add implicit phase dependencies
        # FIX (Codex fresh review): Handle empty intermediate phases correctly
        # Link each phase to ALL prior phases' components, not just immediate predecessor
        # This ensures Phase 3 waits for Phase 1 even if Phase 2 is empty
        sorted_phases = sorted(phases, key=lambda p: p.sequence)

        # Collect all components from phases prior to current
        prior_phase_components: list[str] = []

        for phase in sorted_phases:
            curr_comps = phase_components.get(phase.id, [])

            # Every component in current phase depends on all prior phase components
            for curr_id in curr_comps:
                for prev_id in prior_phase_components:
                    # Don't duplicate if explicit edge already exists
                    if prev_id not in self._dependencies[curr_id]:
                        self._add_edge(prev_id, curr_id, "phase")

            # Add current phase components to prior list for next iteration
            prior_phase_components.extend(curr_comps)

        # Step 3: Validate - detect cycles using Kahn's algorithm
        self._validate_no_cycles()
        self._built = True

        logger.info(
            f"Built DAG for feature '{feature_id}': "
            f"{len(all_components)} components, "
            f"{sum(len(deps) for deps in self._dependencies.values())} edges"
        )

    def _add_edge(self, from_id: str, to_id: str, edge_type: str) -> None:
        """Add a dependency edge: from_id must complete before to_id."""
        self._dependents[from_id].append(to_id)
        self._dependencies[to_id].append(from_id)

    def _validate_no_cycles(self) -> None:
        """Validate DAG has no cycles using Kahn's algorithm.

        Raises:
            CyclicDependencyError: If cycle detected, includes cycle members
        """
        # Compute in-degrees
        in_degree = {node: len(deps) for node, deps in self._dependencies.items()}

        # Initialize queue with zero in-degree nodes
        # FIX (PR review): Use deque for O(1) popleft instead of O(n) list.pop(0)
        queue = deque(node for node, deg in in_degree.items() if deg == 0)
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1

            for dependent in self._dependents.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if visited != len(self._components):
            # Remaining nodes with in_degree > 0 form a cycle
            cycle_members = [n for n, deg in in_degree.items() if deg > 0]
            raise CyclicDependencyError(
                f"Circular dependency detected among components: {sorted(cycle_members)}. "
                f"Check depends_on configuration for these components."
            )

    def get_ready_components(self) -> list[Component]:
        """Get components ready for execution (all dependencies satisfied).

        A component is ready when:
        1. Status is 'pending'
        2. All dependencies have status 'complete'

        Returns:
            List of Component objects ready for execution.
            Can be executed in parallel if no file conflicts.

        Note:
            Call build_graph() first or this returns empty list.
            FIX (Codex fresh review): Thread-safe read using _status_lock.
        """
        from supervisor.core.models import ComponentStatus

        if not self._built:
            return []

        # FIX (Codex fresh review): Use lock for thread-safe reads
        # Worker threads may be updating statuses concurrently
        with self._status_lock:
            ready = []
            for comp_id, comp in self._components.items():
                # Skip if not pending
                if comp.status != ComponentStatus.PENDING:
                    continue

                # Check all dependencies are complete
                deps_complete = all(
                    self._components[dep_id].status == ComponentStatus.COMPLETED
                    for dep_id in self._dependencies.get(comp_id, [])
                )

                if deps_complete:
                    ready.append(comp)

            return ready

    def get_parallel_batches(self, ready: list[Component]) -> list[list[Component]]:
        """Group ready components into conflict-free parallel batches.

        Components that modify the same files cannot run in parallel.
        This method groups ready components into batches where each batch
        can be executed in parallel without file conflicts.

        FIX (Codex fresh review): Sort ready components by ID for deterministic batching.
        Without sorting, iteration order depends on dict/set order which can vary.

        Args:
            ready: List of ready components from get_ready_components()

        Returns:
            List of batches. Each batch is a list of components that can
            run in parallel. Batches should be executed sequentially.

        Example:
            ready = [A (files: x.py), B (files: y.py), C (files: x.py)]
            returns: [[A, B], [C]]  # A and B in parallel, then C
        """
        batches: list[list[Component]] = []
        scheduled_files: set[str] = set()
        current_batch: list[Component] = []

        # Sort by component ID for deterministic, repeatable batching
        sorted_ready = sorted(ready, key=lambda c: c.id)

        for comp in sorted_ready:
            # FIX (Codex v4): Use pre-normalized _component_files for conflict detection
            # This ensures ./a.py and a.py are detected as conflicts
            comp_files = self._component_files.get(comp.id, set())

            # Check for file conflict with current batch
            if comp_files & scheduled_files:
                # Conflict - start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [comp]
                scheduled_files = comp_files.copy()
            else:
                # No conflict - add to current batch
                current_batch.append(comp)
                scheduled_files |= comp_files

        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def update_component_status(
        self,
        component_id: str,
        status: ComponentStatus,
        output: str | None = None,
        error: str | None = None,
        workflow_id: str | None = None,
    ) -> None:
        """Update component status and refresh local cache.

        Persists to database and updates local component cache.
        Thread-safe: Uses _status_lock for concurrent updates.

        FIX (Codex review): Added thread-safety with lock to prevent
        race conditions when multiple workers update status concurrently.
        FIX (PAL review): Use append_event instead of non-existent direct updates.

        Args:
            component_id: Component to update
            status: New status
            output: Execution output (for SUCCESS)
            error: Error message (for FAILED)
            workflow_id: Parent workflow ID (phase ID or feature ID)
        """
        from supervisor.core.models import ComponentStatus

        with self._status_lock:
            # Update local cache
            if component_id in self._components:
                self._components[component_id].status = status

            # Persist to database via Event Sourcing
            if status == ComponentStatus.COMPLETED:
                self.db.append_event(
                    Event(
                        workflow_id=workflow_id or "unknown",
                        event_type=EventType.COMPONENT_COMPLETED,
                        component_id=component_id,
                        payload={"output": {"text": output or ""}},
                    )
                )
            elif status == ComponentStatus.FAILED:
                self.db.append_event(
                    Event(
                        workflow_id=workflow_id or "unknown",
                        event_type=EventType.COMPONENT_FAILED,
                        component_id=component_id,
                        payload={"error": error or ""},
                    )
                )
            elif status == ComponentStatus.IMPLEMENTING:
                self.db.append_event(
                    Event(
                        workflow_id=workflow_id or "unknown",
                        event_type=EventType.COMPONENT_STARTED,
                        component_id=component_id,
                    )
                )
            # Other statuses not explicitly handled by events

    def set_component_metadata(
        self,
        component_id: str,
        key: str,
        value: Any,
        workflow_id: str,
    ) -> None:
        """Store arbitrary metadata on a component.

        FIX (v24 - Codex): Added for SKIP persistence in approval flow.
        Used by ApprovalGate to persist SKIP flags for later audit.
        Metadata is stored as JSON in the component's metadata field via event.
        """

        with self._status_lock:
            # Store via event for audit trail
            self.db.append_event(
                Event(
                    workflow_id=workflow_id,
                    event_type=EventType.METADATA_SET,  # FIX (v24): Use METADATA_SET
                    component_id=component_id,
                    payload={
                        "action": "set_metadata",
                        "key": key,
                        "value": value,
                    },
                )
            )

    def is_feature_complete(self) -> bool:
        """Check if all components are complete.

        FIX (Codex fresh review): Thread-safe read using _status_lock.
        """
        from supervisor.core.models import ComponentStatus

        with self._status_lock:
            return all(
                comp.status == ComponentStatus.COMPLETED for comp in self._components.values()
            )

    def get_component(self, component_id: str) -> Component | None:
        """Get a component by ID from the scheduler's cache.

        FIX (PR review): Public method to avoid direct access to _components dict.
        Thread-safe read using _status_lock.

        Args:
            component_id: Component ID to look up

        Returns:
            Component if found, None otherwise
        """
        with self._status_lock:
            return self._components.get(component_id)

    def get_component_count(self) -> int:
        """Get total number of components in the scheduler.

        FIX (PR review): Public method to avoid direct access to _components dict.
        Thread-safe read using _status_lock.

        Returns:
            Total component count
        """
        with self._status_lock:
            return len(self._components)

    def get_completed_count(self) -> int:
        """Get count of completed components.

        FIX (PR review): Public method to avoid direct access to _components dict.
        Thread-safe read using _status_lock.

        Returns:
            Number of components with COMPLETED status
        """
        from supervisor.core.models import ComponentStatus

        with self._status_lock:
            return sum(
                1 for comp in self._components.values() if comp.status == ComponentStatus.COMPLETED
            )

    def is_feature_blocked(self) -> bool:
        """Check if feature is blocked (no ready components but not complete).

        A feature is blocked when:
        1. Not all components are complete
        2. No components are ready to execute
        3. No components are currently in progress

        FIX (Codex review): Also detect "all FAILED" state to avoid infinite loop.
        Previously only checked for pending components, which would return False
        when all components failed, causing the execution loop to spin forever.
        """
        from supervisor.core.models import ComponentStatus

        if self.is_feature_complete():
            return False

        ready = self.get_ready_components()
        if ready:
            return False

        # Check if any components are still in progress
        has_in_progress = any(
            comp.status == ComponentStatus.IMPLEMENTING for comp in self._components.values()
        )
        if has_in_progress:
            return False  # Still working, not blocked

        # Check if there are pending components (blocked by failed deps)
        any(comp.status == ComponentStatus.PENDING for comp in self._components.values())

        # FIX: If no pending and no ready and no in_progress, we're blocked
        # This includes the "all FAILED" case
        return True

    def get_blocking_components(self) -> list[tuple[str, list[str]]]:
        """Get components that are blocking progress.

        FIX (Codex fresh review): Thread-safe read using _status_lock.

        Returns:
            List of (component_id, [failed_dependency_ids]) tuples
            for components that are pending but have failed dependencies.
        """
        from supervisor.core.models import ComponentStatus

        with self._status_lock:
            blocking = []
            for comp_id, comp in self._components.items():
                if comp.status != ComponentStatus.PENDING:
                    continue

                failed_deps = [
                    dep_id
                    for dep_id in self._dependencies.get(comp_id, [])
                    if self._components[dep_id].status == ComponentStatus.FAILED
                ]

                if failed_deps:
                    blocking.append((comp_id, failed_deps))

            return blocking

    def get_execution_order(self) -> list[str]:
        """Get topological sort of all components (for visualization/debugging).

        Returns:
            List of component IDs in valid execution order.
        """
        import heapq

        in_degree = {node: len(deps) for node, deps in self._dependencies.items()}
        heap = [node for node, deg in in_degree.items() if deg == 0]
        heapq.heapify(heap)
        result = []

        while heap:
            node = heapq.heappop(heap)
            result.append(node)

            for dependent in self._dependents.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    heapq.heappush(heap, dependent)

        return result

    def refresh_component(self, component_id: str) -> None:
        """Refresh a component from the database into local cache.

        Useful when external processes update component state.
        """
        with self._status_lock:
            comp = self.db.get_component(component_id)
            if comp:
                self._components[component_id] = comp
