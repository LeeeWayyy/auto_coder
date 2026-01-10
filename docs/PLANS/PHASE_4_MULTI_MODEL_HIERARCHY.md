# Phase 4: Multi-Model & Hierarchy - Implementation Plan

**Status:** APPROVED v10
**Created:** 2026-01-09
**Updated:** 2026-01-09
**Reviewed By:** Codex (v10 - APPROVED), Gemini (fresh - APPROVED)
**Phase:** 4 of 5
**Dependencies:** Phase 1 (Foundation), Phase 2 (Core Workflow), Phase 3 (Gates & Verification)

## Overview

Phase 4 transforms the supervisor from a single-worker orchestrator into a multi-model, hierarchical workflow system. This phase enables complex features to be broken down into phases and components, with parallel execution where dependencies allow, and leverages the strengths of different AI models for different tasks.

### Core Deliverables

From SUPERVISOR_ORCHESTRATOR.md Phase 4:
- [ ] All three CLIs integrated (Claude, Codex, Gemini)
- [ ] Parallel review execution
- [ ] Role-specific context strategies
- [ ] Hierarchical workflow (Feature→Phase→Component)
- [ ] DAG scheduler with phase sequencing enforcement

---

## Current Implementation Status

### Already Implemented (Phase 1-3)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| CLI Adapters | `parser.py:167-266` | Complete | ClaudeAdapter, CodexAdapter, GeminiAdapter |
| SandboxedLLMClient | `executor.py:478-750` | Complete | Supports claude, codex, gemini |
| Feature/Phase/Component Models | `models.py:86-135` | Complete | Pydantic models with status enums |
| Database Tables | `state.py:156-212` | Complete | features, phases, components tables |
| Completion Rollup | `state.py:706-770` | Complete | check_phase_completion, check_feature_completion |
| Role Loader | `roles.py` | Complete | YAML loading with inheritance |
| Context Packer | `context.py` | Complete | Token budgets, priority pruning |
| Gate System | `gates.py`, `gate_*.py` | Complete | Full gate execution with caching |
| RetryPolicy | `engine.py:78-96` | Complete | Exponential backoff with jitter |
| CircuitBreaker | `engine.py:100-170` | Complete | Thread-safe, bounded memory |

### Gaps to Address in Phase 4

1. **DAG Scheduler** - No dependency-aware scheduler for components
2. **Parallel Execution** - No ThreadPoolExecutor for parallel reviews/steps
3. **Context Strategies** - No pluggable, role-specific context strategies
4. **Workflow Orchestration** - No Feature→Phase→Component execution flow
5. **Multi-Model Routing** - No intelligent model selection per task type
6. **Interface Locking** - No mechanism to lock interfaces before parallel work

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW COORDINATOR                          │
│  Manages Feature→Phase→Component hierarchy                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  Feature     │───▶│    Phase     │───▶│  Component   │     │
│   │  (P1T5)      │    │  (P1T5-PH1)  │    │ (P1T5-PH1-C1)│     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                       DAG SCHEDULER                              │
│  Resolves dependencies, enables parallel execution               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────┐      │
│   │              PARALLEL EXECUTOR                        │      │
│   │  ThreadPoolExecutor for concurrent component work     │      │
│   │                                                       │      │
│   │   ┌────────┐  ┌────────┐  ┌────────┐               │      │
│   │   │Worker 1│  │Worker 2│  │Worker 3│               │      │
│   │   │(Claude)│  │(Codex) │  │(Gemini)│               │      │
│   │   └────────┘  └────────┘  └────────┘               │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    CONTEXT STRATEGY                              │
│  Pluggable per-role context selection                            │
│                                                                  │
│   ┌─────────────┐ ┌─────────────────┐ ┌─────────────────┐      │
│   │planner_     │ │implementer_     │ │reviewer_        │      │
│   │docset       │ │targeted         │ │diff             │      │
│   └─────────────┘ └─────────────────┘ └─────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 4 Deliverables

### 4.1 DAG Scheduler

**Goal:** Enable dependency-aware scheduling of components with phase sequencing.

**Files to Create/Modify:**
- `supervisor/core/scheduler.py` (NEW) - DAG scheduler implementation
- `supervisor/core/engine.py` (MODIFY) - Integrate scheduler with execution
- `supervisor/core/state.py` (MODIFY) - Add dependency graph methods

#### Scheduler Design

```python
# supervisor/core/scheduler.py
"""DAG Scheduler for component execution with phase sequencing."""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from supervisor.core.state import Event, EventType

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
    # FIX (PR review): Corrected comments to match implementation
    from_id: str      # The dependency (must complete first)
    to_id: str        # The component that depends on from_id
    edge_type: str    # "explicit" (declared) or "phase" (implicit)


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

    def __init__(self, db: "Database", repo_path: str | None = None):
        self.db = db
        # FIX (Codex v4): Store repo path for path normalization
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        # Adjacency list: node -> list of nodes that depend on it
        self._dependents: dict[str, list[str]] = {}
        # Reverse: node -> list of its dependencies
        self._dependencies: dict[str, list[str]] = {}
        # Component data cache
        self._components: dict[str, "Component"] = {}
        # Files assigned to each component (for conflict detection) - normalized paths
        self._component_files: dict[str, set[str]] = {}
        # Build state
        self._built = False
        # Thread-safety: Lock for status updates (FIX: Codex review - thread-safety gaps)
        # FIX (Codex fresh review): Use RLock for reentrant locking in nested calls
        self._status_lock = threading.RLock()

    def _normalize_path(self, p: str, fail_closed: bool = True) -> str:
        """Normalize a file path to canonical repo-relative form.

        FIX (Codex v4): Ensures consistent path comparison for conflict detection.
        FIX (Codex v5): Fail-closed for out-of-repo paths to prevent missed conflicts.
        ./a.py, a.py, and /full/path/to/repo/a.py all normalize to "a.py".

        Args:
            p: File path (relative or absolute)
            fail_closed: If True, raise on out-of-repo paths (default True)

        Returns:
            Normalized repo-relative path string

        Raises:
            ValueError: If path resolves outside repo root and fail_closed=True
        """
        from pathlib import Path

        path = Path(p)
        repo_root = self.repo_path.resolve()

        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (repo_root / path).resolve()

        try:
            return str(resolved.relative_to(repo_root))
        except ValueError:
            # FIX (Codex v5): Fail-closed - raise instead of warn-and-return
            if fail_closed:
                raise ValueError(
                    f"Path '{p}' resolves outside repo root '{repo_root}'. "
                    "All component files must be within the repository."
                )
            # Only for cases where we want to continue (not used in current impl)
            logger.warning(f"Path '{p}' resolves outside repo root")
            return p

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
        from supervisor.core.models import Phase, Component

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
            c.id: set(self._normalize_path(f) for f in c.files)
            for c in all_components
        }

        # Build component-to-phase mapping
        component_to_phase: dict[str, str] = {}
        phase_components: dict[str, list[str]] = {p.id: [] for p in phases}
        for comp in all_components:
            component_to_phase[comp.id] = comp.phase_id
            phase_components[comp.phase_id].append(comp.id)

        # Step 1: Add explicit dependencies
        all_component_ids = set(self._components.keys())
        for comp in all_components:
            for dep_id in (comp.depends_on or []):
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
        queue = [node for node, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            node = queue.pop(0)
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

    def get_ready_components(self) -> list["Component"]:
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
                    self._components[dep_id].status == ComponentStatus.COMPLETE
                    for dep_id in self._dependencies.get(comp_id, [])
                )

                if deps_complete:
                    ready.append(comp)

            return ready

    def get_parallel_batches(self, ready: list["Component"]) -> list[list["Component"]]:
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
        batches: list[list["Component"]] = []
        scheduled_files: set[str] = set()
        current_batch: list["Component"] = []

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
        status: "ComponentStatus",
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
            if status == ComponentStatus.COMPLETE:
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
            # Other statuses not explicitly handled by events, but could add generic update if needed
            # For now, these cover the key lifecycle transitions

    def is_feature_complete(self) -> bool:
        """Check if all components are complete.

        FIX (Codex fresh review): Thread-safe read using _status_lock.
        """
        from supervisor.core.models import ComponentStatus
        with self._status_lock:
            return all(
                comp.status == ComponentStatus.COMPLETE
                for comp in self._components.values()
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
            comp.status == ComponentStatus.IMPLEMENTING
            for comp in self._components.values()
        )
        if has_in_progress:
            return False  # Still working, not blocked

        # Check if there are pending components (blocked by failed deps)
        has_pending = any(
            comp.status == ComponentStatus.PENDING
            for comp in self._components.values()
        )

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
                    dep_id for dep_id in self._dependencies.get(comp_id, [])
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
```

#### Scheduler Integration with Engine

```python
# Add to supervisor/core/engine.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from supervisor.core.scheduler import DAGScheduler, WorkflowBlockedError
from supervisor.core.state import Event, EventType


class ExecutionEngine:
    # ... existing code ...

    def __init__(
        self,
        # ... existing params ...
        max_parallel_workers: int = 3,
    ):
        # ... existing init ...
        self.max_parallel_workers = max_parallel_workers
        self._scheduler: DAGScheduler | None = None

    def execute_feature(
        self,
        feature_id: str,
        parallel: bool = True,
    ) -> "Feature":
        """Execute all components of a feature in dependency order.

        ALGORITHM:
        1. Build DAG from feature's phases and components
        2. While not complete:
           a. Get ready components (dependencies satisfied)
           b. If parallel: execute batch in ThreadPoolExecutor
           c. If sequential: execute one by one
           d. Update component statuses
        3. Update phase/feature completion status

        Args:
            feature_id: Feature to execute
            parallel: Enable parallel execution (default True)

        Returns:
            Completed Feature object

        Raises:
            WorkflowBlockedError: If no progress can be made
        """
        from supervisor.core.models import Feature, FeatureStatus, ComponentStatus

        # Build DAG - FIX (Codex v5): Pass repo_path for consistent path normalization
        self._scheduler = DAGScheduler(self.db, repo_path=self.repo_path)
        self._scheduler.build_graph(feature_id)

        # Update feature status to in_progress via Event
        self.db.append_event(
             Event(
                 workflow_id=feature_id,
                 event_type=EventType.WORKFLOW_STARTED, # Reusing workflow event for feature start
                 payload={"id": feature_id, "initial_step": "start"},
             )
        )
        # Note: state.py may need a dedicated FEATURE_STARTED event if workflow_started isn't sufficient

        iteration = 0
        # FIX (Codex fresh review): Use component-based limit instead of fixed 1000
        # This scales with feature size and won't trip on large/slow features making progress
        num_components = len(self._scheduler._components)
        max_iterations = max(num_components * 10, 100)  # At least 100, scales with size

        while not self._scheduler.is_feature_complete():
            iteration += 1
            if iteration > max_iterations:
                raise WorkflowBlockedError(
                    f"Feature '{feature_id}' exceeded {max_iterations} iterations "
                    f"({num_components} components). Possible infinite loop."
                )

            # Get ready components
            ready = self._scheduler.get_ready_components()

            if not ready:
                if self._scheduler.is_feature_blocked():
                    blocking = self._scheduler.get_blocking_components()
                    raise WorkflowBlockedError(
                        f"Feature '{feature_id}' is blocked. "
                        f"Components with failed dependencies: {blocking}"
                    )
                # FIX (Codex v4): Add sleep to avoid busy-wait CPU churn
                # All components in-progress, wait before checking again
                import time
                time.sleep(0.5)  # 500ms backoff
                continue

            # Execute ready components
            if parallel and len(ready) > 1:
                self._execute_parallel_batch(ready)
            else:
                for comp in ready:
                    self._execute_component(comp)

        # Update feature status to complete
        # Note: This is handled automatically by _check_feature_completion in state.py
        # when the last phase completes.
        return self.db.get_feature(feature_id)

    def _execute_parallel_batch(self, components: list["Component"]) -> None:
        """Execute a batch of components in parallel.

        Uses ThreadPoolExecutor with max_parallel_workers threads.
        File conflicts are handled by get_parallel_batches() before this.
        """
        from supervisor.core.models import ComponentStatus

        # Get conflict-free batches
        batches = self._scheduler.get_parallel_batches(components)

        for batch in batches:
            if len(batch) == 1:
                # Single component - no parallelism needed
                self._execute_component(batch[0])
            else:
                # Execute batch in parallel
                with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
                    futures = {
                        executor.submit(self._execute_component, comp): comp
                        for comp in batch
                    }

                    for future in as_completed(futures):
                        comp = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Component {comp.id} failed: {e}")
                            # Status already updated in _execute_component

    def _execute_component(self, component: "Component") -> None:
        """Execute a single component.

        Maps component to role execution:
        1. Determine role from component.assigned_role
        2. Build context with component's target files
        3. Run role with component-specific task
        4. Validate file boundaries (FIX: Gemini review)
        5. Update component status based on result
        """
        from supervisor.core.models import ComponentStatus

        try:
            # Update status to implementing
            self._scheduler.update_component_status(
                component.id,
                ComponentStatus.IMPLEMENTING,
                workflow_id=component.phase_id,
            )

            # Determine role and task
            role_name = component.assigned_role or "implementer"
            task_description = f"Implement component: {component.title}\n\n{component.description or ''}"

            # Run the role (existing method)
            result = self.run_role(
                role_name=role_name,
                task_description=task_description,
                workflow_id=component.phase_id,  # Use phase as workflow context
                target_files=component.files,
            )

            # FIX (Gemini review + Codex v2): Validate file boundaries
            # Check that files_changed doesn't exceed declared files
            # FIX (Codex v2): Normalize paths and fail-closed if files_changed missing
            self._validate_file_boundaries(component, result)

            # Success - update status (FIX Codex v3: moved out of _validate_file_boundaries)
            self._scheduler.update_component_status(
                component.id,
                ComponentStatus.COMPLETE,
                output=str(result.model_dump() if hasattr(result, 'model_dump') else result),
                workflow_id=component.phase_id,
            )

            logger.info(f"Component '{component.id}' completed successfully")

        except Exception as e:
            logger.error(f"Component '{component.id}' failed: {e}")
            self._scheduler.update_component_status(
                component.id,
                ComponentStatus.FAILED,
                error=str(e),
                workflow_id=component.phase_id,
            )

    def _validate_file_boundaries(
        self,
        component: "Component",
        result: Any,
    ) -> None:
        """Validate that component only modified declared files.

        FIX (Codex v2): Added path normalization and fail-closed behavior.
        FIX (Codex v3): This method is validation-only, no status updates.
        - Normalizes all paths to repo-relative canonical form
        - Rejects paths outside repo root (fail-closed for declared AND changed)
        - Fails closed if files_changed is missing (requires implementer output schema)

        Args:
            component: Component being validated
            result: Execution result from run_role

        Raises:
            ValueError: If unauthorized files modified or validation fails
        """
        from pathlib import Path

        # Get repo root for normalization
        repo_root = Path(self.repo_path).resolve()

        def normalize_path(p: str) -> str:
            """Normalize path to repo-relative canonical form."""
            path = Path(p)
            if path.is_absolute():
                resolved = path.resolve()
            else:
                resolved = (repo_root / path).resolve()

            # Reject paths outside repo
            try:
                rel = resolved.relative_to(repo_root)
                return str(rel)
            except ValueError:
                raise ValueError(f"Path '{p}' resolves outside repo root")

        # Normalize declared files - FIX (Codex v3): fail-closed, not warn-only
        declared_files = set()
        invalid_declared = []
        for f in component.files:
            try:
                declared_files.add(normalize_path(f))
            except ValueError as e:
                invalid_declared.append(str(e))

        # FIX (Codex v3): Fail-closed on invalid declared files
        if invalid_declared:
            raise ValueError(
                f"Component '{component.id}' has invalid declared files: "
                f"{invalid_declared}. All declared files must be within repo root."
            )

        # Get changed files from result - fail closed if missing
        if not hasattr(result, 'files_changed') or result.files_changed is None:
            # Fail closed: require files_changed for implementer results
            # This ensures we always know what was modified
            raise ValueError(
                f"Component '{component.id}' result missing 'files_changed'. "
                "Implementer outputs must declare modified files for boundary enforcement."
            )

        # Normalize changed files
        changed_files = set()
        for f in result.files_changed:
            try:
                changed_files.add(normalize_path(f))
            except ValueError as e:
                raise ValueError(
                    f"Component '{component.id}' modified file outside repo: {e}"
                )

        # Check for unauthorized modifications
        unauthorized = changed_files - declared_files
        if unauthorized:
            raise ValueError(
                f"Component '{component.id}' modified unauthorized files: "
                f"{unauthorized}. Declared files: {declared_files}"
            )
```

**Acceptance Criteria:**
- [ ] DAG builds correctly from feature/phase/component hierarchy
- [ ] Cycle detection catches circular dependencies
- [ ] get_ready_components() respects both explicit and phase dependencies
- [ ] Parallel batches avoid file conflicts
- [ ] Feature executes to completion with proper status rollup

---

### 4.2 Parallel Review Execution

**Goal:** Execute multiple reviewers simultaneously and aggregate results.

**Files to Create/Modify:**
- `supervisor/core/parallel.py` (NEW) - Parallel execution utilities
- `supervisor/core/engine.py` (MODIFY) - Add parallel review method

#### Parallel Review Implementation

```python
# supervisor/core/parallel.py
"""Parallel execution utilities for multi-model reviews."""

import logging
# FIX (PR review): Import TimeoutError with alias to avoid shadowing built-in
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class ParallelExecutionError(Exception):
    """Error during parallel execution."""
    pass


@dataclass
class ReviewResult:
    """Result from a single reviewer.

    FIX (Codex fresh review): output is Optional since error paths set it to None.
    """
    role_name: str
    cli: str
    output: BaseModel | None  # None on error/timeout
    duration_seconds: float
    success: bool
    error: str | None = None


@dataclass
class AggregatedReviewResult:
    """Aggregated result from multiple parallel reviewers.

    APPROVAL POLICY OPTIONS:
    - ALL_APPROVED: All reviewers must approve
    - ANY_APPROVED: At least one reviewer approves
    - MAJORITY_APPROVED: >50% of reviewers approve
    """
    results: list[ReviewResult]
    approved: bool
    approval_policy: str
    summary: str

    @property
    def all_approved(self) -> bool:
        return all(r.success for r in self.results)

    @property
    def any_approved(self) -> bool:
        return any(r.success for r in self.results)

    @property
    def majority_approved(self) -> bool:
        approved_count = sum(1 for r in self.results if r.success)
        return approved_count > len(self.results) / 2

    def get_rejections(self) -> list[ReviewResult]:
        """Get list of reviewers that rejected."""
        return [r for r in self.results if not r.success]

    def get_issues(self) -> list[str]:
        """Collect all issues from all reviewers.

        FIX (Codex fresh review): Handle None output from failed/timed-out reviewers.
        """
        issues = []
        for result in self.results:
            # Skip if output is None (error/timeout case)
            if result.output is None:
                if result.error:
                    issues.append(f"[{result.role_name}] Error: {result.error}")
                continue
            if hasattr(result.output, 'issues'):
                for issue in result.output.issues:
                    if hasattr(issue, 'description'):
                        issues.append(f"[{result.role_name}] {issue.description}")
                    else:
                        issues.append(f"[{result.role_name}] {issue}")
        return issues


class ParallelReviewer:
    """Execute multiple reviewers in parallel.

    USAGE:
        reviewer = ParallelReviewer(engine, max_workers=3)
        result = reviewer.run_parallel_review(
            roles=["reviewer_gemini", "reviewer_codex"],
            task_description="Review auth implementation",
            target_files=["src/auth.py"],
            approval_policy="ALL_APPROVED",
        )
        if result.approved:
            # Proceed to commit
        else:
            # Handle rejections
    """

    APPROVAL_POLICIES = {"ALL_APPROVED", "ANY_APPROVED", "MAJORITY_APPROVED"}

    def __init__(
        self,
        engine: "ExecutionEngine",
        max_workers: int = 3,
        timeout: float = 300.0,
    ):
        self.engine = engine
        self.max_workers = max_workers
        self.timeout = timeout

    def run_parallel_review(
        self,
        roles: list[str],
        task_description: str,
        workflow_id: str,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
        approval_policy: str = "ALL_APPROVED",
    ) -> AggregatedReviewResult:
        """Run multiple reviewers in parallel and aggregate results.

        Args:
            roles: List of reviewer role names (e.g., ["reviewer_gemini", "reviewer_codex"])
            task_description: What to review
            workflow_id: Workflow context
            target_files: Files to review
            extra_context: Additional context (git_diff, etc.)
            approval_policy: How to determine overall approval

        Returns:
            AggregatedReviewResult with individual results and approval decision
        """
        import time

        if approval_policy not in self.APPROVAL_POLICIES:
            raise ValueError(
                f"Invalid approval_policy '{approval_policy}'. "
                f"Must be one of: {self.APPROVAL_POLICIES}"
            )

        # FIX (Codex fresh review): Validate non-empty roles to avoid false positives
        # all([]) returns True, which would incorrectly approve with no reviewers
        if not roles:
            raise ValueError(
                "At least one reviewer role must be specified. "
                "Empty roles list would result in false approval."
            )

        results: list[ReviewResult] = []

        # FIX (Codex review v2): Use manual executor management to avoid
        # blocking on shutdown. The context manager calls shutdown(wait=True)
        # which defeats the timeout purpose.
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            # Submit all review tasks
            futures: dict[Future, str] = {}
            for role in roles:
                future = executor.submit(
                    self._run_single_review,
                    role,
                    task_description,
                    workflow_id,
                    target_files,
                    extra_context,
                )
                futures[future] = role

            # Collect results as they complete with timeout
            try:
                for future in as_completed(futures, timeout=self.timeout):
                    role = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Reviewer '{role}' failed: {e}")
                        results.append(ReviewResult(
                            role_name=role,
                            cli="unknown",
                            output=None,
                            duration_seconds=0.0,
                            success=False,
                            error=str(e),
                        ))
            except FuturesTimeoutError:
                # Some futures timed out - cancel and return immediately
                logger.warning(f"Parallel review timed out after {self.timeout}s")
                for future, role in futures.items():
                    if not future.done():
                        future.cancel()
                        results.append(ReviewResult(
                            role_name=role,
                            cli="unknown",
                            output=None,
                            duration_seconds=self.timeout,
                            success=False,
                            error=f"Timed out after {self.timeout}s",
                        ))
        finally:
            # Shutdown without waiting - allows immediate return on timeout
            # Note: Running tasks may continue in background but won't block caller
            executor.shutdown(wait=False, cancel_futures=True)

        # FIX (Codex fresh review): Handle empty results as rejection
        # This can happen if all futures fail before any complete
        if not results:
            return AggregatedReviewResult(
                results=[],
                approved=False,
                approval_policy=approval_policy,
                summary="No review results - all reviewers failed or timed out.",
            )

        # Determine overall approval
        if approval_policy == "ALL_APPROVED":
            approved = all(r.success for r in results)
        elif approval_policy == "ANY_APPROVED":
            approved = any(r.success for r in results)
        else:  # MAJORITY_APPROVED
            approved_count = sum(1 for r in results if r.success)
            approved = approved_count > len(results) / 2

        # Build summary
        approved_names = [r.role_name for r in results if r.success]
        rejected_names = [r.role_name for r in results if not r.success]
        summary = f"Approved by: {approved_names}. Rejected by: {rejected_names}."

        return AggregatedReviewResult(
            results=results,
            approved=approved,
            approval_policy=approval_policy,
            summary=summary,
        )

    def _run_single_review(
        self,
        role: str,
        task_description: str,
        workflow_id: str,
        target_files: list[str] | None,
        extra_context: dict[str, str] | None,
    ) -> ReviewResult:
        """Execute a single reviewer and wrap result."""
        import time

        start_time = time.time()
        role_config = self.engine.role_loader.load_role(role)

        try:
            output = self.engine.run_role(
                role_name=role,
                task_description=task_description,
                workflow_id=workflow_id,
                target_files=target_files,
                extra_context=extra_context,
            )

            duration = time.time() - start_time

            # Determine if approved (check output.status or review_status)
            approved = False
            if hasattr(output, 'status'):
                approved = output.status == "APPROVED"
            if hasattr(output, 'review_status'):
                approved = output.review_status == "APPROVED"

            return ReviewResult(
                role_name=role,
                cli=role_config.cli,
                output=output,
                duration_seconds=duration,
                success=approved,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ReviewResult(
                role_name=role,
                cli=role_config.cli,
                output=None,
                duration_seconds=duration,
                success=False,
                error=str(e),
            )
```

**Acceptance Criteria:**
- [ ] Multiple reviewers execute concurrently
- [ ] Results aggregate correctly per approval policy
- [ ] Timeouts handle slow reviewers
- [ ] Errors in one reviewer don't block others

---

### 4.3 Role-Specific Context Strategies

**Goal:** Pluggable context selection strategies per role.

**Files to Create/Modify:**
- `supervisor/core/strategies.py` (NEW) - Context strategy definitions
- `supervisor/config/context_strategies/` (NEW) - Strategy YAML configs
- `supervisor/core/context.py` (MODIFY) - Integrate strategies

#### Strategy System Design

**Integration Note:** Strategies must leverage existing `ContextPacker` capabilities (Repomix, Jinja templates) where applicable, rather than reimplementing basic file packing. The `ContextStrategy.pack` method should ideally configure parameters for `ContextPacker` or use `ContextPacker` as a helper.

```python
# supervisor/core/strategies.py
"""Pluggable context selection strategies."""

import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# FIX (Codex fresh review): Removed unused yaml import

logger = logging.getLogger(__name__)


class StrategyError(Exception):
    """Error in context strategy."""
    pass


@dataclass
class ContextResult:
    """Result of context packing from a strategy."""
    content: str
    token_count: int
    files_included: list[str]
    truncated: bool = False
    truncation_info: str | None = None


class ContextStrategy(ABC):
    """Base class for context selection strategies.

    STRATEGY TYPES:
    - planner_docset: Broad project docs for planning
    - implementer_targeted: Target file + imports + diff
    - reviewer_diff: Git changes + full new files + standards
    - security_audit: Auth code + deps + env handling
    - investigator_wide: Broad scan for exploration
    """

    name: str
    description: str
    token_budget: int

    @abstractmethod
    def pack(
        self,
        repo_path: Path,
        target_files: list[str] | None = None,
        extra_inputs: dict[str, Any] | None = None,
    ) -> ContextResult:
        """Pack context according to this strategy.

        Args:
            repo_path: Repository root path
            target_files: Files being modified (for targeted strategies)
            extra_inputs: Strategy-specific inputs (git_diff, etc.)

        Returns:
            ContextResult with packed content
        """
        pass


class PlannerDocsetStrategy(ContextStrategy):
    """Pack broad project documentation for planning.

    INCLUDES:
    - README.md
    - docs/ARCHITECTURE.md (if exists)
    - Task specifications (docs/TASKS/)
    - ADR documents (docs/ADRs/)
    - File tree (compressed)

    PRIORITY (high to low):
    1. README.md
    2. Task specification
    3. Architecture docs
    4. ADRs
    5. File tree
    """

    name = "planner_docset"
    description = "Broad project docs for planning phase"
    token_budget = 30000

    INCLUDE_PATTERNS = [
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/TASKS/**/*.md",
        "docs/ADRs/**/*.md",
    ]

    PRIORITY_ORDER = [
        "readme",
        "task",
        "architecture",
        "adrs",
        "tree",
    ]

    def pack(
        self,
        repo_path: Path,
        target_files: list[str] | None = None,
        extra_inputs: dict[str, Any] | None = None,
    ) -> ContextResult:
        from supervisor.core.context import ContextPacker

        parts: dict[str, str] = {}
        files_included: list[str] = []

        # README
        readme_path = repo_path / "README.md"
        if readme_path.exists():
            parts["readme"] = f"## README.md\n\n{readme_path.read_text()}"
            files_included.append("README.md")

        # Task specification (if provided in extra_inputs)
        task_spec = extra_inputs.get("task_spec") if extra_inputs else None
        if task_spec:
            parts["task"] = f"## Task Specification\n\n{task_spec}"

        # Architecture docs
        arch_path = repo_path / "docs" / "ARCHITECTURE.md"
        if arch_path.exists():
            parts["architecture"] = f"## Architecture\n\n{arch_path.read_text()}"
            files_included.append("docs/ARCHITECTURE.md")

        # ADRs (collect all)
        adr_dir = repo_path / "docs" / "ADRs"
        if adr_dir.exists():
            adr_content = []
            for adr_file in sorted(adr_dir.glob("*.md")):
                adr_content.append(f"### {adr_file.name}\n\n{adr_file.read_text()}")
                files_included.append(f"docs/ADRs/{adr_file.name}")
            if adr_content:
                parts["adrs"] = "## Architecture Decision Records\n\n" + "\n\n".join(adr_content)

        # File tree (compressed)
        # FIX (PR review): Uses helper method with Python fallback for portability
        tree_output = self._get_file_tree(repo_path)
        if tree_output:
            parts["tree"] = f"## Project Structure\n\n```\n{tree_output}\n```"

        # Combine with budget enforcement
        combined = self._combine_with_budget(parts, self.token_budget)

        return ContextResult(
            content=combined,
            token_count=len(combined) // 4,
            files_included=files_included,
            truncated=len(combined) < sum(len(p) for p in parts.values()),
        )

    def _get_file_tree(self, repo_path: Path, max_depth: int = 3) -> str:
        """Get file tree using tree command or Python fallback.

        FIX (Gemini review): Ensures context is available even without tree command.
        """
        # Try tree command first (better formatting)
        try:
            tree_output = subprocess.run(
                ["tree", "-L", str(max_depth), "-I", "node_modules|.git|__pycache__|.venv|venv"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if tree_output.returncode == 0:
                return tree_output.stdout
        except Exception:
            pass

        # Python fallback using os.walk
        IGNORE_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".pytest_cache"}
        lines = [str(repo_path.name)]

        def walk_tree(path: Path, prefix: str = "", depth: int = 0) -> None:
            if depth >= max_depth:
                return
            entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name))
            entries = [e for e in entries if e.name not in IGNORE_DIRS]
            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir():
                    extension = "    " if is_last else "│   "
                    walk_tree(entry, prefix + extension, depth + 1)

        walk_tree(repo_path)
        return "\n".join(lines)

    def _combine_with_budget(self, parts: dict[str, str], budget: int) -> str:
        """Combine parts within token budget, respecting priority."""
        char_budget = budget * 4

        # Try full content
        full = "\n\n".join(parts.values())
        if len(full) <= char_budget:
            return full

        # Progressive pruning (reverse priority)
        result_parts = parts.copy()
        for key in reversed(self.PRIORITY_ORDER):
            if key in result_parts:
                del result_parts[key]
                current = "\n\n".join(result_parts.values())
                if len(current) <= char_budget:
                    return current

        return "\n\n".join(result_parts.values())


class ImplementerTargetedStrategy(ContextStrategy):
    """Pack target file + resolved imports for implementation.

    INCLUDES:
    - Target file(s) being modified
    - Import dependencies of target files
    - Git diff (configurable source)

    PRIORITY (high to low):
    1. Target files
    2. Direct imports
    3. Git diff
    4. Indirect imports

    FIX (Codex review): Made diff source configurable via extra_inputs["diff_source"].
    Options: "staged" (default), "unstaged", "both", "head"
    """

    name = "implementer_targeted"
    description = "Target file + imports + diff for implementation"
    token_budget = 25000

    # Configurable diff sources
    DIFF_COMMANDS = {
        "staged": ["git", "diff", "--cached"],
        "unstaged": ["git", "diff"],
        "both": ["git", "diff", "HEAD"],
        "head": ["git", "diff", "HEAD~1", "HEAD"],
    }

    def pack(
        self,
        repo_path: Path,
        target_files: list[str] | None = None,
        extra_inputs: dict[str, Any] | None = None,
    ) -> ContextResult:
        parts: dict[str, str] = {}
        files_included: list[str] = []

        # Target files
        if target_files:
            target_content = []
            for tf in target_files:
                path = repo_path / tf
                if path.exists():
                    target_content.append(f"### {tf}\n\n```python\n{path.read_text()}\n```")
                    files_included.append(tf)
                else:
                    target_content.append(f"### {tf}\n\n[File does not exist yet - new file]")
                    files_included.append(tf)
            parts["target_file"] = "## Target Files\n\n" + "\n\n".join(target_content)

        # Resolve imports
        if target_files:
            imports = self._resolve_imports(repo_path, target_files)
            if imports:
                import_content = []
                for imp_file in imports[:10]:  # Limit to 10 imports
                    path = repo_path / imp_file
                    if path.exists():
                        import_content.append(f"### {imp_file}\n\n```python\n{path.read_text()}\n```")
                        files_included.append(imp_file)
                if import_content:
                    parts["imports"] = "## Imported Dependencies\n\n" + "\n\n".join(import_content)

        # Git diff - FIX (Codex review): Configurable diff source
        git_diff = extra_inputs.get("git_diff") if extra_inputs else None
        diff_source = (extra_inputs.get("diff_source", "staged") if extra_inputs else "staged")

        if not git_diff:
            try:
                diff_cmd = self.DIFF_COMMANDS.get(diff_source, self.DIFF_COMMANDS["staged"])
                result = subprocess.run(
                    diff_cmd,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    git_diff = result.stdout
            except Exception:
                pass

        if git_diff:
            source_label = {"staged": "Staged", "unstaged": "Unstaged", "both": "All", "head": "Last Commit"}
            label = source_label.get(diff_source, "Staged")
            parts["git_diff"] = f"## {label} Changes\n\n```diff\n{git_diff}\n```"

        # Combine with budget
        combined = self._combine_with_budget(parts, self.token_budget)

        return ContextResult(
            content=combined,
            token_count=len(combined) // 4,
            files_included=files_included,
        )

    def _resolve_imports(self, repo_path: Path, target_files: list[str]) -> list[str]:
        """Resolve Python imports from target files.

        Uses AST parsing to extract imports and map to file paths.
        """
        import ast

        imports = set()

        for tf in target_files:
            path = repo_path / tf
            if not path.exists() or not tf.endswith('.py'):
                continue

            try:
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
            except SyntaxError:
                continue

        # Convert module names to file paths
        file_paths = []
        for imp in imports:
            # Try direct module file
            parts = imp.split('.')
            potential_paths = [
                '/'.join(parts) + '.py',
                '/'.join(parts) + '/__init__.py',
            ]
            for pp in potential_paths:
                if (repo_path / pp).exists():
                    file_paths.append(pp)
                    break

        return file_paths

    def _combine_with_budget(self, parts: dict[str, str], budget: int) -> str:
        """Combine parts within token budget."""
        char_budget = budget * 4
        priority = ["target_file", "imports", "git_diff"]

        full = "\n\n".join(parts.values())
        if len(full) <= char_budget:
            return full

        result_parts = parts.copy()
        for key in reversed(priority):
            if key in result_parts:
                del result_parts[key]
                current = "\n\n".join(result_parts.values())
                if len(current) <= char_budget:
                    return current

        return "\n\n".join(result_parts.values())


class ReviewerDiffStrategy(ContextStrategy):
    """Pack git changes + standards for code review.

    INCLUDES:
    - Git diff (staged changes)
    - Full content of new files
    - Coding standards docs

    PRIORITY (high to low):
    1. Git diff
    2. New files full content
    3. Standards docs
    """

    name = "reviewer_diff"
    description = "Git changes + full new files + standards for review"
    token_budget = 15000

    def pack(
        self,
        repo_path: Path,
        target_files: list[str] | None = None,
        extra_inputs: dict[str, Any] | None = None,
    ) -> ContextResult:
        parts: dict[str, str] = {}
        files_included: list[str] = []

        # Git diff
        try:
            diff_result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if diff_result.returncode == 0 and diff_result.stdout.strip():
                parts["git_diff"] = f"## Staged Changes\n\n```diff\n{diff_result.stdout}\n```"
        except Exception:
            pass

        # New files (full content)
        try:
            new_files_result = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if new_files_result.returncode == 0 and new_files_result.stdout.strip():
                new_files = new_files_result.stdout.strip().split('\n')
                new_content = []
                for nf in new_files[:5]:  # Limit to 5 new files
                    path = repo_path / nf
                    if path.exists():
                        new_content.append(f"### {nf}\n\n```\n{path.read_text()}\n```")
                        files_included.append(nf)
                if new_content:
                    parts["new_files"] = "## New Files (Full Content)\n\n" + "\n\n".join(new_content)
        except Exception:
            pass

        # Standards docs
        standards_dir = repo_path / "docs" / "STANDARDS"
        if standards_dir.exists():
            standards_content = []
            for std_file in sorted(standards_dir.glob("*.md"))[:3]:  # Limit to 3
                standards_content.append(f"### {std_file.name}\n\n{std_file.read_text()}")
                files_included.append(f"docs/STANDARDS/{std_file.name}")
            if standards_content:
                parts["standards"] = "## Coding Standards\n\n" + "\n\n".join(standards_content)

        # Combine with budget
        combined = self._combine_with_budget(parts, self.token_budget)

        return ContextResult(
            content=combined,
            token_count=len(combined) // 4,
            files_included=files_included,
        )

    def _combine_with_budget(self, parts: dict[str, str], budget: int) -> str:
        char_budget = budget * 4
        priority = ["git_diff", "new_files", "standards"]

        full = "\n\n".join(parts.values())
        if len(full) <= char_budget:
            return full

        result_parts = parts.copy()
        for key in reversed(priority):
            if key in result_parts:
                del result_parts[key]
                current = "\n\n".join(result_parts.values())
                if len(current) <= char_budget:
                    return current

        return "\n\n".join(result_parts.values())


# Strategy registry
STRATEGIES: dict[str, type[ContextStrategy]] = {
    "planner_docset": PlannerDocsetStrategy,
    "implementer_targeted": ImplementerTargetedStrategy,
    "reviewer_diff": ReviewerDiffStrategy,
}


def get_strategy(name: str) -> ContextStrategy:
    """Get a context strategy by name.

    Args:
        name: Strategy name (e.g., "planner_docset")

    Returns:
        Instantiated strategy

    Raises:
        StrategyError: If strategy not found
    """
    if name not in STRATEGIES:
        raise StrategyError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[name]()


def get_strategy_for_role(role_name: str) -> ContextStrategy | None:
    """Get the appropriate strategy for a role.

    MAPPING:
    - planner -> planner_docset
    - implementer -> implementer_targeted
    - reviewer -> reviewer_diff
    - * -> None (use default context packing)
    """
    ROLE_STRATEGY_MAP = {
        "planner": "planner_docset",
        "implementer": "implementer_targeted",
        "reviewer": "reviewer_diff",
        "reviewer_gemini": "reviewer_diff",
        "reviewer_codex": "reviewer_diff",
    }

    strategy_name = ROLE_STRATEGY_MAP.get(role_name)
    if strategy_name:
        return get_strategy(strategy_name)

    # Check for role that extends a base role
    # (Would need RoleLoader integration for full implementation)
    return None
```

**Acceptance Criteria:**
- [ ] Strategies pack correct content per role type
- [ ] Token budgets are respected
- [ ] Import resolution works for Python files
- [ ] Git diff integration works
- [ ] Priority pruning maintains most important content

---

### 4.4 Workflow Orchestration (Feature→Phase→Component)

**Goal:** Full workflow from feature definition to component completion.

**Files to Create/Modify:**
- `supervisor/core/workflow.py` (NEW) - Workflow coordinator
- `supervisor/cli.py` (MODIFY) - Add workflow commands

#### Workflow Coordinator

```python
# supervisor/core/workflow.py
"""Workflow coordinator for Feature→Phase→Component hierarchy."""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from supervisor.core.engine import ExecutionEngine
from supervisor.core.models import (
    Component, ComponentStatus,
    Feature, FeatureStatus,
    Phase, PhaseStatus,
)
from supervisor.core.scheduler import DAGScheduler
from supervisor.core.state import Database

logger = logging.getLogger(__name__)


# FIX (Codex review): Add Pydantic schema for planner output validation
class ComponentPlan(BaseModel):
    """Schema for a component in the planner output."""
    title: str = Field(..., min_length=1, description="Component title")
    files: list[str] = Field(default_factory=list, description="Files to create/modify")
    depends_on: list[str] = Field(default_factory=list, description="Component IDs this depends on")
    role: str = Field(default="implementer", description="Role to execute this component")
    description: str = Field(default="", description="Component description")


class PhasePlan(BaseModel):
    """Schema for a phase in the planner output."""
    title: str = Field(..., min_length=1, description="Phase title")
    components: list[ComponentPlan] = Field(..., min_length=1, description="Components in this phase")
    interfaces: dict[str, Any] = Field(default_factory=dict, description="Interface definitions")


class PlannerOutput(BaseModel):
    """Schema for validating planner role output."""
    phases: list[PhasePlan] = Field(..., min_length=1, description="List of phases")
    summary: str = Field(default="", description="Planning summary")


class WorkflowError(Exception):
    """Error in workflow coordination."""
    pass


class WorkflowCoordinator:
    """Coordinate Feature→Phase→Component workflows.

    WORKFLOW STAGES:
    1. PLANNING: Planner breaks feature into phases
    2. TECH_LEAD: Tech lead breaks phases into components, defines interfaces
    3. IMPLEMENTATION: Components implemented in dependency order
    4. INTEGRATION: Integration tests across components
    5. REVIEW: Final review of complete feature

    USAGE:
        coordinator = WorkflowCoordinator(engine, db)
        feature = coordinator.create_feature(
            title="User Authentication",
            description="Implement user auth with JWT",
        )
        coordinator.run_planning(feature.id)
        coordinator.run_implementation(feature.id)
        coordinator.run_review(feature.id)
    """

    def __init__(
        self,
        engine: ExecutionEngine,
        db: Database,
    ):
        self.engine = engine
        self.db = db

    def create_feature(
        self,
        title: str,
        description: str = "",
    ) -> Feature:
        """Create a new feature for execution.

        Args:
            title: Feature title
            description: Detailed description

        Returns:
            Created Feature object
        """
        import uuid

        feature_id = f"F-{uuid.uuid4().hex[:8].upper()}"

        self.db.create_feature(
            feature_id=feature_id,
            title=title,
            description=description,
        )

        logger.info(f"Created feature '{feature_id}': {title}")
        return self.db.get_feature(feature_id)

    def run_planning(self, feature_id: str) -> list[Phase]:
        """Run planning to break feature into phases.

        Uses planner role to analyze feature and produce phase breakdown.

        Args:
            feature_id: Feature to plan

        Returns:
            List of created Phase objects
        """
        feature = self.db.get_feature(feature_id)

        # Update feature status
        self.db.update_feature_status(feature_id, FeatureStatus.PLANNING)

        # Run planner role
        task = f"""
Analyze and plan the implementation of this feature:

## Feature: {feature.title}

{feature.description}

Break this feature down into sequential phases. Each phase should:
1. Have a clear scope and deliverables
2. Be independently testable
3. Build on previous phases

For each phase, identify:
- Key components to implement
- Dependencies between components
- Interfaces that must be defined before implementation

Output your plan in the required JSON format.
"""

        result = self.engine.run_role(
            role_name="planner",
            task_description=task,
            workflow_id=feature_id,
        )

        # FIX (Codex review): Validate planner output with Pydantic schema
        try:
            if hasattr(result, 'model_dump'):
                validated = PlannerOutput.model_validate(result.model_dump())
            elif isinstance(result, dict):
                validated = PlannerOutput.model_validate(result)
            else:
                raise WorkflowError(
                    f"Planner output is not a valid format: {type(result)}. "
                    "Expected dict or Pydantic model with 'phases' field."
                )
        except ValidationError as e:
            raise WorkflowError(
                f"Planner output validation failed: {e}. "
                "Ensure planner produces valid PhasePlan structure."
            )

        # FIX (Codex fresh review): Two-pass component creation to map dependency IDs
        # Pass 1: Create phases and components, build symbolic->generated ID map
        # Pass 2: Update depends_on with mapped IDs
        phases = []
        symbolic_to_generated: dict[str, str] = {}  # Maps planner's symbolic IDs to generated IDs

        # Pass 1: Create all phases and components
        for i, phase_data in enumerate(validated.phases):
            phase_dict = phase_data.model_dump()
            phase, component_map = self._create_phase_from_plan(
                feature_id=feature_id,
                sequence=i + 1,
                phase_data=phase_dict,
            )
            phases.append(phase)
            symbolic_to_generated.update(component_map)

        # Pass 2: Remap depends_on for all components
        self._remap_component_dependencies(feature_id, symbolic_to_generated)

        logger.info(f"Planning complete for '{feature_id}': {len(phases)} phases created")
        return phases

    def _create_phase_from_plan(
        self,
        feature_id: str,
        sequence: int,
        phase_data: dict[str, Any],
    ) -> tuple[Phase, dict[str, str]]:
        """Create a phase from planner output.

        Returns:
            Tuple of (Phase, symbolic_to_generated_map)
        """
        phase_id = f"{feature_id}-PH{sequence}"

        self.db.create_phase(
            phase_id=phase_id,
            feature_id=feature_id,
            title=phase_data.get("title", f"Phase {sequence}"),
            sequence=sequence,
            interfaces=phase_data.get("interfaces", {}),
        )

        # Create components for this phase, tracking ID mappings
        # FIX (Codex v9 review): Enforce unique symbolic IDs to prevent mis-mapping
        components = phase_data.get("components", [])
        symbolic_map: dict[str, str] = {}

        for j, comp_data in enumerate(components):
            symbolic_id, generated_id = self._create_component_from_plan(
                phase_id=phase_id,
                component_number=j + 1,
                component_data=comp_data,
            )
            if symbolic_id:
                # Check for duplicate symbolic IDs within this phase
                if symbolic_id in symbolic_map:
                    raise WorkflowError(
                        f"Duplicate symbolic_id '{symbolic_id}' in phase '{phase_id}'. "
                        f"Components must have unique symbolic_id or title values. "
                        f"Already mapped to: {symbolic_map[symbolic_id]}, "
                        f"conflicting: {generated_id}"
                    )
                symbolic_map[symbolic_id] = generated_id

        return self.db.get_phase(phase_id), symbolic_map

    def _create_component_from_plan(
        self,
        phase_id: str,
        component_number: int,
        component_data: dict[str, Any],
    ) -> tuple[str | None, str]:
        """Create a component from phase plan output.

        FIX (Codex fresh review): Returns symbolic ID mapping for dependency resolution.
        FIX (Codex v9 review): Prefixes symbolic IDs with phase_id to ensure global uniqueness.

        The planner can specify a 'symbolic_id' field for cross-referencing dependencies.
        If not provided, the title is used as a fallback symbolic ID.
        All symbolic IDs are prefixed with phase_id to ensure uniqueness across phases.

        Returns:
            Tuple of (symbolic_id, generated_id)
        """
        generated_id = f"{phase_id}-C{component_number}"

        # Get symbolic ID from planner output (or use title as fallback)
        raw_symbolic_id = component_data.get("symbolic_id") or component_data.get("title")

        # FIX (Codex v9 review): Prefix with phase_id to ensure global uniqueness
        # This prevents collision when same title appears in different phases
        symbolic_id = f"{phase_id}:{raw_symbolic_id}" if raw_symbolic_id else None

        self.db.create_component(
            component_id=generated_id,
            phase_id=phase_id,
            title=component_data.get("title", f"Component {component_number}"),
            files=component_data.get("files", []),
            depends_on=component_data.get("depends_on", []),  # Will be remapped in pass 2
            assigned_role=component_data.get("role", "implementer"),
        )

        return symbolic_id, generated_id

    def _remap_component_dependencies(
        self,
        feature_id: str,
        symbolic_to_generated: dict[str, str],
    ) -> None:
        """Remap component depends_on from symbolic IDs to generated IDs.

        FIX (Codex fresh review): This second pass ensures dependencies reference
        actual component IDs, not the symbolic names from planner output.
        FIX (Codex v9 review): Handles phase-prefixed symbolic IDs.
        """
        components = self.db.get_components(feature_id)

        for comp in components:
            if not comp.depends_on:
                continue

            # Get component's phase_id for prefixing same-phase deps
            phase_id = comp.phase_id

            remapped_deps = []
            for dep in comp.depends_on:
                # Try direct lookup first (for cross-phase deps with full prefix)
                if dep in symbolic_to_generated:
                    remapped_deps.append(symbolic_to_generated[dep])
                # Try prefixing with current phase (for same-phase deps)
                elif f"{phase_id}:{dep}" in symbolic_to_generated:
                    remapped_deps.append(symbolic_to_generated[f"{phase_id}:{dep}"])
                elif dep.startswith(feature_id):
                    # Already a generated ID format
                    remapped_deps.append(dep)
                else:
                    # FIX (Codex v9 review): Keep unknown deps so DAG builder can catch them
                    logger.warning(
                        f"Component '{comp.id}' depends on unknown '{dep}'. "
                        f"Available: {list(symbolic_to_generated.keys())}"
                    )
                    remapped_deps.append(dep)  # Keep it - DependencyNotFoundError will surface

            if remapped_deps != comp.depends_on:
                self.db.update_component_dependencies(comp.id, remapped_deps)

    def run_implementation(
        self,
        feature_id: str,
        parallel: bool = True,
    ) -> Feature:
        """Run implementation of all components.

        Uses DAG scheduler to execute components in dependency order.

        Args:
            feature_id: Feature to implement
            parallel: Enable parallel execution

        Returns:
            Updated Feature object
        """
        return self.engine.execute_feature(feature_id, parallel=parallel)

    def run_review(
        self,
        feature_id: str,
        reviewers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run final review of completed feature.

        Uses parallel reviewer for multi-model review.

        Args:
            feature_id: Feature to review
            reviewers: List of reviewer roles (default: gemini + codex)

        Returns:
            Review result dict
        """
        from supervisor.core.parallel import ParallelReviewer

        if reviewers is None:
            reviewers = ["reviewer_gemini", "reviewer_codex"]

        feature = self.db.get_feature(feature_id)

        # Collect all files modified
        components = self.db.get_components(feature_id)
        all_files = []
        for comp in components:
            all_files.extend(comp.files)
        all_files = list(set(all_files))

        # Run parallel review
        parallel_reviewer = ParallelReviewer(self.engine)
        result = parallel_reviewer.run_parallel_review(
            roles=reviewers,
            task_description=f"Review complete implementation of feature: {feature.title}",
            workflow_id=feature_id,
            target_files=all_files,
            approval_policy="ALL_APPROVED",
        )

        # Update feature status based on review
        if result.approved:
            self.db.update_feature_status(feature_id, FeatureStatus.COMPLETE)
        else:
            self.db.update_feature_status(feature_id, FeatureStatus.REVIEW)

        return {
            "approved": result.approved,
            "summary": result.summary,
            "issues": result.get_issues(),
        }
```

**Acceptance Criteria:**
- [ ] Features can be created and tracked
- [ ] Planning creates phases and components
- [ ] Implementation follows dependency order
- [ ] Review aggregates results from multiple reviewers

---

### 4.5 Multi-Model Routing

**Goal:** Intelligent model selection based on task type and role.

**Files to Create/Modify:**
- `supervisor/core/routing.py` (NEW) - Model routing logic
- `supervisor/core/engine.py` (MODIFY) - Integrate routing

#### Model Router

```python
# supervisor/core/routing.py
"""Multi-model routing based on task characteristics."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelCapability(str, Enum):
    """Capabilities that different models excel at."""
    REASONING = "reasoning"           # Complex logic, edge cases
    SPEED = "speed"                   # Fast response for simple tasks
    CONTEXT = "context"               # Large context handling
    CODE_GEN = "code_generation"      # Generating new code
    CODE_REVIEW = "code_review"       # Reviewing existing code
    DOCUMENTATION = "documentation"   # Writing docs
    PLANNING = "planning"             # Architecture, planning


@dataclass
class ModelProfile:
    """Profile of a model's capabilities."""
    name: str
    cli: str
    strengths: list[ModelCapability]
    max_context: int  # Approximate token limit
    relative_speed: float  # 1.0 = baseline
    relative_cost: float  # 1.0 = baseline


# Model capability profiles
MODEL_PROFILES = {
    "claude": ModelProfile(
        name="Claude",
        cli="claude",
        strengths=[
            ModelCapability.REASONING,
            ModelCapability.PLANNING,
            ModelCapability.CODE_REVIEW,
        ],
        max_context=200000,
        relative_speed=1.0,
        relative_cost=1.0,
    ),
    "codex": ModelProfile(
        name="Codex",
        cli="codex",
        strengths=[
            ModelCapability.SPEED,
            ModelCapability.CODE_GEN,
        ],
        max_context=128000,
        relative_speed=2.0,  # Faster
        relative_cost=0.5,   # Cheaper
    ),
    "gemini": ModelProfile(
        name="Gemini",
        cli="gemini",
        strengths=[
            ModelCapability.CONTEXT,
            ModelCapability.DOCUMENTATION,
            ModelCapability.CODE_REVIEW,
        ],
        max_context=1000000,
        relative_speed=0.8,
        relative_cost=0.7,
    ),
}


# Role to model mapping (default)
ROLE_MODEL_MAP = {
    "planner": "claude",
    "implementer": "claude",
    "implementer_fast": "codex",
    "reviewer": "claude",
    "reviewer_gemini": "gemini",
    "reviewer_codex": "codex",
    "investigator": "gemini",
    "doc_generator": "gemini",
}


class ModelRouter:
    """Route tasks to appropriate models.

    ROUTING STRATEGY:
    1. If role specifies CLI, use that
    2. Otherwise, match task type to model strengths
    3. Consider context size (large context → Gemini)
    4. Consider speed requirements (fast → Codex)
    """

    def __init__(self, prefer_speed: bool = False, prefer_cost: bool = False):
        self.prefer_speed = prefer_speed
        self.prefer_cost = prefer_cost

    def select_model(
        self,
        role_name: str,
        context_size: int = 0,
        task_type: ModelCapability | None = None,
        role_cli: str | None = None,
    ) -> str:
        """Select the best model for a task.

        Args:
            role_name: Role being executed
            context_size: Estimated context tokens
            task_type: Primary capability needed
            role_cli: CLI specified in role config (takes precedence)

        Returns:
            CLI name to use (claude, codex, gemini)
        """
        # Role config takes precedence
        if role_cli:
            return role_cli

        # Default mapping
        if role_name in ROLE_MODEL_MAP:
            default = ROLE_MODEL_MAP[role_name]
        else:
            default = "claude"

        # Check context size requirements
        if context_size > 128000:
            # Need large context - Gemini
            return "gemini"

        # Speed preference
        if self.prefer_speed and task_type in [ModelCapability.CODE_GEN, ModelCapability.SPEED]:
            return "codex"

        # Cost preference
        if self.prefer_cost:
            # Choose cheapest that has required capability
            if task_type in MODEL_PROFILES["codex"].strengths:
                return "codex"
            if task_type in MODEL_PROFILES["gemini"].strengths:
                return "gemini"

        return default

    def get_profile(self, cli: str) -> ModelProfile | None:
        """Get model profile by CLI name."""
        return MODEL_PROFILES.get(cli)
```

**Acceptance Criteria:**
- [ ] Models selected based on task requirements
- [ ] Context size influences model choice
- [ ] Speed/cost preferences respected
- [ ] Role config CLI takes precedence

---

## Integration Points

### Dependencies Between Deliverables

```
4.1 DAG Scheduler
      │
      ▼
4.4 Workflow Orchestration ◄──── 4.2 Parallel Review
      │
      ▼
4.3 Context Strategies
      │
      ▼
4.5 Multi-Model Routing
```

### New Dependencies

```toml
# Add to pyproject.toml
# FIX (Gemini review): Removed networkx - DAGScheduler uses custom Kahn's
# algorithm implementation, no external dependency needed.
dependencies = [
    # ... existing ...
    # No new dependencies required for Phase 4
]
```

---

## Testing Strategy

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_scheduler.py` | DAG building, cycle detection, ready components |
| `tests/test_parallel.py` | Parallel review execution, result aggregation |
| `tests/test_strategies.py` | Context strategies, token budgets |
| `tests/test_workflow.py` | Feature→Phase→Component flow |
| `tests/test_routing.py` | Model selection logic |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_full_feature_workflow` | Create feature → plan → implement → review |
| `test_parallel_component_execution` | Execute independent components in parallel |
| `test_dependency_ordering` | Components run in correct dependency order |
| `test_multi_model_review` | All three models review same code |

### Example Test Cases

```python
# tests/test_scheduler.py
def test_dag_detects_cycle():
    """Circular dependency raises CyclicDependencyError."""
    # Component A depends on B, B depends on A
    # Should raise CyclicDependencyError


def test_phase_sequencing():
    """Components in Phase 2 wait for all Phase 1 components."""
    # Phase 1: [A, B]
    # Phase 2: [C]
    # C should not be ready until both A and B complete


def test_parallel_batches_avoid_conflicts():
    """Components modifying same file run sequentially."""
    # A modifies x.py, B modifies y.py, C modifies x.py
    # Batches should be [[A, B], [C]] not [[A, B, C]]
```

---

## Security Considerations

### Parallel Execution Safety

```python
# File locking for parallel execution
# Each component writes to isolated files (validated by scheduler)
# Shared state (database) uses transactions
# Git operations are serialized with worktree locks
```

### Model API Security

```python
# All CLI invocations go through SandboxedLLMClient
# Network egress limited to model API endpoints
# No credentials in context (secret scanning from Phase 2)
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Race conditions in parallel execution | Medium | High | File conflict detection, worktree isolation |
| Context overflow with multiple files | Medium | Medium | Strategy budgets, progressive pruning |
| Model API rate limits | Medium | Medium | Retry with backoff, parallel limits |
| DAG cycle detection false positives | Low | Low | Kahn's algorithm is well-understood |
| Deadlock in parallel review | Low | High | Timeouts, independent executor threads |
| Apply lock serialization bottleneck | Medium | Medium | Document limitation, future optimization |

### Known Performance Bottleneck: Apply Lock Serialization

**FIX (Gemini review):** The existing `_apply_lock` in `executor.py` serializes all
file application operations. This means that while components can be executed in
parallel (LLM calls), the final step of applying changes to files is serialized.

**Current behavior:**
```python
# In executor.py (existing code)
async def _apply_changes(self, changes: list[FileChange]) -> None:
    async with self._apply_lock:  # <-- All applications serialized here
        for change in changes:
            await self._write_file(change)
```

**Impact:**
- Parallel component execution benefits are reduced during the apply phase
- For N components completing simultaneously, apply time = N × single_apply_time
- This is acceptable for Phase 4 as most time is spent in LLM calls, not file I/O

**Future optimization (Phase 5+):**
- Use per-file or per-directory locks instead of global lock
- Or: Batch all changes and apply in single transaction
- Or: Use git worktrees for true parallel isolated writes

---

## Completion Criteria

Phase 4 is complete when:

1. [ ] DAG scheduler correctly orders components by dependencies
2. [ ] Parallel execution runs independent components concurrently
3. [ ] Context strategies pack appropriate content per role
4. [ ] Feature→Phase→Component workflow executes end-to-end
5. [ ] Multi-model routing selects appropriate model per task
6. [ ] All unit tests pass
7. [ ] Integration tests demonstrate full workflow
8. [ ] No race conditions in parallel execution

---

## CLI Commands (Preview)

```bash
# Create a new feature
supervisor feature create "User Authentication" --description "JWT-based auth"

# Run planning
supervisor feature plan F-ABC12345

# Execute implementation
supervisor feature implement F-ABC12345 --parallel

# Run review
supervisor feature review F-ABC12345 --reviewers gemini,codex

# Check status
supervisor feature status F-ABC12345

# List all features
supervisor feature list
```

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-09 | Claude | Initial Phase 4 plan based on SUPERVISOR_ORCHESTRATOR.md |
| 2026-01-09 | Claude | v2: Applied fixes from Codex/Gemini review |
| 2026-01-09 | Claude | v3: Applied Codex v2 review fixes |
| 2026-01-09 | Claude | v4: Applied Codex v3 review fixes |
| 2026-01-09 | Claude | v5: Applied Codex v4 review fixes |
| 2026-01-09 | Claude | v6: Applied Codex v5 review fixes |
| 2026-01-09 | Claude | v7: Applied Codex v6 non-blocking docstring fix |
| 2026-01-09 | Claude | v8: Gemini applied Event Sourcing fixes |
| 2026-01-09 | Claude | v9: Fresh review fixes (High: dep mapping, Medium: 5 items, Low: 3 items) |
| 2026-01-09 | Claude | v10: Applied Codex v9 fixes (unknown deps, unique symbolic IDs) - APPROVED |

### v10 Fixes Applied (from Codex v9 review):

**High Priority:**
1. Keep unknown dependencies for DAG validation
   - Previously unknown deps were silently dropped, hiding planner errors
   - Now appended to remapped_deps so DependencyNotFoundError surfaces

**Medium Priority:**
2. Enforce unique symbolic IDs with phase prefixing
   - All symbolic IDs prefixed with `phase_id:` for global uniqueness
   - Duplicate detection raises WorkflowError with clear message
   - Updated remap logic to handle phase-prefixed lookups

**Dependency Resolution Rules:**
- Unprefixed deps (e.g., "auth_service") = same-phase lookup
- Prefixed deps (e.g., "F-123-PH1:auth_service") = cross-phase reference

### v9 Fixes Applied (from Codex fresh review):

**High Priority:**
1. Two-pass component creation for dependency ID mapping
   - Pass 1: Create phases/components, build symbolic→generated ID map
   - Pass 2: Remap depends_on with actual generated IDs

**Medium Priority:**
2. Fixed phase sequencing for empty phases (link ALL prior phases)
3. Component-based max_iterations (scales with feature size)
4. Empty roles validation in parallel review
5. Thread-safe reads using RLock

**Low Priority:**
6. ReviewResult.output type to `BaseModel | None`
7. Deterministic batching via component ID sorting
8. Removed unused yaml import

### v6 Fixes Applied (from Codex v5 review):

**Medium Priority:**
1. Pass `repo_path` from ExecutionEngine to DAGScheduler
   - Changed `DAGScheduler(self.db)` to `DAGScheduler(self.db, repo_path=self.repo_path)`
   - Ensures scheduler path normalization uses same repo root as boundary validation

**Low Priority:**
2. Made `_normalize_path` fail-closed for out-of-repo paths
   - Now raises `ValueError` instead of warning and returning raw path
   - Prevents scheduling conflicts from being missed due to different path forms
   - Added `fail_closed` parameter (defaults to True) for flexibility

### v5 Fixes Applied (from Codex v4 review):

**Medium Priority:**
1. Normalized file paths in scheduler for conflict detection
   - Added `_normalize_path()` method to DAGScheduler
   - `_component_files` now stores normalized paths at build time
   - `get_parallel_batches()` uses normalized paths for conflict detection
   - Ensures `./a.py`, `a.py`, and `/full/path/a.py` are detected as same file

**Low Priority:**
2. Added sleep/backoff in all-in-progress loop
   - Added `time.sleep(0.5)` when no ready components and work is in-progress
   - Prevents CPU churn from busy-wait loop

### v4 Fixes Applied (from Codex v3 review):

**High Priority:**
1. Fixed success path and try/except structure in `_execute_component`
   - Moved success status update out of `_validate_file_boundaries` into `_execute_component`
   - Fixed try/except indentation - exception handler now properly catches validation errors
   - `_validate_file_boundaries` is now validation-only (no status updates)

**Medium Priority:**
2. Made declared file validation fail-closed (not warn-only)
   - Invalid declared files now raise `ValueError` instead of logging warning
   - Collects all invalid paths and reports them in a single error message

### v3 Fixes Applied (from Codex v2 review):

**High Priority:**
1. Fixed timeout handling to use manual executor shutdown with `wait=False, cancel_futures=True`
   - Prevents blocking on shutdown when timeout occurs
   - Allows immediate return with partial results

**Medium Priority:**
2. Added path normalization in file boundary validation using `Path.resolve()`
   - Normalizes `./`, `../`, and absolute paths to repo-relative form
   - Rejects paths that resolve outside repo root
3. Made `files_changed` validation fail-closed (mandatory)
   - Implementer outputs must declare modified files
   - Fails if `files_changed` is missing or None

### v2 Fixes Applied (from Codex/Gemini review):

**High Priority (Codex):**
1. Fixed `is_feature_blocked()` to detect "all FAILED" state (prevents infinite loop)
2. Made `update_component_status()` thread-safe with `_status_lock`
3. Added `TimeoutError` handling in `run_parallel_review()` with future cancellation

**Medium Priority (Codex):**
4. Added file boundary validation in `_execute_component()` to enforce declared files
5. Added `PlannerOutput` Pydantic schema for planner output validation
6. Made diff source configurable in `ImplementerTargetedStrategy` (staged/unstaged/both/head)

**Gemini Review (APPROVED in v2):**
7. Removed unused networkx dependency (DAGScheduler uses custom Kahn's algorithm)
8. Documented `_apply_lock` serialization bottleneck in Risk Assessment section
9. Added file boundary enforcement at runtime (validates `files_changed` against declared)
