"""Workflow coordinator for Feature->Phase->Component hierarchy.

Phase 4 deliverable 4.4: Full workflow orchestration from feature to completion.
"""

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from supervisor.core.models import (
    Component,
    ComponentStatus,
    Feature,
    FeatureStatus,
    Phase,
    PhaseStatus,
)
from supervisor.core.routing import ModelRouter, create_router, AdaptiveConfig
from supervisor.core.scheduler import (
    DAGScheduler,
    WorkflowBlockedError,
)
from supervisor.core.state import Database, Event, EventType
from supervisor.core.utils import normalize_repo_path

if TYPE_CHECKING:
    from supervisor.core.engine import ExecutionEngine
    from supervisor.core.interaction import InteractionBridge
    from supervisor.core.approval import ApprovalGate

logger = logging.getLogger(__name__)


# --- Pydantic Schemas for Planner Output Validation ---


class ComponentPlan(BaseModel):
    """Schema for a component in the planner output."""

    title: str = Field(..., min_length=1, description="Component title")
    symbolic_id: str | None = Field(
        default=None, description="Optional symbolic ID for dependency references"
    )
    files: list[str] = Field(
        default_factory=list, description="Files to create/modify"
    )
    depends_on: list[str] = Field(
        default_factory=list, description="Component IDs this depends on"
    )
    role: str = Field(default="implementer", description="Role to execute this component")
    description: str = Field(default="", description="Component description")


class PhasePlan(BaseModel):
    """Schema for a phase in the planner output."""

    title: str = Field(..., min_length=1, description="Phase title")
    components: list[ComponentPlan] = Field(
        ..., min_length=1, description="Components in this phase"
    )
    interfaces: dict[str, Any] = Field(
        default_factory=dict, description="Interface definitions"
    )


class PlannerOutput(BaseModel):
    """Schema for validating planner role output."""

    phases: list[PhasePlan] = Field(..., min_length=1, description="List of phases")
    summary: str = Field(default="", description="Planning summary")


class WorkflowError(Exception):
    """Error in workflow coordination."""

    pass


class WorkflowCoordinator:
    """Coordinate Feature->Phase->Component workflows.

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
        engine: "ExecutionEngine",
        db: Database,
        repo_path: str | Path | None = None,
        max_parallel_workers: int = 3,
        prefer_speed: bool = False,
        prefer_cost: bool = False,
        max_stall_seconds: float = 600.0,
        component_timeout: float = 300.0,
        # NEW Phase 5 params
        workflow_timeout: float = 3600.0,
        checkpoint_on_timeout: bool = True,
        role_timeouts: dict[str, float] | None = None,
        approval_gate: "ApprovalGate | None" = None,
        interaction_bridge: "InteractionBridge | None" = None,
        adaptive_config: dict[str, Any] | None = None,
        # FIX (v27 - Gemini PR review): Configurable git subprocess timeout
        git_timeout: float = 60.0,
    ):
        self.engine = engine
        self.db = db
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.max_parallel_workers = max_parallel_workers
        # FIX (PR review): Make timeout values configurable
        self.max_stall_seconds = max_stall_seconds  # Max time without progress (default 10 min)
        self.component_timeout = component_timeout  # Per-component timeout (default 5 min)
        self.git_timeout = git_timeout  # Git subprocess timeout (default 60s)
        self._scheduler: DAGScheduler | None = None
        
        # Parse adaptive config
        self.adaptive_config = None
        if adaptive_config:
            self.adaptive_config = AdaptiveConfig(**adaptive_config)

        # FIX (Gemini review): Integrate ModelRouter with adaptive support
        self._router = create_router(
            prefer_speed=prefer_speed, 
            prefer_cost=prefer_cost,
            db=self.db,
            adaptive_config=self.adaptive_config,
        )

        # NEW Phase 5 initialization
        self.workflow_timeout = workflow_timeout or 3600.0
        self.checkpoint_on_timeout = checkpoint_on_timeout
        self.role_timeouts = role_timeouts or {}
        self._workflow_start_time: float | None = None

        self.approval_gate = approval_gate
        self.interaction_bridge = interaction_bridge

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
        feature_id = f"F-{uuid.uuid4().hex[:8].upper()}"

        self.db.create_feature(
            feature_id=feature_id,
            title=title,
            description=description,
        )

        logger.info(f"Created feature '{feature_id}': {title}")
        return self.db.get_feature(feature_id)  # type: ignore

    def run_planning(self, feature_id: str) -> list[Phase]:
        """Run planning to break feature into phases.

        Uses planner role to analyze feature and produce phase breakdown.

        Args:
            feature_id: Feature to plan

        Returns:
            List of created Phase objects
        """
        feature = self.db.get_feature(feature_id)
        if not feature:
            raise WorkflowError(f"Feature '{feature_id}' not found")

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

Output your plan in the required JSON format with 'phases' array.
Each phase needs 'title' and 'components' array.
Each component needs 'title', 'files', 'depends_on', 'role'.
Use 'symbolic_id' for cross-referencing dependencies within same phase.
"""

        result = self.engine.run_role(
            role_name="planner",
            task_description=task,
            workflow_id=feature_id,
        )

        # FIX (Codex review): Validate planner output with Pydantic schema
        try:
            if hasattr(result, "model_dump"):
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
        symbolic_to_generated: dict[str, str] = {}

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

        # Update feature status to in_progress
        self.db.update_feature_status(feature_id, FeatureStatus.IN_PROGRESS)

        logger.info(
            f"Planning complete for '{feature_id}': {len(phases)} phases created"
        )
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

        # FIX (PR review): Convert interfaces dict to list format for Phase model
        # Planner may return {"name": {type, definition}} but Phase expects list[Interface]
        # FIX (Codex review): Ensure 'type' field is present (required by Interface model)
        raw_interfaces = phase_data.get("interfaces", {})
        if isinstance(raw_interfaces, dict):
            interfaces_list = []
            for name, props in raw_interfaces.items():
                if isinstance(props, dict):
                    # Ensure type field exists with default
                    interface = {"name": name, "type": props.get("type", "unknown"), **props}
                else:
                    # String value - use as definition, default type
                    interface = {"name": name, "type": "unknown", "definition": str(props)}
                interfaces_list.append(interface)
        else:
            # Already a list - ensure each item has type field
            interfaces_list = []
            for item in (raw_interfaces if isinstance(raw_interfaces, list) else []):
                if isinstance(item, dict):
                    if "type" not in item:
                        item = {**item, "type": "unknown"}
                    interfaces_list.append(item)
                # Skip non-dict items

        self.db.create_phase(
            phase_id=phase_id,
            feature_id=feature_id,
            title=phase_data.get("title", f"Phase {sequence}"),
            sequence=sequence,
            interfaces=interfaces_list,
        )

        # Create components for this phase, tracking ID mappings
        # FIX (Codex v9 review): Enforce unique symbolic IDs to prevent mis-mapping
        components = phase_data.get("components", [])
        symbolic_map: dict[str, str] = {}

        for j, comp_data in enumerate(components):
            symbolic_id, generated_id = self._create_component_from_plan(
                feature_id=feature_id,
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

        return self.db.get_phase(phase_id), symbolic_map  # type: ignore

    def _create_component_from_plan(
        self,
        feature_id: str,
        phase_id: str,
        component_number: int,
        component_data: dict[str, Any],
    ) -> tuple[str | None, str]:
        """Create a component from phase plan output.

        FIX (Codex fresh review): Returns symbolic ID mapping for dependency resolution.
        FIX (Codex v9 review): Prefixes symbolic IDs with phase_id to ensure global uniqueness.
        FIX (PR review): Accepts feature_id to avoid unnecessary DB lookup in create_component.

        The planner can specify a 'symbolic_id' field for cross-referencing dependencies.
        If not provided, the title is used as a fallback symbolic ID.
        All symbolic IDs are prefixed with phase_id to ensure uniqueness across phases.

        Returns:
            Tuple of (symbolic_id, generated_id)
        """
        generated_id = f"{phase_id}-C{component_number}"

        # Get symbolic ID from planner output (or use title as fallback)
        raw_symbolic_id = component_data.get("symbolic_id") or component_data.get(
            "title"
        )

        # FIX (Codex v9 review): Prefix with phase_id to ensure global uniqueness
        # This prevents collision when same title appears in different phases
        symbolic_id = f"{phase_id}:{raw_symbolic_id}" if raw_symbolic_id else None

        # FIX (PR review): Pass feature_id directly to avoid DB lookup
        self.db.create_component(
            component_id=generated_id,
            phase_id=phase_id,
            title=component_data.get("title", f"Component {component_number}"),
            files=component_data.get("files", []),
            depends_on=component_data.get("depends_on", []),  # Will be remapped in pass 2
            assigned_role=component_data.get("role", "implementer"),
            description=component_data.get("description", ""),
            feature_id=feature_id,
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
        FIX (Gemini/Codex review): Fail fast on unknown dependencies with clear error.
        """
        components = self.db.get_components(feature_id)

        # Build raw symbolic ID map for cross-phase resolution
        # This handles the case where planner uses unprefixed IDs for cross-phase deps
        raw_to_generated: dict[str, list[str]] = {}
        for prefixed_id, generated_id in symbolic_to_generated.items():
            if ":" in prefixed_id:
                _, raw_id = prefixed_id.split(":", 1)
                if raw_id not in raw_to_generated:
                    raw_to_generated[raw_id] = []
                raw_to_generated[raw_id].append(generated_id)

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
                # FIX (Codex review): Try raw ID lookup for cross-phase deps
                elif dep in raw_to_generated:
                    matches = raw_to_generated[dep]
                    if len(matches) == 1:
                        remapped_deps.append(matches[0])
                    else:
                        # Multiple matches - ambiguous, fail fast
                        raise WorkflowError(
                            f"Component '{comp.id}' depends on ambiguous symbolic ID '{dep}'. "
                            f"Multiple components match: {matches}. "
                            f"Use phase-prefixed ID (e.g., 'PHASE_ID:{dep}') to disambiguate."
                        )
                else:
                    # FIX (Gemini review): Fail fast instead of warning and crashing later
                    raise WorkflowError(
                        f"Component '{comp.id}' depends on unknown symbolic ID '{dep}'. "
                        f"Available symbolic IDs: {sorted(symbolic_to_generated.keys())}. "
                        f"Check planner output for typos or missing components."
                    )

            if remapped_deps != list(comp.depends_on):
                self.db.update_component_dependencies(comp.id, remapped_deps)

    def run_implementation(
        self,
        feature_id: str,
        parallel: bool = True,
    ) -> Feature:
        """Run implementation of all components.

        Uses DAG scheduler to execute components in dependency order.
        FIX (Gemini review): Uses continuous scheduling instead of batch-wait.
        As each component completes, immediately checks for newly ready components.

        Args:
            feature_id: Feature to implement
            parallel: Enable parallel execution

        Returns:
            Updated Feature object
        """
        # FIX (v15 - Codex): ALWAYS reset timer per run_implementation call
        self._workflow_start_time = time.time()

        # Check workflow timeout (elapsed will be ~0 at start, this is for resume)
        elapsed = time.time() - self._workflow_start_time
        if elapsed > self.workflow_timeout:
            self._handle_workflow_timeout(feature_id, elapsed)

        # Build DAG - FIX (Codex v5): Pass repo_path for consistent path normalization
        self._scheduler = DAGScheduler(self.db, repo_path=self.repo_path)
        self._scheduler.build_graph(feature_id)

        # Update feature status
        self.db.update_feature_status(feature_id, FeatureStatus.IN_PROGRESS)

        # FIX (PR review): Use public method instead of accessing _components directly
        num_components = self._scheduler.get_component_count()

        try:
            workflow_deadline = self._workflow_start_time + self.workflow_timeout
            if not parallel:
                # Sequential execution - simple loop with timeout
                self._run_sequential_with_timeout(feature_id, workflow_deadline=workflow_deadline)
            else:
                # FIX (Gemini review): Continuous parallel scheduling
                self._run_continuous_parallel(
                    feature_id,
                    num_components,
                    workflow_deadline=workflow_deadline
                )

        except Exception as e:
            # Check for CancellationError which might be raised by timeouts
            if type(e).__name__ == "CancellationError":
                # Checkpoint already saved in _handle_workflow_timeout
                raise
            raise e

        # Feature complete
        self.db.update_feature_status(feature_id, FeatureStatus.REVIEW)
        logger.info(f"Implementation complete for '{feature_id}'")
        return self.db.get_feature(feature_id)  # type: ignore

    def _run_sequential_with_timeout(
        self,
        feature_id: str,
        workflow_deadline: float | None = None,
    ) -> None:
        """Run components sequentially with workflow and role timeout checks."""
        iteration = 0
        max_iterations = max(self._scheduler.get_component_count() * 10, 100)

        while not self._scheduler.is_feature_complete():
            # NEW (v9): Check workflow-level timeout before each component
            if workflow_deadline and time.time() > workflow_deadline:
                elapsed = time.time() - (self._workflow_start_time or 0)
                self._handle_workflow_timeout(feature_id, elapsed)

            iteration += 1
            if iteration > max_iterations:
                raise WorkflowBlockedError(
                    f"Feature '{feature_id}' exceeded {max_iterations} iterations. "
                    "Possible infinite loop."
                )

            ready = self._scheduler.get_ready_components()
            if not ready:
                if self._scheduler.is_feature_blocked():
                    blocking = self._scheduler.get_blocking_components()
                    raise WorkflowBlockedError(
                        f"Feature '{feature_id}' is blocked. "
                        f"Components with failed dependencies: {blocking}"
                    )
                time.sleep(0.1)
                continue

            # Execute one component at a time with per-role timeout
            comp = ready[0]
            role_name = comp.assigned_role or "implementer"
            role_timeout = self.role_timeouts.get(role_name, self.component_timeout)

            # Use single-threaded executor for timeout enforcement
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._execute_component, comp, feature_id)
                try:
                    future.result(timeout=role_timeout)
                except TimeoutError:
                    future.cancel()
                    self._scheduler.update_component_status(
                        comp.id,
                        ComponentStatus.FAILED,
                        error=f"Component timed out after {role_timeout}s",
                        workflow_id=feature_id,
                    )
                    logger.error(f"Component '{comp.id}' timed out after {role_timeout}s (sequential mode)")

    def _run_continuous_parallel(
        self,
        feature_id: str,
        num_components: int,
        workflow_deadline: float | None = None,
    ) -> None:
        """Run components with continuous parallel scheduling.

        FIX (Gemini review): Instead of waiting for entire batches to complete,
        this approach continuously checks for newly ready components as each
        completes. This prevents slow components from blocking unrelated work.

        ALGORITHM:
        1. Maintain a pool of active futures (up to max_parallel_workers)
        2. As each future completes, immediately check for newly ready components
        3. Submit new ready components without waiting for all current ones
        4. Track files in use to prevent conflicts
        """
        # FIX (Codex review v2): Wall-clock stall detector for hanging components
        # Tracks last progress time to detect hangs even with active futures
        # FIX (PR review): Use configurable timeout values from __init__
        last_progress_time = time.time()
        last_completed_count = 0
        active_futures: dict[Future, Component] = {}
        future_start_times: dict[Future, float] = {}  # Track when each future started
        future_timeouts: dict[Future, float] = {}  # NEW (v9): Track per-component deadlines
        # FIX (Codex review): Track active component IDs for O(1) lookup instead of O(n) scan
        active_component_ids: set[str] = set()
        # FIX (Codex review v4): Track timed-out futures to release file locks ONLY when thread completes
        # Don't release locks immediately - the thread may still be writing files
        timed_out_futures: dict[Future, Component] = {}
        # Track files currently being modified to prevent conflicts
        files_in_progress: set[str] = set()
        files_lock = threading.Lock()

        # FIX (PR review): Use shared utility for consistent path normalization
        def normalize_path(p: str) -> str:
            """Normalize path using shared utility."""
            return normalize_repo_path(p, self.repo_path, fail_closed=False)

        executor = ThreadPoolExecutor(max_workers=self.max_parallel_workers)
        try:
            while not self._scheduler.is_feature_complete():
                now = time.time()

                # NEW: Check workflow-level timeout
                if workflow_deadline and now > workflow_deadline:
                    elapsed = now - (self._workflow_start_time or 0)
                    self._handle_workflow_timeout(feature_id, elapsed)

                # Check for progress (any new completions)
                # FIX (PR review): Use public method instead of accessing _components directly
                current_completed = self._scheduler.get_completed_count()
                if current_completed > last_completed_count:
                    # Progress made - reset wall-clock timer
                    last_progress_time = time.time()
                    last_completed_count = current_completed

                # FIX (PR review): Check per-component timeouts independently of stall detection
                # This ensures hung components are caught even when other work progresses
                # FIX (Codex review): Only mark as timed out if future is NOT done
                # A future that completed just before timeout should be processed normally
                timed_out = [
                    (f, c) for f, c in active_futures.items()
                    if now > future_timeouts.get(f, now + 9999) # Use stored deadline
                    and not f.done()  # Don't timeout completed futures
                ]
                if timed_out:
                    # FIX (Codex review v4): Mark components failed but DON'T release file locks
                    # The thread may still be writing - keep locks until future.done()
                    for future, comp in timed_out:
                        future.cancel()  # Request cancellation (won't stop running thread)
                        # Move to timed_out_futures - DON'T release file locks yet
                        del active_futures[future]
                        future_start_times.pop(future, None)
                        future_timeouts.pop(future, None)
                        active_component_ids.discard(comp.id)  # FIX (Codex review): O(1) tracking
                        timed_out_futures[future] = comp  # Track for later cleanup
                        self._scheduler.update_component_status(
                            comp.id,
                            ComponentStatus.FAILED,
                            error=f"Component timed out",
                            workflow_id=feature_id,
                        )
                        logger.error(f"Component '{comp.id}' timed out")
                    last_progress_time = time.time()  # Reset after handling timeouts

                # FIX (Codex review): Wall-clock stall detection for deadlocks
                elapsed_since_progress = time.time() - last_progress_time
                if elapsed_since_progress > self.max_stall_seconds and not active_futures:
                    raise WorkflowBlockedError(
                        f"Feature '{feature_id}' stalled for {elapsed_since_progress:.0f}s "
                        "with no progress. Possible deadlock."
                    )

                # FIX (Codex review v4): Check for completed timed-out futures and release their file locks
                completed_timed_out = [f for f in timed_out_futures if f.done()]
                for future in completed_timed_out:
                    comp = timed_out_futures.pop(future)
                    with files_lock:
                        for file in comp.files or []:
                            files_in_progress.discard(normalize_path(file))
                    logger.debug(f"Timed-out component '{comp.id}' thread finished - released file locks")

                # Get ready components that don't conflict with in-progress work
                ready = self._scheduler.get_ready_components()
                submittable = []

                with files_lock:
                    # FIX (PR review): Track files from components being added to submittable
                    # to prevent co-scheduling components that conflict with each other
                    pending_files: set[str] = set()
                    for comp in ready:
                        # Normalize component file paths for comparison
                        comp_files = {normalize_path(f) for f in (comp.files or [])}
                        # FIX (Codex review): O(1) check instead of O(n) scan
                        if comp.id in active_component_ids:
                            continue
                        # Check for file conflicts with in-progress AND pending-to-submit
                        if comp_files & files_in_progress or comp_files & pending_files:
                            continue  # Skip - has file conflict
                        submittable.append(comp)
                        pending_files.update(comp_files)  # Track for subsequent checks

                # Submit new work (respect max_workers limit)
                slots_available = self.max_parallel_workers - len(active_futures)
                for comp in submittable[:slots_available]:
                    with files_lock:
                        # Normalize paths when adding to files_in_progress
                        files_in_progress.update(normalize_path(f) for f in (comp.files or []))
                    
                    # NEW (v9): Look up role-specific timeout
                    role_name = comp.assigned_role or "implementer"
                    timeout = self.role_timeouts.get(role_name, self.component_timeout)
                    
                    future = executor.submit(self._execute_component, comp, feature_id)
                    active_futures[future] = comp
                    active_component_ids.add(comp.id)  # FIX (Codex review): O(1) tracking
                    future_start_times[future] = time.time()  # Track start time for timeout
                    future_timeouts[future] = time.time() + timeout # NEW (v9): Store deadline

                # If no active work and nothing ready, check if blocked
                if not active_futures:
                    if not ready:
                        if self._scheduler.is_feature_blocked():
                            blocking = self._scheduler.get_blocking_components()
                            raise WorkflowBlockedError(
                                f"Feature '{feature_id}' is blocked. "
                                f"Components with failed dependencies: {blocking}"
                            )
                        # All done or waiting for dependencies
                        if self._scheduler.is_feature_complete():
                            break
                    time.sleep(0.1)
                    continue

                # Wait for at least one to complete (short timeout for responsiveness)
                completed = []
                try:
                    for future in as_completed(active_futures, timeout=0.5):
                        completed.append(future)
                        # Process one completion then check for new ready components
                        break
                except TimeoutError:
                    pass  # No completions yet, continue loop

                # Process completed futures
                for future in completed:
                    comp = active_futures.pop(future)
                    active_component_ids.discard(comp.id)  # FIX (Codex review): O(1) tracking
                    future_start_times.pop(future, None)  # Clean up start time tracking
                    future_timeouts.pop(future, None)
                    with files_lock:
                        for f in comp.files or []:
                            files_in_progress.discard(normalize_path(f))
                    try:
                        future.result()  # Raise any exception
                    except Exception as e:
                        logger.error(f"Component {comp.id} failed: {e}")
                        # Status already updated in _execute_component

        finally:
            # Wait for any remaining active futures to complete
            for future in active_futures:
                try:
                    future.result(timeout=60)
                except Exception:
                    pass
            # FIX (Codex review v4): Also wait for timed-out futures before shutdown
            # This ensures their threads finish and file locks can be released
            for future in timed_out_futures:
                try:
                    future.result(timeout=30)  # Shorter timeout for already-timed-out
                except Exception:
                    pass
            executor.shutdown(wait=True)

    # FIX (v27 - Gemini PR review): Helper methods for _execute_component

    def _run_component_role(
        self,
        component: Component,
        feature_id: str,
    ) -> Any:
        """Run the role for a component.

        Returns the role execution result.
        """
        role_name = component.assigned_role or "implementer"
        task_description = (
            f"Implement component: {component.title}\n\n"
            f"Description: {component.description or 'No description provided.'}\n\n"
            f"Files to create/modify: {', '.join(component.files) if component.files else 'As needed'}"
        )

        role_config = self.engine.role_loader.load_role(role_name)
        role_cli = role_config.cli if role_config else None

        estimated_context = len(component.files) * 5000 if component.files else 10000
        selected_model = self._router.select_model(
            role_name=role_name,
            role_cli=role_cli,
            context_size=estimated_context,
        )
        logger.debug(
            f"Component '{component.id}': Router selected '{selected_model}' for role '{role_name}'"
        )

        def is_cancelled() -> bool:
            comp = self._scheduler.get_component(component.id)
            return comp is not None and comp.status == ComponentStatus.FAILED

        return self.engine.run_role(
            role_name=role_name,
            task_description=task_description,
            workflow_id=feature_id,
            target_files=component.files,
            cli_override=selected_model,
            cancellation_check=is_cancelled,
        )

    def _handle_post_execution_approval(
        self,
        component: Component,
        feature_id: str,
        baseline_set: set[str],
        baseline_hashes: dict[str, str],
        baseline_contents: dict[str, bytes] | None = None,
    ) -> tuple[bool, list[str], list[str]]:
        """Handle post-execution approval flow.

        FIX (v27 - Codex PR review): Track changes by content hash, not just filename.
        Files that were already modified but changed again by this component are
        now detected and included in rollback.
        FIX (v27 - Codex PR review): Pass baseline_contents to rollback for restoration.

        Returns:
            Tuple of (approved, component_changed, component_untracked)
        """
        current_changed, current_untracked = self._get_changed_files(None)
        current_hashes = self._get_file_hashes(current_changed)

        # New files (not in baseline)
        new_changed = [f for f in current_changed if f not in baseline_set]
        new_untracked = [f for f in current_untracked if f not in baseline_set]

        # Files that were in baseline but have different hash now (re-modified)
        re_modified = [
            f for f in current_changed
            if f in baseline_hashes and current_hashes.get(f) != baseline_hashes[f]
        ]

        # Combine: files this component touched
        component_changed = list(set(new_changed + re_modified))
        component_untracked = new_untracked
        all_component_changes = component_changed + component_untracked

        # FIX (v27 - Codex PR review P2): Show diff for all detected changes, not just component.files
        # FIX (v27 - Codex PR review): Pass untracked files separately to show their content
        diff_lines = self._get_worktree_diff(
            component_changed if component_changed else None,
            component_untracked if component_untracked else None,
        )

        if self.approval_gate:
            if not self._check_approval_gate(
                feature_id, component, all_component_changes, diff_lines, component_untracked
            ):
                # FIX (v27 - Codex PR review): Surface rollback failures
                rollback_ok = self._rollback_worktree_changes(
                    component_changed, component_untracked, baseline_contents
                )
                if not rollback_ok:
                    logger.warning(
                        f"Rollback for component '{component.id}' had partial failures. "
                        "Some files may not be fully restored."
                    )
                return False, component_changed, component_untracked

        return True, component_changed, component_untracked

    def _get_file_hashes(self, files: list[str]) -> dict[str, str]:
        """Get SHA1 hashes of file contents for change detection.

        FIX (v27 - Codex PR review): Track changes by content hash, not just filename.
        """
        import hashlib
        import os
        hashes = {}
        for filepath in files:
            full_path = os.path.join(self.repo_path or ".", filepath)
            try:
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    with open(full_path, "rb") as f:
                        hashes[filepath] = hashlib.sha1(f.read()).hexdigest()
            except Exception:
                pass
        return hashes

    def _save_file_contents(self, files: list[str]) -> dict[str, bytes]:
        """Save file contents for baseline restoration on rollback.

        FIX (v27 - Codex PR review): Save actual content instead of relying on git checkout.
        This allows restoring to baseline state, not HEAD, preserving prior component changes.
        """
        import os
        contents = {}
        for filepath in files:
            full_path = os.path.join(self.repo_path or ".", filepath)
            try:
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    with open(full_path, "rb") as f:
                        contents[filepath] = f.read()
            except Exception as e:
                logger.warning(f"Failed to save baseline content for {filepath}: {e}")
        return contents

    def _execute_component(self, component: Component, feature_id: str) -> None:
        """Execute a single component.

        FIX (v27 - Gemini PR review): Refactored into smaller helper methods.
        FIX (v27 - Codex PR review): Track changes by content hash, not just filename.
        FIX (v27 - Codex PR review): Save baseline contents to restore on rollback.
        """
        # Capture baseline BEFORE running role - filenames, hashes, and contents
        baseline_changed, baseline_untracked = self._get_changed_files(None)
        baseline_set = set(baseline_changed + baseline_untracked)
        baseline_hashes = self._get_file_hashes(baseline_changed)
        # Save actual content for restoring on rollback (not git checkout to HEAD)
        baseline_contents = self._save_file_contents(baseline_changed)

        try:
            # Update status to implementing
            self._scheduler.update_component_status(
                component.id,
                ComponentStatus.IMPLEMENTING,
                workflow_id=feature_id,
            )

            # Run the role
            result = self._run_component_role(component, feature_id)

            # Check if timed out while running
            comp = self._scheduler.get_component(component.id)
            if comp and comp.status == ComponentStatus.FAILED:
                logger.warning(
                    f"Component '{component.id}' completed after timeout - ignoring late result"
                )
                return

            # Handle approval flow
            approved, component_changed, component_untracked = self._handle_post_execution_approval(
                component, feature_id, baseline_set, baseline_hashes, baseline_contents
            )

            if not approved:
                self._scheduler.update_component_status(
                    component.id,
                    ComponentStatus.FAILED,
                    error="Approval rejected by user",
                    workflow_id=feature_id,
                )
                logger.info(f"Component '{component.id}' rejected by approval gate, changes rolled back")
                return

            # Approval granted - mark as completed
            self._scheduler.update_component_status(
                component.id,
                ComponentStatus.COMPLETED,
                output=str(
                    result.model_dump() if hasattr(result, "model_dump") else result
                ),
                workflow_id=feature_id,
            )

            logger.info(f"Component '{component.id}' completed successfully")

        except Exception as e:
            logger.error(f"Component '{component.id}' failed: {e}")
            # Rollback only THIS component's changes using hash-based detection
            exc_changed, exc_untracked = self._get_changed_files(None)
            exc_hashes = self._get_file_hashes(exc_changed)

            new_changed = [f for f in exc_changed if f not in baseline_set]
            re_modified = [
                f for f in exc_changed
                if f in baseline_hashes and exc_hashes.get(f) != baseline_hashes[f]
            ]
            component_exc_changed = list(set(new_changed + re_modified))
            component_exc_untracked = [f for f in exc_untracked if f not in baseline_set]
            # FIX (v27 - Codex PR review): Surface rollback failures
            rollback_ok = self._rollback_worktree_changes(
                component_exc_changed, component_exc_untracked, baseline_contents
            )
            if not rollback_ok:
                logger.warning(
                    f"Rollback for component '{component.id}' had partial failures. "
                    "Some files may not be fully restored."
                )

            comp = self._scheduler.get_component(component.id)
            if comp and comp.status == ComponentStatus.FAILED:
                logger.warning(
                    f"Component '{component.id}' raised exception after timeout - ignoring"
                )
                return
            self._scheduler.update_component_status(
                component.id,
                ComponentStatus.FAILED,
                error=str(e),
                workflow_id=feature_id,
            )

    def _handle_workflow_timeout(self, feature_id: str, elapsed: float) -> None:
        """Handle workflow-level timeout.

        NEW method - saves checkpoint and raises CancellationError.
        """
        logger.error(f"Workflow timeout: {elapsed:.1f}s > {self.workflow_timeout:.1f}s")

        if self.checkpoint_on_timeout:
            self._save_timeout_checkpoint(feature_id, f"Workflow timeout after {elapsed:.1f}s")

        from supervisor.core.engine import CancellationError
        raise CancellationError(f"Workflow timeout after {elapsed:.1f}s")

    def _save_timeout_checkpoint(self, feature_id: str, reason: str) -> str:
        """Save checkpoint on timeout using existing create_checkpoint method."""
        # Get current state for checkpoint
        feature = self.db.get_feature(feature_id)
        phases = self.db.get_phases(feature_id)
        components = self.db.get_components(feature_id)

        # FIX (Codex v3): context must be dict, not JSON string
        context = {
            "reason": reason,
            "feature_status": feature.status.value if feature else "unknown",
            "phases_completed": sum(1 for p in phases if p.status == PhaseStatus.COMPLETED),
            "components_completed": sum(1 for c in components if c.status == ComponentStatus.COMPLETED),
            "resumable": True,
        }

        # FIX (Codex v3): Use correct create_checkpoint signature
        checkpoint_id = self.db.create_checkpoint(
            workflow_id=feature_id,
            step_id=None,  # FIX: step_id is required param (can be None)
            git_sha=f"timeout-{uuid.uuid4().hex[:8]}",
            context=context,  # FIX: dict not JSON string
            status="timeout",
        )

        logger.info(f"Saved timeout checkpoint: {checkpoint_id}")
        return str(checkpoint_id)

    def reset_workflow_timer(self) -> None:
        """Reset workflow timer for new feature execution."""
        self._workflow_start_time = None

    def _check_approval_gate(
        self,
        feature_id: str,
        component: Component,
        changed_files: list[str],
        diff_lines: list[str] | None = None,
        untracked_files: list[str] | None = None,
    ) -> bool:
        """Check if approval gate should be invoked and get decision.

        FIX (v25 - Codex): Updated signature and docstring to reflect new flow.
        CALL SITE: This method is called from _execute_component() AFTER
        running the component's role, BEFORE committing. This allows showing
        the actual diff to the user for review.

        Args:
            feature_id: Workflow/feature ID
            component: Component being executed
            changed_files: List of changed file paths (for risk assessment and counts)
            diff_lines: Full git diff output (for display in TUI/CLI)
            untracked_files: Newly created files (shown separately in approval UI)

        Returns True if workflow should proceed (APPROVE or SKIP).
        Returns False if workflow should halt (REJECT or EDIT).
        """
        from datetime import datetime
        from supervisor.core.interaction import ApprovalDecision

        if not self.approval_gate:
            return True

        # FIX (v23): Add operation to context for require_approval_for policy
        operation = component.assigned_role or "implement"
        if "commit" in operation.lower():
            operation = "commit"
        elif "deploy" in operation.lower():
            operation = "deploy"

        # FIX (v25): Pass file list for risk assessment, not diff lines
        context = {
            "changes": changed_files,
            "component": component.id,
            "operation": operation,
        }
        if not self.approval_gate.requires_approval(context):
            return True  # Low risk / excluded operation, auto-approve

        # FIX (v25): Build review summary with untracked file warning
        review_summary = f"Review changes for component {component.id} ({component.title})"
        if untracked_files:
            review_summary += f"\n\nNote: {len(untracked_files)} new file(s) will be created: {', '.join(untracked_files)}"

        # FIX (v25): Request approval with both file list (for risk) and diff (for display)
        decision = self.approval_gate.request_approval(
            feature_id=feature_id,
            title=f"Approve {component.title}",
            changes=changed_files,
            diff_lines=diff_lines,
            review_summary=review_summary,
            component_id=component.id,
            bridge=self.interaction_bridge,
        )

        # FIX (v22): Use is_proceed() semantic helper for clear decision handling
        if decision.is_proceed():
            if decision == ApprovalDecision.SKIP:
                logger.warning(f"Component '{component.id}' skipped approval - flagged for review")
                # FIX (v24): Persist SKIP flag in component metadata for later review
                self._scheduler.set_component_metadata(
                    component.id,
                    key="approval_skipped",
                    value=True,
                    workflow_id=feature_id,
                )
                self._scheduler.set_component_metadata(
                    component.id,
                    key="approval_skipped_at",
                    value=datetime.now().isoformat(),
                    workflow_id=feature_id,
                )
            return True  # Proceed with APPROVE or SKIP

        # REJECT or EDIT - block execution
        if decision == ApprovalDecision.EDIT:
            logger.info(f"Component '{component.id}' awaiting user edits (EDIT not implemented)")
        else:  # REJECT
            logger.info(f"Component '{component.id}' rejected by user")
        return False

    def _get_changed_files(
        self, target_files: list[str] | None
    ) -> tuple[list[str], list[str]]:
        """Get lists of changed tracked files and new untracked files.

        FIX (v26 - Codex): Include both staged (--cached) and unstaged changes.

        Returns:
            (changed_files, untracked_files) tuple where:
            - changed_files: Modified/deleted tracked files (staged + unstaged)
            - untracked_files: Newly created files from git status --porcelain
        """
        import subprocess
        changed: set[str] = set()
        untracked = []
        try:
            # Get unstaged modified tracked files
            cmd = ["git", "diff", "--name-only"]
            if target_files:
                cmd.extend(["--", *target_files])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.git_timeout, cwd=self.repo_path)
            if result.returncode == 0 and result.stdout.strip():
                changed.update(result.stdout.strip().split("\n"))

            # FIX (v26 - Codex): Also get staged changes
            cmd_cached = ["git", "diff", "--name-only", "--cached"]
            if target_files:
                cmd_cached.extend(["--", *target_files])
            result = subprocess.run(cmd_cached, capture_output=True, text=True, timeout=self.git_timeout, cwd=self.repo_path)
            if result.returncode == 0 and result.stdout.strip():
                changed.update(result.stdout.strip().split("\n"))

            # Get untracked files (newly created)
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=self.git_timeout, cwd=self.repo_path,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("??"):
                        filepath = line[3:].strip()
                        # If target_files specified, filter to those paths
                        if target_files:
                            if any(filepath.startswith(tf) or tf.startswith(filepath)
                                   for tf in target_files):
                                untracked.append(filepath)
                        else:
                            untracked.append(filepath)
        except Exception as e:
            logger.warning(f"Failed to get changed files: {e}")
        return list(changed), untracked

    def _get_worktree_diff(
        self,
        target_files: list[str] | None,
        untracked_files: list[str] | None = None,
    ) -> list[str]:
        """Capture actual git diff from worktree after role execution.

        FIX (v26 - Codex): Include both staged and unstaged changes in diff.
        FIX (v27 - Codex PR review): Include content for untracked (new) files.

        Args:
            target_files: Tracked files to show diff for
            untracked_files: Untracked (new) files to show content for

        Returns a list of diff lines that can be shown to the user in the
        approval gate UI. This enables reviewing REAL changes, not just
        file names from the component spec.
        """
        import os
        import subprocess
        all_diff_lines: list[str] = []
        try:
            # Get unstaged changes for tracked files
            cmd = ["git", "diff", "--no-color"]
            if target_files:
                cmd.extend(["--", *target_files])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.git_timeout,
                cwd=self.repo_path,
            )
            if result.returncode == 0 and result.stdout.strip():
                all_diff_lines.extend(result.stdout.strip().split("\n"))

            # FIX (v26 - Codex): Also get staged changes
            cmd_cached = ["git", "diff", "--no-color", "--cached"]
            if target_files:
                cmd_cached.extend(["--", *target_files])
            result = subprocess.run(
                cmd_cached,
                capture_output=True,
                text=True,
                timeout=self.git_timeout,
                cwd=self.repo_path,
            )
            if result.returncode == 0 and result.stdout.strip():
                if all_diff_lines:
                    all_diff_lines.append("")  # Separator
                    all_diff_lines.append("# Staged changes:")
                all_diff_lines.extend(result.stdout.strip().split("\n"))

            # FIX (v27 - Codex PR review): Show content for untracked (new) files
            # FIX (v27 - Codex PR review): Use repo-relative paths to avoid leaking absolute paths
            if untracked_files:
                for filepath in untracked_files:
                    full_path = os.path.join(self.repo_path or ".", filepath)
                    if os.path.isfile(full_path):
                        # Use git diff --no-index with repo-relative path
                        # The -- separator ensures paths are treated as paths, not options
                        cmd_untracked = [
                            "git", "diff", "--no-index", "--no-color",
                            "--", "/dev/null", filepath
                        ]
                        result = subprocess.run(
                            cmd_untracked,
                            capture_output=True,
                            text=True,
                            timeout=self.git_timeout,
                            cwd=self.repo_path,
                        )
                        # git diff --no-index returns 1 when files differ, which is expected
                        if result.stdout.strip():
                            if all_diff_lines:
                                all_diff_lines.append("")  # Separator
                            all_diff_lines.append(f"# New file: {filepath}")
                            all_diff_lines.extend(result.stdout.strip().split("\n"))

            return all_diff_lines
        except Exception as e:
            logger.warning(f"Failed to capture diff: {e}")
            return []

    def _rollback_worktree_changes(
        self,
        target_files: list[str] | None,
        untracked_files: list[str] | None = None,
        baseline_contents: dict[str, bytes] | None = None,
    ) -> bool:
        """Rollback uncommitted worktree changes after rejection.

        FIX (v25 - Codex): Now handles both tracked and untracked files.
        FIX (v26 - Codex): Handle directories with shutil.rmtree.
        FIX (v27 - Codex PR review P1): Restore from baseline contents, not git checkout.
        FIX (v27 - Codex PR review): Return bool to surface rollback failures.
        - Tracked files with baseline: Restore from saved baseline contents
        - Tracked files without baseline (new to tracking): git checkout to discard
        - Untracked files/directories: Removes newly created files and directories

        Returns:
            True if rollback succeeded, False if any files failed to restore.
        """
        import os
        import shutil
        import subprocess
        restore_failed = False
        try:
            # FIX (v27 - Codex PR review P1): Restore files from baseline contents
            # This preserves prior component changes instead of resetting to HEAD
            if target_files:
                files_to_checkout: list[str] = []
                for filepath in target_files:
                    if baseline_contents and filepath in baseline_contents:
                        # Restore from saved baseline content
                        full_path = os.path.join(self.repo_path or ".", filepath)
                        try:
                            # FIX (v27 - Codex PR review): Ensure parent dirs exist
                            # Component may have deleted directories during execution
                            parent_dir = os.path.dirname(full_path)
                            if parent_dir:
                                os.makedirs(parent_dir, exist_ok=True)
                            with open(full_path, "wb") as f:
                                f.write(baseline_contents[filepath])
                            logger.debug(f"Restored {filepath} from baseline content")
                        except Exception as restore_err:
                            # FIX (v27 - Codex PR review): Don't fall back to git checkout
                            # for files we have baseline content for - that defeats P1 fix
                            # FIX (v27 - Codex PR review): Track failures to surface to caller
                            logger.error(f"Failed to restore {filepath} from baseline: {restore_err}")
                            restore_failed = True
                    else:
                        # File wasn't in baseline - use git checkout to discard
                        files_to_checkout.append(filepath)

                # Git checkout for files not in baseline (new modifications)
                if files_to_checkout:
                    cmd = ["git", "checkout", "--"]
                    cmd.extend(files_to_checkout)
                    subprocess.run(cmd, capture_output=True, timeout=self.git_timeout, cwd=self.repo_path)
                    logger.debug(f"Rolled back tracked changes via git checkout for: {files_to_checkout}")
            else:
                # No tracked files to rollback - this is expected when component
                # only created new files or edited files already in baseline
                logger.debug("No tracked files to rollback")

            # FIX (v25): Remove untracked files created by role
            # FIX (v26): Handle directories with shutil.rmtree
            if untracked_files:
                for filepath in untracked_files:
                    full_path = os.path.join(self.repo_path or ".", filepath)
                    try:
                        if os.path.isdir(full_path):
                            shutil.rmtree(full_path)
                            logger.debug(f"Removed untracked directory: {filepath}")
                        elif os.path.exists(full_path):
                            os.remove(full_path)
                            logger.debug(f"Removed untracked file: {filepath}")
                    except Exception as file_err:
                        logger.warning(f"Failed to remove untracked path {filepath}: {file_err}")
                        restore_failed = True

            return not restore_failed
        except Exception as e:
            logger.error(f"Failed to rollback worktree changes: {e}")
            return False

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
        if not feature:
            raise WorkflowError(f"Feature '{feature_id}' not found")

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
            self.db.update_feature_status(feature_id, FeatureStatus.COMPLETED)
        else:
            self.db.update_feature_status(feature_id, FeatureStatus.REVIEW)

        return {
            "approved": result.approved,
            "summary": result.summary,
            "issues": result.get_issues(),
        }

    def get_feature_status(self, feature_id: str) -> dict[str, Any]:
        """Get detailed status of a feature.

        Returns:
            Dict with feature status, phase statuses, component statuses
        """
        feature = self.db.get_feature(feature_id)
        if not feature:
            raise WorkflowError(f"Feature '{feature_id}' not found")

        phases = self.db.get_phases(feature_id)
        components = self.db.get_components(feature_id)

        # Group components by phase
        phase_components: dict[str, list[dict]] = {}
        for comp in components:
            if comp.phase_id not in phase_components:
                phase_components[comp.phase_id] = []
            phase_components[comp.phase_id].append(
                {
                    "id": comp.id,
                    "title": comp.title,
                    "status": comp.status.value,
                    "assigned_role": comp.assigned_role,
                    "error": comp.error,
                }
            )

        return {
            "feature": {
                "id": feature.id,
                "title": feature.title,
                "status": feature.status.value,
                "created_at": feature.created_at.isoformat(),
                "completed_at": (
                    feature.completed_at.isoformat() if feature.completed_at else None
                ),
            },
            "phases": [
                {
                    "id": p.id,
                    "title": p.title,
                    "sequence": p.sequence,
                    "status": p.status.value,
                    "components": phase_components.get(p.id, []),
                }
                for p in phases
            ],
            "summary": {
                "total_phases": len(phases),
                "total_components": len(components),
                "completed_components": sum(
                    1 for c in components if c.status == ComponentStatus.COMPLETED
                ),
                "failed_components": sum(
                    1 for c in components if c.status == ComponentStatus.FAILED
                ),
                "pending_components": sum(
                    1 for c in components if c.status == ComponentStatus.PENDING
                ),
            },
        }
