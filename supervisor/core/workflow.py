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
from supervisor.core.routing import ModelRouter, create_router
from supervisor.core.scheduler import (
    DAGScheduler,
    WorkflowBlockedError,
)
from supervisor.core.state import Database, Event, EventType
from supervisor.core.utils import normalize_repo_path

if TYPE_CHECKING:
    from supervisor.core.engine import ExecutionEngine

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
    ):
        self.engine = engine
        self.db = db
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.max_parallel_workers = max_parallel_workers
        # FIX (PR review): Make timeout values configurable
        self.max_stall_seconds = max_stall_seconds  # Max time without progress (default 10 min)
        self.component_timeout = component_timeout  # Per-component timeout (default 5 min)
        self._scheduler: DAGScheduler | None = None
        # FIX (Gemini review): Integrate ModelRouter for intelligent model selection
        self._router = create_router(prefer_speed=prefer_speed, prefer_cost=prefer_cost)

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
        raw_interfaces = phase_data.get("interfaces", {})
        if isinstance(raw_interfaces, dict):
            interfaces_list = [
                {"name": name, **props} if isinstance(props, dict) else {"name": name, "definition": str(props)}
                for name, props in raw_interfaces.items()
            ]
        else:
            interfaces_list = raw_interfaces if isinstance(raw_interfaces, list) else []

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
        # Build DAG - FIX (Codex v5): Pass repo_path for consistent path normalization
        self._scheduler = DAGScheduler(self.db, repo_path=self.repo_path)
        self._scheduler.build_graph(feature_id)

        # Update feature status
        self.db.update_feature_status(feature_id, FeatureStatus.IN_PROGRESS)

        # FIX (PR review): Use public method instead of accessing _components directly
        num_components = self._scheduler.get_component_count()

        if not parallel:
            # Sequential execution - simple loop
            self._run_sequential(feature_id, num_components)
        else:
            # FIX (Gemini review): Continuous parallel scheduling
            # Instead of batch-wait, continuously submit newly ready components
            self._run_continuous_parallel(feature_id, num_components)

        # Feature complete
        self.db.update_feature_status(feature_id, FeatureStatus.REVIEW)
        logger.info(f"Implementation complete for '{feature_id}'")
        return self.db.get_feature(feature_id)  # type: ignore

    def _run_sequential(self, feature_id: str, num_components: int) -> None:
        """Run components sequentially (no parallelism)."""
        iteration = 0
        max_iterations = max(num_components * 10, 100)

        while not self._scheduler.is_feature_complete():
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

            # Execute one component at a time
            self._execute_component(ready[0], feature_id)

    def _run_continuous_parallel(self, feature_id: str, num_components: int) -> None:
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
                # Check for progress (any new completions)
                # FIX (PR review): Use public method instead of accessing _components directly
                current_completed = self._scheduler.get_completed_count()
                if current_completed > last_completed_count:
                    # Progress made - reset wall-clock timer
                    last_progress_time = time.time()
                    last_completed_count = current_completed

                # FIX (PR review): Check per-component timeouts independently of stall detection
                # This ensures hung components are caught even when other work progresses
                now = time.time()
                timed_out = [
                    (f, c) for f, c in active_futures.items()
                    if now - future_start_times.get(f, now) > self.component_timeout
                ]
                if timed_out:
                    # FIX (Codex review v4): Mark components failed but DON'T release file locks
                    # The thread may still be writing - keep locks until future.done()
                    for future, comp in timed_out:
                        future.cancel()  # Request cancellation (won't stop running thread)
                        # Move to timed_out_futures - DON'T release file locks yet
                        del active_futures[future]
                        future_start_times.pop(future, None)
                        timed_out_futures[future] = comp  # Track for later cleanup
                        self._scheduler.update_component_status(
                            comp.id,
                            ComponentStatus.FAILED,
                            error=f"Component timed out after {self.component_timeout}s",
                            workflow_id=feature_id,
                        )
                        logger.error(f"Component '{comp.id}' timed out after {self.component_timeout}s")
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
                        # Check if already submitted (in active_futures)
                        if any(c.id == comp.id for c in active_futures.values()):
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
                    future = executor.submit(self._execute_component, comp, feature_id)
                    active_futures[future] = comp
                    future_start_times[future] = time.time()  # Track start time for timeout

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
                    future_start_times.pop(future, None)  # Clean up start time tracking
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

    def _execute_component(self, component: Component, feature_id: str) -> None:
        """Execute a single component.

        Maps component to role execution:
        1. Determine role from component.assigned_role
        2. Use ModelRouter for intelligent model selection (FIX: Gemini review)
        3. Build context with component's target files
        4. Run role with component-specific task
        5. Update component status based on result
        """
        try:
            # Update status to implementing
            self._scheduler.update_component_status(
                component.id,
                ComponentStatus.IMPLEMENTING,
                workflow_id=feature_id,
            )

            # Determine role and task
            role_name = component.assigned_role or "implementer"
            task_description = (
                f"Implement component: {component.title}\n\n"
                f"Description: {component.description or 'No description provided.'}\n\n"
                f"Files to create/modify: {', '.join(component.files) if component.files else 'As needed'}"
            )

            # FIX (PR review): Load role config to get its configured CLI
            # This ensures custom/overlay roles with explicit CLI settings are respected
            role_config = self.engine.role_loader.load_role(role_name)
            role_cli = role_config.cli if role_config else None

            # FIX (Gemini review): Use ModelRouter for intelligent model selection
            # Estimate context size based on number of files
            estimated_context = len(component.files) * 5000 if component.files else 10000
            selected_model = self._router.select_model(
                role_name=role_name,
                role_cli=role_cli,  # Pass role's configured CLI to respect explicit config
                context_size=estimated_context,
            )
            logger.debug(
                f"Component '{component.id}': Router selected '{selected_model}' for role '{role_name}'"
            )

            # Run the role with ModelRouter-selected CLI
            result = self.engine.run_role(
                role_name=role_name,
                task_description=task_description,
                workflow_id=feature_id,
                target_files=component.files,
                cli_override=selected_model,
            )

            # FIX (Codex review v3): Check current status before updating to COMPLETED
            # A timed-out component may still complete; ignore late results
            # FIX (PR review): Use public method instead of accessing _components directly
            comp = self._scheduler.get_component(component.id)
            if comp and comp.status == ComponentStatus.FAILED:
                logger.warning(
                    f"Component '{component.id}' completed after timeout - ignoring late result"
                )
                return  # Don't overwrite FAILED status

            # Success - update status
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
            # FIX (Codex review v4): Don't overwrite timeout error with exception error
            # FIX (PR review): Use public method instead of accessing _components directly
            comp = self._scheduler.get_component(component.id)
            if comp and comp.status == ComponentStatus.FAILED:
                logger.warning(
                    f"Component '{component.id}' raised exception after timeout - ignoring"
                )
                return  # Don't overwrite timeout error
            self._scheduler.update_component_status(
                component.id,
                ComponentStatus.FAILED,
                error=str(e),
                workflow_id=feature_id,
            )

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
