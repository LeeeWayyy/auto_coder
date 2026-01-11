# Phase 5: Polish & Advanced - Implementation Plan

**Status:** Draft v25 (approval after execution with real diff, untracked file handling, SKIP persistence)
**Created:** 2026-01-10
**Updated:** 2026-01-11
**Reviewed By:** Codex (REJECT v22)
**Phase:** 5 of 5
**Dependencies:** Phase 1 (Foundation), Phase 2 (Core Workflow), Phase 3 (Gates & Verification), Phase 4 (Multi-Model & Hierarchy)

## Overview

Phase 5 transforms the supervisor from a functional orchestration system into a production-ready tool with observability, human interaction capabilities, and adaptive intelligence. This phase focuses on operational excellence, user experience, and intelligent automation.

**Key Principle:** Phase 5 **extends existing implementations** rather than creating parallel systems. The codebase already has:
- Timeout handling (`workflow.py:component_timeout`, `engine.py:CancellationError`)
- Approval events (`state.py:APPROVAL_REQUESTED/GRANTED/DENIED`)
- Model routing (`routing.py:ModelRouter`)
- Rich CLI patterns (`cli.py`)

We build on these foundations rather than replacing them.

### Core Deliverables

From SUPERVISOR_ORCHESTRATOR.md Phase 5:
- [ ] Timeout handling
- [ ] Human interrupt interface (TUI)
- [ ] Metrics dashboard
- [ ] Adaptive role assignment

### Extended Scope (From Advanced Features Section)

Based on the architecture document's "Advanced Features" section, Phase 5 also includes:
- [ ] Workflow timeout management with graceful degradation
- [ ] Interactive TUI for approval gates and workflow control
- [ ] Comprehensive metrics collection and visualization
- [ ] ML-assisted model selection based on historical performance
- [ ] Human approval policy configuration

---

## Current Implementation Status

### Already Implemented (Phases 1-4)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| CircuitBreaker | `engine.py:100-170` | Complete | Thread-safe, bounded memory |
| RetryPolicy | `engine.py:78-96` | Complete | Exponential backoff with jitter |
| CancellationError | `engine.py:37-42` | Complete | Used by `cancellation_check` parameter |
| Event Sourcing | `state.py` | Complete | Full audit trail via SQLite |
| Approval Events | `state.py:EventType` | Complete | `APPROVAL_REQUESTED/GRANTED/DENIED` defined |
| DAG Scheduler | `scheduler.py` | Complete | Dependency-aware execution |
| Parallel Reviewer | `parallel.py` | Complete | Multi-model concurrent reviews with timeout |
| Workflow Coordinator | `workflow.py` | Complete | Feature→Phase→Component with `component_timeout` |
| Component Timeout | `workflow.py:__init__` | Complete | `component_timeout=300.0`, `max_stall_seconds=600.0` |
| Cancellation Event | `workflow.py/parallel.py` | Complete | `threading.Event()` for timeout signaling |
| Gate System | `gates.py` | Complete | Orchestrator-enforced verification |
| Context Strategies | `strategies.py` | Complete | Role-specific context packing |
| Model Routing | `routing.py` | Complete | Static capability-based model selection |
| CLI Foundation | `cli.py` | Complete | Rich-based CLI with init/plan/run/roles/status |

### Gaps to Address in Phase 5

1. **Timeout Handling** - Component-level exists in `workflow.py`, but need:
   - Workflow-level (total feature) timeout
   - Checkpoint save on timeout for resume
   - Per-role timeout overrides
   - **EXTEND existing `component_timeout` and `CancellationError`**
   - **NOTE (v6):** Phase-level timeout is OUT OF SCOPE - would require refactoring DAGScheduler to support `is_phase_complete()`. The existing scheduler executes feature-wide, not phase-by-phase.

2. **Human Interrupt TUI** - Existing `APPROVAL_*` events need:
   - Interactive TUI interface for approval/rejection
   - Integration with existing EventType enum
   - Diff viewer for code changes
   - **BUILD ON existing approval event types in `state.py`**

3. **Metrics Dashboard** - No aggregated metrics, need:
   - **FIX (v20): Metrics table and record_metric() ALREADY EXIST in state.py:217-291**
   - Collection integrated with `ExecutionEngine.run_role()` (instrumentation)
   - Dashboard CLI command (visualization)
   - **USE existing `state.py` schema - no new tables needed**

4. **Adaptive Assignment** - Static routing in `routing.py`, need:
   - Historical performance tracking
   - Weighted scoring algorithm
   - **EXTEND existing `ModelRouter` class, not replace**

5. **Approval Policies** - Gate system exists, need:
   - Risk assessment logic
   - Configurable policy rules

---

## Architecture Overview

Phase 5 **extends existing components** (shown with `[EXTEND]`) rather than creating new systems.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUPERVISOR CLI (cli.py)                       │
│  supervisor run | supervisor workflow --tui <feature-id> [NEW] | supervisor metrics [NEW]│
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────┐      │
│   │  [EXTEND] WorkflowCoordinator (workflow.py)           │      │
│   │                                                       │      │
│   │   EXISTING              │  NEW Phase 5                │      │
│   │   ─────────────────────┼────────────────────────     │      │
│   │   component_timeout    │  workflow_timeout           │      │
│   │   max_stall_seconds    │  checkpoint_on_timeout      │      │
│   │   threading.Event      │  _save_timeout_checkpoint   │      │
│   │   CancellationError    │  role_timeouts (per-role)   │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                  │
│   ┌──────────────────────────────────────────────────────┐      │
│   │  [NEW] Human Interrupt TUI (tui/app.py)               │      │
│   │                                                       │      │
│   │  Uses EXISTING:          │  NEW:                      │      │
│   │   EventType.APPROVAL_*   │  SupervisorTUI class       │      │
│   │   Rich console patterns  │  ApprovalGate integration  │      │
│   │   gates.py Gate system   │  Diff viewer               │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                  │
│   ┌──────────────────────────────────────────────────────┐      │
│   │  [EXTEND] State (state.py) + [NEW] Metrics            │      │
│   │                                                       │      │
│   │  EXISTING SQLite:        │  NEW:                      │      │
│   │   events table           │  metrics table             │      │
│   │   checkpoints table      │  MetricsCollector          │      │
│   │   append_event()         │  MetricsAggregator         │      │
│   │                          │  MetricsDashboard          │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                  │
│   ┌──────────────────────────────────────────────────────┐      │
│   │  [EXTEND] ModelRouter (routing.py)                    │      │
│   │                                                       │      │
│   │  EXISTING:               │  NEW Phase 5:              │      │
│   │   select_model()         │  _select_adaptive()        │      │
│   │   MODEL_PROFILES         │  _calculate_scores()       │      │
│   │   ROLE_MODEL_MAP         │  AdaptiveConfig            │      │
│   │   prefer_speed/cost      │  exploration_rate          │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 5 Deliverables

### 5.1 Timeout Handling

**Goal:** Extend existing timeout handling to workflow level with checkpoint save on timeout, plus per-role timeout overrides.

**EXISTING IMPLEMENTATION:**
- `workflow.py:WorkflowCoordinator.__init__` - Has `component_timeout=300.0` and `max_stall_seconds=600.0`
- `workflow.py:_run_continuous_parallel` - Uses `threading.Event()` for cancellation, tracks `future_start_times`
- `parallel.py:ParallelReviewer` - Uses `cancelled = threading.Event()` for timeout signaling
- `engine.py:CancellationError` - Exception raised when `cancellation_check()` returns True
- `engine.py:run_role` - Accepts `cancellation_check: Callable[[], bool]` parameter

**Files to Create/Modify:**
- `supervisor/core/workflow.py` (MODIFY) - Add workflow-level timeout + role_timeouts wiring
- `supervisor/core/state.py` (MODIFY) - Add checkpoint save helper for timeout recovery
- `supervisor/config/limits.yaml` (MODIFY) - Add timeout configuration

#### Design: Extend WorkflowCoordinator

The existing `WorkflowCoordinator` already has component-level timeout handling via `_run_continuous_parallel`. We extend it with workflow-level timeout and per-role timeout overrides (phase-level timeout is OUT OF SCOPE as DAGScheduler executes feature-wide).

**IMPORTANT API NOTE:** The actual coordinator uses `run_planning()`, `run_implementation()`, `run_review()` - NOT `run_feature()`. The plan must work with these existing methods.

```python
# supervisor/core/workflow.py - MODIFICATIONS to existing class

class WorkflowCoordinator:
    """Coordinates feature execution across phases.

    EXISTING API (Phase 4):
    - run_planning(feature_id) -> list[Phase]
    - run_implementation(feature_id, parallel=True) -> Feature
    - run_review(feature_id) -> Feature
    - _run_continuous_parallel(feature_id, num_components) - internal parallel execution
    - component_timeout: Per-component execution limit (default 300s)
    - max_stall_seconds: Total stall detection for parallel execution (default 600s)

    NEW Phase 5 params:
    - workflow_timeout: Total execution limit for run_implementation (implementation stage)
      NOTE (v10): Scope is implementation only; run_planning/run_review are typically fast
    - checkpoint_on_timeout: Save state for resume on timeout
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
        # NEW Phase 5 params (5.1 Timeout)
        workflow_timeout: float = 3600.0,     # 1 hour total for entire feature
        checkpoint_on_timeout: bool = True,   # Save state on timeout
        role_timeouts: dict[str, float] | None = None,  # Per-role timeout overrides
        # NEW Phase 5 params (5.2 TUI/Approval) - FIX (v13 - Gemini: consolidate)
        approval_gate: "ApprovalGate | None" = None,
        interaction_bridge: "InteractionBridge | None" = None,
        # NEW Phase 5 params (5.4 Adaptive)
        adaptive_config: "AdaptiveConfig | None" = None,
    ):
        # ... existing init code ...
        self.max_stall_seconds = max_stall_seconds
        self.component_timeout = component_timeout

        # NEW: Workflow-level timeout tracking
        # NOTE (v6): Phase-level timeout removed - DAGScheduler executes feature-wide
        self.workflow_timeout = workflow_timeout
        self.checkpoint_on_timeout = checkpoint_on_timeout
        self.role_timeouts = role_timeouts or {}  # Per-role overrides
        self._workflow_start_time: float | None = None

        # NEW (v13 - Gemini): Approval gate and TUI integration
        self.approval_gate = approval_gate
        self.interaction_bridge = interaction_bridge
        self.adaptive_config = adaptive_config

    def run_implementation(
        self,
        feature_id: str,
        parallel: bool = True,
    ) -> Feature:
        """Run implementation of all components.

        MODIFICATION (v7): Add workflow-level timeout only.
        NOTE: DAGScheduler executes feature-wide via is_feature_complete().
        Phase-by-phase iteration is NOT supported by the existing scheduler.
        """
        # FIX (v15 - Codex): ALWAYS reset timer per run_implementation call
        # Previously only set if None, causing premature timeouts when reusing coordinator
        self._workflow_start_time = time.time()

        # Check workflow timeout (elapsed will be ~0 at start, this is for resume)
        elapsed = time.time() - self._workflow_start_time
        if elapsed > self.workflow_timeout:
            self._handle_workflow_timeout(feature_id, elapsed)

        # EXISTING: Build DAG, run parallel/sequential (UNCHANGED)
        self._scheduler = DAGScheduler(self.db, repo_path=self.repo_path)
        self._scheduler.build_graph(feature_id)
        self.db.update_feature_status(feature_id, FeatureStatus.IN_PROGRESS)
        num_components = self._scheduler.get_component_count()

        try:
            # EXISTING: Run all components feature-wide (not phase-by-phase)
            # NEW: Pass workflow_deadline to enable timeout checks in loop
            workflow_deadline = self._workflow_start_time + self.workflow_timeout
            if parallel:
                self._run_continuous_parallel(
                    feature_id,
                    num_components,
                    workflow_deadline=workflow_deadline,
                )
            else:
                # Sequential mode - FIX (v9): Add workflow timeout check
                self._run_sequential_with_timeout(
                    feature_id,
                    workflow_deadline=workflow_deadline,
                )

        except CancellationError:
            # NOTE (v8): Checkpoint already saved in _handle_workflow_timeout()
            # Do not save again here to avoid duplicate checkpoints
            raise

        self.db.update_feature_status(feature_id, FeatureStatus.REVIEW)
        logger.info(f"Implementation complete for '{feature_id}'")
        return self.db.get_feature(feature_id)

    def _run_continuous_parallel(
        self,
        feature_id: str,
        num_components: int,
        workflow_deadline: float | None = None,  # NEW: Optional workflow deadline
    ) -> None:
        """Run components with continuous parallel scheduling.

        MODIFICATION (v7): Accept workflow_deadline for timeout checks.
        Existing signature: (self, feature_id, num_components)
        New signature: (self, feature_id, num_components, workflow_deadline=None)
        """
        # EXISTING: last_progress_time, active_futures, etc.
        last_progress_time = time.time()
        # ... existing setup code ...

        # FIX (v23 - Codex): Track error/cancel state for conditional cleanup
        workflow_failed = False

        executor = ThreadPoolExecutor(max_workers=self.max_parallel_workers)
        try:
            while not self._scheduler.is_feature_complete():
                now = time.time()

                # FIX (v17 - Codex): Check for human-initiated interrupts via bridge
                if self.interaction_bridge:
                    # Check for cancellation signal
                    if self.interaction_bridge.is_cancelled():
                        workflow_failed = True  # FIX (v23): Set flag
                        logger.info("Workflow cancelled by user via TUI")
                        # FIX (v20 - Codex): Mark all active components as FAILED
                        # This prevents in-flight operations from applying changes
                        self._cancel_active_components(feature_id, "Cancelled by user")

                        # FIX (v22 - Codex): Update feature status and emit terminal event
                        self.db.update_feature_status(feature_id, FeatureStatus.FAILED)
                        from supervisor.core.state import Event, EventType
                        self.db.append_event(
                            Event(
                                workflow_id=feature_id,
                                event_type=EventType.WORKFLOW_FAILED,
                                payload={"reason": "cancelled_by_user"},
                            )
                        )

                        if self.checkpoint_on_timeout:
                            self._save_timeout_checkpoint(feature_id, "Cancelled by user")
                        from supervisor.core.engine import CancellationError
                        raise CancellationError("Cancelled by user via TUI")

                    # Check for pause signal - block until resumed or cancelled
                    self.interaction_bridge.wait_if_paused()

                # NEW: Check workflow-level timeout
                if workflow_deadline and now > workflow_deadline:
                    workflow_failed = True  # FIX (v23): Set flag
                    self._handle_workflow_timeout(feature_id, now - self._workflow_start_time)

                # EXISTING: component timeout checks, stall detection, etc.
                # ... rest of existing implementation unchanged ...

        except Exception:
            workflow_failed = True  # FIX (v23): Any exception means failure
            raise

        finally:
            # FIX (v23 - Codex): Only cancel active components on error/cancel, not success
            if workflow_failed:
                self._cancel_active_components(feature_id, "Workflow terminated due to error")

            # FIX (v21 - Codex): Properly wait for in-flight futures
            # Collect all active futures and wait with bounded timeout
            CLEANUP_TIMEOUT = 5.0  # Max wait for in-flight work
            for future in list(active_futures.keys()):
                if not future.done():
                    try:
                        future.result(timeout=CLEANUP_TIMEOUT / max(len(active_futures), 1))
                    except Exception:
                        pass  # Ignore - component already marked FAILED if workflow_failed

            # Shutdown executor (wait=True is safe now since we've awaited futures)
            executor.shutdown(wait=True)

    def _cancel_active_components(self, feature_id: str, reason: str) -> None:
        """Mark all IMPLEMENTING components as FAILED.

        FIX (v20 - Codex): Ensures cancellation propagates to in-flight components
        so that cancellation_check() in run_role prevents further changes.
        """
        components = self.db.get_components(feature_id)
        for comp in components:
            if comp.status == ComponentStatus.IMPLEMENTING:
                self._scheduler.update_component_status(
                    comp.id,
                    ComponentStatus.FAILED,
                    error=reason,
                    workflow_id=feature_id,
                )
                logger.info(f"Component '{comp.id}' cancelled: {reason}")

    def _handle_workflow_timeout(self, feature_id: str, elapsed: float) -> None:
        """Handle workflow-level timeout.

        NEW method - saves checkpoint and raises CancellationError.
        FIX (v22 - Codex): Updates feature status and emits event before raising.
        """
        logger.error(f"Workflow timeout: {elapsed:.1f}s > {self.workflow_timeout:.1f}s")

        # FIX (v22): Update feature status to FAILED
        self.db.update_feature_status(feature_id, FeatureStatus.FAILED)

        # FIX (v22): Emit terminal event for audit/metrics
        from supervisor.core.state import Event, EventType
        self.db.append_event(
            Event(
                workflow_id=feature_id,
                event_type=EventType.WORKFLOW_FAILED,  # or use existing event type
                payload={"reason": "timeout", "elapsed_seconds": elapsed},
            )
        )

        if self.checkpoint_on_timeout:
            self._save_timeout_checkpoint(feature_id, f"Workflow timeout after {elapsed:.1f}s")

        from supervisor.core.engine import CancellationError
        raise CancellationError(f"Workflow timeout after {elapsed:.1f}s")

    def _save_timeout_checkpoint(self, feature_id: str, reason: str) -> str:
        """Save checkpoint on timeout using existing create_checkpoint method.

        NEW method - uses EXISTING db.create_checkpoint() API.
        FIX (Codex v3): Use correct signature with step_id and dict context.
        """
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
        # def create_checkpoint(workflow_id, step_id, git_sha, context: dict, status)
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

    def _run_sequential_with_timeout(
        self,
        feature_id: str,
        workflow_deadline: float | None = None,
    ) -> None:
        """Run components sequentially with workflow and role timeout checks.

        NEW method (v9): Wraps existing _run_sequential logic with deadline checks.
        FIX (v9): Ensures sequential execution respects workflow_timeout.
        FIX (v11): Also enforces per-role timeouts via ThreadPoolExecutor wrapper.
        """
        iteration = 0
        max_iterations = max(self._scheduler.get_component_count() * 10, 100)

        while not self._scheduler.is_feature_complete():
            # FIX (v17 - Codex): Check for human-initiated interrupts via bridge
            if self.interaction_bridge:
                # Check for cancellation signal
                if self.interaction_bridge.is_cancelled():
                    logger.info("Workflow cancelled by user via TUI")

                    # FIX (v23 - Codex): Mirror parallel cancel sequence
                    # Mark active components as FAILED
                    self._cancel_active_components(feature_id, "Cancelled by user")

                    # Update feature status and emit terminal event
                    self.db.update_feature_status(feature_id, FeatureStatus.FAILED)
                    from supervisor.core.state import Event, EventType
                    self.db.append_event(
                        Event(
                            workflow_id=feature_id,
                            event_type=EventType.WORKFLOW_FAILED,
                            payload={"reason": "cancelled_by_user"},
                        )
                    )

                    if self.checkpoint_on_timeout:
                        self._save_timeout_checkpoint(feature_id, "Cancelled by user")
                    from supervisor.core.engine import CancellationError
                    raise CancellationError("Cancelled by user via TUI")

                # Check for pause signal - block until resumed or cancelled
                self.interaction_bridge.wait_if_paused()

            # NEW (v9): Check workflow-level timeout before each component
            if workflow_deadline and time.time() > workflow_deadline:
                elapsed = time.time() - self._workflow_start_time
                self._handle_workflow_timeout(feature_id, elapsed)

            iteration += 1
            if iteration > max_iterations:
                raise WorkflowBlockedError(
                    f"Feature '{feature_id}' exceeded {max_iterations} iterations."
                )

            ready = self._scheduler.get_ready_components()
            if not ready:
                if self._scheduler.is_feature_blocked():
                    blocking = self._scheduler.get_blocking_components()
                    raise WorkflowBlockedError(
                        f"Feature '{feature_id}' is blocked: {blocking}"
                    )
                time.sleep(0.1)
                continue

            # FIX (v11): Execute with per-role timeout via ThreadPoolExecutor
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
```

#### Key Design Decisions (v7)

| Issue (Codex/Gemini) | Fix Applied |
|---------------------|-------------|
| No `run_feature` method exists | Use existing `run_implementation()` method, add timeout wrapper |
| No `Feature.phases` attribute | Not needed - scheduler executes feature-wide via `is_feature_complete()` |
| No `_cancelled` field | Use existing `CancellationError` and deadline check pattern |
| Checkpoint schema mismatch | Use existing `db.create_checkpoint(workflow_id, step_id, git_sha, context, status)` |
| Phase timeout not practical (v6/v7) | **REMOVED** - DAGScheduler has no `is_phase_complete()`. Focus on workflow + role timeouts |
| No `get_all_components()` method | Not needed - scheduler handles component iteration internally |
| No `_run_sequential_phase()` method | Not needed - scheduler executes feature-wide, not phase-by-phase |

#### Integration Points

The timeout extensions integrate with existing code:

```python
# supervisor/core/workflow.py - _run_continuous_parallel signature change

def _run_continuous_parallel(
    self,
    feature_id: str,
    num_components: int,
    workflow_deadline: float | None = None,  # NEW optional param
) -> None:
    """Run components in parallel with continuous scheduling.

    EXISTING: Uses ThreadPoolExecutor, checks component_timeout, max_stall_seconds
    NEW: Accept workflow_deadline for workflow-level timeout
    """
    # Add deadline check in main loop:
    while not self._scheduler.is_feature_complete():
        # NEW: Workflow-level timeout check
        # FIX (v11): Call _handle_workflow_timeout to save checkpoint before raising
        if workflow_deadline and time.time() > workflow_deadline:
            elapsed = time.time() - self._workflow_start_time
            self._handle_workflow_timeout(feature_id, elapsed)  # Saves checkpoint, raises CancellationError

        # ... rest of existing implementation unchanged ...
```

```python
# supervisor/core/workflow.py - Wire role_timeouts to _run_continuous_parallel (v9)
# FIX (v9): Per-role timeouts MUST be enforced in the timeout check loop,
# not just computed in _execute_component.

class WorkflowCoordinator:
    def __init__(self, ...):
        # ... existing ...
        self.role_timeouts = role_timeouts or {}  # Per-role timeout overrides

    def _run_continuous_parallel(
        self,
        feature_id: str,
        num_components: int,
        workflow_deadline: float | None = None,
    ) -> None:
        """Run components with continuous parallel scheduling.

        MODIFICATION (v9): Track per-component timeout based on role.
        """
        # ... existing setup ...

        # NEW (v9): Track per-component timeouts (keyed by future)
        # This replaces the single self.component_timeout check
        future_timeouts: dict[Future, float] = {}  # future -> deadline

        while not self._scheduler.is_feature_complete():
            # ... existing scheduling code ...

            # When submitting a component, compute its timeout
            for comp in submittable:
                role_name = comp.assigned_role or "implementer"
                # NEW (v9): Look up role-specific timeout
                timeout = self.role_timeouts.get(role_name, self.component_timeout)
                future = executor.submit(self._execute_component, comp, feature_id)
                active_futures[future] = comp
                future_start_times[future] = time.time()
                # NEW (v9): Store the deadline for this specific component
                future_timeouts[future] = time.time() + timeout

            # MODIFIED (v9): Check per-component timeouts using stored deadlines
            now = time.time()
            timed_out = [
                (f, c) for f, c in active_futures.items()
                if now > future_timeouts.get(f, now + 9999)  # Use stored deadline
                and not f.done()
            ]
            # ... rest of timeout handling unchanged ...
```

```python
# supervisor/core/workflow.py - ApprovalGate Integration
#
# FIX (v17 - Codex): EXPLICIT INTEGRATION PATH for approval gating.
#
# EXECUTION FLOW:
# 1. User calls: `supervisor workflow --tui <feature-id>` (CLI)
# 2. CLI creates WorkflowCoordinator with approval_gate and interaction_bridge
# 3. CLI calls: coordinator.run_implementation(feature_id) [may be in TUI thread]
# 4. run_implementation() calls _run_continuous_parallel() or _run_sequential_with_timeout()
# 5. These methods schedule components via executor.submit(self._execute_component, ...)
# 6. _execute_component() is WHERE approval gate check happens (below)
# 7. If approval required: blocks on InteractionBridge.request_approval()
# 8. TUI polls bridge.get_pending_requests() and shows approval dialog
# 9. User decision submitted via bridge.submit_decision()
# 10. _execute_component() continues or fails based on decision

def _execute_component(self, component: Component, feature_id: str) -> None:
    """Execute a single component.

    MODIFICATION (v10/v17): Call _check_approval_gate before running role.

    FIX (v17 - Codex): This is THE CALL SITE for approval gating.
    Called from: _run_continuous_parallel() and _run_sequential_with_timeout()

    FIX (v19 - Codex): Use IMPLEMENTING status, retain ModelRouter selection
    and cancellation_check from existing workflow.py pattern.
    """
    # FIX (v19): Use IMPLEMENTING (not IN_PROGRESS which doesn't exist)
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

    # FIX (v19): Load role config to get its configured CLI (existing pattern)
    role_config = self.engine.role_loader.load_role(role_name)
    role_cli = role_config.cli if role_config else None

    # FIX (v19): Use ModelRouter for intelligent model selection (existing pattern)
    estimated_context = len(component.files) * 5000 if component.files else 10000
    selected_model = self._router.select_model(
        role_name=role_name,
        role_cli=role_cli,
        context_size=estimated_context,
    )
    logger.debug(
        f"Component '{component.id}': Router selected '{selected_model}' for role '{role_name}'"
    )

    # FIX (v19): Create cancellation check to prevent applying changes after timeout
    def is_cancelled() -> bool:
        comp = self._scheduler.get_component(component.id)
        return comp is not None and comp.status == ComponentStatus.FAILED

    # FIX (v24 - Codex): Execute role FIRST, then check approval with actual diff
    # Previous versions checked approval BEFORE execution using only component.files,
    # which prevented showing the actual diff to the user. New flow:
    # 1. Run role (generates changes in worktree, does NOT commit)
    # 2. Capture actual git diff
    # 3. Check approval with real diff
    # 4. If approved: commit. If rejected: rollback worktree changes.
    try:
        # FIX (v19): Run role with router-selected CLI and cancellation check
        # Note: run_role generates changes but does NOT commit automatically
        result = self.engine.run_role(
            role_name=role_name,
            task_description=task_description,
            workflow_id=feature_id,
            target_files=component.files,
            cli_override=selected_model,
            cancellation_check=is_cancelled,
        )

        # FIX (v19): Check current status before updating to COMPLETED
        # A timed-out component may still complete; ignore late results
        comp = self._scheduler.get_component(component.id)
        if comp and comp.status == ComponentStatus.FAILED:
            logger.warning(
                f"Component '{component.id}' completed after timeout, ignoring result"
            )
            return

        # FIX (v25 - Codex): Capture BOTH changed file names AND diff after execution
        # - changed_files: List of file paths for risk assessment and counts
        # - diff_lines: Full diff for display in TUI/CLI
        # - untracked_files: Newly created files (not in git yet)
        changed_files, untracked_files = self._get_changed_files(component.files)
        diff_lines = self._get_worktree_diff(component.files)

        # Combine tracked and untracked for complete picture
        all_changed = changed_files + untracked_files

        # FIX (v24 - Codex): APPROVAL GATE CHECK - AFTER execution, BEFORE commit
        # This is the critical integration point that enables human approval of REAL changes
        if self.approval_gate:
            # Request approval with both file list (for risk) and diff (for display)
            # FIX (v25): Pass changed_files for risk assessment, diff_lines for display
            if not self._check_approval_gate(
                feature_id, component, all_changed, diff_lines, untracked_files
            ):
                # User rejected - rollback worktree changes (tracked + untracked)
                self._rollback_worktree_changes(component.files, untracked_files)
                self._scheduler.update_component_status(
                    component.id,
                    ComponentStatus.FAILED,
                    error="Approval rejected by user",
                    workflow_id=feature_id,
                )
                logger.info(f"Component '{component.id}' rejected by approval gate, changes rolled back")
                return  # Do NOT commit

        # Approval granted (or no gate configured) - commit changes
        # Note: Actual commit is handled by the workflow's commit phase (e.g., committer role)
        # This marks the component as ready for commit
        self._scheduler.update_component_status(
            component.id,
            ComponentStatus.COMPLETED,
            workflow_id=feature_id,
        )

    except Exception as e:
        logger.error(f"Component '{component.id}' failed: {e}")
        # Rollback any partial changes on error (tracked and untracked)
        # FIX (v25): Pass untracked_files to rollback for complete cleanup
        self._rollback_worktree_changes(component.files, untracked_files)
        self._scheduler.update_component_status(
            component.id,
            ComponentStatus.FAILED,
            error=str(e),
            workflow_id=feature_id,
        )

# FIX (v25 - Codex): Helper methods for file detection, diff capture, and rollback
def _get_changed_files(
    self, target_files: list[str] | None
) -> tuple[list[str], list[str]]:
    """Get lists of changed tracked files and new untracked files.

    Returns:
        (changed_files, untracked_files) tuple where:
        - changed_files: Modified/deleted tracked files from git diff --name-only
        - untracked_files: Newly created files from git status --porcelain
    """
    import subprocess
    changed = []
    untracked = []
    try:
        # Get modified tracked files
        cmd = ["git", "diff", "--name-only"]
        if target_files:
            cmd.extend(["--", *target_files])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            changed = result.stdout.strip().split("\n")

        # Get untracked files (newly created)
        # Filter by target_files if provided
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=30,
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
    return changed, untracked

def _get_worktree_diff(self, target_files: list[str] | None) -> list[str]:
    """Capture actual git diff from worktree after role execution.

    Returns a list of diff lines that can be shown to the user in the
    approval gate UI. This enables reviewing REAL changes, not just
    file names from the component spec.
    """
    import subprocess
    try:
        cmd = ["git", "diff", "--no-color"]
        if target_files:
            cmd.extend(["--", *target_files])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")
        return []
    except Exception as e:
        logger.warning(f"Failed to capture diff: {e}")
        return []

def _rollback_worktree_changes(
    self, target_files: list[str] | None, untracked_files: list[str] | None = None
) -> None:
    """Rollback uncommitted worktree changes after rejection.

    FIX (v25 - Codex): Now handles both tracked and untracked files.
    - Tracked files: Uses git checkout to discard changes
    - Untracked files: Removes newly created files

    Args:
        target_files: Tracked files to rollback via git checkout
        untracked_files: Newly created files to remove
    """
    import subprocess
    import os
    try:
        # Rollback tracked file changes
        cmd = ["git", "checkout", "--"]
        if target_files:
            cmd.extend(target_files)
        else:
            cmd.append(".")  # Rollback all changes
        subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
        logger.debug(f"Rolled back tracked changes for: {target_files or 'all files'}")

        # FIX (v25): Remove untracked files created by role
        if untracked_files:
            for filepath in untracked_files:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.debug(f"Removed untracked file: {filepath}")
                except Exception as file_err:
                    logger.warning(f"Failed to remove untracked file {filepath}: {file_err}")
    except Exception as e:
        logger.error(f"Failed to rollback worktree changes: {e}")
```

```python
# supervisor/core/state.py - use EXISTING create_checkpoint method

# The checkpoints table schema (EXISTING):
# CREATE TABLE IF NOT EXISTS checkpoints (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     workflow_id TEXT NOT NULL,
#     step_id TEXT,                     -- Can be NULL for workflow-level checkpoints
#     git_sha TEXT NOT NULL,            -- Use for timeout ID
#     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#     context_snapshot TEXT NOT NULL,   -- JSON state snapshot (serialized from dict)
#     status TEXT DEFAULT 'active',
#     FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
# );

# EXISTING create_checkpoint method signature (v4 FIX):
# def create_checkpoint(
#     self,
#     workflow_id: str,
#     step_id: str | None,     # FIX: step_id is required param
#     git_sha: str,
#     context: dict[str, Any], # FIX: dict not string - internally serialized
#     status: str = "active",
# ) -> int
```

**Acceptance Criteria:**
- [ ] Workflow-level timeout terminates long-running features
- [ ] Per-role timeout limits via `role_timeouts` dict (step-level)
- [ ] Checkpoints are saved on timeout for resume
- [ ] Cancellation token enables cooperative shutdown via existing `CancellationError`
- ~~Phase-level timeout~~ (OUT OF SCOPE - would require DAGScheduler refactor)
- ~~Grace period~~ (FUTURE: config exposed but not wired - could add delay before CancellationError)

---

### 5.2 Human Interrupt Interface (TUI)

**Goal:** Interactive terminal interface for workflow approval, rejection, and control.

**EXISTING IMPLEMENTATION:**
- `state.py:EventType` - Already has `APPROVAL_REQUESTED`, `APPROVAL_GRANTED`, `APPROVAL_DENIED`
- `cli.py` - Rich-based CLI with console output patterns
- `gates.py` - Gate system for verification checkpoints

**Files to Create/Modify:**
- `supervisor/core/interaction.py` (NEW) - Thread-safe bridge for workflow-TUI communication (FIX v13: moved to core to fix layering violation)
- `supervisor/core/approval.py` (NEW) - ApprovalGate and ApprovalPolicy (FIX v16: moved from TUI to Core)
- `supervisor/tui/__init__.py` (NEW) - TUI package
- `supervisor/tui/app.py` (NEW) - Main TUI application using Rich (SYNC, not async)
- `supervisor/cli.py` (MODIFY) - Add `supervisor workflow --tui <feature-id>` command
- `supervisor/cli.py` (MODIFY) - Update `init` command to create Phase 5 config files (FIX v17)

**FIX (v17 - Codex): Init Command Updates**

The existing `supervisor init` command must be updated to create Phase 5 configuration files:

```python
# supervisor/cli.py - Update to init command

@main.command()
def init() -> None:
    """Initialize project for supervisor."""
    repo_path = get_repo_path()
    supervisor_dir = repo_path / ".supervisor"

    # ... existing directory creation ...

    # FIX (v17): Create Phase 5 config files
    # These files are needed by the `workflow` command config loader

    limits_yaml = supervisor_dir / "limits.yaml"
    if not limits_yaml.exists():
        limits_yaml.write_text("""# Timeout configuration for Phase 5
timeout:
  workflow_timeout_seconds: 3600  # 1 hour total for entire feature
  step_timeout_seconds: 300       # 5 min per component (default)
  max_stall_seconds: 600          # 10 min max without progress
  checkpoint_on_timeout: true
  role_timeouts:
    planner: 600       # 10 min for planning
    investigator: 900  # 15 min for investigation
    reviewer: 180      # 3 min for reviews
""")

    adaptive_yaml = supervisor_dir / "adaptive.yaml"
    if not adaptive_yaml.exists():
        adaptive_yaml.write_text("""# Adaptive model routing configuration
adaptive:
  enabled: true
  min_samples_before_adapt: 10
  recalculation_interval: 10
  exploration_rate: 0.1
  score_weights:
    success_rate: 0.6  # FIX (v24 - Codex): Removed approval_rate (not implemented)
    avg_duration: 0.4  # Weights must sum to 1.0
""")

    approval_yaml = supervisor_dir / "approval.yaml"
    if not approval_yaml.exists():
        approval_yaml.write_text("""# Approval policy configuration
approval:
  auto_approve_low_risk: true
  risk_threshold: medium
  require_approval_for:
    - deploy
    - commit
""")

    console.print(
        Panel(
            "[green]Project initialized![/green]\\n\\n"
            f"Created: {supervisor_dir}\\n"
            "- config.yaml: Project configuration\\n"
            "- limits.yaml: Timeout configuration (Phase 5)\\n"
            "- adaptive.yaml: Adaptive routing config (Phase 5)\\n"
            "- approval.yaml: Approval policy config (Phase 5)\\n"
            "- roles/: Custom role definitions\\n"
            "- state.db: Workflow state database",
            title="Supervisor Initialized",
        )
    )
```

**FIX (v20 - Codex): Config Integration Details**

**Config File Locations:**
- `.supervisor/limits.yaml` - Timeout settings (created by `init`)
- `.supervisor/adaptive.yaml` - Adaptive routing settings (created by `init`)
- `.supervisor/approval.yaml` - Approval policy settings (created by `init`)
- `.supervisor/config.yaml` - Existing project config (Phase 1)

**Config Precedence:**
1. CLI flags (e.g., `--config-dir`) take highest precedence
2. `.supervisor/` directory (project-level, created by `init`)
3. Package defaults in `supervisor/config/` (fallback)

**Integration with Existing CLI:**
- `supervisor init` creates all config files in `.supervisor/`
- `supervisor workflow --tui <feature-id>` loads from `.supervisor/` by default
- Existing `supervisor run` command is unaffected (uses `.supervisor/config.yaml`)

```python
# supervisor/config/loader.py - FIX (v17/v20): Default to project config directory

from pathlib import Path
from typing import Any
import yaml

DEFAULT_CONFIG_DIR = Path(__file__).parent  # Package config directory


def load_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load all Phase 5 configuration.

    FIX (v17 - Codex): Default to .supervisor/ (project config), not package config.
    FIX (v20 - Codex): Explicit precedence and fallback behavior.

    Precedence:
    1. Explicit config_dir parameter
    2. Project .supervisor directory (if exists)
    3. Package defaults
    """
    if config_dir is None:
        # Try project .supervisor directory first
        project_config = Path.cwd() / ".supervisor"
        if project_config.exists():
            config_dir = project_config
        else:
            # Fall back to package defaults
            config_dir = DEFAULT_CONFIG_DIR

    config: dict[str, Any] = {}

    # Load each config file, using defaults if not found
    for config_name in ["limits", "adaptive", "approval"]:
        config_file = config_dir / f"{config_name}.yaml"
        if config_file.exists():
            with open(config_file) as f:
                data = yaml.safe_load(f)
                config[config_name] = data.get(config_name, {}) if data else {}
        else:
            # Use empty dict - get_*_config functions provide defaults
            config[config_name] = {}

    return config
```

**CRITICAL (v13/v16):** Core components are in **`supervisor/core/`**, NOT in `tui/`:
- `InteractionBridge`, `ApprovalRequest`, `ApprovalDecision` → `supervisor/core/interaction.py`
- `ApprovalGate`, `ApprovalPolicy` → `supervisor/core/approval.py` (v16)

This ensures Core has no dependency on TUI layer.

**CRITICAL DESIGN DECISION (v3):**
The core orchestration (`WorkflowCoordinator`, `ExecutionEngine`) is **synchronous and blocking**.
The TUI must run the workflow in a **background thread** and use thread-safe communication.
We use `threading.Event` and `queue.Queue` for sync/async bridging - NOT asyncio.

#### TUI Architecture (v3 - Fixed Sync/Async)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Thread (TUI)                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Rich Live Display                                        │   │
│  │  - Polls InteractionBridge.pending_requests              │   │
│  │  - User input via Rich Prompt (blocking OK on main)      │   │
│  │  - Calls bridge.submit_decision() when user decides      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ▲                                   │
│                              │ queue.Queue (thread-safe)         │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  InteractionBridge                                        │   │
│  │  - pending_requests: Queue[ApprovalRequest]              │   │
│  │  - decisions: dict[gate_id, threading.Event + decision]  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ▲                                   │
│                              │ threading.Event (blocking wait)   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Background Thread (Workflow)                             │   │
│  │  - Runs WorkflowCoordinator.run_implementation()         │   │
│  │  - At approval gate, calls bridge.request_approval()     │   │
│  │  - Blocks on threading.Event until TUI submits decision  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### TUI Application Design

```python
# supervisor/core/interaction.py (FIX v13: moved to core layer)
"""Thread-safe bridge for workflow-TUI communication.

FIX (Gemini review): Provides blocking sync API for workflow thread
and polling API for TUI main thread.

FIX (v13 - Gemini): Placed in CORE layer, not TUI layer, to ensure
WorkflowCoordinator can import without depending on presentation layer.
"""

import logging
import queue
import threading
import time  # FIX (v19 - Codex): Added missing import for wait_if_paused
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ApprovalDecision(str, Enum):
    """User decision for approval gate.

    FIX (v17 - Codex): Added explicit semantics for each decision type.

    APPROVE: Proceed with the proposed changes as-is.
             Workflow continues normally.

    REJECT: Stop the proposed changes entirely.
            Component is marked as FAILED, workflow may halt or retry.

    EDIT: User wants to modify the proposed changes before proceeding.
          Workflow should pause, allow user edits, then re-request approval.
          NOTE (v19): NOT YET IMPLEMENTED. Reserved for future enhancement.
          Currently not exposed in TUI UI to avoid confusion.

    SKIP: Skip this approval gate without approving or rejecting.
          Component proceeds but is flagged for later review.
          Used for: timeout fallback, low-confidence auto-decisions.
    """
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    SKIP = "skip"

    def is_proceed(self) -> bool:
        """Returns True if workflow should proceed (APPROVE or SKIP)."""
        return self in (ApprovalDecision.APPROVE, ApprovalDecision.SKIP)

    def is_blocking(self) -> bool:
        """Returns True if this decision blocks workflow (REJECT or EDIT)."""
        return self in (ApprovalDecision.REJECT, ApprovalDecision.EDIT)


@dataclass
class ApprovalRequest:
    """Request for human approval.

    FIX (v19 - Codex): Added expiry_time to prevent stale requests from
    appearing in TUI after they've timed out/cancelled.
    FIX (v25 - Codex): Added diff_lines for displaying actual git diff.
    """
    gate_id: str
    feature_id: str
    component_id: str | None
    title: str
    description: str
    risk_level: str
    changes: list[str]  # File paths (for counts and risk assessment)
    review_summary: str
    diff_lines: list[str] | None = None  # FIX (v25): Actual git diff for display
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
        # FIX (v18): Merged interrupt control events into single __init__
        self._cancelled = threading.Event()
        self._paused = threading.Event()

    def request_approval(
        self,
        request: ApprovalRequest,
        timeout: float = 300.0,
    ) -> ApprovalDecision:
        """Request approval from TUI. BLOCKS until decision received.

        Called from workflow thread.

        FIX (v18 - Codex): Added cancellation check in wait loop so that
        pressing [c] in TUI can interrupt an approval-blocked workflow.
        """
        # Create event for this request
        event = threading.Event()
        with self._lock:
            self._decisions[request.gate_id] = (event, None)

        # FIX (v19): Set expiry time to prevent stale requests in TUI
        request.expiry_time = time.time() + timeout

        # Queue request for TUI
        self._pending_requests.put(request)

        # FIX (v18): Poll with short intervals to check for cancellation
        # Instead of event.wait(timeout), poll in short intervals
        POLL_INTERVAL = 0.5  # Check every 500ms
        elapsed = 0.0

        while elapsed < timeout:
            # Check for cancellation signal from TUI
            if self._cancelled.is_set():
                with self._lock:
                    self._decisions.pop(request.gate_id, None)
                logger.info(f"Approval request '{request.gate_id}' cancelled by user")
                return ApprovalDecision.REJECT  # Treat cancel as rejection

            # Wait for short interval
            if event.wait(timeout=POLL_INTERVAL):
                # Decision received
                with self._lock:
                    _, decision = self._decisions.pop(request.gate_id)
                    return decision if decision else ApprovalDecision.SKIP

            elapsed += POLL_INTERVAL

        # Timeout - clean up and return skip
        with self._lock:
            self._decisions.pop(request.gate_id, None)
        logger.warning(f"Approval request '{request.gate_id}' timed out")
        return ApprovalDecision.SKIP

    def get_pending_requests(self) -> list[ApprovalRequest]:
        """Get all pending approval requests. NON-BLOCKING.

        Called from TUI main thread.

        FIX (v19 - Codex): Filters out expired requests to prevent stale
        prompts after timeout/cancel.
        """
        requests = []
        now = time.time()
        while True:
            try:
                request = self._pending_requests.get_nowait()
                # FIX (v19): Filter out expired requests
                if request.expiry_time is None or now < request.expiry_time:
                    requests.append(request)
                else:
                    logger.debug(f"Discarding expired approval request '{request.gate_id}'")
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

    # FIX (v17/v18 - Codex): Added interrupt control signals for human-initiated workflow control

    def signal_cancellation(self) -> None:
        """Signal workflow cancellation from TUI.

        FIX (v17 - Codex): Enables human-initiated workflow cancel.
        Workflow checks this via is_cancelled() in execution loops.
        """
        self._cancelled.set()

    def signal_pause(self) -> None:
        """Signal workflow pause from TUI.

        FIX (v17 - Codex): Enables human-initiated workflow pause.
        """
        self._paused.set()

    def signal_resume(self) -> None:
        """Signal workflow resume from TUI.

        FIX (v17 - Codex): Clears pause signal to resume execution.
        """
        self._paused.clear()

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested.

        Called from workflow thread in execution loops.
        """
        return self._cancelled.is_set()

    def is_paused(self) -> bool:
        """Check if pause was requested.

        Called from workflow thread in execution loops.
        """
        return self._paused.is_set()

    def wait_if_paused(self, check_interval: float = 0.5) -> None:
        """Block while paused, checking for cancellation.

        Called from workflow thread. Returns when unpaused or cancelled.
        """
        while self._paused.is_set() and not self._cancelled.is_set():
            time.sleep(check_interval)
```

```python
# supervisor/tui/app.py
"""Terminal User Interface for supervisor workflow control.

FIX (Gemini/Codex review v3): SYNCHRONOUS design using Rich Live display.
Workflow runs in background thread, communicates via InteractionBridge.

USES EXISTING:
- EventType.APPROVAL_REQUESTED - Triggered when approval needed
- EventType.APPROVAL_GRANTED - User approves
- EventType.APPROVAL_DENIED - User rejects
- db.get_phases(feature_id) - EXISTING DB method
- db.get_components(feature_id) - EXISTING DB method
- db.get_events(...) - EXISTING DB method
- ComponentStatus.COMPLETED - CORRECT enum value (not COMPLETE)
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt  # FIX (v19): Added Confirm for keyboard handlers
from rich.table import Table
from rich.tree import Tree

from supervisor.core.state import Database
from supervisor.core.models import ComponentStatus, FeatureStatus
from supervisor.core.interaction import ApprovalDecision, ApprovalRequest, InteractionBridge

logger = logging.getLogger(__name__)


class SupervisorTUI:
    """Main TUI application for supervisor.

    FIX (Gemini review): SYNCHRONOUS - runs Rich Live in main thread,
    workflow in background thread via InteractionBridge.

    FIX (v12): Accept external bridge instead of creating own instance.
    This allows CLI to share same bridge with WorkflowCoordinator.

    USAGE:
        # Create shared bridge
        bridge = InteractionBridge()

        # Pass same bridge to coordinator AND TUI
        coordinator = WorkflowCoordinator(..., interaction_bridge=bridge)
        tui = SupervisorTUI(db, bridge=bridge)

        tui.run_with_workflow(
            workflow_fn=lambda: coordinator.run_implementation(feature_id),
            feature_id=feature_id,
        )
    """

    def __init__(self, db: Database, bridge: "InteractionBridge | None" = None):
        self.db = db
        self.console = Console()
        # FIX (v12): Use injected bridge or create default for standalone testing
        self.bridge = bridge if bridge is not None else InteractionBridge()
        self._running = False
        self._workflow_thread: threading.Thread | None = None
        self._current_feature_id: str | None = None

    def run_with_workflow(
        self,
        workflow_fn: Callable[[], None],
        feature_id: str,
    ) -> None:
        """Run TUI with workflow executing in background thread.

        Args:
            workflow_fn: Function to run workflow (e.g., coordinator.run_implementation)
            feature_id: Feature being executed (for status display)
        """
        self._running = True
        self._current_feature_id = feature_id

        # Start workflow in background thread
        self._workflow_thread = threading.Thread(
            target=self._run_workflow_wrapper,
            args=(workflow_fn,),
            daemon=True,
        )
        self._workflow_thread.start()

        # Run TUI in main thread (blocking)
        try:
            self._run_tui_loop()
        finally:
            self._running = False

    def _run_workflow_wrapper(self, workflow_fn: Callable[[], None]) -> None:
        """Wrapper to run workflow and catch exceptions."""
        try:
            workflow_fn()
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
        finally:
            self._running = False  # Signal TUI to stop

    def _run_tui_loop(self) -> None:
        """Main TUI loop - runs in main thread.

        FIX (v17 - Codex): Added keyboard handling for human interrupt controls.
        Supports: [c]ancel, [p]ause, [r]esume, [q]uit

        FIX (v18 - Codex): Use cbreak terminal mode for responsive single-key input.
        """
        import sys

        # FIX (v18): Set up terminal for cbreak mode (single-key reads without Enter)
        old_settings = None
        if sys.platform != "win32":
            import termios
            import tty
            try:
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except termios.error:
                old_settings = None  # Not a TTY, skip cbreak mode

        try:
            with Live(self._generate_layout(), refresh_per_second=4, console=self.console) as live:
                while self._running:
                    live.update(self._generate_layout())

                    # FIX (v17/v18): Check for keyboard input for interrupt controls
                    key = self._check_keyboard_input()
                    if key:
                        self._handle_keyboard_command(key, live)

                    # Check for pending approvals via bridge (NON-BLOCKING)
                    pending = self.bridge.get_pending_requests()
                    for request in pending:
                        # Pause Live display for user input
                        # FIX (v18): Temporarily restore normal terminal mode for Rich prompts
                        live.stop()
                        if old_settings:
                            import termios
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

                        decision = self._handle_approval_sync(request)
                        self.bridge.submit_decision(request.gate_id, decision)

                        # Restore cbreak mode
                        if old_settings:
                            import tty
                            tty.setcbreak(sys.stdin.fileno())
                        live.start()

                    time.sleep(0.25)  # SYNC sleep, not async
        finally:
            # FIX (v18): Restore original terminal settings
            if old_settings:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _check_keyboard_input(self) -> str | None:
        """Check for keyboard input without blocking.

        FIX (v17 - Codex): Non-blocking keyboard check for interrupt controls.
        FIX (v18 - Codex): Works with cbreak mode set by _run_tui_loop.
        """
        import sys
        import select

        # Non-blocking check for stdin
        # In cbreak mode, single keypresses are available immediately
        if sys.platform != "win32":
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
        else:
            # Windows: Use msvcrt for non-blocking key check
            import msvcrt
            if msvcrt.kbhit():
                return msvcrt.getch().decode('utf-8', errors='ignore').lower()
        return None

    def _handle_keyboard_command(self, key: str, live: Live) -> None:
        """Handle keyboard commands for workflow control.

        FIX (v17 - Codex): Implements human interrupt controls.
        Commands:
        - 'c': Cancel workflow (raises CancellationError via bridge)
        - 'p': Pause workflow
        - 'r': Resume paused workflow
        - 'q': Quit TUI (same as cancel)
        """
        if key in ('c', 'q'):
            # Cancel workflow - signal via bridge
            live.stop()
            if Confirm.ask("[bold red]Cancel workflow?[/bold red]", default=False):
                self.bridge.signal_cancellation()
                self._running = False
                self.console.print("[yellow]Workflow cancellation requested[/yellow]")
            live.start()

        elif key == 'p':
            # Pause workflow
            self.bridge.signal_pause()
            self.console.print("[yellow]Workflow paused. Press 'r' to resume.[/yellow]")

        elif key == 'r':
            # Resume workflow
            self.bridge.signal_resume()
            self.console.print("[green]Workflow resumed.[/green]")

    def _generate_layout(self) -> Layout:
        """Generate the TUI layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["header"].update(self._render_header())
        layout["body"].split_row(
            Layout(name="status", ratio=1),
            Layout(name="details", ratio=2),
        )
        layout["status"].update(self._render_workflow_status())
        layout["details"].update(self._render_details())
        layout["footer"].update(self._render_footer())

        return layout

    def _render_header(self) -> Panel:
        """Render header with title and stats."""
        return Panel(
            "[bold blue]Supervisor[/bold blue] - AI Workflow Orchestrator",
            style="blue",
        )

    def _render_workflow_status(self) -> Panel:
        """Render current workflow status.

        FIX (Codex review): Use EXISTING db.get_phases() and db.get_components()
        """
        tree = Tree("[bold]Active Workflows[/bold]")

        if self._current_feature_id:
            feature = self.db.get_feature(self._current_feature_id)
            if feature:
                feature_node = tree.add(f"[cyan]{feature.id}[/cyan]: {feature.title}")

                # FIX: Use EXISTING db.get_phases(feature_id)
                phases = self.db.get_phases(feature.id)
                for phase in phases:
                    status_icon = self._status_icon(phase.status)
                    phase_node = feature_node.add(f"{status_icon} {phase.title}")

                    # FIX: Use db.get_components(feature_id) and filter by phase
                    all_components = self.db.get_components(feature.id)
                    phase_components = [c for c in all_components if c.phase_id == phase.id]
                    # FIX: Use ComponentStatus.COMPLETED (not COMPLETE)
                    completed = sum(1 for c in phase_components if c.status == ComponentStatus.COMPLETED)
                    total = len(phase_components)
                    phase_node.add(f"[dim]{completed}/{total} components[/dim]")

        return Panel(tree, title="Workflow Status")

    def _render_details(self) -> Panel:
        """Render details panel (logs when no approval pending)."""
        return self._render_logs_panel()

    def _render_logs_panel(self) -> Panel:
        """Render recent logs panel.

        FIX (Codex v3): get_events requires workflow_id, has no limit param.
        FIX (v8): PERFORMANCE - Python slice acceptable for MVP since events
        list is bounded by workflow duration. For very long workflows, consider
        adding Database.get_recent_events(workflow_id, limit) with SQL LIMIT.
        """
        table = Table(show_header=True, header_style="bold")
        table.add_column("Time", style="dim")
        table.add_column("Type")
        table.add_column("Details")

        # FIX (Codex v3): get_events(workflow_id) is the actual signature
        # FIX (v8): Fetches all then slices - OK for MVP, SQL LIMIT for scale
        if self._current_feature_id:
            events = self.db.get_events(workflow_id=self._current_feature_id)
            # Show last 20 events (most recent) - Python slice for simplicity
            for event in events[-20:]:
                table.add_row(
                    event.timestamp.strftime("%H:%M:%S") if event.timestamp else "",
                    event.event_type.value,
                    str(event.payload)[:50] + "..." if len(str(event.payload)) > 50 else str(event.payload),
                )

        return Panel(table, title="Recent Events")

    def _status_icon(self, status) -> str:
        """Get icon for status.

        FIX (Codex v3): Map "completed" not "complete" to match ComponentStatus.COMPLETED.
        FIX (v11): Use status.value for Enum types, not str(status).lower().
        """
        status_icons = {
            "pending": "⏳",
            "in_progress": "🔄",
            "implementing": "🔧",
            "testing": "🧪",
            "review": "👀",
            "completed": "✅",  # FIX: was "complete"
            "failed": "❌",
        }
        # FIX (v11): Handle Enum types properly - use .value not str()
        status_key = status.value if hasattr(status, 'value') else str(status).lower()
        return status_icons.get(status_key, "❓")

    def _render_footer(self) -> Panel:
        """Render footer with help."""
        return Panel(
            "[Q]uit  [P]ause  [R]esume  [C]ancel  [?]Help",
            style="dim",
        )

    def _handle_approval_sync(self, request: ApprovalRequest) -> ApprovalDecision:
        """Handle interactive approval request. SYNCHRONOUS.

        FIX (Gemini review): Changed from async to sync method.
        Called from main thread when Live display is paused.
        """
        self.console.print()
        self.console.rule("[bold]Approval Required[/bold]")

        # Show details
        self.console.print(f"\n[bold]Feature:[/bold] {request.feature_id}")
        self.console.print(f"[bold]Gate:[/bold] {request.gate_id}")
        self.console.print(f"[bold]Risk:[/bold] [{self._risk_color(request.risk_level)}]{request.risk_level}[/]")

        self.console.print("\n[bold]Changes:[/bold]")
        for change in request.changes:
            self.console.print(f"  • {change}")

        self.console.print(f"\n[bold]Review Summary:[/bold]\n{request.review_summary}")

        # Get decision (Rich Prompt is blocking, which is fine in main thread)
        # FIX (v19 - Codex): Removed EDIT option until edit workflow is implemented
        # EDIT semantics require: pause, capture edits, re-submit for approval
        self.console.print("\n[dim][a]pprove [r]eject [s]kip[/dim]")
        choice = Prompt.ask(
            "\n[bold]Decision[/bold]",
            choices=["a", "r", "s"],
            default="a",
        )

        decision_map = {
            "a": ApprovalDecision.APPROVE,
            "r": ApprovalDecision.REJECT,
            # "d": ApprovalDecision.EDIT,  # FIX (v19): Disabled until implemented
            "s": ApprovalDecision.SKIP,
        }

        decision = decision_map.get(choice, ApprovalDecision.SKIP)

        # Handle rejection reason
        if decision == ApprovalDecision.REJECT:
            reason = Prompt.ask("[bold]Rejection reason[/bold]")
            logger.info(f"Approval rejected: {reason}")

        return decision

    # NOTE (v8): Async request_approval REMOVED - use InteractionBridge.request_approval() instead
    # The workflow thread calls bridge.request_approval() which is SYNC (blocks on Event)
    # TUI main thread polls bridge.get_pending_requests() and calls bridge.submit_decision()

    # NOTE (v8): Duplicate _status_icon REMOVED - defined once in _render_progress_panel
    # The correct mapping uses "completed" (not "complete") to match ComponentStatus enum

    def _risk_color(self, risk: str) -> str:
        """Get color for risk level."""
        colors = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
            "critical": "bold red",
        }
        return colors.get(risk.lower(), "white")

    def stop(self) -> None:
        """Stop the TUI."""
        self._running = False
```

#### Approval Gate Integration

```python
# supervisor/core/approval.py (FIX v16: Moved from TUI to Core layer)
"""Approval gate integration for human-in-the-loop workflows."""

import logging
from dataclasses import dataclass, field  # FIX (v14): Added field for default_factory
from typing import Any

from supervisor.core.state import Database, Event, EventType
# FIX (v10): ApprovalRequest/ApprovalDecision are in core interaction
from supervisor.core.interaction import ApprovalRequest, ApprovalDecision, InteractionBridge
# FIX (v16): Removed SupervisorTUI import - core should not depend on TUI

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
        if file_count > 20:
            return "high"
        elif file_count > 10:
            return "medium"
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

        # Log decision using EXISTING EventTypes
        # FIX (v18 - Codex): Proper semantic mapping for all decision types
        # APPROVE -> GRANTED, SKIP -> GRANTED (with skip flag), REJECT/EDIT -> DENIED
        if decision in (ApprovalDecision.APPROVE, ApprovalDecision.SKIP):
            event_type = EventType.APPROVAL_GRANTED
        else:
            event_type = EventType.APPROVAL_DENIED

        self.db.append_event(
            Event(
                workflow_id=feature_id,
                event_type=event_type,
                payload={
                    "gate_id": gate_id,
                    "decision": decision.value,
                    "needs_review": decision == ApprovalDecision.SKIP,  # Flag for later review
                },
            )
        )

        return decision

    def _cli_approval_sync(self, request: ApprovalRequest) -> ApprovalDecision:
        """Fallback CLI-based approval. SYNCHRONOUS.

        FIX (Gemini review): Changed from async to sync.
        FIX (v22 - Codex): Added non-interactive safeguard for CI/headless runs.
        """
        import sys
        from rich.console import Console
        from rich.prompt import Confirm

        console = Console()

        # FIX (v22): Non-interactive safeguard for CI/headless runs
        if not sys.stdin.isatty():
            # Non-interactive mode - use policy-based auto-decision
            if self.policy.auto_approve_low_risk and request.risk_level == "low":
                logger.info(f"Non-interactive: auto-approving low-risk request '{request.gate_id}'")
                return ApprovalDecision.APPROVE
            else:
                # Default to SKIP in non-interactive mode to allow workflow to proceed
                # with flagged review later
                logger.warning(
                    f"Non-interactive: skipping approval for '{request.gate_id}' (flagged for review)"
                )
                return ApprovalDecision.SKIP

        # Interactive mode - prompt user
        console.print(f"\n[bold red]Approval Required[/bold red]: {request.title}")
        console.print(f"Risk Level: [{request.risk_level}]")
        console.print(f"Changes: {len(request.changes)} files")

        for change in request.changes[:5]:
            console.print(f"  • {change}")
        if len(request.changes) > 5:
            console.print(f"  ... and {len(request.changes) - 5} more")

        approved = Confirm.ask("Approve?", default=True)
        return ApprovalDecision.APPROVE if approved else ApprovalDecision.REJECT
```

**Acceptance Criteria:**
- [ ] TUI displays workflow status in real-time (sync via Rich Live)
- [ ] Approval gates pause workflow via InteractionBridge (blocking)
- [ ] Users can approve, reject via Rich Prompt
- [ ] Risk assessment determines approval requirements
- [ ] Events are logged for all approval decisions using existing EventTypes

---

### 5.3 Metrics Dashboard

**Goal:** Comprehensive metrics collection, aggregation, and visualization.

**CRITICAL FIX (Codex/Gemini review):** Must explicitly instrument `ExecutionEngine.run_role()` to call the collector.

**Files to Create/Modify:**
- `supervisor/metrics/__init__.py` (NEW) - Metrics package
- `supervisor/metrics/collector.py` (NEW) - Event-based metrics collection
- `supervisor/metrics/aggregator.py` (NEW) - Metrics aggregation
- `supervisor/metrics/dashboard.py` (NEW) - Rich-based dashboard
- `supervisor/core/state.py` (NO CHANGE) - Metrics table and record_metric() ALREADY EXIST (v20 fix)
- `supervisor/core/engine.py` (MODIFY) - Instrument run_role() to record metrics
- `supervisor/cli.py` (MODIFY) - Add `supervisor metrics` command

#### Metrics Schema

```sql
-- Add to supervisor/core/state.py schema

-- Metrics table for performance tracking
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Dimensions
    role TEXT NOT NULL,
    cli TEXT NOT NULL,
    task_type TEXT,  -- FIX (v11): Standardized values: 'plan', 'implement', 'review', 'test', 'investigate', 'document', 'other'
    workflow_id TEXT,

    -- Measures
    success BOOLEAN NOT NULL,
    duration_seconds REAL NOT NULL,
    retry_count INTEGER DEFAULT 0,
    token_usage INTEGER,
    error_category TEXT,

    -- For adaptive routing
    model_score REAL  -- Computed success rate for this role/cli combo
);

CREATE INDEX IF NOT EXISTS idx_metrics_role_cli ON metrics(role, cli);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
```

#### Database API Extension

```python
# supervisor/core/state.py - ADD record_metric method

class Database:
    # ... existing methods ...

    def record_metric(
        self,
        role: str,
        cli: str,
        workflow_id: str,
        success: bool,
        duration_seconds: float,
        task_type: str = "other",
        retry_count: int = 0,
        token_usage: int | None = None,
        error_category: str | None = None,
    ) -> None:
        """Record a metric for role execution.

        NEW method for Phase 5 - uses _connect() context manager like existing methods.
        FIX (v15 - Gemini): MUST use `with self._connect() as conn:` pattern.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics (
                    role, cli, task_type, workflow_id,
                    success, duration_seconds, retry_count,
                    token_usage, error_category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    role, cli, task_type, workflow_id,
                    success, duration_seconds, retry_count,
                    token_usage, error_category,
                )
            )
            # commit handled by context manager

    def get_metrics(
        self,
        days: int = 30,
        role: str | None = None,
        cli: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query metrics with optional filters.

        NEW method for Phase 5 - supports metrics aggregation.
        FIX (v15 - Gemini): MUST use `with self._connect() as conn:` pattern.
        FIX (v14 - Codex): Use SQLite datetime() for cross-format comparison.
        """
        with self._connect() as conn:
            # FIX (v14): Use datetime() for format-agnostic timestamp comparison
            query = f"SELECT * FROM metrics WHERE timestamp > datetime('now', '-{days} days')"
            params: list[Any] = []

            if role:
                query += " AND role = ?"
                params.append(role)
            if cli:
                query += " AND cli = ?"
                params.append(cli)

            query += " ORDER BY timestamp DESC"
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
```

#### Engine Instrumentation

**FIX (Codex v3):** Drop `enable_metrics` toggle - always record when `db` exists.
This avoids requiring changes to all CLI/test instantiation sites.

```python
# supervisor/core/engine.py - MODIFY run_role to record metrics

class ExecutionEngine:
    """Execute roles with circuit breaker, retry, and metrics.

    EXISTING: db attribute already exists (passed in __init__)
    NEW: run_role records metrics in finally block (no new __init__ params)
    """

    # NO CHANGE to __init__ - db already exists as self.db

    def run_role(
        self,
        role_name: str,
        task_description: str,
        workflow_id: str,
        step_id: str | None = None,  # FIX (v8): Keep existing param
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
        retry_policy: RetryPolicy | None = None,  # FIX (v8): Keep existing param
        gates: list[str] | None = None,  # FIX (v8): Keep existing param
        cli_override: str | None = None,  # FIX (v8): Keep existing param
        cancellation_check: Callable[[], bool] | None = None,
    ) -> BaseModel:
        """Execute a role with the specified CLI.

        MODIFICATION: Record metrics after execution.
        FIX (v8): Signature matches EXISTING engine.py - all params preserved.
        """
        import time

        role_config = self.role_loader.load_role(role_name)
        cli = cli_override or role_config.cli  # FIX (v8): Respect cli_override

        start_time = time.time()
        success = False
        error_category = None
        attempt_count = 0  # FIX (v10): Track retry count for metrics

        try:
            # EXISTING: The actual retry loop (NOT _execute_with_retry - that doesn't exist)
            # See engine.py:694-773 for actual implementation:
            # for attempt in range(retry_policy.max_attempts):
            #     with self.workspace.isolated_execution(step_id) as ctx:
            #         result = self._execute_cli(role, prompt, ctx.worktree_path, cli_override)
            #         ... parse output, run gates ...
            #
            # FIX (v9): This section shows WHERE to add metrics instrumentation,
            # not a new method. The metrics recording goes in the EXISTING finally block.

            # ... existing retry loop (tracks attempt_count) ...
            for attempt in range(retry_policy.max_attempts):
                attempt_count = attempt  # FIX (v10): Track for metrics
                # ... existing attempt logic ...
                pass

            result = ...  # placeholder - actual code is in engine.py
            success = True
            return result

        except Exception as e:
            error_category = type(e).__name__
            raise

        finally:
            # NEW: Record metrics if db exists (no toggle needed)
            # FIX (Codex v3): Check hasattr instead of requiring new init param
            if hasattr(self, 'db') and self.db is not None:
                duration = time.time() - start_time
                # FIX (v10): Infer task_type from role name
                task_type = self._infer_task_type(role_name)
                try:
                    self.db.record_metric(
                        role=role_name,
                        cli=cli,
                        workflow_id=workflow_id,
                        success=success,
                        duration_seconds=duration,
                        task_type=task_type,  # FIX (v10): Required for adaptive routing
                        retry_count=attempt_count,  # FIX (v10): Required for adaptive routing
                        error_category=error_category,
                    )
                except Exception as metric_err:
                    # Don't fail execution if metrics recording fails
                    logger.warning(f"Failed to record metric: {metric_err}")

    def _infer_task_type(self, role_name: str) -> str:
        """Infer task type from role name for metrics categorization.

        FIX (v11): Standardized task_type values to match schema and router.
        Values: plan, implement, review, test, investigate, document, other
        """
        role_to_task = {
            "planner": "plan",
            "implementer": "implement",
            "reviewer": "review",
            "investigator": "investigate",
            "doc_generator": "document",
            "tester": "test",
        }
        return role_to_task.get(role_name, "other")
```

#### Metrics Collector (Optional Wrapper)

```python
# supervisor/metrics/collector.py
"""Optional collector wrapper - can derive metrics from events instead.

FIX (Codex review): The primary collection path is via Engine instrumentation.
This collector can optionally derive metrics from events for existing data.

FIX (v13 - Codex): Added class wrapper and proper imports.
"""

import logging
import time
from supervisor.core.state import Database

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Optional helper for metrics timing.

    FIX (v13 - Codex): This class was missing - methods were orphaned.
    Primary metrics collection is via Engine.run_role() instrumentation.
    This class provides optional timing helpers.
    """

    def __init__(self, db: Database):
        self.db = db
        self._step_start_times: dict[str, float] = {}

    def _infer_task_type(self, role: str) -> str:
        """Infer task type from role name.

        FIX (v11): Standardized task_type values across schema, engine, router.
        """
        role_lower = role.lower()
        if "plan" in role_lower:
            return "plan"
        elif "review" in role_lower:
            return "review"
        elif "implement" in role_lower:
            return "implement"
        elif "test" in role_lower:
            return "test"
        elif "investigat" in role_lower:
            return "investigate"
        elif "doc" in role_lower:
            return "document"
        else:
            return "other"

    def start_step(self, step_id: str) -> None:
        """Record step start time for duration calculation."""
        self._step_start_times[step_id] = time.time()

    def end_step(self, step_id: str) -> float:
        """Get duration and cleanup step tracking."""
        start = self._step_start_times.pop(step_id, None)
        if start:
            return time.time() - start
        return 0.0
```

#### Metrics Aggregator

**FIX (Codex v3):** MetricsAggregator uses `db._connect()` pattern instead of non-existent `db.execute()`.

```python
# supervisor/metrics/aggregator.py
"""Metrics aggregation and analysis.

FIX (Codex v3): Uses db._connect() pattern like other Database methods.
Cannot use db.execute() as it doesn't exist.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from supervisor.core.state import Database

logger = logging.getLogger(__name__)


@dataclass
class RolePerformance:
    """Aggregated performance for a role."""
    role: str
    success_rate: float
    avg_duration_seconds: float
    avg_retries: float
    total_executions: int

    @property
    def formatted_success_rate(self) -> str:
        return f"{self.success_rate * 100:.1f}%"


@dataclass
class CLIPerformance:
    """Aggregated performance for a CLI by task type."""
    cli: str
    task_type: str
    success_rate: float
    avg_duration_seconds: float
    total_executions: int


class MetricsAggregator:
    """Aggregate and analyze metrics.

    FIX (Codex v3): Uses db._connect() pattern to access SQLite.

    USAGE:
        aggregator = MetricsAggregator(db)
        roles = aggregator.get_role_performance(days=30)
        cli_stats = aggregator.get_cli_comparison(days=30)
    """

    def __init__(self, db: Database):
        self.db = db

    def get_role_performance(
        self,
        days: int = 30,
    ) -> list[RolePerformance]:
        """Get performance metrics by role.

        FIX (Codex v3): Uses db._connect() pattern.
        FIX (v14): Use datetime() for format-agnostic timestamp comparison.
        """
        # FIX: Use _connect() pattern like other Database methods
        with self.db._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    role,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_seconds) as avg_duration,
                    AVG(retry_count) as avg_retries,
                    COUNT(*) as total
                FROM metrics
                WHERE timestamp > datetime('now', '-{days} days')
                GROUP BY role
                ORDER BY success_rate DESC
                """
            ).fetchall()

        return [
            RolePerformance(
                role=row[0],
                success_rate=row[1] or 0.0,
                avg_duration_seconds=row[2] or 0.0,
                avg_retries=row[3] or 0.0,
                total_executions=row[4],
            )
            for row in rows
        ]

    def get_cli_comparison(
        self,
        days: int = 30,
    ) -> list[CLIPerformance]:
        """Get performance comparison across CLIs.

        FIX (Codex v3): Uses db._connect() pattern.
        FIX (v14): Use datetime() for format-agnostic timestamp comparison.
        """
        with self.db._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    cli,
                    task_type,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_seconds) as avg_duration,
                    COUNT(*) as total
                FROM metrics
                WHERE timestamp > datetime('now', '-{days} days')
                GROUP BY cli, task_type
                ORDER BY task_type, success_rate DESC
                """
            ).fetchall()

        return [
            CLIPerformance(
                cli=row[0],
                task_type=row[1],
                success_rate=row[2] or 0.0,
                avg_duration_seconds=row[3] or 0.0,
                total_executions=row[4],
            )
            for row in rows
        ]

    def get_best_cli_for_task(
        self,
        task_type: str,
        days: int = 30,
        min_samples: int = 10,
    ) -> str | None:
        """Get the best performing CLI for a task type.

        FIX (Codex v3): Uses db._connect() pattern.
        FIX (v14): Use datetime() for format-agnostic timestamp comparison.
        """
        with self.db._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    cli,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_seconds) as avg_duration,
                    COUNT(*) as total
                FROM metrics
                WHERE timestamp > datetime('now', '-{days} days') AND task_type = ?
                GROUP BY cli
                HAVING COUNT(*) >= ?
                ORDER BY success_rate DESC
                LIMIT 1
                """,
                (task_type, min_samples)
            ).fetchone()

        return rows[0] if rows else None

    def get_recent_failures(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent execution failures for debugging.

        FIX (Codex v3): Uses db._connect() pattern.
        """
        with self.db._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    timestamp,
                    role,
                    cli,
                    workflow_id,
                    error_category,
                    duration_seconds
                FROM metrics
                WHERE success = 0
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        return [
            {
                "timestamp": row[0],
                "role": row[1],
                "cli": row[2],
                "workflow_id": row[3],
                "error_category": row[4],
                "duration_seconds": row[5],
            }
            for row in rows
        ]

    def get_summary_stats(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get high-level summary statistics.

        FIX (Codex v3): Uses db._connect() pattern.
        FIX (v14): Use datetime() for format-agnostic timestamp comparison.
        """
        with self.db._connect() as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_executions,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as overall_success_rate,
                    AVG(duration_seconds) as avg_duration,
                    SUM(retry_count) as total_retries,
                    COUNT(DISTINCT workflow_id) as unique_workflows
                FROM metrics
                WHERE timestamp > datetime('now', '-{days} days')
                """
            ).fetchone()

        return {
            "total_executions": row[0],
            "overall_success_rate": f"{(row[1] or 0) * 100:.1f}%",
            "avg_duration_seconds": row[2] or 0,
            "total_retries": row[3] or 0,
            "unique_workflows": row[4] or 0,
            "period_days": days,
        }
```

#### Metrics Dashboard

```python
# supervisor/metrics/dashboard.py
"""Rich-based metrics dashboard."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

from supervisor.metrics.aggregator import MetricsAggregator


class MetricsDashboard:
    """Terminal dashboard for metrics visualization.

    USAGE:
        dashboard = MetricsDashboard(aggregator)
        dashboard.show()  # One-time display
        dashboard.live()  # Live updating display
    """

    def __init__(self, aggregator: MetricsAggregator):
        self.aggregator = aggregator
        self.console = Console()

    def show(self, days: int = 30) -> None:
        """Display metrics dashboard once."""
        self.console.print()
        self.console.rule(f"[bold blue]Supervisor Metrics - Last {days} Days[/bold blue]")

        # Summary stats
        self._show_summary(days)
        self.console.print()

        # Role performance
        self._show_role_performance(days)
        self.console.print()

        # CLI comparison
        self._show_cli_comparison(days)
        self.console.print()

        # Recent failures
        self._show_recent_failures()

    def _show_summary(self, days: int) -> None:
        """Show summary statistics."""
        stats = self.aggregator.get_summary_stats(days)

        table = Table(title="Summary", show_header=False, box=None)
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Total Executions", str(stats["total_executions"]))
        table.add_row("Overall Success Rate", stats["overall_success_rate"])
        table.add_row("Avg Duration", f"{stats['avg_duration_seconds']:.1f}s")
        table.add_row("Total Retries", str(stats["total_retries"]))
        table.add_row("Unique Workflows", str(stats["unique_workflows"]))

        self.console.print(Panel(table))

    def _show_role_performance(self, days: int) -> None:
        """Show role performance table."""
        roles = self.aggregator.get_role_performance(days)

        table = Table(title="Role Performance")
        table.add_column("Role", style="cyan")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg Time", justify="right")
        table.add_column("Avg Retries", justify="right")
        table.add_column("Executions", justify="right")

        for role in roles:
            success_style = "green" if role.success_rate > 0.9 else "yellow" if role.success_rate > 0.7 else "red"
            table.add_row(
                role.role,
                f"[{success_style}]{role.formatted_success_rate}[/]",
                f"{role.avg_duration_seconds:.1f}s",
                f"{role.avg_retries:.1f}",
                str(role.total_executions),
            )

        self.console.print(table)

    def _show_cli_comparison(self, days: int) -> None:
        """Show CLI comparison by task type."""
        stats = self.aggregator.get_cli_comparison(days)

        table = Table(title="CLI Performance by Task Type")
        table.add_column("Task Type", style="bold")
        table.add_column("CLI", style="cyan")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg Time", justify="right")
        table.add_column("Total", justify="right")

        current_task = None
        for stat in stats:
            # Add separator between task types
            if current_task and current_task != stat.task_type:
                table.add_row("", "", "", "", "")
            current_task = stat.task_type

            # Mark best performer
            is_best = stat.success_rate == max(
                s.success_rate for s in stats if s.task_type == stat.task_type
            )
            best_marker = " ✓" if is_best else ""

            success_style = "green" if stat.success_rate > 0.9 else "yellow" if stat.success_rate > 0.7 else "red"

            table.add_row(
                stat.task_type,
                f"{stat.cli}{best_marker}",
                f"[{success_style}]{stat.success_rate * 100:.1f}%[/]",
                f"{stat.avg_duration_seconds:.1f}s",
                str(stat.total_executions),
            )

        self.console.print(table)

    def _show_recent_failures(self, limit: int = 5) -> None:
        """Show recent failures."""
        failures = self.aggregator.get_recent_failures(limit)

        if not failures:
            self.console.print("[dim]No recent failures[/dim]")
            return

        table = Table(title="Recent Failures")
        table.add_column("Time", style="dim")
        table.add_column("Role")
        table.add_column("CLI")
        table.add_column("Workflow")
        table.add_column("Error")

        for f in failures:
            table.add_row(
                f["timestamp"][:19] if f["timestamp"] else "",
                f["role"],
                f["cli"],
                f["workflow_id"][:12] if f["workflow_id"] else "",
                f["error_category"] or "unknown",
            )

        self.console.print(table)
```

**Acceptance Criteria:**
- [ ] Metrics are collected for all role executions
- [ ] Aggregation provides role and CLI performance stats
- [ ] Dashboard displays metrics in readable format
- [ ] Historical data can be queried by time range
- [ ] Best CLI can be determined per task type

---

### 5.4 Adaptive Role Assignment

**Goal:** Extend existing ModelRouter with historical performance-based selection.

**EXISTING IMPLEMENTATION:**
- `routing.py:ModelRouter` - Static model selection with `select_model()` method
- `routing.py:ModelCapability` - Enum for capability types (REASONING, SPEED, CODE_GEN, etc.)
- `routing.py:ModelProfile` - Dataclass with strengths, max_context, relative_speed, relative_cost
- `routing.py:ROLE_MODEL_MAP` - Static role→CLI mapping
- `routing.py:MODEL_PROFILES` - Static CLI→profile mapping

**Files to Create/Modify:**
- `supervisor/core/routing.py` (MODIFY) - Add adaptive methods to existing ModelRouter
- `supervisor/metrics/aggregator.py` (NEW, from 5.3) - Provides historical data
- `supervisor/config/adaptive.yaml` (NEW) - Adaptive configuration

#### Adaptive Router Design

**DEPRECATED (v10/v13):** The standalone `adaptive.py` below is kept as REFERENCE ONLY.
The actual implementation should **EXTEND `supervisor/core/routing.py`** (see "Design: Extend Existing ModelRouter" section).
This avoids creating a parallel system.

```python
# supervisor/core/adaptive.py - DEPRECATED REFERENCE (v13: use routing.py extension instead)
"""Adaptive model routing based on historical performance.

DEPRECATED (v13 - Gemini): This standalone module is for REFERENCE ONLY.
The actual implementation EXTENDS supervisor/core/routing.py to add
adaptive capabilities to the existing ModelRouter class.

Learns from execution history to optimize model selection per task type.
Uses weighted scoring: success_rate * 0.7 + speed_score * 0.2 + cost_score * 0.1
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from supervisor.core.state import Database
from supervisor.metrics.aggregator import MetricsAggregator

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive routing."""
    enabled: bool = True

    # Minimum samples before adapting
    min_samples_before_adapt: int = 10

    # How often to recalculate scores (in executions)
    recalculation_interval: int = 10

    # Exploration vs exploitation
    exploration_rate: float = 0.1  # 10% chance to try non-optimal CLI

    # Scoring weights
    success_weight: float = 0.7
    speed_weight: float = 0.2
    cost_weight: float = 0.1

    # Maximum deviation from default
    max_deviation_from_default: float = 0.2  # Don't stray too far

    # Locked assignments (never adapt these)
    locked_assignments: dict[str, str] = field(default_factory=dict)

    # Lookback period for scoring
    lookback_days: int = 30


@dataclass
class CLIScore:
    """Computed score for a CLI/task combination."""
    cli: str
    task_type: str
    success_rate: float
    avg_duration_seconds: float
    sample_count: int

    # Computed score (0-1)
    score: float

    # Whether this is the recommended choice
    recommended: bool = False


class AdaptiveRouter:
    """Adaptively route tasks to best-performing CLIs.

    DEPRECATED (v10): This standalone class is NOT implemented.
    Instead, use the ModelRouter extension below (see "Design: Extend Existing ModelRouter").
    This section is kept for reference on the algorithm design only.

    ALGORITHM:
    1. Query historical metrics for CLI/task type combinations
    2. Compute weighted score for each CLI
    3. Select highest-scoring CLI (with exploration)
    4. Respect locked assignments and guardrails

    REFERENCE ONLY - Use ModelRouter._select_adaptive() instead.
    """

    def __init__(
        self,
        db: Database,
        config: AdaptiveConfig | None = None,
    ):
        self.db = db
        self.config = config or AdaptiveConfig()
        self.aggregator = MetricsAggregator(db)

        # Cache scores to avoid frequent DB queries
        self._score_cache: dict[str, list[CLIScore]] = {}
        self._execution_count: int = 0

    def select_cli(
        self,
        role_name: str,
        task_type: str,
        context_size: int = 0,
        role_cli: str | None = None,
        default_cli: str = "claude",
    ) -> str:
        """Select best CLI for task based on adaptive scoring.

        Args:
            role_name: Role being executed
            task_type: Type of task (implement, review, plan)
            context_size: Estimated context tokens
            role_cli: CLI specified in role config (takes precedence)
            default_cli: Fallback if no data or disabled

        Returns:
            Selected CLI name
        """
        # Check locked assignments first
        if role_name in self.config.locked_assignments:
            return self.config.locked_assignments[role_name]

        # Role config takes precedence
        if role_cli:
            return role_cli

        # Check if adaptive is enabled
        if not self.config.enabled:
            return default_cli

        # Get scores for task type
        scores = self._get_scores(task_type)

        # Check minimum samples
        total_samples = sum(s.sample_count for s in scores)
        if total_samples < self.config.min_samples_before_adapt:
            logger.debug(
                f"Insufficient samples ({total_samples}) for adaptive routing, "
                f"using default: {default_cli}"
            )
            return default_cli

        # Apply exploration
        import random
        if random.random() < self.config.exploration_rate:
            # Explore: random choice weighted by score
            weights = [max(0.1, s.score) for s in scores]
            selected = random.choices(scores, weights=weights, k=1)[0]
            logger.debug(f"Exploration: selected {selected.cli} for {task_type}")
            return selected.cli

        # Exploit: best score
        best = max(scores, key=lambda s: s.score) if scores else None
        if not best:
            return default_cli

        # Check deviation guardrail
        default_score = next(
            (s for s in scores if s.cli == default_cli),
            None
        )
        if default_score:
            deviation = best.score - default_score.score
            if deviation < self.config.max_deviation_from_default:
                # Not enough improvement to switch
                return default_cli

        logger.info(
            f"Adaptive: selected {best.cli} for {task_type} "
            f"(score: {best.score:.3f}, samples: {best.sample_count})"
        )
        return best.cli

    def _get_scores(self, task_type: str) -> list[CLIScore]:
        """Get CLI scores for task type, with caching."""
        # Increment execution count
        self._execution_count += 1

        # Check if recalculation needed
        if (
            task_type not in self._score_cache or
            self._execution_count % self.config.recalculation_interval == 0
        ):
            self._score_cache[task_type] = self._calculate_scores(task_type)

        return self._score_cache[task_type]

    def _calculate_scores(self, task_type: str) -> list[CLIScore]:
        """Calculate scores from historical data."""
        cli_stats = self.aggregator.get_cli_comparison(
            days=self.config.lookback_days
        )

        # Filter to task type
        task_stats = [s for s in cli_stats if s.task_type == task_type]

        if not task_stats:
            return []

        # Normalize metrics for scoring
        max_duration = max(s.avg_duration_seconds for s in task_stats) or 1.0

        scores = []
        best_score = 0.0

        for stat in task_stats:
            # Compute component scores (0-1)
            success_score = stat.success_rate

            # Invert duration (faster = higher score)
            speed_score = 1.0 - (stat.avg_duration_seconds / max_duration)

            # Cost score (simplified - could use actual cost data)
            cost_scores = {"claude": 0.5, "codex": 0.8, "gemini": 0.7}
            cost_score = cost_scores.get(stat.cli, 0.5)

            # Weighted total
            total_score = (
                success_score * self.config.success_weight +
                speed_score * self.config.speed_weight +
                cost_score * self.config.cost_weight
            )

            best_score = max(best_score, total_score)

            scores.append(CLIScore(
                cli=stat.cli,
                task_type=task_type,
                success_rate=stat.success_rate,
                avg_duration_seconds=stat.avg_duration_seconds,
                sample_count=stat.total_executions,
                score=total_score,
            ))

        # Mark recommended
        for score in scores:
            score.recommended = score.score == best_score

        return scores

    def suggest_improvements(self) -> list[dict[str, Any]]:
        """Analyze metrics and suggest configuration improvements."""
        suggestions = []

        role_stats = self.aggregator.get_role_performance(
            days=self.config.lookback_days
        )

        for role in role_stats:
            # High retry rate
            if role.avg_retries > 2.0:
                suggestions.append({
                    "role": role.role,
                    "issue": f"High retry rate ({role.avg_retries:.1f} per task)",
                    "recommendation": "Consider increasing context budget or simplifying prompts",
                })

            # Low success rate
            if role.success_rate < 0.7:
                suggestions.append({
                    "role": role.role,
                    "issue": f"Low success rate ({role.success_rate * 100:.1f}%)",
                    "recommendation": "Review role configuration and test with different CLIs",
                })

            # Slow execution
            if role.avg_duration_seconds > 180:
                suggestions.append({
                    "role": role.role,
                    "issue": f"Slow execution ({role.avg_duration_seconds:.0f}s avg)",
                    "recommendation": "Consider using faster CLI or reducing context size",
                })

        return suggestions

    def get_routing_report(self) -> dict[str, Any]:
        """Generate report of adaptive routing decisions."""
        report = {
            "enabled": self.config.enabled,
            "exploration_rate": self.config.exploration_rate,
            "locked_assignments": self.config.locked_assignments,
            "scores_by_task": {},
            "suggestions": self.suggest_improvements(),
        }

        # FIX (v11): Use standardized task_type values
        for task_type in ["plan", "implement", "review", "test", "investigate", "document", "other"]:
            scores = self._get_scores(task_type)
            report["scores_by_task"][task_type] = [
                {
                    "cli": s.cli,
                    "score": round(s.score, 3),
                    "success_rate": round(s.success_rate, 3),
                    "samples": s.sample_count,
                    "recommended": s.recommended,
                }
                for s in sorted(scores, key=lambda x: -x.score)
            ]

        return report
```

#### Design: Extend Existing ModelRouter

Rather than creating a separate AdaptiveRouter class, we extend the existing ModelRouter with adaptive capabilities.

```python
# supervisor/core/routing.py - MODIFICATIONS to existing class
# FIX (v13 - Codex): Add `field` import for default_factory
from dataclasses import dataclass, field  # UPDATE existing import


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive routing - NEW dataclass.

    FIX (v10): Consolidated with all fields from adaptive.yaml.
    """
    enabled: bool = True
    min_samples_before_adapt: int = 10
    recalculation_interval: int = 10  # FIX (v10): Added missing field
    exploration_rate: float = 0.1
    max_deviation_from_default: float = 0.2  # FIX (v10): Added missing field
    success_weight: float = 0.7
    speed_weight: float = 0.2
    cost_weight: float = 0.1
    lookback_days: int = 30
    locked_assignments: dict[str, str] = field(default_factory=dict)


class ModelRouter:
    """Route tasks to appropriate models.

    EXISTING: Static routing via ROLE_MODEL_MAP and MODEL_PROFILES
    NEW: Optional adaptive routing based on historical metrics

    ROUTING STRATEGY (UPDATED):
    1. If role specifies CLI, use that (existing)
    2. If adaptive enabled and sufficient data, use learned scores (NEW)
    3. Otherwise, match task type to model strengths (existing)
    4. Consider context size (large context -> Gemini) (existing)
    5. Consider speed requirements (fast -> Codex) (existing)
    """

    def __init__(
        self,
        prefer_speed: bool = False,
        prefer_cost: bool = False,
        # NEW Phase 5 params
        db: "Database | None" = None,
        adaptive_config: AdaptiveConfig | None = None,
    ):
        self.prefer_speed = prefer_speed
        self.prefer_cost = prefer_cost

        # NEW: Adaptive routing support
        self.db = db
        self.adaptive_config = adaptive_config
        self._score_cache: dict[str, list[tuple[str, float]]] = {}
        self._execution_count = 0

    def select_model(
        self,
        role_name: str,
        context_size: int = 0,
        task_type: ModelCapability | None = None,
        role_cli: str | None = None,
    ) -> str:
        """Select the best model for a task.

        EXISTING behavior preserved, NEW adaptive check inserted.
        """
        # Role config takes precedence (EXISTING)
        if role_cli:
            return role_cli

        # NEW: Check locked assignments
        if (
            self.adaptive_config
            and role_name in self.adaptive_config.locked_assignments
        ):
            return self.adaptive_config.locked_assignments[role_name]

        # NEW: Try adaptive routing if enabled
        if self._should_use_adaptive(role_name):
            adaptive_result = self._select_adaptive(role_name, task_type)
            if adaptive_result:
                return adaptive_result

        # Default mapping (EXISTING)
        if role_name in ROLE_MODEL_MAP:
            default = ROLE_MODEL_MAP[role_name]
        else:
            default = "claude"

        # Check context size requirements (EXISTING)
        if context_size > 128000:
            return "gemini"

        # Speed preference (EXISTING)
        if self.prefer_speed and task_type in [
            ModelCapability.CODE_GEN,
            ModelCapability.SPEED,
        ]:
            return "codex"

        # Cost preference (EXISTING - already generic from PR review fix)
        if self.prefer_cost and task_type:
            capable_models = [
                (cli, profile)
                for cli, profile in MODEL_PROFILES.items()
                if task_type in profile.strengths
            ]
            if capable_models:
                capable_models.sort(key=lambda x: x[1].relative_cost)
                return capable_models[0][0]

        return default

    # --- NEW ADAPTIVE METHODS ---

    def _should_use_adaptive(self, role_name: str) -> bool:
        """Check if adaptive routing should be used."""
        if not self.adaptive_config or not self.adaptive_config.enabled:
            return False
        if not self.db:
            return False
        return True

    def _select_adaptive(
        self,
        role_name: str,
        task_type: ModelCapability | None,
    ) -> str | None:
        """Select CLI using adaptive scoring from metrics.

        Returns None if insufficient data, letting static routing handle it.
        """
        from supervisor.metrics.aggregator import MetricsAggregator

        # Infer task type from role name if not provided
        inferred_type = self._infer_task_type(role_name)

        # Query historical performance
        aggregator = MetricsAggregator(self.db)
        cli_stats = aggregator.get_cli_comparison(
            days=self.adaptive_config.lookback_days
        )

        # Filter to task type
        task_stats = [s for s in cli_stats if s.task_type == inferred_type]

        # Check minimum samples
        total_samples = sum(s.total_executions for s in task_stats)
        if total_samples < self.adaptive_config.min_samples_before_adapt:
            return None  # Insufficient data

        # Calculate scores
        scores = self._calculate_scores(task_stats)
        if not scores:
            return None

        # Exploration vs exploitation
        import random
        if random.random() < self.adaptive_config.exploration_rate:
            # Explore: weighted random
            weights = [max(0.1, score) for _, score in scores]
            selected = random.choices(scores, weights=weights, k=1)[0]
            return selected[0]

        # Exploit: best score
        best_cli, best_score = max(scores, key=lambda x: x[1])
        logger.info(f"Adaptive: {best_cli} for {role_name} (score: {best_score:.3f})")
        return best_cli

    def _calculate_scores(
        self,
        stats: list["CLIPerformance"],
    ) -> list[tuple[str, float]]:
        """Calculate weighted scores for CLIs."""
        if not stats:
            return []

        max_duration = max(s.avg_duration_seconds for s in stats) or 1.0
        cost_map = {"claude": 1.0, "codex": 0.5, "gemini": 0.7}

        scores = []
        for stat in stats:
            success_score = stat.success_rate
            speed_score = 1.0 - (stat.avg_duration_seconds / max_duration)
            cost_score = 1.0 - cost_map.get(stat.cli, 0.5)

            total = (
                success_score * self.adaptive_config.success_weight +
                speed_score * self.adaptive_config.speed_weight +
                cost_score * self.adaptive_config.cost_weight
            )
            scores.append((stat.cli, total))

        return scores

    def _infer_task_type(self, role_name: str) -> str:
        """Infer task type from role name.

        FIX (v11): Standardized task_type values across schema, engine, collector.
        Values: plan, implement, review, test, investigate, document, other
        """
        role_lower = role_name.lower()
        if "plan" in role_lower:
            return "plan"
        elif "review" in role_lower:
            return "review"
        elif "implement" in role_lower:
            return "implement"
        elif "test" in role_lower:
            return "test"
        elif "investigat" in role_lower:
            return "investigate"
        elif "doc" in role_lower:
            return "document"
        return "other"
```

#### Factory and WorkflowCoordinator Wiring

**CRITICAL FIX (Codex/Gemini review):** Update `create_router` factory and `WorkflowCoordinator` to pass `db`.

```python
# supervisor/core/routing.py - UPDATE create_router factory

def create_router(
    prefer_speed: bool = False,
    prefer_cost: bool = False,
    db: "Database | None" = None,  # NEW param
    adaptive_config: AdaptiveConfig | None = None,  # NEW param
) -> ModelRouter:
    """Factory function to create a configured model router.

    FIX (Gemini review): Accept db and adaptive_config for adaptive routing.
    """
    return ModelRouter(
        prefer_speed=prefer_speed,
        prefer_cost=prefer_cost,
        db=db,
        adaptive_config=adaptive_config,
    )
```

```python
# supervisor/core/workflow.py - UPDATE WorkflowCoordinator.__init__
# FIX (v8): Added role_timeouts param (was missing in v7)

class WorkflowCoordinator:
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
        # Phase 5 NEW params:
        workflow_timeout: float = 3600.0,
        checkpoint_on_timeout: bool = True,
        role_timeouts: dict[str, float] | None = None,  # FIX (v8): Added
        adaptive_config: AdaptiveConfig | None = None,
    ):
        # ... existing init ...
        self.role_timeouts = role_timeouts or {}  # FIX (v8): Per-role timeout overrides

        # FIX (Gemini review): Pass db to create_router for adaptive routing
        self._router = create_router(
            prefer_speed=prefer_speed,
            prefer_cost=prefer_cost,
            db=db,  # NEW: Pass db for metrics access
            adaptive_config=adaptive_config,  # NEW: Pass adaptive config
        )
```

#### Key Changes from Original Plan

| Original Plan | Updated Plan |
|--------------|--------------|
| New `supervisor/core/adaptive.py` | Extend `supervisor/core/routing.py` |
| New `AdaptiveRouter` class | Add methods to existing `ModelRouter` |
| Separate routing system | Integrated into existing `select_model()` |
| Import AdaptiveRouter | Inline adaptive logic in ModelRouter |
| Factory unchanged | `create_router` updated with `db` param |
| Wiring unchanged | `WorkflowCoordinator` passes `db` to router |

**Acceptance Criteria:**
- [ ] Adaptive routing uses historical metrics
- [ ] Exploration/exploitation balance is configurable
- [ ] Locked assignments are respected
- [ ] Guardrails prevent wild deviations
- [ ] `create_router` factory accepts `db` parameter
- [ ] `WorkflowCoordinator` passes `db` to router

---

## Integration Points

### Dependencies Between Deliverables

```
5.1 Timeout Handling (extends WorkflowCoordinator)
      │
      ├──► 5.2 Human Interrupt TUI (timeout can trigger approval request)
      │
      └──► 5.3 Metrics (timeout events feed metrics)

5.3 Metrics Dashboard (new metrics table + aggregator)
      │
      └──► 5.4 Adaptive Routing (metrics drive model selection)
```

### Building on Existing Code

| Phase 5 Component | Existing Code | Integration Method |
|-------------------|--------------|-------------------|
| Workflow Timeout | `workflow.py:WorkflowCoordinator` | Add `workflow_timeout` param, use existing `CancellationError` |
| Timeout in Parallel | `workflow.py:_run_continuous_parallel` | Add `workflow_deadline` param, check in main loop |
| Checkpoint | `state.py:create_checkpoint` | Use existing `create_checkpoint(workflow_id, step_id, git_sha, context: dict, status)` |
| TUI Events | `state.py:EventType.APPROVAL_*` | Use existing APPROVAL_REQUESTED/GRANTED/DENIED |
| TUI Sync/Async Bridge | NEW `InteractionBridge` | `queue.Queue` + `threading.Event` for thread-safe comms |
| TUI Status Query | `state.py:get_phases/get_components` | Use existing DB query methods |
| Metrics Storage | `state.py` SQLite schema | ALREADY EXISTS: `metrics` table + `record_metric()` (v20) |
| Metrics Collection | `engine.py:run_role` | Add timing instrumentation in finally block |
| Adaptive Routing | `routing.py:ModelRouter` | Extend `select_model()` with `_select_adaptive()` |
| Router Factory | `routing.py:create_router` | Add `db` and `adaptive_config` params |
| Coordinator Wiring | `workflow.py:WorkflowCoordinator` | Pass `db` to `create_router()` |

### New Dependencies

```toml
# Add to pyproject.toml
dependencies = [
    # ... existing ...
    "rich>=13.0.0",  # TUI and dashboard (already used in cli.py)
]
```

### Configuration Loading (FIX Codex v3)

The config files need a loader and wiring plan:

```python
# supervisor/config/loader.py
"""Configuration loading for Phase 5 features.

Loads limits.yaml, adaptive.yaml, approval.yaml and wires to components.
FIX (Codex v4): Handle None from yaml.safe_load and include all config keys.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from supervisor.core.routing import AdaptiveConfig
from supervisor.core.approval import ApprovalPolicy

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path(__file__).parent


# FIX (v23 - Codex): CONSOLIDATED - load_config is defined above in the
# "FIX (v20 - Codex): Config Integration Details" section (line ~896).
# That implementation has correct precedence: CLI > .supervisor/ > package defaults.
# See that section for the canonical implementation.
#
# from supervisor.config.loader import load_config  # Use the canonical version


def get_adaptive_config(config: dict[str, Any]) -> AdaptiveConfig:
    """Extract AdaptiveConfig from loaded config.

    FIX (Codex v4): Guard against None with 'or {}'.
    FIX (v10): Include ALL config fields to match AdaptiveConfig dataclass.
    """
    adaptive = config.get("adaptive") or {}  # FIX: Handle None
    return AdaptiveConfig(
        enabled=adaptive.get("enabled", True),
        min_samples_before_adapt=adaptive.get("min_samples_before_adapt", 10),
        recalculation_interval=adaptive.get("recalculation_interval", 10),  # FIX (v10)
        exploration_rate=adaptive.get("exploration_rate", 0.1),
        max_deviation_from_default=adaptive.get("max_deviation_from_default", 0.2),  # FIX (v10)
        success_weight=adaptive.get("success_weight", 0.7),
        speed_weight=adaptive.get("speed_weight", 0.2),
        cost_weight=adaptive.get("cost_weight", 0.1),
        lookback_days=adaptive.get("lookback_days", 30),
        locked_assignments=adaptive.get("locked_assignments") or {},  # FIX
    )


def get_timeout_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract timeout settings from limits config.

    FIX (v7): Removed phase_timeout - not supported by DAGScheduler.
    """
    limits = config.get("limits") or {}  # FIX: Handle None
    timeout = limits.get("timeout") or {}  # FIX: Handle None

    return {
        # Core timeouts (v7: removed phase_timeout)
        "workflow_timeout": timeout.get("workflow_timeout_seconds", 3600.0),
        "component_timeout": timeout.get("step_timeout_seconds", 300.0),
        "max_stall_seconds": timeout.get("max_stall_seconds", 600.0),
        # Additional settings
        "grace_period_seconds": timeout.get("grace_period_seconds", 30.0),
        "checkpoint_on_timeout": timeout.get("checkpoint_on_timeout", True),
        "role_timeouts": timeout.get("role_timeouts") or {},  # Per-role timeout overrides
    }


def get_approval_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract approval policy settings from config.

    FIX (v13 - Codex): Approval policy was loaded but never wired.
    This function extracts settings from approval.yaml config.
    """
    approval = config.get("approval") or {}

    return {
        "auto_approve_low_risk": approval.get("auto_approve_low_risk", True),
        "risk_threshold": approval.get("risk_threshold", "medium"),
        "require_approval_for": approval.get("require_approval_for", ["deploy", "commit"]),
    }
```

```python
# supervisor/cli.py - NEW workflow command (FIX v13: don't conflict with existing `run` command)
#
# IMPORTANT: Existing CLI uses `main` as group (not `cli`) and `run` for single-role execution.
# This adds a NEW `workflow` command to avoid breaking existing CLI contract.

from supervisor.config.loader import load_config, get_adaptive_config, get_timeout_config, get_approval_config


@main.command()
@click.option("--tui", "enable_tui", is_flag=True, default=False, help="Enable interactive TUI mode")
@click.option("--config-dir", type=click.Path(exists=True), default=None, help="Config directory")
@click.argument("feature_id")
def workflow(enable_tui: bool, config_dir: str | None, feature_id: str):
    """Execute a feature workflow with Phase 5 capabilities.

    FEATURE_ID is the feature to execute (e.g., "feat-auth-login").

    This is a NEW command (FIX v13) that doesn't conflict with existing
    `supervisor run <role> <task>` command for single-role execution.

    Example:
        supervisor workflow feat-auth-login
        supervisor workflow --tui feat-auth-login
    """
    config = load_config(Path(config_dir) if config_dir else None)
    repo_path = get_repo_path()

    # Initialize database and engine
    # FIX (v17): Use state.db to match existing cli.py init and other commands
    db = Database(config.get("database_path", str(repo_path / ".supervisor" / "state.db")))
    engine = ExecutionEngine(repo_path, db=db)

    # Create coordinator with loaded config
    timeout_cfg = get_timeout_config(config)
    adaptive_cfg = get_adaptive_config(config)

    # FIX (v21 - Codex): Always create approval gate (CLI fallback if no TUI)
    # Bridge is only created for TUI mode
    from supervisor.core.approval import ApprovalGate, ApprovalPolicy

    approval_cfg = get_approval_config(config)
    approval_policy = ApprovalPolicy(
        auto_approve_low_risk=approval_cfg.get("auto_approve_low_risk", True),
        risk_threshold=approval_cfg.get("risk_threshold", "medium"),
        require_approval_for=approval_cfg.get("require_approval_for", ["deploy", "commit"]),
    )
    approval_gate = ApprovalGate(db, policy=approval_policy)

    # Bridge is only needed for TUI mode
    interaction_bridge = None
    if enable_tui:
        from supervisor.core.interaction import InteractionBridge
        interaction_bridge = InteractionBridge()

    # FIX (v21 - Codex): Check feature exists before running implementation
    # This provides clear UX when feature hasn't been created/planned yet
    feature = db.get_feature(feature_id)
    if not feature:
        console.print(f"[red]Error: Feature '{feature_id}' not found.[/red]")
        console.print("\n[yellow]To create a feature, use:[/yellow]")
        console.print(f"  supervisor plan \"<task description>\" -w {feature_id}")
        console.print("\n[yellow]This will:[/yellow]")
        console.print("  1. Create the feature in the database")
        console.print("  2. Generate phases and components")
        console.print("  3. Prepare the workflow for implementation")
        sys.exit(1)

    # Show feature status
    console.print(f"[bold]Feature:[/bold] {feature.title or feature_id}")
    console.print(f"[bold]Status:[/bold] {feature.status.value}")

    coordinator = WorkflowCoordinator(
        engine=engine,
        db=db,
        # FIX (v7): Removed phase_timeout - not supported by DAGScheduler
        workflow_timeout=timeout_cfg["workflow_timeout"],
        component_timeout=timeout_cfg["component_timeout"],
        max_stall_seconds=timeout_cfg["max_stall_seconds"],
        checkpoint_on_timeout=timeout_cfg["checkpoint_on_timeout"],
        role_timeouts=timeout_cfg["role_timeouts"],  # Per-role timeout overrides
        adaptive_config=adaptive_cfg,
        # FIX (v11/v21): Pass approval gate and bridge for human-in-the-loop
        # FIX (v21): approval_gate enabled even without TUI for CLI fallback
        approval_gate=approval_gate,
        interaction_bridge=interaction_bridge,
    )

    # FIX (v12): Start TUI with shared bridge if enabled
    if enable_tui:
        from supervisor.tui.app import SupervisorTUI

        # Pass SAME bridge to TUI that coordinator uses
        tui = SupervisorTUI(db, bridge=interaction_bridge)
        tui.run_with_workflow(
            workflow_fn=lambda: coordinator.run_implementation(feature_id),
            feature_id=feature_id,
        )
    else:
        # Non-interactive mode - run directly
        coordinator.run_implementation(feature_id)
```

### ExecutionEngine Factory Methods (FIX v12)

The existing `ExecutionEngine.create_workflow_coordinator()` and `ExecutionEngine.execute_feature()` methods need to be updated to pass Phase 5 parameters:

```python
# supervisor/core/engine.py - Updates to factory methods

def create_workflow_coordinator(
    self,
    max_parallel_workers: int = 3,
    prefer_speed: bool = False,
    prefer_cost: bool = False,
    max_stall_seconds: float = 600.0,
    component_timeout: float = 300.0,
    # NEW Phase 5 parameters
    workflow_timeout: float | None = None,
    role_timeouts: dict[str, float] | None = None,
    checkpoint_on_timeout: bool = True,
    adaptive_config: "AdaptiveConfig | None" = None,
    approval_gate: "ApprovalGate | None" = None,
    interaction_bridge: "InteractionBridge | None" = None,
) -> "WorkflowCoordinator":
    """Create a WorkflowCoordinator for multi-model hierarchical workflows.

    Phase 4 integration point for Feature->Phase->Component execution.
    FIX (v12): Extended with Phase 5 timeout, adaptive, and approval params.
    FIX (v16): Normalize workflow_timeout to avoid None TypeError.
    """
    from supervisor.core.workflow import WorkflowCoordinator

    return WorkflowCoordinator(
        engine=self,
        db=self.db,
        repo_path=self.repo_path,
        max_parallel_workers=max_parallel_workers,
        prefer_speed=prefer_speed,
        prefer_cost=prefer_cost,
        max_stall_seconds=max_stall_seconds,
        component_timeout=component_timeout,
        # Phase 5 parameters - FIX (v16): Normalize None to default
        workflow_timeout=workflow_timeout or 3600.0,  # Default 1 hour if None
        role_timeouts=role_timeouts,
        checkpoint_on_timeout=checkpoint_on_timeout,
        adaptive_config=adaptive_config,
        approval_gate=approval_gate,
        interaction_bridge=interaction_bridge,
    )


def execute_feature(
    self,
    feature_id: str,
    parallel: bool = True,
    max_parallel_workers: int = 3,
    # ... existing params ...
    # NEW Phase 5 parameters
    workflow_timeout: float | None = None,
    role_timeouts: dict[str, float] | None = None,
    approval_gate: "ApprovalGate | None" = None,
    interaction_bridge: "InteractionBridge | None" = None,
) -> "Feature":
    """Execute all components of a feature in dependency order.

    FIX (v12): Extended signature to support Phase 5 features.
    """
    coordinator = self.create_workflow_coordinator(
        max_parallel_workers=max_parallel_workers,
        # ... pass through Phase 5 params ...
        workflow_timeout=workflow_timeout,
        role_timeouts=role_timeouts,
        approval_gate=approval_gate,
        interaction_bridge=interaction_bridge,
    )
    return coordinator.run_implementation(feature_id)
```

### ApprovalGate Integration (FIX Codex v3)

ApprovalGate needs to be called from WorkflowCoordinator or GateExecutor:

```python
# supervisor/core/workflow.py - Integration point for ApprovalGate
# FIX (v14 - Codex): Added import for ApprovalDecision
# FIX (v25 - Codex): Added datetime import for SKIP persistence
from datetime import datetime
from supervisor.core.interaction import ApprovalDecision


class WorkflowCoordinator:
    def __init__(
        self,
        # ... existing params ...
        approval_gate: "ApprovalGate | None" = None,  # NEW
        interaction_bridge: "InteractionBridge | None" = None,  # NEW
    ):
        self.approval_gate = approval_gate
        self.interaction_bridge = interaction_bridge

    def _check_approval_gate(
        self,
        feature_id: str,
        component: Component,
        changed_files: list[str],
        diff_lines: list[str] | None = None,
        untracked_files: list[str] | None = None,
    ) -> bool:
        """Check if approval gate should be invoked and get decision.

        FIX (v17 - Codex): Explicit semantics for all ApprovalDecision values.

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
        if not self.approval_gate:
            return True  # No gate configured

        # FIX (v23 - Codex): Add operation to context for require_approval_for policy
        # Operation is derived from component's assigned_role or defaults to "implement"
        operation = component.assigned_role or "implement"
        # Map common operations: "committer" -> "commit", "deployer" -> "deploy"
        if "commit" in operation.lower():
            operation = "commit"
        elif "deploy" in operation.lower():
            operation = "deploy"

        # FIX (v25 - Codex): Pass file list for risk assessment, not diff lines
        # This fixes the issue where file_count and pattern checks were broken
        context = {
            "changes": changed_files,  # File paths for risk scoring
            "component": component.id,
            "operation": operation,  # FIX (v23): Added for require_approval_for policy
        }
        if not self.approval_gate.requires_approval(context):
            return True  # Low risk / excluded operation, auto-approve

        # FIX (v25 - Codex): Build review summary with untracked file warning
        review_summary = "Review required before proceeding"
        if untracked_files:
            review_summary += f"\n\nNote: {len(untracked_files)} new file(s) will be created: {', '.join(untracked_files)}"

        # FIX (v21 - Codex): Request approval via bridge OR CLI fallback
        # If bridge is None, ApprovalGate uses _cli_approval_sync() fallback
        # FIX (v25): Pass changed_files for context and diff_lines for display
        decision = self.approval_gate.request_approval(
            feature_id=feature_id,
            title=f"Approve {component.title}",
            changes=changed_files,  # File paths for ApprovalRequest.changes
            diff_lines=diff_lines,  # FIX (v25): Actual diff for display
            review_summary=review_summary,
            component_id=component.id,
            bridge=self.interaction_bridge,  # May be None for non-TUI mode
        )

        # FIX (v22 - Codex): Use is_proceed() semantic helper for clear decision handling
        # This explicitly handles APPROVE/SKIP (proceed) vs REJECT/EDIT (block)
        if decision.is_proceed():
            if decision == ApprovalDecision.SKIP:
                logger.warning(f"Component '{component.id}' skipped approval - flagged for review")
                # FIX (v24 - Codex): Persist SKIP flag in component metadata for later review
                # This allows post-workflow audits to identify components that bypassed approval
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
```

**FIX (v24 - Codex): Required DAGScheduler extension for SKIP persistence**

The `set_component_metadata` method must be added to DAGScheduler:

```python
# supervisor/core/dag_scheduler.py - REQUIRED extension for SKIP persistence

def set_component_metadata(
    self,
    component_id: str,
    key: str,
    value: Any,
    workflow_id: str,
) -> None:
    """Store arbitrary metadata on a component.

    Used by ApprovalGate to persist SKIP flags for later audit.
    Metadata is stored as JSON in the component's metadata field.
    """
    with self.db._connect() as conn:
        # Read existing metadata
        row = conn.execute(
            "SELECT metadata FROM components WHERE id = ? AND workflow_id = ?",
            (component_id, workflow_id),
        ).fetchone()

        if row:
            import json
            metadata = json.loads(row[0]) if row[0] else {}
            metadata[key] = value
            conn.execute(
                "UPDATE components SET metadata = ? WHERE id = ? AND workflow_id = ?",
                (json.dumps(metadata), component_id, workflow_id),
            )
```

---

## Testing Strategy

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_workflow_timeout.py` | WorkflowCoordinator timeout extensions |
| `tests/test_tui.py` | TUI rendering, InteractionBridge, approval flow |
| `tests/test_metrics.py` | Collection via Engine instrumentation, aggregation |
| `tests/test_adaptive.py` | ModelRouter adaptive methods, scoring |

**FIX (Codex review):** Test files updated to match v3 design (no TimeoutManager/CancellationToken).

### Integration Tests

| Test | Description |
|------|-------------|
| `test_workflow_timeout` | run_implementation with workflow_timeout |
| `test_approval_gate_bridge` | ApprovalGate + InteractionBridge sync flow |
| `test_metrics_from_engine` | Engine.run_role → db.record_metric |
| `test_adaptive_via_modelrouter` | ModelRouter._select_adaptive with metrics |

### Example Test Cases

```python
# tests/test_workflow_timeout.py
def test_workflow_timeout_triggers_cancellation():
    """Workflow timeout should raise CancellationError.

    FIX (Codex review): Uses CancellationError, not TimeoutError.
    Tests WorkflowCoordinator, not separate TimeoutManager.
    """
    from supervisor.core.engine import CancellationError

    # Very short timeout for testing
    coordinator = WorkflowCoordinator(
        engine=mock_engine,
        db=mock_db,
        workflow_timeout=0.1,  # 100ms
    )

    with pytest.raises(CancellationError, match="Workflow timeout"):
        # This should timeout
        coordinator.run_implementation(feature_id)


def test_checkpoint_saved_on_timeout():
    """Checkpoint should be saved when timeout occurs."""
    coordinator = WorkflowCoordinator(
        engine=mock_engine,
        db=mock_db,
        workflow_timeout=0.1,
        checkpoint_on_timeout=True,
    )

    try:
        coordinator.run_implementation(feature_id)
    except CancellationError:
        pass

    # Check checkpoint was created
    checkpoint = mock_db.get_latest_checkpoint(feature_id)
    assert checkpoint is not None
    assert "timeout" in checkpoint["git_sha"]


# tests/test_tui.py
def test_interaction_bridge_sync():
    """InteractionBridge should block workflow until TUI responds."""
    bridge = InteractionBridge()
    request = ApprovalRequest(
        gate_id="test-gate",
        feature_id="F-123",
        # ... other fields
    )

    # Submit from workflow thread (would block)
    def workflow_thread():
        return bridge.request_approval(request, timeout=5.0)

    import threading
    result = [None]
    t = threading.Thread(target=lambda: result.__setitem__(0, workflow_thread()))
    t.start()

    # Simulate TUI submitting decision
    time.sleep(0.1)
    pending = bridge.get_pending_requests()
    assert len(pending) == 1
    bridge.submit_decision("test-gate", ApprovalDecision.APPROVE)

    t.join(timeout=1.0)
    assert result[0] == ApprovalDecision.APPROVE


# tests/test_adaptive.py
def test_modelrouter_adaptive_select():
    """ModelRouter._select_adaptive should use metrics.

    FIX (Codex review): Tests ModelRouter, not separate AdaptiveRouter.
    """
    db = mock_db_with_metrics([
        {"cli": "claude", "task_type": "implement", "success_rate": 0.8},
        {"cli": "codex", "task_type": "implement", "success_rate": 0.92},
    ])

    router = ModelRouter(db=db, adaptive_config=AdaptiveConfig(enabled=True))
    selected = router.select_model("implementer", task_type=ModelCapability.CODE_GEN)

    assert selected == "codex"  # Higher success rate


def test_modelrouter_respects_locked_assignments():
    """Locked assignments should override adaptive selection."""
    config = AdaptiveConfig(locked_assignments={"planner": "claude"})
    router = ModelRouter(db=mock_db, adaptive_config=config)

    selected = router.select_model("planner")
    assert selected == "claude"
```

---

## Security Considerations

### TUI Security

```python
# User input is never executed directly
# Rich library handles terminal escape sequences safely
# Approval decisions are logged for audit trail
```

### Metrics Security

```python
# No PII stored in metrics
# Workflow IDs are internal references
# Error messages are sanitized before storage
```

### Timeout Security

```python
# Graceful shutdown prevents data corruption
# Checkpoints enable safe resume
# Cancellation is cooperative, not forceful
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TUI compatibility issues | Medium | Low | Use Rich library (cross-platform) |
| Metrics table grows large | Medium | Medium | Add index, implement retention policy |
| Adaptive routing oscillates | Low | Medium | Exploration rate dampening, guardrails |
| Timeout checkpoint corruption | Low | High | Atomic writes, validation on resume |
| Memory pressure from metrics | Low | Medium | Bounded cache, periodic flush |

---

## CLI Commands (Preview)

```bash
# Start interactive TUI
supervisor workflow --tui <feature-id>

# View metrics dashboard
supervisor metrics --days 30

# View specific role metrics
supervisor metrics --role implementer

# Show adaptive routing report
supervisor metrics --adaptive

# Configure timeouts
supervisor config set timeout.workflow 7200
supervisor config set timeout.step 600

# Lock specific role to CLI
supervisor config set adaptive.locked.planner claude

# Enable/disable adaptive routing
supervisor config set adaptive.enabled true
```

---

## Completion Criteria

Phase 5 is complete when:

1. [ ] Workflow-level timeouts work with graceful degradation
2. [ ] TUI displays status and handles approvals
3. [ ] Metrics are collected and displayed in dashboard
4. [ ] Adaptive routing improves with usage history
5. [ ] All unit tests pass
6. [ ] Integration tests demonstrate end-to-end flow
7. [ ] Documentation updated with new commands
8. [ ] No regressions in Phases 1-4 functionality

---

## Configuration Reference

### Timeout Configuration (limits.yaml)

```yaml
# config/limits.yaml additions (v7: removed phase_timeout_seconds)
timeout:
  workflow_timeout_seconds: 3600  # 1 hour total for entire feature
  step_timeout_seconds: 300       # 5 min per component (default)
  max_stall_seconds: 600          # 10 min max without progress
  grace_period_seconds: 30        # Cleanup grace period
  checkpoint_on_timeout: true     # Save state on timeout

  # Per-role timeout overrides (v7: wired to _execute_component)
  role_timeouts:
    planner: 600       # 10 min for planning
    investigator: 900  # 15 min for investigation
    reviewer: 180      # 3 min for reviews (faster)
```

### Adaptive Configuration (adaptive.yaml)

```yaml
# config/adaptive.yaml
adaptive:
  enabled: true
  min_samples_before_adapt: 10
  recalculation_interval: 10
  exploration_rate: 0.1
  max_deviation_from_default: 0.2
  lookback_days: 30

  # Scoring weights
  success_weight: 0.7
  speed_weight: 0.2
  cost_weight: 0.1

  # Never change these
  locked_assignments:
    planner: claude  # Always Claude for planning
```

### Approval Policy Configuration (approval.yaml)

FIX (v15 - Codex): Simplified schema to match `get_approval_config()` loader and `ApprovalPolicy` dataclass.
The complex `approval_gates` structure was removed in favor of a flat, simpler schema.

```yaml
# config/approval.yaml
# FIX (v15): Simplified schema matching ApprovalPolicy dataclass

approval:
  # Auto-approve low-risk changes without user prompt
  auto_approve_low_risk: true

  # Minimum risk level that triggers approval requirement
  # Options: "low", "medium", "high", "critical"
  risk_threshold: medium

  # Operations that ALWAYS require approval regardless of risk level
  require_approval_for:
    - deploy
    - commit
    - production_config_change
```

---

## Open Questions

1. Should metrics have a retention policy? (e.g., delete after 90 days)
2. Should TUI support remote connections? (SSH, tmux)
3. Should adaptive routing consider cost from actual billing data?
