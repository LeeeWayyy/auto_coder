"""
Execution engine for the Supervisor orchestrator.

Coordinates:
- Context packing
- Worker invocation (via sandbox)
- Output parsing
- Gate verification
- State updates
"""

import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from pydantic import BaseModel

from supervisor.core.context import ContextPacker
from supervisor.core.feedback import StructuredFeedbackGenerator
from supervisor.core.gates import (
    GateExecutor,
    GateFailAction,
    GateLoader,
    GateNotFoundError,
    GateResult,
    GateSeverity,
    GateStatus,
)
from supervisor.core.models import ErrorAction, ErrorCategory, Step, StepStatus
from supervisor.core.parser import (
    GenericOutput,
    InvalidOutputError,
    ParsingError,
    ROLE_SCHEMAS,
    get_adapter,
    parse_role_output,
)
from supervisor.core.roles import RoleConfig, RoleLoader
from supervisor.core.routing import _infer_task_type
from supervisor.core.state import Database, Event, EventType
from supervisor.core.workspace import ApplyError, GateFailedError, IsolatedWorkspace, _truncate_output
from supervisor.sandbox.executor import (
    DockerNotAvailableError,
    ExecutionResult,
    SandboxConfig,
    SandboxedExecutor,
    SandboxedLLMClient,
    get_sandboxed_executor,
    require_docker,
)


class EngineError(Exception):
    """Error in execution engine."""

    pass


class RetryExhaustedError(Exception):
    """All retry attempts exhausted."""

    pass


class CircuitOpenError(Exception):
    """Circuit breaker is open, refusing to execute."""

    pass


# FIX (v27 - Gemini PR review): Import from models to avoid circular imports
# Re-export for backwards compatibility
from supervisor.core.models import CancellationError  # noqa: F401


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 2.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.1

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt (0-indexed)."""
        import random

        delay = min(
            self.initial_delay * (self.backoff_multiplier**attempt),
            self.max_delay,
        )
        jitter = random.uniform(-self.jitter * delay, self.jitter * delay)
        return delay + jitter


@dataclass
class CircuitBreaker:
    """Prevent runaway error loops.

    Thread-safe implementation using a lock to protect _failures dict.
    Uses bounded LRU-style eviction to prevent unbounded memory growth.

    Memory bounds:
    - max_keys: Maximum number of tracked keys (default 1000)
    - _cleanup_interval: How often to run cleanup (every N calls)

    When max_keys is exceeded, oldest keys (by last failure time) are evicted.
    """

    max_failures: int = 5
    reset_timeout: int = 300  # seconds
    max_keys: int = 1000  # Maximum tracked keys to bound memory
    _cleanup_interval: int = 100  # Cleanup every N record_failure calls
    _failures: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _call_count: int = field(default=0)

    def _cleanup_stale_keys(self, now: float) -> None:
        """Remove stale keys and enforce max_keys limit (call within lock).

        Uses LRU-style eviction: removes keys with oldest last-failure time first.
        """
        # First pass: remove keys with no recent failures
        stale_keys = [
            key for key, timestamps in self._failures.items()
            if not timestamps or all(now - t >= self.reset_timeout for t in timestamps)
        ]
        for key in stale_keys:
            del self._failures[key]

        # Second pass: if still over max_keys, evict oldest by last failure time
        if len(self._failures) > self.max_keys:
            # Sort by most recent failure timestamp (ascending = oldest first)
            sorted_keys = sorted(
                self._failures.keys(),
                key=lambda k: max(self._failures[k]) if self._failures[k] else 0.0
            )
            # Evict oldest keys until under limit
            evict_count = len(self._failures) - self.max_keys
            for key in sorted_keys[:evict_count]:
                del self._failures[key]

    def record_failure(self, step_id: str) -> None:
        now = time.time()
        with self._lock:
            if step_id not in self._failures:
                self._failures[step_id] = []

            # Clean old failures outside reset window
            self._failures[step_id] = [
                t for t in self._failures[step_id] if now - t < self.reset_timeout
            ]
            self._failures[step_id].append(now)

            # Periodically cleanup stale keys and enforce max_keys
            self._call_count += 1
            if self._call_count >= self._cleanup_interval:
                self._cleanup_stale_keys(now)
                self._call_count = 0

    def is_open(self, step_id: str) -> bool:
        """Returns True if circuit is open (should not retry)."""
        with self._lock:
            if step_id not in self._failures:
                return False
            # Clean old failures before checking
            now = time.time()
            self._failures[step_id] = [
                t for t in self._failures[step_id] if now - t < self.reset_timeout
            ]
            # Remove key if empty to prevent memory growth
            if not self._failures[step_id]:
                del self._failures[step_id]
                return False
            return len(self._failures[step_id]) >= self.max_failures

    def reset(self, step_id: str) -> None:
        """Reset circuit breaker for a step."""
        with self._lock:
            self._failures.pop(step_id, None)


class CircuitState(str, Enum):
    """Circuit breaker state."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker behavior."""

    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    current_failures: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    state_changes: int = 0


class EnhancedCircuitBreaker:
    """Enhanced circuit breaker with half-open recovery and optional persistence."""

    def __init__(
        self,
        max_failures: int = 5,
        reset_timeout: int = 300,
        half_open_timeout: int = 60,
        db: Database | None = None,
    ) -> None:
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout  # Timeout for stuck HALF_OPEN state
        self.db = db
        self._lock = threading.RLock()
        self._states: dict[str, CircuitState] = {}
        self._metrics: dict[str, CircuitBreakerMetrics] = {}
        self._half_open_used: dict[str, int] = {}
        self._half_open_start: dict[str, float] = {}  # Track when HALF_OPEN started

        # DRY: Check persistence capability once at init
        self._persistence_enabled = (
            self.db is not None and hasattr(self.db, "transaction")
        )

        if self._persistence_enabled:
            self._ensure_table()
            self._load_state()

    def _ensure_table(self) -> None:
        """Create persistence table if needed."""
        if not self._persistence_enabled:
            return
        with self.db.transaction() as conn:  # type: ignore[union-attr]
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                    key TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    last_failure REAL,
                    last_success REAL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _load_state(self) -> None:
        """Load persisted state into memory."""
        if not self._persistence_enabled:
            return
        try:
            with self.db.transaction() as conn:  # type: ignore[union-attr]
                rows = conn.execute(
                    "SELECT key, state, failure_count, success_count, last_failure, last_success "
                    "FROM circuit_breaker_state"
                ).fetchall()
        except Exception as e:
            logger.warning(f"Failed to load circuit breaker state from DB: {e}")
            return

        with self._lock:
            for row in rows:
                try:
                    key = row["key"]
                    state_value = row["state"]
                    state = CircuitState(state_value)
                    failure_count = int(row["failure_count"] or 0)
                    success_count = int(row["success_count"] or 0)
                    last_failure = row["last_failure"]
                    last_success = row["last_success"]
                except Exception as e:
                    logger.debug(f"Skipping malformed circuit breaker row: {e}")
                    continue

                metrics = CircuitBreakerMetrics(
                    total_calls=failure_count + success_count,
                    total_failures=failure_count,
                    total_successes=success_count,
                    current_failures=failure_count,
                    last_failure_time=last_failure,
                    last_success_time=last_success,
                    state_changes=0,
                )
                self._states[key] = state
                self._metrics[key] = metrics
                if state == CircuitState.HALF_OPEN:
                    self._half_open_used[key] = 0
                    # Assume HALF_OPEN started at load time (best effort for persistence)
                    self._half_open_start[key] = time.time()

    def _persist_state(self, key: str) -> None:
        """Persist state to SQLite if available."""
        if not self._persistence_enabled:
            return
        metrics = self._metrics.get(key)
        state = self._states.get(key, CircuitState.CLOSED)
        if metrics is None:
            # Key was reset/deleted - remove from DB
            try:
                with self.db.transaction() as conn:  # type: ignore[union-attr]
                    conn.execute(
                        "DELETE FROM circuit_breaker_state WHERE key = ?",
                        (key,),
                    )
            except Exception as e:
                logger.warning(f"Failed to delete circuit breaker state for key '{key}': {e}")
            return
        try:
            with self.db.transaction() as conn:  # type: ignore[union-attr]
                conn.execute(
                    """
                    INSERT INTO circuit_breaker_state
                        (key, state, failure_count, success_count, last_failure, last_success, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        state=excluded.state,
                        failure_count=excluded.failure_count,
                        success_count=excluded.success_count,
                        last_failure=excluded.last_failure,
                        last_success=excluded.last_success,
                        updated_at=excluded.updated_at
                    """,
                    (
                        key,
                        state.value,
                        metrics.current_failures,
                        metrics.total_successes,
                        metrics.last_failure_time,
                        metrics.last_success_time,
                        time.time(),
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to persist circuit breaker state for key '{key}': {e}")
            return

    def get_state(self, key: str) -> CircuitState:
        """Return state, transitioning OPEN->HALF_OPEN after timeout."""
        with self._lock:
            state = self._states.get(key, CircuitState.CLOSED)
            metrics = self._metrics.get(key)
            now = time.time()

            if state == CircuitState.OPEN and metrics and metrics.last_failure_time:
                if now - metrics.last_failure_time >= self.reset_timeout:
                    self._states[key] = CircuitState.HALF_OPEN
                    self._half_open_used[key] = 0
                    self._half_open_start[key] = now  # Track when HALF_OPEN started
                    metrics.state_changes += 1
                    self._persist_state(key)
                    return CircuitState.HALF_OPEN

            # HALF_OPEN timeout: if probe was issued but never reported, reset probe allowance
            if state == CircuitState.HALF_OPEN:
                half_open_start = self._half_open_start.get(key)
                used = self._half_open_used.get(key, 0)
                if half_open_start and used >= 1:
                    # Probe was issued (used >= 1) - check if it timed out
                    if now - half_open_start >= self.half_open_timeout:
                        # Probe never reported - allow a new probe
                        self._half_open_used[key] = 0
                        self._half_open_start[key] = now

            return state

    def can_execute(self, key: str) -> bool:
        """Return True if execution is allowed for this key.

        Thread-safe: entire check-then-act sequence is atomic via RLock.
        """
        with self._lock:
            state = self.get_state(key)  # This also handles HALF_OPEN timeout
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.OPEN:
                return False
            # HALF_OPEN: allow one probe
            used = self._half_open_used.get(key, 0)
            if used >= 1:
                return False
            self._half_open_used[key] = used + 1
            self._half_open_start[key] = time.time()  # Track probe start time
            return True

    def is_open(self, key: str) -> bool:
        """Compatibility helper: True if circuit should block execution."""
        return not self.can_execute(key)

    def record_failure(self, key: str) -> None:
        now = time.time()
        with self._lock:
            metrics = self._metrics.setdefault(key, CircuitBreakerMetrics())

            # SECURITY: Expire old failure counts based on reset_timeout.
            # This prevents sporadic failures spread over long periods from
            # accumulating and opening the circuit unexpectedly.
            if metrics.last_failure_time is not None:
                if now - metrics.last_failure_time >= self.reset_timeout:
                    metrics.current_failures = 0

            metrics.total_calls += 1
            metrics.total_failures += 1
            metrics.current_failures += 1
            metrics.last_failure_time = now

            state = self._states.get(key, CircuitState.CLOSED)
            if state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                if metrics.current_failures >= self.max_failures or state == CircuitState.HALF_OPEN:
                    self._states[key] = CircuitState.OPEN
                    metrics.state_changes += 1
            self._persist_state(key)

    def record_success(self, key: str) -> None:
        now = time.time()
        with self._lock:
            metrics = self._metrics.setdefault(key, CircuitBreakerMetrics())
            metrics.total_calls += 1
            metrics.total_successes += 1
            metrics.current_failures = 0
            metrics.last_success_time = now

            state = self._states.get(key, CircuitState.CLOSED)
            if state == CircuitState.HALF_OPEN:
                self._states[key] = CircuitState.CLOSED
                metrics.state_changes += 1
                self._half_open_used[key] = 0
                self._half_open_start.pop(key, None)  # Clean up tracking
            self._persist_state(key)

    def reset(self, key: str) -> None:
        """Reset breaker state and metrics for a key."""
        with self._lock:
            self._states.pop(key, None)
            self._metrics.pop(key, None)
            self._half_open_used.pop(key, None)
            self._half_open_start.pop(key, None)  # Clean up tracking
            self._persist_state(key)

    def get_metrics(self, key: str) -> CircuitBreakerMetrics:
        """Return metrics for a key (default if missing)."""
        with self._lock:
            return self._metrics.get(key, CircuitBreakerMetrics())


class ErrorClassifier:
    """Classify errors for appropriate handling."""

    # All patterns are lowercase for case-insensitive matching
    TRANSIENT_PATTERNS = [
        r"timed? ?out",
        r"connection (refused|reset|closed)",
        r"temporary failure",
        r"rate limit",
        r"overloaded",
    ]

    VALIDATION_PATTERNS = [
        r"invalid json",
        r"validation error",
        r"missing required field",
        r"no json block found",
    ]

    FATAL_PATTERNS = [
        r"blocked:",
        r"cannot proceed",
        r"permission denied",
        r"authentication failed",
    ]

    def classify(self, error: str) -> tuple[ErrorCategory, ErrorAction]:
        import re

        error_lower = error.lower()

        # Check fatal first
        for pattern in self.FATAL_PATTERNS:
            if re.search(pattern, error_lower):
                return ErrorCategory.LOGIC, ErrorAction.ESCALATE

        # Check validation (retry with feedback)
        for pattern in self.VALIDATION_PATTERNS:
            if re.search(pattern, error_lower):
                return ErrorCategory.VALIDATION, ErrorAction.RETRY_WITH_FEEDBACK

        # Check transient (retry same)
        for pattern in self.TRANSIENT_PATTERNS:
            if re.search(pattern, error_lower):
                return ErrorCategory.NETWORK, ErrorAction.RETRY_SAME

        # Unknown - retry once
        return ErrorCategory.LOGIC, ErrorAction.RETRY_ONCE


def _generate_circuit_key(
    workflow_id: str,
    role_name: str,
    task_description: str,
) -> str:
    """Generate a deterministic key for circuit breaker tracking.

    Uses a hash of (workflow_id, role_name, task_description) to ensure
    the same logical task always gets the same key. This allows the circuit
    breaker to properly aggregate failures across retries.
    """
    identity = f"{workflow_id}:{role_name}:{task_description}"
    hash_base = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return f"{workflow_id}-{role_name}-{hash_base}"


class ExecutionEngine:
    """Main execution engine for the Supervisor orchestrator.

    SECURITY: Always runs in Docker sandbox. Requires Docker to be available.
    """

    def __init__(
        self,
        repo_path: Path,
        db: Database | None = None,
        sandbox_config: SandboxConfig | None = None,
    ):
        self.repo_path = Path(repo_path).absolute()
        self.db = db or Database(self.repo_path / ".supervisor/state.db")
        self.role_loader = RoleLoader()
        self.context_packer = ContextPacker(self.repo_path)

        # Configure sandbox with allowed workdir roots for security
        # Workdirs are only allowed under the repo's .worktrees directory
        if sandbox_config is None:
            self.sandbox_config = SandboxConfig(
                allowed_workdir_roots=[str(self.repo_path / ".worktrees")]
            )
        else:
            # If user provided config, add worktrees to allowed roots if not set
            if not sandbox_config.allowed_workdir_roots:
                sandbox_config.allowed_workdir_roots = [
                    str(self.repo_path / ".worktrees")
                ]
            self.sandbox_config = sandbox_config

        # Validate Docker is available at startup
        require_docker()

        # Create sandboxed executor for gates/tests
        self.executor = get_sandboxed_executor(self.sandbox_config)
        self.workspace = IsolatedWorkspace(self.repo_path, self.executor, self.db)
        self.circuit_breaker = EnhancedCircuitBreaker(db=self.db)
        self.error_classifier = ErrorClassifier()

        # Gate execution system - uses GateExecutor for caching, integrity, dependencies
        # Enable project-specific gate configs from .supervisor/gates.yaml
        self.gate_loader = GateLoader(self.repo_path, allow_project_gates=True)
        self.gate_executor = GateExecutor(
            executor=self.executor,
            gate_loader=self.gate_loader,
            db=self.db,
        )

        # CLI clients (lazily initialized per CLI name)
        self._cli_clients: dict[str, SandboxedLLMClient] = {}
        self._cli_clients_lock = threading.Lock()  # Thread-safe client initialization

    def _get_cli_client(self, cli_name: str) -> SandboxedLLMClient:
        """Get or create sandboxed CLI client.

        Thread-safe using double-checked locking pattern.
        """
        # Fast path: already initialized (no lock needed)
        if cli_name in self._cli_clients:
            return self._cli_clients[cli_name]

        # Slow path: need to initialize (use lock to prevent race)
        with self._cli_clients_lock:
            # Re-check after acquiring lock (another thread may have initialized)
            if cli_name not in self._cli_clients:
                self._cli_clients[cli_name] = SandboxedLLMClient(
                    cli_name=cli_name,
                    config=self.sandbox_config,
                )
            return self._cli_clients[cli_name]

    # FIX (v27 - Gemini PR review): Helper methods to reduce run_role complexity

    def _setup_role_execution(
        self,
        role_name: str,
        task_description: str,
        workflow_id: str,
        step_id: str | None,
        target_files: list[str] | None,
        extra_context: dict[str, str] | None,
        retry_policy: RetryPolicy | None,
        gates: list[str] | None,
    ) -> tuple[str, RetryPolicy, list[str], str, RoleConfig, str]:
        """Setup role execution context.

        Returns:
            Tuple of (step_id, retry_policy, gates, circuit_key, role, prompt)
        """
        step_id = step_id or f"{workflow_id}-{role_name}-{uuid.uuid4().hex[:8]}"
        retry_policy = retry_policy or RetryPolicy()
        gates = gates or []

        circuit_key = _generate_circuit_key(workflow_id, role_name, task_description)

        if self.circuit_breaker.is_open(circuit_key):
            raise CircuitOpenError(
                f"Circuit breaker open for {circuit_key}. "
                f"Too many failures. Manual intervention required."
            )

        role = self.role_loader.load_role(role_name)

        template_name = self._get_template_for_role(role)
        if template_name:
            prompt = self.context_packer.build_full_prompt(
                template_name, role, task_description, target_files, extra_context,
            )
        else:
            prompt = self.context_packer.pack_context(
                role=role, task_description=task_description,
                target_files=target_files, extra_context=extra_context,
            )

        return step_id, retry_policy, gates, circuit_key, role, prompt

    def _build_gate_overrides(
        self,
        gates: list[str],
        role: RoleConfig,
        gate_loader: GateLoader,
    ) -> dict[str, GateFailAction]:
        """Build effective gate fail action overrides.

        Handles role overrides, gate severity defaults, and optional gate propagation.
        """
        resolved_gates = gate_loader.resolve_execution_order(gates)
        effective_overrides: dict[str, GateFailAction] = {}

        for gate_name in resolved_gates:
            if gate_name in role.on_fail_overrides:
                effective_overrides[gate_name] = role.on_fail_overrides[gate_name]
            else:
                try:
                    gate_config = gate_loader.get_gate(gate_name)
                    if gate_config.severity in (GateSeverity.WARNING, GateSeverity.INFO):
                        effective_overrides[gate_name] = GateFailAction.WARN
                    else:
                        effective_overrides[gate_name] = GateFailAction.BLOCK
                except GateNotFoundError:
                    effective_overrides[gate_name] = GateFailAction.BLOCK

        # Propagate optional behavior to dependencies
        deps_cache: dict[str, set[str]] = {}

        def collect_deps(gn: str) -> set[str]:
            if gn in deps_cache:
                return deps_cache[gn]
            deps: set[str] = set()
            try:
                gc = gate_loader.get_gate(gn)
                for dep in gc.depends_on:
                    if dep not in deps:
                        deps.add(dep)
                        deps.update(collect_deps(dep))
            except GateNotFoundError:
                pass
            deps_cache[gn] = deps
            return deps

        for gate_name in gates:
            if effective_overrides.get(gate_name) == GateFailAction.WARN:
                for dep in collect_deps(gate_name):
                    if effective_overrides.get(dep) == GateFailAction.BLOCK:
                        effective_overrides[dep] = GateFailAction.WARN

        return effective_overrides

    def _apply_and_finalize_step(
        self,
        workflow_id: str,
        step_id: str,
        role_name: str,
        output: BaseModel,
        ctx: Any,  # IsolatedContext
        cancellation_check: "Callable[[], bool] | None",
    ) -> list[str]:
        """Apply changes to main tree and record success.

        Returns list of changed files.

        Raises:
            CancellationError: If cancelled before/during apply
            ApplyError: If apply fails
        """
        if cancellation_check and cancellation_check():
            logger.warning(
                f"Step '{step_id}' cancelled before applying changes - "
                "discarding worktree changes"
            )
            raise CancellationError(
                f"Step '{step_id}' was cancelled (likely timeout). "
                "Changes were not applied to avoid out-of-policy mutations."
            )

        self.db.append_event(
            Event(
                workflow_id=workflow_id,
                event_type=EventType.STEP_APPLYING,
                step_id=step_id,
                role=role_name,
                payload={"original_head": ctx.original_head},
            )
        )

        try:
            with self.workspace._apply_lock:
                if cancellation_check and cancellation_check():
                    raise CancellationError(
                        f"Step '{step_id}' cancelled after acquiring apply lock. "
                        "Changes not applied."
                    )
                changed_files = self.workspace._apply_changes(
                    ctx.worktree_path, ctx.original_head
                )
        except CancellationError:
            raise
        except Exception as apply_err:
            raise ApplyError(
                f"Apply failed (repository may be in inconsistent state): {apply_err}"
            ) from apply_err

        try:
            self.db.append_event(
                Event(
                    workflow_id=workflow_id,
                    event_type=EventType.STEP_COMPLETED,
                    step_id=step_id,
                    role=role_name,
                    payload={
                        "output": output.model_dump() if hasattr(output, "model_dump") else output,
                        "files_changed": changed_files,
                    },
                )
            )
        except Exception as db_error:
            import sys
            print(
                f"CRITICAL: Step {step_id} applied changes but DB update failed: {db_error}. "
                f"Changed files: {changed_files}. Manual remediation may be required.",
                file=sys.stderr,
            )
            try:
                self.db.append_event(
                    Event(
                        workflow_id=workflow_id,
                        event_type=EventType.STEP_FAILED,
                        step_id=step_id,
                        role=role_name,
                        payload={
                            "error": f"DB update failed after apply: {db_error}",
                            "files_changed": changed_files,
                            "inconsistent_state": True,
                        },
                    )
                )
            except Exception:
                pass
            raise

        return changed_files

    def run_role(
        self,
        role_name: str,
        task_description: str,
        workflow_id: str,
        step_id: str | None = None,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
        retry_policy: RetryPolicy | None = None,
        gates: list[str] | None = None,
        cli_override: str | None = None,
        cancellation_check: "Callable[[], bool] | None" = None,
    ) -> BaseModel:
        """Run a role with full context packing, isolation, and error handling.

        SECURITY: All CLI execution happens in an isolated worktree.
        Changes are only applied to main tree after gates pass.

        Args:
            role_name: Name of the role to run
            task_description: Description of the task
            workflow_id: ID of the parent workflow
            step_id: Optional step ID (generated if not provided)
            target_files: Specific files to focus on
            extra_context: Additional context (git diff, test output, etc.)
            retry_policy: Retry configuration
            gates: Optional gates to run (e.g., ["test", "lint"])
            cli_override: Override CLI to use instead of role's default (from ModelRouter)
            cancellation_check: Optional callback that returns True if operation should be
                cancelled (e.g., due to timeout). Checked before applying changes.

        Returns:
            Parsed output from the worker

        Raises:
            RetryExhaustedError: All retry attempts failed
            CircuitOpenError: Too many failures, circuit breaker open
            GateFailedError: Gate verification failed
            CancellationError: Operation was cancelled before applying changes
        """
        import time
        start_time = time.time()
        success = False
        error_category = None
        attempt_count = 0
        
        # Load role configuration upfront for metrics
        try:
            role_config = self.role_loader.load_role(role_name)
            cli_used = cli_override or role_config.cli
        except Exception:
            cli_used = "unknown"

        try:
            # FIX (v27 - Gemini PR review): Use helper method for setup
            step_id, retry_policy, gates, circuit_key, role, prompt = self._setup_role_execution(
                role_name, task_description, workflow_id, step_id,
                target_files, extra_context, retry_policy, gates,
            )

            # Record step started
            self.db.append_event(
                Event(
                    workflow_id=workflow_id,
                    event_type=EventType.STEP_STARTED,
                    step_id=step_id,
                    role=role_name,
                    payload={"task": task_description[:500], "gates": gates},
                )
            )

            last_error: Exception | None = None
            feedback: str | None = None

            for attempt in range(retry_policy.max_attempts):
                attempt_count = attempt
                try:
                    # Add feedback from previous attempt if any
                    effective_prompt = prompt
                    if feedback:
                        effective_prompt = f"{prompt}\n\n## Previous Attempt Feedback\n\n{feedback}"

                    # Execute CLI in ISOLATED worktree
                    with self.workspace.isolated_execution(step_id) as ctx:
                        result = self._execute_cli(role, effective_prompt, ctx.worktree_path, cli_override)

                        if result.returncode != 0 and not result.timed_out:
                            raise EngineError(
                                f"CLI exited with code {result.returncode}: {result.stderr}"
                            )

                        if result.timed_out:
                            raise EngineError(f"CLI timed out: {result.stderr}")

                        # PHASE 2: Use CLI adapter for parsing
                        # FIX (PR review): Use the actual CLI that executed, not role.cli
                        # When cli_override is used, output format matches the override CLI
                        effective_cli = cli_override or role.cli
                        adapter = get_adapter(effective_cli)
                        # NOTE: Use _get_schema_for_role to resolve overlay extends chain
                        schema = self._get_schema_for_role(role)
                        output = adapter.parse_output(result.stdout, schema)

                        # Run gates IN THE WORKTREE using GateExecutor
                        # FIX (v27 - Gemini PR review): Use helper for gate setup
                        worktree_gate_loader = GateLoader(ctx.worktree_path, allow_project_gates=True)
                        worktree_gate_executor = GateExecutor(
                            executor=self.executor,
                            gate_loader=worktree_gate_loader,
                            db=self.db,
                        )

                        # Build effective overrides using helper method
                        effective_overrides = self._build_gate_overrides(gates, role, worktree_gate_loader)

                        gate_results = worktree_gate_executor.run_gates(
                            gate_names=gates,
                            worktree_path=ctx.worktree_path,
                            workflow_id=workflow_id,
                            step_id=step_id,
                            on_fail_overrides=effective_overrides,
                        )

                        # Check for blocking failures (GateExecutor already logged events)
                        for gate_result in gate_results:
                            if gate_result.status == GateStatus.FAILED:
                                # All gates (including dependencies) resolved upfront
                                on_fail = effective_overrides.get(
                                    gate_result.gate_name, GateFailAction.BLOCK
                                )

                                if on_fail == GateFailAction.WARN:
                                    # Already logged by GateExecutor, continue
                                    continue
                                # BLOCK or RETRY_WITH_FEEDBACK: raise exception
                                raise GateFailedError(gate_result.gate_name, gate_result.output)

                        # FIX (v27 - Gemini PR review): Use helper for apply and finalize
                        changed_files = self._apply_and_finalize_step(
                            workflow_id, step_id, role_name, output, ctx, cancellation_check,
                        )

                    self.circuit_breaker.reset(circuit_key)
                    success = True
                    return output

                except GateFailedError as e:
                    # WARN gates are handled inline before raising GateFailedError.
                    # This exception handler only handles BLOCK and RETRY_WITH_FEEDBACK actions.
                    on_fail_action = role.on_fail_overrides.get(e.gate_name, GateFailAction.BLOCK)

                    if on_fail_action == GateFailAction.RETRY_WITH_FEEDBACK:
                        # Generate structured feedback and continue retry loop
                        gate_result = GateResult(
                            gate_name=e.gate_name,
                            status=GateStatus.FAILED,
                            output=e.output,
                            duration_seconds=0.0,  # Not tracked at this level
                        )
                        feedback_gen = StructuredFeedbackGenerator()
                        feedback = feedback_gen.generate(
                            gate_result, context=task_description
                        )
                        last_error = e
                        self.circuit_breaker.record_failure(circuit_key)

                        if attempt < retry_policy.max_attempts - 1:
                            time.sleep(retry_policy.get_delay(attempt))
                            continue  # Explicitly continue to next attempt
                        # On last attempt, fall through to retry exhausted logic

                    else:
                        # BLOCK (or unexpected WARN - which shouldn't reach here):
                        # Gate failures block execution
                        self.db.append_event(
                            Event(
                                workflow_id=workflow_id,
                                event_type=EventType.STEP_FAILED,
                                step_id=step_id,
                                role=role_name,
                                payload={
                                    "error": f"Gate '{e.gate_name}' failed (blocking)",
                                    "gate": e.gate_name,
                                },
                            )
                        )
                        raise

                except ApplyError as e:
                    # CRITICAL: Apply errors are FATAL - do NOT retry
                    # Repository may be in inconsistent state; retrying would compound corruption
                    self.db.append_event(
                        Event(
                            workflow_id=workflow_id,
                            event_type=EventType.STEP_FAILED,
                            step_id=step_id,
                            role=role_name,
                            payload={
                                "error": str(e),
                                "fatal": True,
                                "reason": "Apply failure - repository may be corrupted",
                            },
                        )
                    )
                    raise  # Do not retry - immediate failure

                except (ParsingError, InvalidOutputError) as e:
                    last_error = e
                    category, action = self.error_classifier.classify(str(e))

                    if action == ErrorAction.RETRY_WITH_FEEDBACK:
                        feedback = self._build_feedback(e, role)
                        self.circuit_breaker.record_failure(circuit_key)
                    elif action == ErrorAction.ESCALATE:
                        break  # Don't retry
                    else:
                        self.circuit_breaker.record_failure(circuit_key)

                    if attempt < retry_policy.max_attempts - 1:
                        time.sleep(retry_policy.get_delay(attempt))

                except CancellationError:
                    # FIX (Codex review): Don't retry cancelled operations
                    # CancellationError means component was timed out - no point retrying
                    logger.info(f"Step '{step_id}' cancelled, aborting without retry")
                    self.db.append_event(
                        Event(
                            workflow_id=workflow_id,
                            event_type=EventType.STEP_FAILED,
                            step_id=step_id,
                            role=role_name,
                            payload={"error": "Cancelled (timeout)", "cancelled": True},
                        )
                    )
                    raise  # Re-raise without retry

                except Exception as e:
                    last_error = e
                    self.circuit_breaker.record_failure(circuit_key)

                    if attempt < retry_policy.max_attempts - 1:
                        time.sleep(retry_policy.get_delay(attempt))

            # All retries exhausted
            self.db.append_event(
                Event(
                    workflow_id=workflow_id,
                    event_type=EventType.STEP_FAILED,
                    step_id=step_id,
                    role=role_name,
                    payload={"error": str(last_error)},
                )
            )
            
            error_category = type(last_error).__name__ if last_error else "RetryExhausted"
            raise RetryExhaustedError(
                f"Role '{role_name}' failed after {retry_policy.max_attempts} attempts: {last_error}"
            )

        except Exception as e:
            error_category = type(e).__name__
            raise e

        finally:
            if hasattr(self, 'db') and self.db is not None:
                duration = time.time() - start_time
                task_type = _infer_task_type(role_name)
                try:
                    self.db.record_metric(
                        role=role_name,
                        cli=cli_used,
                        workflow_id=workflow_id,
                        success=success,
                        duration_seconds=duration,
                        task_type=task_type,
                        retry_count=attempt_count,
                        error_category=error_category,
                    )
                except Exception as metric_err:
                    logger.warning(f"Failed to record metric: {metric_err}")

    def _execute_cli(
        self,
        role: RoleConfig,
        prompt: str,
        workdir: Path,
        cli_override: str | None = None,
    ) -> ExecutionResult:
        """Execute CLI for a role in sandbox.

        Args:
            role: Role configuration
            prompt: The prompt to send
            workdir: Working directory (must be a worktree for isolation)
            cli_override: Optional CLI to use instead of role's default

        Note: workdir is required and must be under allowed_workdir_roots
              (typically repo/.worktrees). The repo root is intentionally
              NOT allowed to enforce worktree isolation.
        """
        cli_name = cli_override or role.cli
        client = self._get_cli_client(cli_name)
        return client.execute(prompt, workdir)

    def _get_template_for_role(self, role: RoleConfig) -> str | None:
        """Map role to template, resolving overlays via base_role.

        For overlay roles (e.g., reviewer-python extends reviewer),
        uses the base role's template.

        base_role is computed by RoleLoader and handles multi-level overlays:
        - reviewer-python (extends reviewer) -> base_role = "reviewer"
        - my-reviewer (extends reviewer-python extends reviewer) -> base_role = "reviewer"

        Returns None only for truly unknown roles to use legacy pack_context.
        """
        templates = {
            "planner": "planning.j2",
            "implementer": "implement.j2",
            "reviewer": "review_strict.j2",
        }

        # First try exact role name (for base roles)
        if role.name in templates:
            return templates[role.name]

        # Then try base_role (for overlays - handles multi-level extends)
        if role.base_role and role.base_role in templates:
            return templates[role.base_role]

        # Unknown role - fallback to legacy pack_context
        return None

    def _get_schema_for_role(self, role: RoleConfig) -> type[BaseModel]:
        """Get output schema for role, resolving overlays via base_role.

        For overlay roles (e.g., implementer-python extends implementer),
        uses the base role's schema.

        base_role is computed by RoleLoader and handles multi-level overlays.
        """
        # First try exact role name (for base roles)
        if role.name in ROLE_SCHEMAS:
            return ROLE_SCHEMAS[role.name]

        # Then try base_role (for overlays - handles multi-level extends)
        if role.base_role and role.base_role in ROLE_SCHEMAS:
            return ROLE_SCHEMAS[role.base_role]

        # Unknown role - use GenericOutput
        return GenericOutput

    def _build_feedback(self, error: Exception, role: RoleConfig) -> str:
        """Build feedback message for retry.

        IMPORTANT: Always includes a warning that file changes were discarded,
        since each retry starts with a fresh worktree from HEAD.
        """
        # Common warning about worktree reset - agent must re-apply all changes
        worktree_warning = (
            "\n\n⚠️ IMPORTANT: Your previous file modifications have been DISCARDED. "
            "The workspace has been reset to a clean state. "
            "You MUST re-apply ALL code changes in this attempt."
        )

        if isinstance(error, ParsingError):
            return (
                f"Your previous output failed to parse:\n{error}\n\n"
                f"REQUIRED: End your response with a JSON code block:\n"
                f"```json\n{{\n  \"status\": \"...\",\n  ...\n}}\n```"
                f"{worktree_warning}"
            )
        elif isinstance(error, InvalidOutputError):
            return (
                f"Your previous output failed schema validation:\n{error}\n\n"
                f"Please ensure your JSON output matches the required schema."
                f"{worktree_warning}"
            )
        else:
            return f"Previous attempt failed:\n{error}{worktree_warning}"

    def run_step_isolated(
        self,
        step: Step,
        worker_fn: Callable[[Step, Path], dict[str, Any]],
    ) -> dict[str, Any]:
        """Run a step with full workspace isolation.

        Uses git worktrees for isolation. Gates run in worktree before
        applying changes to main tree.
        """
        return self.workspace.execute_step(step, worker_fn, step.gates)

    def run_plan(self, task_description: str, workflow_id: str) -> Any:
        """Run the planner role to create a feature plan."""
        return self.run_role(
            role_name="planner",
            task_description=task_description,
            workflow_id=workflow_id,
            extra_context={
                "file_tree": self.context_packer.get_file_tree(),
            },
        )

    # --- Phase 4: Feature->Phase->Component Workflow Integration ---

    def create_workflow_coordinator(
        self,
        max_parallel_workers: int = 3,
        prefer_speed: bool = False,
        prefer_cost: bool = False,
        max_stall_seconds: float = 600.0,
        component_timeout: float = 300.0,
        # NEW Phase 5 params
        workflow_timeout: float = 3600.0,
        checkpoint_on_timeout: bool = True,
        role_timeouts: dict[str, float] | None = None,
        approval_gate: Any = None, # Type hint looser to avoid cyclic import
        interaction_bridge: Any = None,
        adaptive_config: dict[str, Any] | None = None,
    ) -> "WorkflowCoordinator":
        """Create a WorkflowCoordinator for multi-model hierarchical workflows.

        Phase 4 integration point for Feature->Phase->Component execution.

        Args:
            max_parallel_workers: Maximum parallel component executions
            prefer_speed: Prioritize faster models when possible
            prefer_cost: Prioritize cheaper models when possible
            max_stall_seconds: Max time to wait for any progress before stalling
            component_timeout: Max time for a single component to run

        Returns:
            Configured WorkflowCoordinator instance
        """
        from supervisor.core.workflow import WorkflowCoordinator

        # FIX (PR review): Pass through all configuration parameters
        return WorkflowCoordinator(
            engine=self,
            db=self.db,
            repo_path=self.repo_path,
            max_parallel_workers=max_parallel_workers,
            prefer_speed=prefer_speed,
            prefer_cost=prefer_cost,
            max_stall_seconds=max_stall_seconds,
            component_timeout=component_timeout,
            workflow_timeout=workflow_timeout,
            checkpoint_on_timeout=checkpoint_on_timeout,
            role_timeouts=role_timeouts,
            approval_gate=approval_gate,
            interaction_bridge=interaction_bridge,
            adaptive_config=adaptive_config,
        )

    def execute_feature(
        self,
        feature_id: str,
        parallel: bool = True,
        max_parallel_workers: int = 3,
        prefer_speed: bool = False,
        prefer_cost: bool = False,
        max_stall_seconds: float = 600.0,
        component_timeout: float = 300.0,
    ) -> "Feature":
        """Execute all components of a feature in dependency order.

        Phase 4 convenience method that creates a WorkflowCoordinator
        and runs the implementation phase.

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
            max_parallel_workers: Maximum parallel workers
            prefer_speed: Prioritize faster models when possible
            prefer_cost: Prioritize cheaper models when possible
            max_stall_seconds: Max time to wait for any progress before stalling
            component_timeout: Max time for a single component to run

        Returns:
            Completed Feature object

        Raises:
            WorkflowBlockedError: If no progress can be made
        """
        from supervisor.core.models import Feature
        # FIX (Gemini review): Use factory method instead of duplicating instantiation
        coordinator = self.create_workflow_coordinator(
            max_parallel_workers=max_parallel_workers,
            prefer_speed=prefer_speed,
            prefer_cost=prefer_cost,
            max_stall_seconds=max_stall_seconds,
            component_timeout=component_timeout,
        )

        return coordinator.run_implementation(feature_id, parallel=parallel)

    def run_parallel_review(
        self,
        roles: list[str],
        task_description: str,
        workflow_id: str,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
        approval_policy: str = "ALL_APPROVED",
        timeout: float = 300.0,
        max_workers: int = 3,
    ) -> "AggregatedReviewResult":
        """Run multiple reviewers in parallel and aggregate results.

        Phase 4 convenience method for parallel multi-model review.

        Args:
            roles: List of reviewer role names (e.g., ["reviewer_gemini", "reviewer_codex"])
            task_description: What to review
            workflow_id: Workflow context
            target_files: Files to review
            extra_context: Additional context (git_diff, etc.)
            approval_policy: How to determine overall approval
            timeout: Maximum time to wait for all reviewers
            max_workers: Maximum parallel reviewers (default 3)

        Returns:
            AggregatedReviewResult with individual results and approval decision
        """
        from supervisor.core.parallel import AggregatedReviewResult, ParallelReviewer

        reviewer = ParallelReviewer(
            engine=self,
            max_workers=max_workers,
            timeout=timeout,
        )

        return reviewer.run_parallel_review(
            roles=roles,
            task_description=task_description,
            workflow_id=workflow_id,
            target_files=target_files,
            extra_context=extra_context,
            approval_policy=approval_policy,
        )

    # FIX (v27 - Gemini PR review): Removed duplicate _infer_task_type
    # Now using shared function from supervisor.core.routing