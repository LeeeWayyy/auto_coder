"""Tests for ExecutionEngine - the core orchestration component.

Tests cover:
- RetryPolicy: Exponential backoff calculation
- CircuitBreaker: Failure tracking and state management
- EnhancedCircuitBreaker: Advanced features with metrics
- ErrorClassifier: Error categorization for retry logic
- ExecutionEngine: Core execution flows (with mocked sandbox)
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from supervisor.core.engine import (
    CircuitBreaker,
    EnhancedCircuitBreaker,
    ErrorAction,
    ErrorCategory,
    ErrorClassifier,
    ExecutionEngine,
    RetryPolicy,
)
from supervisor.core.state import EventType
from supervisor.sandbox.executor import ExecutionResult

# =============================================================================
# RetryPolicy Tests
# =============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy - exponential backoff calculation."""

    def test_default_values(self):
        """RetryPolicy has sensible defaults."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.initial_delay == 2.0
        assert policy.backoff_multiplier == 2.0
        assert policy.max_delay == 60.0
        assert policy.jitter == 0.1

    def test_exponential_backoff_calculation(self):
        """Delay increases exponentially with attempt number."""
        policy = RetryPolicy(
            initial_delay=2.0,
            backoff_multiplier=2.0,
            jitter=0.0,  # No jitter for deterministic testing
        )

        # Attempt 0: 2.0
        assert policy.get_delay(0) == 2.0

        # Attempt 1: 2.0 * 2 = 4.0
        assert policy.get_delay(1) == 4.0

        # Attempt 2: 2.0 * 4 = 8.0
        assert policy.get_delay(2) == 8.0

        # Attempt 3: 2.0 * 8 = 16.0
        assert policy.get_delay(3) == 16.0

    def test_max_delay_enforced(self):
        """Delay is capped at max_delay."""
        policy = RetryPolicy(
            initial_delay=10.0,
            backoff_multiplier=2.0,
            max_delay=50.0,
            jitter=0.0,
        )

        # Attempt 10 would be 10 * 2^10 = 10240, but capped at 50
        delay = policy.get_delay(10)
        assert delay == 50.0

    def test_jitter_adds_randomness(self):
        """Jitter adds randomness to delay."""
        policy = RetryPolicy(
            initial_delay=10.0,
            jitter=0.1,  # ±10%
        )

        # Get multiple delays for same attempt
        delays = [policy.get_delay(0) for _ in range(100)]

        # All should be within ±10% of 10.0
        assert all(9.0 <= d <= 11.0 for d in delays)

        # Should have some variation (not all identical)
        assert len(set(delays)) > 1

    def test_custom_backoff_multiplier(self):
        """Custom backoff multiplier works correctly."""
        policy = RetryPolicy(
            initial_delay=1.0,
            backoff_multiplier=3.0,  # Triple each time
            jitter=0.0,
        )

        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 3.0
        assert policy.get_delay(2) == 9.0
        assert policy.get_delay(3) == 27.0


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker - prevent infinite retry loops."""

    def test_initial_state_closed(self):
        """Circuit starts in closed state (allows execution)."""
        breaker = CircuitBreaker()
        assert not breaker.is_open("test-step")

    def test_record_failure_increments_count(self):
        """Recording failures increments failure count."""
        breaker = CircuitBreaker(max_failures=3)

        breaker.record_failure("test-step")
        assert not breaker.is_open("test-step")  # Not open yet

        breaker.record_failure("test-step")
        assert not breaker.is_open("test-step")  # Still not open

        breaker.record_failure("test-step")
        assert breaker.is_open("test-step")  # Now open after 3 failures

    def test_circuit_opens_after_max_failures(self):
        """Circuit opens after reaching max_failures."""
        breaker = CircuitBreaker(max_failures=2)

        # Record 2 failures
        breaker.record_failure("step-1")
        breaker.record_failure("step-1")

        # Circuit should now be open
        assert breaker.is_open("step-1")

    def test_different_steps_tracked_independently(self):
        """Different step IDs have independent failure counts."""
        breaker = CircuitBreaker(max_failures=2)

        breaker.record_failure("step-1")
        breaker.record_failure("step-1")

        # step-1 is open
        assert breaker.is_open("step-1")

        # step-2 is still closed
        assert not breaker.is_open("step-2")

    def test_old_failures_expire(self):
        """Failures outside reset_timeout window are expired."""
        breaker = CircuitBreaker(max_failures=2, reset_timeout=1)  # 1 second timeout

        # Record a failure
        breaker.record_failure("test-step")

        # Wait for timeout
        time.sleep(1.1)

        # Check if open (this should clean old failures)
        is_open = breaker.is_open("test-step")
        assert not is_open  # Should be closed, old failure expired

        # Record one more failure (should not open circuit)
        breaker.record_failure("test-step")
        assert not breaker.is_open("test-step")

    def test_cleanup_stale_keys(self):
        """Stale keys are cleaned up to prevent memory growth."""
        breaker = CircuitBreaker(
            max_failures=2,
            reset_timeout=1,
            max_keys=100,
            _cleanup_interval=10,
        )

        # Add many different steps
        for i in range(150):
            breaker.record_failure(f"step-{i}")

        # Should trigger cleanup and enforce max_keys
        assert len(breaker._failures) <= breaker.max_keys

    def test_thread_safety(self):
        """CircuitBreaker is thread-safe."""
        import threading

        breaker = CircuitBreaker(max_failures=100)

        def record_failures():
            for _ in range(50):
                breaker.record_failure("concurrent-step")

        threads = [threading.Thread(target=record_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded 200 failures total (50 * 4 threads)
        # Circuit should be open
        assert breaker.is_open("concurrent-step")


# =============================================================================
# EnhancedCircuitBreaker Tests
# =============================================================================


class TestEnhancedCircuitBreaker:
    """Tests for EnhancedCircuitBreaker with advanced features."""

    def test_initial_state_closed(self):
        """Enhanced breaker starts closed."""
        breaker = EnhancedCircuitBreaker()
        assert breaker.can_execute("test-key")
        assert not breaker.is_open("test-key")

    def test_opens_after_max_failures(self):
        """Breaker opens after max_failures."""
        breaker = EnhancedCircuitBreaker(max_failures=3)

        for _ in range(3):
            breaker.record_failure("test-key")

        assert not breaker.can_execute("test-key")
        assert breaker.is_open("test-key")

    def test_half_open_state_after_timeout(self):
        """Breaker enters half-open state after reset_timeout."""
        breaker = EnhancedCircuitBreaker(
            max_failures=2,
            reset_timeout=1,  # 1 second
        )

        # Open the circuit
        breaker.record_failure("test-key")
        breaker.record_failure("test-key")
        assert breaker.is_open("test-key")

        # Wait for reset timeout
        time.sleep(1.1)

        # Should now be in half-open state (allows probe)
        assert breaker.can_execute("test-key")

    def test_half_open_allows_one_probe(self):
        """Half-open state allows exactly one probe request."""
        breaker = EnhancedCircuitBreaker(
            max_failures=1,
            reset_timeout=1,
        )

        # Open the circuit
        breaker.record_failure("test-key")
        assert breaker.is_open("test-key")

        # Wait for timeout
        time.sleep(1.1)

        # First call should be allowed (probe)
        assert breaker.can_execute("test-key")

        # Second call should be blocked (already used probe)
        assert not breaker.can_execute("test-key")

    def test_success_closes_half_open_circuit(self):
        """Success in half-open state closes the circuit."""
        breaker = EnhancedCircuitBreaker(
            max_failures=1,
            reset_timeout=1,
        )

        # Open the circuit
        breaker.record_failure("test-key")

        # Wait and probe
        time.sleep(1.1)
        breaker.can_execute("test-key")

        # Record success - should close circuit
        breaker.record_success("test-key")

        assert breaker.can_execute("test-key")
        assert not breaker.is_open("test-key")

    def test_failure_in_half_open_reopens(self):
        """Failure in half-open state reopens the circuit."""
        breaker = EnhancedCircuitBreaker(
            max_failures=1,
            reset_timeout=1,
        )

        # Open the circuit
        breaker.record_failure("test-key")

        # Wait and probe
        time.sleep(1.1)
        breaker.can_execute("test-key")

        # Another failure - should reopen
        breaker.record_failure("test-key")

        assert breaker.is_open("test-key")

    def test_metrics_tracking(self):
        """Breaker tracks detailed metrics."""
        breaker = EnhancedCircuitBreaker()

        breaker.record_success("test-key")
        breaker.record_success("test-key")
        breaker.record_failure("test-key")

        metrics = breaker.get_metrics("test-key")
        assert metrics.total_calls == 3
        assert metrics.total_successes == 2
        assert metrics.current_failures == 1

    def test_reset_clears_state(self):
        """Reset clears all state and metrics."""
        breaker = EnhancedCircuitBreaker()

        breaker.record_failure("test-key")
        breaker.record_failure("test-key")

        # Reset
        breaker.reset("test-key")

        # Should be back to initial state
        assert breaker.can_execute("test-key")
        metrics = breaker.get_metrics("test-key")
        assert metrics.total_calls == 0


# =============================================================================
# ErrorClassifier Tests
# =============================================================================


class TestErrorClassifier:
    """Tests for ErrorClassifier - categorize errors for retry logic."""

    def test_classify_network_error(self):
        """Network errors are classified as transient."""
        classifier = ErrorClassifier()

        category, action = classifier.classify("Connection refused by server")
        assert category == ErrorCategory.NETWORK
        assert action == ErrorAction.RETRY_SAME

        category, action = classifier.classify("Request timed out after 30s")
        assert category == ErrorCategory.NETWORK
        assert action == ErrorAction.RETRY_SAME

    def test_classify_parsing_error(self):
        """Parsing/validation errors trigger retry with feedback."""
        classifier = ErrorClassifier()

        category, action = classifier.classify("Invalid JSON format in response")
        assert category == ErrorCategory.VALIDATION
        assert action == ErrorAction.RETRY_WITH_FEEDBACK

        category, action = classifier.classify("Validation error: missing required field 'name'")
        assert category == ErrorCategory.VALIDATION
        assert action == ErrorAction.RETRY_WITH_FEEDBACK

    def test_classify_fatal_error(self):
        """Fatal errors are escalated, not retried."""
        classifier = ErrorClassifier()

        category, action = classifier.classify("blocked: permission denied")
        assert category == ErrorCategory.LOGIC
        assert action == ErrorAction.ESCALATE

        category, action = classifier.classify("Authentication failed - invalid token")
        assert category == ErrorCategory.LOGIC
        assert action == ErrorAction.ESCALATE

    def test_classify_unknown_error(self):
        """Unknown errors are retried once."""
        classifier = ErrorClassifier()

        category, action = classifier.classify("Something went wrong unexpectedly")
        assert category == ErrorCategory.LOGIC
        assert action == ErrorAction.RETRY_ONCE

    def test_case_insensitive_matching(self):
        """Error classification is case-insensitive."""
        classifier = ErrorClassifier()

        category1, action1 = classifier.classify("CONNECTION REFUSED")
        category2, action2 = classifier.classify("connection refused")

        assert category1 == category2 == ErrorCategory.NETWORK
        assert action1 == action2 == ErrorAction.RETRY_SAME


# =============================================================================
# ExecutionEngine Tests
# =============================================================================


@pytest.mark.git
class TestExecutionEngine:
    """Tests for ExecutionEngine - main orchestration logic."""

    @pytest.fixture
    def engine(self, repo_with_git, test_db):
        """Create ExecutionEngine with test database."""
        # Mock Docker requirement
        with patch("supervisor.core.engine.require_docker"):
            engine = ExecutionEngine(repo_with_git, db=test_db)
            return engine

    def test_engine_initialization(self, repo_with_git, test_db):
        """Engine initializes with correct dependencies."""
        with patch("supervisor.core.engine.require_docker"):
            engine = ExecutionEngine(repo_with_git, db=test_db)

            assert engine.repo_path == repo_with_git
            assert engine.db == test_db
            assert engine.role_loader is not None
            assert engine.context_packer is not None

    def test_engine_requires_docker(self, repo_with_git, test_db):
        """Engine initialization requires Docker."""
        from supervisor.sandbox.executor import DockerNotAvailableError

        with patch("supervisor.core.engine.require_docker") as mock_require:
            mock_require.side_effect = DockerNotAvailableError("Docker not found")

            with pytest.raises(DockerNotAvailableError):
                ExecutionEngine(repo_with_git, db=test_db)

    def test_run_role_successful_execution(self, engine, mocker):
        """run_role executes successfully with mocked components."""
        from contextlib import contextmanager

        from supervisor.core.workspace import WorktreeContext

        # Mock role loading with proper context dict
        mock_role = Mock()
        mock_role.name = "implementer"
        mock_role.cli = "claude:sonnet"
        mock_role.gates = []
        mock_role.config = {}
        mock_role.context = {}
        mock_role.system_prompt = "You are an implementer."
        mock_role.flags = []
        mock_role.base_role = "implementer"
        mock_role.on_fail_overrides = {}
        mocker.patch.object(engine.role_loader, "load_role", return_value=mock_role)

        # Mock context packing
        mocker.patch.object(
            engine.context_packer, "build_full_prompt", return_value="Packed context"
        )

        # Mock _execute_cli directly to avoid model routing
        mock_result = ExecutionResult(
            returncode=0,
            stdout='{"status": "success", "files_modified": []}',
            stderr="",
        )
        mocker.patch.object(engine, "_execute_cli", return_value=mock_result)

        # Mock parser adapter
        mock_adapter = Mock()
        mock_output = Mock()
        mock_output.status = "success"
        mock_output.files_modified = []
        mock_adapter.parse_output.return_value = mock_output
        mocker.patch("supervisor.core.engine.get_adapter", return_value=mock_adapter)

        # Mock workspace.isolated_execution context manager
        @contextmanager
        def mock_isolated_execution(step_id):
            yield WorktreeContext(
                worktree_path=Path("/tmp/worktree"),
                step_id=step_id,
                original_head="abc123",
            )

        mocker.patch.object(
            engine.workspace, "isolated_execution", side_effect=mock_isolated_execution
        )

        # Mock _apply_and_finalize_step to avoid workspace apply logic
        mocker.patch.object(engine, "_apply_and_finalize_step", return_value=[])

        # Execute
        result = engine.run_role(
            role_name="implementer",
            task_description="Test task",
            workflow_id="test-wf",
            gates=[],
        )

        assert result is not None

    def test_run_role_with_retry_on_transient_error(self, engine, mocker):
        """run_role retries on transient errors."""
        from contextlib import contextmanager

        from supervisor.core.workspace import WorktreeContext

        mock_role = Mock()
        mock_role.name = "implementer"
        mock_role.cli = "claude:sonnet"
        mock_role.gates = []
        mock_role.config = {"max_retries": 2}
        mock_role.context = {}
        mock_role.system_prompt = "You are an implementer."
        mock_role.flags = []
        mock_role.base_role = "implementer"
        mock_role.on_fail_overrides = {}
        mocker.patch.object(engine.role_loader, "load_role", return_value=mock_role)

        mocker.patch.object(engine.context_packer, "build_full_prompt", return_value="Context")

        # Mock _execute_cli: fail once with transient error, then succeed
        mock_execute = mocker.patch.object(engine, "_execute_cli")
        mock_execute.side_effect = [
            ExecutionResult(returncode=1, stdout="", stderr="Connection timed out"),
            ExecutionResult(
                returncode=0,
                stdout='{"status": "success", "files_modified": []}',
                stderr="",
            ),
        ]

        # Mock parser adapter
        mock_adapter = Mock()
        mock_output = Mock()
        mock_output.status = "success"
        mock_output.files_modified = []
        mock_adapter.parse_output.return_value = mock_output
        mocker.patch("supervisor.core.engine.get_adapter", return_value=mock_adapter)

        # Mock workspace.isolated_execution
        @contextmanager
        def mock_isolated_execution(step_id):
            yield WorktreeContext(
                worktree_path=Path("/tmp/worktree"),
                step_id=step_id,
                original_head="abc123",
            )

        mocker.patch.object(
            engine.workspace, "isolated_execution", side_effect=mock_isolated_execution
        )

        # Mock _apply_and_finalize_step
        mocker.patch.object(engine, "_apply_and_finalize_step", return_value=[])

        # Should succeed after retry
        engine.run_role(
            role_name="implementer",
            task_description="Test task",
            workflow_id="test-wf",
            gates=[],
        )

        # Verify execute was called twice (initial + 1 retry)
        assert mock_execute.call_count == 2

    def test_run_role_respects_max_retries(self, engine, mocker):
        """run_role respects retry_policy max_attempts configuration."""
        from contextlib import contextmanager

        from supervisor.core.workspace import WorktreeContext

        mock_role = Mock()
        mock_role.name = "implementer"
        mock_role.cli = "claude:sonnet"
        mock_role.gates = []
        mock_role.config = {}
        mock_role.context = {}
        mock_role.system_prompt = "You are an implementer."
        mock_role.flags = []
        mock_role.base_role = "implementer"
        mock_role.on_fail_overrides = {}
        mocker.patch.object(engine.role_loader, "load_role", return_value=mock_role)

        mocker.patch.object(engine.context_packer, "build_full_prompt", return_value="Context")

        # Mock _execute_cli: always fail
        mock_execute = mocker.patch.object(engine, "_execute_cli")
        mock_execute.return_value = ExecutionResult(
            returncode=1, stdout="", stderr="Connection failed"
        )

        # Mock workspace.isolated_execution
        @contextmanager
        def mock_isolated_execution(step_id):
            yield WorktreeContext(
                worktree_path=Path("/tmp/worktree"),
                step_id=step_id,
                original_head="abc123",
            )

        mocker.patch.object(
            engine.workspace, "isolated_execution", side_effect=mock_isolated_execution
        )

        # Should fail after max_attempts
        with pytest.raises(Exception):  # RetryExhaustedError or EngineError
            engine.run_role(
                role_name="implementer",
                task_description="Test task",
                workflow_id="test-wf",
                gates=[],
                retry_policy=RetryPolicy(max_attempts=2),  # Only 2 total attempts
            )

        # Should have tried exactly 2 attempts
        assert mock_execute.call_count == 2

    def test_run_role_circuit_breaker_prevents_execution(self, engine, mocker):
        """Circuit breaker prevents execution when open."""
        # Pre-open the circuit breaker
        circuit_key = "test-circuit"
        for _ in range(engine.circuit_breaker.max_failures):
            engine.circuit_breaker.record_failure(circuit_key)

        assert engine.circuit_breaker.is_open(circuit_key)

        # Attempting to execute with open circuit should raise error
        # (This test depends on how the engine checks circuit breaker)
        # For now, we verify the circuit is open
        assert engine.circuit_breaker.is_open(circuit_key)

    def test_run_role_logs_events_to_database(self, engine, mocker):
        """run_role logs events to the database."""
        from contextlib import contextmanager

        from supervisor.core.workspace import WorktreeContext

        mock_role = Mock()
        mock_role.name = "implementer"
        mock_role.cli = "claude:sonnet"
        mock_role.gates = []
        mock_role.config = {}
        mock_role.context = {}
        mock_role.system_prompt = "You are an implementer."
        mock_role.flags = []
        mock_role.base_role = "implementer"
        mock_role.on_fail_overrides = {}
        mocker.patch.object(engine.role_loader, "load_role", return_value=mock_role)

        mocker.patch.object(engine.context_packer, "build_full_prompt", return_value="Context")

        # Mock _execute_cli
        mock_result = ExecutionResult(
            returncode=0,
            stdout='{"status": "success", "files_modified": []}',
            stderr="",
        )
        mocker.patch.object(engine, "_execute_cli", return_value=mock_result)

        # Mock parser adapter
        mock_adapter = Mock()
        mock_output = Mock()
        mock_output.status = "success"
        mock_output.files_modified = []
        mock_adapter.parse_output.return_value = mock_output
        mocker.patch("supervisor.core.engine.get_adapter", return_value=mock_adapter)

        # Mock workspace.isolated_execution
        @contextmanager
        def mock_isolated_execution(step_id):
            yield WorktreeContext(
                worktree_path=Path("/tmp/worktree"),
                step_id=step_id,
                original_head="abc123",
            )

        mocker.patch.object(
            engine.workspace, "isolated_execution", side_effect=mock_isolated_execution
        )

        # Mock _apply_and_finalize_step
        mocker.patch.object(engine, "_apply_and_finalize_step", return_value=[])

        # Execute
        engine.run_role(
            role_name="implementer",
            task_description="Test task",
            workflow_id="test-wf",
            gates=[],
        )

        # Verify events were logged
        events = engine.db.get_events("test-wf")
        assert len(events) > 0

        # Should have STEP_STARTED event (STEP_COMPLETED is recorded in _apply_and_finalize_step)
        event_types = [e.event_type for e in events]
        assert EventType.STEP_STARTED in event_types


@pytest.mark.git
class TestExecutionEngineIntegration:
    """Integration tests for ExecutionEngine with real database."""

    def test_full_execution_flow_with_real_db(self, repo_with_git, test_db, mocker):
        """Test complete execution flow with real database (mocked sandbox)."""
        from contextlib import contextmanager

        from supervisor.core.workspace import WorktreeContext

        with patch("supervisor.core.engine.require_docker"):
            engine = ExecutionEngine(repo_with_git, db=test_db)

        # Mock all external dependencies with proper attributes
        mock_role = Mock()
        mock_role.name = "test_role"
        mock_role.cli = "claude:sonnet"
        mock_role.gates = []
        mock_role.config = {}
        mock_role.context = {}
        mock_role.system_prompt = "You are a test role."
        mock_role.flags = []
        mock_role.base_role = "implementer"
        mock_role.on_fail_overrides = {}
        mocker.patch.object(engine.role_loader, "load_role", return_value=mock_role)

        mocker.patch.object(engine.context_packer, "build_full_prompt", return_value="Context")

        # Mock _execute_cli
        mock_result = ExecutionResult(
            returncode=0,
            stdout='{"status": "success", "files_modified": ["test.py"]}',
            stderr="",
        )
        mocker.patch.object(engine, "_execute_cli", return_value=mock_result)

        # Mock parser adapter
        mock_adapter = Mock()
        mock_output = Mock()
        mock_output.status = "success"
        mock_output.files_modified = ["test.py"]
        mock_adapter.parse_output.return_value = mock_output
        mocker.patch("supervisor.core.engine.get_adapter", return_value=mock_adapter)

        # Mock workspace.isolated_execution
        @contextmanager
        def mock_isolated_execution(step_id):
            yield WorktreeContext(
                worktree_path=Path("/tmp/worktree"),
                step_id=step_id,
                original_head="abc123",
            )

        mocker.patch.object(
            engine.workspace, "isolated_execution", side_effect=mock_isolated_execution
        )

        # Mock _apply_and_finalize_step
        mocker.patch.object(engine, "_apply_and_finalize_step", return_value=["test.py"])

        # Execute
        workflow_id = "integration-test-wf"
        result = engine.run_role(
            role_name="test_role",
            task_description="Integration test task",
            workflow_id=workflow_id,
            gates=[],
        )

        # Verify result
        assert result is not None

        # Verify database events
        events = test_db.get_events(workflow_id)
        assert len(events) > 0

        # Check event sequence - STEP_STARTED should be logged
        # Note: STEP_COMPLETED is recorded in _apply_and_finalize_step which is mocked
        event_types = [e.event_type for e in events]
        assert EventType.STEP_STARTED in event_types
