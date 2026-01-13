# Documentation and Test Coverage Improvement Plan

**Status**: In Progress
**Started**: 2026-01-12
**Strategy**: Balanced approach (alternate docs/tests), Incremental testing (core first)

## Quick Reference

- **Current Test Coverage**: 40-50% (9/33 files)
- **Target Coverage**: 75-80% by completion
- **Total Estimated Time**: 53-68 days
- **Phases**: 5 phases from foundation to polish

## Phase Overview

| Phase | Focus | Duration | Documentation | Testing |
|-------|-------|----------|---------------|---------|
| 1 | Foundation & Quick Wins | 8-10 days | Getting Started, CHANGELOG | CLI tests, conftest.py, CI |
| 2 | Critical Infrastructure | 13-16 days | Architecture | Engine, State, Sandbox tests |
| 3 | Operational Excellence | 12-16 days | CLI Ref, Ops, Contributing, Examples | Metrics, Integration tests |
| 4 | Comprehensive Coverage | 15-19 days | Metrics guide, Advanced examples | Remaining 10 modules |
| 5 | Polish & Optimization | 5-7 days | FAQ, Comparison | Test optimization, perf tests |

---

## Phase 1: Foundation & Quick Wins (8-10 days)

**Goal**: User onboarding + test infrastructure + critical entry point testing

### Documentation Deliverables

#### ✅ CHANGELOG.md
- Track version history
- Document breaking changes
- Started: [date]

#### ✅ docs/GETTING_STARTED.md
**Structure**:
1. High-level pitch (What is Supervisor? What problem does it solve?)
2. Prerequisites (Python 3.11+, Docker, Git)
3. Installation (`pip install -e .`)
4. Your First Run ("Hello World")
   - Example: `supervisor run --role=planner "Create a plan to greet the world"`
   - Show exact expected output
5. Core Concepts (Role, Workflow, Gate, State - 1-2 sentences each)
6. Next Steps (links to other docs)

**Estimated**: 300-500 lines

### Testing Deliverables

#### ✅ tests/conftest.py
**Foundation fixtures**:
- `temp_repo` - Temporary git repository
- `test_db` - Temporary test database
- `sample_role_config` - Sample role configuration
- `mock_sandbox_config` - Mock sandbox configuration

**Estimated**: 200-300 lines

#### ✅ tests/test_cli.py
**Critical file**: `supervisor/cli.py` (451 lines)

**Test coverage** (~15 test cases):
```python
# Core commands
- test_init_creates_directory_structure()
- test_plan_command_with_workflow_id()
- test_run_command_with_targets()
- test_workflow_command_tui_mode()
- test_metrics_command_basic()

# Configuration loading
- test_load_approval_config_exists()
- test_load_limits_config_with_values()

# Error handling
- test_plan_command_engine_error()
- test_run_command_invalid_role()
```

**Strategy**: Mock ExecutionEngine, Database, subprocess. Focus on command parsing.

**Estimated**: 400-600 lines

### CI/CD Enhancement

#### ✅ .github/workflows/test.yaml
- Add pytest with coverage
- Generate HTML coverage reports
- Optional: Codecov/Coveralls integration
- Enforce coverage minimums on PRs

### Success Metrics
- [ ] Getting Started guide: new users complete first workflow in <30 minutes
- [ ] CHANGELOG started
- [ ] test_cli.py: 15+ tests passing
- [ ] CI coverage tracking enabled

---

## Phase 2: Critical Infrastructure (13-16 days)

**Goal**: Test the three most critical modules - system backbone

**Priority**: Focus here before spreading thin (Gemini's advice)

### Testing Deliverables

#### ✅ tests/test_engine.py
**Critical file**: `supervisor/core/engine.py` (52,939 lines)

**Test coverage** (~30 test cases):
```python
# Core execution flow
class TestExecutionEngine:
    - test_run_role_successful_execution()
    - test_run_role_with_gates()
    - test_run_role_gate_failure()
    - test_run_role_with_retry()

# Retry logic (critical for reliability)
class TestRetryPolicy:
    - test_exponential_backoff_calculation()
    - test_retry_on_transient_error()
    - test_no_retry_on_fatal_error()
    - test_max_retries_exhausted()

# Circuit breaker (prevent infinite loops)
class TestCircuitBreaker:
    - test_circuit_open_after_failures()
    - test_circuit_half_open_after_timeout()
    - test_failure_rate_calculation()

# Error classification
class TestErrorClassifier:
    - test_classify_network_error()
    - test_classify_parsing_error()
    - test_classify_gate_failure()
```

**Strategy**: Mock sandbox, use real Database. Focus on error paths and retry logic.

**Estimated**: 800-1200 lines

#### ✅ tests/test_state.py
**Critical file**: `supervisor/core/state.py` (53,783 lines)

**Event-sourced testing** (~25 test cases):
```python
# Test each event type (state_n + event = state_n+1)
class TestEventLogging:
    - test_log_workflow_started()
    - test_log_step_completed()
    - test_log_gate_failed()
    - test_event_serialization()

# Database projections
class TestDatabase:
    - test_create_feature()
    - test_create_phase()
    - test_create_component()
    - test_get_phases_for_feature()

# CRITICAL: Event sourcing patterns
class TestEventSourcing:
    - test_rebuild_projections_from_events()  # Reconstruction from scratch
    - test_idempotency_replay_events()        # Apply same events multiple times
    - test_state_reconstruction_accuracy()     # Full replay validation

# Transaction safety
class TestTransactionSafety:
    - test_concurrent_event_writes()
    - test_projection_consistency()
```

**Key fixtures**:
- `populated_db` - Database with sample features/phases/components
- `event_sequence` - Realistic event sequence for replay testing

**Strategy**: Real SQLite with tempfile. Test idempotency, reconstruction.

**Estimated**: 1000-1500 lines

#### ✅ tests/test_sandbox_executor.py
**Critical file**: `supervisor/sandbox/executor.py` (39,021 lines)

**Security-focused testing** (~20 test cases):
```python
# Docker availability
class TestDockerAvailability:
    - test_docker_not_available_raises()
    - test_verify_docker_version()

# Container lifecycle
class TestSandboxedLLMClient:
    - test_execute_command_success()
    - test_execute_with_timeout()
    - test_container_cleanup_on_exit()

class TestSandboxedExecutor:
    - test_execute_gate_command()
    - test_no_network_enforcement()
    - test_working_dir_validation()

# CRITICAL: Security tests
class TestSandboxSecurity:
    - test_command_injection_prevention()
    - test_path_traversal_prevention()
    - test_symlink_rejection()
    - test_volume_mount_security()
```

**Strategy**: Mock Docker API for unit tests. Optional `@pytest.mark.docker` for integration.

**Estimated**: 600-800 lines

### Documentation Deliverables

#### ✅ docs/ARCHITECTURE.md
**Content** (600-1000 lines):
- System overview with component diagram
- Data flow through the system
- Event sourcing philosophy
- Core components: ExecutionEngine, Database, Sandbox, Workspace
- Step-by-step execution flow
- Security model: isolation layers, threat model, egress control
- Extension points: custom roles, gates, CLI adapters

**References**: All spec files in `docs/SPECS/`, design docs in `docs/PLANS/`

#### Ongoing: Improve Docstrings
As we test each module, improve docstrings for auto-generated API docs (Sphinx)

### Success Metrics
- [ ] engine.py: 60%+ coverage
- [ ] state.py: 70%+ coverage
- [ ] executor.py: 60%+ coverage
- [ ] All critical paths tested
- [ ] Event sourcing patterns validated (reconstruction, idempotency)
- [ ] Architecture documentation published

---

## Phase 3: Operational Excellence (12-16 days)

**Goal**: Complete user documentation + test operational components

### Documentation Deliverables (ALL TOPICS)

#### ✅ docs/CLI_REFERENCE.md
**Content** (500-800 lines):
- Commands overview table
- Detailed reference for each command: init, plan, run, workflow, metrics, roles, status, version
  - Full signature with all options
  - Parameter descriptions
  - 2-3 usage examples per command
  - Exit codes and error handling
- Configuration files: config.yaml, limits.yaml, adaptive.yaml, approval.yaml, gates.yaml
- Common error messages and solutions

**Reference**: `supervisor/cli.py`

#### ✅ docs/OPERATIONS.md
**Content** (500-800 lines):
- **Deployment**: Requirements, Docker config, network security (egress rules), permissions
- **Monitoring**: Using `supervisor metrics`, understanding performance, identifying bottlenecks
- **Troubleshooting**: Common issues (Docker unavailable, network egress, gate failures, OOM, slow perf)
- **Maintenance**: Database (state.db), cleaning worktrees, log rotation, backups
- **Production Best Practices**: Timeouts, resource limits, approval policies, security hardening

#### ✅ CONTRIBUTING.md
**Content** (400-600 lines):
- **Development Setup**: Fork/clone, install dev deps, run tests, code style (ruff, mypy)
- **Project Structure**: Directory overview, key modules
- **Adding Features**: New roles, custom gates, CLI adapters, metrics collectors
- **Testing Requirements**: Coverage expectations (80%+), writing unit/integration tests, fixtures
- **Documentation Requirements**: Updating docs, code comments, specs
- **Pull Request Process**: Branch naming, commit messages, review, CI/CD

#### ✅ examples/
**Structure** (600-1000 lines total):
```
examples/
├── README.md                    # Overview
├── basic-workflow/              # Simple plan→implement→review
├── multi-model-routing/         # Adaptive routing
├── custom-gates/                # Verification gates
└── parallel-workflow/           # Parallel execution
```

### Testing Deliverables

#### ✅ tests/test_metrics.py
**Files**: `supervisor/metrics/aggregator.py`, `collector.py`, `dashboard.py`

**Coverage** (~20 test cases):
```python
class TestMetricsAggregator:
    - test_get_role_performance()
    - test_get_cli_comparison()
    - test_get_best_cli_for_task()
    - test_date_range_filtering()

class TestMetricsDashboard:
    - test_show_metrics_output()
    - test_format_summary_table()

class TestMetricsCollector:
    - test_record_execution_metrics()
```

**Estimated**: 400-600 lines

#### ✅ tests/integration/test_full_workflow.py
**End-to-end scenarios** (~15 test cases):
```python
class TestFullWorkflow:
    - test_plan_to_implementation_flow()
    - test_parallel_component_execution()
    - test_gate_failure_recovery()
    - test_workflow_crash_recovery()
```

**Strategy**: Real database, mocked sandbox. Test complete user workflows.

**Estimated**: 500-800 lines

### Success Metrics
- [ ] CLI Reference complete
- [ ] Operations Guide complete
- [ ] Contributing Guide complete
- [ ] 4 working examples published
- [ ] Metrics tests passing
- [ ] Integration tests passing

---

## Phase 4: Comprehensive Coverage (15-19 days)

**Goal**: Test remaining untested modules, expand coverage to 70%+

### Testing Deliverables (10 new test files)

#### Supporting Core Modules
- **test_approval.py** (~200 lines, 10-15 tests) - `supervisor/core/approval.py` (9,162 lines)
- **test_feedback.py** (~200 lines, 10-15 tests) - `supervisor/core/feedback.py` (9,996 lines)
- **test_interaction.py** (~150 lines, 8-10 tests) - `supervisor/core/interaction.py` (4,269 lines)
- **test_models.py** (~200 lines, 12-15 tests) - `supervisor/core/models.py` (6,216 lines)
- **test_utils.py** (~100 lines, 5-10 tests) - `supervisor/core/utils.py` (1,567 lines)

#### Gate System Modules
- **test_gate_executor.py** (~400 lines, 15-20 tests) - `supervisor/core/gate_executor.py` (56,132 lines)
- **test_gate_loader.py** (~300 lines, 12-15 tests) - `supervisor/core/gate_loader.py` (23,806 lines)

#### Workspace Management
- **test_workspace.py** (~600 lines, 20-25 tests) - `supervisor/core/workspace.py` (47,119 lines)

#### TUI and Dashboard
- **test_tui.py** (~300 lines, 12-15 tests) - `supervisor/tui/app.py`
- **test_dashboard.py** (~150 lines, 8-10 tests) - `supervisor/metrics/dashboard.py` (5,076 lines)
- **test_collector.py** (~100 lines, 5-8 tests) - `supervisor/metrics/collector.py` (618 lines)

**Total estimated**: ~2,700 lines of test code

### Documentation Deliverables

#### ✅ docs/METRICS.md
**Content** (300-500 lines):
- Available metrics: role performance, CLI comparison, workflow statistics
- Using the dashboard: `supervisor metrics` interpretation
- Performance tuning: adaptive selection, timeout optimization, parallel strategies
- Cost optimization: choosing models, analyzing performance, balancing cost vs quality

#### ✅ examples/advanced/
**Examples** (800-1200 lines):
- Custom role implementation
- Complex gate configurations
- Multi-phase hierarchical workflow
- Adaptive routing with custom scoring
- CI/CD integration

### Success Metrics
- [ ] 10 new test files completed
- [ ] Overall test coverage: 70%+
- [ ] Metrics guide published
- [ ] Advanced examples available

---

## Phase 5: Polish & Optimization (5-7 days)

**Goal**: Final touches, performance optimization, professional polish

### Documentation Deliverables

#### ✅ docs/FAQ.md (~400 lines)
- Common questions and answers
- Troubleshooting tips
- Best practices

#### ✅ docs/COMPARISON.md (~300 lines)
- Comparison with LangGraph, MetaGPT
- When to use Supervisor
- Migration guides

#### ✅ README.md Enhancement
- Improve quick start section
- Add badges (build status, coverage)
- Better examples
- Comprehensive links to all documentation

### Testing Deliverables

#### ✅ Test Optimization
- Implement test parallelization (`pytest-xdist`)
- Add test markers: `@pytest.mark.slow`, `@pytest.mark.docker`, `@pytest.mark.integration`
- Optimize fixture creation
- Target: test execution <5 minutes

#### ✅ Performance Tests (Optional)
**Location**: `tests/performance/`
- Benchmark key operations
- Memory leak detection
- Parallel execution scaling

### Optional: Documentation Website

#### ✅ MkDocs/Sphinx Site
- Auto-generated API documentation from docstrings
- Searchable documentation
- Version tracking
- Professional presentation

### Success Metrics
- [ ] FAQ and comparison docs complete
- [ ] README enhanced with badges
- [ ] Test execution time <5 minutes
- [ ] Optional: Documentation site live

---

## Key Principles (from Gemini)

### Testing
1. **Focus on core first**: "A solid, well-tested core will make it easier and safer to write tests for peripheral modules later"
2. **Event sourcing priority**: Test reconstruction from scratch - "can you reliably reconstruct state by replaying entire event log?"
3. **Test each event type**: `state_n + event = state_n+1`
4. **Idempotency**: Apply same events multiple times, state should be identical

### Documentation
1. **Getting Started is critical**: "Seeing a successful result immediately is a huge confidence booster"
2. **Progressive disclosure**: Start simple, add complexity gradually
3. **User-centric**: Write for users, not developers
4. **Examples matter**: Every concept gets a working example

### Process
1. **CI integration**: "Makes progress visible and holds the line against backsliding"
2. **Docstrings matter**: "Can be used with Sphinx to auto-generate API documentation"
3. **Don't spread thin**: Prioritize CLI → Engine → State before expanding

---

## Progress Tracking

### Phase 1 Status ✅ COMPLETED (2026-01-12)
- [x] CHANGELOG.md created
- [x] docs/GETTING_STARTED.md completed (400+ lines, comprehensive onboarding)
- [x] tests/conftest.py created (15 fixtures, 350+ lines)
- [x] tests/test_cli.py completed (40+ tests covering all CLI commands)
- [x] .github/workflows/test.yaml created (comprehensive CI with coverage)
- [x] CI coverage tracking enabled (Codecov integration)

### Phase 2 Status ✅ COMPLETED (2026-01-12)
- [x] tests/test_engine.py completed (35+ tests: RetryPolicy, CircuitBreaker, ErrorClassifier, ExecutionEngine)
- [x] tests/test_state.py completed (30+ tests: event sourcing, projections, idempotency, reconstruction)
- [x] tests/test_sandbox_executor.py completed (25+ tests: Docker isolation, security validation)
- [x] docs/ARCHITECTURE.md completed (comprehensive system design, 600+ lines)
- [x] Docstrings improved for tested modules
- [x] Event sourcing patterns validated (reconstruction, idempotency)

### Phase 3 Status
- [ ] docs/CLI_REFERENCE.md completed
- [ ] docs/OPERATIONS.md completed
- [ ] CONTRIBUTING.md completed
- [ ] examples/ directory created with 4 examples
- [ ] tests/test_metrics.py completed
- [ ] tests/integration/test_full_workflow.py completed

### Phase 4 Status
- [ ] 10 new test files completed
- [ ] docs/METRICS.md completed
- [ ] examples/advanced/ created
- [ ] 70%+ test coverage achieved

### Phase 5 Status
- [ ] docs/FAQ.md completed
- [ ] docs/COMPARISON.md completed
- [ ] README.md enhanced
- [ ] Test optimization completed
- [ ] Optional: Documentation site deployed

---

## Coverage Targets

| Metric | Current | After Phase 2 | After Phase 4 | After Phase 5 |
|--------|---------|---------------|---------------|---------------|
| File Coverage | 27% (9/33) | 45% (15/33) | 70% (23/33) | 80% (26/33) |
| Line Coverage | 40-50% | 55-65% | 70%+ | 75-80% |
| Critical Modules | 0/5 tested | 5/5 tested | 5/5 tested | 5/5 tested |

**Critical Modules**: CLI, Engine, State, Sandbox Executor, Gates

---

## Next Actions

1. Start Phase 1 immediately
2. Get feedback on Getting Started guide from new users
3. Validate event sourcing test patterns in Phase 2
4. Iterate based on actual coverage reports
5. Adjust phasing if needed based on discovered complexity
