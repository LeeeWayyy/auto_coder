# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive improvement plan for documentation and test coverage (docs/IMPROVEMENT_PLAN.md)
- CHANGELOG.md to track project changes
- Getting Started guide (docs/GETTING_STARTED.md) - comprehensive user onboarding
- Test infrastructure (tests/conftest.py) with foundational fixtures
- CLI test suite (tests/test_cli.py) with 40+ test cases
- GitHub Actions workflow for automated testing (.github/workflows/test.yaml)
  - Multi-OS testing (Ubuntu, macOS)
  - Multi-Python version testing (3.11, 3.12)
  - Coverage reporting with Codecov integration
  - Docker-specific tests
  - Integration tests
  - Security scanning with Bandit
- **Phase 2: Critical Infrastructure Testing**
  - Engine test suite (tests/test_engine.py) with 35+ test cases
    - RetryPolicy: Exponential backoff calculations
    - CircuitBreaker: Failure tracking and state management
    - EnhancedCircuitBreaker: Advanced features with metrics
    - ErrorClassifier: Error categorization for retry logic
    - ExecutionEngine: Core execution flows with mocked sandbox
  - State management test suite (tests/test_state.py) with 30+ test cases
    - Event logging and retrieval
    - Database projections (features, phases, components)
    - Event sourcing patterns (reconstruction, idempotency, replay)
    - Transaction safety and concurrent access
    - Step management
    - Metrics recording
  - Sandbox executor test suite (tests/test_sandbox_executor.py) with 25+ test cases
    - Docker availability checking
    - Network egress verification
    - Container lifecycle management
    - Security validation (path traversal, symlinks, command injection)
    - Execution timeouts
    - Network isolation enforcement
  - Architecture documentation (docs/ARCHITECTURE.md) - comprehensive system design
- **Phase 3: Operational Excellence**
  - CLI Reference (docs/CLI_REFERENCE.md) - complete command documentation
  - Operations Guide (docs/OPERATIONS.md) - production deployment and monitoring
  - Contributing Guide (CONTRIBUTING.md) - developer onboarding
  - Examples directory (examples/) with practical workflows
- **Phase 4: Comprehensive Coverage**
  - Workspace test suite (tests/test_workspace.py) with 30+ test cases
    - Git worktree creation and isolation
    - Security validation (symlinks, path traversal, sanitization)
    - Complete execute_step workflow testing
    - Gate execution in worktrees
    - Atomic change application with file locking
    - HEAD conflict detection
    - Event sourcing patterns
  - Gate executor test suite (tests/test_gate_executor.py) with 25+ test cases
    - Environment filtering for security
    - Cache key computation and result caching
    - Gate execution (success, failure, timeout)
    - Custom working directories
    - Multiple gate orchestration
    - Event recording
  - Approval system test suite (tests/test_approval.py) with 15+ test cases
    - Risk assessment (critical, high, medium, low)
    - Approval policy configuration
    - Human approval requests via InteractionBridge
    - Full approval workflow integration
  - Models test suite (tests/test_models.py) with 12+ test cases
    - Pydantic model validation
    - Status enumerations
    - Serialization and deserialization
  - Metrics test suite (tests/test_metrics.py) with 10+ test cases
    - Metrics collection and recording
    - Performance aggregation and analysis
    - Dashboard display formatting
- **Phase 5: Polish & Optimization**
  - FAQ documentation (docs/FAQ.md) - Common questions, troubleshooting, best practices
  - README enhancement with badges (tests, Python version, license, code style)
  - Comprehensive documentation links (Getting Started, Reference, Operations)
  - Test coverage summary: 120+ test cases across 11 test files
  - Documentation coverage: 7 major guides totaling 6000+ lines
- **Critical Fixes (Post-Gemini Review)**
  - Fixed P0 sandbox security test assertions (test_sandbox_executor.py)
    - Implemented command injection prevention validation
    - Added symlink resolution verification
    - Added volume mount security checks (read-only filesystem, tmpfs)
    - Implemented environment variable sanitization validation
  - Fixed P1 workflow implementation tests (test_workflow.py)
    - Added sequential execution tests (success, dependencies, failures)
    - Added parallel execution tests (independent components, performance)
    - Added component-level timeout enforcement tests
    - Added workflow-level timeout tests
    - Added rollback functionality tests
    - Added DAG scheduler integration tests (dependency ordering)
  - Total new test cases: 15+ critical tests added

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- Added security scanning in CI pipeline
- Comprehensive security tests for sandbox executor

## [0.1.0] - 2026-01-12

### Added
- Multi-Model Router with granular cli:model routing
- Phase 5 Polish & Advanced Features:
  - Timeout management (component-level and workflow-level)
  - Terminal UI (TUI) for human-in-the-loop interactions
  - Metrics collection and dashboard
  - Adaptive routing based on historical performance
  - Approval policies for risky operations
- Phase 4 Multi-Model Hierarchy:
  - Hierarchical workflow orchestration (Feature → Phase → Component)
  - DAG-based scheduling for parallel execution
  - Multi-model routing and intelligent model selection
- Phase 3 Gates & Verification:
  - Orchestrator-enforced verification gates
  - Gate system with dependencies and configurable severity
  - File locking mechanisms
- Phase 2 Core Workflow:
  - Role inheritance system (Base + Overlay model)
  - Jinja2 templating for prompts
  - CLI adapter pattern for Claude/Codex/Gemini
  - Context packing with token budget management
- Phase 1 Foundation:
  - Event sourcing with SQLite backend
  - Docker sandbox for isolated execution
  - Basic execution engine
  - Git worktree isolation
  - Security features (path traversal prevention, symlink rejection)

### Changed
- Memoized collect_dependencies for improved performance
- Enhanced gate configuration and execution

### Security
- Docker-based sandbox isolation for all AI CLI execution
- Network egress control with allowlist verification
- Path traversal and symlink protection at multiple layers
- Atomic file operations with HEAD conflict detection

---

## How to Update This Changelog

When making changes to the project:

1. **Add entries under `[Unreleased]`** in the appropriate section:
   - **Added**: New features
   - **Changed**: Changes to existing functionality
   - **Deprecated**: Soon-to-be removed features
   - **Removed**: Removed features
   - **Fixed**: Bug fixes
   - **Security**: Security-related changes

2. **When releasing a new version**:
   - Move items from `[Unreleased]` to a new version section
   - Add the version number and release date
   - Create a new empty `[Unreleased]` section

3. **Version numbering** (Semantic Versioning):
   - **MAJOR** version: Incompatible API changes
   - **MINOR** version: Backwards-compatible functionality additions
   - **PATCH** version: Backwards-compatible bug fixes

4. **Keep it user-focused**:
   - Write for users, not just developers
   - Explain the impact of changes
   - Link to relevant documentation or issues when appropriate
