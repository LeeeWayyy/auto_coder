# CLI Reference

Complete reference for all Auto Coder (Supervisor CLI) commands, options, and configuration files. For step-by-step operational guidance, see the [Runbook](../runbook/README.md).

## Command Overview

| Command | Description |
|---------|-------------|
| `supervisor init` | Initialize project for Supervisor |
| `supervisor plan` | Run planner on a task |
| `supervisor run` | Execute a specific role |
| `supervisor workflow` | Run a feature workflow |
| `supervisor metrics` | View performance metrics |
| `supervisor roles` | List available roles |
| `supervisor status` | Show workflow status |
| `supervisor visualize` | Render a workflow graph in the terminal |
| `supervisor run-graph` | Execute a declarative workflow graph (YAML) |
| `supervisor inspect` | Inspect execution node details |
| `supervisor studio` | Launch the Studio web console |
| `supervisor version` | Show version information |

---

## Commands

### `supervisor init`

Initialize a project for Supervisor. Creates `.supervisor/` directory with configuration files and database.

**Usage**:
```bash
supervisor init
```

**Creates**:
```
.supervisor/
├── config.yaml          # Project configuration
├── limits.yaml          # Timeout configuration
├── adaptive.yaml        # Model selection rules
├── approval.yaml        # Approval policies
├── gates.yaml           # Verification gates (if customized)
├── roles/               # Custom role definitions
├── templates/           # Custom prompt templates
└── state.db             # SQLite database
```

**Exit Codes**:
- `0`: Success
- `1`: Error

**Examples**:
```bash
# Initialize in current directory
supervisor init

# Already initialized warning
supervisor init  # Shows: "Project already initialized"
```

---

### `supervisor plan`

Run the planner role on a task description. Breaks features into phases and components.

**Usage**:
```bash
supervisor plan TASK [OPTIONS]
```

**Arguments**:
- `TASK` (required): Task description

**Options**:
- `-w, --workflow-id TEXT`: Custom workflow ID (auto-generated if not provided)
- `--dry-run`: Show what would be done without executing

**Exit Codes**:
- `0`: Success
- `1`: Execution error

**Examples**:
```bash
# Plan a feature
supervisor plan "Add user authentication with JWT"

# Plan with custom workflow ID
supervisor plan "Add feature X" --workflow-id feat-auth-001

# Dry run (show plan without executing)
supervisor plan "Add feature X" --dry-run
```

**Output**:
```
Planning task: Add user authentication with JWT
Workflow ID: wf-abc12345

✓ Created worktree: /repo/.worktrees/wf-abc12345-step-001
✓ Packed context: 2.5k tokens
✓ Running planner role...

Feature Plan:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1: Implementation
  Component: jwt_utils
    - Create src/utils/jwt.py
    - Implement token generation/validation

Phase 2: Testing
  Component: auth_tests
    - Create tests/test_auth.py
    - Test token lifecycle

✓ Plan created: feat-abc12345
```

---

### `supervisor run`

Execute a specific role on a task.

**Usage**:
```bash
supervisor run ROLE TASK [OPTIONS]
```

**Arguments**:
- `ROLE` (required): Role name (planner, implementer, reviewer, or custom)
- `TASK` (required): Task description

**Options**:
- `-w, --workflow-id TEXT`: Workflow ID
- `-t, --target FILE`: Target files to focus on (can be used multiple times)

**Exit Codes**:
- `0`: Success
- `1`: Execution error
- `2`: Gate failure

**Examples**:
```bash
# Run implementer role
supervisor run implementer "Add login endpoint"

# Target specific files
supervisor run implementer "Fix bug in auth" -t src/api/auth.py

# Multiple targets
supervisor run implementer "Refactor auth" \
  -t src/api/auth.py \
  -t src/models/user.py

# Run with custom workflow ID
supervisor run reviewer "Review PR changes" -w wf-review-123
```

**Output**:
```
Running role: implementer
Task: Add login endpoint
Workflow ID: wf-abc12345

✓ Created worktree: /repo/.worktrees/wf-abc12345-step-002
✓ Packed context: 18.2k tokens
✓ Running implementer role...
✓ Code generated: src/api/auth.py (67 lines)

Running verification gates...
  ✓ lint passed (0.8s)
  ✓ type_check passed (1.2s)
  ✓ test passed (3.4s)

✓ Changes applied to repository
✓ Cleaned up worktree

Role completed!

Files modified:
  - src/api/auth.py

Files created:
  - tests/test_auth.py
```

---

### `supervisor workflow`

Execute a complete feature workflow (planning → implementation → review).

**Usage**:
```bash
supervisor workflow FEATURE_ID [OPTIONS]
```

**Arguments**:
- `FEATURE_ID` (required): Feature ID from planning phase

**Options**:
- `--tui`: Run with interactive terminal UI
- `--parallel` / `--sequential`: Parallel or sequential component execution (default: parallel)
- `--timeout SECONDS`: Workflow timeout in seconds (default: from limits.yaml or 3600)

**Exit Codes**:
- `0`: Success
- `1`: Execution error
- `2`: Timeout
- `3`: User cancelled (in TUI mode)

**Examples**:
```bash
# Run workflow with default settings
supervisor workflow feat-abc12345

# Run with interactive TUI
supervisor workflow feat-abc12345 --tui

# Sequential execution (no parallelism)
supervisor workflow feat-abc12345 --sequential

# Custom timeout (2 hours)
supervisor workflow feat-abc12345 --timeout 7200

# Combined options
supervisor workflow feat-abc12345 --tui --parallel --timeout 3600
```

**TUI Mode Output**:
```
┌─────────────────────────────────────────────────────┐
│ Feature: User Authentication (feat-abc12345)        │
├─────────────────────────────────────────────────────┤
│                                                     │
│ ✓ Phase 1: Implementation                          │
│   ✓ Component: jwt_utils (2.3s)                    │
│   ✓ Component: login_endpoint (4.1s)               │
│   ● Component: password_hashing (in progress...)   │
│                                                     │
│ ○ Phase 2: Integration                             │
│   ○ Component: integration_tests (pending)         │
│                                                     │
│ ○ Phase 3: Review                                  │
│   ○ Component: code_review (pending)               │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Progress: 3/6 components (50%)                     │
│ Elapsed: 6.4s / 3600s                              │
└─────────────────────────────────────────────────────┘
```

---

### `supervisor metrics`

View performance metrics and statistics.

**Usage**:
```bash
supervisor metrics [OPTIONS]
```

**Options**:
- `--days INTEGER`: Number of days to analyze (default: 30)
- `--live`: Live updating display (not yet implemented)

**Exit Codes**:
- `0`: Success
- `1`: Database not found or error

**Examples**:
```bash
# View last 30 days
supervisor metrics

# Last 7 days
supervisor metrics --days 7

# Last year
supervisor metrics --days 365
```

**Output**:
```
Performance Metrics (Last 7 Days)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Summary
┌────────────────────┬─────────┐
│ Metric             │ Value   │
├────────────────────┼─────────┤
│ Total Executions   │ 47      │
│ Success Rate       │ 91.5%   │
│ Avg Duration       │ 4.2s    │
│ Total Retries      │ 8       │
└────────────────────┴─────────┘

Role Performance
┌─────────────┬────────┬──────────┬──────────┐
│ Role        │ Runs   │ Success  │ Avg Time │
├─────────────┼────────┼──────────┼──────────┤
│ planner     │ 12     │ 100.0%   │ 3.1s     │
│ implementer │ 28     │ 89.3%    │ 4.8s     │
│ reviewer    │ 7      │ 85.7%    │ 3.5s     │
└─────────────┴────────┴──────────┴──────────┘

CLI Comparison
┌──────────────┬────────┬──────────┬──────────┐
│ CLI          │ Runs   │ Success  │ Avg Time │
├──────────────┼────────┼──────────┼──────────┤
│ claude:opus  │ 15     │ 93.3%    │ 5.2s     │
│ claude:sonnet│ 24     │ 91.7%    │ 3.8s     │
│ claude:haiku │ 8      │ 87.5%    │ 2.1s     │
└──────────────┴────────┴──────────┴──────────┘
```

---

### `supervisor roles`

List all available roles (base roles + custom roles).

**Usage**:
```bash
supervisor roles
```

**Exit Codes**:
- `0`: Success

**Examples**:
```bash
supervisor roles
```

**Output**:
```
Available Roles
┌────────────────────┬─────────────────────────────┬──────────────┐
│ Role               │ Description                  │ CLI          │
├────────────────────┼─────────────────────────────┼──────────────┤
│ planner            │ Break features into phases   │ claude       │
│ implementer        │ Code implementation with TDD │ claude       │
│ reviewer           │ Code review for quality      │ claude:opus  │
│ security_reviewer  │ Security-focused review      │ claude:opus  │
│ frontend_impl      │ Frontend specialist          │ claude:sonnet│
└────────────────────┴─────────────────────────────┴──────────────┘
```

---

### `supervisor status`

Show workflow execution status.

**Usage**:
```bash
supervisor status [OPTIONS]
```

**Options**:
- `-w, --workflow-id TEXT`: Filter by workflow ID
- `-e, --execution-id TEXT`: Show a specific graph execution

**Exit Codes**:
- `0`: Success
- `1`: Database not found

**Examples**:
```bash
# Show recent executions (optionally filtered by workflow id)
supervisor status
supervisor status --workflow-id wf-abc12345

# Show a specific execution
supervisor status --execution-id exec_1234
```

---

### `supervisor visualize`

Render a declarative workflow graph in the terminal.

**Usage**:
```bash
supervisor visualize WORKFLOW_FILE
```

**Examples**:
```bash
supervisor visualize examples/workflows/basic_workflow.yaml
```

---

### `supervisor run-graph`

Execute a declarative workflow graph (YAML).

**Usage**:
```bash
supervisor run-graph WORKFLOW_FILE --workflow-id ID [OPTIONS]
```

**Options**:
- `--workflow-id TEXT` (required): Execution label
- `--validate-only`: Validate graph and exit
- `--live`: Show live execution monitor

**Examples**:
```bash
# Validate only
supervisor run-graph examples/workflows/basic_workflow.yaml --workflow-id wf-1234 --validate-only

# Execute with live monitor
supervisor run-graph examples/workflows/basic_workflow.yaml --workflow-id wf-1234 --live
```

---

### `supervisor inspect`

Inspect a graph execution and node details.

**Usage**:
```bash
supervisor inspect EXECUTION_ID [OPTIONS]
```

**Options**:
- `-n, --node TEXT`: Inspect a specific node ID
- `-i, --interactive`: Interactive inspection mode

**Examples**:
```bash
# Summary view
supervisor inspect exec_1234

# Node-specific
supervisor inspect exec_1234 --node node_1

# Interactive
supervisor inspect exec_1234 --interactive
```

---

### `supervisor studio`

Launch the Supervisor Studio web console.

**Usage**:
```bash
supervisor studio [OPTIONS]
```

**Options**:
- `--host TEXT`: Host to bind to (default: 127.0.0.1)
- `--port INTEGER`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development

**Examples**:
```bash
supervisor studio
supervisor studio --port 8000
supervisor studio --reload
```

---

### `supervisor version`

Show version information.

**Usage**:
```bash
supervisor version
```

**Examples**:
```bash
supervisor version
```

**Output**:
```
Supervisor v0.1.0
AI CLI Orchestrator
```

---

## Configuration Files

### `config.yaml`

Main project configuration file.

**Location**: `.supervisor/config.yaml`

**Format**:
```yaml
# Default CLI to use
default_cli: claude

# Workflow settings
workflow:
  parallel_execution: true   # Run independent components in parallel
  require_tests: true        # Require tests to pass
  human_approval: true       # Ask for approval on risky operations

# Role overrides for this project
roles:
  implementer:
    cli: claude:sonnet  # Use Sonnet for implementation (faster/cheaper)
  reviewer:
    cli: claude:opus    # Use Opus for review (higher quality)

# Git settings
git:
  auto_branch: true
  branch_pattern: "ai/{task_id}-{description}"
```

**Options**:
- `default_cli`: Default AI CLI to use (`claude`, `codex`, `gemini`)
- `workflow.parallel_execution`: Enable parallel component execution
- `workflow.require_tests`: Require tests to pass before applying changes
- `workflow.human_approval`: Require human approval for risky operations
- `roles`: Override role configurations for this project
- `git.auto_branch`: Automatically create branches for workflows
- `git.branch_pattern`: Branch naming pattern

---

### `limits.yaml`

Timeout and resource limit configuration.

**Location**: `.supervisor/limits.yaml`

**Format**:
```yaml
# Timeout configuration (seconds)
workflow_timeout: 3600    # Total workflow timeout (1 hour)
component_timeout: 300    # Per-component timeout (5 minutes)

# Role-specific timeouts
role_timeouts:
  planner: 600           # 10 minutes for planning
  implementer: 300       # 5 minutes for implementation
  reviewer: 180          # 3 minutes for review
```

**Options**:
- `workflow_timeout`: Maximum time for entire workflow (seconds)
- `component_timeout`: Maximum time per component (seconds)
- `role_timeouts`: Per-role timeout overrides (seconds)

---

### `adaptive.yaml`

Adaptive model selection configuration.

**Location**: `.supervisor/adaptive.yaml`

**Format**:
```yaml
# Adaptive model selection
adaptive:
  enabled: true
  min_samples_before_adapt: 10
  recalculation_interval: 10
  exploration_rate: 0.1

  # Score weights for model selection
  score_weights:
    success_rate: 0.6
    avg_duration: 0.4

  # Task-specific routing
  routing:
    architecture:
      - claude:opus
      - gemini:pro

    code_gen:
      - claude:sonnet
      - codex:gpt-4

    debugging:
      - claude:opus
      - gemini:pro

    refactoring:
      - claude:sonnet
      - gemini:pro

    testing:
      - claude:sonnet
      - codex:gpt-4

    security:
      - claude:opus

    documentation:
      - claude:sonnet
      - claude:haiku
```

**Options**:
- `adaptive.enabled`: Enable adaptive model selection
- `adaptive.min_samples_before_adapt`: Minimum samples before adapting
- `adaptive.recalculation_interval`: How often to recalculate scores
- `adaptive.exploration_rate`: Exploration vs exploitation (0-1)
- `adaptive.score_weights`: Weights for scoring models
- `adaptive.routing`: Task-type to model mapping

---

### `approval.yaml`

Approval policy configuration.

**Location**: `.supervisor/approval.yaml`

**Format**:
```yaml
# Approval policies
approval:
  auto_approve_low_risk: true
  risk_threshold: medium  # low, medium, high, critical

  # Always require approval
  require_approval_for:
    - deploy
    - commit
    - database_migration

  # Never require approval (fully automated)
  never_approve:
    - tests
    - lint
    - "*.md"  # Documentation

  # Risk-based approval
  risk_thresholds:
    low: 50      # Lines changed
    medium: 200
    high: 500
```

**Options**:
- `auto_approve_low_risk`: Automatically approve low-risk changes
- `risk_threshold`: Minimum risk level requiring approval
- `require_approval_for`: Patterns always requiring approval
- `never_approve`: Patterns never requiring approval
- `risk_thresholds`: Lines changed thresholds for risk levels

---

### `gates.yaml`

Custom gate configuration (optional - defaults are used if not present).

**Location**: `.supervisor/gates.yaml`

**Format**:
```yaml
gates:
  # Custom validation gate
  api_validate:
    command: ["python", "scripts/validate_api.py"]
    timeout: 60
    description: "Validate API contracts"
    severity: error
    fail_action: fail
    depends_on: [lint]

  # Database migration check
  migration_check:
    command: ["python", "manage.py", "makemigrations", "--check", "--dry-run"]
    timeout: 30
    description: "Check for missing migrations"
    severity: warning
    fail_action: warn

  #Override default test gate
  test:
    command: ["pytest", "-v", "--maxfail=3"]
    timeout: 600  # 10 minutes
    description: "Run test suite"
    severity: error
    fail_action: fail
    cache: true
```

**Gate Options**:
- `command`: Command to execute (list of strings)
- `timeout`: Timeout in seconds
- `description`: Human-readable description
- `severity`: `error` or `warning`
- `fail_action`: `fail`, `warn`, or `skip`
- `depends_on`: List of gate dependencies
- `cache`: Enable caching based on file content
- `parallel_safe`: Can run in parallel with other gates
- `allowed_writes`: Paths allowed for writing

**Built-in Gates**:
- `test`: Run pytest
- `lint`: Run ruff
- `type_check`: Run mypy
- `security`: Run bandit
- `format_check`: Check code formatting

---

## Environment Variables

Supervisor respects the following environment variables:

- `ANTHROPIC_API_KEY`: API key for Claude
- `OPENAI_API_KEY`: API key for Codex/GPT
- `GEMINI_API_KEY`: API key for Gemini
- `SUPERVISOR_LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `SUPERVISOR_DB_PATH`: Override database path
- `DOCKER_HOST`: Docker daemon socket (if not default)

---

## Exit Codes

Standard exit codes used by all commands:

- `0`: Success
- `1`: General error (execution failed, invalid arguments, etc.)
- `2`: Gate failure (tests or linting failed)
- `3`: User cancelled (in interactive mode)
- `4`: Timeout
- `130`: Interrupted (Ctrl+C)

---

## Common Patterns

### Feature Development Workflow

```bash
# 1. Initialize (once per project)
supervisor init

# 2. Plan the feature
supervisor plan "Add user authentication with JWT"
# Returns: feat-abc12345

# 3. Run the workflow
supervisor workflow feat-abc12345 --tui

# 4. Review metrics
supervisor metrics --days 1
```

### Targeted Implementation

```bash
# Fix a specific bug
supervisor run implementer "Fix null pointer in user profile" \
  -t src/models/user.py

# Add a new feature
supervisor run implementer "Add password reset functionality" \
  -t src/api/auth.py \
  -t src/email/templates.py
```

### Code Review

```bash
# Review recent changes
supervisor run reviewer "Review authentication implementation"

# Security-focused review
supervisor run security_reviewer "Review API endpoints for vulnerabilities" \
  -t src/api/
```

### Troubleshooting

```bash
# Check workflow status
supervisor status --workflow-id wf-abc12345

# View recent metrics
supervisor metrics --days 1

# List available roles
supervisor roles

# Re-run with dry-run to see what would happen
supervisor plan "Test task" --dry-run
```

---

## Tips and Best Practices

### 1. Use Target Files

Always specify target files when possible:
```bash
# Good - focused context
supervisor run implementer "Add feature" -t src/feature.py

# Less optimal - sends entire codebase
supervisor run implementer "Add feature"
```

### 2. Custom Workflow IDs

Use meaningful workflow IDs for tracking:
```bash
supervisor plan "Add auth" -w feat-authentication-001
```

### 3. Role-Specific CLIs

Configure different models for different roles:
```yaml
# config.yaml
roles:
  planner:
    cli: claude:opus      # High quality for planning
  implementer:
    cli: claude:sonnet    # Fast for implementation
  reviewer:
    cli: claude:opus      # High quality for review
```

### 4. Parallel Execution

Enable parallel execution for faster workflows:
```yaml
# config.yaml
workflow:
  parallel_execution: true
```

### 5. Custom Gates

Add project-specific verification:
```yaml
# gates.yaml
gates:
  openapi_validate:
    command: ["openapi-spec-validator", "api/openapi.yaml"]
    timeout: 30
    description: "Validate OpenAPI spec"
```

---

## Troubleshooting

### Command Not Found

```bash
$ supervisor: command not found
```

**Solution**: Install Supervisor:
```bash
pip install -e .
```

### Database Not Found

```bash
$ supervisor metrics
No supervisor database found. Run 'supervisor init' first.
```

**Solution**: Initialize the project:
```bash
supervisor init
```

### Docker Not Available

```bash
$ supervisor run implementer "Task"
Error: Docker is required but not available
```

**Solution**: Install and start Docker:
```bash
# macOS
brew install docker
open -a Docker

# Linux
sudo systemctl start docker
```

### Gate Failures

```bash
✗ test failed (exit code 1)
  Output: 3 tests failed
```

**Solution**: Fix the failing tests and retry, or check gate output for details.

### Timeout Errors

```bash
Error: Workflow timed out after 3600s
```

**Solution**: Increase timeout in limits.yaml:
```yaml
workflow_timeout: 7200  # 2 hours
```

---

## See Also

- [Runbook](../runbook/README.md) - Operational guide
- [Getting Started](GETTING_STARTED.md) - Quickstart guide
- [Architecture](ARCHITECTURE.md) - System design
- [Operations](OPERATIONS.md) - Production deployment
- [Contributing](../CONTRIBUTING.md) - Development guide
