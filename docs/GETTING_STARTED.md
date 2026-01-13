# Getting Started with Supervisor

## What is Supervisor?

Supervisor is an AI CLI orchestrator that treats AI tools (Claude, Codex, Gemini) as **stateless workers** rather than conversational partners. It solves the "context dilution" problem: when AI CLIs maintain long conversations, their context windows grow and they progressively ignore earlier instructions.

Instead of one long conversation, Supervisor spawns fresh, short-lived AI instances for each task step, feeding only relevant context, getting structured output, and immediately terminating them. Think of it as an **operating system for AI-assisted development** - managing AI workers with the same rigor you'd manage processes, memory, and I/O.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** - Check with `python --version`
- **Docker** - Required for sandbox execution (verify with `docker --version`)
- **Git repository** - Supervisor works within a git repository
- **AI CLI tool** - At least one of:
  - Claude CLI (recommended)
  - GitHub Copilot CLI (Codex)
  - Gemini CLI (via apple-gemini or similar)

### Docker Setup

Supervisor requires Docker for isolated execution. If you don't have Docker installed:

**macOS/Windows**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Linux**:
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Verify installation
docker --version
```

## Installation

Clone or navigate to your project directory, then install Supervisor:

```bash
# From the repository root
pip install -e .

# Verify installation
supervisor --version
```

You should see: `supervisor, version 0.1.0`

## Your First Workflow: "Hello World"

Let's walk through a complete example to get you comfortable with Supervisor.

### Step 1: Initialize Your Project

```bash
# Navigate to your git repository
cd /path/to/your/project

# Initialize Supervisor
supervisor init
```

This creates a `.supervisor/` directory with default configuration:
```
.supervisor/
├── config.yaml         # Project settings
├── limits.yaml         # Timeout configuration
├── adaptive.yaml       # Model selection rules
├── approval.yaml       # Approval policies
├── roles/              # Custom role definitions
└── templates/          # Custom prompt templates
```

### Step 2: List Available Roles

```bash
supervisor roles
```

You'll see three base roles:
- **planner** - Breaks features into phases and components
- **implementer** - Writes code following TDD principles
- **reviewer** - Reviews code for quality and correctness

### Step 3: Run Your First Command

Let's use the planner role to create a simple plan:

```bash
supervisor plan "Add a greeting function that takes a name and returns 'Hello, {name}!'"
```

**What happens**:
1. Supervisor creates an isolated git worktree
2. Packs relevant context (your codebase structure, task description)
3. Spawns a fresh Claude instance in a Docker container
4. Claude generates a structured plan
5. The plan is saved to the database and displayed

**Expected output**:
```
✓ Created worktree: /tmp/supervisor_abc123
✓ Packed context: 2.5k tokens
✓ Running planner role...

Feature Plan:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1: Implementation
  Component: greeting_function
    - Create src/greet.py
    - Implement greet(name: str) -> str
    - Add docstring and type hints

Phase 2: Testing
  Component: tests
    - Create tests/test_greet.py
    - Test basic greeting
    - Test edge cases (empty string, None)

✓ Plan created: feature_abc123
```

### Step 4: Implement the Feature

Now let's have the implementer write the code:

```bash
supervisor run implementer "Implement the greeting function from the plan" -t src/greet.py
```

**What happens**:
1. Creates a new worktree
2. Packs context including the plan
3. Runs implementer role (fresh Claude instance)
4. Claude writes the code following the plan
5. **Verification gates run**:
   - Linting (ruff)
   - Type checking (mypy)
   - Tests (pytest)
6. If gates pass, changes are applied to your repository

**Expected output**:
```
✓ Created worktree: /tmp/supervisor_def456
✓ Running implementer role...
✓ Code generated: src/greet.py (42 lines)

Running verification gates...
  ✓ lint passed (0.3s)
  ✓ type_check passed (0.8s)
  ✓ test passed (1.2s)

✓ Changes applied to repository
✓ Cleaned up worktree
```

### Step 5: Check Your Work

```bash
# See what was created
cat src/greet.py

# Run tests locally
pytest tests/test_greet.py
```

You'll find a fully implemented function with tests, type hints, and documentation!

## Core Concepts

### Roles

**Roles** define how AI workers behave. Each role has:
- **System prompt**: Instructions for the AI
- **CLI**: Which AI model to use (`claude`, `codex`, `gemini`)
- **Context**: What files/information to include
- **Gates**: Verification steps (tests, linting, etc.)
- **Configuration**: Retries, timeouts, etc.

Base roles are in `supervisor/config/base_roles/`. You can create custom roles in `.supervisor/roles/`.

**Example role configuration** (`.supervisor/roles/custom_reviewer.yaml`):
```yaml
name: custom_reviewer
description: "Custom code reviewer for our team standards"
extends: reviewer  # Inherit from base reviewer role

cli: claude:opus  # Use Claude Opus for higher quality

# Add project-specific review checklist
system_prompt: |
  Review code for:
  - Team coding standards (see STANDARDS.md)
  - Security vulnerabilities
  - Performance concerns
  - Test coverage

gates: [lint, type_check, security]
```

### Workflows

**Workflows** orchestrate multiple roles in sequence:
1. **Planning** - Planner breaks feature into phases/components
2. **Implementation** - Implementer writes code for each component
3. **Integration** - Integration tests run
4. **Review** - Reviewer checks the code

**Hierarchical structure**:
```
Feature
  └─ Phase 1
      ├─ Component A (can run in parallel)
      └─ Component B
  └─ Phase 2
      └─ Component C (depends on Phase 1)
```

**Run a full workflow**:
```bash
supervisor workflow feat-authentication --tui
```

The `--tui` flag opens a terminal UI showing real-time progress.

### Gates

**Gates** are verification steps enforced by the orchestrator (not trusted to AI):
- `test` - Run pytest
- `lint` - Run ruff
- `type_check` - Run mypy
- `security` - Run bandit
- `format_check` - Check code formatting

Gates are defined in `.supervisor/gates.yaml` and run in isolated worktrees.

**Custom gate example**:
```yaml
# .supervisor/gates.yaml
gates:
  custom_check:
    command: "./scripts/custom_validation.sh"
    timeout: 60
    fail_action: fail
    description: "Run custom validation"
```

### State Management

Supervisor uses **event sourcing** with SQLite:
- All actions are recorded as events (workflow started, step completed, gate passed, etc.)
- Current state is derived from the event log
- Full audit trail of what happened
- Can replay events to reconstruct state

Database location: `.supervisor/state.db`

**View current status**:
```bash
supervisor status
```

### Isolation

Every AI execution runs in isolation:
- **Git worktree**: Separate filesystem, safe to make changes
- **Docker container**: Sandboxed execution, network controls
- **Fresh instance**: No conversation history, no context dilution

**Security features**:
- Path traversal prevention
- Symlink rejection
- Network egress allowlist (AI APIs only)
- Atomic file operations

### Context Packing

Supervisor intelligently selects which files to send to the AI:
- Uses Repomix for smart file selection
- Respects token budgets (default 20k-30k tokens per role)
- Priority-based pruning when over budget
- Always includes task description and system prompt

**Context templates** use Jinja2:
- `supervisor/prompts/planning.j2` - Planning role
- `supervisor/prompts/implement.j2` - Implementation role
- `supervisor/prompts/review_strict.j2` - Review role

## Common Workflows

### Feature Development
```bash
# 1. Plan the feature
supervisor plan "Add user authentication"

# 2. Implement each component
supervisor run implementer "Implement login endpoint" -t src/api/auth.py
supervisor run implementer "Add password hashing" -t src/utils/crypto.py

# 3. Review the changes
supervisor run reviewer "Review authentication implementation"
```

### Bug Fixing
```bash
# Fix a specific bug
supervisor run implementer "Fix null pointer in user profile" -t src/models/user.py
```

### Code Review
```bash
# Review recent changes
supervisor run reviewer "Review changes in PR #123"
```

### Full Hierarchical Workflow
```bash
# Run complete workflow with TUI
supervisor workflow feat-$(uuid) --tui --parallel

# The TUI will show:
# - Feature breakdown
# - Phase progress
# - Component status
# - Gate results
# - Real-time updates
```

## Configuration

### Project Configuration (`.supervisor/config.yaml`)

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

### Timeout Configuration (`.supervisor/limits.yaml`)

```yaml
workflow_timeout: 3600  # 1 hour max for entire workflow
component_timeout: 300  # 5 minutes per component

role_timeouts:
  planner: 600      # 10 minutes for planning
  implementer: 300  # 5 minutes for implementation
  reviewer: 180     # 3 minutes for review
```

### Adaptive Model Selection (`.supervisor/adaptive.yaml`)

```yaml
# Automatically select best model based on task type
adaptive:
  enabled: true

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
      - codex:gpt-4
```

## Viewing Metrics

Supervisor tracks performance metrics for all executions:

```bash
# View metrics dashboard
supervisor metrics

# View metrics for last 7 days
supervisor metrics --days 7

# View metrics for specific role
supervisor metrics --role implementer
```

**Metrics include**:
- Success rates by role/CLI
- Average execution time
- Retry counts
- Token usage
- Model performance comparisons

## Troubleshooting

### Docker Not Available
```
Error: Docker is not available
```

**Solution**: Install Docker and ensure the daemon is running:
```bash
docker --version  # Should show version
docker ps         # Should connect to daemon
```

### Network Egress Not Configured
```
Warning: Cannot verify network egress rules
```

**Solution**: This is informational. Supervisor will work but won't validate Docker network isolation. For production, configure Docker network policies.

### Gate Failures
```
✗ test failed (exit code 1)
```

**Solution**: Check the gate output for details. Fix the failing tests/linting and retry. Supervisor shows full output from gates.

### Out of Context Budget
```
Warning: Context exceeds budget, pruning files...
```

**Solution**: This is normal. Supervisor prioritizes important files. To increase budget, edit your role configuration:
```yaml
# .supervisor/roles/my_role.yaml
context:
  token_budget: 40000  # Increase from default 25000
```

### Slow Performance
```bash
# Check metrics to identify bottlenecks
supervisor metrics

# Consider using faster models for routine tasks
# Edit .supervisor/config.yaml:
roles:
  implementer:
    cli: claude:haiku  # Faster, cheaper for simple tasks
```

## Next Steps

Now that you understand the basics:

1. **Read the [CLI Reference](CLI_REFERENCE.md)** - Complete command documentation
2. **Review [Architecture](ARCHITECTURE.md)** - Understand the system design
3. **Check out [Examples](../examples/)** - Practical workflow examples
4. **Read [Operations Guide](OPERATIONS.md)** - Production deployment and monitoring
5. **See [Contributing Guide](../CONTRIBUTING.md)** - Extend Supervisor for your needs

## Advanced Topics

### Custom Roles

Create specialized roles for your domain:

```yaml
# .supervisor/roles/frontend_implementer.yaml
name: frontend_implementer
description: "Frontend specialist with React expertise"
extends: implementer

cli: claude:sonnet

system_prompt: |
  You are a frontend developer specializing in React and TypeScript.
  Follow our component patterns in src/components/README.md.

  Guidelines:
  - Use functional components with hooks
  - TypeScript for all components
  - CSS Modules for styling
  - Comprehensive Storybook stories
  - Accessibility (ARIA, keyboard nav)

context:
  include:
    - "src/components/**/*.tsx"
    - "src/components/README.md"
    - "src/types/**/*.ts"
  token_budget: 30000

gates: [lint, type_check, test, format_check]
```

### Parallel Execution

Run independent components in parallel for faster workflows:

```yaml
# .supervisor/config.yaml
workflow:
  parallel_execution: true
```

Components with no dependencies run concurrently. Supervisor uses a DAG scheduler to maximize parallelism.

### Approval Policies

Configure when to ask for human approval:

```yaml
# .supervisor/approval.yaml
approval:
  # Always require approval for these patterns
  always_approve:
    - "src/database/migrations/**"
    - "*.sql"
    - "package.json"

  # Never require approval (fully automated)
  never_approve:
    - "tests/**"
    - "*.md"

  # Risk-based approval (depends on change size)
  risk_threshold:
    low: 50      # Lines changed
    medium: 200
    high: 500
```

### Integration with CI/CD

Use Supervisor in your CI pipeline:

```yaml
# .github/workflows/ai-review.yaml
name: AI Code Review

on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Supervisor
        run: pip install -e .

      - name: Run AI Review
        run: supervisor run reviewer "Review PR changes"
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: Report bugs at [GitHub Issues](https://github.com/LeeeWayyy/auto_coder/issues)
- **Examples**: Check `examples/` for complete workflows
- **Architecture**: Read `docs/ARCHITECTURE.md` for deep dive

## What Makes Supervisor Different?

| Traditional AI CLI | Supervisor |
|-------------------|------------|
| Long conversation | Fresh instance per step |
| Context grows unbounded | Curated context per task |
| Instructions dilute over time | Same instructions every time |
| Manual verification | Automated gates (tests, lint) |
| Conversational partner | Managed worker |
| No audit trail | Full event log (SQLite) |
| Ad-hoc execution | Structured workflows |

**Bottom line**: Supervisor treats AI as infrastructure, not as a chatbot. This enables reliable, auditable, scalable AI-assisted development.

---

Ready to dive deeper? Continue with:
- [CLI Reference](CLI_REFERENCE.md) for complete command documentation
- [Operations Guide](OPERATIONS.md) for production deployment
- [Architecture](ARCHITECTURE.md) for system internals
