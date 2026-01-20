# Frequently Asked Questions (FAQ)

Common questions and answers about Auto Coder (Supervisor CLI). For operational guidance, see the [Runbook](../runbook/README.md).

## Table of Contents

- [General](#general)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Security](#security)
- [Development](#development)

---

## General

### What is Supervisor?

Supervisor is an AI-powered orchestration system that coordinates multiple AI models to plan, implement, review, and deploy code changes. It uses event sourcing, isolated execution environments, and verification gates to ensure safe, reliable automation.

### Does it include a web console?

Yes. **Supervisor Studio** is a web UI for visual workflow management (graph editing, execution, and live monitoring). Start it with:
```bash
supervisor studio --port 8000
```
Studio is localhost-only and has no authentication. See the runbook for full setup details.

### How does Supervisor differ from LangGraph or MetaGPT?

| Feature | Supervisor | LangGraph | MetaGPT |
|---------|-----------|-----------|---------|
| **Focus** | Code workflow orchestration | General agent graphs | Role-based AI collaboration |
| **Isolation** | Docker + git worktrees | Not built-in | Not built-in |
| **Event Sourcing** | SQLite-based | Custom | Custom |
| **Gate System** | Built-in verification | Manual | Manual |
| **Multi-Model** | Native support (Claude, Codex, Gemini) | Possible | Possible |
| **Production Ready** | Yes (file locking, conflict detection) | Research-oriented | Research-oriented |

**When to use Supervisor**:
- You need safe, auditable code automation
- You want isolated execution (worktrees, Docker)
- You need multi-step verification (lint, test, security gates)
- You need production-grade reliability (event sourcing, atomic operations)

**When to use LangGraph**:
- You're building custom agent workflows outside of code automation
- You need fine-grained control over agent graph structure
- Your use case is research or experimentation

**When to use MetaGPT**:
- You want AI to simulate a full software company (PM, architect, engineer, etc.)
- Your focus is on design documents and specification generation

### Is Supervisor free and open source?

Yes, Supervisor is open source under the MIT License. You can use it freely for personal or commercial projects.

### What AI models does Supervisor support?

Supervisor supports:
- **Claude** (Anthropic): Opus, Sonnet, Haiku
- **Codex** (OpenAI): GPT-4, GPT-3.5
- **Gemini** (Google): Pro, Flash

You can configure different models for different roles (e.g., Opus for complex planning, Haiku for simple reviews).

---

## Installation and Setup

### What are the system requirements?

- **Python**: 3.11 or 3.12
- **Docker**: 24.0.0+ (required for sandbox isolation)
- **Git**: 2.30+ (for worktree management)
- **OS**: Linux (Ubuntu 20.04+), macOS 12+, or Windows with WSL2
- **Disk**: 10GB+ for Docker images and worktrees
- **Memory**: 4GB+ RAM (8GB+ recommended for parallel execution)

### Do I really need Docker?

Yes, Docker is required for security. All AI CLI execution and gate commands run in isolated containers to prevent:
- Accidental file system modifications
- Network access to unauthorized endpoints
- Execution of malicious code

If Docker is not available, Supervisor will not start.

### How do I set up API keys?

Create a `.env` file in your project root:

```bash
ANTHROPIC_API_KEY=your-claude-api-key
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key
```

Then source it:
```bash
source .env
supervisor plan "Your task"
```

Alternatively, export as environment variables in your shell profile.

### Can I use Supervisor without an API key?

If you're using subscription-based CLI tools (like Claude Pro, Gemini Advanced, or Codex subscriptions), the CLI handles authentication automatically - no separate API key is needed.

For API access (programmatic usage), you need at least one AI API key configured. All roles (planner, implementer, reviewer) use AI models.

---

## Usage

### How do I start a new feature?

```bash
# 1. Initialize project
cd your-repo
supervisor init

# 2. Plan the feature
supervisor plan "Add user authentication with JWT tokens"
# Returns: feat-abc12345

# 3. Execute the workflow
supervisor workflow feat-abc12345

# Or run individual roles:
supervisor run implementer "Implement JWT middleware" -t api/middleware/auth.py
```

### What's the difference between `supervisor run` and `supervisor workflow`?

- **`supervisor run`**: Execute a single role (planner, implementer, reviewer) on a specific task
- **`supervisor workflow`**: Execute a complete multi-phase workflow from a feature plan

Use `supervisor run` for one-off tasks. Use `supervisor workflow` for full feature implementation.

### How do I customize gate verification?

Edit `.supervisor/gates.yaml`:

```yaml
gates:
  my_custom_gate:
    command: ["python", "scripts/validate.py"]
    timeout: 60
    description: "Custom validation"
    severity: error
    fail_action: fail
```

Then reference it in your role configuration:
```yaml
# .supervisor/roles/implementer.yaml
gates: [test, lint, my_custom_gate]
```

### Can I skip gates for testing?

Yes, use `--skip-gates` or `--gates []`:

```bash
# Skip all gates
supervisor run implementer "Quick test" --skip-gates

# Run only specific gates
supervisor run implementer "Test task" --gates lint test
```

**Warning**: Skipping gates bypasses verification. Only use for development/testing.

### How do I target specific files?

Use the `-t` or `--target-files` flag:

```bash
supervisor run implementer "Add validation" -t api/routes.py -t models/user.py
```

This provides context to the AI about which files to modify.

---

## Troubleshooting

### Why is my workflow slow?

Common causes:
1. **Large context**: Reduce `context.token_budget` in role configuration
2. **Expensive model**: Use faster models (Haiku instead of Opus) for simple tasks
3. **Slow gates**: Profile gate execution, enable caching
4. **Sequential execution**: Enable `workflow.parallel_execution: true` in config

Diagnostic:
```bash
supervisor metrics --days 1
```

### Gates are failing but code looks correct

Check gate output:
```bash
supervisor status --workflow-id wf-abc123
# Shows gate output for each step
```

Common issues:
- **Test failures**: Missing test dependencies, incorrect test paths
- **Lint failures**: Code style issues (run `ruff check .` manually)
- **Timeout**: Increase gate timeout in `.supervisor/gates.yaml`

### Docker container killed (exit code 137)

This is an out-of-memory error. Solutions:
1. Increase Docker memory limit (Docker Desktop → Settings → Resources)
2. Reduce parallel execution:
   ```yaml
   # .supervisor/config.yaml
   workflow:
     parallel_execution: false
   ```
3. Reduce container memory:
   ```yaml
   sandbox:
     memory_limit: 2g  # Reduce from 4g
   ```

### Changes aren't being applied to my repository

Check for:
1. **Gate failures**: Run `supervisor status` to see which gates failed
2. **HEAD conflicts**: Another process modified the repository
   ```bash
   git status  # Check for unexpected changes
   git log -1  # Verify HEAD hasn't moved
   ```
3. **Permission errors**: Ensure write access to repository

### "Database is locked" error

This occurs with concurrent access. Solutions:
1. Wait a few seconds and retry
2. Reduce parallel execution
3. Check for stale lock files:
   ```bash
   rm .supervisor/.apply.lock
   ```

---

## Performance

### How can I speed up execution?

**1. Use appropriate models per role**:
```yaml
# .supervisor/config.yaml
roles:
  planner: claude-opus    # Complex reasoning
  implementer: claude-sonnet  # Balanced
  reviewer: claude-haiku   # Fast, simple checks
```

**2. Enable caching**:
```yaml
# .supervisor/gates.yaml
gates:
  test:
    cache: true  # Skip if code unchanged
```

**3. Reduce context size**:
```yaml
# .supervisor/roles/implementer.yaml
context:
  token_budget: 15000  # Reduce from 25000
```

**4. Enable parallel execution**:
```yaml
# .supervisor/config.yaml
workflow:
  parallel_execution: true
```

### How much do API calls cost?

Approximate costs per workflow (varies by complexity):
- **Planning** (Opus): $0.10-0.50
- **Implementation** (Sonnet): $0.20-1.00 per component
- **Review** (Haiku): $0.05-0.20

**Total for medium feature**: $1-5

**Cost optimization**:
- Use cheaper models (Haiku) for simple tasks
- Enable caching to avoid redundant API calls
- Use adaptive routing to automatically select cost-effective models

### How long does a typical workflow take?

Approximate times (without caching):
- **Planning**: 30-90 seconds
- **Implementation** (per component): 1-3 minutes
- **Review**: 30-60 seconds
- **Full workflow** (3-5 components): 10-30 minutes

**Time optimization**: See performance section above.

---

## Security

### Is it safe to run AI-generated code?

Supervisor includes multiple security layers:
1. **Docker isolation**: All execution in sandboxed containers
2. **Network egress control**: Whitelist allowed endpoints
3. **File system isolation**: Git worktrees prevent corruption of main repository
4. **Verification gates**: Mandatory testing, linting, security scanning
5. **Human approval**: Configurable approval for high-risk changes

**Best practices**:
- Always run security gates (bandit, semgrep, etc.)
- Enable approval policies for production changes
- Review AI output before deploying

### Can Supervisor access my secrets?

No. Supervisor explicitly filters sensitive environment variables (PATH, LD_PRELOAD, SUPERVISOR_*, etc.) before passing to AI CLIs.

**Best practices**:
- Use `.env` files for API keys (not in code)
- Add `.env` to `.gitignore`
- Use secrets managers (AWS Secrets Manager, etc.) for production

### What data is sent to AI APIs?

Supervisor sends:
- **System prompt**: Role description
- **Task description**: User-provided task
- **File context**: Selected files based on `context.include` patterns

Supervisor does NOT send:
- Files outside of context patterns
- Binary files
- Files in `.gitignore` (unless explicitly included)
- Environment variables or secrets

**Privacy**: Review context packing in `.supervisor/roles/{role}.yaml`

### How do I audit what changes were made?

Supervisor uses event sourcing - all changes are logged:

```bash
# Query event log
sqlite3 .supervisor/state.db "SELECT * FROM events WHERE workflow_id='wf-abc123';"

# View step output
supervisor status --workflow-id wf-abc123 --verbose

# Check git history
git log --oneline --all
```

All events include timestamp, role, files modified, gate results, and AI output.

---

## Development

### How do I contribute to Supervisor?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Development setup
- Coding standards
- Testing requirements
- Pull request process

### How do I add a custom role?

1. Create role configuration:
```yaml
# .supervisor/roles/my_role.yaml
name: my_role
description: "My custom role"
cli: claude-sonnet

system_prompt: |
  You are a specialist in...

context:
  include:
    - "src/**/*.py"
  token_budget: 25000

gates: [test, lint]
```

2. Create prompt template:
```jinja
{# .supervisor/prompts/my_role.j2 #}
{% extends "_base.j2" %}

{% block system_prompt %}
{{ system_prompt }}
{% endblock %}

{% block task %}
Task: {{ task_description }}
{% endblock %}
```

3. Use the role:
```bash
supervisor run my_role "Your task description"
```

### How do I run tests?

```bash
# All tests
pytest

# With coverage
pytest --cov=supervisor --cov-report=term-missing

# Specific test file
pytest tests/test_engine.py

# Fast tests only (skip slow integration tests)
pytest -m "not slow"
```

### Where can I get help?

- **Documentation**: See `docs/` directory
- **GitHub Issues**: https://github.com/LeeeWayyy/auto_coder/issues
- **Discussions**: https://github.com/LeeeWayyy/auto_coder/discussions

---

## Common Error Messages

### "Docker is required but not available"

**Cause**: Docker daemon is not running.

**Fix**:
```bash
# Linux
sudo systemctl start docker

# macOS
open -a Docker

# Verify
docker ps
```

### "HEAD has moved - possible concurrent modification"

**Cause**: Another process modified the repository while Supervisor was running.

**Fix**:
1. Check git status: `git status`
2. Resolve conflicts if any
3. Retry the workflow

### "Gate 'test' failed: 3 tests failed"

**Cause**: Tests failed verification.

**Fix**:
1. View gate output: `supervisor status --workflow-id wf-abc123`
2. Run tests manually: `pytest -v`
3. Fix failing tests
4. Retry

### "FileNotFoundError: [Errno 2] No such file or directory: '.supervisor'"

**Cause**: Project not initialized.

**Fix**:
```bash
supervisor init
```

### "ValidationError: Invalid gate configuration"

**Cause**: Malformed YAML in `.supervisor/gates.yaml`.

**Fix**:
1. Validate YAML syntax: `yamllint .supervisor/gates.yaml`
2. Check required fields: `command`, `timeout`, `description`
3. Review [CLI Reference](CLI_REFERENCE.md#gate-configuration)

---

## Best Practices

### 1. Start Small

Begin with simple tasks:
```bash
supervisor run implementer "Add a docstring to function foo()" -t utils.py
```

Gradually increase complexity as you learn the system.

### 2. Use Descriptive Task Descriptions

**Bad**: "Fix bug"
**Good**: "Fix authentication bug where expired tokens are accepted as valid"

More context = better AI output.

### 3. Enable All Relevant Gates

Minimum recommended gates:
```yaml
gates: [test, lint, type_check]
```

Production workflows:
```yaml
gates: [test, lint, type_check, security, coverage]
```

### 4. Review AI Output

Always review generated code before deploying:
```bash
supervisor run reviewer "Review the authentication implementation"
```

### 5. Use Version Control

Commit after each successful workflow:
```bash
git add .
git commit -m "feat: Add authentication (AI-assisted)"
```

This creates audit trail and allows rollback.

---

## Migration and Compatibility

### Can I use Supervisor with existing projects?

Yes! Supervisor works with any git repository:
```bash
cd your-existing-project
supervisor init
supervisor plan "Add feature X"
```

Supervisor creates `.supervisor/` directory but doesn't modify existing files until you run workflows.

### Does Supervisor work with monorepos?

Yes. Initialize Supervisor at the repo root:
```bash
cd monorepo/
supervisor init
```

Then use `-t` to target specific packages:
```bash
supervisor run implementer "Add validation" -t packages/api/src/routes.py
```

### Can I integrate Supervisor with CI/CD?

Yes. Example GitHub Actions:
```yaml
- name: AI Code Review
  run: supervisor run reviewer "Review PR changes"
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

See [Operations Guide](OPERATIONS.md#cicd-integration) for full examples.

---

## See Also

- [Getting Started](GETTING_STARTED.md) - Quickstart guide
- [CLI Reference](CLI_REFERENCE.md) - Command documentation
- [Architecture](ARCHITECTURE.md) - System design
- [Operations](OPERATIONS.md) - Production deployment
- [Contributing](../CONTRIBUTING.md) - Development guide
