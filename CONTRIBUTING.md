# Contributing to Supervisor

Thank you for your interest in contributing to Supervisor! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Adding Features](#adding-features)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Code Style](#code-style)
8. [Pull Request Process](#pull-request-process)

---

## Development Setup

### Prerequisites

- Python 3.11 or 3.12
- Docker 24.0.0+
- Git 2.30+
- A code editor (VS Code, PyCharm, etc.)

### Setup Instructions

1. **Fork and Clone**:
```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/auto_coder.git
cd auto_coder

# Add upstream remote
git remote add upstream https://github.com/LeeeWayyy/auto_coder.git
```

2. **Create Virtual Environment**:
```bash
# Create venv
python3.11 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

3. **Install Dependencies**:
```bash
# Install in development mode
pip install -e ".[dev]"

# Verify installation
supervisor --version
pytest --version
ruff --version
```

4. **Install Pre-commit Hooks** (optional but recommended):
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

5. **Verify Setup**:
```bash
# Run tests
pytest

# Check code style
ruff check .

# Type check
mypy supervisor/
```

---

## Project Structure

Understanding the codebase organization:

```
supervisor/
├── supervisor/              # Main package
│   ├── core/               # Core orchestration logic
│   │   ├── engine.py       # ExecutionEngine (main coordinator)
│   │   ├── state.py        # Database and event sourcing
│   │   ├── context.py      # Context packing
│   │   ├── parser.py       # Output parsing
│   │   ├── roles.py        # Role configuration
│   │   ├── gates.py        # Gate system
│   │   ├── workflow.py     # Hierarchical workflows
│   │   ├── routing.py      # Multi-model routing
│   │   └── workspace.py    # Git worktree management
│   ├── sandbox/            # Docker isolation
│   │   └── executor.py     # Sandboxed execution
│   ├── metrics/            # Performance tracking
│   │   ├── aggregator.py
│   │   ├── collector.py
│   │   └── dashboard.py
│   ├── tui/                # Terminal UI
│   │   └── app.py
│   ├── config/             # Base roles and gates
│   ├── prompts/            # Jinja2 templates
│   └── cli.py              # CLI entry point
├── tests/                  # Test suite
│   ├── conftest.py         # Shared fixtures
│   ├── test_*.py           # Unit tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation
│   ├── SPECS/              # Technical specifications
│   └── PLANS/              # Implementation plans
├── examples/               # Example workflows
└── pyproject.toml          # Project configuration
```

### Key Modules and Responsibilities

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| `engine.py` | Orchestrates execution flow | `ExecutionEngine`, `CircuitBreaker`, `RetryPolicy` |
| `state.py` | Event sourcing and database | `Database`, `Event` |
| `context.py` | Intelligent file selection | `ContextPacker` |
| `parser.py` | Parse AI output | `parse_role_output`, adapters |
| `roles.py` | Role configuration | `RoleLoader`, `RoleConfig` |
| `gates.py` | Verification gates | `GateLoader`, `GateExecutor` |
| `workspace.py` | Git worktree isolation | `IsolatedWorkspace` |
| `executor.py` | Docker sandbox | `SandboxedLLMClient`, `SandboxedExecutor` |

---

## Development Workflow

### Creating a Feature Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### Making Changes

1. **Write Code**: Implement your feature or fix
2. **Write Tests**: Add tests for your changes (required)
3. **Update Documentation**: Update relevant docs
4. **Test Locally**: Run full test suite
5. **Commit**: Make atomic, well-described commits

### Commit Message Format

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples**:
```
feat(engine): add timeout support for role execution

- Add timeout parameter to ExecutionEngine.run_role()
- Configure timeouts via limits.yaml
- Add tests for timeout enforcement

Closes #123
```

```
fix(parser): handle malformed JSON in Claude output

- Add fallback parsing for incomplete JSON blocks
- Improve error messages
- Add test case for malformed output

Fixes #456
```

---

## Adding Features

### Adding a New Role

1. **Create Role Configuration**:
```yaml
# supervisor/config/base_roles/my_new_role.yaml
name: my_new_role
description: "Description of the new role"
cli: claude

system_prompt: |
  You are a specialist in ...

context:
  include:
    - "src/**/*.py"
  token_budget: 25000

gates: [test, lint]

config:
  max_retries: 3
  timeout: 300
```

2. **Create Prompt Template**:
```jinja
{# supervisor/prompts/my_new_role.j2 #}
{% extends "_base.j2" %}

{% block system_prompt %}
{{ system_prompt }}
{% endblock %}

{% block task %}
Task: {{ task_description }}
{% endblock %}

{% block context %}
{# Custom context for this role #}
{% endblock %}
```

3. **Add Tests**:
```python
# tests/test_roles.py
def test_load_my_new_role():
    loader = RoleLoader()
    role = loader.load_role("my_new_role")

    assert role.name == "my_new_role"
    assert role.cli == "claude"
    assert "test" in role.gates
```

4. **Update Documentation**:
- Add to `docs/CLI_REFERENCE.md`
- Add example to `examples/`

### Adding a Custom Gate

1. **Define Gate Configuration**:
```yaml
# supervisor/config/gates.yaml (or project .supervisor/gates.yaml)
gates:
  my_custom_gate:
    command: ["python", "scripts/my_validation.py"]
    timeout: 60
    description: "My custom validation"
    severity: error
    fail_action: fail
    depends_on: []
```

2. **Implement Gate Script**:
```python
# scripts/my_validation.py
import sys

def validate():
    # Your validation logic
    if error_found:
        print("Error: validation failed")
        return 1
    print("Validation passed")
    return 0

if __name__ == "__main__":
    sys.exit(validate())
```

3. **Add Tests**:
```python
# tests/test_gates.py
def test_my_custom_gate(temp_gate_file):
    gates = {"my_custom_gate": {...}}
    write_gates(temp_gate_file, gates)

    loader = GateLoader()
    gate = loader.load_gate("my_custom_gate")

    assert gate.command == ["python", "scripts/my_validation.py"]
```

### Adding a CLI Adapter

Support a new AI CLI:

1. **Implement Adapter**:
```python
# supervisor/core/parser.py

class NewAIAdapter:
    """Adapter for NewAI CLI output format."""

    def parse(self, output: str) -> GenericOutput:
        """Parse NewAI output format."""
        # Extract structured output
        # Validate against schema
        # Return GenericOutput
        pass

# Register adapter
ADAPTERS["newai"] = NewAIAdapter()
```

2. **Add Schema**:
```python
# supervisor/core/parser.py

ROLE_SCHEMAS["implementer_newai"] = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "files_modified": {"type": "array"},
        # ...
    },
    "required": ["status"]
}
```

3. **Add Tests**:
```python
# tests/test_parser.py

def test_newai_adapter():
    adapter = NewAIAdapter()
    output = '{"status": "success", ...}'

    result = adapter.parse(output)

    assert result.status == "success"
```

### Extending Metrics

Add new metrics:

1. **Update Database Schema**:
```python
# supervisor/core/state.py

class Database:
    SCHEMA = """
    ...
    -- Add new metric columns
    ALTER TABLE metrics ADD COLUMN my_new_metric REAL;
    """
```

2. **Record Metrics**:
```python
# supervisor/core/engine.py

def run_role(...):
    # ...
    self.db.record_metric(
        workflow_id=workflow_id,
        # ... existing fields ...
        my_new_metric=calculated_value
    )
```

3. **Display in Dashboard**:
```python
# supervisor/metrics/dashboard.py

def show(self, days: int):
    # ... existing code ...
    # Add new metric visualization
```

---

## Testing

### Testing Requirements

- **Coverage**: 80%+ line coverage required
- **Unit Tests**: All new functions/classes must have unit tests
- **Integration Tests**: Complex interactions need integration tests
- **Security Tests**: Security-sensitive code needs dedicated tests

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=supervisor --cov-report=term-missing

# Run specific test file
pytest tests/test_engine.py

# Run specific test
pytest tests/test_engine.py::TestRetryPolicy::test_exponential_backoff

# Run with markers
pytest -m "not slow"  # Skip slow tests
pytest -m docker      # Only Docker tests
pytest -m integration # Only integration tests

# Verbose output
pytest -v
pytest -vv  # Extra verbose
```

### Writing Tests

**Unit Test Example**:
```python
# tests/test_my_feature.py

import pytest
from supervisor.core.my_module import MyClass

class TestMyClass:
    """Tests for MyClass."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        obj = MyClass()
        result = obj.do_something()

        assert result == expected_value

    def test_error_handling(self):
        """Test error handling."""
        obj = MyClass()

        with pytest.raises(ValueError):
            obj.invalid_operation()

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_multiple_inputs(self, input, expected):
        """Test with multiple inputs."""
        obj = MyClass()
        assert obj.double(input) == expected
```

**Integration Test Example**:
```python
# tests/integration/test_my_workflow.py

import pytest

@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete workflows."""

    def test_end_to_end_flow(self, temp_repo, test_db, mocker):
        """Test complete execution flow."""
        # Setup
        engine = ExecutionEngine(temp_repo, db=test_db)
        mocker.patch("supervisor.core.engine.SandboxedLLMClient")

        # Execute
        result = engine.run_role(...)

        # Verify
        assert result is not None
        events = test_db.get_events(workflow_id)
        assert len(events) > 0
```

**Using Fixtures**:
```python
def test_with_fixtures(temp_repo, test_db, sample_role_config):
    """Test using shared fixtures from conftest.py."""
    # temp_repo: Temporary git repository
    # test_db: Test database
    # sample_role_config: Sample role configuration

    # Your test code
    pass
```

### Test Best Practices

1. **Isolation**: Tests should be independent
2. **Descriptive Names**: Use clear, descriptive test names
3. **Single Assertion**: One logical assertion per test (when possible)
4. **Mock External Dependencies**: Mock Docker, network calls, etc.
5. **Test Edge Cases**: Include error cases, boundary values
6. **Fast Tests**: Keep unit tests fast (<1s each)

---

## Documentation

### Documentation Requirements

All new features must include:

1. **Docstrings**: All public functions/classes
2. **Type Hints**: All function signatures
3. **User Documentation**: Updates to relevant guides
4. **Examples**: Working code examples

### Docstring Format

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """Short description of function.

    Longer description if needed. Explain behavior, edge cases,
    and any important notes.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
        RuntimeError: When operation fails

    Example:
        >>> my_function("test", 42)
        True
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")

    # Implementation
    return True
```

### Updating Documentation

When adding features, update:

- `docs/GETTING_STARTED.md`: If it affects user workflow
- `docs/CLI_REFERENCE.md`: If adding/changing CLI commands
- `docs/ARCHITECTURE.md`: If changing system design
- `docs/OPERATIONS.md`: If affecting production deployment
- `README.md`: For major features
- `CHANGELOG.md`: Always update unreleased section

---

## Code Style

### Python Style Guide

Follow PEP 8 with project conventions:

- **Line length**: 100 characters (configured in ruff)
- **Imports**: Organized and sorted (ruff handles this)
- **Type hints**: Required for all public APIs
- **Docstrings**: Required for all public functions/classes

### Linting and Formatting

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check --fix .

# Type checking
mypy supervisor/

# All checks
ruff check . && mypy supervisor/
```

### Pre-commit Configuration

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Code Review Checklist

Before submitting PR, verify:

- [ ] Code follows style guide (ruff passes)
- [ ] Type hints are present (mypy passes)
- [ ] Tests are written and passing (pytest passes)
- [ ] Coverage is maintained or improved (>80%)
- [ ] Docstrings are present and accurate
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Commit messages are clear and follow format
- [ ] No security vulnerabilities introduced
- [ ] No hardcoded secrets or credentials

---

## Pull Request Process

### Before Submitting

1. **Sync with Upstream**:
```bash
git checkout main
git pull upstream main
git checkout your-feature-branch
git rebase main
```

2. **Run All Checks**:
```bash
# Tests
pytest --cov=supervisor

# Linting
ruff check .

# Type checking
mypy supervisor/
```

3. **Update CHANGELOG.md**:
```markdown
## [Unreleased]

### Added
- Description of your feature (#PR_NUMBER)
```

### Submitting Pull Request

1. **Push to Fork**:
```bash
git push origin your-feature-branch
```

2. **Open Pull Request** on GitHub:
   - Clear title: "feat: Add feature X" or "fix: Fix bug Y"
   - Description: What, why, and how
   - Link related issues: "Closes #123"
   - Screenshots/demos if applicable

3. **PR Template**:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] All tests passing
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guide
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commits follow format
```

### Review Process

1. **Automated Checks**: CI must pass (tests, linting, type checking)
2. **Code Review**: At least one approval required
3. **Address Feedback**: Respond to review comments
4. **Squash and Merge**: Maintainers will merge when approved

### After Merge

1. **Update Local**:
```bash
git checkout main
git pull upstream main
git branch -d your-feature-branch
```

2. **Delete Remote Branch**:
```bash
git push origin --delete your-feature-branch
```

---

## Development Tips

### Debugging

```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use breakpoint
breakpoint()  # Python 3.7+
```

### Testing in Isolation

```bash
# Create test virtualenv
python -m venv test-venv
source test-venv/bin/activate
pip install -e ".[dev]"

# Test specific scenario
pytest tests/test_engine.py -v
```

### Working with Docker

```bash
# Build custom test image
docker build -t supervisor-test:latest -f Dockerfile.test .

# Run tests in Docker
docker run --rm -v $(pwd):/app supervisor-test pytest
```

---

## Getting Help

- **GitHub Issues**: https://github.com/LeeeWayyy/auto_coder/issues
- **Discussions**: https://github.com/LeeeWayyy/auto_coder/discussions
- **Documentation**: https://supervisor.readthedocs.io (when available)

---

## Code of Conduct

Be respectful, inclusive, and professional. See `CODE_OF_CONDUCT.md` for details.

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Supervisor! 🎉
