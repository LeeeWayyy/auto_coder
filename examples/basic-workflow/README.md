# Basic Workflow Example

A simple demonstration of Supervisor's core workflow: plan → implement → review.

## Objective

Implement a greeting function that:
- Takes a name as input
- Returns a personalized greeting
- Includes proper error handling
- Has complete test coverage

## Prerequisites

- Supervisor installed (`pip install -e .`)
- Docker running
- Claude API key configured

## Setup

1. Navigate to this directory:
```bash
cd examples/basic-workflow
```

2. Initialize (already done for you):
```bash
# The .supervisor/ directory is pre-configured
ls .supervisor/
```

3. Review the configuration:
```bash
cat .supervisor/config.yaml
```

## Running the Example

### Step 1: Plan the Feature

```bash
supervisor plan "Implement a greeting function that takes a name and returns a personalized greeting"
```

**What happens**:
- Planner role analyzes the task
- Breaks it into phases and components
- Returns a feature ID (e.g., `feat-abc12345`)

**Expected output**:
```
✓ Planning task: Implement a greeting function...

Feature Plan:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1: Implementation
  Component: greeting_function
    - Create src/greet.py
    - Implement greet(name: str) -> str
    - Add docstring and type hints

Phase 2: Testing
  Component: tests
    - Create tests/test_greet.py
    - Test basic greeting
    - Test edge cases (empty string, None, special characters)

✓ Plan created: feat-abc12345
```

### Step 2: Implement the Greeting Function

```bash
supervisor run implementer "Implement the greeting function from the plan" -t src/greet.py
```

**What happens**:
- Creates isolated git worktree
- Implementer role writes the code
- Runs verification gates (lint, type_check, test)
- Applies changes if all gates pass

**Expected output**:
```
✓ Running implementer role...
✓ Code generated: src/greet.py (42 lines)

Running verification gates...
  ✓ lint passed (0.3s)
  ✓ type_check passed (0.8s)
  ✓ test passed (1.2s)

✓ Changes applied to repository
```

**Generated file** (`src/greet.py`):
```python
"""Greeting module."""

def greet(name: str) -> str:
    """Generate a personalized greeting.

    Args:
        name: Name of the person to greet

    Returns:
        Personalized greeting string

    Raises:
        ValueError: If name is empty or None

    Example:
        >>> greet("Alice")
        'Hello, Alice! Welcome!'
    """
    if not name:
        raise ValueError("Name cannot be empty")

    # Sanitize input
    name = name.strip()

    return f"Hello, {name}! Welcome!"
```

### Step 3: Implement Tests

```bash
supervisor run implementer "Implement comprehensive tests for the greeting function" -t tests/test_greet.py
```

**Generated file** (`tests/test_greet.py`):
```python
"""Tests for greeting module."""

import pytest
from src.greet import greet

def test_basic_greeting():
    """Test basic greeting."""
    result = greet("Alice")
    assert result == "Hello, Alice! Welcome!"

def test_greeting_with_whitespace():
    """Test greeting strips whitespace."""
    result = greet("  Bob  ")
    assert result == "Hello, Bob! Welcome!"

def test_empty_name_raises():
    """Test empty name raises ValueError."""
    with pytest.raises(ValueError):
        greet("")

def test_none_raises():
    """Test None raises ValueError."""
    with pytest.raises(ValueError):
        greet(None)

@pytest.mark.parametrize("name,expected", [
    ("Alice", "Hello, Alice! Welcome!"),
    ("Bob", "Hello, Bob! Welcome!"),
    ("Charlie-Smith", "Hello, Charlie-Smith! Welcome!"),
])
def test_multiple_names(name, expected):
    """Test with multiple inputs."""
    assert greet(name) == expected
```

### Step 4: Review the Implementation

```bash
supervisor run reviewer "Review the greeting implementation for quality and completeness"
```

**What happens**:
- Reviewer role analyzes the code
- Checks for best practices, edge cases, documentation
- Provides feedback

**Expected output**:
```
✓ Running reviewer role...

Review Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Code Quality: ✓ Excellent
- Clear function signature with type hints
- Comprehensive docstring
- Good error handling

Test Coverage: ✓ Complete
- All branches covered
- Edge cases tested
- Parametrized tests for multiple inputs

Recommendations:
1. Consider adding tests for international characters
2. Add example usage to README
```

### Step 5: Run Tests Locally

```bash
# Run the test suite
pytest tests/test_greet.py -v

# Check coverage
pytest tests/test_greet.py --cov=src
```

**Expected output**:
```
tests/test_greet.py::test_basic_greeting PASSED
tests/test_greet.py::test_greeting_with_whitespace PASSED
tests/test_greet.py::test_empty_name_raises PASSED
tests/test_greet.py::test_none_raises PASSED
tests/test_greet.py::test_multiple_names[Alice-...] PASSED
tests/test_greet.py::test_multiple_names[Bob-...] PASSED
tests/test_greet.py::test_multiple_names[Charlie-Smith-...] PASSED

---------- coverage: 100% ----------
```

## What You Learned

- ✅ How to plan a feature with `supervisor plan`
- ✅ How to implement code with `supervisor run implementer`
- ✅ How to target specific files with `-t`
- ✅ How verification gates work (lint, type_check, test)
- ✅ How to review code with `supervisor run reviewer`
- ✅ How Supervisor creates isolated worktrees for each step

## Configuration Details

### `.supervisor/config.yaml`

```yaml
# Using default configuration
default_cli: claude

workflow:
  parallel_execution: false  # Simple sequential workflow
  require_tests: true
  human_approval: false  # Fully automated for this example
```

### `.supervisor/limits.yaml`

```yaml
# Standard timeouts
workflow_timeout: 3600
component_timeout: 300

role_timeouts:
  planner: 600
  implementer: 300
  reviewer: 180
```

## Next Steps

1. **Try modifying the task**: "Add support for multiple languages"
2. **Add error handling**: "Handle special characters and emojis"
3. **Explore other examples**: See [custom-gates](../custom-gates/) for adding verification

## Troubleshooting

**Tests failing**:
```bash
# Check what went wrong
cd .supervisor/.worktrees/latest-worktree
pytest -v
```

**Want to start over**:
```bash
# Reset the repository
git checkout -- src/ tests/
git clean -fd
```

**View metrics**:
```bash
supervisor metrics --days 1
```

---

**Next Example**: [Custom Gates](../custom-gates/) - Learn to add project-specific verification
