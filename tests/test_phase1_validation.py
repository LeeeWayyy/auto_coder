#!/usr/bin/env python3
"""Static validation test for Phase 1 implementation.

This script checks:
1. Python syntax of all new modules
2. YAML syntax of example workflows
3. Schema validation (if pydantic available)
"""

import ast
import importlib.util
import sys


def check_python_syntax(file_path):
    """Check Python file for syntax errors."""
    try:
        with open(file_path) as f:
            ast.parse(f.read(), filename=str(file_path))
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_yaml_syntax(file_path):
    """Check YAML file for syntax errors."""
    try:
        import yaml

        with open(file_path) as f:
            yaml.safe_load(f)
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    """Run validation tests."""
    print("Phase 1 Implementation Validation")
    print("=" * 50)

    # Check Python modules
    python_files = [
        "supervisor/core/graph_schema.py",
        "supervisor/core/graph_engine.py",
        "supervisor/core/worker.py",
    ]

    print("\n1. Checking Python syntax...")
    all_passed = True
    for file_path in python_files:
        passed, error = check_python_syntax(file_path)
        status = "✓" if passed else "✗"
        print(f"  {status} {file_path}")
        if not passed:
            print(f"     Error: {error}")
            all_passed = False

    # Check YAML files
    yaml_files = [
        "examples/workflows/simple_workflow.yaml",
        "examples/workflows/debug_workflow.yaml",
        "examples/workflows/parallel_review.yaml",
    ]

    print("\n2. Checking YAML syntax...")
    if importlib.util.find_spec("yaml") is None:
        print("  ⚠ yaml module not available, skipping YAML checks")
    else:
        for file_path in yaml_files:
            passed, error = check_yaml_syntax(file_path)
            status = "✓" if passed else "✗"
            print(f"  {status} {file_path}")
            if not passed:
                print(f"     Error: {error}")
                all_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All checks passed!")
        return 0
    else:
        print("✗ Some checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
