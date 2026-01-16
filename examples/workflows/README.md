# Workflow Examples

This directory contains example declarative workflow graphs for Supervisor Studio Phase 1.

## Available Workflows

### 1. Simple Workflow (`simple_workflow.yaml`)

A basic linear workflow demonstrating the fundamentals:
- Plan → Implement → Review
- Shows basic task nodes and sequential execution
- Good for testing the basic graph execution

**Usage:**
```bash
supervisor run-graph examples/workflows/simple_workflow.yaml --workflow-id test-simple
```

### 2. Debug Workflow with Retry Loop (`debug_workflow.yaml`)

Demonstrates loop control and branching:
- Analyze → Fix → Test → Check (branch)
- If tests fail, loop back to Analyze (max 3 iterations)
- If tests pass, proceed to Review
- Shows BRANCH nodes with loop control

**Usage:**
```bash
supervisor run-graph examples/workflows/debug_workflow.yaml --workflow-id debug-session-1
```

**Key Features:**
- Loop control with `max_iterations: 3`
- Conditional edges based on test results
- Gate node for test execution

### 3. Parallel Review Workflow (`parallel_review.yaml`)

Demonstrates parallel execution and merge:
- Implement → Split into 3 parallel gates (lint, test, security)
- Join results → Human approval → Merge
- Shows PARALLEL, MERGE, and HUMAN nodes

**Usage:**
```bash
supervisor run-graph examples/workflows/parallel_review.yaml --workflow-id feature-xyz
```

**Key Features:**
- Parallel gate execution
- Merge strategies (union)
- Human approval point
- Conditional merge based on approval

## Validation Only

To validate a workflow without executing it:

```bash
supervisor run-graph examples/workflows/simple_workflow.yaml --workflow-id test --validate-only
```

## Node Types Reference

- **TASK**: Execute a role (planner, implementer, reviewer, debugger)
- **GATE**: Run verification gates (test, lint, security)
- **BRANCH**: Conditional branching with loop support
- **MERGE**: Synchronization point for parallel branches
- **PARALLEL**: Fan-out to multiple parallel branches
- **HUMAN**: Human approval/intervention point
- **SUBGRAPH**: Nested workflow (not yet implemented)

## Edge Conditions

Edges can have conditions to control flow:

```yaml
edges:
  - id: e1
    source: check_node
    target: next_node
    condition:
      field: test_status
      operator: "=="
      value: "passed"
```

**Supported operators:**
- `==`, `!=`, `>`, `<`, `>=`, `<=`
- `in`, `not_in`
- `contains`, `starts_with`, `ends_with`

## Loop Control

Branch nodes support loop control to prevent infinite loops:

```yaml
branch_config:
  condition:
    field: status
    operator: "!="
    value: "success"
    max_iterations: 3  # Prevent infinite loops
  on_true: retry_node
  on_false: success_node
```

## Configuration

Global workflow config:

```yaml
config:
  max_parallel_nodes: 4  # Maximum concurrent node execution
  fail_fast: true        # Stop on first failure
```

## Tips

1. **Start Simple**: Use `simple_workflow.yaml` as a template
2. **Validate First**: Always use `--validate-only` to check for errors
3. **Loop Safety**: Always set `max_iterations` for loops
4. **Exit Points**: Ensure your workflow has clear exit points
5. **Gate Safety**: Gates run BEFORE changes are applied to main repo

## Creating Custom Workflows

See the plan document at `docs/PLANS/SUPERVISOR_STUDIO_PHASE1.md` for complete schema reference and advanced patterns.
