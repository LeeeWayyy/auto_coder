# Daily Usage

These are the most common commands for day-to-day work.

## Plan → Implement → Review

```bash
# Plan
supervisor plan "Add user authentication with JWT"

# Implement
supervisor run implementer "Add login endpoint" -t src/api/auth.py
supervisor run implementer "Add JWT utilities" -t src/utils/jwt.py

# Review
supervisor run reviewer "Review authentication implementation"
```

## Check Status

```bash
supervisor status
```

## View Metrics

```bash
supervisor metrics
supervisor metrics --days 7
supervisor metrics --role implementer
```

## Role Inventory

```bash
supervisor roles
```

## Graph Workflows (YAML)

```bash
# Validate and visualize a workflow graph
supervisor visualize examples/workflows/basic_workflow.yaml

# Execute a graph
supervisor run-graph examples/workflows/basic_workflow.yaml --workflow-id wf-1234
```

## Inspect Executions

```bash
# Summary view
supervisor status --execution-id EXEC_ID

# Inspect node details
supervisor inspect EXEC_ID --node node_1

# Interactive inspector
supervisor inspect EXEC_ID --interactive
```

## Tips

- Keep task prompts short and specific.
- Use `-t` to target files and reduce context size.
- Prefer smaller scopes per run for faster verification gates.
