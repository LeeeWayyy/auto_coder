# Workflows

Workflows orchestrate multiple roles in sequence (or parallel) using the Supervisor engine.

## Run a Full Workflow

```bash
supervisor workflow feat-authentication --tui --parallel
```

## Declarative Graph Workflows

Supervisor also supports declarative YAML graphs (used by Studio):

```bash
supervisor run-graph examples/workflows/basic_workflow.yaml --workflow-id wf-1234
supervisor run-graph examples/workflows/basic_workflow.yaml --workflow-id wf-1234 --live
```

Use `supervisor visualize` to render a graph in the terminal.

## Typical Phases

1. Plan feature
2. Implement components
3. Run gates
4. Review changes

## Parallel Execution

Enable parallel execution in `.supervisor/config.yaml`:
```yaml
workflow:
  parallel_execution: true
```

## Workflow Conventions

- Use a stable workflow ID for large features.
- Keep components small enough to be reviewed and gated quickly.
- Avoid long-running gates in a single component; split if needed.
