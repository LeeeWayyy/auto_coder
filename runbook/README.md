# Auto Coder Runbook

Operational runbook for **Auto Coder** (the Supervisor CLI). Use this folder as the step-by-step guide for setup, daily usage, and production operations.

## Start Here

1. [Installation](01-installation.md)
2. [Initial Project Setup](02-initial-setup.md)
3. [Daily Usage](03-daily-usage.md)
4. [Workflows](04-workflows.md)
5. [Operations](05-operations.md)
6. [Troubleshooting](06-troubleshooting.md)
7. [Security & Approvals](07-security.md)
8. [Upgrades](08-upgrades.md)
9. [Studio Web Console](09-studio-web-console.md)

## Quick Command Map

```bash
# Initialize in a repo
supervisor init

# Plan a feature
supervisor plan "Add user authentication"

# Implement a task
supervisor run implementer "Add login endpoint" -t src/api/auth.py

# Review
supervisor run reviewer "Review recent changes"

# Full workflow
supervisor workflow feat-authentication --tui --parallel

# Status and metrics
supervisor status
supervisor metrics --days 7

# Studio web console
supervisor studio --port 8000
```

## Related Docs

- [README](../README.md)
- [Getting Started](../docs/GETTING_STARTED.md)
- [CLI Reference](../docs/CLI_REFERENCE.md)
- [Operations Guide](../docs/OPERATIONS.md)
- [FAQ](../docs/FAQ.md)
- [Examples](../examples/README.md)
