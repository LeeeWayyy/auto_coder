# Supervisor - AI CLI Orchestrator

Treats AI CLIs (Claude, Codex, Gemini) as **stateless workers** to prevent context dilution in long workflows.

## The Problem

When using AI CLI tools for long development workflows, **context dilution** causes the AI to progressively ignore instructions. The larger the context window grows, the less reliably the AI follows the original rules.

## The Solution

For every step of your workflow, spin up a **fresh short-lived instance**, feed it only the relevant rules/context, get the output, and kill it.

## Quick Start

```bash
# Install
pip install -e .

# Initialize project
supervisor init

# Plan a feature
supervisor plan "Add user authentication with JWT"

# Run a specific role
supervisor run implementer "Add login endpoint" -t src/api/auth.py

# List available roles
supervisor roles
```

## Architecture

```
                        ┌──────────────┐
                        │  Supervisor  │
                        │     CLI      │
                        └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │  Claude  │    │  Codex   │    │  Gemini  │
        │   CLI    │    │   CLI    │    │   CLI    │
        └──────────┘    └──────────┘    └──────────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                        ┌──────────────┐
                        │    SQLite    │
                        │  Event Log   │
                        └──────────────┘
```

## Key Features

- **Stateless Workers**: Fresh CLI instance per task step
- **SQLite + Event Sourcing**: Full audit trail, reproducible state
- **Docker Sandboxing**: Isolated execution with egress allowlist
- **Git Worktree Isolation**: Clean workspace per step, atomic commits
- **Schema-Enforced Outputs**: Pydantic validation, no magic strings
- **Circuit Breakers**: Prevent infinite retry loops

## Documentation

See `docs/PLANS/SUPERVISOR_ORCHESTRATOR.md` for the full architecture plan.

## License

MIT
