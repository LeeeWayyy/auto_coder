# Supervisor - AI CLI Orchestrator

[![Tests](https://github.com/LeeeWayyy/auto_coder/workflows/Tests/badge.svg)](https://github.com/LeeeWayyy/auto_coder/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Treats AI CLIs (Claude, Codex, Gemini) as **stateless workers** to prevent context dilution in long workflows.

> **AI-powered code orchestration with event sourcing, Docker isolation, and verification gates**

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

**Getting Started**:
- [Getting Started Guide](docs/GETTING_STARTED.md) - Your first workflow in 5 minutes
- [Examples](examples/) - Practical workflow examples

**Reference**:
- [CLI Reference](docs/CLI_REFERENCE.md) - Complete command documentation
- [Architecture](docs/ARCHITECTURE.md) - System design and internals
- [FAQ](docs/FAQ.md) - Common questions and troubleshooting

**Operations**:
- [Operations Guide](docs/OPERATIONS.md) - Production deployment and monitoring
- [Contributing Guide](CONTRIBUTING.md) - Development and contribution guide

**Design Documents**:
- [Full Architecture Plan](docs/PLANS/SUPERVISOR_ORCHESTRATOR.md) - Complete technical specification

## Quick Links

- [Report an Issue](https://github.com/LeeeWayyy/auto_coder/issues)
- [View Changelog](CHANGELOG.md)
- [License](LICENSE)

## License

MIT
