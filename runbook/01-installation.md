# Installation

This runbook assumes you are installing Auto Coder locally and using the Supervisor CLI.

## Prerequisites

- Python 3.11+
- Docker (required for sandboxed execution)
- Git
- At least one AI CLI configured (Claude, Codex, Gemini)

Verify:
```bash
python --version
docker --version
git --version
```

## Install From Source

```bash
# From the repo root
pip install -e .

# Confirm CLI
supervisor --version
```

If `npm` is available, the Studio frontend is built automatically during install.
If npm is not available, run the dev server or build manually (see `runbook/09-studio-web-console.md`).

## Environment Variables

Set API keys for the CLIs you plan to use:
```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

## Optional: Dev Dependencies

```bash
pip install -e ".[dev]"
```

## Smoke Test

```bash
supervisor roles
```

Expected: the base roles list (planner, implementer, reviewer).

## First Run Preflight (New)

On first run, the CLI will:
- create the `supervisor-egress` Docker network if missing
- prompt to build missing sandbox images (CLI + executor)
- verify the CLI binaries inside the sandbox image

You can auto-build without a prompt:
```bash
SUPERVISOR_AUTO_BUILD_IMAGES=1 supervisor roles
```

To skip the CLI binary check (not recommended):
```bash
SUPERVISOR_SKIP_CLI_CHECKS=1 supervisor roles
```
