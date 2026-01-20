# Initial Project Setup

Use this when onboarding a new repository.

## Initialize Supervisor

```bash
cd /path/to/your/repo
supervisor init
```

This creates `.supervisor/` with default configuration and the SQLite event store.

## Key Files to Review

- `.supervisor/config.yaml` - default CLI, workflow behavior, git settings
- `.supervisor/limits.yaml` - timeouts
- `.supervisor/adaptive.yaml` - model routing (optional)
- `.supervisor/approval.yaml` - approval policies
- `.supervisor/gates.yaml` - verification gates (optional)
- `.supervisor/roles/` - custom role definitions

## Minimal Configuration Checklist

- Set `default_cli` in `.supervisor/config.yaml`
- Confirm gate commands match your repo tooling
- Update approval policies for sensitive paths

## First Plan

```bash
supervisor plan "Add a simple healthcheck endpoint"
```

If planning succeeds, you are ready to run implementation roles.

## Start The System (Quick Path)

```bash
# 1) Install + preflight (prompts to build Docker images if missing)
supervisor roles

# 2) Start Studio (optional web UI)
supervisor studio
```

If you prefer automatic image builds without prompts:
```bash
SUPERVISOR_AUTO_BUILD_IMAGES=1 supervisor studio
```
