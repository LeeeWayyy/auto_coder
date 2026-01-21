# Operations

Operational guidance for running Auto Coder in shared or production environments.

## Core Requirements

- Docker available and running
- Stable outbound HTTPS to AI API endpoints
- Disk space for worktrees and Docker images

## Startup Checklist

- Verify Docker daemon (`docker ps`)
- Export required API keys
- Ensure `.supervisor/` config files are present
- Confirm gates align with repo tooling

## Monitoring

```bash
supervisor metrics
supervisor status
```

Track:
- Gate pass/fail rate
- Execution time per role
- Retry counts

## Studio Web Console (Optional)

- Requires `uvicorn` and (for dev) Node.js + npm.
- **Bind to localhost only**; Studio has no authentication.
- See `runbook/09-studio-web-console.md` for startup steps.

## Backups

Back up the event store regularly:
- `.supervisor/state.db`

## Cleanup

If a run is interrupted, remove stale worktrees and retry:
```bash
# Example cleanup (adjust for your environment)
rm -rf /tmp/supervisor_*
```

## Production Tips

- Use dedicated Docker images for deterministic tooling
- Enable approval policies for high-risk paths
- Pin model versions in role configs
