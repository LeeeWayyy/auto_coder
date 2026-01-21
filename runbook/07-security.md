# Security & Approvals

## Sandbox Isolation

Supervisor uses Docker containers and git worktrees to isolate execution. Keep Docker updated and restrict outbound access to required AI endpoints.

## Approval Policies

Configure approvals in `.supervisor/approval.yaml`:
```yaml
approval:
  always_approve:
    - "src/database/migrations/**"
    - "*.sql"
  never_approve:
    - "tests/**"
  risk_threshold:
    low: 50
    medium: 200
    high: 500
```

## Gates

Use gates to enforce verification outside the AI output:
- `test`
- `lint`
- `type_check`
- `security`

## Secrets

- Do not commit API keys
- Prefer environment variables or a local `.env`
- Restrict `.supervisor/` configuration as needed for your org
