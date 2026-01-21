# Upgrades

## Update the Tool

```bash
# From the repo root
pip install -e .
```

## Post-Upgrade Checklist

- Run `supervisor --version`
- Re-run `supervisor roles` to ensure role configs load
- Verify `.supervisor/` config files still match your environment
- Run a small plan/implement cycle to validate gates

## Schema Changes

If `.supervisor/state.db` schema changes, review the release notes and apply any migration steps described there.
