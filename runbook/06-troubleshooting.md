# Troubleshooting

## Docker Not Available

**Symptom**:
```
Error: Docker is not available
```

**Fix**:
```bash
docker --version
docker ps
```
If the daemon is not running, start Docker and retry.

## Gate Failures

**Symptom**:
```
✗ test failed (exit code 1)
```

**Fix**:
- Read the gate output
- Fix the failing tests/lint
- Re-run the command

## Context Budget Exceeded

**Symptom**:
```
Warning: Context exceeds budget, pruning files...
```

**Fix**:
- Target files with `-t`
- Increase `context.token_budget` in your role config

## CLI Not Found

**Symptom**:
```
command not found: supervisor
```

**Fix**:
```bash
pip install -e .
```

## Studio Won’t Start (uvicorn missing)

**Symptom**:
```
Error: uvicorn is required for the studio.
```

**Fix**:
```bash
pip install uvicorn
```

## Studio UI Loads Blank

**Likely cause**: Frontend build missing.

**Fix**:
```bash
cd supervisor/studio/frontend
npm install
npm run build
```

## Common Next Steps

- Check `.supervisor/state.db` for recent events
- Review `.supervisor/config.yaml` for model and gate settings
- Run `supervisor roles` to ensure role configs are valid
