# Docker Sandboxing

**File:** `supervisor/sandbox/executor.py`

## Overview

Provides sandboxed execution for AI CLIs and arbitrary commands using Docker containers. Two container types with different security profiles:

1. **SandboxedLLMClient**: For AI CLI calls (needs network egress to APIs)
2. **SandboxedExecutor**: For tests/commands (no network, fully isolated)

## Key Classes

### `SandboxConfig`

Configuration for sandbox containers:

```python
@dataclass
class SandboxConfig:
    cli_image: str = "supervisor-cli:latest"
    executor_image: str = "supervisor-executor:latest"
    egress_network: str = "supervisor-egress"
    memory_limit: str = "4g"
    cpu_limit: str = "2"
    cli_timeout: int = 300  # 5 minutes
    executor_timeout: int = 600  # 10 minutes
    allowed_egress: list[str]  # API endpoints
    require_docker: bool = True
    verify_egress_rules: bool = True
    allowed_workdir_roots: list[str]  # Required, cannot be empty
```

### `SandboxedLLMClient`

Executes AI CLI in isolated container with controlled egress.

```python
client = SandboxedLLMClient(cli_name="claude", config=config)
result = client.execute(prompt, workdir)
```

**Supported CLIs:**
- `claude`: Argument-based (`claude -p <prompt>`)
- `codex`: Stdin-based (`codex exec --stdin`)
- `gemini`: Stdin-based (`gemini -o json`)

### `SandboxedExecutor`

Executes arbitrary commands with NO network access.

```python
executor = SandboxedExecutor(config)
result = executor.run(command=["make", "test"], workdir=worktree_path)
```

### `ExecutionResult`

Result of sandboxed execution:

```python
class ExecutionResult(BaseModel):
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
```

### `LocalExecutor`

**UNSAFE** fallback for unit tests only. Requires `SUPERVISOR_ALLOW_LOCAL_EXECUTOR=1` environment variable.

## Security Features

### Docker Hardening

All containers run with:
- `--read-only`: Read-only root filesystem
- `--cap-drop=ALL`: No Linux capabilities
- `--security-opt=no-new-privileges:true`: No privilege escalation
- `--tmpfs=/tmp:size=1g`: Writable temp space
- `--tmpfs=/home:size=100m`: Writable home directory
- `--env=HOME=/home`: Set HOME to tmpfs mount

### Network Isolation

- **CLI containers**: Connected to `supervisor-egress` network
- **Executor containers**: `--network=none` (no network access)

### Workdir Validation

```python
_validate_workdir(workdir, allowed_roots)
```

- **Mandatory**: `allowed_workdir_roots` cannot be empty
- Prevents mounting arbitrary host paths into containers
- Validates existence and directory status before mount

### Prompt Security

For argument-based CLIs (claude):
- Prompt written to temp file via stdin (not visible in `ps`)
- Variable assignment prevents command injection: `PROMPT="$(cat file)" && cli "$PROMPT"`
- ARG_MAX enforcement (1.5MB limit)

## Container Registry

Global registry for container cleanup on exit:

```python
class ContainerRegistry:
    def add(container_id: str)
    def remove(container_id: str)
    def cleanup_all()  # Called on atexit, SIGTERM, SIGINT
```

Uses `threading.RLock` to prevent deadlock in signal handlers.

## Error Classes

- `SandboxError`: General sandbox error
- `DockerNotAvailableError`: Docker not installed or not running
- `EgressNotConfiguredError`: Egress allowlist not properly configured

## Platform Support

- **Linux/macOS**: Uses `--user=UID:GID` for file permissions
- **Windows**: Skips `--user` flag (Docker Desktop handles permissions)
- Override via `SUPERVISOR_DOCKER_USER` environment variable
