"""Sandboxed execution for AI CLIs and arbitrary commands.

Two container types:
1. SandboxedLLMClient - For AI CLI calls (needs network egress to APIs)
2. SandboxedExecutor - For tests/commands (no network, fully isolated)

SECURITY: Docker is REQUIRED. LocalExecutor is only for unit tests.
"""

import atexit
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class SandboxError(Exception):
    """Error in sandbox execution."""

    pass


class DockerNotAvailableError(SandboxError):
    """Docker is required but not available."""

    pass


class EgressNotConfiguredError(SandboxError):
    """Egress allowlist is not properly configured."""

    pass


def _verify_egress_rules(network_name: str, allowed_egress: list[str]) -> bool | None:
    """Verify that iptables/nftables egress rules exist for the Docker network.

    SECURITY: This is a best-effort check.

    Returns:
        True: Egress rules appear to be configured
        False: Rules definitely not configured (network exists but no rules)
        None: Cannot verify (permission denied, commands missing) - caller should warn

    NOTE: iptables/nftables commands typically require root privileges.
    When running as non-root, this returns None to indicate verification wasn't possible.
    """
    import json
    import sys

    # Get the bridge interface for the Docker network
    try:
        result = subprocess.run(
            ["docker", "network", "inspect", network_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False

        network_info = json.loads(result.stdout)
        if not network_info:
            return False

        # Get the bridge name (e.g., "br-abc123")
        options = network_info[0].get("Options", {})
        bridge_name = options.get("com.docker.network.bridge.name")
        if not bridge_name:
            # Docker generates a default name like "br-<network_id[:12]>"
            network_id = network_info[0].get("Id", "")[:12]
            bridge_name = f"br-{network_id}"

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, IndexError):
        return False

    # Check for iptables rules on the bridge interface
    # Look for FORWARD chain rules that restrict egress
    permission_denied = False
    try:
        # Try iptables first
        result = subprocess.run(
            ["iptables", "-L", "FORWARD", "-n", "-v"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Check if there are DROP rules or restrictive rules for the bridge
            output = result.stdout.lower()
            if bridge_name.lower() in output and ("drop" in output or "reject" in output):
                return True
        elif "permission denied" in result.stderr.lower() or result.returncode == 4:
            # iptables exit code 4 or "permission denied" = needs root
            permission_denied = True

        # Try nftables if iptables check didn't find rules
        result = subprocess.run(
            ["nft", "list", "ruleset"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if bridge_name.lower() in output and ("drop" in output or "reject" in output):
                return True
        elif "permission denied" in result.stderr.lower():
            permission_denied = True

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # If we couldn't verify due to permission issues, return None
    # to indicate verification wasn't possible (not necessarily that rules are missing)
    if permission_denied:
        print(
            "WARNING: Cannot verify egress rules (permission denied - requires root). "
            "Assuming egress is configured. Verify manually if needed.",
            file=sys.stderr,
        )
        return None

    return False


class ExecutionResult(BaseModel):
    """Result of a sandboxed execution."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _truncate_output(output: str, max_bytes: int) -> str:
    """Truncate output to max_bytes, adding truncation notice if needed.

    Prevents downstream memory issues from unbounded command output.
    """
    if len(output.encode("utf-8", errors="replace")) <= max_bytes:
        return output

    # Truncate by bytes, ensuring we don't cut in the middle of a UTF-8 sequence
    truncated = output.encode("utf-8", errors="replace")[:max_bytes].decode(
        "utf-8", errors="ignore"
    )
    return truncated + f"\n\n[OUTPUT TRUNCATED - exceeded {max_bytes} bytes]"


@dataclass
class SandboxConfig:
    """Configuration for sandbox containers."""

    # Docker images
    cli_image: str = "supervisor-cli:latest"
    executor_image: str = "supervisor-executor:latest"

    # Network settings
    egress_network: str = "supervisor-egress"

    # Resource limits
    memory_limit: str = "4g"
    cpu_limit: str = "2"

    # Timeouts
    cli_timeout: int = 300  # 5 minutes for CLI calls
    executor_timeout: int = 600  # 10 minutes for tests

    # Output limits (prevent OOM from unbounded output)
    max_output_bytes: int = 10 * 1024 * 1024  # 10MB max stdout/stderr

    # Process limits (prevent fork-bomb DoS)
    pids_limit: int = 256  # Max PIDs per container
    ulimit_nproc: int = 256  # Max processes

    # Egress allowlist for CLI containers (must be enforced via iptables)
    allowed_egress: list[str] = field(
        default_factory=lambda: [
            "api.anthropic.com:443",
            "api.openai.com:443",
            "generativelanguage.googleapis.com:443",
        ]
    )

    # Security settings
    require_docker: bool = True  # Fail if Docker unavailable
    verify_egress_rules: bool = True  # Verify iptables rules exist
    # SECURITY: Default to fail-closed when egress rules can't be verified
    # Set to False only if you've manually verified iptables/nftables rules
    fail_on_unverified_egress: bool = True

    # Allowed workdir roots - if set, workdir must be under one of these paths
    # This prevents mounting arbitrary host paths into containers
    allowed_workdir_roots: list[str] = field(default_factory=list)


class ContainerRegistry:
    """Track running containers for cleanup on exit.

    Uses RLock (reentrant lock) to prevent deadlock when signal handlers
    call cleanup_all() while the lock is already held by the same thread.

    On initialization, cleans up ONLY exited/dead autocoder-supervisor-* containers
    from previous runs. Does NOT touch running containers to prevent
    killing active containers from other concurrent supervisor instances.
    """

    def __init__(self) -> None:
        self._containers: set[str] = set()
        self._lock = threading.RLock()  # RLock allows same thread to re-acquire

        # Clean up ONLY exited orphaned containers from previous runs
        self._cleanup_orphaned_containers()

        # Register cleanup handlers
        atexit.register(self.cleanup_all)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _cleanup_orphaned_containers(self) -> None:
        """Remove exited/dead autocoder-supervisor-* containers from previous runs.

        IMPORTANT: Only cleans up containers that are NOT running.
        This prevents killing active containers from other concurrent
        supervisor instances (e.g., parallel CI jobs, multiple developers).

        Called on startup to clean up containers that were left behind
        due to SIGKILL, power loss, or other hard termination where
        atexit handlers don't run.
        """
        try:
            # List ONLY exited/dead containers with autocoder-supervisor-* prefix
            # Using --filter status=exited and status=dead to avoid killing running ones
            result = subprocess.run(
                [
                    "docker", "ps", "-a",
                    "--filter", "name=autocoder-supervisor-",
                    "--filter", "status=exited",
                    "--format", "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return  # Docker not available or error, skip cleanup

            exited = [name.strip() for name in result.stdout.strip().split("\n") if name.strip()]

            # Also check for dead containers
            result_dead = subprocess.run(
                [
                    "docker", "ps", "-a",
                    "--filter", "name=autocoder-supervisor-",
                    "--filter", "status=dead",
                    "--format", "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result_dead.returncode == 0:
                dead = [name.strip() for name in result_dead.stdout.strip().split("\n") if name.strip()]
                exited.extend(dead)

            if not exited:
                return

            # Remove only exited/dead containers (safe - they're not running)
            removed_count = 0
            for container_name in exited:
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        capture_output=True,
                        timeout=5,
                    )
                    removed_count += 1
                except Exception:
                    pass

            if removed_count > 0:
                import sys
                print(
                    f"Cleaned up {removed_count} exited supervisor containers from previous run.",
                    file=sys.stderr,
                )

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # Docker not available or timed out

    def add(self, container_id: str) -> None:
        with self._lock:
            self._containers.add(container_id)

    def remove(self, container_id: str) -> None:
        with self._lock:
            self._containers.discard(container_id)

    def cleanup_all(self) -> None:
        """Kill all tracked containers."""
        with self._lock:
            for container_id in list(self._containers):
                try:
                    subprocess.run(
                        ["docker", "kill", container_id],
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    pass
            self._containers.clear()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self.cleanup_all()
        raise SystemExit(128 + signum)


# Global container registry
_registry = ContainerRegistry()


def _validate_docker() -> None:
    """Validate Docker is available. Raises if not."""
    if not shutil.which("docker"):
        raise DockerNotAvailableError("Docker binary not found in PATH")

    result = subprocess.run(
        ["docker", "version"],
        capture_output=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise DockerNotAvailableError(
            f"Docker is not running or not accessible: {result.stderr.decode()}"
        )


def _sanitize_path_component(name: str) -> str:
    """Sanitize a string for use as a path component.

    Prevents path traversal attacks by removing/replacing dangerous characters.
    """
    # Replace path separators and null bytes
    sanitized = re.sub(r"[/\\\x00]", "-", name)
    # Remove leading dots (hidden files / parent traversal)
    sanitized = sanitized.lstrip(".")
    # Limit length
    sanitized = sanitized[:64]
    # If empty after sanitization, use UUID
    if not sanitized:
        sanitized = uuid.uuid4().hex[:16]
    return sanitized


def _sanitize_container_name_component(name: str) -> str:
    """Sanitize a string for use in Docker container names.

    Docker container names must match: [a-zA-Z0-9][a-zA-Z0-9_.-]*

    Prevents DoS where invalid characters cause Docker to fail.
    """
    # Replace invalid characters with dashes
    # Docker allows: a-z, A-Z, 0-9, underscore, period, hyphen
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "-", name)
    # Container names must start with alphanumeric
    sanitized = sanitized.lstrip("_.-")
    # Collapse multiple dashes
    sanitized = re.sub(r"-+", "-", sanitized)
    # Limit length (Docker has a limit, but we'll use 64 to be safe)
    sanitized = sanitized[:64]
    # If empty after sanitization, use a default
    if not sanitized:
        sanitized = "cli"
    return sanitized.lower()


def _get_docker_user() -> str | None:
    """Get the current user's UID:GID for Docker --user flag.

    Uses the actual host user's UID/GID to avoid permission issues
    with files created in mounted volumes.

    Returns:
        "UID:GID" string on Linux/macOS, or None on Windows.
        Can be overridden via SUPERVISOR_DOCKER_USER env var.

    On Windows, returns None because Docker Desktop handles file
    permissions automatically and --user can cause issues.
    """
    # Allow override via environment variable
    env_override = os.environ.get("SUPERVISOR_DOCKER_USER")
    if env_override:
        return env_override

    # Windows doesn't have getuid/getgid and Docker Desktop handles permissions
    if os.name == "nt":
        return None

    # Linux/macOS: use actual UID:GID
    uid = os.getuid()
    gid = os.getgid()
    return f"{uid}:{gid}"


def _validate_workdir(workdir: Path, allowed_roots: list[str]) -> Path:
    """Validate workdir is under one of the allowed root paths.

    SECURITY: Prevents mounting arbitrary host paths into containers.
    Workdir should only be a worktree directory, not arbitrary paths.

    TOCTOU MITIGATION: Returns the resolved path to be used for Docker mount.
    Caller MUST use the returned path (not the original) to minimize the
    race window between validation and Docker mount.

    Args:
        workdir: Absolute path to the working directory
        allowed_roots: List of allowed parent directories

    Returns:
        The resolved (canonical) path to use for Docker mount

    Raises:
        SandboxError: If workdir is not under any allowed root or if
                      allowed_roots is empty (security footgun)
    """
    if not allowed_roots:
        raise SandboxError(
            "allowed_workdir_roots is empty. "
            "SECURITY: You must specify allowed roots to prevent mounting arbitrary host paths. "
            "Example: SandboxConfig(allowed_workdir_roots=['/path/to/repo/.worktrees'])"
        )

    # SECURITY: Reject if workdir itself is a symlink
    if workdir.is_symlink():
        raise SandboxError(
            f"SECURITY: Workdir is a symlink: {workdir}. "
            "Symlinks are not allowed to prevent mount redirection attacks."
        )

    # SECURITY: Check all ancestors for symlinks (TOCTOU mitigation)
    current = workdir.parent
    while current != current.parent:  # Stop at root
        if current.is_symlink():
            raise SandboxError(
                f"SECURITY: Workdir ancestor is a symlink: {current}. "
                "Symlinks in path are not allowed to prevent mount redirection attacks."
            )
        current = current.parent

    workdir_resolved = workdir.resolve()

    for root in allowed_roots:
        root_resolved = Path(root).resolve()
        try:
            workdir_resolved.relative_to(root_resolved)
            # TOCTOU MITIGATION: Return resolved path for Docker mount
            return workdir_resolved
        except ValueError:
            continue  # Not under this root, try next

    raise SandboxError(
        f"Workdir '{workdir}' is not under any allowed root. "
        f"Allowed roots: {allowed_roots}"
    )


class SandboxedLLMClient:
    """Execute AI CLI in isolated container with controlled egress.

    CLI containers need network access to reach model APIs, but are otherwise
    isolated. Uses an egress allowlist to restrict outbound connections.

    IMPORTANT: Egress allowlist must be enforced via iptables rules on the host.
    This class creates the network but cannot enforce egress rules without root.
    See docs/PLANS/SUPERVISOR_ORCHESTRATOR.md for iptables configuration.

    Updated (v28): Added model_id support for granular model selection.
    """

    # ANSI escape code pattern for stripping terminal colors
    ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def __init__(
        self,
        cli_name: str,
        config: SandboxConfig | None = None,
        model_id: str | None = None,
    ):
        """Initialize sandboxed CLI client.

        Args:
            cli_name: CLI binary name (claude, codex, gemini)
            config: Sandbox configuration
            model_id: Optional model ID to pass via --model flag
                      (e.g., "claude-opus-4-5-20251101", "gpt-5.2-codex")
        """
        self.cli_name = cli_name
        self.model_id = model_id
        self.config = config or SandboxConfig()
        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate Docker is available, network exists, and egress rules configured.

        SECURITY: If verify_egress_rules is True, this method will verify that
        iptables/nftables rules exist for the egress network. If not, it raises
        EgressNotConfiguredError to fail closed (rather than running insecurely).
        """
        if self.config.require_docker:
            _validate_docker()
        self._ensure_network_exists()

        # SECURITY: Verify egress rules if configured to do so
        if self.config.verify_egress_rules:
            egress_status = _verify_egress_rules(self.config.egress_network, self.config.allowed_egress)
            if egress_status is False:
                # Definitely not configured - fail closed
                raise EgressNotConfiguredError(
                    f"Egress rules not configured for network '{self.config.egress_network}'. "
                    f"SECURITY: CLI containers may have unrestricted network access. "
                    f"Configure iptables/nftables rules for the egress allowlist, "
                    f"or set verify_egress_rules=False to bypass (NOT RECOMMENDED). "
                    f"See docs/PLANS/SUPERVISOR_ORCHESTRATOR.md for configuration."
                )
            elif egress_status is None and self.config.fail_on_unverified_egress:
                # Couldn't verify - fail closed if configured to do so
                raise EgressNotConfiguredError(
                    f"Cannot verify egress rules for network '{self.config.egress_network}' "
                    f"(permission denied - requires root). "
                    f"SECURITY: fail_on_unverified_egress is True, refusing to proceed. "
                    f"Either run with root privileges to verify rules, "
                    f"or set fail_on_unverified_egress=False if you've verified rules manually."
                )
            # egress_status is True (rules found) or None with fail_on_unverified_egress=False

    def _ensure_network_exists(self) -> None:
        """Ensure the egress network exists.

        Handles race conditions where multiple processes try to create the
        network simultaneously by tolerating "already exists" errors.
        """
        result = subprocess.run(
            ["docker", "network", "inspect", self.config.egress_network],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            # Try to create the network
            create_result = subprocess.run(
                [
                    "docker", "network", "create",
                    "--driver", "bridge",
                    "--opt", "com.docker.network.bridge.enable_icc=false",
                    self.config.egress_network,
                ],
                capture_output=True,
                timeout=10,
            )

            # Check if creation failed
            if create_result.returncode != 0:
                stderr = create_result.stderr.decode() if create_result.stderr else ""
                # Tolerate "already exists" error (race condition with another process)
                if "already exists" not in stderr.lower():
                    # Verify network exists now (another process may have created it)
                    verify = subprocess.run(
                        ["docker", "network", "inspect", self.config.egress_network],
                        capture_output=True,
                        timeout=5,
                    )
                    if verify.returncode != 0:
                        raise SandboxError(
                            f"Failed to create network '{self.config.egress_network}': {stderr}"
                        )
            else:
                # Network was created successfully, warn about egress rules
                import sys
                print(
                    f"WARNING: Created network '{self.config.egress_network}' but egress "
                    f"rules must be configured manually via iptables. See documentation.",
                    file=sys.stderr,
                )

    def _get_cli_config(self) -> tuple[list[str], bool]:
        """Get CLI command and whether it uses stdin for prompt.

        Updated (v28): Includes --model flag when model_id is set.

        Returns:
            Tuple of (command_args, uses_stdin).
            If uses_stdin=False, prompt is appended as final argument.

        NOTE: CLI stdin support varies:
        - claude: Requires prompt as argument after -p (uses_stdin=False)
        - codex: Reads from stdin with --stdin flag (uses_stdin=True)
        - gemini: Reads from stdin (uses_stdin=True)

        Model flag formats:
        - claude: --model <model_id> (before -p)
        - codex: --model <model_id> (after exec)
        - gemini: --model <model_id> (before -o)
        """
        # Base configs without model flag
        base_configs: dict[str, tuple[list[str], bool]] = {
            # Claude Code CLI: prompt must be argument, not stdin
            "claude": (["claude", "-p"], False),
            # Codex: supports stdin with --stdin flag
            "codex": (["codex", "exec", "--json", "--stdin"], True),
            # Gemini: reads prompt from stdin
            "gemini": (["gemini", "-o", "json"], True),
        }

        cmd_args, uses_stdin = base_configs.get(self.cli_name, ([self.cli_name], True))

        # Add model flag if model_id is specified
        # Insert into existing command list to avoid duplication with base_configs
        if self.model_id:
            model_flag = ["--model", self.model_id]
            if self.cli_name == "codex":
                # codex exec --model <id> --json --stdin (insert after 'exec')
                cmd_args = cmd_args[:2] + model_flag + cmd_args[2:]
            else:
                # For claude, gemini, and generic: insert after cli name
                cmd_args = cmd_args[:1] + model_flag + cmd_args[1:]

        return cmd_args, uses_stdin

    def execute(
        self,
        prompt: str,
        workdir: Path | str,
    ) -> ExecutionResult:
        """Execute CLI with prompt in sandboxed container.

        SECURITY NOTES:
        - For stdin-capable CLIs: prompt passed via stdin (not visible in `ps`)
        - For argument-based CLIs (e.g., claude -p): prompt passed via env var
          and shell expansion to avoid `ps` exposure

        Args:
            prompt: The prompt to send to the CLI
            workdir: Working directory to mount (should be a worktree, not main repo)

        Returns:
            ExecutionResult with stdout, stderr, and return code
        """
        workdir = Path(workdir).absolute()

        # SECURITY: Validate workdir is under allowed roots
        # Returns resolved path to use for Docker mount (TOCTOU mitigation)
        workdir_resolved = _validate_workdir(workdir, self.config.allowed_workdir_roots)

        # Validate workdir exists and is a directory before mounting
        # Use resolved path for consistency
        if not workdir_resolved.exists():
            raise SandboxError(f"Workdir does not exist: {workdir_resolved}")
        if not workdir_resolved.is_dir():
            raise SandboxError(f"Workdir is not a directory: {workdir_resolved}")

        # Sanitize cli_name for Docker container name to prevent DoS from invalid chars
        # Use specific prefix to avoid collision with other tools
        safe_cli_name = _sanitize_container_name_component(self.cli_name)
        container_name = f"autocoder-supervisor-cli-{safe_cli_name}-{uuid.uuid4().hex[:8]}"

        cli_args, uses_stdin = self._get_cli_config()

        # Base docker command
        # TOCTOU MITIGATION: Use resolved path for Docker mount
        cmd = [
            "docker", "run",
            "--rm",
            f"--name={container_name}",
            f"--network={self.config.egress_network}",
            # Mount ONLY the specified workdir (should be worktree, not main repo)
            f"--volume={workdir_resolved}:/workspace:rw",
            "--workdir=/workspace",
        ]

        # Add --user flag only on Linux/macOS (Windows Docker Desktop handles this)
        docker_user = _get_docker_user()
        if docker_user:
            cmd.append(f"--user={docker_user}")

        cmd.extend([
            "--read-only",
            "--tmpfs=/tmp:size=1g",
            "--tmpfs=/home:size=100m",
            f"--memory={self.config.memory_limit}",
            f"--cpus={self.config.cpu_limit}",
            # Process limits (prevent fork-bomb DoS)
            f"--pids-limit={self.config.pids_limit}",
            f"--ulimit=nproc={self.config.ulimit_nproc}:{self.config.ulimit_nproc}",
            # Security options
            "--security-opt=no-new-privileges:true",
            "--cap-drop=ALL",
            # Environment variables
            # HOME must match tmpfs mount so CLI tools can write config/cache
            "--env=HOME=/home",
            # API keys passed through from host
            # SECURITY NOTE: Using "--env=KEY" (without value) passes through from host
            # environment, so the actual secret is NOT visible in `ps` output.
            # However, `docker inspect` can still reveal the value.
            #
            # KNOWN LIMITATION: docker inspect exposure remains acceptable for this use case
            # because: (1) requires Docker API access, (2) containers are short-lived,
            # (3) same-user isolation is assumed.
            #
            # FUTURE: Consider using --env-file with a tmpfs-backed temporary file,
            # or mounting secrets via Docker secrets/volumes for enhanced isolation.
            "--env=ANTHROPIC_API_KEY",
            "--env=OPENAI_API_KEY",
            "--env=GOOGLE_API_KEY",
        ])

        # Handle prompt passing based on CLI capability
        # SECURITY: Prompt is always passed via stdin to avoid `ps` exposure.
        stdin_input = prompt

        # ARG_MAX for argument-based CLIs (like claude -p)
        # While Linux typically allows ~2MB, shell variable expansion and argument
        # passing can fail at lower limits. POSIX guarantees only 128KB minimum.
        # Use 128KB to ensure compatibility across systems and prevent DoS.
        ARG_MAX_SAFE = 131_072  # 128KB (POSIX minimum guarantee)
        if not uses_stdin and len(prompt) > ARG_MAX_SAFE:
            raise SandboxError(
                f"Prompt size ({len(prompt)} bytes) exceeds ARG_MAX limit ({ARG_MAX_SAFE} bytes) "
                f"for CLI '{self.cli_name}' which requires argument-based prompt passing. "
                f"Use a stdin-capable CLI (codex, gemini) or reduce prompt size."
            )

        if uses_stdin:
            cmd.append("-i")  # Keep stdin open for prompt input
            cmd.extend([self.config.cli_image, *cli_args])
        else:
            # For CLIs that need prompt as argument (e.g., claude -p),
            # write stdin to temp file, then read and pass as argument.
            # SECURITY: File is written via stdin, not exposed in ps output.
            # SECURITY: chmod 600 restricts file to owner-only access.
            # SECURITY: Variable assignment + quoted expansion prevents command injection.
            #   - PROMPT="$(cat file)" captures output literally into variable
            #   - "$PROMPT" expands without further shell parsing
            #   - This is safe even if prompt contains quotes, $, backticks, etc.
            # NOTE: Command substitution $(cat file) still subject to ARG_MAX (~2MB).
            cmd.append("-i")  # Keep stdin open
            cmd.extend([
                self.config.cli_image,
                "sh", "-c",
                # Write stdin to file, capture into variable, pass safely quoted
                f'cat > /tmp/prompt.txt && chmod 600 /tmp/prompt.txt && '
                f'PROMPT="$(cat /tmp/prompt.txt)" && {shlex.join(cli_args)} "$PROMPT"',
            ])

        _registry.add(container_name)
        try:
            # KNOWN LIMITATION: subprocess.run with capture_output=True buffers all
            # output in memory before _truncate_output is applied. A malicious or
            # runaway command could emit gigabytes before timeout, causing OOM.
            #
            # MITIGATIONS IN PLACE:
            # (1) Container memory limit (--memory) caps total container memory
            # (2) Timeout kills container before unbounded output accumulates
            # (3) Post-capture truncation prevents downstream memory issues
            #
            # FUTURE: Implement streaming capture with Popen and incremental reads,
            # enforcing byte limit during capture rather than after. This would
            # require significant refactoring of the execution flow.
            result = subprocess.run(
                cmd,
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=self.config.cli_timeout,
            )
            # Truncate output to prevent downstream memory issues
            max_bytes = self.config.max_output_bytes
            return ExecutionResult(
                returncode=result.returncode,
                stdout=_truncate_output(self._strip_ansi(result.stdout), max_bytes),
                stderr=_truncate_output(result.stderr, max_bytes),
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            # Kill container on timeout
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"CLI execution timed out after {self.config.cli_timeout}s",
                timed_out=True,
            )
        finally:
            _registry.remove(container_name)

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from output."""
        return self.ANSI_ESCAPE.sub("", text)


class SandboxedExecutor:
    """Execute arbitrary commands (tests, lint) in fully isolated container.

    NO network access - tests should not need external connectivity.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
    ):
        self.config = config or SandboxConfig()
        if self.config.require_docker:
            _validate_docker()

    def run(
        self,
        command: list[str],
        workdir: str | Path,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run command in fully isolated container (no network).

        Args:
            command: Command and arguments to run
            workdir: Working directory to mount (should be a worktree)
            timeout: Optional timeout override
            env: Optional environment variables

        Returns:
            ExecutionResult with stdout, stderr, and return code
        """
        workdir = Path(workdir).absolute()

        # SECURITY: Validate workdir is under allowed roots
        # Returns resolved path to use for Docker mount (TOCTOU mitigation)
        workdir_resolved = _validate_workdir(workdir, self.config.allowed_workdir_roots)

        # Validate workdir exists and is a directory before mounting
        # Use resolved path for consistency
        if not workdir_resolved.exists():
            raise SandboxError(f"Workdir does not exist: {workdir_resolved}")
        if not workdir_resolved.is_dir():
            raise SandboxError(f"Workdir is not a directory: {workdir_resolved}")

        # Use specific prefix to avoid collision with other tools
        container_name = f"autocoder-supervisor-exec-{uuid.uuid4().hex[:8]}"
        effective_timeout = timeout or self.config.executor_timeout

        # Build command string for shell execution
        # Use bash -lc to get a login shell with proper PATH
        shell_command = " ".join(shlex.quote(arg) for arg in command)

        # TOCTOU MITIGATION: Use resolved path for Docker mount
        cmd = [
            "docker", "run",
            "--rm",
            f"--name={container_name}",
            "--network=none",  # NO network access
            # Mount ONLY the specified workdir
            f"--volume={workdir_resolved}:/workspace:rw",
            "--workdir=/workspace",
        ]

        # Add --user flag only on Linux/macOS (Windows Docker Desktop handles this)
        docker_user = _get_docker_user()
        if docker_user:
            cmd.append(f"--user={docker_user}")

        cmd.extend([
            "--read-only",
            "--tmpfs=/tmp:size=1g",
            "--tmpfs=/home:size=100m",
            f"--memory={self.config.memory_limit}",
            f"--cpus={self.config.cpu_limit}",
            # Process limits (prevent fork-bomb DoS)
            f"--pids-limit={self.config.pids_limit}",
            f"--ulimit=nproc={self.config.ulimit_nproc}:{self.config.ulimit_nproc}",
            # Security options
            "--security-opt=no-new-privileges:true",
            "--cap-drop=ALL",
            # HOME must match tmpfs mount so tools can write config/cache
            "--env=HOME=/home",
        ])

        # Add environment variables
        if env:
            for key, value in env.items():
                cmd.extend(["--env", f"{key}={value}"])

        # Add image and command
        cmd.extend([
            self.config.executor_image,
            "bash", "-lc", shell_command,
        ])

        _registry.add(container_name)
        try:
            # KNOWN LIMITATION: Same buffering issue as SandboxedLLMClient.execute().
            # See comments there for details and future plans.
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            # Truncate output to prevent downstream memory issues
            max_bytes = self.config.max_output_bytes
            return ExecutionResult(
                returncode=result.returncode,
                stdout=_truncate_output(result.stdout, max_bytes),
                stderr=_truncate_output(result.stderr, max_bytes),
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {effective_timeout}s",
                timed_out=True,
            )
        finally:
            _registry.remove(container_name)


class LocalExecutor:
    """UNSAFE fallback executor for unit tests only.

    WARNING: This is NOT sandboxed. NEVER use in production.
    Only use for testing the orchestrator logic without Docker.

    Requires SUPERVISOR_ALLOW_LOCAL_EXECUTOR=1 environment variable to instantiate.
    This prevents accidental use in production code.
    """

    _UNIT_TEST_ONLY = True  # Flag to indicate this is for tests only

    def __init__(self, workdir: Path | None = None):
        # SECURITY: Require explicit opt-in via environment variable
        # This prevents accidental use in production code
        if os.environ.get("SUPERVISOR_ALLOW_LOCAL_EXECUTOR") != "1":
            raise SandboxError(
                "LocalExecutor requires SUPERVISOR_ALLOW_LOCAL_EXECUTOR=1 environment variable. "
                "This executor is NOT sandboxed and should ONLY be used for unit tests. "
                "In production, use SandboxedExecutor instead."
            )

        self.workdir = Path(workdir).absolute() if workdir else Path.cwd()
        import sys
        print(
            "WARNING: Using LocalExecutor - commands run WITHOUT SANDBOX. "
            "This should only be used for unit tests.",
            file=sys.stderr,
        )

    def run(
        self,
        command: list[str],
        workdir: str | Path | None = None,
        timeout: int = 600,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run command locally (NOT SANDBOXED - unit tests only)."""
        effective_workdir = Path(workdir) if workdir else self.workdir

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=effective_workdir,
                timeout=timeout,
                env=env,
            )
            return ExecutionResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                timed_out=True,
            )


def require_docker() -> None:
    """Validate Docker is available. Call at startup."""
    _validate_docker()


def get_sandboxed_executor(config: SandboxConfig | None = None) -> SandboxedExecutor:
    """Get a sandboxed executor. Raises if Docker unavailable."""
    return SandboxedExecutor(config)


def get_executor_for_testing(workdir: Path) -> LocalExecutor:
    """Get a LOCAL executor for unit tests only. NOT SANDBOXED."""
    return LocalExecutor(workdir)
