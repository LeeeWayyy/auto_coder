"""Gate execution with caching, worktree integrity verification, and artifact storage."""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from copy import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from supervisor.core.gate_locks import ArtifactLock, WorktreeLock
from supervisor.core.gate_models import (
    ConcurrentGateExecutionError,
    GateConfig,
    GateConfigError,
    GateFailAction,
    GateResult,
    GateSeverity,
    GateStatus,
    WorktreeBaseline,
)

if TYPE_CHECKING:
    from supervisor.core.gate_loader import GateLoader
    from supervisor.core.state import Database
    from supervisor.sandbox.executor import SandboxedExecutor

logger = logging.getLogger(__name__)


class GateExecutor:
    """Execute verification gates in sandboxed containers (basic structure)."""

    # Cache settings
    CACHE_MAX_SIZE = 100
    CACHE_TTL_SECONDS = 3600

    # Cache key computation limits
    CACHE_KEY_TIMEOUT = 30
    CACHE_KEY_MAX_UNTRACKED_FILES = 500
    CACHE_KEY_MAX_FILE_SIZE = 1_000_000  # 1MB
    CACHE_KEY_SKIP_PATTERNS = frozenset(
        [
            "node_modules/",
            "node_modules\\",
            ".venv/",
            ".venv\\",
            "venv/",
            "venv\\",
            "__pycache__/",
            "__pycache__\\",
            "build/",
            "build\\",
            "dist/",
            "dist\\",
            ".git/",
            ".git\\",
            "vendor/",
            "vendor\\",
            "target/",
            "target\\",
            ".next/",
            ".next\\",
        ]
    )

    # Protected env vars that cannot be overridden by gate config (security)
    ENV_DENYLIST = frozenset(
        {
            "PATH",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "HOME",
            "USER",
        }
    )
    ENV_DENYLIST_PREFIXES = ("SUPERVISOR_",)

    # Git config overrides to disable repo-configured helpers
    GIT_SAFE_CONFIG_BASE = [
        "-c",
        "core.fsmonitor=false",
        "-c",
        "diff.external=",
        "-c",
        "diff.textconv=",
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.clean=",
    ]

    _safe_hooks_dir: Path | None = None

    class GateTimeout(Exception):
        """Cache key computation timed out."""

        pass

    def __init__(
        self,
        executor: "SandboxedExecutor",
        gate_loader: "GateLoader",
        db: "Database",
    ) -> None:
        self.executor = executor
        self.gate_loader = gate_loader
        self.db = db
        self._cache: dict[str, tuple[GateResult, float]] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def _get_safe_hooks_dir(cls) -> Path:
        if cls._safe_hooks_dir is None:
            cls._safe_hooks_dir = Path(
                tempfile.mkdtemp(prefix="supervisor_safe_hooks_")
            )
        return cls._safe_hooks_dir

    def _filter_env(self, env: dict[str, str]) -> dict[str, str]:
        """Filter out protected env vars from gate config."""
        filtered: dict[str, str] = {}
        for key, value in env.items():
            key_upper = key.upper()
            if key_upper in self.ENV_DENYLIST:
                continue
            if any(key_upper.startswith(prefix) for prefix in self.ENV_DENYLIST_PREFIXES):
                continue
            filtered[key] = value
        return filtered

    def _get_safe_git_env(self) -> tuple[dict[str, str], list[str]]:
        """Get sanitized environment and safe config for git subprocess calls."""
        git_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        git_env["GIT_CONFIG_NOSYSTEM"] = "1"
        git_env["GIT_CONFIG_GLOBAL"] = "NUL" if sys.platform == "win32" else "/dev/null"

        safe_hooks_dir = self._get_safe_hooks_dir()
        git_safe_config = list(self.GIT_SAFE_CONFIG_BASE) + [
            "-c",
            f"core.hooksPath={safe_hooks_dir}",
        ]
        return git_env, git_safe_config

    def _evict_cache(self) -> None:
        """Evict oldest cache entry to maintain max size."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]

    def _get_cached_result(self, key: str) -> GateResult | None:
        """Fetch a cached result if present and not expired."""
        with self._cache_lock:
            if key not in self._cache:
                return None
            result, timestamp = self._cache[key]
            if time.time() - timestamp > self.CACHE_TTL_SECONDS:
                del self._cache[key]
                return None
            return result

    def _cache_result(self, key: str, result: GateResult) -> None:
        """Store a result in the cache with FIFO eviction."""
        with self._cache_lock:
            if len(self._cache) >= self.CACHE_MAX_SIZE:
                self._evict_cache()
            self._cache[key] = (result, time.time())

    def _compute_cache_key(self, worktree_path: Path, config: GateConfig) -> str | None:
        """Compute deterministic cache key for gate result."""
        if not config.cache:
            return None

        deadline = time.monotonic() + self.CACHE_KEY_TIMEOUT

        def _check_deadline() -> None:
            if time.monotonic() > deadline:
                raise self.GateTimeout("Cache key computation timed out")

        try:
            _check_deadline()
            git_env, git_safe_config = self._get_safe_git_env()

            def _run_git(args: list[str], timeout: int = 5) -> subprocess.CompletedProcess:
                _check_deadline()
                return subprocess.run(
                    ["git"] + git_safe_config + args,
                    cwd=worktree_path,
                    capture_output=True,
                    timeout=timeout,
                    env=git_env,
                )

            head_result = _run_git(["rev-parse", "HEAD"], timeout=5)
            if head_result.returncode != 0:
                return None
            head_commit = head_result.stdout.decode(
                "utf-8", errors="surrogateescape"
            ).strip()
            _check_deadline()

            staged_result = _run_git(["diff", "--cached"], timeout=10)
            if staged_result.returncode != 0:
                return None
            _check_deadline()

            unstaged_result = _run_git(["diff"], timeout=10)
            if unstaged_result.returncode != 0:
                return None
            _check_deadline()

            untracked_result = _run_git(
                ["ls-files", "--others", "--exclude-standard", "-z"], timeout=5
            )
            if untracked_result.returncode != 0:
                return None
            _check_deadline()

            content_hasher = hashlib.sha256()

            # Include deterministic config inputs first
            filtered_env = self._filter_env(config.env or {})
            sorted_env = sorted(filtered_env.items())
            config_blob = json.dumps(
                {
                    "gate": config.name,
                    "command": config.command,
                    "env": sorted_env,
                    "working_dir": config.working_dir or "",
                    "timeout": config.timeout,
                    # Include integrity rules so tightening allowed_writes invalidates cache
                    "allowed_writes": sorted(config.allowed_writes or []),
                },
                separators=(",", ":"),
                ensure_ascii=True,
            )
            content_hasher.update(config_blob.encode("utf-8"))
            content_hasher.update(b"\0")
            content_hasher.update(head_commit.encode("utf-8"))
            content_hasher.update(b"\0")
            content_hasher.update(staged_result.stdout)
            content_hasher.update(b"\0")
            content_hasher.update(unstaged_result.stdout)
            content_hasher.update(b"\0")

            # Hash untracked file contents
            files_processed = 0
            if untracked_result.stdout:
                for filename_bytes in untracked_result.stdout.split(b"\0"):
                    if not filename_bytes:
                        continue
                    _check_deadline()
                    filename = filename_bytes.decode("utf-8", errors="surrogateescape")
                    normalized = filename.replace("\\", "/")

                    skip = False
                    for pattern in self.CACHE_KEY_SKIP_PATTERNS:
                        pattern_norm = pattern.replace("\\", "/").strip("/")
                        # Check if pattern matches as a path component (gitignore-style)
                        if normalized.startswith(pattern_norm + "/") or pattern_norm in normalized.split("/"):
                            skip = True
                            break
                    if skip:
                        continue

                    files_processed += 1
                    if files_processed > self.CACHE_KEY_MAX_UNTRACKED_FILES:
                        return None

                    file_path = worktree_path / filename
                    if not file_path.is_file():
                        continue

                    # SECURITY: Skip symlinks to avoid escaping worktree
                    if file_path.is_symlink():
                        continue

                    stat = file_path.stat()
                    if stat.st_size > self.CACHE_KEY_MAX_FILE_SIZE:
                        return None

                    file_hasher = hashlib.sha256()
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            file_hasher.update(chunk)
                    file_hash = file_hasher.hexdigest()[:32]
                    content_hasher.update(f"{filename}:{file_hash}\n".encode("utf-8"))

            # Include cache_inputs: additional files/patterns that affect the gate
            if config.cache_inputs:
                _check_deadline()
                cache_input_files: list[str] = []
                for pattern in config.cache_inputs:
                    _check_deadline()
                    # Expand glob patterns
                    pattern_norm = pattern.replace("\\", "/")
                    for root, _dirs, files in os.walk(worktree_path):
                        _check_deadline()
                        rel_root = Path(root).relative_to(worktree_path)
                        for fname in files:
                            rel_path = str(rel_root / fname).replace("\\", "/")
                            if rel_path.startswith("./"):
                                rel_path = rel_path[2:]
                            if fnmatch.fnmatch(rel_path, pattern_norm):
                                cache_input_files.append(rel_path)

                # Sort for determinism and limit count
                cache_input_files = sorted(set(cache_input_files))
                if len(cache_input_files) > self.CACHE_KEY_MAX_UNTRACKED_FILES:
                    if config.force_hash_large_cache_inputs:
                        cache_input_files = cache_input_files[
                            : self.CACHE_KEY_MAX_UNTRACKED_FILES
                        ]
                    else:
                        return None  # Too many files to hash

                for rel_path in cache_input_files:
                    _check_deadline()
                    file_path = worktree_path / rel_path
                    if not file_path.is_file():
                        continue
                    # Skip symlinks for cache key (security)
                    if file_path.is_symlink():
                        continue
                    stat = file_path.stat()
                    if stat.st_size > self.CACHE_KEY_MAX_FILE_SIZE:
                        if not config.force_hash_large_cache_inputs:
                            return None  # File too large
                        # force_hash_large_cache_inputs=True: hash the full file
                        # even though it's large (may be slow but ensures correctness)

                    # Hash the file contents
                    file_hasher = hashlib.sha256()
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            file_hasher.update(chunk)
                    file_hash = file_hasher.hexdigest()[:32]
                    content_hasher.update(
                        f"cache_input:{rel_path}:{file_hash}\n".encode("utf-8")
                    )

            executor_image_id = getattr(self.executor, "image_id", None)
            if executor_image_id:
                content_hasher.update(f"image:{executor_image_id}".encode("utf-8"))

            return content_hasher.hexdigest()[:32]
        except self.GateTimeout:
            return None
        except Exception:
            return None

    # ReDoS protection constants
    _MAX_GLOB_PATTERN_LENGTH = 1024
    _MAX_CONSECUTIVE_WILDCARDS = 10
    _MAX_CHARACTER_CLASS_LENGTH = 64

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _glob_to_regex(pattern: str) -> re.Pattern[str] | None:
        """Convert a glob pattern to a compiled regex (cached).

        Includes ReDoS protection:
        - Pattern length limit
        - Consecutive wildcard limit
        - Character class length limit
        """
        # SECURITY: ReDoS protection - limit pattern length
        if len(pattern) > GateExecutor._MAX_GLOB_PATTERN_LENGTH:
            return None

        # SECURITY: ReDoS protection - detect consecutive wildcards
        wildcard_count = 0
        for char in pattern:
            if char in "*?":
                wildcard_count += 1
                if wildcard_count > GateExecutor._MAX_CONSECUTIVE_WILDCARDS:
                    return None
            else:
                wildcard_count = 0

        regex_parts: list[str] = []
        i = 0
        while i < len(pattern):
            if i + 1 < len(pattern) and pattern[i : i + 2] == "**":
                if i + 2 < len(pattern) and pattern[i + 2] == "/":
                    # **/ matches zero or more directory components
                    regex_parts.append("(?:.*/)?")
                    i += 3
                elif i + 2 == len(pattern):
                    # ** at end of pattern matches everything remaining
                    regex_parts.append(".*")
                    i += 2
                else:
                    # SECURITY: ** not followed by / and not at end is ambiguous.
                    # e.g., "build**bar" could match "build_evil_bar" unexpectedly.
                    # Treat as literal "**" to prevent overly broad matching.
                    regex_parts.append(re.escape("**"))
                    i += 2
            elif pattern[i] == "*":
                regex_parts.append("[^/]*")
                i += 1
            elif pattern[i] == "?":
                regex_parts.append("[^/]")
                i += 1
            elif pattern[i] == "[":
                j = i + 1
                if j < len(pattern) and pattern[j] in "!^":
                    j += 1
                if j < len(pattern) and pattern[j] == "]":
                    j += 1
                while j < len(pattern) and pattern[j] != "]":
                    j += 1
                if j < len(pattern):
                    # SECURITY: ReDoS protection - limit character class length
                    char_class = pattern[i : j + 1]
                    if len(char_class) > GateExecutor._MAX_CHARACTER_CLASS_LENGTH:
                        return None
                    regex_parts.append(char_class)
                    i = j + 1
                else:
                    regex_parts.append(re.escape(pattern[i]))
                    i += 1
            else:
                regex_parts.append(re.escape(pattern[i]))
                i += 1

        regex_str = "^" + "".join(regex_parts) + "$"
        try:
            return re.compile(regex_str)
        except re.error:
            return None

    @staticmethod
    def _path_match(path: str, pattern: str) -> bool:
        """Git-style glob matching for allowed_writes patterns."""

        def _normalize(value: str) -> str:
            value = value.replace("\\", "/")
            while value.startswith("./"):
                value = value[2:]
            while "//" in value:
                value = value.replace("//", "/")
            return value

        path = _normalize(path)
        pattern = _normalize(pattern)

        compiled = GateExecutor._glob_to_regex(pattern)
        if compiled is None:
            return path == pattern
        return bool(compiled.match(path))

    # Baseline capture size limits (prevents memory exhaustion)
    _MAX_BASELINE_FILES = 10000  # Maximum files to track in baseline
    _MAX_BASELINE_FILE_SIZE = 50 * 1024 * 1024  # 50MB - skip hash for larger files
    _MAX_GIT_OUTPUT_SIZE = 10 * 1024 * 1024  # 10MB git status output limit

    def _capture_worktree_baseline(self, worktree_path: Path) -> WorktreeBaseline | None:
        """Capture file states before gate execution.

        Size limits are enforced to prevent memory exhaustion:
        - Max files: _MAX_BASELINE_FILES
        - Max file size for hashing: _MAX_BASELINE_FILE_SIZE
        - Max git output: _MAX_GIT_OUTPUT_SIZE
        """
        try:
            git_env, git_safe_config = self._get_safe_git_env()
            result = subprocess.run(
                ["git"] + git_safe_config + ["status", "--porcelain", "-z", "-uall"],
                cwd=worktree_path,
                capture_output=True,
                timeout=10,
                env=git_env,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="surrogateescape")
                logger.warning(f"Baseline capture failed (git status): {stderr}")
                return None

            # SECURITY: Limit git output size to prevent memory exhaustion
            if len(result.stdout) > self._MAX_GIT_OUTPUT_SIZE:
                logger.warning(
                    f"Baseline capture aborted: git status output too large "
                    f"({len(result.stdout)} > {self._MAX_GIT_OUTPUT_SIZE} bytes)"
                )
                return None

            baseline: dict[str, tuple[int, int, str | None]] = {}
            has_tracked_changes = False
            entries = result.stdout.split(b"\0")
            i = 0
            file_count = 0
            while i < len(entries):
                # SECURITY: Limit number of files to prevent memory exhaustion
                if file_count >= self._MAX_BASELINE_FILES:
                    logger.warning(
                        f"Baseline capture truncated: too many files "
                        f"({file_count} >= {self._MAX_BASELINE_FILES})"
                    )
                    break

                entry = entries[i]
                if not entry or len(entry) < 3:
                    i += 1
                    continue

                entry_str = entry.decode("utf-8", errors="surrogateescape")
                space_idx = entry_str.find(" ")
                if space_idx < 2:
                    i += 1
                    continue

                status_token = entry_str[:space_idx]
                status = status_token[:2]
                path = entry_str[space_idx + 1 :]

                is_rename_copy = status[0] in ("R", "C") or status[1] in ("R", "C")
                if is_rename_copy and (i + 1) < len(entries):
                    new_path_bytes = entries[i + 1]
                    path = (
                        new_path_bytes.decode("utf-8", errors="surrogateescape")
                        if new_path_bytes
                        else path
                    )
                    i += 2
                else:
                    i += 1

                if status.strip() and not status.startswith("??"):
                    has_tracked_changes = True

                file_path = worktree_path / path
                file_count += 1
                try:
                    # SECURITY: Track symlinks separately to detect symlink attacks
                    # Use size=-2 to indicate symlink (vs -1 for non-file/error)
                    if file_path.is_symlink():
                        # Record symlink target for change detection
                        try:
                            symlink_target = str(file_path.readlink())
                            # Hash the symlink target for change detection
                            target_hash = hashlib.sha256(symlink_target.encode()).hexdigest()[:32]
                            baseline[path] = (0, -2, target_hash)  # -2 indicates symlink
                        except OSError:
                            baseline[path] = (0, -2, None)
                    elif file_path.is_file():
                        stat = file_path.stat()
                        mtime = int(stat.st_mtime * 1000)
                        size = stat.st_size

                        # SECURITY: Skip hashing for very large files (use mtime+size only)
                        if size > self._MAX_BASELINE_FILE_SIZE:
                            baseline[path] = (mtime, size, None)
                        else:
                            file_hasher = hashlib.sha256()
                            with open(file_path, "rb") as f:
                                for chunk in iter(lambda: f.read(8192), b""):
                                    file_hasher.update(chunk)
                            content_hash = file_hasher.hexdigest()[:32]
                            baseline[path] = (mtime, size, content_hash)
                    else:
                        baseline[path] = (0, -1, None)
                except OSError:
                    baseline[path] = (0, -1, None)

            return WorktreeBaseline(
                files=baseline, pre_tracked_clean=not has_tracked_changes
            )
        except Exception as e:
            logger.warning(f"Baseline capture failed: {e}")
            return None

    def _verify_worktree_integrity(
        self,
        worktree_path: Path,
        baseline: WorktreeBaseline,
        config: GateConfig,
    ) -> tuple[bool, list[str]]:
        """Check if gate modified files it shouldn't have."""
        violations: list[str] = []
        allowed = config.allowed_writes or []

        def _is_allowed(path: str) -> bool:
            if not allowed:
                return False
            normalized = path.replace("\\", "/")
            return any(self._path_match(normalized, pattern) for pattern in allowed)

        try:
            git_env, git_safe_config = self._get_safe_git_env()

            # SECURITY: Get list of tracked files to prevent allowed_writes from
            # exempting modifications to clean tracked files (P1 security fix)
            tracked_result = subprocess.run(
                ["git"] + git_safe_config + ["ls-files", "-z"],
                cwd=worktree_path,
                capture_output=True,
                timeout=10,
                env=git_env,
            )
            tracked_files: set[str] = set()
            if tracked_result.returncode == 0:
                for entry in tracked_result.stdout.split(b"\0"):
                    if entry:
                        tracked_files.add(entry.decode("utf-8", errors="surrogateescape"))

            result = subprocess.run(
                ["git"] + git_safe_config + ["status", "--porcelain", "-z", "-uall"],
                cwd=worktree_path,
                capture_output=True,
                timeout=10,
                env=git_env,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="surrogateescape")
                logger.warning(f"Integrity check failed (git status): {stderr}")
                return (False, ["<unknown - git status failed>"])

            entries = result.stdout.split(b"\0")
            current_paths: set[str] = set()
            i = 0
            while i < len(entries):
                entry = entries[i]
                if not entry or len(entry) < 3:
                    i += 1
                    continue

                entry_str = entry.decode("utf-8", errors="surrogateescape")
                space_idx = entry_str.find(" ")
                if space_idx < 2:
                    i += 1
                    continue

                status_token = entry_str[:space_idx]
                status = status_token[:2]
                path = entry_str[space_idx + 1 :]

                is_rename_copy = status[0] in ("R", "C") or status[1] in ("R", "C")
                if is_rename_copy and (i + 1) < len(entries):
                    new_path_bytes = entries[i + 1]
                    path = (
                        new_path_bytes.decode("utf-8", errors="surrogateescape")
                        if new_path_bytes
                        else path
                    )
                    i += 2
                else:
                    i += 1

                if status.strip():
                    current_paths.add(path)

            resolved_worktree = worktree_path.resolve()

            for path in sorted(current_paths):
                if path not in baseline.files:
                    # SECURITY (P1): Check if file is tracked. Tracked files that were
                    # clean before gate execution are NOT exempt from integrity checks,
                    # even if they match allowed_writes. Only truly NEW untracked files
                    # can be exempted by allowed_writes patterns.
                    is_tracked = path in tracked_files
                    if not is_tracked and _is_allowed(path):
                        # Truly new untracked file matching allowed_writes - OK
                        continue
                    # Check if it's a symlink escaping worktree
                    file_path = worktree_path / path
                    if file_path.is_symlink():
                        try:
                            resolved_target = file_path.resolve()
                            resolved_target.relative_to(resolved_worktree)
                        except (ValueError, OSError):
                            # SECURITY: New symlink escapes worktree
                            violations.append(f"{path} (new symlink escapes worktree)")
                            continue
                    if is_tracked:
                        violations.append(f"{path} (tracked file modified)")
                    else:
                        violations.append(path)
                    continue

                # SECURITY: Tracked files in baseline are ALWAYS checked.
                # Untracked files in baseline that match allowed_writes are exempt
                # (for idempotency - gates can update their own artifacts).
                is_tracked_file = path in tracked_files
                is_exempt_untracked = not is_tracked_file and _is_allowed(path)

                pre_mtime, pre_size, pre_hash = baseline.files[path]
                file_path = worktree_path / path

                # SECURITY: Check if file/symlink still exists using is_symlink() first
                # to avoid false "deleted" reports for broken symlinks.
                # A broken symlink (target missing) returns False for exists() but
                # True for is_symlink(), so we check symlink status first.
                is_symlink = file_path.is_symlink()
                file_exists = is_symlink or file_path.exists()

                if not file_exists:
                    # File was deleted - only allow if it's an exempt untracked file
                    if pre_size not in (-1,) and not is_exempt_untracked:
                        violations.append(path)
                    continue

                # SECURITY: Check if file type changed (file <-> symlink)
                was_symlink = pre_size == -2

                if is_symlink != was_symlink:
                    # Type changed - violation unless exempt untracked
                    if not is_exempt_untracked:
                        violations.append(path)
                    continue

                if is_symlink:
                    # SECURITY: Symlink handling - check stays in worktree FIRST (before hash)
                    # This ordering mitigates TOCTOU: security check happens atomically with resolve()
                    try:
                        # SECURITY: Verify symlink doesn't escape worktree (do this FIRST)
                        resolved_target = file_path.resolve()
                        try:
                            resolved_target.relative_to(resolved_worktree)
                        except ValueError:
                            # Symlink points outside worktree - always a violation (security)
                            violations.append(f"{path} (symlink escapes worktree)")
                            continue

                        # Change detection: hash the symlink target
                        symlink_target = str(file_path.readlink())
                        cur_target_hash = hashlib.sha256(symlink_target.encode()).hexdigest()[:32]

                        # Check if symlink target changed (unless exempt untracked)
                        if pre_hash is not None and cur_target_hash != pre_hash:
                            if not is_exempt_untracked:
                                violations.append(path)
                            continue
                    except OSError:
                        # Symlink is broken (can't resolve/readlink)
                        # Only flag as violation if it wasn't already broken in baseline
                        # pre_hash is None for broken symlinks in baseline (size=-2)
                        if pre_hash is not None and not is_exempt_untracked:
                            # Was valid before, now broken - violation
                            violations.append(path)
                elif file_path.is_file():
                    try:
                        stat = file_path.stat()
                        cur_mtime = int(stat.st_mtime * 1000)
                        cur_size = stat.st_size
                        if cur_mtime != pre_mtime or cur_size != pre_size:
                            if not is_exempt_untracked:
                                violations.append(path)
                            continue

                        if pre_hash is not None:
                            file_hasher = hashlib.sha256()
                            with open(file_path, "rb") as f:
                                for chunk in iter(lambda: f.read(8192), b""):
                                    file_hasher.update(chunk)
                            cur_hash = file_hasher.hexdigest()[:32]
                            if cur_hash != pre_hash and not is_exempt_untracked:
                                violations.append(path)
                    except OSError:
                        if not is_exempt_untracked:
                            violations.append(path)
                else:
                    if pre_size >= 0 and not is_exempt_untracked:  # Was a regular file
                        violations.append(path)

            # Check for deleted files - files in baseline that are no longer present
            # SECURITY: Tracked files are ALWAYS checked. Untracked files matching
            # allowed_writes can be deleted (for idempotency).
            for path, (pre_mtime, pre_size, pre_hash) in baseline.files.items():
                if path in current_paths:
                    continue
                file_path = worktree_path / path
                if not file_path.exists() and pre_size != -1:
                    # Only flag deletion if it's a tracked file or doesn't match allowed_writes
                    is_tracked_file = path in tracked_files
                    is_exempt_untracked = not is_tracked_file and _is_allowed(path)
                    if not is_exempt_untracked:
                        violations.append(path)

            return (len(violations) == 0, violations)
        except Exception as e:
            logger.warning(f"Integrity check failed: {e}")
            return (False, ["<unknown - integrity check failed>"])

    # Common secret patterns to redact from gate output/artifacts.
    SECRET_PATTERNS = [
        (re.compile(r"ghp_[a-zA-Z0-9]{36,}"), "[REDACTED:github_pat]"),
        (re.compile(r"github_pat_[a-zA-Z0-9_]+"), "[REDACTED:github_pat_v2]"),
        (re.compile(r"gho_[a-zA-Z0-9]{36,}"), "[REDACTED:github_oauth]"),
        (re.compile(r"sk-[a-zA-Z0-9]{48,}"), "[REDACTED:openai_key]"),
        (re.compile(r"sk-proj-[a-zA-Z0-9\-_]+"), "[REDACTED:openai_project]"),
        (re.compile(r"sk-ant-[a-zA-Z0-9\-_]+"), "[REDACTED:anthropic_key]"),
        (re.compile(r"AKIA[A-Z0-9]{16}"), "[REDACTED:aws_access_key]"),
        (re.compile(r"xox[baprs]-[a-zA-Z0-9\-]+"), "[REDACTED:slack_token]"),
        (re.compile(r"(?i)(api[_-]?key|apikey)[\"']?\s*[:=]\s*[\"']?[\w\-]+"), r"\1=[REDACTED]"),
        (re.compile(r"(?i)(token|secret|password|passwd|pwd)[\"']?\s*[:=]\s*[\"']?[\w\-]+"), r"\1=[REDACTED]"),
        (re.compile(r"(?i)(authorization|auth)[\"']?\s*[:=]\s*[\"']?bearer\s+[\w\-\.]+"), r"\1=Bearer [REDACTED]"),
    ]

    def _redact_secrets(self, text: str) -> str:
        """Redact common secret patterns from output."""
        for pattern, replacement in self.SECRET_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def _store_artifact(
        self,
        workflow_id: str,
        gate_name: str,
        output: str,
        worktree_path: Path,
    ) -> str | None:
        """Store full gate output as artifact file.

        Path: {worktree}/.supervisor/artifacts/gates/{hashed_workflow_id}/{gate_name}-{timestamp}.log
        """
        try:
            resolved_worktree = worktree_path.resolve()
            safe_workflow_id = hashlib.sha256(workflow_id.encode("utf-8")).hexdigest()[:32]

            # Validate existing path components for symlink safety.
            path_parts = [".supervisor", ".supervisor/artifacts", ".supervisor/artifacts/gates"]
            for part in path_parts:
                check_path = resolved_worktree / part
                if check_path.exists():
                    if check_path.is_symlink():
                        logger.warning(
                            f"SECURITY: {part} is a symlink. Artifact storage disabled."
                        )
                        return None
                    try:
                        check_path.resolve().relative_to(resolved_worktree)
                    except ValueError:
                        logger.warning(
                            f"SECURITY: {part} resolves outside worktree. Artifact storage disabled."
                        )
                        return None

            artifacts_dir = (
                resolved_worktree
                / ".supervisor"
                / "artifacts"
                / "gates"
                / safe_workflow_id
            )

            # Concurrency protection.
            with ArtifactLock(worktree_path, exclusive=True) as lock:
                if not lock.acquired:
                    logger.warning(
                        f"Artifact storage skipped for gate '{gate_name}': lock unavailable."
                    )
                    return None

                artifacts_dir.mkdir(parents=True, exist_ok=True)
                try:
                    artifacts_dir.resolve().relative_to(resolved_worktree)
                except ValueError:
                    logger.warning(
                        "SECURITY: Artifact directory escapes worktree. Storage disabled."
                    )
                    return None

                # Enforce per-workflow artifact count (LRU by mtime).
                try:
                    existing = sorted(
                        artifacts_dir.glob("*.log"),
                        key=lambda p: p.stat().st_mtime,
                    )
                    while len(existing) >= GateResult.ARTIFACT_MAX_COUNT_PER_WORKFLOW:
                        oldest = existing.pop(0)
                        if oldest.is_symlink():
                            continue
                        try:
                            oldest.resolve().relative_to(resolved_worktree)
                        except ValueError:
                            continue
                        try:
                            oldest.unlink()
                        except OSError:
                            pass
                except OSError:
                    pass

                # Redact secrets and enforce size cap.
                redacted = self._redact_secrets(output)
                output_bytes = redacted.encode("utf-8", errors="replace")
                max_bytes = GateResult.ARTIFACT_MAX_SIZE
                if len(output_bytes) > max_bytes:
                    marker = (
                        f"...[artifact truncated to {GateResult.ARTIFACT_MAX_SIZE} bytes]...\n"
                    ).encode("utf-8")
                    tail_budget = max_bytes - len(marker)
                    tail = output_bytes[-tail_budget:] if tail_budget > 0 else b""
                    redacted = marker.decode("utf-8") + tail.decode(
                        "utf-8", errors="replace"
                    )

                # Unique filename to avoid collisions.
                timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
                unique_suffix = uuid.uuid4().hex[:8]
                artifact_filename = f"{gate_name}-{timestamp}-{unique_suffix}.log"
                artifact_file = artifacts_dir / artifact_filename

                # Atomic write via temp file + replace.
                fd, temp_path = tempfile.mkstemp(
                    prefix=".artifact_",
                    suffix=".tmp",
                    dir=str(artifacts_dir),
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(redacted)

                    final_resolved = artifact_file.resolve()
                    try:
                        final_resolved.relative_to(resolved_worktree)
                    except ValueError:
                        os.unlink(temp_path)
                        logger.warning(
                            "SECURITY: Final artifact path escapes worktree. Storage disabled."
                        )
                        return None

                    os.replace(temp_path, str(artifact_file))
                    return str(artifact_file)
                except Exception:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    raise
        except Exception as e:
            logger.warning(f"Failed to store artifact for gate '{gate_name}': {e}")
            return None

    @staticmethod
    def cleanup_artifacts(
        worktree_path: Path,
        retention_days: int = GateResult.ARTIFACT_RETENTION_DAYS,
    ) -> int:
        """Remove expired artifacts and enforce global size cap."""
        resolved_worktree = worktree_path.resolve()
        supervisor_dir = resolved_worktree / ".supervisor"
        if supervisor_dir.exists() and supervisor_dir.is_symlink():
            try:
                supervisor_dir.resolve().relative_to(resolved_worktree)
            except ValueError:
                logger.warning(
                    "SECURITY: .supervisor is a symlink outside worktree. Cleanup disabled."
                )
                return 0

        artifacts_dir = resolved_worktree / ".supervisor" / "artifacts" / "gates"
        if not artifacts_dir.exists():
            return 0
        try:
            artifacts_dir.resolve().relative_to(resolved_worktree)
        except ValueError:
            logger.warning("SECURITY: Artifacts dir escapes worktree. Cleanup disabled.")
            return 0

        deleted = 0
        cutoff = datetime.now() - timedelta(days=retention_days)

        with ArtifactLock(worktree_path, exclusive=True) as lock:
            if not lock.acquired:
                logger.info("Artifact cleanup skipped: lock unavailable.")
                return 0

            for workflow_dir in artifacts_dir.iterdir():
                if workflow_dir.is_symlink() or not workflow_dir.is_dir():
                    continue
                try:
                    workflow_dir.resolve().relative_to(resolved_worktree)
                except ValueError:
                    continue

                for artifact in workflow_dir.iterdir():
                    if artifact.is_symlink():
                        continue
                    try:
                        artifact.resolve().relative_to(resolved_worktree)
                    except ValueError:
                        continue
                    try:
                        mtime = datetime.fromtimestamp(artifact.lstat().st_mtime)
                        if mtime < cutoff:
                            artifact.unlink()
                            deleted += 1
                    except OSError:
                        pass

                try:
                    if not any(workflow_dir.iterdir()):
                        workflow_dir.rmdir()
                except OSError:
                    pass

            # Enforce global size cap (LRU eviction across workflows).
            try:
                all_artifacts: list[tuple[Path, float, int]] = []
                for wf_dir in artifacts_dir.iterdir():
                    if wf_dir.is_symlink() or not wf_dir.is_dir():
                        continue
                    for artifact in wf_dir.iterdir():
                        if artifact.is_symlink():
                            continue
                        try:
                            stat = artifact.lstat()
                            all_artifacts.append((artifact, stat.st_mtime, stat.st_size))
                        except OSError:
                            pass

                total_size = sum(a[2] for a in all_artifacts)
                if total_size > GateResult.ARTIFACT_MAX_TOTAL_SIZE:
                    all_artifacts.sort(key=lambda x: x[1])  # oldest first
                    while total_size > GateResult.ARTIFACT_MAX_TOTAL_SIZE and all_artifacts:
                        artifact_path, _, size = all_artifacts.pop(0)
                        try:
                            artifact_path.resolve().relative_to(resolved_worktree)
                        except ValueError:
                            continue
                        try:
                            artifact_path.unlink()
                            total_size -= size
                            deleted += 1
                        except OSError:
                            pass
            except Exception as e:
                logger.warning(f"Artifact size cleanup failed: {e}")

        return deleted

    def run_gate(
        self,
        config: GateConfig,
        worktree_path: Path,
        workflow_id: str,
        step_id: str,
    ) -> GateResult:
        def _truncate(text: str, max_chars: int) -> str:
            if len(text) <= max_chars:
                return text
            return f"...[truncated {len(text) - max_chars} chars]...\n{text[-max_chars:]}"

        # Resolve and validate working directory.
        try:
            if config.working_dir:
                workdir = (worktree_path / config.working_dir).resolve()
                worktree_resolved = worktree_path.resolve()
                workdir.relative_to(worktree_resolved)
                if not workdir.exists() or not workdir.is_dir():
                    raise GateConfigError(
                        f"working_dir '{config.working_dir}' does not exist or is not a directory"
                    )
            else:
                workdir = worktree_path.resolve()
        except (OSError, ValueError, GateConfigError) as e:
            error_output = f"Invalid gate working_dir: {e}"
            gate_result = GateResult(
                gate_name=config.name,
                status=GateStatus.FAILED,
                output=self._redact_secrets(error_output),
                duration_seconds=0,
                returncode=None,
                timed_out=False,
                retry_count=0,
                cached=False,
                cache_key=None,
                artifact_path=None,
                integrity_violation=False,
            )
            from supervisor.core.state import Event, EventType

            self.db.append_event(
                Event(
                    workflow_id=workflow_id,
                    event_type=EventType.GATE_FAILED,
                    step_id=step_id,
                    payload={
                        "gate": gate_result.gate_name,
                        "output": _truncate(
                            gate_result.output, GateResult.EVENT_OUTPUT_MAX_CHARS
                        ),
                        "duration": gate_result.duration_seconds,
                        "returncode": gate_result.returncode,
                        "timed_out": gate_result.timed_out,
                        "cached": gate_result.cached,
                        "artifact_path": gate_result.artifact_path,
                    },
                )
            )
            return gate_result

        cache_key = self._compute_cache_key(worktree_path, config)
        if cache_key is not None:
            cached = self._get_cached_result(cache_key)
            if cached:
                result = copy(cached)
                result.cached = True
                result.cache_key = cache_key
                # Clear artifact_path - it's workflow-specific and may not exist
                # in the current workflow's artifact directory
                result.artifact_path = None
                from supervisor.core.state import Event, EventType

                self.db.append_event(
                    Event(
                        workflow_id=workflow_id,
                        event_type=EventType.GATE_PASSED
                        if result.status == GateStatus.PASSED
                        else EventType.GATE_FAILED,
                        step_id=step_id,
                        payload={
                            "gate": result.gate_name,
                            "output": _truncate(
                                result.output, GateResult.EVENT_OUTPUT_MAX_CHARS
                            ),
                            "duration": result.duration_seconds,
                            "returncode": result.returncode,
                            "timed_out": result.timed_out,
                            "cached": result.cached,
                            "artifact_path": result.artifact_path,
                        },
                    )
                )
                return result

        baseline = self._capture_worktree_baseline(worktree_path)
        if baseline is None:
            baseline = WorktreeBaseline(files={}, pre_tracked_clean=False)

        filtered_env = self._filter_env(config.env)
        start_time = time.time()
        exec_result = None
        error_output = None
        timed_out = False

        try:
            exec_result = self.executor.run(
                command=config.command,
                workdir=workdir,
                timeout=config.timeout,
                env=filtered_env,
            )
        except Exception as e:
            timed_out = isinstance(e, TimeoutError)
            error_output = f"Gate executor error: {type(e).__name__}: {e}"

        duration = time.time() - start_time

        if exec_result is not None:
            raw_output = f"{exec_result.stdout}\n{exec_result.stderr}".strip()
            returncode = exec_result.returncode
            timed_out = getattr(exec_result, "timed_out", False)
        else:
            raw_output = error_output or "Gate executor error"
            returncode = None

        integrity_ok, violations = self._verify_worktree_integrity(
            worktree_path, baseline, config
        )
        integrity_violation = not integrity_ok
        if violations:
            raw_output += (
                "\n\nWORKTREE INTEGRITY VIOLATION: "
                + ", ".join(sorted(set(violations)))
            )

        redacted_output = self._redact_secrets(raw_output)
        artifact_path = None
        if len(redacted_output) > GateResult.OUTPUT_MAX_CHARS:
            artifact_path = self._store_artifact(
                workflow_id, config.name, raw_output, worktree_path
            )

        output = _truncate(redacted_output, GateResult.OUTPUT_MAX_CHARS)
        status = (
            GateStatus.PASSED
            if exec_result is not None and returncode == 0 and integrity_ok
            else GateStatus.FAILED
        )

        gate_result = GateResult(
            gate_name=config.name,
            status=status,
            output=output,
            duration_seconds=duration,
            returncode=returncode,
            timed_out=timed_out,
            retry_count=0,
            cached=False,
            cache_key=cache_key,
            artifact_path=artifact_path,
            integrity_violation=integrity_violation,
        )

        if gate_result.passed and cache_key is not None:
            self._cache_result(cache_key, gate_result)

        from supervisor.core.state import Event, EventType

        self.db.append_event(
            Event(
                workflow_id=workflow_id,
                event_type=EventType.GATE_PASSED
                if gate_result.status == GateStatus.PASSED
                else EventType.GATE_FAILED,
                step_id=step_id,
                payload={
                    "gate": gate_result.gate_name,
                    "output": _truncate(
                        gate_result.output, GateResult.EVENT_OUTPUT_MAX_CHARS
                    ),
                    "duration": gate_result.duration_seconds,
                    "returncode": gate_result.returncode,
                    "timed_out": gate_result.timed_out,
                    "cached": gate_result.cached,
                    "artifact_path": gate_result.artifact_path,
                },
            )
        )

        return gate_result

    def _get_on_fail_action(
        self,
        config: GateConfig,
        on_fail_overrides: dict[str, GateFailAction] | None = None,
    ) -> GateFailAction:
        """Resolve on-fail action for a gate with overrides and severity defaults."""
        if on_fail_overrides and config.name in on_fail_overrides:
            return on_fail_overrides[config.name]
        if config.severity in (GateSeverity.WARNING, GateSeverity.INFO):
            return GateFailAction.WARN
        return GateFailAction.BLOCK

    def run_gates(
        self,
        gate_names: list[str],
        worktree_path: Path,
        workflow_id: str,
        step_id: str,
        on_fail_overrides: dict[str, GateFailAction] | None = None,
    ) -> list[GateResult]:
        """Run multiple gates in dependency order with skip-on-blocking behavior."""
        def _truncate(text: str, max_chars: int) -> str:
            if len(text) <= max_chars:
                return text
            return f"...[truncated {len(text) - max_chars} chars]...\n{text[-max_chars:]}"

        with WorktreeLock(worktree_path) as lock:
            if not lock.acquired:
                raise ConcurrentGateExecutionError(
                    f"Cannot acquire worktree lock for {worktree_path}."
                )

            ordered = self.gate_loader.resolve_execution_order(gate_names)
            results: list[GateResult] = []
            blocking_failures: set[str] = set()

            for gate_name in ordered:
                config = self.gate_loader.get_gate(gate_name)

                if config.skip_on_dependency_failure:
                    blocking_deps = [
                        dep for dep in config.depends_on if dep in blocking_failures
                    ]
                    if blocking_deps:
                        skip_result = GateResult(
                            gate_name=gate_name,
                            status=GateStatus.SKIPPED,
                            output=(
                                "SKIPPED: Dependencies had blocking failures: "
                                f"{blocking_deps}"
                            ),
                            duration_seconds=0,
                            cached=False,
                            cache_key=None,
                        )
                        results.append(skip_result)

                        from supervisor.core.state import Event, EventType

                        self.db.append_event(
                            Event(
                                workflow_id=workflow_id,
                                event_type=EventType.GATE_SKIPPED,
                                step_id=step_id,
                                payload={
                                    "gate": skip_result.gate_name,
                                    "output": _truncate(
                                        skip_result.output,
                                        GateResult.EVENT_OUTPUT_MAX_CHARS,
                                    ),
                                    "duration": skip_result.duration_seconds,
                                    "returncode": skip_result.returncode,
                                    "timed_out": skip_result.timed_out,
                                    "cached": skip_result.cached,
                                    "artifact_path": skip_result.artifact_path,
                                },
                            )
                        )
                        blocking_failures.add(gate_name)
                        continue

                result = self.run_gate(config, worktree_path, workflow_id, step_id)
                results.append(result)

                if result.integrity_violation:
                    blocking_failures.add(gate_name)
                    return results

                if result.status == GateStatus.FAILED:
                    action = self._get_on_fail_action(
                        config, on_fail_overrides=on_fail_overrides
                    )
                    if action in (
                        GateFailAction.BLOCK,
                        GateFailAction.RETRY_WITH_FEEDBACK,
                    ):
                        blocking_failures.add(gate_name)
                        return results

            return results
