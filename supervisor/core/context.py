"""Context packing for AI workers.

Packs relevant files and context for each role based on configuration.
Uses Repomix for file packing when available, falls back to simple packing.
"""

import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from supervisor.core.roles import RoleConfig


class ContextPackerError(Exception):
    """Error during context packing."""

    pass


class ContextPacker:
    """Pack context for AI workers based on role configuration."""

    # Pin repomix version for supply-chain safety
    # Update this version after security review of new releases
    #
    # KNOWN LIMITATION: npx executes on the host, not in a sandbox. Even with
    # a pinned version, npm package installation runs lifecycle scripts which
    # could be compromised in a supply-chain attack.
    #
    # MITIGATIONS IN PLACE:
    # (1) Version pinning prevents auto-upgrade to malicious versions
    # (2) repomix is a well-maintained, widely-used package
    # (3) Fallback to _pack_simple if repomix unavailable
    #
    # FUTURE: Consider one of these approaches for enhanced security:
    # - Vendor repomix as a checked-in binary with checksum verification
    # - Run repomix inside the Docker sandbox instead of on host
    # - Use npm --ignore-scripts with integrity hash verification
    # - Replace with a pure-Python implementation
    REPOMIX_VERSION = "0.2.20"

    # File size limits to prevent memory exhaustion
    MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB per file
    MAX_TOTAL_SIZE = 10 * 1024 * 1024  # 10MB total for all files

    # Priority order for context pruning when over budget
    PRIORITY_ORDER = [
        "system_prompt",
        "task",
        "target_file",
        "imports",
        "related",
        "tree",
    ]

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path).absolute()
        self._repomix_available = self._check_repomix()

        # Template environment for prompts
        template_dir = Path(__file__).parent.parent / "prompts"
        if template_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self.jinja_env = None

    def _check_repomix(self) -> bool:
        """Check if repomix is available."""
        try:
            # Use pinned version for supply-chain safety
            result = subprocess.run(
                ["npx", "--yes", f"repomix@{self.REPOMIX_VERSION}", "--version"],
                capture_output=True,
                timeout=30,  # First run may need to download
            )
            return result.returncode == 0
        except Exception:
            return False

    def pack_context(
        self,
        role: RoleConfig,
        task_description: str,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
    ) -> str:
        """Pack context for a role.

        Args:
            role: Role configuration
            task_description: Description of the task
            target_files: Specific files to focus on
            extra_context: Additional context (e.g., git diff, test output)

        Returns:
            Packed context string ready for the AI worker
        """
        context_parts: dict[str, str] = {}

        # 1. System prompt (always included first)
        context_parts["system_prompt"] = role.system_prompt

        # 2. Task description
        context_parts["task"] = f"## Task\n\n{task_description}"

        # 3. Pack files based on role configuration
        file_context = self._pack_files(role, target_files)
        if file_context:
            context_parts["files"] = file_context

        # 4. Add extra context (git diff, test output, etc.)
        if extra_context:
            for key, value in extra_context.items():
                context_parts[key] = f"## {key.replace('_', ' ').title()}\n\n{value}"

        # 5. Enforce token budget with progressive pruning
        return self._pack_with_budget(context_parts, role.token_budget)

    def _pack_files(
        self,
        role: RoleConfig,
        target_files: list[str] | None = None,
    ) -> str:
        """Pack files using repomix or fallback to simple packing."""
        include_patterns = role.include_patterns.copy()
        exclude_patterns = role.exclude_patterns.copy()

        # Add target files to includes
        if target_files:
            include_patterns.extend(target_files)

        # Replace special variables
        include_patterns = self._resolve_patterns(include_patterns, target_files)

        if not include_patterns:
            return ""

        if self._repomix_available:
            return self._pack_with_repomix(include_patterns, exclude_patterns)
        else:
            return self._pack_simple(include_patterns, exclude_patterns)

    def _resolve_patterns(
        self,
        patterns: list[str],
        target_files: list[str] | None,
    ) -> list[str]:
        """Resolve special pattern variables."""
        resolved = []
        for pattern in patterns:
            if pattern == "$TARGET":
                if target_files:
                    resolved.extend(target_files)
            elif pattern == "$TARGET_IMPORTS":
                # TODO: Implement import resolution
                pass
            elif pattern.startswith("$"):
                # Skip unresolved variables
                pass
            else:
                resolved.append(pattern)
        return resolved

    def _sanitize_pattern(self, pattern: str) -> str | None:
        """Sanitize a pattern to prevent argument injection and path traversal.

        Returns None if pattern is unsafe and should be skipped.

        Security checks:
        - Patterns starting with '-' are prefixed with './' to prevent flag injection
        - Absolute paths are rejected (could read files outside repo)
        - Patterns containing '..' are rejected (path traversal)
        """
        # SECURITY: Reject absolute paths
        if pattern.startswith("/"):
            return None

        # SECURITY: Reject path traversal attempts
        # Check for '..' as a path component (not just substring to allow '...' in names)
        parts = pattern.replace("\\", "/").split("/")
        if ".." in parts:
            return None

        # Patterns starting with '-' could be interpreted as flags
        if pattern.startswith("-"):
            return f"./{pattern}"

        return pattern

    def _pack_with_repomix(
        self,
        include_patterns: list[str],
        exclude_patterns: list[str],
    ) -> str:
        """Pack files using repomix."""
        # Use pinned version for supply-chain safety
        cmd = ["npx", "--yes", f"repomix@{self.REPOMIX_VERSION}", "--style", "xml"]

        # SECURITY: Pass each pattern separately to avoid comma-in-filename issues
        # Using multiple --include flags instead of comma-separated values
        # Also sanitize patterns to prevent argument injection and path traversal
        for pattern in include_patterns:
            safe_pattern = self._sanitize_pattern(pattern)
            if safe_pattern is not None:
                cmd.extend(["--include", safe_pattern])
            # Skip unsafe patterns silently (absolute paths, path traversal)

        for pattern in exclude_patterns:
            safe_pattern = self._sanitize_pattern(pattern)
            if safe_pattern is not None:
                cmd.extend(["--ignore", safe_pattern])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return f"## Repository Context\n\n{result.stdout}"
            else:
                # Fallback to simple packing on error
                return self._pack_simple(include_patterns, exclude_patterns)

        except subprocess.TimeoutExpired:
            return self._pack_simple(include_patterns, exclude_patterns)

    def _validate_file_path(self, file_path: Path) -> Path | None:
        """Validate file path is safe to read.

        Returns resolved path if safe, None if unsafe.
        """
        try:
            # SECURITY: Resolve to canonical path
            resolved = file_path.resolve()

            # SECURITY: Must be within repo_path
            repo_resolved = self.repo_path.resolve()
            try:
                resolved.relative_to(repo_resolved)
            except ValueError:
                return None  # Path escapes repo

            # SECURITY: Reject symlinks pointing outside repo
            if file_path.is_symlink():
                link_target = file_path.resolve()
                try:
                    link_target.relative_to(repo_resolved)
                except ValueError:
                    return None  # Symlink target escapes repo

            return resolved
        except (OSError, RuntimeError):
            # Symlink loop or other filesystem error
            return None

    def _pack_simple(
        self,
        include_patterns: list[str],
        exclude_patterns: list[str],
    ) -> str:
        """Simple file packing without repomix."""
        from fnmatch import fnmatch

        packed_files = []

        # Find matching files
        for pattern in include_patterns:
            if "*" in pattern or "?" in pattern:
                # Glob pattern
                for file_path in self.repo_path.rglob("*"):
                    if file_path.is_file():
                        # SECURITY: Validate path before processing
                        validated = self._validate_file_path(file_path)
                        if validated is None:
                            continue

                        rel_path = str(file_path.relative_to(self.repo_path))
                        if fnmatch(rel_path, pattern):
                            # Check excludes
                            if not any(fnmatch(rel_path, ex) for ex in exclude_patterns):
                                packed_files.append(validated)
            else:
                # Exact path - SECURITY: Validate before joining
                # Normalize to prevent path traversal
                file_path = self.repo_path / pattern

                # SECURITY: Validate resolved path stays within repo
                validated = self._validate_file_path(file_path)
                if validated is not None and validated.is_file():
                    packed_files.append(validated)

        if not packed_files:
            return ""

        # Pack files into context with size limits
        parts = ["## Repository Context\n"]
        repo_resolved = self.repo_path.resolve()
        total_size = 0
        skipped_large = 0
        skipped_total_limit = 0

        for file_path in sorted(set(packed_files)):
            try:
                # SECURITY: Check file size before reading to prevent memory exhaustion
                file_size = file_path.stat().st_size
                if file_size > self.MAX_FILE_SIZE:
                    skipped_large += 1
                    continue

                # Check if adding this file would exceed total limit
                if total_size + file_size > self.MAX_TOTAL_SIZE:
                    skipped_total_limit += 1
                    continue

                content = file_path.read_text()
                total_size += len(content.encode("utf-8"))

                rel_path = file_path.relative_to(repo_resolved)
                parts.append(f"### {rel_path}\n\n```\n{content}\n```\n")
            except Exception:
                pass  # Skip unreadable files (binary, permission denied, etc.)

        # Add notice if files were skipped
        if skipped_large > 0 or skipped_total_limit > 0:
            notice = f"\n[Skipped {skipped_large} files exceeding {self.MAX_FILE_SIZE // 1024}KB limit"
            if skipped_total_limit > 0:
                notice += f", {skipped_total_limit} files due to total size limit"
            notice += "]\n"
            parts.append(notice)

        return "\n".join(parts)

    def _pack_with_budget(
        self,
        context_parts: dict[str, str],
        budget: int,
    ) -> str:
        """Pack context within token budget, pruning if necessary.

        Uses simple character-based estimation (4 chars ~= 1 token).
        """
        # Rough token estimation
        char_budget = budget * 4

        # Try full context first
        full = self._combine(context_parts)
        if len(full) <= char_budget:
            return full

        # Progressive pruning by priority
        for drop_key in reversed(self.PRIORITY_ORDER):
            if drop_key in context_parts:
                del context_parts[drop_key]
                pruned = self._combine(context_parts)
                if len(pruned) <= char_budget:
                    return pruned

        # Still too large - truncate what's left
        return full[:char_budget] + "\n\n[Context truncated due to token limit]"

    def _combine(self, context_parts: dict[str, str]) -> str:
        """Combine context parts in priority order."""
        result = []
        for key in self.PRIORITY_ORDER:
            if key in context_parts:
                result.append(context_parts[key])

        # Add remaining parts not in priority order
        for key, value in context_parts.items():
            if key not in self.PRIORITY_ORDER:
                result.append(value)

        return "\n\n---\n\n".join(result)

    def render_prompt(
        self,
        template_name: str,
        role: RoleConfig,
        task_description: str,
        **kwargs: Any,
    ) -> str:
        """Render a Jinja2 prompt template."""
        if self.jinja_env is None:
            # Fallback to simple prompt building
            context = self.pack_context(role, task_description, **kwargs)
            return f"{context}\n\n## Instructions\n\n{task_description}"

        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(
                role=role,
                task=task_description,
                **kwargs,
            )
        except Exception:
            # Fallback on template error
            context = self.pack_context(role, task_description, **kwargs)
            return context

    def get_git_diff(self, staged: bool = True) -> str:
        """Get git diff for context."""
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    def get_file_tree(self, max_depth: int = 3) -> str:
        """Get repository file tree for context."""
        try:
            result = subprocess.run(
                ["find", ".", "-maxdepth", str(max_depth), "-type", "f"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                # Filter out hidden files and format
                files = [
                    f for f in result.stdout.strip().split("\n")
                    if not any(part.startswith(".") for part in f.split("/")[1:])
                ]
                return "\n".join(sorted(files))
            return ""
        except Exception:
            return ""
