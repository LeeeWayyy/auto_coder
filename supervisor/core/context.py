"""Context packing for AI workers.

Packs relevant files and context for each role based on configuration.
Uses Repomix for file packing when available, falls back to simple packing.
"""

import subprocess
from pathlib import Path
from typing import Any

from jinja2 import FileSystemLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from supervisor.core.roles import RoleConfig


# SECURITY: Allowlist of valid template names (shipped with package)
ALLOWED_TEMPLATES = frozenset([
    "_base.j2",
    "_output_schema.j2",
    "planning.j2",
    "implement.j2",
    "review_strict.j2",
])


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

    # Cap for always_include to prevent unbounded growth
    MAX_ALWAYS_INCLUDE_SIZE = 2 * 1024 * 1024  # 2MB total for always_include

    # Reserve budget for system_prompt + task + output_requirements template overhead
    TEMPLATE_OVERHEAD_CHARS = 4000  # ~1000 tokens for template boilerplate

    # Sentinel patterns that are NOT file patterns (handled separately)
    SENTINELS = frozenset(["git diff --cached", "$CHANGED_FILES"])

    # Priority order for context pruning when over budget (pack_context legacy flow)
    PRIORITY_ORDER = [
        "system_prompt",
        "task",
        "target_file",
        "imports",
        "related",
        "tree",
    ]

    # Keys that are NEVER pruned, regardless of budget
    PROTECTED_KEYS = frozenset(["system_prompt", "task", "always_include"])

    # Priority order for pruning (only non-protected keys are pruned)
    PRUNABLE_PRIORITY_ORDER = [
        "target_file",     # High priority (last to prune)
        "imports",
        "related",
        "files",
        "tree",            # Low priority (first to prune)
    ]

    # File-context specific priority order (maps from role-level keys to file-context keys)
    FILE_CONTEXT_PRIORITY = [
        "target_file",     # High priority - the file being modified
        "git_diff",        # Changes in progress
        "changed_files",   # Full content of changed files
        "imports",         # Import dependencies
        "related",         # Related files
        "files",           # General file context
        "tree",            # Directory tree (low priority - first to prune)
    ]

    # Mapping from role priority_order keys to file-context keys
    PRIORITY_KEY_MAP = {
        "target_file": "target_file",
        "git_diff": "git_diff",
        "changed_files": "changed_files",
        "imports": "imports",
        "related": "related",
        "files": "files",
        "tree": "tree",
        # Role-level keys that don't apply to file context (ignored)
        "system_prompt": None,
        "task_description": None,
        "task": None,
        "readme": None,
        "docs": None,
        "standards": None,
    }

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path).absolute()
        self._repomix_available = self._check_repomix()

        # SECURITY: Template directory is package-internal, not user-controlled
        template_dir = Path(__file__).parent.parent / "prompts"

        # SECURITY: Use SandboxedEnvironment to prevent arbitrary code execution
        # StrictUndefined raises errors on undefined variables (catches typos)
        if template_dir.exists():
            self.jinja_env = SandboxedEnvironment(
                loader=FileSystemLoader(template_dir),
                undefined=StrictUndefined,
                autoescape=False,  # Not HTML, no XSS concern
                trim_blocks=True,
                lstrip_blocks=True,
            )
            # Register custom filters
            self.jinja_env.filters["truncate_lines"] = self._truncate_lines
            self.jinja_env.filters["format_diff"] = self._format_diff
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

    # --- Template-based prompt building (Phase 2) ---

    def build_full_prompt(
        self,
        template_name: str,
        role: RoleConfig,
        task_description: str,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
    ) -> str:
        """Build complete prompt: pack context + render template.

        This is the main entry point for prompt generation.
        It combines context packing and template rendering in one call.

        IMPORTANT: Template renders ALL content (system_prompt, task, context).
        pack_file_context() only returns file content, NOT system_prompt/task.
        This prevents duplication.

        Usage:
            packer = ContextPacker(repo_path)
            prompt = packer.build_full_prompt(
                "planning.j2",
                role=planner_role,
                task_description="Implement feature X",
                target_files=["src/main.py"],
            )
        """
        # Centralize fallback logic: if templates are unavailable, use legacy packing
        if self.jinja_env is None:
            return self.pack_context(role, task_description, target_files, extra_context)

        # Step 1: Pack FILE context only (not system_prompt/task - template handles those)
        file_context = self.pack_file_context(role, target_files, extra_context)

        # Step 2: Render template with role, task, and file context
        # Template handles: system_prompt, task, context (files), output requirements
        try:
            return self.render_prompt(template_name, role, task_description, context=file_context)
        except Exception:
            # Fallback: Use pack_context with full access to target_files and extra_context
            # This ensures no context is lost on template rendering failures
            return self.pack_context(role, task_description, target_files, extra_context)

    def pack_file_context(
        self,
        role: RoleConfig,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
    ) -> str:
        """Pack FILE context only, with budget reservation for template overhead.

        Used when templates handle system_prompt/task rendering.
        Budget is reduced by TEMPLATE_OVERHEAD_CHARS to reserve space for
        system_prompt, task, and output_requirements in the final prompt.
        """
        context_parts: dict[str, str] = {}

        # Pack always_include files (protected)
        always_include = role.context.get("always_include", [])
        if always_include:
            always_content = self._pack_always_include(always_include)
            if always_content:
                context_parts["always_include"] = always_content

        # Pack target files
        if target_files:
            target_content = self._pack_target_files(target_files)
            if target_content:
                context_parts["target_file"] = target_content

        # Handle role-specific context directives
        # Check for git_diff boolean flag OR "git diff --cached" in include patterns
        include_patterns = role.context.get("include", [])
        if role.context.get("git_diff") or "git diff --cached" in include_patterns:
            git_diff = self.get_git_diff(staged=True)
            if git_diff:
                context_parts["git_diff"] = f"## Git Diff (Staged)\n\n```diff\n{git_diff}\n```"

        # Handle $CHANGED_FILES - resolve to list of files from git diff
        if "$CHANGED_FILES" in include_patterns:
            changed_files = self._get_changed_files()
            if changed_files:
                changed_content = self._pack_target_files(changed_files)
                if changed_content:
                    context_parts["changed_files"] = changed_content

        # Pack role-specific files
        # NOTE: Skip $TARGET patterns since target_files are already packed above
        # This prevents duplication when role includes have $TARGET
        file_context = self._pack_files(role, target_files, skip_targets=True)
        if file_context:
            context_parts["files"] = file_context

        # Add extra context (test output, etc.)
        if extra_context:
            for key, value in extra_context.items():
                context_parts[key] = f"## {key.replace('_', ' ').title()}\n\n{value}"

        # Budget for file context = total budget - template overhead
        # Clamp to prevent negative budget for small token_budget values
        file_budget = max(0, role.token_budget - (self.TEMPLATE_OVERHEAD_CHARS // 4))
        # Let _combine_file_context handle all budget scenarios including zero budget
        return self._combine_file_context(context_parts, file_budget, role)

    def _combine_file_context(
        self,
        context_parts: dict[str, str],
        budget: int,
        role: RoleConfig,
    ) -> str:
        """Combine file context parts within budget, using file-specific priority.

        Maps role.context.priority_order keys to file-context keys where applicable.
        Falls back to FILE_CONTEXT_PRIORITY when no overlap.
        Protected keys (always_include) are never pruned.
        """
        char_budget = budget * 4  # 4 chars ~= 1 token

        # Map role priority to file-context keys, filtering out non-applicable keys
        role_priority = role.context.get("priority_order", [])
        mapped_priority = []
        for key in role_priority:
            mapped = self.PRIORITY_KEY_MAP.get(key, key)  # Default: use key as-is
            if mapped is not None and mapped in context_parts:
                mapped_priority.append(mapped)

        # If no overlap with file context keys, use default file priority
        priority_order = mapped_priority if mapped_priority else self.FILE_CONTEXT_PRIORITY

        # Protected keys for file context (always_include is protected)
        protected = {"always_include"}
        protected_parts = {k: v for k, v in context_parts.items() if k in protected}
        prunable_parts = {k: v for k, v in context_parts.items() if k not in protected}

        def assemble_in_priority_order(parts: dict[str, str]) -> str:
            """Assemble parts in priority order, then any remaining keys."""
            result = []
            seen = set()
            # First: protected content (always_include)
            for key in protected:
                if key in parts:
                    result.append(parts[key])
                    seen.add(key)
            # Second: prunable content in priority order
            for key in priority_order:
                if key in parts and key not in seen:
                    result.append(parts[key])
                    seen.add(key)
            # Third: any remaining keys not in priority order
            for key, value in parts.items():
                if key not in seen:
                    result.append(value)
            return "\n\n".join(result)

        # Try full context first
        full = assemble_in_priority_order(context_parts)
        if len(full) <= char_budget:
            return full

        # Progressive pruning by role priority (reverse = low priority first)
        for drop_key in reversed(priority_order):
            if drop_key in prunable_parts:
                del prunable_parts[drop_key]
                current = assemble_in_priority_order({**protected_parts, **prunable_parts})
                if len(current) <= char_budget:
                    return current

        # Return protected content only
        return assemble_in_priority_order(protected_parts)

    def _get_changed_files(self) -> list[str]:
        """Get list of changed files from git diff --name-only --cached.

        Used to resolve $CHANGED_FILES pattern in role include patterns.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--cached"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")
            return []
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return []

    def _pack_target_files(self, target_files: list[str]) -> str:
        """Pack specific target files that are being modified.

        These get highest priority in context (after system_prompt and task).
        """
        parts = ["## Target Files\n"]

        for file_path in target_files:
            full_path = self.repo_path / file_path

            # Validate path stays within repo
            validated = self._validate_file_path(full_path)
            if validated is None:
                continue

            try:
                # Check file size
                if validated.stat().st_size > self.MAX_FILE_SIZE:
                    parts.append(f"### {file_path}\n\n[File exceeds size limit]\n")
                    continue

                content = validated.read_text()
                parts.append(f"### {file_path}\n\n```\n{content}\n```\n")
            except (IOError, OSError, UnicodeDecodeError):
                parts.append(f"### {file_path}\n\n[Could not read file]\n")

        return "\n".join(parts) if len(parts) > 1 else ""

    def _pack_always_include(self, patterns: list[str]) -> str:
        """Pack always_include files - protected but size-capped.

        Used for critical context like risk_limits.py, config files, etc.
        Never pruned by budget logic, but has its own hard cap.
        """
        parts = ["## Required Context\n"]
        total_size = 0

        for pattern in patterns:
            # Sanitize and resolve pattern
            safe_pattern = self._sanitize_pattern(pattern)
            if safe_pattern is None:
                continue

            # Handle glob patterns vs exact paths
            if "*" in safe_pattern or "?" in safe_pattern:
                for file_path in self.repo_path.glob(safe_pattern):
                    validated = self._validate_file_path(file_path)
                    if validated and validated.is_file():
                        added = self._append_file_content_with_cap(
                            parts, validated, total_size, self.MAX_ALWAYS_INCLUDE_SIZE
                        )
                        total_size += added
            else:
                file_path = self.repo_path / safe_pattern
                validated = self._validate_file_path(file_path)
                if validated and validated.is_file():
                    added = self._append_file_content_with_cap(
                        parts, validated, total_size, self.MAX_ALWAYS_INCLUDE_SIZE
                    )
                    total_size += added

        if total_size >= self.MAX_ALWAYS_INCLUDE_SIZE:
            parts.append(
                f"\n[WARNING: always_include capped at {self.MAX_ALWAYS_INCLUDE_SIZE // 1024}KB]"
            )

        return "\n".join(parts) if len(parts) > 1 else ""

    def _append_file_content_with_cap(
        self, parts: list[str], file_path: Path, current_size: int, max_size: int
    ) -> int:
        """Append file content with size tracking. Returns bytes added."""
        try:
            file_size = file_path.stat().st_size

            # Check per-file limit
            if file_size > self.MAX_FILE_SIZE:
                rel_path = file_path.relative_to(self.repo_path)
                parts.append(f"### {rel_path}\n\n[File exceeds size limit]\n")
                return 0

            # Check total limit
            if current_size + file_size > max_size:
                rel_path = file_path.relative_to(self.repo_path)
                parts.append(f"### {rel_path}\n\n[Skipped: would exceed total size cap]\n")
                return 0

            content = file_path.read_text()
            rel_path = file_path.relative_to(self.repo_path)
            parts.append(f"### {rel_path}\n\n```\n{content}\n```\n")
            return len(content.encode("utf-8"))
        except (IOError, OSError, UnicodeDecodeError):
            return 0  # Skip unreadable files

    def _truncate_lines(self, text: str, max_lines: int) -> str:
        """Truncate text to max_lines, adding notice if truncated."""
        lines = text.split("\n")
        if len(lines) <= max_lines:
            return text
        return "\n".join(lines[:max_lines]) + f"\n\n[Truncated {len(lines) - max_lines} lines]"

    def _format_diff(self, diff: str) -> str:
        """Format git diff for prompt inclusion."""
        return f"```diff\n{diff}\n```"

    # --- Legacy context packing (still used for unknown roles) ---

    def pack_context(
        self,
        role: RoleConfig,
        task_description: str,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
    ) -> str:
        """Pack context for a role.

        Legacy/fallback method for unknown roles. Returns complete prompt
        including system_prompt, task, and files.

        For known roles (planner, implementer, reviewer), use build_full_prompt()
        which uses templates.

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

        # 3. Pack target files (high priority - files being modified)
        if target_files:
            target_content = self._pack_target_files(target_files)
            if target_content:
                context_parts["target_file"] = target_content

        # 4. Pack always_include files (protected, never pruned)
        always_include = role.context.get("always_include", [])
        if always_include:
            always_content = self._pack_always_include(always_include)
            if always_content:
                context_parts["always_include"] = always_content

        # 5. Handle role-specific context directives (same as pack_file_context)
        include_patterns = role.context.get("include", [])
        if role.context.get("git_diff") or "git diff --cached" in include_patterns:
            git_diff = self.get_git_diff(staged=True)
            if git_diff:
                context_parts["git_diff"] = f"## Git Diff (Staged)\n\n```diff\n{git_diff}\n```"

        if "$CHANGED_FILES" in include_patterns:
            changed_files = self._get_changed_files()
            if changed_files:
                changed_content = self._pack_target_files(changed_files)
                if changed_content:
                    context_parts["changed_files"] = changed_content

        # 6. Pack role-specific files (skip_targets=True to avoid duplication)
        file_context = self._pack_files(role, target_files, skip_targets=True)
        if file_context:
            context_parts["files"] = file_context

        # 7. Add extra context (test output, etc.)
        if extra_context:
            for key, value in extra_context.items():
                context_parts[key] = f"## {key.replace('_', ' ').title()}\n\n{value}"

        # 8. Enforce token budget with progressive pruning
        return self._pack_with_budget(context_parts, role.token_budget)

    def _pack_files(
        self,
        role: RoleConfig,
        target_files: list[str] | None = None,
        skip_targets: bool = False,
    ) -> str:
        """Pack files using repomix or fallback to simple packing.

        Args:
            role: Role configuration
            target_files: Specific files to focus on
            skip_targets: If True, skip $TARGET expansion (already packed via _pack_target_files)
        """
        include_patterns = role.include_patterns.copy()
        exclude_patterns = role.exclude_patterns.copy()

        # Add target files to includes (unless skip_targets=True)
        if target_files and not skip_targets:
            include_patterns.extend(target_files)

        # IMPORTANT: Pass skip_targets to _resolve_patterns to ignore $TARGET expansion
        include_patterns = self._resolve_patterns(include_patterns, target_files, skip_targets)

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
        skip_targets: bool = False,
    ) -> list[str]:
        """Resolve special pattern variables.

        Handles:
        - $TARGET: Expands to target_files (unless skip_targets=True)
        - $TARGET_IMPORTS: TODO
        - $CHANGED_FILES: Sentinel for changed files, not a file (stripped here)
        - "git diff --cached": Sentinel for git diff, not a file (stripped here)
        """
        resolved = []
        for pattern in patterns:
            if pattern in self.SENTINELS:
                # Skip sentinels - they're handled separately, not file patterns
                continue
            elif pattern == "$TARGET":
                # Skip $TARGET expansion when already packed via _pack_target_files
                if not skip_targets and target_files:
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
        """Render a Jinja2 prompt template.

        SECURITY: Template name is validated against allowlist.

        NOTE: Prefer using build_full_prompt() which handles context packing.
        Use this directly only when you have pre-built context.
        """
        # SECURITY: Validate template name against allowlist
        if template_name not in ALLOWED_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Allowed: {sorted(ALLOWED_TEMPLATES)}"
            )

        # jinja_env availability is checked in build_full_prompt
        # Let exceptions propagate to build_full_prompt for robust fallback
        template = self.jinja_env.get_template(template_name)
        return template.render(
            role=role,
            task=task_description,
            **kwargs,
        )

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
