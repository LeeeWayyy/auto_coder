"""Pluggable context selection strategies.

Phase 4 deliverable 4.3: Role-specific context packing strategies.
"""

import ast
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# FIX (PR review): Module-level helper to avoid code duplication
def detect_language(filepath: str) -> str:
    """Detect language from file extension for syntax highlighting.

    Args:
        filepath: Path to file

    Returns:
        Language identifier for syntax highlighting (e.g., "python", "javascript")
    """
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".md": "markdown",
        ".sql": "sql",
        ".css": "css",
        ".html": "html",
    }
    ext = Path(filepath).suffix.lower()
    return ext_map.get(ext, "")


class StrategyError(Exception):
    """Error in context strategy."""

    pass


@dataclass
class ContextResult:
    """Result of context packing from a strategy."""

    content: str
    token_count: int
    files_included: list[str]
    truncated: bool = False
    truncation_info: str | None = None


class ContextStrategy(ABC):
    """Base class for context selection strategies.

    STRATEGY TYPES:
    - planner_docset: Broad project docs for planning
    - implementer_targeted: Target file + imports + diff
    - reviewer_diff: Git changes + full new files + standards
    - security_audit: Auth code + deps + env handling
    - investigator_wide: Broad scan for exploration
    """

    name: str
    description: str
    token_budget: int

    @abstractmethod
    def pack(
        self,
        repo_path: Path,
        target_files: list[str] | None = None,
        extra_inputs: dict[str, Any] | None = None,
    ) -> ContextResult:
        """Pack context according to this strategy.

        Args:
            repo_path: Repository root path
            target_files: Files being modified (for targeted strategies)
            extra_inputs: Strategy-specific inputs (git_diff, etc.)

        Returns:
            ContextResult with packed content
        """
        pass


class PlannerDocsetStrategy(ContextStrategy):
    """Pack broad project documentation for planning.

    INCLUDES:
    - README.md
    - docs/ARCHITECTURE.md (if exists)
    - Task specifications (docs/TASKS/)
    - ADR documents (docs/ADRs/)
    - File tree (compressed)

    PRIORITY (high to low):
    1. README.md
    2. Task specification
    3. Architecture docs
    4. ADRs
    5. File tree
    """

    name = "planner_docset"
    description = "Broad project docs for planning phase"
    token_budget = 30000

    INCLUDE_PATTERNS = [
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/TASKS/**/*.md",
        "docs/ADRs/**/*.md",
    ]

    PRIORITY_ORDER = [
        "readme",
        "task",
        "architecture",
        "adrs",
        "tree",
    ]

    def pack(
        self,
        repo_path: Path,
        target_files: list[str] | None = None,
        extra_inputs: dict[str, Any] | None = None,
    ) -> ContextResult:
        parts: dict[str, str] = {}
        files_included: list[str] = []

        # README
        readme_path = repo_path / "README.md"
        if readme_path.exists():
            parts["readme"] = f"## README.md\n\n{readme_path.read_text()}"
            files_included.append("README.md")

        # Task specification (if provided in extra_inputs)
        task_spec = extra_inputs.get("task_spec") if extra_inputs else None
        if task_spec:
            parts["task"] = f"## Task Specification\n\n{task_spec}"

        # Architecture docs
        arch_path = repo_path / "docs" / "ARCHITECTURE.md"
        if arch_path.exists():
            parts["architecture"] = f"## Architecture\n\n{arch_path.read_text()}"
            files_included.append("docs/ARCHITECTURE.md")

        # ADRs (collect all)
        adr_dir = repo_path / "docs" / "ADRs"
        if adr_dir.exists():
            adr_content = []
            for adr_file in sorted(adr_dir.glob("*.md")):
                adr_content.append(f"### {adr_file.name}\n\n{adr_file.read_text()}")
                files_included.append(f"docs/ADRs/{adr_file.name}")
            if adr_content:
                parts["adrs"] = (
                    "## Architecture Decision Records\n\n" + "\n\n".join(adr_content)
                )

        # File tree (compressed)
        # FIX (Gemini review): Add Python fallback for portability
        tree_content = self._get_file_tree(repo_path)
        if tree_content:
            parts["tree"] = f"## Project Structure\n\n```\n{tree_content}\n```"

        # Combine with budget enforcement
        combined = self._combine_with_budget(parts, self.token_budget)

        return ContextResult(
            content=combined,
            token_count=len(combined) // 4,
            files_included=files_included,
            truncated=len(combined) < sum(len(p) for p in parts.values()),
        )

    def _get_file_tree(self, repo_path: Path, max_depth: int = 3) -> str:
        """Get file tree using tree command or Python fallback.

        FIX (Gemini review): Ensures context is available even without tree command.
        """
        # Try tree command first (better formatting)
        try:
            tree_output = subprocess.run(
                ["tree", "-L", str(max_depth), "-I", "node_modules|.git|__pycache__|.venv|venv"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if tree_output.returncode == 0:
                return tree_output.stdout
        except Exception:
            pass

        # Python fallback using os.walk
        IGNORE_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".pytest_cache"}
        lines = [str(repo_path.name)]

        def walk_tree(path: Path, prefix: str = "", depth: int = 0) -> None:
            if depth >= max_depth:
                return

            try:
                entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError:
                return

            entries = [e for e in entries if e.name not in IGNORE_DIRS]
            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")

                if entry.is_dir():
                    extension = "    " if is_last else "│   "
                    walk_tree(entry, prefix + extension, depth + 1)

        walk_tree(repo_path)
        return "\n".join(lines) if len(lines) > 1 else ""

    def _combine_with_budget(self, parts: dict[str, str], budget: int) -> str:
        """Combine parts within token budget, respecting priority."""
        char_budget = budget * 4

        # Try full content
        full = "\n\n".join(parts.values())
        if len(full) <= char_budget:
            return full

        # Progressive pruning (reverse priority)
        result_parts = parts.copy()
        for key in reversed(self.PRIORITY_ORDER):
            if key in result_parts:
                del result_parts[key]
                current = "\n\n".join(result_parts.values())
                if len(current) <= char_budget:
                    return current

        return "\n\n".join(result_parts.values())


class ImplementerTargetedStrategy(ContextStrategy):
    """Pack target file + resolved imports for implementation.

    INCLUDES:
    - Target file(s) being modified
    - Import dependencies of target files
    - Git diff (configurable source)

    PRIORITY (high to low):
    1. Target files
    2. Direct imports
    3. Git diff
    4. Indirect imports

    FIX (Codex review): Made diff source configurable via extra_inputs["diff_source"].
    Options: "staged" (default), "unstaged", "both", "head"
    """

    name = "implementer_targeted"
    description = "Target file + imports + diff for implementation"
    token_budget = 25000

    # Configurable diff sources
    DIFF_COMMANDS = {
        "staged": ["git", "diff", "--cached"],
        "unstaged": ["git", "diff"],
        "both": ["git", "diff", "HEAD"],
        "head": ["git", "diff", "HEAD~1", "HEAD"],
    }

    def pack(
        self,
        repo_path: Path,
        target_files: list[str] | None = None,
        extra_inputs: dict[str, Any] | None = None,
    ) -> ContextResult:
        parts: dict[str, str] = {}
        files_included: list[str] = []

        # Target files
        if target_files:
            target_content = []
            for tf in target_files:
                path = repo_path / tf
                if path.exists():
                    # Detect language from extension
                    lang = detect_language(tf)
                    target_content.append(
                        f"### {tf}\n\n```{lang}\n{path.read_text()}\n```"
                    )
                    files_included.append(tf)
                else:
                    target_content.append(f"### {tf}\n\n[File does not exist yet - new file]")
                    files_included.append(tf)
            parts["target_file"] = "## Target Files\n\n" + "\n\n".join(target_content)

        # Resolve imports
        if target_files:
            imports = self._resolve_imports(repo_path, target_files)
            if imports:
                import_content = []
                for imp_file in imports[:10]:  # Limit to 10 imports
                    path = repo_path / imp_file
                    if path.exists():
                        lang = detect_language(imp_file)
                        import_content.append(
                            f"### {imp_file}\n\n```{lang}\n{path.read_text()}\n```"
                        )
                        files_included.append(imp_file)
                if import_content:
                    parts["imports"] = (
                        "## Imported Dependencies\n\n" + "\n\n".join(import_content)
                    )

        # Git diff - FIX (Codex review): Configurable diff source
        git_diff = extra_inputs.get("git_diff") if extra_inputs else None
        diff_source = (
            extra_inputs.get("diff_source", "staged") if extra_inputs else "staged"
        )

        if not git_diff:
            try:
                diff_cmd = self.DIFF_COMMANDS.get(
                    diff_source, self.DIFF_COMMANDS["staged"]
                )
                result = subprocess.run(
                    diff_cmd,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    git_diff = result.stdout
            except Exception:
                pass

        if git_diff:
            source_label = {
                "staged": "Staged",
                "unstaged": "Unstaged",
                "both": "All",
                "head": "Last Commit",
            }
            label = source_label.get(diff_source, "Staged")
            parts["git_diff"] = f"## {label} Changes\n\n```diff\n{git_diff}\n```"

        # Combine with budget
        combined = self._combine_with_budget(parts, self.token_budget)

        return ContextResult(
            content=combined,
            token_count=len(combined) // 4,
            files_included=files_included,
        )

    def _resolve_imports(self, repo_path: Path, target_files: list[str]) -> list[str]:
        """Resolve Python imports from target files.

        Uses AST parsing to extract imports and map to file paths.
        """
        imports = set()

        for tf in target_files:
            path = repo_path / tf
            if not path.exists() or not tf.endswith(".py"):
                continue

            try:
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
            except SyntaxError:
                continue

        # Convert module names to file paths
        file_paths = []
        for imp in imports:
            # Try direct module file
            parts = imp.split(".")
            potential_paths = [
                "/".join(parts) + ".py",
                "/".join(parts) + "/__init__.py",
            ]
            for pp in potential_paths:
                if (repo_path / pp).exists():
                    file_paths.append(pp)
                    break

        return file_paths

    def _combine_with_budget(self, parts: dict[str, str], budget: int) -> str:
        """Combine parts within token budget."""
        char_budget = budget * 4
        priority = ["target_file", "imports", "git_diff"]

        full = "\n\n".join(parts.values())
        if len(full) <= char_budget:
            return full

        result_parts = parts.copy()
        for key in reversed(priority):
            if key in result_parts:
                del result_parts[key]
                current = "\n\n".join(result_parts.values())
                if len(current) <= char_budget:
                    return current

        return "\n\n".join(result_parts.values())


class ReviewerDiffStrategy(ContextStrategy):
    """Pack git changes + standards for code review.

    INCLUDES:
    - Git diff (staged changes)
    - Full content of new files
    - Coding standards docs

    PRIORITY (high to low):
    1. Git diff
    2. New files full content
    3. Standards docs
    """

    name = "reviewer_diff"
    description = "Git changes + full new files + standards for review"
    token_budget = 15000

    def pack(
        self,
        repo_path: Path,
        target_files: list[str] | None = None,
        extra_inputs: dict[str, Any] | None = None,
    ) -> ContextResult:
        parts: dict[str, str] = {}
        files_included: list[str] = []

        # Git diff
        try:
            diff_result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if diff_result.returncode == 0 and diff_result.stdout.strip():
                parts["git_diff"] = (
                    f"## Staged Changes\n\n```diff\n{diff_result.stdout}\n```"
                )
        except Exception:
            pass

        # New files (full content)
        try:
            new_files_result = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if new_files_result.returncode == 0 and new_files_result.stdout.strip():
                new_files = new_files_result.stdout.strip().split("\n")
                new_content = []
                for nf in new_files[:5]:  # Limit to 5 new files
                    path = repo_path / nf
                    if path.exists():
                        lang = detect_language(nf)
                        new_content.append(
                            f"### {nf}\n\n```{lang}\n{path.read_text()}\n```"
                        )
                        files_included.append(nf)
                if new_content:
                    parts["new_files"] = (
                        "## New Files (Full Content)\n\n" + "\n\n".join(new_content)
                    )
        except Exception:
            pass

        # Standards docs
        standards_dir = repo_path / "docs" / "STANDARDS"
        if standards_dir.exists():
            standards_content = []
            for std_file in sorted(standards_dir.glob("*.md"))[:3]:  # Limit to 3
                standards_content.append(
                    f"### {std_file.name}\n\n{std_file.read_text()}"
                )
                files_included.append(f"docs/STANDARDS/{std_file.name}")
            if standards_content:
                parts["standards"] = (
                    "## Coding Standards\n\n" + "\n\n".join(standards_content)
                )

        # Combine with budget
        combined = self._combine_with_budget(parts, self.token_budget)

        return ContextResult(
            content=combined,
            token_count=len(combined) // 4,
            files_included=files_included,
        )

    def _combine_with_budget(self, parts: dict[str, str], budget: int) -> str:
        char_budget = budget * 4
        priority = ["git_diff", "new_files", "standards"]

        full = "\n\n".join(parts.values())
        if len(full) <= char_budget:
            return full

        result_parts = parts.copy()
        for key in reversed(priority):
            if key in result_parts:
                del result_parts[key]
                current = "\n\n".join(result_parts.values())
                if len(current) <= char_budget:
                    return current

        return "\n\n".join(result_parts.values())


# Strategy registry
STRATEGIES: dict[str, type[ContextStrategy]] = {
    "planner_docset": PlannerDocsetStrategy,
    "implementer_targeted": ImplementerTargetedStrategy,
    "reviewer_diff": ReviewerDiffStrategy,
}


def get_strategy(name: str) -> ContextStrategy:
    """Get a context strategy by name.

    Args:
        name: Strategy name (e.g., "planner_docset")

    Returns:
        Instantiated strategy

    Raises:
        StrategyError: If strategy not found
    """
    if name not in STRATEGIES:
        raise StrategyError(
            f"Unknown strategy '{name}'. " f"Available: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[name]()


def get_strategy_for_role(role_name: str) -> ContextStrategy | None:
    """Get the appropriate strategy for a role.

    MAPPING:
    - planner -> planner_docset
    - implementer -> implementer_targeted
    - reviewer -> reviewer_diff
    - * -> None (use default context packing)
    """
    ROLE_STRATEGY_MAP = {
        "planner": "planner_docset",
        "implementer": "implementer_targeted",
        "reviewer": "reviewer_diff",
        "reviewer_gemini": "reviewer_diff",
        "reviewer_codex": "reviewer_diff",
    }

    strategy_name = ROLE_STRATEGY_MAP.get(role_name)
    if strategy_name:
        return get_strategy(strategy_name)

    # Check for role that extends a base role
    # (Would need RoleLoader integration for full implementation)
    return None


def register_strategy(name: str, strategy_class: type[ContextStrategy]) -> None:
    """Register a custom context strategy.

    Args:
        name: Strategy name for lookup
        strategy_class: Class implementing ContextStrategy
    """
    STRATEGIES[name] = strategy_class
