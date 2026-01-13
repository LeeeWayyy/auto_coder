"""Shared utility functions for supervisor core modules.

FIX (PR review): Extracted shared utilities to avoid duplication and circular imports.
"""

from pathlib import Path


def normalize_repo_path(
    file_path: str,
    repo_root: Path,
    fail_closed: bool = False,
) -> str:
    """Normalize a file path to canonical repo-relative form.

    FIX (PR review): Extracted to shared utility to avoid duplication between
    DAGScheduler._normalize_path and WorkflowCoordinator.execute_feature.

    Ensures consistent path comparison for conflict detection:
    ./a.py, a.py, and /full/path/to/repo/a.py all normalize to "a.py".

    Args:
        file_path: File path (relative or absolute)
        repo_root: Repository root path
        fail_closed: If True, raise on out-of-repo paths (default False)

    Returns:
        Normalized repo-relative path string

    Raises:
        ValueError: If path resolves outside repo root and fail_closed=True
    """
    path = Path(file_path)
    resolved_root = repo_root.resolve()

    resolved = path.resolve() if path.is_absolute() else (resolved_root / path).resolve()

    try:
        return str(resolved.relative_to(resolved_root))
    except ValueError:
        if fail_closed:
            raise ValueError(
                f"Path '{file_path}' resolves outside repo root '{resolved_root}'. "
                "All component files must be within the repository."
            )
        # Path outside repo - use resolved absolute path for tracking
        return str(resolved)
