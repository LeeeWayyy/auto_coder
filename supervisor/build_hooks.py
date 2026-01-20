"""Build hooks for packaging.

Ensures the Studio frontend is built and included in the wheel/sdist when
npm is available. This runs during PEP 517 builds (e.g., pip install).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class BuildHook(BuildHookInterface):
    """Build the Studio frontend before packaging.

    Behavior:
    - If dist/ already exists, do nothing.
    - If npm is unavailable, emit a warning and continue.
    - If AUTO_CODER_SKIP_FRONTEND_BUILD=1, skip.
    - If AUTO_CODER_REQUIRE_FRONTEND_BUILD=1, fail on npm missing or build error.
    """

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if os.environ.get("AUTO_CODER_SKIP_FRONTEND_BUILD") == "1":
            return

        frontend_dir = Path(self.root) / "supervisor" / "studio" / "frontend"
        dist_dir = frontend_dir / "dist"
        if dist_dir.exists():
            return

        npm = shutil.which("npm")
        require = os.environ.get("AUTO_CODER_REQUIRE_FRONTEND_BUILD") == "1"
        if not npm:
            message = (
                "Studio frontend not built (npm not found). "
                "Install Node.js/npm or set AUTO_CODER_SKIP_FRONTEND_BUILD=1 to silence."
            )
            if require:
                raise RuntimeError(message)
            print(f"[auto_coder] {message}")
            return

        try:
            subprocess.run([npm, "install"], cwd=frontend_dir, check=True)
            subprocess.run([npm, "run", "build"], cwd=frontend_dir, check=True)
        except subprocess.CalledProcessError as exc:
            message = f"Studio frontend build failed: {exc}"
            if require:
                raise RuntimeError(message) from exc
            print(f"[auto_coder] {message}")
