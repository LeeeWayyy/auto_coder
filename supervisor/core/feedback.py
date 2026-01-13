"""Structured feedback generation for gate failures."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

from supervisor.core.gates import GateResult


class FeedbackGenerator(Protocol):
    """Protocol for generating structured feedback from gate failures."""

    def generate(self, gate_result: GateResult, context: str = "") -> str:
        """Generate feedback from gate result.

        Args:
            gate_result: The failed gate result
            context: Optional additional context (e.g., task description)

        Returns:
            Structured feedback string for retry
        """
        ...


class StructuredFeedbackGenerator:
    """Generate structured feedback with parsing for common output formats."""

    MAX_ISSUES = 20
    MAX_OUTPUT_CHARS = 4000

    # Windows path regex part: handles drive letters (C:\), UNC (\\server\share),
    # and extended paths (\\?\C:\). Used across multiple parsers.
    _WIN_PATH_RE = r"(?:[A-Za-z]:|\\\\(?:[?.]\\)?(?:[A-Za-z]:)?)?[^:\n]+"

    def __init__(self) -> None:
        self._parsers: dict[str, Callable[[str], list[dict[str, str]]]] = {
            "pytest": self._parse_pytest,
            "ruff": self._parse_ruff,
            "mypy": self._parse_mypy,
            "bandit": self._parse_bandit,
        }

    def generate(self, gate_result: GateResult, context: str = "") -> str:
        """Generate structured feedback with parsed output."""
        output = gate_result.output or ""
        gate_name = gate_result.gate_name
        output_type = self._detect_output_type(output, gate_name)
        parser = self._parsers.get(output_type)
        parsed = parser(output) if parser else []
        return self._format_feedback(gate_result, parsed, context)

    def _detect_output_type(self, output: str, gate_name: str) -> str:
        """Detect output type from content patterns."""
        lowered = (output or "").lower()
        gate_name = (gate_name or "").lower()

        if "pytest" in lowered or "failed" in lowered and "collected" in lowered:
            return "pytest"
        if "ruff" in lowered or re.search(
            rf"^{self._WIN_PATH_RE}:\d+:\d+:\s*[A-Z]\d+\s+", output, re.M
        ):
            return "ruff"
        if "mypy" in lowered or re.search(rf"^{self._WIN_PATH_RE}:\d+:\s*error:", output, re.M):
            return "mypy"
        if "bandit" in lowered or "severity:" in lowered and "confidence:" in lowered:
            return "bandit"

        # Fall back to gate name hints
        if "test" in gate_name:
            return "pytest"
        if "lint" in gate_name or "ruff" in gate_name:
            return "ruff"
        if "type" in gate_name or "mypy" in gate_name:
            return "mypy"
        if "security" in gate_name or "bandit" in gate_name:
            return "bandit"

        return ""

    def _parse_pytest(self, output: str) -> list[dict[str, str]]:
        """Parse pytest output into structured failures.

        Prioritizes detailed failure sections (E lines) over summary FAILED lines
        to avoid duplicate entries and provide more specific error context.
        """
        detailed_issues: list[dict[str, str]] = []
        summary_issues: list[dict[str, str]] = []
        summary_re = re.compile(r"^FAILED\s+(?P<test>[^\s]+)\s*-?\s*(?P<msg>.*)$")
        section_re = re.compile(r"^_{3,}\s*(?P<test>.+?)\s*_{3,}$")

        lines = output.splitlines()

        # Parse detailed failure sections (E lines) - preferred source
        current_test: str | None = None
        for line in lines:
            sec = section_re.match(line.strip())
            if sec:
                current_test = sec.group("test").strip()
                continue
            if current_test and line.startswith("E   "):
                detailed_issues.append(
                    {
                        "test": current_test,
                        "message": line[4:].strip() or "(error)",
                    }
                )
                # Don't reset current_test - continue collecting E lines for this test

        # Parse summary FAILED lines - fallback only
        for line in lines:
            m = summary_re.match(line.strip())
            if m:
                summary_issues.append(
                    {
                        "test": m.group("test"),
                        "message": m.group("msg").strip() or "(no message)",
                    }
                )

        # Prioritize detailed issues; use summary only as fallback
        issues = detailed_issues if detailed_issues else summary_issues
        return issues[: self.MAX_ISSUES]

    def _parse_ruff(self, output: str) -> list[dict[str, str]]:
        """Parse ruff/flake8 output into structured violations."""
        issues: list[dict[str, str]] = []
        pattern = re.compile(
            rf"^(?P<file>{self._WIN_PATH_RE}):(?P<line>\d+):(?P<col>\d+):\s*(?P<code>[A-Z]\d+)\s+(?P<msg>.+)$",
            re.M,
        )
        for m in pattern.finditer(output):
            issues.append(
                {
                    "file": m.group("file"),
                    "line": m.group("line"),
                    "col": m.group("col"),
                    "code": m.group("code"),
                    "message": m.group("msg").strip(),
                }
            )
        return issues[: self.MAX_ISSUES]

    def _parse_mypy(self, output: str) -> list[dict[str, str]]:
        """Parse mypy output into structured type errors."""
        issues: list[dict[str, str]] = []
        pattern = re.compile(
            rf"^(?P<file>{self._WIN_PATH_RE}):(?P<line>\d+):(?:\s*(?P<col>\d+):)?\s*error:\s*(?P<msg>.+?)(?:\s+\[(?P<code>[^\]]+)\])?$",
            re.M,
        )
        for m in pattern.finditer(output):
            issues.append(
                {
                    "file": m.group("file"),
                    "line": m.group("line"),
                    "col": (m.group("col") or "").strip(),
                    "code": (m.group("code") or "").strip(),
                    "message": m.group("msg").strip(),
                }
            )
        return issues[: self.MAX_ISSUES]

    def _parse_bandit(self, output: str) -> list[dict[str, str]]:
        """Parse bandit output into structured security issues."""
        issues: list[dict[str, str]] = []
        # Bandit text format blocks
        issue_re = re.compile(r"^Issue:\s*(?P<issue>.+)$", re.M)
        severity_re = re.compile(
            r"^Severity:\s*(?P<severity>\w+)\s+Confidence:\s*(?P<confidence>\w+)", re.M
        )
        # For bandit location, use [^:]+ instead of [^:\n]+ since format is single-line
        bandit_path_re = self._WIN_PATH_RE.replace("[^:\n]+", "[^:]+")
        location_re = re.compile(rf"^Location:\s*(?P<file>{bandit_path_re}):(?P<line>\d+)", re.M)

        issue_iter = list(issue_re.finditer(output))
        if not issue_iter:
            return issues

        for idx, match in enumerate(issue_iter):
            start = match.start()
            end = issue_iter[idx + 1].start() if idx + 1 < len(issue_iter) else len(output)
            block = output[start:end]

            sev = severity_re.search(block)
            loc = location_re.search(block)

            issues.append(
                {
                    "issue": match.group("issue").strip(),
                    "severity": sev.group("severity") if sev else "",
                    "confidence": sev.group("confidence") if sev else "",
                    "file": loc.group("file") if loc else "",
                    "line": loc.group("line") if loc else "",
                }
            )

        return issues[: self.MAX_ISSUES]

    def _format_feedback(
        self,
        gate_result: GateResult,
        parsed_issues: list[dict[str, str]],
        context: str,
    ) -> str:
        """Format parsed issues into structured feedback."""
        header = f"Gate '{gate_result.gate_name}' failed."
        if gate_result.timed_out:
            header += " (timed out)"
        parts = [header]

        if context.strip():
            parts.append("")
            parts.append("Context:")
            parts.append(context.strip())

        if parsed_issues:
            parts.append("")
            parts.append("Issues:")
            for issue in parsed_issues[: self.MAX_ISSUES]:
                parts.append(f"- {self._format_issue(issue)}")
        else:
            parts.append("")
            parts.append("Issues: (none parsed)")

        output = (gate_result.output or "").strip()
        if output:
            parts.append("")
            parts.append("Gate output (truncated):")
            parts.append(self._truncate_output(output, self.MAX_OUTPUT_CHARS))

        return "\n".join(parts).strip() + "\n"

    @staticmethod
    def _format_issue(issue: dict[str, str]) -> str:
        """Format a single parsed issue into a readable line."""
        fields = []
        if "file" in issue and issue.get("file"):
            loc = issue["file"]
            if issue.get("line"):
                loc += f":{issue['line']}"
                if issue.get("col"):
                    loc += f":{issue['col']}"
            fields.append(loc)
        if issue.get("code"):
            fields.append(issue["code"])
        if issue.get("severity"):
            fields.append(f"severity={issue['severity']}")
        if issue.get("confidence"):
            fields.append(f"confidence={issue['confidence']}")

        msg = issue.get("message") or issue.get("issue") or "(no message)"
        if fields:
            return f"{' '.join(fields)}: {msg}"
        return msg

    @staticmethod
    def _truncate_output(output: str, max_chars: int) -> str:
        if len(output) <= max_chars:
            return output
        # Keep head and tail for better context
        truncated = len(output) - max_chars
        half = max_chars // 2
        head = output[:half]
        tail = output[-half:]
        return f"{head}\n...[truncated {truncated} chars]...\n{tail}"
