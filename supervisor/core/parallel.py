"""Parallel execution utilities for multi-model reviews.

Phase 4 deliverable 4.2: Execute multiple reviewers simultaneously and aggregate results.
"""

import logging
import threading
import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from supervisor.core.engine import ExecutionEngine

logger = logging.getLogger(__name__)


class ParallelExecutionError(Exception):
    """Error during parallel execution."""

    pass


@dataclass
class ReviewResult:
    """Result from a single reviewer.

    FIX (Codex fresh review): output is Optional since error paths set it to None.
    """

    role_name: str
    cli: str
    output: BaseModel | None  # None on error/timeout
    duration_seconds: float
    success: bool
    error: str | None = None


@dataclass
class AggregatedReviewResult:
    """Aggregated result from multiple parallel reviewers.

    APPROVAL POLICY OPTIONS:
    - ALL_APPROVED: All reviewers must approve
    - ANY_APPROVED: At least one reviewer approves
    - MAJORITY_APPROVED: >50% of reviewers approve
    """

    results: list[ReviewResult]
    approved: bool
    approval_policy: str
    summary: str

    @property
    def all_approved(self) -> bool:
        return all(r.success for r in self.results)

    @property
    def any_approved(self) -> bool:
        return any(r.success for r in self.results)

    @property
    def majority_approved(self) -> bool:
        approved_count = sum(1 for r in self.results if r.success)
        return approved_count > len(self.results) / 2

    def get_rejections(self) -> list[ReviewResult]:
        """Get list of reviewers that rejected."""
        return [r for r in self.results if not r.success]

    def get_issues(self) -> list[str]:
        """Collect all issues from all reviewers.

        FIX (Codex fresh review): Handle None output from failed/timed-out reviewers.
        """
        issues = []
        for result in self.results:
            # Skip if output is None (error/timeout case)
            if result.output is None:
                if result.error:
                    issues.append(f"[{result.role_name}] Error: {result.error}")
                continue
            if hasattr(result.output, "issues"):
                for issue in result.output.issues:
                    if hasattr(issue, "description"):
                        issues.append(f"[{result.role_name}] {issue.description}")
                    else:
                        issues.append(f"[{result.role_name}] {issue}")
        return issues


class ParallelReviewer:
    """Execute multiple reviewers in parallel.

    USAGE:
        reviewer = ParallelReviewer(engine, max_workers=3)
        result = reviewer.run_parallel_review(
            roles=["reviewer_gemini", "reviewer_codex"],
            task_description="Review auth implementation",
            target_files=["src/auth.py"],
            approval_policy="ALL_APPROVED",
        )
        if result.approved:
            # Proceed to commit
        else:
            # Handle rejections
    """

    APPROVAL_POLICIES = {"ALL_APPROVED", "ANY_APPROVED", "MAJORITY_APPROVED"}

    def __init__(
        self,
        engine: "ExecutionEngine",
        max_workers: int = 3,
        timeout: float = 300.0,
    ):
        self.engine = engine
        self.max_workers = max_workers
        self.timeout = timeout

    def run_parallel_review(
        self,
        roles: list[str],
        task_description: str,
        workflow_id: str,
        target_files: list[str] | None = None,
        extra_context: dict[str, str] | None = None,
        approval_policy: str = "ALL_APPROVED",
    ) -> AggregatedReviewResult:
        """Run multiple reviewers in parallel and aggregate results.

        Args:
            roles: List of reviewer role names (e.g., ["reviewer_gemini", "reviewer_codex"])
            task_description: What to review
            workflow_id: Workflow context
            target_files: Files to review
            extra_context: Additional context (git_diff, etc.)
            approval_policy: How to determine overall approval

        Returns:
            AggregatedReviewResult with individual results and approval decision
        """
        if approval_policy not in self.APPROVAL_POLICIES:
            raise ValueError(
                f"Invalid approval_policy '{approval_policy}'. "
                f"Must be one of: {self.APPROVAL_POLICIES}"
            )

        # FIX (Codex fresh review): Validate non-empty roles to avoid false positives
        # all([]) returns True, which would incorrectly approve with no reviewers
        if not roles:
            raise ValueError(
                "At least one reviewer role must be specified. "
                "Empty roles list would result in false approval."
            )

        results: list[ReviewResult] = []
        # FIX (Codex review): Cancellation event to signal timeout to workers
        # and prevent late results from being processed
        cancelled = threading.Event()
        # Track which roles have been processed to ignore late results
        processed_roles: set[str] = set()
        results_lock = threading.Lock()

        # FIX (Codex review v2): Use manual executor management to avoid
        # blocking on shutdown. The context manager calls shutdown(wait=True)
        # which defeats the timeout purpose.
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            # Submit all review tasks with cancellation event
            futures: dict[Future, str] = {}
            for role in roles:
                future = executor.submit(
                    self._run_single_review_with_cancel,
                    role,
                    task_description,
                    workflow_id,
                    target_files,
                    extra_context,
                    cancelled,
                )
                futures[future] = role

            # Collect results as they complete with timeout
            try:
                for future in as_completed(futures, timeout=self.timeout):
                    role = futures[future]
                    with results_lock:
                        if role in processed_roles:
                            continue  # Already processed (shouldn't happen, but guard)
                        processed_roles.add(role)

                    try:
                        result = future.result()
                        with results_lock:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Reviewer '{role}' failed: {e}")
                        with results_lock:
                            results.append(
                                ReviewResult(
                                    role_name=role,
                                    cli="unknown",
                                    output=None,
                                    duration_seconds=0.0,
                                    success=False,
                                    error=str(e),
                                )
                            )
            except FuturesTimeoutError:
                # Signal cancellation to any still-running workers
                cancelled.set()
                logger.warning(f"Parallel review timed out after {self.timeout}s")

                # Add timeout results for unprocessed roles
                # FIX (Codex review): Check future.done() before marking as timed out
                # A future may have completed between as_completed timeout and now
                with results_lock:
                    for future, role in futures.items():
                        if role not in processed_roles:
                            processed_roles.add(role)
                            if future.done():
                                # Future completed, get its actual result
                                try:
                                    result = future.result(timeout=0)
                                    results.append(result)
                                except Exception as e:
                                    logger.error(f"Reviewer '{role}' failed: {e}")
                                    results.append(
                                        ReviewResult(
                                            role_name=role,
                                            cli="unknown",
                                            output=None,
                                            duration_seconds=0.0,
                                            success=False,
                                            error=str(e),
                                        )
                                    )
                            else:
                                # Future still running, mark as timed out
                                future.cancel()
                                results.append(
                                    ReviewResult(
                                        role_name=role,
                                        cli="unknown",
                                        output=None,
                                        duration_seconds=self.timeout,
                                        success=False,
                                        error=f"Timed out after {self.timeout}s",
                                    )
                                )
        finally:
            # Signal cancellation and shutdown with grace period
            cancelled.set()
            # FIX (Codex review): Add brief grace period for in-progress work to settle
            # Reviewer threads may still be executing external CLI processes with side
            # effects (file writes, DB events, logs). Give them a short window to complete
            # gracefully before proceeding.
            #
            # Note: Python threads cannot be forcefully killed, and external CLI processes
            # spawned by run_role() will run to completion. The cancelled event signals
            # to workers that their results will be ignored, but cannot interrupt them.
            # We use a bounded wait to let in-progress work settle.
            GRACE_PERIOD_SECONDS = 5.0
            executor.shutdown(wait=False, cancel_futures=True)
            # Brief grace period - wait for any futures that might complete soon
            for future in futures:
                if not future.done():
                    try:
                        future.result(timeout=GRACE_PERIOD_SECONDS / len(futures))
                    except Exception:
                        pass  # Ignore - we've already recorded timeout results

        # FIX (Codex fresh review): Handle empty results as rejection
        # This can happen if all futures fail before any complete
        if not results:
            return AggregatedReviewResult(
                results=[],
                approved=False,
                approval_policy=approval_policy,
                summary="No review results - all reviewers failed or timed out.",
            )

        # Determine overall approval
        if approval_policy == "ALL_APPROVED":
            approved = all(r.success for r in results)
        elif approval_policy == "ANY_APPROVED":
            approved = any(r.success for r in results)
        else:  # MAJORITY_APPROVED
            approved_count = sum(1 for r in results if r.success)
            approved = approved_count > len(results) / 2

        # Build summary
        approved_names = [r.role_name for r in results if r.success]
        rejected_names = [r.role_name for r in results if not r.success]
        summary = f"Approved by: {approved_names}. Rejected by: {rejected_names}."

        return AggregatedReviewResult(
            results=results,
            approved=approved,
            approval_policy=approval_policy,
            summary=summary,
        )

    # FIX (Gemini review): Removed unused _run_single_review method
    # run_parallel_review uses _run_single_review_with_cancel instead

    def _run_single_review_with_cancel(
        self,
        role: str,
        task_description: str,
        workflow_id: str,
        target_files: list[str] | None,
        extra_context: dict[str, str] | None,
        cancelled: threading.Event,
    ) -> ReviewResult:
        """Execute a single reviewer with cancellation support.

        FIX (Codex review): Checks cancellation event before and after execution.
        If cancelled before execution, returns immediately with cancelled error.
        If cancelled after execution completes, result is still returned
        (but caller will ignore it via processed_roles check).

        Note: We cannot interrupt run_role mid-execution since it's calling
        external CLI processes. The cancellation event serves to:
        1. Prevent new work from starting after timeout
        2. Signal to caller that this result should be ignored
        """
        # Check if already cancelled before starting
        if cancelled.is_set():
            return ReviewResult(
                role_name=role,
                cli="unknown",
                output=None,
                duration_seconds=0.0,
                success=False,
                error="Cancelled before execution",
            )

        start_time = time.time()
        role_config = self.engine.role_loader.load_role(role)

        try:
            output = self.engine.run_role(
                role_name=role,
                task_description=task_description,
                workflow_id=workflow_id,
                target_files=target_files,
                extra_context=extra_context,
            )

            duration = time.time() - start_time

            # Check if cancelled during execution - result will be ignored by caller
            if cancelled.is_set():
                logger.debug(f"Reviewer '{role}' completed after timeout, result ignored")

            # Determine if approved (check output.status or review_status)
            approved = False
            if hasattr(output, "status"):
                approved = output.status == "APPROVED"
            if hasattr(output, "review_status"):
                approved = output.review_status == "APPROVED"

            return ReviewResult(
                role_name=role,
                cli=role_config.cli,
                output=output,
                duration_seconds=duration,
                success=approved,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ReviewResult(
                role_name=role,
                cli=role_config.cli,
                output=None,
                duration_seconds=duration,
                success=False,
                error=str(e),
            )


# FIX (PR review): Removed unused ParallelComponentExecutor class
# WorkflowCoordinator implements its own parallel execution in _run_continuous_parallel
