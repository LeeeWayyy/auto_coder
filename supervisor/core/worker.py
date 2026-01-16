"""Background worker for graph workflow execution.

This module implements workers that poll the database for active executions
and process them incrementally. Supports both single-execution and daemon modes.
"""

import asyncio
import logging

from supervisor.core.graph_engine import GraphOrchestrator

logger = logging.getLogger(__name__)


class WorkflowWorker:
    """
    Background worker that executes workflows.

    Design:
    - Polls DB for active executions
    - Processes one batch at a time per execution
    - Can run multiple workers for horizontal scaling
    """

    def __init__(self, orchestrator: GraphOrchestrator, poll_interval: float = 1.0):
        self.orchestrator = orchestrator
        self.poll_interval = poll_interval
        self.running = False

    async def run_until_complete(self, execution_id: str) -> str:
        """
        Run a single execution until completion.
        Returns final status.
        """
        while True:
            status = self.orchestrator.get_execution_status(execution_id)
            if status in ["completed", "failed", "cancelled"]:
                return status

            executed = await self.orchestrator.execute_next_batch(execution_id)

            # Only sleep when no work was done (waiting for external events or
            # async completions). When work is done, immediately check for more.
            if executed == 0:
                await asyncio.sleep(self.poll_interval)

    async def start_daemon(self):
        """
        Start daemon mode - process all active executions.
        """
        self.running = True

        while self.running:
            try:
                # Find all running executions
                active = self._get_active_executions()

                for execution_id in active:
                    await self.orchestrator.execute_next_batch(execution_id)

            except Exception as e:
                logger.error(f"Worker error: {e}")

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        """Stop the worker daemon"""
        self.running = False

    def _get_active_executions(self) -> list[str]:
        """Get list of active execution IDs from database."""
        with self.orchestrator.db._connect() as conn:
            rows = conn.execute("SELECT id FROM graph_executions WHERE status='running'").fetchall()
            return [row[0] for row in rows]
