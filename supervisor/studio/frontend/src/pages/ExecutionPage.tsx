/**
 * Page for monitoring a workflow execution.
 */

import { useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useExecution } from '../hooks/useExecutions';
import { useWorkflow } from '../hooks/useWorkflows';
import { ExecutionMonitor } from '../components/ExecutionMonitor';
import type { ExecutionStatus } from '../types/workflow';

export function ExecutionPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  // Fetch execution
  const { data: execution, isLoading: execLoading, error: execError } = useExecution(id);

  // Fetch workflow (using graph_id from execution)
  const { data: workflow, isLoading: workflowLoading } = useWorkflow(
    execution?.graph_id
  );

  const handleComplete = useCallback(
    (status: ExecutionStatus) => {
      console.log('Execution completed with status:', status);
      // Could show a toast notification here
    },
    []
  );

  if (execLoading || workflowLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Loading execution...</div>
      </div>
    );
  }

  if (execError) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <div className="text-red-500">Error loading execution</div>
        <Link to="/" className="text-blue-600 hover:text-blue-700">
          Go back to workflows
        </Link>
      </div>
    );
  }

  if (!execution || !workflow) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <div className="text-red-500">Execution not found</div>
        <Link to="/" className="text-blue-600 hover:text-blue-700">
          Go back to workflows
        </Link>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Breadcrumb */}
      <div className="px-4 py-2 bg-gray-50 border-b text-sm">
        <Link to="/" className="text-gray-500 hover:text-gray-700">
          Workflows
        </Link>
        <span className="mx-2 text-gray-400">/</span>
        <Link
          to={`/workflows/${execution.workflow_id}`}
          className="text-gray-500 hover:text-gray-700"
        >
          {execution.workflow_id}
        </Link>
        <span className="mx-2 text-gray-400">/</span>
        <span className="text-gray-900">Execution {id?.slice(0, 8)}</span>
      </div>

      {/* Monitor */}
      <div className="flex-1">
        <ExecutionMonitor
          executionId={id!}
          workflow={workflow}
          onComplete={handleComplete}
        />
      </div>
    </div>
  );
}
