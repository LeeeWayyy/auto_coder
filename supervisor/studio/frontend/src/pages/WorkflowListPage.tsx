/**
 * Page listing all workflows.
 */

import { Link } from 'react-router-dom';
import { useWorkflows, useDeleteWorkflow } from '../hooks/useWorkflows';
import type { WorkflowGraph } from '../types/workflow';

export function WorkflowListPage() {
  const { data: workflows, isLoading, error } = useWorkflows();
  const deleteMutation = useDeleteWorkflow();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading workflows...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 text-red-700 rounded-md">
        Error loading workflows: {error.message}
      </div>
    );
  }

  const handleDelete = (id: string, name: string) => {
    if (confirm(`Delete workflow "${name}"?`)) {
      deleteMutation.mutate(id);
    }
  };

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Workflows</h1>
        <Link
          to="/workflows/new"
          className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
        >
          New Workflow
        </Link>
      </div>

      {!workflows || workflows.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <p className="text-gray-500 mb-4">No workflows yet</p>
          <Link
            to="/workflows/new"
            className="text-blue-600 hover:text-blue-700 font-medium"
          >
            Create your first workflow
          </Link>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {workflows.map((workflow: WorkflowGraph) => (
            <WorkflowCard
              key={workflow.id}
              workflow={workflow}
              onDelete={() => handleDelete(workflow.id, workflow.name)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface WorkflowCardProps {
  workflow: WorkflowGraph;
  onDelete: () => void;
}

function WorkflowCard({ workflow, onDelete }: WorkflowCardProps) {
  return (
    <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <div>
          <h3 className="font-medium text-gray-900">{workflow.name}</h3>
          <p className="text-xs text-gray-500 font-mono">{workflow.id}</p>
        </div>
        {workflow.version && (
          <span className="text-xs px-2 py-1 bg-gray-100 rounded text-gray-600">
            v{workflow.version}
          </span>
        )}
      </div>

      {workflow.description && (
        <p className="text-sm text-gray-600 mb-3 line-clamp-2">
          {workflow.description}
        </p>
      )}

      <div className="flex items-center gap-2 text-xs text-gray-500 mb-4">
        <span>{workflow.nodes.length} nodes</span>
        <span>|</span>
        <span>{workflow.edges.length} edges</span>
      </div>

      <div className="flex items-center gap-2">
        <Link
          to={`/workflows/${workflow.id}`}
          className="flex-1 px-3 py-1.5 text-sm text-center font-medium text-blue-600 bg-blue-50 rounded hover:bg-blue-100"
        >
          Edit
        </Link>
        <Link
          to={`/workflows/${workflow.id}/run`}
          className="flex-1 px-3 py-1.5 text-sm text-center font-medium text-emerald-600 bg-emerald-50 rounded hover:bg-emerald-100"
        >
          Run
        </Link>
        <button
          onClick={onDelete}
          className="px-3 py-1.5 text-sm font-medium text-red-600 bg-red-50 rounded hover:bg-red-100"
        >
          Delete
        </button>
      </div>
    </div>
  );
}
