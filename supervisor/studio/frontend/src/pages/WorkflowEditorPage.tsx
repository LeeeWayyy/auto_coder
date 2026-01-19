/**
 * Page for editing a workflow.
 */

import { useCallback, useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useWorkflow, useUpdateWorkflow, useCreateWorkflow } from '../hooks/useWorkflows';
import { useExecuteWorkflow } from '../hooks/useExecutions';
import { WorkflowCanvas } from '../components/WorkflowCanvas';
import type { WorkflowGraph } from '../types/workflow';

// Default workflow for "new" page
const defaultWorkflow: WorkflowGraph = {
  id: '',
  name: 'New Workflow',
  description: '',
  version: '1.0.0',
  nodes: [
    {
      id: 'start',
      type: 'task',
      label: 'Start',
      position: { x: 200, y: 100 },
      task_config: {
        role: 'implementer',
        description: 'Starting task',
      },
    },
  ],
  edges: [],
  entry_point: 'start',
  exit_points: ['start'],
  config: {
    fail_fast: true,
    max_parallel_nodes: 4,
  },
};

export function WorkflowEditorPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const isNew = id === 'new';

  // Fetch existing workflow
  const { data: existingWorkflow, isLoading } = useWorkflow(isNew ? undefined : id);
  const updateMutation = useUpdateWorkflow();
  const createMutation = useCreateWorkflow();
  const executeMutation = useExecuteWorkflow();

  // Local state for editing
  const [workflow, setWorkflow] = useState<WorkflowGraph | null>(null);
  const [hasChanges, setHasChanges] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [newId, setNewId] = useState(
    `workflow-${Date.now().toString(36)}`
  );

  // Reset state when navigating to a different workflow
  // React Router reuses the component, so we need to clear stale state
  useEffect(() => {
    setWorkflow(null);
    setHasChanges(false);
    setSaveError(null);
    setNewId(`workflow-${Date.now().toString(36)}`);
  }, [id]);

  // Initialize workflow from fetched data
  const currentWorkflow = workflow ?? existingWorkflow ?? (isNew ? { ...defaultWorkflow, id: newId } : null);

  // Handle workflow changes
  const handleChange = useCallback((updated: WorkflowGraph) => {
    setWorkflow(updated);
    setHasChanges(true);
  }, []);

  // Handle save - returns the saved workflow ID on success, undefined on failure
  const handleSave = useCallback(async (): Promise<string | undefined> => {
    if (!currentWorkflow) return undefined;

    setSaveError(null); // Clear previous errors
    try {
      if (isNew) {
        const created = await createMutation.mutateAsync({
          ...currentWorkflow,
          id: newId,
        });
        setHasChanges(false);
        navigate(`/workflows/${created.id}`, { replace: true });
        return created.id;
      } else {
        await updateMutation.mutateAsync({
          graphId: currentWorkflow.id,
          workflow: currentWorkflow,
        });
        setHasChanges(false);
        return currentWorkflow.id;
      }
    } catch (error) {
      // Extract error message for display
      const message = error instanceof Error ? error.message : 'Save failed';
      setSaveError(message);
      console.error('Save failed:', error);
      return undefined;
    }
  }, [currentWorkflow, isNew, newId, createMutation, updateMutation, navigate]);

  // Handle run
  const handleRun = useCallback(async () => {
    if (!currentWorkflow) return;

    // Determine which graph_id to use for execution
    let graphIdToRun = currentWorkflow.id;

    // Always save first if new workflow OR if there are unsaved changes.
    // New workflows must be saved before execution (they don't exist in DB yet).
    if (isNew || hasChanges) {
      const savedId = await handleSave();
      if (!savedId) {
        // Save failed, don't proceed with execution
        console.error('Cannot run: save failed');
        return;
      }
      graphIdToRun = savedId;
    }

    try {
      const execution = await executeMutation.mutateAsync({
        graph_id: graphIdToRun,
      });
      navigate(`/executions/${execution.execution_id}`);
    } catch (error) {
      console.error('Execution failed:', error);
    }
  }, [currentWorkflow, isNew, hasChanges, handleSave, executeMutation, navigate]);

  if (!isNew && isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Loading workflow...</div>
      </div>
    );
  }

  if (!currentWorkflow) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-500">Workflow not found</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b bg-white">
        <div className="flex items-center gap-4">
          <Link
            to="/"
            className="text-gray-500 hover:text-gray-700"
          >
            Back
          </Link>
          <div>
            <input
              type="text"
              value={currentWorkflow.name}
              onChange={(e) =>
                handleChange({ ...currentWorkflow, name: e.target.value })
              }
              className="text-lg font-semibold bg-transparent border-b border-transparent hover:border-gray-300 focus:border-blue-500 focus:outline-none"
            />
            <div className="text-xs text-gray-500 font-mono">
              {isNew ? (
                <input
                  type="text"
                  value={newId}
                  onChange={(e) => setNewId(e.target.value)}
                  placeholder="workflow-id"
                  className="bg-transparent border-b border-gray-300 focus:border-blue-500 focus:outline-none"
                />
              ) : (
                currentWorkflow.id
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {saveError && (
            <span className="text-sm text-red-600">{saveError}</span>
          )}
          {hasChanges && !saveError && (
            <span className="text-sm text-amber-600">Unsaved changes</span>
          )}
          <button
            onClick={handleSave}
            disabled={createMutation.isPending || updateMutation.isPending}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50"
          >
            {createMutation.isPending || updateMutation.isPending
              ? 'Saving...'
              : 'Save'}
          </button>
          <button
            onClick={handleRun}
            disabled={executeMutation.isPending || createMutation.isPending || updateMutation.isPending}
            className="px-4 py-2 text-sm font-medium text-white bg-emerald-600 rounded-md hover:bg-emerald-700 disabled:opacity-50"
          >
            {executeMutation.isPending ? 'Starting...' : (createMutation.isPending || updateMutation.isPending) ? 'Saving...' : 'Run'}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1">
        <WorkflowCanvas
          workflow={currentWorkflow}
          onWorkflowChange={handleChange}
        />
      </div>
    </div>
  );
}
