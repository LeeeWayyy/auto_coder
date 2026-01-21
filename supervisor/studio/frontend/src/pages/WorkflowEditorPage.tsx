/**
 * Page for editing a workflow.
 */

import { useCallback, useState, useEffect, useMemo } from 'react';
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
      description: 'Starting task',
      position: { x: 200, y: 100 },
      task_config: {
        role: 'implementer',
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
  const isNew = !id || id === 'new';

  // Fetch existing workflow
  const { data: existingWorkflow, isLoading, error: loadError } = useWorkflow(
    isNew ? undefined : id
  );
  const updateMutation = useUpdateWorkflow();
  const createMutation = useCreateWorkflow();
  const executeMutation = useExecuteWorkflow();

  // Local state for editing
  const [workflow, setWorkflow] = useState<WorkflowGraph | null>(null);
  const [hasChanges, setHasChanges] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [lastSavedSnapshot, setLastSavedSnapshot] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [newId, setNewId] = useState(
    `workflow-${Date.now().toString(36)}`
  );
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [edgeFrom, setEdgeFrom] = useState<string>('');
  const [edgeTo, setEdgeTo] = useState<string>('');
  const [runGoal, setRunGoal] = useState('');
  const [runInputs, setRunInputs] = useState('');
  const [runLabel, setRunLabel] = useState('');

  // Reset state when navigating to a different workflow
  // React Router reuses the component, so we need to clear stale state
  useEffect(() => {
    // Reset editor state only when the route param changes.
    setWorkflow(null);
    setHasChanges(false);
    setSaveError(null);
    setLastSavedSnapshot(null);
    setRunError(null);
    if (isNew) {
      setNewId(`workflow-${Date.now().toString(36)}`);
    }
  }, [id, isNew]);

  // Initialize workflow from fetched data. Memoize draft to avoid regenerating
  // a "new" workflow on every render which can reset the canvas state.
  const draftWorkflow = useMemo(
    () => ({ ...defaultWorkflow, id: newId }),
    [newId]
  );
  const currentWorkflow = workflow ?? existingWorkflow ?? (isNew ? draftWorkflow : null);

  useEffect(() => {
    if (isNew && !workflow) {
      setWorkflow(draftWorkflow);
      setHasChanges(true);
      setLastSavedSnapshot(null);
    }
  }, [isNew, workflow, draftWorkflow]);

  useEffect(() => {
    if (!isNew && existingWorkflow) {
      const serialized = JSON.stringify(existingWorkflow);
      setWorkflow(existingWorkflow);
      setLastSavedSnapshot(serialized);
      setHasChanges(false);
    }
  }, [isNew, existingWorkflow]);

  // Handle workflow changes
  const handleChange = useCallback((updated: WorkflowGraph) => {
    const serialized = JSON.stringify(updated);
    if (lastSavedSnapshot) {
      setHasChanges(serialized !== lastSavedSnapshot);
    } else {
      setHasChanges(true);
    }
    setWorkflow(updated);
  }, [lastSavedSnapshot]);

  // Handle save - returns the saved workflow ID on success, undefined on failure
  const handleSave = useCallback(async (): Promise<string | undefined> => {
    if (!currentWorkflow) return undefined;

    setSaveError(null); // Clear previous errors
    try {
      if (isNew) {
        const cleanedId = newId.trim();
        if (!cleanedId) {
          setSaveError('Workflow ID cannot be empty');
          return undefined;
        }
        const created = await createMutation.mutateAsync({
          ...currentWorkflow,
          id: cleanedId,
        });
        const createdSnapshot = JSON.stringify(created);
        setLastSavedSnapshot(createdSnapshot);
        setHasChanges(false);
        setWorkflow(created);
        const createdId = created.id || cleanedId;
        setNewId(createdId);
        navigate(`/workflows/${createdId}`, { replace: true });
        return createdId;
      } else {
        await updateMutation.mutateAsync({
          graphId: currentWorkflow.id,
          workflow: currentWorkflow,
        });
        setLastSavedSnapshot(JSON.stringify(currentWorkflow));
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

    setRunError(null);
    let inputData: Record<string, unknown> = {};
    if (runInputs.trim()) {
      try {
        const parsed = JSON.parse(runInputs);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          inputData = parsed as Record<string, unknown>;
        } else {
          setRunError('Run inputs must be a JSON object.');
          return;
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Invalid JSON input.';
        setRunError(message);
        return;
      }
    }
    if (runGoal.trim()) {
      inputData = { ...inputData, goal: runGoal.trim() };
    }

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
      const label =
        runLabel.trim() || `${graphIdToRun}-${Date.now().toString(36)}`;
      const execution = await executeMutation.mutateAsync({
        graph_id: graphIdToRun,
        workflow_id: label,
        input_data: inputData,
      });
      navigate(`/executions/${execution.execution_id}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Execution failed';
      setRunError(message);
      console.error('Execution failed:', error);
    }
  }, [
    currentWorkflow,
    isNew,
    hasChanges,
    handleSave,
    executeMutation,
    navigate,
    runGoal,
    runInputs,
  ]);

  const selectedNode = useMemo(() => {
    if (!currentWorkflow || !selectedNodeId) return null;
    return currentWorkflow.nodes.find((n) => n.id === selectedNodeId) || null;
  }, [currentWorkflow, selectedNodeId]);

  const updateSelectedNode = useCallback(
    (patch: Partial<WorkflowGraph['nodes'][number]>) => {
      if (!currentWorkflow || !selectedNode) return;
      const updatedNodes = currentWorkflow.nodes.map((node) =>
        node.id === selectedNode.id ? { ...node, ...patch } : node
      );
      handleChange({ ...currentWorkflow, nodes: updatedNodes });
    },
    [currentWorkflow, selectedNode, handleChange]
  );

  const updateTaskConfig = useCallback(
    (patch: Partial<NonNullable<WorkflowGraph['nodes'][number]['task_config']>>) => {
      if (!currentWorkflow || !selectedNode) return;
      const task_config = { ...(selectedNode.task_config || { role: 'implementer' }), ...patch };
      updateSelectedNode({ task_config });
    },
    [currentWorkflow, selectedNode, updateSelectedNode]
  );

  const updateGateConfig = useCallback(
    (patch: Partial<NonNullable<WorkflowGraph['nodes'][number]['gate_config']>>) => {
      if (!currentWorkflow || !selectedNode) return;
      const gate_config = {
        ...(selectedNode.gate_config || { gate_type: 'test' }),
        ...patch,
      };
      updateSelectedNode({ gate_config });
    },
    [currentWorkflow, selectedNode, updateSelectedNode]
  );

  // Counter to ensure unique IDs even within same millisecond
  const nodeIdCounterRef = React.useRef(0);

  const addTaskNode = useCallback(() => {
    if (!currentWorkflow) return;
    // Generate unique ID using timestamp + counter to prevent collisions
    const timestamp = Date.now().toString(36);
    const suffix = (nodeIdCounterRef.current++).toString(36);
    const nextId = `task-${timestamp}-${suffix}`;
    const newNode = {
      id: nextId,
      type: 'task' as const,
      label: `Task ${currentWorkflow.nodes.length + 1}`,
      description: '',
      position: { x: 200, y: 100 + currentWorkflow.nodes.length * 120 },
      task_config: { role: 'implementer' },
    };
    handleChange({ ...currentWorkflow, nodes: [...currentWorkflow.nodes, newNode] });
    setSelectedNodeId(nextId);
    if (!edgeFrom) {
      setEdgeFrom(nextId);
    }
  }, [currentWorkflow, handleChange]);

  useEffect(() => {
    if (!currentWorkflow || currentWorkflow.nodes.length === 0) return;
    if (!edgeFrom) {
      setEdgeFrom(currentWorkflow.nodes[0].id);
    }
    if (!edgeTo && currentWorkflow.nodes.length > 1) {
      setEdgeTo(currentWorkflow.nodes[1].id);
    }
  }, [currentWorkflow, edgeFrom, edgeTo]);

  const addEdge = useCallback(() => {
    if (!currentWorkflow) return;
    if (!edgeFrom || !edgeTo || edgeFrom === edgeTo) return;
    const edgeId = `${edgeFrom}-${edgeTo}-${Date.now().toString(36)}`;
    const newEdge = {
      id: edgeId,
      source: edgeFrom,
      target: edgeTo,
    };
    handleChange({ ...currentWorkflow, edges: [...currentWorkflow.edges, newEdge] });
  }, [currentWorkflow, edgeFrom, edgeTo, handleChange]);

  const removeSelectedNode = useCallback(() => {
    if (!currentWorkflow || !selectedNode) return;
    const removedId = selectedNode.id;
    const nodes = currentWorkflow.nodes.filter((node) => node.id !== removedId);
    const edges = currentWorkflow.edges.filter(
      (edge) => edge.source !== removedId && edge.target !== removedId
    );
    handleChange({ ...currentWorkflow, nodes, edges });
    setSelectedNodeId(null);
    if (edgeFrom === removedId) {
      setEdgeFrom(nodes[0]?.id || '');
    }
    if (edgeTo === removedId) {
      setEdgeTo(nodes[1]?.id || nodes[0]?.id || '');
    }
  }, [currentWorkflow, selectedNode, handleChange, edgeFrom, edgeTo]);

  if (!isNew && isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Loading workflow...</div>
      </div>
    );
  }

  if (!currentWorkflow) {
    if (isNew) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-gray-500">Initializing new workflow...</div>
        </div>
      );
    }
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-500">
          {loadError ? `Workflow load failed: ${loadError.message}` : 'Workflow not found'}
        </div>
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
              id="workflow-name"
              name="workflow-name"
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
                  id="workflow-id"
                  name="workflow-id"
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
          <button
            onClick={addTaskNode}
            className="px-3 py-2 text-sm font-medium text-blue-700 bg-blue-50 rounded-md hover:bg-blue-100"
          >
            Add Task
          </button>
          {saveError && (
            <span className="text-sm text-red-600">{saveError}</span>
          )}
          {runError && (
            <span className="text-sm text-red-600">{runError}</span>
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

      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Canvas */}
        <div className="flex-1 min-h-0">
          <WorkflowCanvas
            workflow={currentWorkflow}
            onWorkflowChange={handleChange}
            onNodeSelect={setSelectedNodeId}
          />
        </div>

        {/* Sidebar */}
        <div className="w-80 border-l bg-white p-4 overflow-y-auto">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Node Details</h3>
          {!selectedNode ? (
            <div className="text-sm text-gray-500">Select a node to edit its settings.</div>
          ) : (
            <div className="space-y-4">
              <div>
                <div className="text-xs text-gray-500 mb-1">Node ID</div>
                <div className="text-sm font-mono text-gray-800">{selectedNode.id}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">Type</div>
                <div className="text-sm text-gray-800">{selectedNode.type}</div>
              </div>
              <label className="block">
                <span className="text-xs text-gray-500">Label</span>
                <input
                  id="node-label"
                  name="node-label"
                  value={selectedNode.label || ''}
                  onChange={(e) => updateSelectedNode({ label: e.target.value })}
                  className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm"
                />
              </label>
              <label className="block">
                <span className="text-xs text-gray-500">Description</span>
                <input
                  id="node-description"
                  name="node-description"
                  value={selectedNode.description || ''}
                  onChange={(e) => updateSelectedNode({ description: e.target.value })}
                  className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm"
                />
              </label>
              <button
                type="button"
                onClick={removeSelectedNode}
                className="w-full px-3 py-1.5 text-sm font-medium text-red-700 bg-red-50 rounded hover:bg-red-100"
              >
                Remove Node
              </button>

              {selectedNode.type === 'task' && (
                <label className="block">
                  <span className="text-xs text-gray-500">Role</span>
                  <input
                    id="task-role"
                    name="task-role"
                    value={selectedNode.task_config?.role || ''}
                    onChange={(e) => updateTaskConfig({ role: e.target.value })}
                    className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm"
                    list="role-suggestions"
                    placeholder="implementer"
                  />
                  <datalist id="role-suggestions">
                    <option value="planner" />
                    <option value="implementer" />
                    <option value="reviewer" />
                    <option value="security_reviewer" />
                    <option value="tester" />
                    <option value="debugger" />
                  </datalist>
                </label>
              )}

              {selectedNode.type === 'gate' && (
                <label className="block">
                  <span className="text-xs text-gray-500">Gate Type</span>
                  <input
                    id="gate-type"
                    name="gate-type"
                    value={selectedNode.gate_config?.gate_type || ''}
                    onChange={(e) => updateGateConfig({ gate_type: e.target.value })}
                    className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm"
                    placeholder="test"
                  />
                </label>
              )}

              <div className="pt-4 border-t">
                <div className="text-xs text-gray-500 mb-2">Add Edge</div>
                <div className="space-y-2">
                  <select
                    id="edge-from"
                    name="edge-from"
                    value={edgeFrom}
                    onChange={(e) => setEdgeFrom(e.target.value)}
                    className="w-full rounded border border-gray-200 px-2 py-1 text-sm"
                  >
                    <option value="">From…</option>
                    {currentWorkflow.nodes.map((node) => (
                      <option key={node.id} value={node.id}>
                        {node.label || node.id}
                      </option>
                    ))}
                  </select>
                  <select
                    id="edge-to"
                    name="edge-to"
                    value={edgeTo}
                    onChange={(e) => setEdgeTo(e.target.value)}
                    className="w-full rounded border border-gray-200 px-2 py-1 text-sm"
                  >
                    <option value="">To…</option>
                    {currentWorkflow.nodes.map((node) => (
                      <option key={node.id} value={node.id}>
                        {node.label || node.id}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={addEdge}
                    className="w-full px-3 py-1.5 text-sm font-medium text-emerald-700 bg-emerald-50 rounded hover:bg-emerald-100"
                  >
                    Add Edge
                  </button>
                </div>
              </div>

              <div className="pt-4 border-t">
                <div className="text-xs text-gray-500 mb-2">Run Inputs</div>
                <div className="space-y-2">
                  <label className="block">
                    <span className="text-xs text-gray-500">Goal</span>
                    <input
                      id="run-goal"
                      name="run-goal"
                      value={runGoal}
                      onChange={(e) => setRunGoal(e.target.value)}
                      className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm"
                      placeholder="Describe the desired outcome"
                    />
                  </label>
                  <label className="block">
                    <span className="text-xs text-gray-500">Run Label (optional)</span>
                    <input
                      id="run-label"
                      name="run-label"
                      value={runLabel}
                      onChange={(e) => setRunLabel(e.target.value)}
                      className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm font-mono"
                      placeholder="auto-generated if empty"
                    />
                  </label>
                  <label className="block">
                    <span className="text-xs text-gray-500">JSON Inputs</span>
                    <textarea
                      id="run-inputs"
                      name="run-inputs"
                      value={runInputs}
                      onChange={(e) => setRunInputs(e.target.value)}
                      className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm font-mono"
                      rows={4}
                      placeholder='{"key":"value"}'
                    />
                    <div className="mt-1 text-[11px] text-gray-400">
                      Must be a JSON object (not an array).
                    </div>
                  </label>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
