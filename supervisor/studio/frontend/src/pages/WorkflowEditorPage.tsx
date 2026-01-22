/**
 * Page for editing a workflow.
 */

import { useCallback, useState, useEffect, useMemo, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useWorkflow, useUpdateWorkflow, useCreateWorkflow } from '../hooks/useWorkflows';
import { useExecuteWorkflow } from '../hooks/useExecutions';
import { WorkflowCanvas } from '../components/WorkflowCanvas';
import { NodePalette, getDefaultConfig } from '../components/NodePalette';
import { PropertiesPanel, type PropertiesPanelHandle } from '../components/PropertiesPanel';
import type { WorkflowGraph, Node, NodeType } from '../types/workflow';

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
  const [runGoal, setRunGoal] = useState('');
  const [runInputs, setRunInputs] = useState('');
  const [runLabel, setRunLabel] = useState('');
  const [runInputError, setRunInputError] = useState<string | null>(null);
  const [nodeFieldErrors, setNodeFieldErrors] = useState<Record<string, string | null>>({});
  const [paletteOpen, setPaletteOpen] = useState(true);
  const [propertiesOpen, setPropertiesOpen] = useState(true);
  const propertiesPanelRef = useRef<PropertiesPanelHandle | null>(null);

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
    if (propertiesPanelRef.current) {
      const valid = propertiesPanelRef.current.validateNodeFields();
      if (!valid) return undefined;
    }

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
          setRunInputError('Run inputs must be a JSON object.');
          return;
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Invalid JSON input.';
        setRunInputError(message);
        return;
      }
    }
    if (runGoal.trim()) {
      inputData = { ...inputData, goal: runGoal.trim() };
    }
    setRunInputError(null);

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
    runLabel,
    runInputs,
  ]);

  const selectedNode = useMemo(() => {
    if (!currentWorkflow || !selectedNodeId) return null;
    return currentWorkflow.nodes.find((n) => n.id === selectedNodeId) || null;
  }, [currentWorkflow, selectedNodeId]);

  const handleNodeUpdate = useCallback(
    (patch: Partial<Node>) => {
      if (!currentWorkflow || !selectedNode) return;
      const updatedNodes = currentWorkflow.nodes.map((node) =>
        node.id === selectedNode.id ? { ...node, ...patch } : node
      );
      handleChange({ ...currentWorkflow, nodes: updatedNodes });
    },
    [currentWorkflow, selectedNode, handleChange]
  );

  // Counter to ensure unique IDs even within same millisecond
  const nodeIdCounterRef = useRef(0);

  const generateNodeId = useCallback((nodeType: NodeType) => {
    const timestamp = Date.now().toString(36);
    const suffix = (nodeIdCounterRef.current++).toString(36);
    return `${nodeType}-${timestamp}-${suffix}`;
  }, []);

  const handleNodeAdd = useCallback(
    (nodeType: NodeType, position: { x: number; y: number }) => {
      if (!currentWorkflow) return;
      const nextId = generateNodeId(nodeType);
      const countForType =
        currentWorkflow.nodes.filter((node) => node.type === nodeType).length + 1;
      const label = `${nodeType.charAt(0).toUpperCase()}${nodeType.slice(1)} ${countForType}`;
      const newNode: Node = {
        id: nextId,
        type: nodeType,
        label,
        description: '',
        position,
        ...getDefaultConfig(nodeType),
      };
      handleChange({ ...currentWorkflow, nodes: [...currentWorkflow.nodes, newNode] });
      setSelectedNodeId(nextId);
    },
    [currentWorkflow, generateNodeId, handleChange]
  );

  const addTaskNode = useCallback(() => {
    if (!currentWorkflow) return;
    handleNodeAdd('task', {
      x: 200,
      y: 100 + currentWorkflow.nodes.length * 120,
    });
  }, [currentWorkflow, handleNodeAdd]);

  const addEdge = useCallback(
    (source: string, target: string) => {
      if (!currentWorkflow) return;
      if (!source || !target || source === target) return;
      const edgeId = `${source}-${target}-${Date.now().toString(36)}`;
      const newEdge = {
        id: edgeId,
        source,
        target,
      };
      handleChange({ ...currentWorkflow, edges: [...currentWorkflow.edges, newEdge] });
    },
    [currentWorkflow, handleChange]
  );

  const removeSelectedNode = useCallback(() => {
    if (!currentWorkflow || !selectedNode) return;
    const removedId = selectedNode.id;
    const nodes = currentWorkflow.nodes.filter((node) => node.id !== removedId);
    const edges = currentWorkflow.edges.filter(
      (edge) => edge.source !== removedId && edge.target !== removedId
    );
    handleChange({ ...currentWorkflow, nodes, edges });
    setSelectedNodeId(null);
  }, [currentWorkflow, selectedNode, handleChange]);

  const canSave = !Object.values(nodeFieldErrors).some((error) => error !== null);
  const canRun = !runInputError;

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
            disabled={!canSave || createMutation.isPending || updateMutation.isPending}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50"
          >
            {createMutation.isPending || updateMutation.isPending
              ? 'Saving...'
              : 'Save'}
          </button>
          <button
            onClick={handleRun}
            disabled={!canRun || executeMutation.isPending || createMutation.isPending || updateMutation.isPending}
            className="px-4 py-2 text-sm font-medium text-white bg-emerald-600 rounded-md hover:bg-emerald-700 disabled:opacity-50"
          >
            {executeMutation.isPending ? 'Starting...' : (createMutation.isPending || updateMutation.isPending) ? 'Saving...' : 'Run'}
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden min-h-0">
        <div className={`border-r bg-white flex flex-col ${paletteOpen ? 'w-56 min-w-48' : 'w-10'}`}>
          <button
            type="button"
            onClick={() => setPaletteOpen((prev) => !prev)}
            className="h-10 text-xs font-semibold text-gray-600 border-b hover:bg-gray-50"
          >
            {paletteOpen ? '<' : '>'}
          </button>
          {paletteOpen && (
            <div className="flex-1 overflow-y-auto">
              <NodePalette onNodeAdd={handleNodeAdd} />
            </div>
          )}
        </div>

        <div className="flex-1 min-h-0 min-w-[400px]">
          <WorkflowCanvas
            workflow={currentWorkflow}
            onWorkflowChange={handleChange}
            onNodeSelect={setSelectedNodeId}
            onNodeAdd={handleNodeAdd}
          />
        </div>

        <div className={`border-l bg-white flex flex-col ${propertiesOpen ? 'w-80 min-w-64' : 'w-10'}`}>
          <button
            type="button"
            onClick={() => setPropertiesOpen((prev) => !prev)}
            className="h-10 text-xs font-semibold text-gray-600 border-b hover:bg-gray-50"
          >
            {propertiesOpen ? '>' : '<'}
          </button>
          {propertiesOpen && (
            <div className="flex-1 overflow-y-auto">
              <PropertiesPanel
                ref={propertiesPanelRef}
                selectedNode={selectedNode}
                workflow={currentWorkflow}
                onNodeUpdate={handleNodeUpdate}
                onNodeDelete={removeSelectedNode}
                onEdgeAdd={addEdge}
                runGoal={runGoal}
                onRunGoalChange={setRunGoal}
                runInputs={runInputs}
                onRunInputsChange={(value) => {
                  setRunInputs(value);
                  setRunInputError(null);
                }}
                runLabel={runLabel}
                onRunLabelChange={setRunLabel}
                runInputError={runInputError}
                onRunInputErrorChange={setRunInputError}
                onNodeErrorsChange={setNodeFieldErrors}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
