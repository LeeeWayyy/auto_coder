/**
 * PropertiesPanel component for editing node properties and run inputs.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  forwardRef,
  useImperativeHandle,
} from 'react';
import type { Node, NodeType, WorkflowGraph } from '../types/workflow';

export interface PropertiesPanelHandle {
  validateNodeFields: () => boolean;
}

export interface PropertiesPanelProps {
  selectedNode: Node | null;
  workflow: WorkflowGraph;
  onNodeUpdate: (patch: Partial<Node>) => void;
  onNodeDelete: () => void;
  onEdgeAdd: (source: string, target: string) => void;
  runGoal: string;
  onRunGoalChange: (goal: string) => void;
  runInputs: string;
  onRunInputsChange: (inputs: string) => void;
  runLabel: string;
  onRunLabelChange: (label: string) => void;
  runInputError?: string | null;
  onRunInputErrorChange?: (error: string | null) => void;
  onNodeErrorsChange?: (errors: Record<string, string | null>) => void;
}

type ConditionValue =
  | string
  | number
  | boolean
  | Array<string | number | boolean>;

const BRANCH_OPERATORS = [
  { value: '==', label: 'equals (==)' },
  { value: '!=', label: 'not equals (!=)' },
  { value: '>', label: 'greater than (>)' },
  { value: '<', label: 'less than (<)' },
  { value: '>=', label: 'greater or equal (>=)' },
  { value: '<=', label: 'less or equal (<=)' },
  { value: 'in', label: 'in array (in)' },
  { value: 'not_in', label: 'not in array (not_in)' },
  { value: 'contains', label: 'contains' },
  { value: 'starts_with', label: 'starts with' },
  { value: 'ends_with', label: 'ends with' },
] as const;

function serializeConditionValue(value: ConditionValue | undefined): string {
  if (value === undefined || value === null) return '';
  if (Array.isArray(value)) return JSON.stringify(value);
  return String(value);
}

function parseConditionValue(input: string): ConditionValue {
  const trimmed = input.trim();

  if (!trimmed) return '';

  if (trimmed === 'true') return true;
  if (trimmed === 'false') return false;

  const num = Number(trimmed);
  if (!Number.isNaN(num) && trimmed !== '') return num;

  if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) return parsed as ConditionValue;
    } catch {
      // fall through to string
    }
  }

  return trimmed;
}

function parseJson<T>(value: string, fieldName: string): { data: T | null; error: string | null } {
  if (!value.trim()) {
    return { data: null, error: null };
  }
  try {
    const parsed = JSON.parse(value) as T;
    return { data: parsed, error: null };
  } catch (error) {
    return {
      data: null,
      error: `Invalid JSON in ${fieldName}: ${error instanceof Error ? error.message : 'Parse error'}`,
    };
  }
}

const NODE_TYPE_BADGES: Record<NodeType, string> = {
  task: 'T',
  gate: 'G',
  human: 'H',
  branch: 'B',
  merge: 'M',
  parallel: 'P',
  subgraph: 'S',
};

export const PropertiesPanel = forwardRef<PropertiesPanelHandle, PropertiesPanelProps>(
  function PropertiesPanel(
    {
      selectedNode,
      workflow,
      onNodeUpdate,
      onNodeDelete,
      onEdgeAdd,
      runGoal,
      onRunGoalChange,
      runInputs,
      onRunInputsChange,
      runLabel,
      onRunLabelChange,
      runInputError,
      onRunInputErrorChange,
      onNodeErrorsChange,
    },
    ref
  ) {
    const [nodeFieldErrors, setNodeFieldErrors] = useState<Record<string, string | null>>({});
    const [inputMappingText, setInputMappingText] = useState('');
    const [outputMappingText, setOutputMappingText] = useState('');
    const [runInputsOpen, setRunInputsOpen] = useState(true);
    const [edgeFrom, setEdgeFrom] = useState('');
    const [edgeTo, setEdgeTo] = useState('');

    useEffect(() => {
      onNodeErrorsChange?.(nodeFieldErrors);
    }, [nodeFieldErrors, onNodeErrorsChange]);

    useEffect(() => {
      setNodeFieldErrors({});
      if (selectedNode?.type === 'subgraph') {
        const inputMapping = selectedNode.subgraph_config?.input_mapping;
        const outputMapping = selectedNode.subgraph_config?.output_mapping;
        setInputMappingText(
          inputMapping ? JSON.stringify(inputMapping, null, 2) : ''
        );
        setOutputMappingText(
          outputMapping ? JSON.stringify(outputMapping, null, 2) : ''
        );
      } else {
        setInputMappingText('');
        setOutputMappingText('');
      }
    }, [selectedNode?.id, selectedNode?.type]);

    useEffect(() => {
      const nodeIds = new Set(workflow.nodes.map((node) => node.id));
      if (!edgeFrom || !nodeIds.has(edgeFrom)) {
        setEdgeFrom(workflow.nodes[0]?.id || '');
      }
      if (!edgeTo || !nodeIds.has(edgeTo)) {
        setEdgeTo(workflow.nodes[1]?.id || workflow.nodes[0]?.id || '');
      }
    }, [workflow.nodes, edgeFrom, edgeTo]);

    const handleJsonBlur = useCallback((value: string, field: string) => {
      const { error } = parseJson(value, field);
      setNodeFieldErrors((prev) => ({ ...prev, [field]: error }));
    }, []);

    const validateNodeFields = useCallback(() => {
      let hasErrors = false;
      const errors: Record<string, string | null> = {};

      if (selectedNode?.type === 'subgraph') {
        const inputResult = parseJson<Record<string, string>>(
          inputMappingText,
          'input_mapping'
        );
        errors.input_mapping = inputResult.error;
        if (inputResult.error) {
          hasErrors = true;
        } else {
          const subgraph_config = {
            ...(selectedNode.subgraph_config || { workflow_name: '' }),
            input_mapping: inputResult.data ?? undefined,
          };
          onNodeUpdate({ subgraph_config });
        }

        const outputResult = parseJson<Record<string, string>>(
          outputMappingText,
          'output_mapping'
        );
        errors.output_mapping = outputResult.error;
        if (outputResult.error) {
          hasErrors = true;
        } else {
          const subgraph_config = {
            ...(selectedNode.subgraph_config || { workflow_name: '' }),
            output_mapping: outputResult.data ?? undefined,
          };
          onNodeUpdate({ subgraph_config });
        }
      }

      setNodeFieldErrors(errors);
      return !hasErrors;
    }, [inputMappingText, onNodeUpdate, outputMappingText, selectedNode]);

    useImperativeHandle(
      ref,
      () => ({
        validateNodeFields,
      }),
      [validateNodeFields]
    );

    const branchConfig = useMemo(() => {
      return (
        selectedNode?.branch_config || {
          condition: { field: '', operator: '==', value: '' },
          on_true: '',
          on_false: '',
        }
      );
    }, [selectedNode?.branch_config]);

    const parallelBranches = selectedNode?.parallel_config?.branches || [];

    const handleRunInputsBlur = useCallback(() => {
      const { error } = parseJson<Record<string, unknown>>(runInputs, 'run inputs');
      onRunInputErrorChange?.(error);
    }, [runInputs, onRunInputErrorChange]);

    return (
      <div className="p-4 space-y-4">
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Node Details</h3>
          {!selectedNode ? (
            <div className="text-sm text-gray-500">Select a node to edit its settings.</div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-gray-500 mb-1">Node ID</div>
                  <div className="text-sm font-mono text-gray-800">{selectedNode.id}</div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-gray-600 bg-gray-100 rounded px-2 py-1">
                    {NODE_TYPE_BADGES[selectedNode.type]}
                  </span>
                  <span className="text-xs text-gray-500">{selectedNode.type}</span>
                </div>
              </div>
              <label className="block">
                <span className="text-xs text-gray-500">Label</span>
                <input
                  id="node-label"
                  name="node-label"
                  value={selectedNode.label || ''}
                  onChange={(e) => onNodeUpdate({ label: e.target.value })}
                  className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm"
                />
              </label>
              <label className="block">
                <span className="text-xs text-gray-500">Description</span>
                <input
                  id="node-description"
                  name="node-description"
                  value={selectedNode.description || ''}
                  onChange={(e) => onNodeUpdate({ description: e.target.value })}
                  className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm"
                />
              </label>
              <button
                type="button"
                onClick={onNodeDelete}
                className="w-full px-3 py-1.5 text-sm font-medium text-red-700 bg-red-50 rounded hover:bg-red-100"
              >
                Remove Node
              </button>

              {selectedNode.type === 'task' && (
                <div className="space-y-3">
                  <label className="block">
                    <span className="text-xs text-gray-500">Role</span>
                    <input
                      id="task-role"
                      name="task-role"
                      value={selectedNode.task_config?.role || ''}
                      onChange={(e) =>
                        onNodeUpdate({
                          task_config: {
                            ...(selectedNode.task_config || { role: 'implementer' }),
                            role: e.target.value,
                          },
                        })
                      }
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
                  <label className="block">
                    <span className="text-xs text-gray-500">Task Template</span>
                    <textarea
                      id="task-template"
                      name="task-template"
                      value={selectedNode.task_config?.task_template || ''}
                      onChange={(e) =>
                        onNodeUpdate({
                          task_config: {
                            ...(selectedNode.task_config || { role: 'implementer' }),
                            task_template: e.target.value,
                          },
                        })
                      }
                      className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm font-mono"
                      rows={4}
                    />
                  </label>
                </div>
              )}

              {selectedNode.type === 'gate' && (
                <div className="space-y-3">
                  <label className="block">
                    <span className="text-xs text-gray-500">Gate Type</span>
                    <input
                      id="gate-type"
                      name="gate-type"
                      value={selectedNode.gate_config?.gate_type || ''}
                      onChange={(e) =>
                        onNodeUpdate({
                          gate_config: {
                            ...(selectedNode.gate_config || { gate_type: 'test' }),
                            gate_type: e.target.value,
                          },
                        })
                      }
                      className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm"
                      placeholder="test"
                    />
                  </label>
                  <label className="flex items-center gap-2 text-sm text-gray-600">
                    <input
                      type="checkbox"
                      checked={Boolean(selectedNode.gate_config?.auto_approve)}
                      onChange={(e) =>
                        onNodeUpdate({
                          gate_config: {
                            ...(selectedNode.gate_config || { gate_type: 'test' }),
                            auto_approve: e.target.checked,
                          },
                        })
                      }
                    />
                    Auto approve
                  </label>
                </div>
              )}

              {selectedNode.type === 'human' && (
                <div className="space-y-3">
                  <label className="block">
                    <span className="text-xs text-gray-500">Title</span>
                    <input
                      id="human-title"
                      name="human-title"
                      value={selectedNode.human_config?.title || ''}
                      onChange={(e) =>
                        onNodeUpdate({
                          human_config: {
                            ...(selectedNode.human_config || { title: 'Review Required' }),
                            title: e.target.value,
                          },
                        })
                      }
                      className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm"
                      placeholder="Review Required"
                    />
                  </label>
                  <label className="block">
                    <span className="text-xs text-gray-500">Description</span>
                    <textarea
                      id="human-description"
                      name="human-description"
                      value={selectedNode.human_config?.description || ''}
                      onChange={(e) =>
                        onNodeUpdate({
                          human_config: {
                            ...(selectedNode.human_config || { title: 'Review Required' }),
                            description: e.target.value,
                          },
                        })
                      }
                      className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm"
                      rows={3}
                    />
                  </label>
                </div>
              )}

              {selectedNode.type === 'branch' && (
                <div className="space-y-3">
                  <label className="block text-sm">
                    <span className="text-gray-700">Field</span>
                    <input
                      type="text"
                      value={branchConfig.condition.field || ''}
                      onChange={(e) =>
                        onNodeUpdate({
                          branch_config: {
                            ...branchConfig,
                            condition: {
                              ...branchConfig.condition,
                              field: e.target.value,
                            },
                          },
                        })
                      }
                      className="mt-1 block w-full rounded border-gray-300"
                      placeholder="e.g., status, result.success"
                    />
                  </label>

                  <label className="block text-sm">
                    <span className="text-gray-700">Operator</span>
                    <select
                      value={branchConfig.condition.operator || '=='}
                      onChange={(e) =>
                        onNodeUpdate({
                          branch_config: {
                            ...branchConfig,
                            condition: {
                              ...branchConfig.condition,
                              operator: e.target.value as typeof branchConfig.condition.operator,
                            },
                          },
                        })
                      }
                      className="mt-1 block w-full rounded border-gray-300"
                    >
                      {BRANCH_OPERATORS.map((op) => (
                        <option key={op.value} value={op.value}>
                          {op.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="block text-sm">
                    <span className="text-gray-700">Value</span>
                    <input
                      type="text"
                      value={serializeConditionValue(branchConfig.condition.value)}
                      onChange={(e) =>
                        onNodeUpdate({
                          branch_config: {
                            ...branchConfig,
                            condition: {
                              ...branchConfig.condition,
                              value: parseConditionValue(e.target.value),
                            },
                          },
                        })
                      }
                      className="mt-1 block w-full rounded border-gray-300"
                      placeholder="e.g., success, true, 100, [1,2,3]"
                    />
                    <span className="text-xs text-gray-500">
                      Supports: strings, numbers, booleans (true/false), JSON arrays
                    </span>
                  </label>

                  <label className="block text-sm">
                    <span className="text-gray-700">On True (target node)</span>
                    <select
                      value={branchConfig.on_true || ''}
                      onChange={(e) =>
                        onNodeUpdate({
                          branch_config: { ...branchConfig, on_true: e.target.value },
                        })
                      }
                      className="mt-1 block w-full rounded border-gray-300"
                    >
                      <option value="">-- Select node --</option>
                      {workflow.nodes
                        .filter((n) => n.id !== selectedNode.id)
                        .map((n) => (
                          <option key={n.id} value={n.id}>
                            {n.label || n.id}
                          </option>
                        ))}
                    </select>
                  </label>

                  <label className="block text-sm">
                    <span className="text-gray-700">On False (target node)</span>
                    <select
                      value={branchConfig.on_false || ''}
                      onChange={(e) =>
                        onNodeUpdate({
                          branch_config: { ...branchConfig, on_false: e.target.value },
                        })
                      }
                      className="mt-1 block w-full rounded border-gray-300"
                    >
                      <option value="">-- Select node --</option>
                      {workflow.nodes
                        .filter((n) => n.id !== selectedNode.id)
                        .map((n) => (
                          <option key={n.id} value={n.id}>
                            {n.label || n.id}
                          </option>
                        ))}
                    </select>
                  </label>
                </div>
              )}

              {selectedNode.type === 'merge' && (
                <div className="space-y-3">
                  <label className="block text-sm">
                    <span className="text-gray-700">Wait For</span>
                    <select
                      value={selectedNode.merge_config?.wait_for || 'all'}
                      onChange={(e) =>
                        onNodeUpdate({
                          merge_config: {
                            ...(selectedNode.merge_config || { wait_for: 'all' }),
                            wait_for: e.target.value as 'all' | 'any',
                          },
                        })
                      }
                      className="mt-1 block w-full rounded border-gray-300"
                    >
                      <option value="all">All</option>
                      <option value="any">Any</option>
                    </select>
                  </label>
                  <label className="block text-sm">
                    <span className="text-gray-700">Merge Strategy</span>
                    <select
                      value={selectedNode.merge_config?.merge_strategy || 'union'}
                      onChange={(e) =>
                        onNodeUpdate({
                          merge_config: {
                            ...(selectedNode.merge_config || { merge_strategy: 'union' }),
                            merge_strategy: e.target.value as 'union' | 'intersection' | 'first',
                          },
                        })
                      }
                      className="mt-1 block w-full rounded border-gray-300"
                    >
                      <option value="union">Union</option>
                      <option value="intersection">Intersection</option>
                      <option value="first">First</option>
                    </select>
                  </label>
                </div>
              )}

              {selectedNode.type === 'parallel' && (
                <div className="space-y-3">
                  <label className="block text-sm">
                    <span className="text-gray-700">Branches</span>
                    <select
                      multiple
                      value={parallelBranches}
                      onChange={(e) => {
                        const selected = Array.from(e.target.selectedOptions).map(
                          (opt) => opt.value
                        );
                        onNodeUpdate({
                          parallel_config: {
                            ...(selectedNode.parallel_config || { wait_for: 'all' }),
                            branches: selected,
                          },
                        });
                      }}
                      className="mt-1 block w-full rounded border-gray-300"
                    >
                      {workflow.nodes
                        .filter((n) => n.id !== selectedNode.id)
                        .map((n) => (
                          <option key={n.id} value={n.id}>
                            {n.label || n.id}
                          </option>
                        ))}
                    </select>
                    <div className="text-xs text-gray-500">Hold Cmd/Ctrl to select multiple.</div>
                  </label>
                  <label className="block text-sm">
                    <span className="text-gray-700">Wait For</span>
                    <select
                      value={selectedNode.parallel_config?.wait_for || 'all'}
                      onChange={(e) =>
                        onNodeUpdate({
                          parallel_config: {
                            ...(selectedNode.parallel_config || { wait_for: 'all' }),
                            wait_for: e.target.value as 'all' | 'any' | 'first',
                          },
                        })
                      }
                      className="mt-1 block w-full rounded border-gray-300"
                    >
                      <option value="all">All</option>
                      <option value="any">Any</option>
                      <option value="first">First</option>
                    </select>
                  </label>
                </div>
              )}

              {selectedNode.type === 'subgraph' && (
                <div className="space-y-3">
                  <label className="block text-sm">
                    <span className="text-gray-700">Workflow Name</span>
                    <input
                      type="text"
                      value={selectedNode.subgraph_config?.workflow_name || ''}
                      onChange={(e) =>
                        onNodeUpdate({
                          subgraph_config: {
                            ...(selectedNode.subgraph_config || { workflow_name: '' }),
                            workflow_name: e.target.value,
                          },
                        })
                      }
                      className="mt-1 block w-full rounded border-gray-300"
                      placeholder="workflow-id"
                    />
                  </label>
                  <label className="block text-sm">
                    <span className="text-gray-700">Input Mapping (JSON)</span>
                    <textarea
                      value={inputMappingText}
                      onChange={(e) => setInputMappingText(e.target.value)}
                      onBlur={(e) => handleJsonBlur(e.target.value, 'input_mapping')}
                      className="mt-1 block w-full rounded border-gray-300 font-mono text-xs"
                      rows={4}
                      placeholder='{"field":"source"}'
                    />
                    {nodeFieldErrors.input_mapping && (
                      <div className="text-xs text-red-600 mt-1">
                        {nodeFieldErrors.input_mapping}
                      </div>
                    )}
                  </label>
                  <label className="block text-sm">
                    <span className="text-gray-700">Output Mapping (JSON)</span>
                    <textarea
                      value={outputMappingText}
                      onChange={(e) => setOutputMappingText(e.target.value)}
                      onBlur={(e) => handleJsonBlur(e.target.value, 'output_mapping')}
                      className="mt-1 block w-full rounded border-gray-300 font-mono text-xs"
                      rows={4}
                      placeholder='{"result":"target"}'
                    />
                    {nodeFieldErrors.output_mapping && (
                      <div className="text-xs text-red-600 mt-1">
                        {nodeFieldErrors.output_mapping}
                      </div>
                    )}
                  </label>
                </div>
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
                    {workflow.nodes.map((node) => (
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
                    {workflow.nodes.map((node) => (
                      <option key={node.id} value={node.id}>
                        {node.label || node.id}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => {
                      if (edgeFrom && edgeTo && edgeFrom !== edgeTo) {
                        onEdgeAdd(edgeFrom, edgeTo);
                      }
                    }}
                    className="w-full px-3 py-1.5 text-sm font-medium text-emerald-700 bg-emerald-50 rounded hover:bg-emerald-100"
                  >
                    Add Edge
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="pt-4 border-t">
          <button
            type="button"
            onClick={() => setRunInputsOpen((prev) => !prev)}
            className="w-full flex items-center justify-between text-xs font-semibold text-gray-600"
          >
            <span>Run Inputs</span>
            <span>{runInputsOpen ? '-' : '+'}</span>
          </button>
          {runInputsOpen && (
            <div className="mt-3 space-y-2">
              <label className="block">
                <span className="text-xs text-gray-500">Goal</span>
                <input
                  id="run-goal"
                  name="run-goal"
                  value={runGoal}
                  onChange={(e) => onRunGoalChange(e.target.value)}
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
                  onChange={(e) => onRunLabelChange(e.target.value)}
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
                  onChange={(e) => onRunInputsChange(e.target.value)}
                  onBlur={handleRunInputsBlur}
                  className="mt-1 w-full rounded border border-gray-200 px-2 py-1 text-sm font-mono"
                  rows={4}
                  placeholder='{"key":"value"}'
                />
                <div className="mt-1 text-[11px] text-gray-400">
                  Must be a JSON object (not an array).
                </div>
                {runInputError && (
                  <div className="mt-1 text-xs text-red-600">{runInputError}</div>
                )}
              </label>
            </div>
          )}
        </div>
      </div>
    );
  }
);
