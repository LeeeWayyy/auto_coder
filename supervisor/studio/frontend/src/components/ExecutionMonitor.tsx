/**
 * ExecutionMonitor component for live execution tracking.
 */

import { useMemo, useCallback, useState } from 'react';
import type {
  WorkflowGraph,
  NodeStatus,
  ExecutionStatus,
  NodeExecutionStatus,
} from '../types/workflow';
import {
  useExecution,
  useExecutionNodes,
  useCancelExecution,
} from '../hooks/useExecutions';
import { useExecutionWebSocket } from '../hooks/useExecutionWebSocket';
import { WorkflowCanvas } from './WorkflowCanvas';

interface ExecutionMonitorProps {
  executionId: string;
  workflow: WorkflowGraph;
  onComplete?: (status: ExecutionStatus) => void;
}

export function ExecutionMonitor({
  executionId,
  workflow,
  onComplete,
}: ExecutionMonitorProps) {
  // Fetch execution and node data
  const { data: execution, isLoading: execLoading } = useExecution(executionId);
  const { data: nodes } = useExecutionNodes(executionId);
  const cancelMutation = useCancelExecution();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Handle WebSocket updates
  const handleComplete = useCallback(
    (status: ExecutionStatus) => {
      onComplete?.(status);
    },
    [onComplete]
  );

  const { isConnected } = useExecutionWebSocket(executionId, {
    enabled: execution?.status === 'running',
    onExecutionComplete: handleComplete,
  });

  // Build node status map from execution nodes
  const nodeStatuses = useMemo<Record<string, NodeStatus>>(() => {
    if (!nodes) return {};
    return nodes.reduce(
      (acc, node) => {
        acc[node.node_id] = node.status;
        return acc;
      },
      {} as Record<string, NodeStatus>
    );
  }, [nodes]);

  const nodeOutputs = useMemo(() => {
    const map = new Map<string, NodeExecutionStatus>();
    for (const node of nodes || []) {
      map.set(node.node_id, node);
    }
    return map;
  }, [nodes]);

  // Handle cancel button
  const handleCancel = useCallback(() => {
    cancelMutation.mutate(executionId);
  }, [executionId, cancelMutation]);

  if (execLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading execution...</div>
      </div>
    );
  }

  if (!execution) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-500">Execution not found</div>
      </div>
    );
  }

  const statusColors: Record<ExecutionStatus, string> = {
    running: 'text-amber-600',
    completed: 'text-emerald-600',
    failed: 'text-red-600',
    cancelled: 'text-gray-600',
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b bg-white">
        <div>
          <h2 className="text-lg font-semibold">Execution: {executionId.slice(0, 8)}</h2>
          <div className="flex items-center gap-4 text-sm text-gray-600">
            <span>
              Status:{' '}
              <span className={`font-medium ${statusColors[execution.status]}`}>
                {execution.status}
              </span>
            </span>
            <span>
              Started:{' '}
              {execution.started_at
                ? new Date(execution.started_at).toLocaleTimeString()
                : '—'}
            </span>
            {execution.completed_at && (
              <span>
                Completed: {new Date(execution.completed_at).toLocaleTimeString()}
              </span>
            )}
            {isConnected && (
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                Live
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {execution.status === 'running' && (
            <button
              onClick={handleCancel}
              disabled={cancelMutation.isPending}
              className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-md hover:bg-red-700 disabled:opacity-50"
            >
              {cancelMutation.isPending ? 'Cancelling...' : 'Cancel'}
            </button>
          )}
        </div>
      </div>

      {/* Error display */}
      {execution.error && (
        <div className="p-4 bg-red-50 border-b border-red-200">
          <div className="text-sm text-red-700">
            <span className="font-medium">Error:</span> {execution.error}
          </div>
        </div>
      )}

      {/* Canvas - read-only during execution */}
      <div className="flex-1 min-h-0">
        <WorkflowCanvas
          workflow={workflow}
          nodeStatuses={nodeStatuses}
          readOnly
          onNodeSelect={setSelectedNodeId}
        />
      </div>

      <NodeOutputPanel
        workflow={workflow}
        nodeStatuses={nodeStatuses}
        nodeOutputs={nodeOutputs}
        selectedNodeId={selectedNodeId}
        onSelectNode={setSelectedNodeId}
      />

      {/* Node status summary */}
      <NodeStatusSummary nodes={nodes || []} />
    </div>
  );
}

interface NodeStatusSummaryProps {
  nodes: NodeExecutionStatus[];
}

function NodeStatusSummary({ nodes }: NodeStatusSummaryProps) {
  const counts = useMemo(() => {
    const result: Record<NodeStatus, number> = {
      pending: 0,
      ready: 0,
      running: 0,
      completed: 0,
      failed: 0,
      skipped: 0,
    };
    for (const node of nodes) {
      result[node.status]++;
    }
    return result;
  }, [nodes]);

  const statusColors: Record<NodeStatus, string> = {
    pending: 'bg-gray-200 text-gray-700',
    ready: 'bg-blue-100 text-blue-700',
    running: 'bg-amber-100 text-amber-700',
    completed: 'bg-emerald-100 text-emerald-700',
    failed: 'bg-red-100 text-red-700',
    skipped: 'bg-gray-100 text-gray-600',
  };

  return (
    <div className="flex items-center gap-4 p-4 border-t bg-gray-50">
      <span className="text-sm font-medium text-gray-600">Nodes:</span>
      {(Object.entries(counts) as [NodeStatus, number][])
        .filter(([, count]) => count > 0)
        .map(([status, count]) => (
          <span
            key={status}
            className={`px-2 py-1 text-xs font-medium rounded ${statusColors[status]}`}
          >
            {status}: {count}
          </span>
        ))}
    </div>
  );
}

interface NodeOutputPanelProps {
  workflow: WorkflowGraph;
  nodeStatuses: Record<string, NodeStatus>;
  nodeOutputs: Map<string, NodeExecutionStatus>;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
}

function NodeOutputPanel({
  workflow,
  nodeStatuses,
  nodeOutputs,
  selectedNodeId,
  onSelectNode,
}: NodeOutputPanelProps) {
  const nodeMap = useMemo(() => {
    return new Map(workflow.nodes.map((n) => [n.id, n]));
  }, [workflow.nodes]);

  const selectedMeta = selectedNodeId ? nodeMap.get(selectedNodeId) : null;
  const selectedOutput = selectedNodeId ? nodeOutputs.get(selectedNodeId) : null;
  const selectedStatus = selectedNodeId
    ? nodeStatuses[selectedNodeId] || 'pending'
    : null;

  return (
    <div className="border-t bg-white">
      <div className="flex items-center justify-between px-4 py-2 border-b">
        <div className="text-sm font-medium text-gray-700">Node Output</div>
        {selectedNodeId && (
          <div className="text-xs text-gray-500">
            {selectedNodeId} • {selectedStatus}
          </div>
        )}
      </div>
      <div className="flex h-56">
        <div className="w-56 border-r overflow-y-auto">
          {workflow.nodes.length === 0 ? (
            <div className="p-3 text-xs text-gray-400">No nodes yet.</div>
          ) : (
            workflow.nodes.map((node) => {
              const label = node.label || node.id;
              const isActive = node.id === selectedNodeId;
              const status = nodeStatuses[node.id] || 'pending';
              return (
                <button
                  key={node.id}
                  type="button"
                  onClick={() => onSelectNode(node.id)}
                  className={`w-full text-left px-3 py-2 text-xs border-b hover:bg-gray-50 ${
                    isActive ? 'bg-gray-100 font-medium' : 'text-gray-600'
                  }`}
                >
                  <div className="truncate">{label}</div>
                  <div className="text-[10px] text-gray-400">{status}</div>
                </button>
              );
            })
          )}
        </div>
        <div className="flex-1 p-3 overflow-y-auto">
          {!selectedNodeId ? (
            <div className="text-xs text-gray-400">Select a node to view output.</div>
          ) : (
            <div className="space-y-3 text-xs">
              <div>
                <div className="text-gray-500">Label</div>
                <div className="font-mono text-gray-800">
                  {selectedMeta?.label || selectedNodeId}
                </div>
              </div>
              {selectedOutput?.error && (
                <div>
                  <div className="text-red-600">Error</div>
                  <pre className="whitespace-pre-wrap text-red-700">
                    {selectedOutput.error}
                  </pre>
                </div>
              )}
              <div>
                <div className="text-gray-500">Output</div>
                <pre className="whitespace-pre-wrap text-gray-800">
                  {selectedOutput?.output
                    ? JSON.stringify(selectedOutput.output, null, 2)
                    : '—'}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
