/**
 * ExecutionMonitor component for live execution tracking.
 */

import { useMemo, useCallback, useState, useEffect } from 'react';
import type {
  WorkflowGraph,
  NodeStatus,
  ExecutionStatus,
  NodeExecutionStatus,
  TraceEvent,
} from '../types/workflow';
import {
  useExecution,
  useExecutionNodes,
  useCancelExecution,
} from '../hooks/useExecutions';
import { useExecutionWebSocket } from '../hooks/useExecutionWebSocket';
import { WorkflowCanvas } from './WorkflowCanvas';
import { TraceTimeline } from './TraceTimeline';
import { StateInspector } from './StateInspector';

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
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);

  useEffect(() => {
    setTraceEvents([]);
  }, [executionId]);

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
    onInitialState: () => setTraceEvents([]),
    onNodeUpdate: (nodeId, status, _output, timestamp) => {
      const nodeInfo = workflow.nodes.find((node) => node.id === nodeId);
      const eventId = `${nodeId}-${timestamp}`;
      setTraceEvents((prev) => {
        if (prev.some((event) => event.id === eventId)) return prev;
        const next: TraceEvent = {
          id: eventId,
          timestamp,
          nodeId,
          nodeLabel: nodeInfo?.label || nodeId,
          nodeType: nodeInfo?.type || 'task',
          status,
        };
        const updated = [...prev, next];
        updated.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
        return updated;
      });
    },
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

  const globalState = useMemo(() => {
    const state: Record<string, unknown> = {};
    for (const [nodeId, nodeStatus] of nodeOutputs) {
      if (nodeStatus.status === 'completed' && nodeStatus.output) {
        state[nodeId] = nodeStatus.output;
      }
    }
    return state;
  }, [nodeOutputs]);

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

      <div className="flex-1 min-h-0 flex">
        <div className="w-64 border-r bg-white">
          <TraceTimeline
            events={traceEvents}
            selectedNodeId={selectedNodeId}
            onNodeSelect={setSelectedNodeId}
            isLive={execution.status === 'running'}
          />
        </div>
        <div className="flex-1 min-h-0 flex flex-col">
          <div className="flex-1 min-h-0">
            <WorkflowCanvas
              workflow={workflow}
              nodeStatuses={nodeStatuses}
              readOnly
              onNodeSelect={setSelectedNodeId}
            />
          </div>
          <div className="h-56">
            <StateInspector
              workflow={workflow}
              nodeOutputs={nodeOutputs}
              selectedNodeId={selectedNodeId}
              globalState={globalState}
            />
          </div>
        </div>
      </div>

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
