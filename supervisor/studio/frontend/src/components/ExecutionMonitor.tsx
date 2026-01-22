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
  NodeType,
} from '../types/workflow';
import {
  useExecution,
  useExecutionNodes,
  useCancelExecution,
  useExecutionHistory,
} from '../hooks/useExecutions';
import { useExecutionWebSocket } from '../hooks/useExecutionWebSocket';
import { WorkflowCanvas } from './WorkflowCanvas';
import { TraceTimeline } from './TraceTimeline';
import { StateInspector } from './StateInspector';
import { ApprovalBanner } from './ApprovalBanner';
import { ApprovalModal } from './ApprovalModal';
import { respondToHumanNode } from '../api/client';

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
  const { data: historyEvents } = useExecutionHistory(executionId, 2000);
  const cancelMutation = useCancelExecution();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);
  const [historyMode, setHistoryMode] = useState(false);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [humanWaiting, setHumanWaiting] = useState<{
    nodeId: string;
    title: string;
    description?: string;
    currentOutput?: Record<string, unknown>;
  } | null>(null);
  const [approvalOpen, setApprovalOpen] = useState(false);
  const [approvalSubmitting, setApprovalSubmitting] = useState(false);
  const [approvalError, setApprovalError] = useState<string | null>(null);

  useEffect(() => {
    setTraceEvents([]);
  }, [executionId]);

  useEffect(() => {
    if (execution?.status !== 'interrupted') {
      setHumanWaiting(null);
      setApprovalOpen(false);
      setApprovalSubmitting(false);
    }
  }, [execution?.status]);

  // Handle WebSocket updates
  const handleComplete = useCallback(
    (status: ExecutionStatus) => {
      onComplete?.(status);
    },
    [onComplete]
  );

  const { isConnected } = useExecutionWebSocket(executionId, {
    enabled: execution?.status === 'running' || execution?.status === 'interrupted',
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
    onHumanWaiting: (nodeId, title, description, currentOutput) => {
      setHumanWaiting({ nodeId, title, description, currentOutput });
      setSelectedNodeId(nodeId);
    },
    onHumanResolved: () => {
      setHumanWaiting(null);
      setApprovalOpen(false);
      setApprovalSubmitting(false);
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

  const nodeTypeById = useMemo(() => {
    return new Map(workflow.nodes.map((node) => [node.id, node.type]));
  }, [workflow.nodes]);

  const historyTraceEvents = useMemo(() => {
    if (!historyEvents) return [];
    return historyEvents
      .filter((event) => event.event_type === 'node_update' && event.node_id)
      .map((event) => ({
        id: String(event.id),
        timestamp: event.timestamp,
        nodeId: event.node_id as string,
        nodeLabel:
          workflow.nodes.find((node) => node.id === event.node_id)?.label ||
          (event.node_id as string),
        nodeType:
          (event.node_type as NodeType | null) ||
          nodeTypeById.get(event.node_id as string) ||
          'task',
        status: (event.status as NodeStatus) || 'pending',
      }));
  }, [historyEvents, workflow.nodes, nodeTypeById]);

  useEffect(() => {
    if (historyMode && historyTraceEvents.length > 0) {
      setHistoryIndex(historyTraceEvents.length - 1);
    }
  }, [historyMode, historyTraceEvents.length]);

  const snapshotFromHistory = useMemo(() => {
    if (!historyMode || historyTraceEvents.length === 0) {
      return null;
    }
    const idx = Math.min(Math.max(historyIndex, 0), historyTraceEvents.length - 1);
    const statuses: Record<string, NodeStatus> = {};
    const outputs = new Map<string, NodeExecutionStatus>();
    for (let i = 0; i <= idx; i += 1) {
      const event = historyTraceEvents[i];
      statuses[event.nodeId] = event.status;
      const raw = historyEvents?.find((ev) => String(ev.id) === event.id);
      const payload = raw?.payload as Record<string, unknown> | undefined;
      const output = payload?.output as Record<string, unknown> | undefined;
      outputs.set(event.nodeId, {
        node_id: event.nodeId,
        node_type: (nodeTypeById.get(event.nodeId) ?? 'task') as NodeExecutionStatus['node_type'],
        status: event.status,
        output: output ?? null,
        error: (payload?.error as string | undefined) ?? null,
        version: raw?.version ?? 0,
      });
    }
    return { statuses, outputs };
  }, [historyMode, historyIndex, historyTraceEvents, historyEvents, nodeTypeById]);

  const globalState = useMemo(() => {
    const state: Record<string, unknown> = {};
    const source = historyMode && snapshotFromHistory ? snapshotFromHistory.outputs : nodeOutputs;
    for (const [nodeId, nodeStatus] of source) {
      if (nodeStatus.status === 'completed' && nodeStatus.output) {
        state[nodeId] = nodeStatus.output;
      }
    }
    return state;
  }, [nodeOutputs, historyMode, snapshotFromHistory]);

  const activeNodeIds = useMemo(() => {
    const active = new Set<string>();
    const statuses = historyMode && snapshotFromHistory ? snapshotFromHistory.statuses : nodeStatuses;
    for (const [nodeId, status] of Object.entries(statuses)) {
      if (status === 'running') {
        active.add(nodeId);
      }
    }
    return active;
  }, [nodeStatuses, historyMode, snapshotFromHistory]);

  const viewNodeStatuses = historyMode && snapshotFromHistory ? snapshotFromHistory.statuses : nodeStatuses;
  const viewNodeOutputs = historyMode && snapshotFromHistory ? snapshotFromHistory.outputs : nodeOutputs;
  const viewTraceEvents = historyMode ? historyTraceEvents : traceEvents;

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
    interrupted: 'text-amber-600',
  };

  const handleHumanAction = useCallback(
    async (action: 'approve' | 'reject' | 'edit', payload?: { feedback?: string; editedData?: Record<string, unknown> }) => {
      if (!humanWaiting) return;
      setApprovalSubmitting(true);
      setApprovalError(null);
      try {
        await respondToHumanNode({
          executionId,
          nodeId: humanWaiting.nodeId,
          action,
          feedback: payload?.feedback,
          editedData: payload?.editedData,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Submission failed';
        setApprovalError(message);
        setApprovalSubmitting(false);
      }
    },
    [executionId, humanWaiting]
  );

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

      {execution.status === 'interrupted' && humanWaiting && (
        <ApprovalBanner
          nodeId={humanWaiting.nodeId}
          title={humanWaiting.title}
          description={humanWaiting.description}
          currentOutput={humanWaiting.currentOutput}
          onApprove={() => handleHumanAction('approve')}
          onReject={(reason) => handleHumanAction('reject', { feedback: reason })}
          onEdit={() => setApprovalOpen(true)}
          isSubmitting={approvalSubmitting}
        />
      )}
      {approvalError && (
        <div className="px-4 py-2 text-xs text-red-600 bg-red-50 border-b border-red-200">
          {approvalError}
        </div>
      )}

      <div className="flex-1 min-h-0 flex">
        <div className="w-64 border-r bg-white">
          <div className="px-3 py-2 border-b space-y-2">
            <div className="flex items-center gap-2 text-xs">
              <button
                type="button"
                onClick={() => setHistoryMode(false)}
                className={`px-2 py-1 rounded border ${
                  historyMode ? 'border-gray-200 text-gray-500' : 'border-emerald-300 text-emerald-700 bg-emerald-50'
                }`}
              >
                Live
              </button>
              <button
                type="button"
                onClick={() => setHistoryMode(true)}
                className={`px-2 py-1 rounded border ${
                  historyMode ? 'border-blue-300 text-blue-700 bg-blue-50' : 'border-gray-200 text-gray-500'
                }`}
                disabled={!historyTraceEvents.length}
              >
                History
              </button>
            </div>
            {historyMode && historyTraceEvents.length > 0 && (
              <div className="space-y-1">
                <input
                  type="range"
                  min={0}
                  max={historyTraceEvents.length - 1}
                  value={Math.min(historyIndex, historyTraceEvents.length - 1)}
                  onChange={(e) => setHistoryIndex(Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-[10px] text-gray-500">
                  Event {Math.min(historyIndex, historyTraceEvents.length - 1) + 1} / {historyTraceEvents.length}
                </div>
              </div>
            )}
          </div>
          <TraceTimeline
            events={viewTraceEvents}
            selectedNodeId={selectedNodeId}
            onNodeSelect={setSelectedNodeId}
            isLive={!historyMode && execution.status === 'running'}
          />
        </div>
        <div className="flex-1 min-h-0 flex flex-col">
          <div className="flex-1 min-h-0">
            <WorkflowCanvas
              workflow={workflow}
              nodeStatuses={viewNodeStatuses}
              activeNodeIds={activeNodeIds}
              readOnly
              onNodeSelect={setSelectedNodeId}
            />
          </div>
          <div className="h-56">
            <StateInspector
              workflow={workflow}
              nodeOutputs={viewNodeOutputs}
              selectedNodeId={selectedNodeId}
              globalState={globalState}
            />
          </div>
        </div>
      </div>

      {/* Node status summary */}
      <NodeStatusSummary nodes={nodes || []} />

      {humanWaiting && (
        <ApprovalModal
          isOpen={approvalOpen}
          onClose={() => setApprovalOpen(false)}
          currentOutput={humanWaiting.currentOutput || {}}
          onSubmit={(editedData) => handleHumanAction('edit', { editedData })}
          isSubmitting={approvalSubmitting}
        />
      )}
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
