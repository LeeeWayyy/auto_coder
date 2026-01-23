/**
 * ExecutionMonitor component for live execution tracking.
 */

import { useMemo, useCallback, useState, useEffect, useRef } from 'react';
import type {
  WorkflowGraph,
  NodeStatus,
  ExecutionStatus,
  NodeExecutionStatus,
  TraceEvent,
  ExecutionEvent,
  NodeType,
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
import { ApprovalBanner } from './ApprovalBanner';
import { ApprovalModal } from './ApprovalModal';
import { respondToHumanNode, createExecutionEventStream, getExecutionHistory } from '../api/client';

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
  const [historyMode, setHistoryMode] = useState(false);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [historyEvents, setHistoryEvents] = useState<ExecutionEvent[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyHasMore, setHistoryHasMore] = useState(true);
  const [historyJump, setHistoryJump] = useState('');
  const [streamedOutputs, setStreamedOutputs] = useState(
    new Map<string, Record<string, unknown>>()
  );
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
    setStreamedOutputs(new Map());
  }, [executionId]);

  useEffect(() => {
    setHistoryEvents([]);
    setHistoryHasMore(true);
    setHistoryLoading(false);
    setHistoryIndex(0);
  }, [executionId]);

  const HISTORY_PAGE_SIZE = 500;

  const loadHistoryPage = useCallback(
    async (reset = false) => {
      if (historyLoading) return;
      setHistoryLoading(true);
      try {
        const sinceId = reset
          ? undefined
          : historyEvents.length > 0
          ? historyEvents[historyEvents.length - 1].id
          : undefined;
        const data = await getExecutionHistory(executionId, sinceId, HISTORY_PAGE_SIZE);
        setHistoryEvents((prev) => (reset ? data : [...prev, ...data]));
        setHistoryHasMore(data.length === HISTORY_PAGE_SIZE);
      } catch {
        setHistoryHasMore(false);
      } finally {
        setHistoryLoading(false);
      }
    },
    [executionId, historyEvents, historyLoading]
  );

  useEffect(() => {
    if (historyMode && historyEvents.length === 0) {
      void loadHistoryPage(true);
    }
  }, [historyMode, historyEvents.length, loadHistoryPage]);

  // FIX (code review): Use ref to track last event ID so reconnections use the latest value
  // (state would capture stale value in closure)
  const lastEventIdRef = useRef<number>(0);

  // FIX (code review): Reset lastEventIdRef when executionId changes to avoid
  // skipping events for new executions (old ID would be larger than new execution's IDs)
  useEffect(() => {
    lastEventIdRef.current = 0;
  }, [executionId]);

  useEffect(() => {
    // FIX (code review): Only depend on executionId to avoid reconnecting on status changes.
    // The SSE connection is stable throughout the execution lifecycle. We handle terminal
    // status by closing inside the effect when we detect completion events.
    let source: EventSource | null = null;
    let closed = false;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

    const handleEvent = (event: MessageEvent) => {
      if (closed) return;
      try {
        const data = JSON.parse(event.data) as ExecutionEvent;

        // Track event ID in ref to avoid duplicates and enable proper reconnect
        if (data.id && Number(data.id) > lastEventIdRef.current) {
          lastEventIdRef.current = Number(data.id);
        }

        // Close stream on terminal status
        if (
          data.event_type === 'execution_complete' &&
          ['completed', 'failed', 'cancelled'].includes(String(data.status))
        ) {
          closed = true;
          source?.close();
          return;
        }

        if (data.event_type === 'node_update' && data.node_id) {
          // FIX (code review): Runtime type validation instead of unsafe type assertions
          const payload = data.payload;
          if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
            const output = (payload as Record<string, unknown>).output;
            if (output && typeof output === 'object' && !Array.isArray(output)) {
              const validOutput = output as Record<string, unknown>;
              if (Object.keys(validOutput).length > 0) {
                setStreamedOutputs((prev) => {
                  const next = new Map(prev);
                  next.set(data.node_id as string, validOutput);
                  // FIX (code review): Evict oldest entries if Map grows too large
                  // (defense in depth - normally bounded by workflow node count)
                  const MAX_STREAMED_NODES = 100;
                  if (next.size > MAX_STREAMED_NODES) {
                    const oldestKey = next.keys().next().value;
                    if (oldestKey) next.delete(oldestKey);
                  }
                  return next;
                });
              }
            }
          }
        }
        if (data.event_type === 'stream_chunk' && data.node_id) {
          // FIX (code review): Runtime type validation for stream chunks
          const payload = data.payload;
          if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
            const payloadObj = payload as Record<string, unknown>;
            const stream = String(payloadObj.stream || 'stdout');
            const chunk = String(payloadObj.chunk || '');
            if (!chunk) return;
            setStreamedOutputs((prev) => {
              const next = new Map(prev);
              const existing = next.get(data.node_id as string) || {};
              const existingObj =
                typeof existing === 'object' && !Array.isArray(existing)
                  ? (existing as Record<string, unknown>)
                  : {};
              const currentText = String(existingObj[stream] || '');
              const combined = (currentText + chunk).slice(-8000);
              next.set(data.node_id as string, { ...existingObj, [stream]: combined });
              // FIX (code review): Evict oldest entries if Map grows too large
              const MAX_STREAMED_NODES = 100;
              if (next.size > MAX_STREAMED_NODES) {
                const oldestKey = next.keys().next().value;
                if (oldestKey) next.delete(oldestKey);
              }
              return next;
            });
          }
        }
      } catch (error) {
        // FIX (code review): Log malformed events for debugging instead of silent swallow
        console.warn('Malformed execution event:', event.data, error);
      }
    };

    // FIX (code review): Create connection with manual reconnect using latest event ID
    const connect = () => {
      if (closed) return;
      // Use ref value so reconnect uses the latest event ID, not stale closure value
      const sinceId = lastEventIdRef.current > 0 ? lastEventIdRef.current : undefined;
      source = createExecutionEventStream(executionId, sinceId);
      source.addEventListener('execution_event', handleEvent);
      // FIX (code review): Handle errors with manual reconnect using latest event ID
      source.onerror = () => {
        if (closed) return;
        source?.close();
        // Reconnect after delay, using ref to get latest event ID
        reconnectTimeout = setTimeout(connect, 2000);
      };
    };

    connect();

    return () => {
      closed = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      source?.removeEventListener('execution_event', handleEvent);
      source?.close();
    };
  }, [executionId]); // FIX (code review): Removed execution?.status to avoid reconnects

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
    // FIX (code review): Don't clear trace events on initial_state.
    // This can race with node_update events that arrive before initial_state,
    // causing lost timeline entries. We already clear on executionId change.
    onInitialState: () => {
      // No-op: trace events are cleared when executionId changes
    },
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
  const nodeLabelById = useMemo(() => {
    return new Map(workflow.nodes.map((node) => [node.id, node.label || node.id]));
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
  const streamedOutput =
    !historyMode && selectedNodeId ? streamedOutputs.get(selectedNodeId) || null : null;
  const createdFiles = useMemo(() => {
    const seen = new Set<string>();
    const items: Array<{ path: string; nodeId: string; nodeLabel: string }> = [];
    for (const [nodeId, nodeStatus] of viewNodeOutputs) {
      const output = nodeStatus.output as Record<string, unknown> | null;
      const files = Array.isArray(output?.files_created) ? output?.files_created : [];
      for (const file of files) {
        if (typeof file !== 'string' || seen.has(file)) continue;
        seen.add(file);
        items.push({
          path: file,
          nodeId,
          nodeLabel: nodeLabelById.get(nodeId) || nodeId,
        });
      }
    }
    return items;
  }, [viewNodeOutputs, nodeLabelById]);

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
        // FIX (code review): Log error for debugging instead of silent swallow
        console.error('Human action failed:', error);
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
          {/* FIX (code review): Allow cancel from both running and interrupted states */}
          {(execution.status === 'running' || execution.status === 'interrupted') && (
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
            {historyMode && (
              <div className="space-y-2">
                <div className="text-[10px] text-gray-500">
                  Loaded {historyTraceEvents.length} events
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => loadHistoryPage(false)}
                    disabled={!historyHasMore || historyLoading}
                    className="px-2 py-1 text-[10px] border rounded text-gray-600 disabled:opacity-50"
                  >
                    {historyLoading ? 'Loading...' : historyHasMore ? 'Load more' : 'All loaded'}
                  </button>
                  <input
                    type="text"
                    value={historyJump}
                    onChange={(e) => setHistoryJump(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key !== 'Enter') return;
                      const value = historyJump.trim();
                      if (!value) return;
                      let targetIndex = -1;
                      if (value.startsWith('id:')) {
                        const id = Number(value.replace('id:', '').trim());
                        const found = historyTraceEvents.findIndex(
                          (event) => Number(event.id) === id
                        );
                        targetIndex = found;
                      } else {
                        const idx = Number(value);
                        if (!Number.isNaN(idx)) {
                          targetIndex = idx - 1;
                        }
                      }
                      if (targetIndex >= 0 && targetIndex < historyTraceEvents.length) {
                        setHistoryIndex(targetIndex);
                      }
                    }}
                    className="px-2 py-1 text-[10px] border rounded w-28"
                    placeholder="Jump # or id:123"
                  />
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
          <div className="h-72 flex flex-col">
            <ExecutionFilesPanel files={createdFiles} />
            <div className="flex-1 min-h-0">
              <StateInspector
                workflow={workflow}
                nodeOutputs={viewNodeOutputs}
                selectedNodeId={selectedNodeId}
                globalState={globalState}
                streamedOutput={streamedOutput}
              />
            </div>
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

interface ExecutionFilesPanelProps {
  files: Array<{ path: string; nodeId: string; nodeLabel: string }>;
}

function ExecutionFilesPanel({ files }: ExecutionFilesPanelProps) {
  return (
    <div className="border-t bg-white">
      <div className="flex items-center justify-between px-4 py-2 border-b">
        <div className="text-xs font-semibold text-gray-600">Files created</div>
        <div className="text-[10px] text-gray-500">{files.length}</div>
      </div>
      <div className="px-4 py-2 max-h-28 overflow-auto">
        {files.length === 0 ? (
          <div className="text-xs text-gray-400">No files created yet.</div>
        ) : (
          <div className="space-y-1">
            {files.map((file) => (
              <div key={file.path} className="flex items-center gap-2 text-xs">
                <a
                  href={`/api/files?path=${encodeURIComponent(file.path)}`}
                  target="_blank"
                  rel="noreferrer"
                  className="min-w-0 flex-1 truncate text-blue-600 hover:underline"
                  title={file.path}
                >
                  {file.path}
                </a>
                <span className="text-[10px] text-gray-400 whitespace-nowrap">
                  {file.nodeLabel}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
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
