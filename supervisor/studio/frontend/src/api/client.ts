/**
 * API client for Supervisor Studio backend.
 * All functions return Promises for use with React Query.
 */

import type {
  WorkflowGraph,
  ExecutionResponse,
  NodeExecutionStatus,
  ExecutionEvent,
} from '../types/workflow';

const API_BASE = '/api';

/**
 * Custom error class for API errors with status code and details.
 */
export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string | Record<string, unknown>
  ) {
    super(typeof detail === 'string' ? detail : JSON.stringify(detail));
    this.name = 'ApiError';
  }
}

/**
 * Fetch wrapper with error handling.
 */
async function fetchApi<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let detail: string | Record<string, unknown>;
    try {
      const json = await response.json();
      detail = json.detail || json;
    } catch {
      detail = response.statusText;
    }
    throw new ApiError(response.status, detail);
  }

  return response.json();
}

// ========== Workflow CRUD ==========

export async function listWorkflows(): Promise<WorkflowGraph[]> {
  return fetchApi<WorkflowGraph[]>('/workflows');
}

export async function getWorkflow(graphId: string): Promise<WorkflowGraph> {
  return fetchApi<WorkflowGraph>(`/workflows/${encodeURIComponent(graphId)}`);
}

export async function createWorkflow(
  workflow: WorkflowGraph
): Promise<WorkflowGraph> {
  return fetchApi<WorkflowGraph>('/workflows', {
    method: 'POST',
    body: JSON.stringify({ graph: workflow }),
  });
}

export async function updateWorkflow(
  graphId: string,
  workflow: WorkflowGraph
): Promise<WorkflowGraph> {
  return fetchApi<WorkflowGraph>(`/workflows/${encodeURIComponent(graphId)}`, {
    method: 'PUT',
    body: JSON.stringify({ graph: workflow }),
  });
}

export async function deleteWorkflow(
  graphId: string
): Promise<{ status: string; id: string }> {
  return fetchApi<{ status: string; id: string }>(
    `/workflows/${encodeURIComponent(graphId)}`,
    { method: 'DELETE' }
  );
}

// ========== Execution Operations ==========

export interface ExecuteWorkflowParams {
  graph_id: string;
  workflow_id?: string;
  input_data?: Record<string, unknown>;
}

export async function executeWorkflow(
  params: ExecuteWorkflowParams
): Promise<ExecutionResponse> {
  return fetchApi<ExecutionResponse>('/execute', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function cancelExecution(
  executionId: string
): Promise<{ status: string; execution_id: string }> {
  return fetchApi<{ status: string; execution_id: string }>(
    `/executions/${encodeURIComponent(executionId)}/cancel`,
    { method: 'POST' }
  );
}

export interface HumanResponseParams {
  executionId: string;
  nodeId: string;
  action: 'approve' | 'reject' | 'edit';
  feedback?: string;
  editedData?: Record<string, unknown>;
}

export async function respondToHumanNode(
  params: HumanResponseParams
): Promise<{ status: string }> {
  return fetchApi<{ status: string }>(
    `/executions/${encodeURIComponent(params.executionId)}/respond`,
    {
      method: 'POST',
      body: JSON.stringify({
        node_id: params.nodeId,
        action: params.action,
        feedback: params.feedback,
        edited_data: params.editedData,
      }),
    }
  );
}

export interface ListExecutionsParams {
  workflow_id?: string;
  source_graph_id?: string;
  limit?: number;
}

export async function listExecutions(
  params?: ListExecutionsParams
): Promise<ExecutionResponse[]> {
  const searchParams = new URLSearchParams();
  if (params?.workflow_id) {
    searchParams.set('workflow_id', params.workflow_id);
  }
  if (params?.source_graph_id) {
    searchParams.set('source_graph_id', params.source_graph_id);
  }
  if (params?.limit) {
    searchParams.set('limit', String(params.limit));
  }

  const queryString = searchParams.toString();
  const url = queryString ? `/executions?${queryString}` : '/executions';
  return fetchApi<ExecutionResponse[]>(url);
}

export async function getExecution(
  executionId: string
): Promise<ExecutionResponse> {
  return fetchApi<ExecutionResponse>(
    `/executions/${encodeURIComponent(executionId)}`
  );
}

export async function getExecutionNodes(
  executionId: string,
  sinceVersion?: number
): Promise<NodeExecutionStatus[]> {
  const url = sinceVersion !== undefined
    ? `/executions/${encodeURIComponent(executionId)}/nodes?since_version=${sinceVersion}`
    : `/executions/${encodeURIComponent(executionId)}/nodes`;
  return fetchApi<NodeExecutionStatus[]>(url);
}

export async function getExecutionHistory(
  executionId: string,
  sinceId?: number,
  limit?: number
): Promise<ExecutionEvent[]> {
  const searchParams = new URLSearchParams();
  if (sinceId !== undefined) {
    searchParams.set('since_id', String(sinceId));
  }
  if (limit !== undefined) {
    searchParams.set('limit', String(limit));
  }
  const query = searchParams.toString();
  const url = query
    ? `/executions/${encodeURIComponent(executionId)}/history?${query}`
    : `/executions/${encodeURIComponent(executionId)}/history`;
  return fetchApi<ExecutionEvent[]>(url);
}

// ========== WebSocket Connection ==========

/**
 * Create a WebSocket connection for execution updates.
 *
 * @param executionId - The execution ID to monitor
 * @param onMessage - Callback for incoming messages
 * @param onError - Callback for errors
 * @param onClose - Callback when connection closes
 * @returns WebSocket instance
 */
export function createExecutionWebSocket(
  executionId: string,
  onMessage: (data: unknown) => void,
  onError?: (error: Event) => void,
  onClose?: (event: CloseEvent) => void
): WebSocket {
  // Use relative WebSocket URL (works with Vite proxy)
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/executions/${encodeURIComponent(
    executionId
  )}`;

  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      console.error('Failed to parse WebSocket message:', event.data);
    }
  };

  // Setup ping/pong for keep-alive with proper cleanup
  let pingInterval: ReturnType<typeof setInterval> | null = null;

  const clearPingInterval = () => {
    if (pingInterval) {
      clearInterval(pingInterval);
      pingInterval = null;
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    clearPingInterval();
    onError?.(error);
  };

  ws.onclose = (event) => {
    if (event.code !== 1000) {
      console.warn('WebSocket closed abnormally:', event.code, event.reason);
    }
    clearPingInterval();
    onClose?.(event);
  };

  ws.onopen = () => {
    pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      }
    }, 30000);
  };

  const originalClose = ws.close.bind(ws);
  ws.close = (...args) => {
    clearPingInterval();
    originalClose(...args);
  };

  return ws;
}
