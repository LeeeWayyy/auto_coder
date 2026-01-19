/**
 * TypeScript types that mirror the Python WorkflowGraph schema.
 * These types define the structure for visual workflow editing.
 */

export type NodeType =
  | 'task'
  | 'gate'
  | 'branch'
  | 'merge'
  | 'parallel'
  | 'human'
  | 'subgraph';

export type NodeStatus =
  | 'pending'
  | 'ready'
  | 'running'
  | 'completed'
  | 'failed'
  | 'skipped';

export type ExecutionStatus = 'running' | 'completed' | 'failed' | 'cancelled';

export interface TaskConfig {
  role: string;
  prompt_template?: string;
  description?: string;
  target_files?: string[];
  output_format?: string;
  timeout?: number;
}

export interface GateConfig {
  checks: string[];
  description?: string;
}

export interface BranchConfig {
  condition_expr: string;
  on_true: string;
  on_false: string;
  max_iterations?: number;
}

export interface Node {
  id: string;
  type: NodeType;
  label?: string;
  task_config?: TaskConfig;
  gate_config?: GateConfig;
  branch_config?: BranchConfig;
  wait_for_incoming?: 'all' | 'any';
  position?: {
    x: number;
    y: number;
  };
}

export interface Edge {
  source: string;
  target: string;
  condition?: string;
  is_loop_edge?: boolean;
  data_mapping?: Record<string, string>;
}

export interface WorkflowConfig {
  fail_fast?: boolean;
  max_parallel_nodes?: number;
  retry_policy?: {
    max_retries: number;
    backoff_seconds: number;
  };
}

export interface WorkflowGraph {
  id: string;
  name: string;
  description?: string;
  version?: string;
  nodes: Node[];
  edges: Edge[];
  entry_point: string;
  exit_points?: string[];
  config?: WorkflowConfig;
}

// API Response types

export interface ExecutionResponse {
  execution_id: string;
  workflow_id: string;
  graph_id: string;
  status: ExecutionStatus;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
}

export interface NodeExecutionStatus {
  node_id: string;
  node_type: NodeType;
  status: NodeStatus;
  output?: Record<string, unknown> | null;
  error?: string | null;
  version: number;
}

// WebSocket message types

export interface WSNodeUpdate {
  type: 'node_update';
  node_id: string;
  status: NodeStatus;
  output: Record<string, unknown>;
  version: number;
  timestamp: string;
}

export interface WSExecutionComplete {
  type: 'execution_complete';
  status: ExecutionStatus;
  final_nodes?: Array<{
    node_id: string;
    status: NodeStatus;
    version: number;
  }>;
  error?: string;
}

export interface WSInitialState {
  type: 'initial_state';
  nodes: NodeExecutionStatus[];
  status: ExecutionStatus;
}

export interface WSHeartbeat {
  type: 'heartbeat';
}

export interface WSPong {
  type: 'pong';
}

export interface WSError {
  type: 'error';
  code: number;
  detail: string;
}

export type WSMessage =
  | WSNodeUpdate
  | WSExecutionComplete
  | WSInitialState
  | WSHeartbeat
  | WSPong
  | WSError;
