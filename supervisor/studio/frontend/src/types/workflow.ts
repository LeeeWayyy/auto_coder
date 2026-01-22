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

export type ExecutionStatus = 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';

export interface TaskConfig {
  role: string;
  task_template?: string;
  timeout?: number;
  gates?: string[];
  worktree_id?: string | null;
}

export interface GateConfig {
  gate_type: string;
  auto_approve?: boolean;
  requires_worktree?: boolean;
}

export interface LoopCondition {
  field: string;
  operator:
    | '=='
    | '!='
    | '>'
    | '<'
    | '>='
    | '<='
    | 'in'
    | 'not_in'
    | 'contains'
    | 'starts_with'
    | 'ends_with';
  value: string | number | boolean | Array<string | number | boolean>;
  max_iterations?: number;
}

export interface BranchConfig {
  condition: LoopCondition;
  on_true: string;
  on_false: string;
}

export interface Node {
  id: string;
  type: NodeType;
  label?: string;
  description?: string;
  task_config?: TaskConfig;
  gate_config?: GateConfig;
  branch_config?: BranchConfig;
  merge_config?: {
    wait_for?: 'all' | 'any';
    merge_strategy?: 'union' | 'intersection' | 'first';
  };
  parallel_config?: {
    branches?: string[];
    wait_for?: 'all' | 'any' | 'first';
  };
  subgraph_config?: {
    workflow_name: string;
    input_mapping?: Record<string, string>;
    output_mapping?: Record<string, string>;
  };
  human_config?: {
    title: string;
    description?: string | null;
  };
  ui_metadata?: Record<string, unknown> | null;
  wait_for_incoming?: 'all' | 'any';
  position?: {
    x: number;
    y: number;
  };
}

export interface Edge {
  id: string;
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

export interface TraceEvent {
  id: string;
  timestamp: string;
  nodeId: string;
  nodeLabel: string;
  nodeType: NodeType;
  status: NodeStatus;
}

export interface ExecutionEvent {
  id: number;
  execution_id: string;
  event_type: string;
  node_id?: string | null;
  node_type?: NodeType | null;
  status?: string | null;
  payload?: Record<string, unknown> | null;
  version?: number | null;
  timestamp: string;
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
  completed_at?: string;
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

export interface WSHumanWaiting {
  type: 'human_waiting';
  node_id: string;
  title: string;
  description?: string;
  current_output?: Record<string, unknown>;
}

export interface WSHumanResolved {
  type: 'human_resolved';
  node_id: string;
  action: 'approve' | 'reject' | 'edit';
  status: ExecutionStatus;
}

export type WSMessage =
  | WSNodeUpdate
  | WSExecutionComplete
  | WSInitialState
  | WSHeartbeat
  | WSPong
  | WSError
  | WSHumanWaiting
  | WSHumanResolved;
