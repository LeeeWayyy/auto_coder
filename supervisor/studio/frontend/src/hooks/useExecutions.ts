/**
 * React Query hooks for execution operations.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type {
  ExecutionResponse,
  NodeExecutionStatus,
} from '../types/workflow';
import {
  listExecutions,
  getExecution,
  getExecutionNodes,
  executeWorkflow,
  cancelExecution,
  type ListExecutionsParams,
  type ExecuteWorkflowParams,
} from '../api/client';

// Query keys for cache management
export const executionKeys = {
  all: ['executions'] as const,
  lists: () => [...executionKeys.all, 'list'] as const,
  list: (filters?: ListExecutionsParams) =>
    [...executionKeys.lists(), filters] as const,
  details: () => [...executionKeys.all, 'detail'] as const,
  detail: (id: string) => [...executionKeys.details(), id] as const,
  nodes: (id: string) => [...executionKeys.detail(id), 'nodes'] as const,
};

/**
 * Fetch executions with optional filters.
 */
export function useExecutions(params?: ListExecutionsParams) {
  return useQuery({
    queryKey: executionKeys.list(params),
    queryFn: () => listExecutions(params),
    staleTime: 10000, // Consider data fresh for 10 seconds
  });
}

/**
 * Fetch a specific execution.
 */
export function useExecution(executionId: string | undefined) {
  return useQuery({
    queryKey: executionKeys.detail(executionId ?? ''),
    queryFn: () => getExecution(executionId!),
    enabled: !!executionId,
    staleTime: 5000,
    // Refetch while running
    refetchInterval: (query) => {
      const data = query.state.data as ExecutionResponse | undefined;
      return data?.status === 'running' ? 2000 : false;
    },
  });
}

/**
 * Fetch node statuses for an execution.
 */
export function useExecutionNodes(executionId: string | undefined) {
  return useQuery({
    queryKey: executionKeys.nodes(executionId ?? ''),
    queryFn: () => getExecutionNodes(executionId!),
    enabled: !!executionId,
    staleTime: 5000,
  });
}

/**
 * Start workflow execution.
 */
export function useExecuteWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (params: ExecuteWorkflowParams) => executeWorkflow(params),
    onSuccess: () => {
      // Invalidate execution list
      queryClient.invalidateQueries({ queryKey: executionKeys.lists() });
    },
  });
}

/**
 * Cancel a running execution.
 */
export function useCancelExecution() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (executionId: string) => cancelExecution(executionId),
    onSuccess: (_, executionId) => {
      // Refetch execution status
      queryClient.invalidateQueries({
        queryKey: executionKeys.detail(executionId),
      });
      queryClient.invalidateQueries({
        queryKey: executionKeys.nodes(executionId),
      });
    },
  });
}

/**
 * Update node statuses in the cache (for WebSocket updates).
 */
export function useUpdateNodeStatus() {
  const queryClient = useQueryClient();

  return (executionId: string, nodeId: string, status: string, output?: Record<string, unknown>) => {
    queryClient.setQueryData<NodeExecutionStatus[]>(
      executionKeys.nodes(executionId),
      (old) => {
        if (!old) return old;
        return old.map((node) =>
          node.node_id === nodeId
            ? { ...node, status: status as NodeExecutionStatus['status'], output: output ?? node.output }
            : node
        );
      }
    );
  };
}
