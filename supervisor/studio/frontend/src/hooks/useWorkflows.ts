/**
 * React Query hooks for workflow operations.
 */

import {
  useQuery,
  useMutation,
  useQueryClient,
} from '@tanstack/react-query';
import type { WorkflowGraph } from '../types/workflow';
import {
  listWorkflows,
  getWorkflow,
  createWorkflow,
  updateWorkflow,
  deleteWorkflow,
} from '../api/client';

// Query keys for cache management
export const workflowKeys = {
  all: ['workflows'] as const,
  lists: () => [...workflowKeys.all, 'list'] as const,
  list: () => [...workflowKeys.lists()] as const,
  details: () => [...workflowKeys.all, 'detail'] as const,
  detail: (id: string) => [...workflowKeys.details(), id] as const,
};

/**
 * Fetch all workflows.
 */
export function useWorkflows() {
  return useQuery({
    queryKey: workflowKeys.list(),
    queryFn: listWorkflows,
    staleTime: 30000, // Consider data fresh for 30 seconds
  });
}

/**
 * Fetch a specific workflow by ID.
 */
export function useWorkflow(graphId: string | undefined) {
  return useQuery({
    queryKey: workflowKeys.detail(graphId ?? ''),
    queryFn: () => getWorkflow(graphId!),
    enabled: !!graphId,
    staleTime: 30000,
  });
}

/**
 * Create a new workflow.
 */
export function useCreateWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (workflow: WorkflowGraph) => createWorkflow(workflow),
    onSuccess: (data) => {
      // Invalidate list to refetch
      queryClient.invalidateQueries({ queryKey: workflowKeys.lists() });
      // Pre-populate detail cache
      queryClient.setQueryData(workflowKeys.detail(data.id), data);
    },
  });
}

/**
 * Update an existing workflow.
 */
export function useUpdateWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ graphId, workflow }: { graphId: string; workflow: WorkflowGraph }) =>
      updateWorkflow(graphId, workflow),
    onSuccess: (data) => {
      // Update both list and detail caches
      queryClient.invalidateQueries({ queryKey: workflowKeys.lists() });
      queryClient.setQueryData(workflowKeys.detail(data.id), data);
    },
  });
}

/**
 * Delete a workflow.
 */
export function useDeleteWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (graphId: string) => deleteWorkflow(graphId),
    onSuccess: (_, graphId) => {
      // Remove from cache and refetch list
      queryClient.removeQueries({ queryKey: workflowKeys.detail(graphId) });
      queryClient.invalidateQueries({ queryKey: workflowKeys.lists() });
    },
  });
}
