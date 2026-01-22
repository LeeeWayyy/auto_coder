/**
 * Hook for WebSocket connection to execution updates.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { createExecutionWebSocket } from '../api/client';
import { executionKeys } from './useExecutions';
import type {
  WSMessage,
  NodeExecutionStatus,
  ExecutionResponse,
  NodeStatus,
  ExecutionStatus,
} from '../types/workflow';

export interface UseExecutionWebSocketOptions {
  enabled?: boolean;
  onInitialState?: () => void;
  onNodeUpdate?: (nodeId: string, status: NodeStatus, output: Record<string, unknown>, timestamp: string) => void;
  onExecutionComplete?: (status: ExecutionStatus, finalNodes?: Array<{ node_id: string; status: NodeStatus; version: number }>) => void;
  onError?: (error: string) => void;
}

export function useExecutionWebSocket(
  executionId: string | undefined,
  options: UseExecutionWebSocketOptions = {}
) {
  const {
    enabled = true,
    onInitialState,
    onNodeUpdate,
    onExecutionComplete,
    onError,
  } = options;

  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Guard to prevent duplicate completion callbacks (e.g., initial_state shows completed,
  // then execution_complete message arrives). Only invoke once per connection.
  const didCompleteRef = useRef(false);

  // Store callbacks in refs to prevent WebSocket reconnection on callback changes.
  // This prevents the useEffect from re-running when callers pass inline functions.
  const onNodeUpdateRef = useRef(onNodeUpdate);
  const onExecutionCompleteRef = useRef(onExecutionComplete);
  const onErrorRef = useRef(onError);
  const onInitialStateRef = useRef(onInitialState);
  onNodeUpdateRef.current = onNodeUpdate;
  onExecutionCompleteRef.current = onExecutionComplete;
  onErrorRef.current = onError;
  onInitialStateRef.current = onInitialState;

  const handleMessage = useCallback(
    (data: unknown) => {
      const message = data as WSMessage;

      switch (message.type) {
        case 'initial_state':
          onInitialStateRef.current?.();
          // Smart merge initial state with existing cache, respecting version numbers.
          // This handles the race condition where a node_update arrives before
          // initial_state due to network timing - we don't overwrite newer data.
          queryClient.setQueryData<NodeExecutionStatus[]>(
            executionKeys.nodes(executionId!),
            (old) => {
              if (!old || old.length === 0) {
                // No existing cache - use initial state directly
                return message.nodes;
              }
              // Merge: for each node, keep status/version from the higher version,
              // but always take node_type from initial_state (authoritative source)
              const oldMap = new Map(old.map((n) => [n.node_id, n]));
              return message.nodes.map((newNode) => {
                const oldNode = oldMap.get(newNode.node_id);
                if (oldNode && oldNode.version > newNode.version) {
                  // Cached node has newer version - keep status/version but use
                  // node_type from initial_state (fixes placeholder 'task' type)
                  return { ...oldNode, node_type: newNode.node_type };
                }
                return newNode;
              });
            }
          );
          // Also update execution status to handle late-connect scenario
          // (when execution already completed before WebSocket connection).
          // Only update if current status is not already terminal to prevent
          // race condition where execution_complete arrives before initial_state.
          queryClient.setQueryData<ExecutionResponse>(
            executionKeys.detail(executionId!),
            (old) => {
              if (!old) return old;
              // Don't overwrite terminal status with potentially stale initial_state
              const terminalStatuses = ['completed', 'failed', 'cancelled'];
              if (terminalStatuses.includes(old.status)) {
                return old;
              }
              return { ...old, status: message.status };
            }
          );
          // If execution is already complete, notify callback (once only)
          if (message.status !== 'running' && !didCompleteRef.current) {
            didCompleteRef.current = true;
            onExecutionCompleteRef.current?.(message.status);
          }
          break;

        case 'node_update':
          // Update specific node in cache using server-provided version
          // If cache is empty (update arrived before initial_state), seed it with this node
          queryClient.setQueryData<NodeExecutionStatus[]>(
            executionKeys.nodes(executionId!),
            (old) => {
              const updatedNode: NodeExecutionStatus = {
                node_id: message.node_id,
                node_type: 'task', // Default, will be corrected by initial_state
                status: message.status,
                output: message.output,
                version: message.version,
              };

              if (!old || old.length === 0) {
                // Cache is empty - seed with this node (initial_state will merge later)
                return [updatedNode];
              }

              // Check if node exists in cache
              const exists = old.some((n) => n.node_id === message.node_id);
              if (!exists) {
                // Node not in cache yet - add it
                return [...old, updatedNode];
              }

              // Update existing node only if incoming version is newer
              // This protects against out-of-order WebSocket messages
              return old.map((node) => {
                if (node.node_id === message.node_id) {
                  // Only update if incoming version is greater than cached version
                  // Preserve node_type from cache (authoritative from initial_state)
                  // since node_update doesn't include the actual node_type
                  return message.version > node.version
                    ? { ...node, status: message.status, output: message.output, version: message.version }
                    : node;
                }
                return node;
              });
            }
          );
          onNodeUpdateRef.current?.(message.node_id, message.status, message.output, message.timestamp);
          break;

        case 'execution_complete':
          // Update execution status, completed_at, and error in cache
          queryClient.setQueryData<ExecutionResponse>(
            executionKeys.detail(executionId!),
            (old) => {
              if (!old) return old;
              return {
                ...old,
                status: message.status,
                // Include completion details if present
                ...(message.completed_at ? { completed_at: message.completed_at } : {}),
                ...(message.error ? { error: message.error } : {}),
              };
            }
          );

          // Update final node states if provided (for cancellation)
          if (message.final_nodes) {
            queryClient.setQueryData<NodeExecutionStatus[]>(
              executionKeys.nodes(executionId!),
              (old) => {
                if (!old) return old;
                const finalMap = new Map(
                  message.final_nodes?.map((n) => [n.node_id, n])
                );
                return old.map((node) => {
                  const final = finalMap.get(node.node_id);
                  return final
                    ? { ...node, status: final.status, version: final.version }
                    : node;
                });
              }
            );
          }

          // Only invoke completion callback once (guards against race with initial_state)
          if (!didCompleteRef.current) {
            didCompleteRef.current = true;
            onExecutionCompleteRef.current?.(message.status, message.final_nodes);
          }
          // Close WebSocket after execution completes - no more updates expected
          // This prevents resource leaks from idle connections
          wsRef.current?.close();
          break;

        case 'error':
          onErrorRef.current?.(message.detail);
          break;

        case 'heartbeat':
        case 'pong':
          // Ignore heartbeat and pong messages
          break;
      }
    },
    [executionId, queryClient]
  );

  useEffect(() => {
    if (!executionId || !enabled) {
      return;
    }

    // Reset completion guard for new connection
    didCompleteRef.current = false;

    const ws = createExecutionWebSocket(
      executionId,
      handleMessage,
      (error) => {
        setIsConnected(false);
        // Forward transport errors to onError callback
        onErrorRef.current?.(`WebSocket connection error: ${error.type}`);
      },
      () => setIsConnected(false)
    );

    // Use addEventListener to avoid overwriting the ping interval setup in createExecutionWebSocket
    ws.addEventListener('open', () => {
      setIsConnected(true);
    });

    // Handle race condition: if WebSocket opened before listener was attached,
    // the open event would be missed. Check readyState immediately.
    if (ws.readyState === WebSocket.OPEN) {
      setIsConnected(true);
    }

    wsRef.current = ws;

    return () => {
      ws.close();
      wsRef.current = null;
      setIsConnected(false);
    };
  }, [executionId, enabled, handleMessage]);

  return {
    isConnected,
    close: () => {
      wsRef.current?.close();
    },
  };
}
