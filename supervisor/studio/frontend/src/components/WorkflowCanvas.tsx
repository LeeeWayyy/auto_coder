/**
 * WorkflowCanvas component with ReactFlow for visual workflow editing.
 */

import { useCallback, useMemo, useState, useEffect, useRef } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Edge as FlowEdge,
  type Node as FlowNode,
  type OnNodesChange,
  type OnEdgesChange,
  BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { nodeTypes } from './nodes';
import type { WorkflowGraph, Node, Edge, NodeStatus } from '../types/workflow';

export interface WorkflowCanvasProps {
  workflow: WorkflowGraph;
  nodeStatuses?: Record<string, NodeStatus>;
  readOnly?: boolean;
  onWorkflowChange?: (workflow: WorkflowGraph) => void;
  onNodeSelect?: (nodeId: string | null) => void;
}

/**
 * Convert workflow schema to ReactFlow nodes.
 */
function toFlowNodes(
  nodes: Node[],
  nodeStatuses: Record<string, NodeStatus>
): FlowNode[] {
  return nodes.map((node, index) => ({
    id: node.id,
    type: node.type,
    position: node.position || {
      x: 100 + (index % 4) * 200,
      y: 100 + Math.floor(index / 4) * 150,
    },
    data: {
      label: node.label || node.id,
      nodeType: node.type,
      status: nodeStatuses[node.id] || 'pending',
      description: node.task_config?.description || node.gate_config?.description,
      role: node.task_config?.role,
      checks: node.gate_config?.checks,
      conditionExpr: node.branch_config?.condition_expr,
      onTrue: node.branch_config?.on_true,
      onFalse: node.branch_config?.on_false,
    },
  }));
}

/**
 * Convert workflow schema to ReactFlow edges.
 * Stores original edge metadata in data property to preserve it on round-trip.
 */
function toFlowEdges(edges: Edge[]): FlowEdge[] {
  return edges.map((edge, index) => ({
    id: `${edge.source}-${edge.target}-${index}`,
    source: edge.source,
    target: edge.target,
    animated: edge.is_loop_edge,
    style: edge.is_loop_edge
      ? { strokeDasharray: '5,5' }
      : undefined,
    label: edge.condition,
    // Preserve original edge metadata for round-trip conversion
    data: {
      condition: edge.condition,
      is_loop_edge: edge.is_loop_edge,
      data_mapping: edge.data_mapping,
    },
  }));
}

/**
 * Convert ReactFlow nodes back to workflow schema.
 * Handles both existing nodes (preserves original config) and newly created nodes.
 */
function fromFlowNodes(flowNodes: FlowNode[], originalNodes: Node[]): Node[] {
  const originalMap = new Map(originalNodes.map((n) => [n.id, n]));

  return flowNodes.map((flowNode) => {
    const original = originalMap.get(flowNode.id);
    const baseNode: Node = {
      id: flowNode.id,
      type: (flowNode.type as Node['type']) || 'task',
      label: flowNode.data.label as string,
      position: { x: flowNode.position.x, y: flowNode.position.y },
    };

    // For existing nodes, preserve original config (task_config, gate_config, etc.)
    // For new nodes (no original), just return the base node
    if (original) {
      return { ...original, ...baseNode };
    }

    // New node - add default task_config
    return {
      ...baseNode,
      task_config: {
        role: 'implementer',
        description: 'New task',
      },
    };
  });
}

/**
 * Convert ReactFlow edges back to workflow schema.
 * Reads edge metadata from data property (preserved from toFlowEdges).
 * Falls back to original edge lookup for edges created via UI (which lack data).
 */
function fromFlowEdges(flowEdges: FlowEdge[], originalEdges: Edge[]): Edge[] {
  // Fallback map for edges that don't have data (e.g., newly created via UI)
  // Use edge ID for precise matching since source-target can have duplicates
  const originalById = new Map(
    originalEdges.map((e, idx) => [`${e.source}-${e.target}-${idx}`, e])
  );

  return flowEdges.map((flowEdge) => {
    // Prefer data from the flow edge (preserved during toFlowEdges)
    const edgeData = flowEdge.data as { condition?: string; is_loop_edge?: boolean; data_mapping?: Record<string, string> } | undefined;

    if (edgeData && (edgeData.condition !== undefined || edgeData.is_loop_edge !== undefined || edgeData.data_mapping !== undefined)) {
      return {
        source: flowEdge.source,
        target: flowEdge.target,
        condition: edgeData.condition,
        is_loop_edge: edgeData.is_loop_edge,
        data_mapping: edgeData.data_mapping,
      };
    }

    // Fallback for new edges created via UI (no data property)
    // Try to match by ID first, then by source-target
    const original = originalById.get(flowEdge.id);

    return {
      source: flowEdge.source,
      target: flowEdge.target,
      condition: original?.condition,
      is_loop_edge: original?.is_loop_edge,
      data_mapping: original?.data_mapping,
    };
  });
}

export function WorkflowCanvas({
  workflow,
  nodeStatuses = {},
  readOnly = false,
  onWorkflowChange,
  onNodeSelect,
}: WorkflowCanvasProps) {
  // Initialize ReactFlow state from workflow
  // Note: nodeStatuses is not in deps because useNodesState only uses initialNodes
  // on mount. Status updates are handled by the useEffect below that watches nodeStatuses.
  const initialNodes = useMemo(
    () => toFlowNodes(workflow.nodes, nodeStatuses),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [workflow.nodes]
  );

  const initialEdges = useMemo(
    () => toFlowEdges(workflow.edges),
    [workflow.edges]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Ref for debounce timeout to enable cleanup on unmount
  const debounceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Track previous workflow ID to detect workflow changes
  const prevWorkflowIdRef = useRef<string>(workflow.id);

  // Refs to track latest nodes/edges/workflow for use in callbacks
  // This prevents stale closure issues in debounced updates
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const workflowRef = useRef(workflow);
  nodesRef.current = nodes;
  edgesRef.current = edges;
  workflowRef.current = workflow;

  // Reset nodes/edges when workflow changes (e.g., navigating to different workflow)
  useEffect(() => {
    if (prevWorkflowIdRef.current !== workflow.id) {
      // Clear any pending debounced update to prevent stale edits being applied
      // to the new workflow
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
        debounceTimeoutRef.current = null;
      }
      setNodes(toFlowNodes(workflow.nodes, nodeStatuses));
      setEdges(toFlowEdges(workflow.edges));
      setSelectedNodeId(null);
      prevWorkflowIdRef.current = workflow.id;
    }
  }, [workflow.id, workflow.nodes, workflow.edges, nodeStatuses, setNodes, setEdges]);

  // Update nodes when statuses change
  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: {
          ...node.data,
          status: nodeStatuses[node.id] || 'pending',
        },
      }))
    );
  }, [nodeStatuses, setNodes]);

  // Clean up debounce timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, []);

  // Debounced workflow change callback
  // Uses workflowRef to access latest workflow state when timeout fires,
  // preventing stale closure issues when metadata changes during debounce period.
  const debouncedUpdate = useCallback(
    (updatedNodes: FlowNode[], updatedEdges: FlowEdge[]) => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
      debounceTimeoutRef.current = setTimeout(() => {
        if (onWorkflowChange) {
          const currentWorkflow = workflowRef.current;
          onWorkflowChange({
            ...currentWorkflow,
            nodes: fromFlowNodes(updatedNodes, currentWorkflow.nodes),
            edges: fromFlowEdges(updatedEdges, currentWorkflow.edges),
          });
        }
      }, 500);
    },
    [onWorkflowChange]
  );

  // Handle node changes with read-only support
  const handleNodesChange: OnNodesChange = useCallback(
    (changes) => {
      if (readOnly) {
        // In read-only mode, only allow selection changes
        const selectChanges = changes.filter((c) => c.type === 'select');
        if (selectChanges.length > 0) {
          onNodesChange(selectChanges);
        }
        return;
      }
      onNodesChange(changes);
      // Schedule debounced update using ref for latest edges to avoid stale closure
      setNodes((current) => {
        debouncedUpdate(current, edgesRef.current);
        return current;
      });
    },
    [readOnly, onNodesChange, debouncedUpdate, setNodes]
  );

  // Handle edge changes
  const handleEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      if (readOnly) return;
      onEdgesChange(changes);
      // Use ref for latest nodes to avoid stale closure
      setEdges((current) => {
        debouncedUpdate(nodesRef.current, current);
        return current;
      });
    },
    [readOnly, onEdgesChange, debouncedUpdate, setEdges]
  );

  // Handle new connections
  const onConnect = useCallback(
    (params: Connection) => {
      if (readOnly) return;
      // Use ref for latest nodes to avoid stale closure
      setEdges((eds) => {
        const newEdges = addEdge(params, eds);
        debouncedUpdate(nodesRef.current, newEdges);
        return newEdges;
      });
    },
    [readOnly, setEdges, debouncedUpdate]
  );

  // Handle node selection
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: FlowNode) => {
      setSelectedNodeId(node.id);
      onNodeSelect?.(node.id);
    },
    [onNodeSelect]
  );

  // Handle background click (deselect)
  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    onNodeSelect?.(null);
  }, [onNodeSelect]);

  // Handle edge deletion
  const onEdgesDelete = useCallback(
    (deletedEdges: FlowEdge[]) => {
      if (readOnly) return;
      const deletedIds = new Set(deletedEdges.map((e) => e.id));
      // Use ref for latest nodes to avoid stale closure
      setEdges((eds) => {
        const newEdges = eds.filter((e) => !deletedIds.has(e.id));
        debouncedUpdate(nodesRef.current, newEdges);
        return newEdges;
      });
    },
    [readOnly, setEdges, debouncedUpdate]
  );

  // Handle node deletion
  const onNodesDelete = useCallback(
    (deletedNodes: FlowNode[]) => {
      if (readOnly) return;
      const deletedIds = new Set(deletedNodes.map((n) => n.id));
      setNodes((nds) => {
        const newNodes = nds.filter((n) => !deletedIds.has(n.id));
        // Also remove edges connected to deleted nodes
        setEdges((eds) => {
          const newEdges = eds.filter(
            (e) => !deletedIds.has(e.source) && !deletedIds.has(e.target)
          );
          debouncedUpdate(newNodes, newEdges);
          return newEdges;
        });
        return newNodes;
      });
    },
    [readOnly, setNodes, setEdges, debouncedUpdate]
  );

  return (
    <div className="w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onEdgesDelete={onEdgesDelete}
        onNodesDelete={onNodesDelete}
        nodesDraggable={!readOnly}
        nodesConnectable={!readOnly}
        elementsSelectable={true}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2}
        defaultEdgeOptions={{
          type: 'smoothstep',
        }}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            const status = (node.data as { status?: string }).status || 'pending';
            const colors: Record<string, string> = {
              pending: '#6b7280',
              ready: '#3b82f6',
              running: '#f59e0b',
              completed: '#10b981',
              failed: '#ef4444',
              skipped: '#9ca3af',
            };
            return colors[status] || '#6b7280';
          }}
          maskColor="rgba(0, 0, 0, 0.2)"
        />
      </ReactFlow>
    </div>
  );
}
