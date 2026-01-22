/**
 * WorkflowCanvas component with ReactFlow for visual workflow editing.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  useReactFlow,
  type ReactFlowInstance,
  type Connection,
  type Edge as FlowEdge,
  type Node as FlowNode,
  type OnNodesChange,
  type OnEdgesChange,
  MarkerType,
  BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// Use default ReactFlow node renderer for stability
import type { WorkflowGraph, Node, Edge, NodeStatus, NodeType } from '../types/workflow';
import { nodeTypes } from './nodes';

const EMPTY_STATUSES: Record<string, NodeStatus> = {};

export interface WorkflowCanvasProps {
  workflow: WorkflowGraph;
  nodeStatuses?: Record<string, NodeStatus>;
  activeNodeIds?: Set<string>;
  readOnly?: boolean;
  onWorkflowChange?: (workflow: WorkflowGraph) => void;
  onNodeSelect?: (nodeId: string | null) => void;
  onNodeAdd?: (nodeType: NodeType, position: { x: number; y: number }) => void;
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
    type: node.type || 'task',
    position: node.position || {
      x: 100 + (index % 4) * 200,
      y: 100 + Math.floor(index / 4) * 150,
    },
    data: {
      label: node.label || node.id,
      nodeType: node.type || 'task',
      status: nodeStatuses[node.id] || 'pending',
      description: node.description,
      role: node.task_config?.role,
      gateType: node.gate_config?.gate_type,
      conditionExpr: node.branch_config?.condition
        ? JSON.stringify(node.branch_config.condition)
        : undefined,
      onTrue: node.branch_config?.on_true,
      onFalse: node.branch_config?.on_false,
    },
  }));
}

/**
 * Convert workflow schema to ReactFlow edges.
 * Stores original edge metadata in data property to preserve it on round-trip.
 */
function toFlowEdges(edges: Edge[], activeNodeIds?: Set<string>): FlowEdge[] {
  const active = activeNodeIds || new Set<string>();
  return edges.map((edge, index) => ({
    id: edge.id || `${edge.source}-${edge.target}-${index}`,
    source: edge.source,
    target: edge.target,
    animated: (active.has(edge.source) || active.has(edge.target)) || edge.is_loop_edge,
    style:
      active.has(edge.source) || active.has(edge.target)
        ? { stroke: '#f59e0b', strokeWidth: 2 }
        : edge.is_loop_edge
        ? { strokeDasharray: '5,5' }
        : undefined,
    label: edge.condition,
    // Preserve original edge metadata for round-trip conversion
    data: {
      id: edge.id,
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
      type: (original?.type || (flowNode.type as Node['type']) || 'task'),
      label: (flowNode.data?.label as string) || original?.label || flowNode.id,
      position: { x: flowNode.position.x, y: flowNode.position.y },
    };

    // For existing nodes, preserve original config (task_config, gate_config, etc.)
    // For new nodes (no original), just return the base node
    if (original) {
      return { ...original, ...baseNode, type: original.type };
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
    originalEdges.map((e, idx) => [e.id || `${e.source}-${e.target}-${idx}`, e])
  );

  return flowEdges.map((flowEdge) => {
    // Prefer data from the flow edge (preserved during toFlowEdges)
    const edgeData = flowEdge.data as { id?: string; condition?: string; is_loop_edge?: boolean; data_mapping?: Record<string, string> } | undefined;

    if (edgeData && (edgeData.condition !== undefined || edgeData.is_loop_edge !== undefined || edgeData.data_mapping !== undefined)) {
      return {
        id: edgeData.id || flowEdge.id,
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
      id: original?.id || flowEdge.id,
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
  nodeStatuses = EMPTY_STATUSES,
  activeNodeIds,
  readOnly = false,
  onWorkflowChange,
  onNodeSelect,
  onNodeAdd,
}: WorkflowCanvasProps) {
  const reactFlow = useReactFlow();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<FlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<FlowEdge>([]);
  const [viewport, setViewport] = useState({ x: 0, y: 0, zoom: 1 });
  const [debugEnabled, setDebugEnabled] = useState(() => {
    try {
      return localStorage.getItem('studio.debug') === '1';
    } catch {
      return false;
    }
  });

  // Ref for debounce timeout to enable cleanup on unmount
  const debounceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Track previous workflow ID to detect workflow changes
  const prevWorkflowIdRef = useRef<string>(workflow.id);

  // Refs to track latest nodes/edges/workflow for use in callbacks
  // This prevents stale closure issues in debounced updates
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const workflowRef = useRef(workflow);
  const rfInstanceRef = useRef<ReactFlowInstance | null>(null);
  const didAutoFitRef = useRef(false);
  nodesRef.current = nodes;
  edgesRef.current = edges;
  workflowRef.current = workflow;

  // Reset nodes/edges when workflow changes (e.g., navigating to different workflow)
  useEffect(() => {
    const workflowChanged = prevWorkflowIdRef.current !== workflow.id;
    if (workflowChanged) {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
        debounceTimeoutRef.current = null;
      }
      prevWorkflowIdRef.current = workflow.id;
      didAutoFitRef.current = false;
    }

    // Sync ReactFlow state from workflow model updates.
    setNodes(toFlowNodes(workflow.nodes, nodeStatuses));
    setEdges(toFlowEdges(workflow.edges, activeNodeIds));
  }, [workflow.id, workflow.nodes, workflow.edges, nodeStatuses, activeNodeIds, setNodes, setEdges]);

  useEffect(() => {
    if (!rfInstanceRef.current) return;
    if (workflow.nodes.length === 0) return;
    if (didAutoFitRef.current) return;
    const instance = rfInstanceRef.current;
    const nodePos = workflow.nodes[0].position;
    const fit = () => {
      if (nodePos) {
        instance.setCenter(nodePos.x, nodePos.y, { zoom: 1 });
      } else {
        instance.fitView({ padding: 0.2 });
      }
    };
    // Defer to next frame so nodes have dimensions.
    requestAnimationFrame(() => requestAnimationFrame(fit));
    didAutoFitRef.current = true;
  }, [workflow.id, workflow.nodes]);

  useEffect(() => {
    try {
      localStorage.setItem('studio.debug', debugEnabled ? '1' : '0');
    } catch {
      // Ignore storage failures (e.g. private mode).
    }
  }, [debugEnabled]);

  const copySnapshot = useCallback(() => {
    const size = containerRef.current
      ? {
          w: containerRef.current.clientWidth,
          h: containerRef.current.clientHeight,
        }
      : undefined;
    const snap = {
      workflowId: workflow.id,
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.type,
        label: String(n.data?.label ?? ''),
        position: n.position,
      })),
      edges: edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
      })),
      viewport,
      size,
    };
    navigator.clipboard.writeText(JSON.stringify(snap, null, 2));
  }, [workflow.id, nodes, edges, viewport]);

  // Sync nodes/edges when workflow changes (including sidebar edits)

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
      const onlySelection = changes.every((c) => c.type === 'select');
      onNodesChange(changes);
      if (!onlySelection) {
        // Schedule debounced update using ref for latest edges to avoid stale closure
        setNodes((current) => {
          debouncedUpdate(current, edgesRef.current);
          return current;
        });
      }
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
      onNodeSelect?.(node.id);
    },
    [onNodeSelect]
  );

  // Handle background click (deselect)
  const onPaneClick = useCallback(() => {
    onNodeSelect?.(null);
  }, [onNodeSelect]);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      if (readOnly) return;
      const nodeType = event.dataTransfer.getData('application/reactflow-node-type');
      if (!nodeType || !onNodeAdd) return;
      const position = reactFlow.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      onNodeAdd(nodeType as NodeType, position);
    },
    [onNodeAdd, reactFlow, readOnly]
  );

  const onDragOver = useCallback(
    (event: React.DragEvent) => {
      if (readOnly) {
        event.dataTransfer.dropEffect = 'none';
        return;
      }
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
    },
    [readOnly]
  );

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
    <div ref={containerRef} className="w-full h-full min-h-0 relative">
      <div className="absolute right-4 top-4 z-10 flex gap-2">
        <button
          type="button"
          onClick={() => reactFlow.fitView({ padding: 0.2 })}
          className="px-3 py-1.5 text-xs font-medium text-gray-700 bg-white border border-gray-200 rounded shadow-sm hover:bg-gray-50"
        >
          Zoom to Fit
        </button>
        <button
          type="button"
          onClick={() => setDebugEnabled((prev) => !prev)}
          className="px-3 py-1.5 text-xs font-medium text-gray-700 bg-white border border-gray-200 rounded shadow-sm hover:bg-gray-50"
        >
          Debug {debugEnabled ? 'On' : 'Off'}
        </button>
      </div>
      {debugEnabled && (
        <div className="absolute left-4 top-4 z-10 text-[11px] font-mono bg-white/90 border border-gray-200 rounded px-2 py-1">
          nodes:{nodes.length} edges:{edges.length} view:{Math.round(viewport.x)},{Math.round(viewport.y)}@{viewport.zoom.toFixed(2)}
          <button
            type="button"
            onClick={copySnapshot}
            className="ml-2 px-2 py-0.5 text-[10px] bg-gray-100 border border-gray-200 rounded"
          >
            copy snapshot
          </button>
        </div>
      )}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        style={{ width: '100%', height: '100%' }}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onInit={(instance) => {
          rfInstanceRef.current = instance;
          if (nodes.length > 0 && !didAutoFitRef.current) {
            instance.fitView({ padding: 0.2 });
            didAutoFitRef.current = true;
          }
          setViewport(instance.getViewport());
        }}
        onMoveEnd={() => {
          const instance = rfInstanceRef.current;
          if (instance) {
            setViewport(instance.getViewport());
          }
        }}
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
        minZoom={0.1}
        maxZoom={2}
        defaultEdgeOptions={{
          type: 'smoothstep',
          markerEnd: { type: MarkerType.ArrowClosed },
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
