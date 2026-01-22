/**
 * NodePalette component for draggable node types.
 */

import type { Node, NodeType } from '../types/workflow';

export interface NodePaletteProps {
  onNodeAdd: (nodeType: NodeType, position: { x: number; y: number }) => void;
}

// NOTE: No icon fields - using text badges instead.
const NODE_CATEGORIES: Record<
  string,
  Array<{ type: NodeType; label: string; description: string }>
> = {
  Tasks: [
    { type: 'task', label: 'Task', description: 'Execute an AI agent task' },
    { type: 'subgraph', label: 'Subgraph', description: 'Embed another workflow' },
  ],
  'Control Flow': [
    { type: 'gate', label: 'Gate', description: 'Quality gate checkpoint' },
    { type: 'branch', label: 'Branch', description: 'Conditional branching' },
    { type: 'merge', label: 'Merge', description: 'Merge parallel paths' },
    { type: 'parallel', label: 'Parallel', description: 'Execute nodes in parallel' },
  ],
  'Human-in-Loop': [
    { type: 'human', label: 'Human', description: 'Human approval/input' },
  ],
};

const NODE_TYPE_BADGES: Record<NodeType, string> = {
  task: 'T',
  gate: 'G',
  human: 'H',
  branch: 'B',
  merge: 'M',
  parallel: 'P',
  subgraph: 'S',
};

export function getDefaultConfig(nodeType: NodeType): Partial<Node> {
  switch (nodeType) {
    case 'task':
      return { task_config: { role: 'implementer' } };
    case 'gate':
      return { gate_config: { gate_type: 'test', auto_approve: false } };
    case 'human':
      return { human_config: { title: 'Review Required' } };
    case 'branch':
      return {
        branch_config: {
          condition: { field: 'status', operator: '==', value: 'success' },
          on_true: '',
          on_false: '',
        },
      };
    case 'merge':
      return { merge_config: { wait_for: 'all', merge_strategy: 'union' } };
    case 'parallel':
      return { parallel_config: { branches: [], wait_for: 'all' } };
    case 'subgraph':
      return {
        subgraph_config: {
          workflow_name: '',
          input_mapping: {},
          output_mapping: {},
        },
      };
    default:
      return {};
  }
}

export function NodePalette({ onNodeAdd }: NodePaletteProps) {
  return (
    <div className="p-3">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
        Node Types
      </div>
      <div className="space-y-4">
        {Object.entries(NODE_CATEGORIES).map(([category, nodes]) => (
          <div key={category}>
            <div className="text-[11px] font-medium text-gray-500 mb-2">
              {category}
            </div>
            <div className="space-y-2">
              {nodes.map((node) => (
                <div
                  key={node.type}
                  draggable
                  onDragStart={(event) => {
                    event.dataTransfer.setData(
                      'application/reactflow-node-type',
                      node.type
                    );
                    event.dataTransfer.effectAllowed = 'move';
                  }}
                  onDoubleClick={() =>
                    onNodeAdd(node.type, { x: 120, y: 120 })
                  }
                  title={node.description}
                  className="flex items-center gap-2 rounded border border-gray-200 px-2 py-2 text-sm text-gray-700 bg-white hover:bg-gray-50 cursor-grab"
                >
                  <div className="w-6 h-6 rounded bg-gray-100 text-gray-700 flex items-center justify-center text-xs font-semibold">
                    {NODE_TYPE_BADGES[node.type]}
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-800">
                      {node.label}
                    </div>
                    <div className="text-[11px] text-gray-500">
                      {node.description}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
