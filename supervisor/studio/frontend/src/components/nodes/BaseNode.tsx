/**
 * Base node component with shared styling and status indicators.
 */

import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { NodeStatus, NodeType } from '../../types/workflow';

export interface BaseNodeData {
  label: string;
  nodeType: NodeType;
  status?: NodeStatus;
  description?: string;
}

const statusColors: Record<NodeStatus, string> = {
  pending: 'bg-gray-200 border-gray-400',
  ready: 'bg-blue-100 border-blue-400',
  running: 'bg-amber-100 border-amber-400 animate-pulse',
  completed: 'bg-emerald-100 border-emerald-400',
  failed: 'bg-red-100 border-red-400',
  skipped: 'bg-gray-100 border-gray-300',
};

const statusIcons: Record<NodeStatus, string> = {
  pending: '',
  ready: '',
  running: '',
  completed: '',
  failed: '',
  skipped: '',
};

const typeColors: Record<NodeType, string> = {
  task: 'border-l-blue-500',
  gate: 'border-l-violet-500',
  branch: 'border-l-amber-500',
  merge: 'border-l-emerald-500',
  parallel: 'border-l-cyan-500',
  human: 'border-l-pink-500',
  subgraph: 'border-l-indigo-500',
};

const typeIcons: Record<NodeType, string> = {
  task: '',
  gate: '',
  branch: '',
  merge: '',
  parallel: '',
  human: '',
  subgraph: '',
};

interface BaseNodeProps extends NodeProps {
  data: BaseNodeData;
  children?: React.ReactNode;
}

function BaseNodeComponent({ data, selected, children }: BaseNodeProps) {
  const status = data.status || 'pending';
  const typeColor = typeColors[data.nodeType] || 'border-l-gray-500';
  const statusColor = statusColors[status];

  return (
    <div
      className={`
        px-4 py-2 rounded-lg shadow-md border-2 border-l-4
        min-w-[140px] max-w-[200px]
        ${typeColor}
        ${statusColor}
        ${selected ? 'ring-2 ring-blue-500 ring-offset-2' : ''}
        transition-all duration-200
      `}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 bg-gray-400 border-2 border-white"
      />

      <div className="flex items-center gap-2">
        <span className="text-lg" title={data.nodeType}>
          {typeIcons[data.nodeType]}
        </span>
        <div className="flex-1 min-w-0">
          <div className="font-medium text-sm truncate">{data.label}</div>
          {data.description && (
            <div className="text-xs text-gray-500 truncate">{data.description}</div>
          )}
        </div>
        {status !== 'pending' && (
          <span className="text-sm" title={status}>
            {statusIcons[status]}
          </span>
        )}
      </div>

      {children}

      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 bg-gray-400 border-2 border-white"
      />
    </div>
  );
}

export const BaseNode = memo(BaseNodeComponent);
