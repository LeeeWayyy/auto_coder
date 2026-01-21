/**
 * Branch node component for conditional routing.
 */

import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { BaseNodeData } from './BaseNode';

export interface BranchNodeData extends BaseNodeData {
  conditionExpr?: string;
  onTrue?: string;
  onFalse?: string;
}

const statusColors: Record<string, string> = {
  pending: 'bg-gray-200 border-gray-400',
  ready: 'bg-blue-100 border-blue-400',
  running: 'bg-amber-100 border-amber-400 animate-pulse',
  completed: 'bg-emerald-100 border-emerald-400',
  failed: 'bg-red-100 border-red-400',
  skipped: 'bg-gray-100 border-gray-300',
};

function BranchNodeComponent({ data, selected }: NodeProps) {
  const nodeData = (data ?? {}) as BranchNodeData;
  const status = nodeData.status || 'pending';
  const statusColor = statusColors[status];

  return (
    <div
      className={`
        relative
        ${selected ? 'ring-2 ring-blue-500 ring-offset-2' : ''}
      `}
    >
      {/* Diamond shape for branch */}
      <div
        className={`
          w-24 h-24 rotate-45 rounded-lg shadow-md border-2
          ${statusColor}
          border-l-4 border-l-amber-500
        `}
      >
        <Handle
          type="target"
          position={Position.Top}
          className="w-3 h-3 bg-gray-400 border-2 border-white -rotate-45"
          style={{ top: -6, left: '50%', transform: 'translateX(-50%) rotate(-45deg)' }}
        />
      </div>

      {/* Label overlay (not rotated) */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="text-center">
          <div className="font-medium text-sm">{nodeData.label || 'Branch'}</div>
          {nodeData.conditionExpr && (
            <div className="text-xs text-amber-600 font-mono truncate max-w-[80px]">
              {nodeData.conditionExpr}
            </div>
          )}
        </div>
      </div>

      {/* True/False handles */}
      <Handle
        type="source"
        position={Position.Right}
        id="true"
        className="w-3 h-3 bg-emerald-500 border-2 border-white"
        style={{ right: -6, top: '50%' }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="false"
        className="w-3 h-3 bg-red-500 border-2 border-white"
        style={{ bottom: -6, left: '50%' }}
      />
    </div>
  );
}

export const BranchNode = memo(BranchNodeComponent);
