/**
 * Task node component for AI role execution.
 */

import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { BaseNode, type BaseNodeData } from './BaseNode';

export interface TaskNodeData extends BaseNodeData {
  role?: string;
  promptTemplate?: string;
}

function TaskNodeComponent(props: NodeProps) {
  const data = props.data as TaskNodeData;

  return (
    <BaseNode {...props} data={{ ...data, nodeType: 'task' }}>
      {data.role && (
        <div className="mt-1 text-xs text-blue-600 font-mono">{data.role}</div>
      )}
    </BaseNode>
  );
}

export const TaskNode = memo(TaskNodeComponent);
