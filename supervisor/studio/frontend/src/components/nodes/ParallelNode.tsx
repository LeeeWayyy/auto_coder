/**
 * Parallel node component for concurrent execution.
 */

import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { BaseNode, type BaseNodeData } from './BaseNode';

function ParallelNodeComponent(props: NodeProps) {
  const data = props.data as BaseNodeData;

  return (
    <BaseNode {...props} data={{ ...data, nodeType: 'parallel' }}>
      <div className="mt-1 text-xs text-cyan-600">Fork</div>
    </BaseNode>
  );
}

export const ParallelNode = memo(ParallelNodeComponent);
