/**
 * Human node component for user approval/input.
 */

import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { BaseNode, type BaseNodeData } from './BaseNode';

function HumanNodeComponent(props: NodeProps) {
  const data = (props.data ?? {}) as BaseNodeData;

  return (
    <BaseNode {...props} data={{ ...data, nodeType: 'human', label: data.label ?? 'Human' }}>
      <div className="mt-1 text-xs text-pink-600">Requires approval</div>
    </BaseNode>
  );
}

export const HumanNode = memo(HumanNodeComponent);
