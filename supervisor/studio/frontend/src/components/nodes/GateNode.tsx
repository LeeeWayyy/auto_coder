/**
 * Gate node component for validation checks.
 */

import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { BaseNode, type BaseNodeData } from './BaseNode';

export interface GateNodeData extends BaseNodeData {
  checks?: string[];
}

function GateNodeComponent(props: NodeProps) {
  const data = props.data as GateNodeData;

  return (
    <BaseNode {...props} data={{ ...data, nodeType: 'gate' }}>
      {data.checks && data.checks.length > 0 && (
        <div className="mt-1 text-xs text-violet-600">
          {data.checks.length} check(s)
        </div>
      )}
    </BaseNode>
  );
}

export const GateNode = memo(GateNodeComponent);
