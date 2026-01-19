/**
 * Subgraph node component for nested workflows.
 */

import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { BaseNode, type BaseNodeData } from './BaseNode';

export interface SubgraphNodeData extends BaseNodeData {
  subgraphId?: string;
}

function SubgraphNodeComponent(props: NodeProps) {
  const data = props.data as SubgraphNodeData;

  return (
    <BaseNode {...props} data={{ ...data, nodeType: 'subgraph' }}>
      {data.subgraphId && (
        <div className="mt-1 text-xs text-indigo-600 font-mono truncate">
          {data.subgraphId}
        </div>
      )}
    </BaseNode>
  );
}

export const SubgraphNode = memo(SubgraphNodeComponent);
