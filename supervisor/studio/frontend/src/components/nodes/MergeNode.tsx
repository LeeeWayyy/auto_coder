/**
 * Merge node component for joining parallel branches.
 */

import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { BaseNode, type BaseNodeData } from './BaseNode';

function MergeNodeComponent(props: NodeProps) {
  const data = props.data as BaseNodeData;

  return <BaseNode {...props} data={{ ...data, nodeType: 'merge' }} />;
}

export const MergeNode = memo(MergeNodeComponent);
