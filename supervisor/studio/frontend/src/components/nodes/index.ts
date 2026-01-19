/**
 * Custom node types for ReactFlow.
 */

import type { NodeTypes } from '@xyflow/react';
import { TaskNode } from './TaskNode';
import { GateNode } from './GateNode';
import { BranchNode } from './BranchNode';
import { MergeNode } from './MergeNode';
import { ParallelNode } from './ParallelNode';
import { HumanNode } from './HumanNode';
import { SubgraphNode } from './SubgraphNode';

export { BaseNode } from './BaseNode';
export { TaskNode } from './TaskNode';
export { GateNode } from './GateNode';
export { BranchNode } from './BranchNode';
export { MergeNode } from './MergeNode';
export { ParallelNode } from './ParallelNode';
export { HumanNode } from './HumanNode';
export { SubgraphNode } from './SubgraphNode';

/**
 * Node type registry for ReactFlow.
 * Maps node type strings to React components.
 */
export const nodeTypes: NodeTypes = {
  task: TaskNode,
  gate: GateNode,
  branch: BranchNode,
  merge: MergeNode,
  parallel: ParallelNode,
  human: HumanNode,
  subgraph: SubgraphNode,
};
