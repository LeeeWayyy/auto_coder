/**
 * StateInspector component for persisted node outputs and global state.
 */

import { useMemo } from 'react';
import type { WorkflowGraph, NodeExecutionStatus } from '../types/workflow';

export interface StateInspectorProps {
  selectedNodeId: string | null;
  nodeOutputs: Map<string, NodeExecutionStatus>;
  workflow: WorkflowGraph;
  globalState: Record<string, unknown>;
  streamedOutput?: Record<string, unknown> | null;
}

function copyToClipboard(data: unknown) {
  navigator.clipboard.writeText(JSON.stringify(data, null, 2));
}

export function StateInspector({
  selectedNodeId,
  nodeOutputs,
  workflow,
  globalState,
  streamedOutput,
}: StateInspectorProps) {
  const nodeMap = useMemo(() => new Map(workflow.nodes.map((n) => [n.id, n])), [
    workflow.nodes,
  ]);
  const selectedMeta = selectedNodeId ? nodeMap.get(selectedNodeId) : null;
  const selectedOutput = selectedNodeId ? nodeOutputs.get(selectedNodeId) : null;

  return (
    <div className="border-t bg-white">
      <div className="flex items-center justify-between px-4 py-2 border-b">
        <div className="text-sm font-medium text-gray-700">State Inspector</div>
        {selectedNodeId && (
          <div className="text-xs text-gray-500">
            {selectedMeta?.label || selectedNodeId}
          </div>
        )}
      </div>
      <div className="grid grid-cols-2 gap-4 p-4 text-xs">
        <div className="space-y-2">
          {streamedOutput && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold text-gray-600">Live Output (stream)</div>
                <button
                  type="button"
                  onClick={() => copyToClipboard(streamedOutput)}
                  className="text-[10px] text-blue-600 hover:underline"
                >
                  Copy
                </button>
              </div>
              <div className="rounded border border-blue-100 bg-blue-50 p-2 overflow-auto max-h-32">
                <pre className="whitespace-pre-wrap text-gray-800">
                  {JSON.stringify(streamedOutput, null, 2)}
                </pre>
              </div>
            </div>
          )}
          <div className="flex items-center justify-between">
            <div className="text-xs font-semibold text-gray-600">Selected Node Output</div>
            <button
              type="button"
              onClick={() => copyToClipboard(selectedOutput?.output || {})}
              className="text-[10px] text-blue-600 hover:underline"
              disabled={!selectedOutput?.output}
            >
              Copy
            </button>
          </div>
          {!selectedNodeId ? (
            <div className="text-xs text-gray-400">Select a node to view output.</div>
          ) : (
            <div className="rounded border border-gray-200 bg-gray-50 p-2 overflow-auto max-h-44">
              {selectedOutput?.error && (
                <div className="mb-2 text-red-600 whitespace-pre-wrap">
                  {selectedOutput.error}
                </div>
              )}
              <pre className="whitespace-pre-wrap text-gray-800">
                {selectedOutput?.output
                  ? JSON.stringify(selectedOutput.output, null, 2)
                  : '—'}
              </pre>
            </div>
          )}
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="text-xs font-semibold text-gray-600">Global State</div>
            <button
              type="button"
              onClick={() => copyToClipboard(globalState)}
              className="text-[10px] text-blue-600 hover:underline"
            >
              Copy
            </button>
          </div>
          <div className="rounded border border-gray-200 bg-gray-50 p-2 overflow-auto max-h-44">
            <pre className="whitespace-pre-wrap text-gray-800">
              {Object.keys(globalState).length > 0
                ? JSON.stringify(globalState, null, 2)
                : '—'}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
