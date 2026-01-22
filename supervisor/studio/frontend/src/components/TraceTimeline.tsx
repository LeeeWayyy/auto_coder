/**
 * TraceTimeline component for live execution events.
 */

import { useEffect, useMemo, useRef } from 'react';
import type { TraceEvent } from '../types/workflow';

export interface TraceTimelineProps {
  events: TraceEvent[];
  selectedNodeId: string | null;
  onNodeSelect: (nodeId: string) => void;
  isLive: boolean;
}

function formatRelativeTime(timestamp: string): string {
  const ts = Date.parse(timestamp);
  if (Number.isNaN(ts)) return 'just now';
  const diffMs = Date.now() - ts;
  const diffSec = Math.max(0, Math.floor(diffMs / 1000));
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

const statusColors: Record<string, string> = {
  pending: 'bg-gray-100 text-gray-600',
  ready: 'bg-blue-100 text-blue-700',
  running: 'bg-amber-100 text-amber-700',
  completed: 'bg-emerald-100 text-emerald-700',
  failed: 'bg-red-100 text-red-700',
  skipped: 'bg-gray-100 text-gray-600',
};

export function TraceTimeline({
  events,
  selectedNodeId,
  onNodeSelect,
  isLive,
}: TraceTimelineProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sortedEvents = useMemo(() => events, [events]);

  useEffect(() => {
    if (!isLive) return;
    if (!containerRef.current) return;
    containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [isLive, sortedEvents]);

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-2 border-b text-xs font-semibold text-gray-600 flex items-center justify-between">
        <span>Trace Timeline</span>
        {isLive && (
          <span className="flex items-center gap-1 text-[10px] text-emerald-600">
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
            Live
          </span>
        )}
      </div>
      <div className="px-3 py-2 text-[11px] text-gray-500 border-b">
        Live events only
      </div>
      <div ref={containerRef} className="flex-1 overflow-y-auto">
        {sortedEvents.length === 0 ? (
          <div className="p-4 text-sm text-gray-500 text-center">
            <div className="font-medium">No events yet</div>
            <div className="text-xs mt-1">
              Events appear here during live execution.
              {isLive ? ' Waiting for updates...' : ' Refresh to reconnect.'}
            </div>
            <div className="text-xs mt-2 text-gray-400">
              For persisted data, see Node Outputs below.
            </div>
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {sortedEvents.map((event) => {
              const isSelected = event.nodeId === selectedNodeId;
              return (
                <button
                  key={event.id}
                  type="button"
                  onClick={() => onNodeSelect(event.nodeId)}
                  className={`w-full text-left rounded border px-2 py-2 text-xs transition ${
                    isSelected
                      ? 'border-blue-300 bg-blue-50'
                      : 'border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="font-medium text-gray-700 truncate">
                      {event.nodeLabel}
                    </div>
                    <div className="text-[10px] text-gray-400">
                      {formatRelativeTime(event.timestamp)}
                    </div>
                  </div>
                  <div className="mt-1 flex items-center gap-2">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                        statusColors[event.status] || 'bg-gray-100 text-gray-600'
                      }`}
                    >
                      {event.status}
                    </span>
                    <span className="text-[10px] text-gray-400">{event.nodeType}</span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
