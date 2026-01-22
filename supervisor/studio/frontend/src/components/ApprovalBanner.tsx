/**
 * ApprovalBanner component for human-in-the-loop execution.
 */

import { useState } from 'react';

export interface ApprovalBannerProps {
  nodeId: string;
  title: string;
  description?: string;
  currentOutput?: Record<string, unknown>;
  onApprove: () => void;
  onReject: (reason: string) => void;
  onEdit: () => void;
  isSubmitting: boolean;
}

export function ApprovalBanner({
  nodeId,
  title,
  description,
  onApprove,
  onReject,
  onEdit,
  isSubmitting,
}: ApprovalBannerProps) {
  const [showReject, setShowReject] = useState(false);
  const [rejectReason, setRejectReason] = useState('');

  return (
    <div className="sticky top-0 z-10 border-b border-amber-200 bg-amber-50">
      <div className="px-4 py-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-amber-800">Approval Required</div>
          <div className="text-xs text-amber-700">{title}</div>
          {description && (
            <div className="text-xs text-amber-700 mt-1">{description}</div>
          )}
          <div className="text-[11px] text-amber-600 mt-1">Node: {nodeId}</div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onApprove}
            disabled={isSubmitting}
            className="px-3 py-1.5 text-xs font-semibold text-white bg-emerald-600 rounded hover:bg-emerald-700 disabled:opacity-50"
          >
            Approve
          </button>
          <button
            type="button"
            onClick={() => setShowReject((prev) => !prev)}
            disabled={isSubmitting}
            className="px-3 py-1.5 text-xs font-semibold text-white bg-red-600 rounded hover:bg-red-700 disabled:opacity-50"
          >
            Reject
          </button>
          <button
            type="button"
            onClick={onEdit}
            disabled={isSubmitting}
            className="px-3 py-1.5 text-xs font-semibold text-white bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50"
          >
            Edit & Submit
          </button>
        </div>
      </div>
      {showReject && (
        <div className="px-4 pb-3">
          <label className="block text-xs text-amber-700 mb-1">Rejection Reason</label>
          <textarea
            value={rejectReason}
            onChange={(e) => setRejectReason(e.target.value)}
            className="w-full rounded border border-amber-200 px-2 py-1 text-xs"
            rows={2}
            placeholder="Explain why this should be rejected"
          />
          <div className="mt-2 flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                onReject(rejectReason.trim());
                setRejectReason('');
                setShowReject(false);
              }}
              disabled={isSubmitting}
              className="px-3 py-1 text-xs font-semibold text-white bg-red-600 rounded hover:bg-red-700 disabled:opacity-50"
            >
              Submit Reject
            </button>
            <button
              type="button"
              onClick={() => setShowReject(false)}
              className="px-3 py-1 text-xs font-semibold text-amber-700 bg-amber-100 rounded hover:bg-amber-200"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
