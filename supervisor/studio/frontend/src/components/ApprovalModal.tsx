/**
 * ApprovalModal component for editing human-in-the-loop output.
 */

import { useEffect, useState } from 'react';

export interface ApprovalModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentOutput: Record<string, unknown>;
  onSubmit: (editedData: Record<string, unknown>) => void;
  isSubmitting: boolean;
}

export function ApprovalModal({
  isOpen,
  onClose,
  currentOutput,
  onSubmit,
  isSubmitting,
}: ApprovalModalProps) {
  const [text, setText] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setText(JSON.stringify(currentOutput || {}, null, 2));
      setError(null);
    }
  }, [isOpen, currentOutput]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div className="w-full max-w-2xl rounded bg-white shadow-lg">
        <div className="flex items-center justify-between px-4 py-2 border-b">
          <div className="text-sm font-semibold text-gray-700">Edit Output</div>
          <button
            type="button"
            onClick={onClose}
            className="text-xs text-gray-500 hover:text-gray-700"
          >
            Close
          </button>
        </div>
        <div className="p-4 space-y-3">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full h-64 rounded border border-gray-200 px-2 py-1 text-xs font-mono"
          />
          {error && <div className="text-xs text-red-600">{error}</div>}
          <div className="flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 text-xs font-semibold text-gray-700 bg-gray-100 rounded hover:bg-gray-200"
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => {
                try {
                  const parsed = JSON.parse(text) as Record<string, unknown>;
                  setError(null);
                  onSubmit(parsed);
                } catch (err) {
                  const message = err instanceof Error ? err.message : 'Invalid JSON';
                  setError(message);
                }
              }}
              className="px-3 py-1.5 text-xs font-semibold text-white bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Submitting...' : 'Submit'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
