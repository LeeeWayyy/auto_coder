/**
 * Main App component with routing and providers.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ReactFlowProvider } from '@xyflow/react';
import { WorkflowListPage, WorkflowEditorPage, ExecutionPage } from './pages';

// Create a client with reasonable defaults
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 10000,
    },
  },
});

export function App() {
  const hostCliEnabled =
    typeof window !== 'undefined' &&
    ((window as unknown as { __SUPERVISOR_HOST_CLI__?: boolean }).__SUPERVISOR_HOST_CLI__ ?? false);

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="h-screen flex flex-col bg-gray-100">
          {/* Header */}
          <header className="bg-white shadow-sm border-b">
            <div className="px-4 py-3">
              <h1 className="text-xl font-bold text-gray-900">
                Supervisor Studio
              </h1>
            </div>
          </header>

          {hostCliEnabled && (
            <div className="bg-amber-100 border-b border-amber-200 text-amber-900 text-sm px-4 py-2">
              Host CLI mode is enabled. AI CLIs run UNSANDBOXED on the host.
            </div>
          )}

          {/* Main content */}
          <main className="flex-1 overflow-hidden">
            <ReactFlowProvider>
              <Routes>
                <Route path="/" element={<WorkflowListPage />} />
                <Route path="/workflows/new" element={<WorkflowEditorPage />} />
                <Route path="/workflows/:id" element={<WorkflowEditorPage />} />
                <Route path="/workflows/:id/run" element={<WorkflowEditorPage />} />
                <Route path="/executions/:id" element={<ExecutionPage />} />
              </Routes>
            </ReactFlowProvider>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
