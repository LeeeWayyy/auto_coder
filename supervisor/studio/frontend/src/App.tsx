/**
 * Main App component with routing and providers.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
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

          {/* Main content */}
          <main className="flex-1 overflow-hidden">
            <Routes>
              <Route path="/" element={<WorkflowListPage />} />
              <Route path="/workflows/new" element={<WorkflowEditorPage />} />
              <Route path="/workflows/:id" element={<WorkflowEditorPage />} />
              <Route path="/workflows/:id/run" element={<WorkflowEditorPage />} />
              <Route path="/executions/:id" element={<ExecutionPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
