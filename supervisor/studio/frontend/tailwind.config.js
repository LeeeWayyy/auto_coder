/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Status colors for node execution states
        status: {
          pending: '#6b7280', // gray-500
          ready: '#3b82f6', // blue-500
          running: '#f59e0b', // amber-500
          completed: '#10b981', // emerald-500
          failed: '#ef4444', // red-500
          skipped: '#9ca3af', // gray-400
        },
        // Node type colors
        node: {
          task: '#3b82f6', // blue-500
          gate: '#8b5cf6', // violet-500
          branch: '#f59e0b', // amber-500
          merge: '#10b981', // emerald-500
          parallel: '#06b6d4', // cyan-500
          human: '#ec4899', // pink-500
          subgraph: '#6366f1', // indigo-500
        },
      },
    },
  },
  plugins: [],
};
