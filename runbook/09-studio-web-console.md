# Studio Web Console

Supervisor Studio is the web console for visual workflow management (graph editing, execution, and live monitoring).

## Prerequisites

- Python with Supervisor installed
- `uvicorn` available (`pip install uvicorn`)
- Node.js + npm for frontend development (only needed for dev builds)

## Start the Backend

```bash
# Bind to localhost only (recommended)
supervisor studio

# Custom port
supervisor studio --port 8000

# Dev auto-reload (backend only)
supervisor studio --reload
```

The backend serves:
- API: `http://127.0.0.1:8000/api`
- WebSocket: `ws://127.0.0.1:8000/ws/...`

On first run, the CLI will:
- create the `supervisor-egress` Docker network if missing
- prompt to build missing sandbox images (CLI + executor)
- verify the CLI binaries inside the sandbox image

To auto-build without prompts:
```bash
SUPERVISOR_AUTO_BUILD_IMAGES=1 supervisor studio
```

To skip the CLI binary check (not recommended):
```bash
SUPERVISOR_SKIP_CLI_CHECKS=1 supervisor studio
```

## Frontend: Dev Mode

```bash
cd supervisor/studio/frontend
npm install
npm run dev
```

Vite runs on `http://127.0.0.1:5173` and proxies `/api` and `/ws` to the backend.

## Frontend: Production Build

```bash
cd supervisor/studio/frontend
npm install
npm run build
```

The build outputs to `supervisor/studio/frontend/dist`. When that folder exists,
`supervisor studio` serves it as a static SPA.

## Security Notes

- Studio has **no authentication**.
- Bind to localhost only. If you bind to `0.0.0.0`, you are exposing the UI.
- The backend enforces a restrictive CORS allowlist for localhost origins.

## Run Inputs

The Run panel supports:
- **Goal**: a short description of what to achieve (`input_data.goal`)
- **JSON Inputs**: additional input data (must be a JSON object)
- **Run Label**: optional label for the execution. If omitted, Studio auto-generates
  a unique label for each run (helps avoid circuit-breaker blocks on retries).

## Host CLI Mode (Subscription Login)

If you need subscription login, run the CLI on the host instead of Docker:

```bash
SUPERVISOR_USE_HOST_CLI=1 supervisor studio
```

This bypasses Docker sandboxing. Use only on trusted machines.

## Real-Time Updates Limitation

WebSocket updates are **in-memory**. Only executions started by the same Studio
server instance receive live updates. Executions started elsewhere will show in
lists but won’t stream live updates.
