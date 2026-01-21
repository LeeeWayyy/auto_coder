# Supervisor Studio (Web Console)

**Backend:** `supervisor/studio/server.py`  
**Frontend:** `supervisor/studio/frontend/`

## Overview

Supervisor Studio is the web UI for visual workflow management: graph editing, execution, and live monitoring.

## Backend (FastAPI)

Capabilities:
- Workflow CRUD (`/api/workflows`)
- Execution control (`/api/execute`, `/api/executions`, `/api/executions/{id}`)
- Node status queries (`/api/executions/{id}/nodes`)
- WebSocket updates (`/ws/executions/{id}`)
- Static SPA hosting if `frontend/dist` exists

Security properties:
- Localhost-only intent (no authentication)
- CORS allowlist for local dev origins
- Origin/host validation for HTTP and WebSocket

Limitations:
- WebSocket updates are in-memory callbacks; only executions started by the same
  server instance receive live updates.

## Frontend (Vite + React)

- Dev server on `http://127.0.0.1:5173`
- Proxies `/api` and `/ws` to backend (see `vite.config.ts`)
- Production build output in `frontend/dist` served by backend

## CLI Entry Point

`supervisor studio --host 127.0.0.1 --port 8000` starts the backend and serves
static assets if built.
