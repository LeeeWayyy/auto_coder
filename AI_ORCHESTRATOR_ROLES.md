# AI Orchestrator: Dynamic Roles & Generalization Strategy

## 1. Dynamic Role System Architecture
**Recommendation:** "Base + Overlay" Inheritance Model.

We define roles not as hardcoded enums, but as **Configurable Personas** defined by YAML/JSON.

### Structure
*   **Base Roles (Core):** `Planner`, `Implementer`, `Reviewer`. These handle the fundamental loop (Think -> Do -> Check).
*   **Domain Overlays (Mixin):** These are "skill packs" injected into Base Roles.
    *   *Example:* A `Trader` is an `Implementer` + `Finance Knowledge Base` + `Risk Constraints`.

### Role Definition Schema (YAML)
```yaml
# .supervisor/roles/trader.yaml
extends: implementer
description: "Handles execution of trading logic with strict risk adherence."
system_prompt_additions: |
  You are a quantitative developer.
  CRITICAL: You must check 'risk_limits.py' before placing orders.
  NEVER remove stop-loss checks.
allowed_tools:
  - read_file
  - run_backtest_simulation  # Custom tool
context_rules:
  always_include:
    - "config/risk_limits.py"
    - "core/order_execution.py"
```

## 2. Project Configuration Hierarchy
**Recommendation:** Cascading Configuration (Global -> Project -> Task).

### Directory Structure
```text
project_root/
├── .supervisor/
│   ├── config.yaml       # Project-specific overrides
│   ├── roles/            # Custom role definitions (domain specific)
│   │   ├── trader.yaml
│   │   └── data_eng.yaml
│   └── templates/        # Jinja2 prompt templates
└── ...
```

### Inheritance Logic
1.  **Global (`~/.supervisor/config.yaml`):** Default LLM models, timeout settings, base system prompts.
2.  **Project (`./.supervisor/config.yaml`):** Overrides models (e.g., "Use Claude for everything"), defines project domain.
3.  **Task (Runtime):** User says `run --role=junior_dev`, overriding specific params for one run.

## 3. Role Discovery (Auto-Detection)
**Recommendation:** Heuristic Suggestions + Explicit Confirmation.

Don't auto-configure silently. Run a "Scan" step that suggests a setup.

**Heuristics Logic:**
*   `package.json` has `react`/`vue` → Suggest **Frontend Roles** (`ui_designer`, `component_builder`).
*   `requirements.txt` has `pandas`/`airflow` → Suggest **Data Roles** (`etl_architect`).
*   `docker-compose.yml` has `postgres`/`redis` → Suggest **Backend Roles** (`db_optimizer`).
*   `*.sol` files → Suggest **Smart Contract Roles** (`auditor`).

**Workflow:**
`supervisor init` → Scans files → "I detected a React + Python project. Enable 'Web App' and 'Backend' role packs? [Y/n]"

## 4. Addressing Open Questions

### Streaming vs. Complete Output
*   **Decision:** **Stream to UI, Buffer for Logic.**
*   **Why:** The user needs to see progress (Streaming). The Orchestrator needs the full JSON/Text to parse next steps (Buffering).
*   **Implementation:** The subprocess pipe reads line-by-line, prints to stdout (for user), and appends to a `buffer` string. Once process exits, parse `buffer`.

### Context Overlap (The "Stale Reviewer" Problem)
*   **Strategy:** **Differential Context Injection.**
*   **Mechanism:**
    1.  Orchestrator snapshots file hashes *before* Implementer runs.
    2.  Implementer finishes. Orchestrator detects changed files.
    3.  **Reviewer Context:** automatically gets:
        *   `git diff` of the changes.
        *   The *full new content* of changed files.
        *   The *original plan* (to verify intent).
    *   *Note:* The Reviewer does *not* need the whole repo, just the "Blast Radius".

### Error Classification
*   **Strategy:** **Regex Classification + Retry Counter.**
*   *Category A: Worker Confused (Retry)*
    *   **Triggers:** Invalid JSON output, Markdown formatting errors, Tool arguments malformed.
    *   **Action:** Feed error message back to worker. "You formatted JSON wrong. Fix it." (Max 3 retries).
*   *Category B: Task Impossible (Escalate)*
    *   **Triggers:** "File not found" (after search), "API Authentication Failed", "ImportError" (missing library).
    *   **Action:** Pause execution. Ask Human: "Worker needs 'pandas' but it's not installed. Install it? [y/n]"

## 5. Domain-Specific Role Matrix

| Domain | Role Name | Specialization (Prompt/Context) | Key Files/Tools |
| :--- | :--- | :--- | :--- |
| **Trading** | `RiskManager` | Validates logic against capital preservation rules. Checks for race conditions. | `risk.py`, `oms.py` |
| **Trading** | `QuantDev` | Focuses on vectorization (numpy) and minimizing latency. | `strategy.py`, `backtest/` |
| **Web App** | `UXDesigner` | Focuses on accessibility, CSS frameworks (Tailwind), and component state. | `tailwind.config.js`, `components/` |
| **Web App** | `StateArchitect`| Manages data flow (Redux/Zustand/Context). | `store/`, `hooks/` |
| **CLI Tool** | `ArgParser` | Focuses on DX (Developer Experience), help flags, and error messages. | `main.py`, `args.py` |
| **ETL/Data** | `SchemaGuard` | Validates data types, handles NULLs, ensures idempotency. | `models.py`, `migrations/` |
| **Mobile** | `PlatformSpec` | Knows iOS vs Android nuances (Permissions, UI paradigms). | `Info.plist`, `AndroidManifest.xml` |
