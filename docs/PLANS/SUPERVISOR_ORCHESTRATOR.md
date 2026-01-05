# Supervisor/Orchestrator System Plan

**Status:** Draft
**Created:** 2025-01-03
**Last Updated:** 2025-01-03
**Contributors:** Claude, Gemini, Codex (collaborative design)

## Problem Statement

When using AI CLI tools (Claude Code, Codex, Gemini CLI) for long development workflows, **context dilution** causes the AI to progressively ignore instructions. The larger the context window grows, the less reliably the AI follows the original rules and constraints.

**Root Cause:** AI CLIs maintain conversation state, and as context accumulates, earlier instructions lose priority.

**Solution:** Treat AI CLIs as **Stateless Workers** - spin up fresh instances per task step, feed only relevant context, get output, terminate.

---

## Core Philosophy

> For every step of your workflow, spin up a fresh short-lived instance, feed it only the relevant rules/context, get the output, and kill it.

The orchestrator acts as an "Operating System" for the development process, managing:
- State transitions with checkpointing
- Context packing per role
- Worker invocation (stateless subprocess)
- Gate enforcement (orchestrator verifies, not workers)
- Error recovery with retry policies

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER                                     │
│                          │                                       │
│                          ▼                                       │
│                   ┌──────────────┐                               │
│                   │  Supervisor  │                               │
│                   │     CLI      │                               │
│                   └──────┬───────┘                               │
│                          │                                       │
│         ┌────────────────┼────────────────┐                      │
│         ▼                ▼                ▼                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│   │  State   │    │  Config  │    │  Prompts │                   │
│   │ (SQLite) │    │  (YAML)  │    │ (Jinja2) │                   │
│   └──────────┘    └──────────┘    └──────────┘                   │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                           │
│              │   Execution Engine    │                           │
│              ├───────────────────────┤                           │
│              │ 1. Pack Context       │◄── Repomix                │
│              │ 2. Build Prompt       │◄── Jinja2 Templates       │
│              │ 3. Run Worker         │◄── Subprocess Wrapper     │
│              │ 4. Parse Output       │◄── Structured Extraction  │
│              │ 5. Verify Gates       │◄── Orchestrator-Enforced  │
│              └───────────┬───────────┘                           │
│                          │                                       │
│         ┌────────────────┼────────────────┐                      │
│         ▼                ▼                ▼                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│   │  Claude  │    │  Codex   │    │  Gemini  │                   │
│   │   CLI    │    │   CLI    │    │   CLI    │                   │
│   └──────────┘    └──────────┘    └──────────┘                   │
│                          │                                       │
│                          ▼                                       │
│                   ┌──────────────┐                               │
│                   │    State     │                               │
│                   │   Update     │                               │
│                   └──────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Patterns (Inspired by LangGraph & MetaGPT)

### From LangGraph: Durable Execution & Checkpointing

**Key Insight:** Checkpoint at every step so workflows can resume from exactly where they left off.

```python
# Checkpoint after every node execution
class CheckpointingExecutor:
    """LangGraph-inspired checkpointing for CLI orchestration."""

    def execute_step(self, step: Step, state: WorkflowState) -> WorkflowState:
        # 1. Load checkpoint if resuming
        checkpoint = self.load_checkpoint(state.workflow_id, step.id)
        if checkpoint and checkpoint.status == 'completed':
            return checkpoint.result_state  # Skip already-completed steps

        # 2. Execute the step
        try:
            result = self.run_worker(step, state)
            new_state = state.apply(result)

            # 3. Checkpoint immediately after success
            self.save_checkpoint(state.workflow_id, step.id, new_state, status='completed')
            return new_state

        except Exception as e:
            # 4. Checkpoint failure state for later retry
            self.save_checkpoint(state.workflow_id, step.id, state, status='failed', error=str(e))
            raise
```

**Small Nodes Principle:** Smaller steps = more frequent checkpoints = less work to repeat on failure.

### From LangGraph: Retry Policies with Backoff

```python
from dataclasses import dataclass
from typing import Callable
import random
import time

@dataclass
class RetryPolicy:
    """LangGraph-style retry policy configuration."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.1  # Add randomness to prevent thundering herd
    retryable_exceptions: tuple = (TimeoutError, ConnectionError)

    def execute_with_retry(self, func: Callable, *args, **kwargs):
        delay = self.initial_delay
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.retryable_exceptions as e:
                last_exception = e
                if attempt == self.max_attempts - 1:
                    raise

                # Add jitter to delay
                actual_delay = delay + random.uniform(-self.jitter * delay, self.jitter * delay)
                time.sleep(actual_delay)
                delay = min(delay * self.backoff_multiplier, self.max_delay)

        raise last_exception
```

### From LangGraph: Human-in-the-Loop via Interrupt

```python
class InterruptibleWorkflow:
    """LangGraph-inspired interrupt/resume pattern."""

    def execute_with_approval(self, step: Step, state: WorkflowState):
        # Check if step requires human approval
        if step.requires_approval:
            # Save state and interrupt
            self.save_checkpoint(state, status='awaiting_approval')

            # Raise interrupt - workflow pauses here
            raise HumanApprovalRequired(
                step_id=step.id,
                message=f"Approval required for: {step.description}",
                resume_command=f"supervisor resume {state.workflow_id}"
            )

    def resume(self, workflow_id: str, approval: Approval):
        """Resume from interrupt with human decision."""
        checkpoint = self.load_checkpoint(workflow_id)

        if approval.approved:
            # Continue from where we left off
            return self.continue_from_checkpoint(checkpoint)
        else:
            # Handle rejection
            return self.handle_rejection(checkpoint, approval.reason)
```

### ~~From MetaGPT: Publish-Subscribe Messaging~~ (DISCARDED)

> **REMOVED:** Pub/Sub conflicts with our stateless worker model. Workers are short-lived subprocesses that die after each step - they cannot subscribe to messages. Instead, use SQLite event log + DAG scheduler as the sole communication channel. The orchestrator pulls next steps from the DAG based on completed dependencies.

### From MetaGPT: Schema-Enforced Structured Outputs

**Key Insight:** Agents output structured documents, not free-form text. Downstream agents can reliably parse these outputs.

```python
from pydantic import BaseModel, Field
from typing import Literal

class PlanOutput(BaseModel):
    """Schema for Planner role output - enforced, not optional."""
    status: Literal["COMPLETE", "NEEDS_REFINEMENT", "BLOCKED"]
    phases: list[PhaseDefinition]
    dependencies: list[Dependency]
    estimated_components: int
    risks: list[str] = Field(default_factory=list)

class ImplementationOutput(BaseModel):
    """Schema for Implementer role output."""
    status: Literal["SUCCESS", "PARTIAL", "FAILED"]
    files_created: list[str]
    files_modified: list[str]
    tests_written: list[str]
    blockers: list[str] = Field(default_factory=list)

class ReviewOutput(BaseModel):
    """Schema for Reviewer role output."""
    status: Literal["APPROVED", "CHANGES_REQUESTED", "REJECTED"]
    issues: list[ReviewIssue]
    suggestions: list[str]
    security_concerns: list[str] = Field(default_factory=list)

# Enforce output parsing
def parse_worker_output(raw_output: str, schema: type[BaseModel]) -> BaseModel:
    """Extract and validate structured output from worker response."""
    # Extract JSON block from output
    json_str = extract_json_block(raw_output)

    # Parse and validate against schema
    try:
        return schema.model_validate_json(json_str)
    except ValidationError as e:
        raise WorkerOutputError(f"Output doesn't match schema: {e}")
```

### From MetaGPT: Executable Feedback Loops

**Key Insight:** Don't just review code - execute tests and feed real failures back to the worker.

```python
class ExecutableFeedbackLoop:
    """MetaGPT-inspired: Run tests, feed failures back for fixing."""

    MAX_FIX_ATTEMPTS = 3

    def implement_with_feedback(self, task: Task) -> ImplementationResult:
        for attempt in range(self.MAX_FIX_ATTEMPTS):
            # 1. Run implementer
            impl_result = self.run_role("implementer", task)

            # 2. Execute tests (orchestrator runs, not worker)
            test_result = subprocess.run(
                ["pytest", "-v", "--tb=short"],
                capture_output=True, text=True
            )

            if test_result.returncode == 0:
                return impl_result  # Success!

            # 3. Feed failure back to implementer
            task.context["test_failure"] = {
                "attempt": attempt + 1,
                "stdout": test_result.stdout,
                "stderr": test_result.stderr,
                "instruction": "Fix the failing tests. Focus on the error messages above."
            }

        # Max attempts exhausted
        raise ImplementationFailed(f"Could not fix tests after {self.MAX_FIX_ATTEMPTS} attempts")
```

### From MetaGPT: Dependency-Based Activation

**Key Insight:** An agent only activates when ALL its prerequisites are met.

```python
class DependencyGate:
    """Agent activation only when prerequisites are complete."""

    def can_activate(self, component_id: str) -> tuple[bool, list[str]]:
        """Check if component's dependencies are all satisfied."""
        component = self.db.get_component(component_id)
        missing = []

        for dep_id in component.depends_on:
            dep = self.db.get_component(dep_id)
            if dep.status != 'completed':
                missing.append(dep_id)

        return len(missing) == 0, missing

    def get_ready_components(self) -> list[Component]:
        """Get all components whose dependencies are satisfied."""
        pending = self.db.get_components_by_status('pending')
        ready = []

        for comp in pending:
            can_run, _ = self.can_activate(comp.id)
            if can_run:
                ready.append(comp)

        return ready
```

---

## Repository Structure

```
supervisor/                          # The orchestrator package (installed globally or per-project)
├── config/
│   ├── base_roles/                  # Built-in base roles (shipped with supervisor)
│   │   ├── planner.yaml
│   │   ├── implementer.yaml
│   │   └── reviewer.yaml
│   ├── context_strategies/          # Pluggable context selection strategies
│   │   ├── planner_docset.yaml
│   │   ├── implementer_targeted.yaml
│   │   ├── reviewer_diff.yaml
│   │   └── security_audit.yaml
│   ├── workflow.yaml                # Default workflow state machine
│   ├── limits.yaml                  # Budgets, timeouts, retry limits
│   └── role_schema.json             # JSON Schema for role validation
├── core/
│   ├── __init__.py
│   ├── llm_client.py                # Subprocess wrapper with ANSI strip, timeout
│   ├── context.py                   # Dynamic Repomix context packing
│   ├── state.py                     # SQLite + event sourcing
│   ├── gates.py                     # Orchestrator-enforced verification
│   ├── parser.py                    # Structured output extraction
│   ├── role_loader.py               # Role plugin system with inheritance
│   └── project_detector.py          # Auto-detect project type for role suggestions
├── prompts/
│   ├── _base.j2                     # Common output format requirements
│   ├── _output_schema.j2            # Mandatory JSON output block
│   ├── planning.j2
│   ├── implement.j2
│   └── review_strict.j2
├── sandbox/
│   └── Dockerfile
├── tests/
├── supervisor.py                    # Main CLI entry point
├── tui.py                           # Human interrupt interface
└── requirements.txt

# Per-project configuration (in target project repo)
project_root/
├── .supervisor/
│   ├── config.yaml                  # Project-specific overrides
│   ├── roles/                       # Custom domain roles
│   │   ├── trader.yaml              # Trading-specific role
│   │   ├── risk_manager.yaml
│   │   └── quant_dev.yaml
│   ├── templates/                   # Custom Jinja2 prompts
│   └── workflow.yaml                # Project-specific workflow (optional)
└── ...

# Global user configuration
~/.supervisor/
├── config.yaml                      # User defaults (preferred CLI, API keys)
└── roles/                           # User's shared custom roles
```

---

## Key Components

### 1. LLM Client (Subprocess Wrapper)

**File:** `core/llm_client.py`

Wraps CLI invocations with:
- ANSI escape code stripping
- Timeout handling
- Error capture and classification
- Retry logic with exponential backoff

```python
class LLMClient:
    """Subprocess wrapper for AI CLI tools."""

    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def __init__(self, executable: str, flags: list[str], timeout: int = 300):
        self.executable = executable
        self.flags = flags
        self.timeout = timeout

    def generate(self, prompt: str) -> dict:
        """Execute CLI and return structured result."""
        ...
```

**CLI Invocation Flags:**

| CLI | Non-Interactive | JSON Output | Notes |
|-----|-----------------|-------------|-------|
| Claude Code | `claude -p "..."` | `--output-format json` | Use `--dangerously-skip-permissions` for automation |
| Codex | `codex exec` | `--json` | JSONL event stream |
| Gemini | `gemini` | `-o json` | Use `--yolo` for auto-approve |

### 2. State Management (SQLite + Event Sourcing)

**File:** `core/state.py`

Why SQLite over JSON:
- ACID compliance (no corruption on crash)
- Event log for debugging/replay
- Atomic transactions with rollback

**Schema:**
```sql
-- Current workflow state
CREATE TABLE workflow (
    id INTEGER PRIMARY KEY,
    current_step TEXT,
    component TEXT,
    status TEXT,  -- 'active', 'completed', 'failed'
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Event log (immutable)
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    workflow_id INTEGER,
    event_type TEXT,  -- 'transition', 'review', 'error', 'retry', 'gate'
    role TEXT,
    status TEXT,
    payload JSON,
    cost_estimate REAL,
    timestamp TIMESTAMP
);

-- Checkpoints for recovery
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY,
    workflow_id INTEGER,
    git_sha TEXT,
    context_snapshot TEXT,
    timestamp TIMESTAMP
);
```

### 3. Context Packing (Role-Based)

**File:** `core/context.py`

Dynamic context selection based on role:

| Role | Context Includes | Token Budget |
|------|------------------|--------------|
| Planner | README, file tree, task docs, ADRs | 30k |
| Implementer | Target file + imports + git diff | 25k |
| Reviewer | Staged changes + standards docs | 15k |
| UI Designer | Frontend code + style guide | 20k |

**Algorithm:**
1. Read role config → get include patterns
2. For dynamic roles (implementer), resolve imports/dependencies
3. Run Repomix with computed patterns
4. Enforce token budget (prioritize: system > task > target > related)

### 4. Orchestrator-Enforced Gates

**File:** `core/gates.py`

**Critical principle:** Never trust worker output. The orchestrator verifies.

**IMPORTANT:** All gates run INSIDE the sandbox container, not on the host.

```python
class WorkflowGates:
    """All verification runs inside sandboxed containers."""

    def __init__(self, sandbox: SandboxedExecutor):
        self.sandbox = sandbox

    def verify_tests(self, worktree_path: str) -> tuple[bool, str]:
        """Orchestrator runs tests inside sandbox, not on host."""
        result = self.sandbox.run(
            command=["make", "test"],
            workdir=worktree_path
        )
        return result.returncode == 0, result.stdout + result.stderr

    def verify_lint(self, worktree_path: str) -> tuple[bool, str]:
        """Lint check inside sandbox."""
        result = self.sandbox.run(
            command=["make", "lint"],
            workdir=worktree_path
        )
        return result.returncode == 0, result.stdout

    def verify_review(self, raw_output: str) -> ReviewOutput:
        """Parse and validate review output - NO MAGIC STRING FALLBACK."""
        json_str = extract_json_block(raw_output)
        if not json_str:
            raise InvalidOutputError("No JSON block found - review rejected")

        # Strict schema validation only
        return ReviewOutput.model_validate_json(json_str)
```

**Gate Sequence:**
```
implement → [GATE: tests pass in sandbox] → review → [GATE: JSON approved] → [GATE: CI pass] → commit
```

### 5. Structured Output Parsing

**File:** `core/parser.py`

All role templates require a JSON output block. **No fallback to marker detection.**

```json
{
  "status": "SUCCESS" | "NEEDS_REVISION" | "BLOCKED",
  "action_taken": "description",
  "files_modified": ["path/to/file.py"],
  "review_status": "APPROVED" | "CHANGES_REQUESTED" | null,
  "blockers": ["blocker 1"] | null,
  "next_step": "suggested action"
}
```

```python
def parse_worker_output(raw_output: str, schema: type[BaseModel]) -> BaseModel:
    """Extract and validate - STRICT MODE, NO FALLBACK."""
    json_str = extract_json_block(raw_output)
    if not json_str:
        raise ParsingError("No JSON block found in worker output")

    try:
        return schema.model_validate_json(json_str)
    except ValidationError as e:
        raise ParsingError(f"Output doesn't match schema: {e}")
```

---

## Dynamic Role System

### Philosophy: Base + Overlay Model

Roles are **Configurable Personas** defined in YAML, not hardcoded. The system uses inheritance:

- **Base Roles (Core):** `planner`, `implementer`, `reviewer` - fundamental development loop
- **Domain Overlays:** "Skill packs" that extend base roles with domain knowledge
  - Example: `trader` = `implementer` + finance knowledge + risk constraints

### Role Definition Schema

```yaml
# .supervisor/roles/trader.yaml
extends: implementer                    # Inherits from base role
description: "Trading logic with strict risk adherence"

system_prompt_additions: |
  You are a quantitative developer.
  CRITICAL: Check 'risk_limits.py' before placing orders.
  NEVER remove stop-loss checks.
  All order sizes must respect position limits.

context:
  always_include:                       # Added to parent's context
    - "config/risk_limits.py"
    - "core/order_execution.py"
    - "libs/risk/**"
  exclude:
    - "**/backtest/**"                  # Don't include historical data

allowed_tools:                          # Restrict available tools
  - read_file
  - edit_file
  - run_backtest_simulation             # Custom tool

gates:                                  # Additional gates for this role
  - name: risk_check
    command: "python -m risk.validate"
    on_fail: block
```

### Configuration Hierarchy (Cascading)

```
Priority (highest to lowest):
┌─────────────────────────────────────┐
│ 1. Task Runtime (--role=junior_dev) │  ← CLI override
├─────────────────────────────────────┤
│ 2. Project (.supervisor/config.yaml)│  ← Per-project settings
├─────────────────────────────────────┤
│ 3. User (~/.supervisor/config.yaml) │  ← User preferences
├─────────────────────────────────────┤
│ 4. Package (supervisor/config/)     │  ← Built-in defaults
└─────────────────────────────────────┘
```

**Merge Semantics:**

| Field Type | Merge Strategy |
|------------|----------------|
| Scalars (cli, timeout) | Override (child wins) |
| Lists (include, exclude) | Append (parent + child) |
| Dicts (context) | Deep merge |
| system_prompt | Concatenate (parent + additions) |

### Role Plugin System

**File:** `core/role_loader.py`

```python
class RoleLoader:
    """Load and merge role definitions with inheritance."""

    SEARCH_PATHS = [
        Path(".supervisor/roles"),      # Project-specific
        Path.home() / ".supervisor/roles",  # User-global
        PACKAGE_DIR / "config/base_roles",  # Built-in
    ]

    def load_role(self, name: str) -> RoleConfig:
        """Load role with inheritance resolution."""
        role_file = self._find_role_file(name)
        config = yaml.safe_load(role_file.read_text())

        if "extends" in config:
            parent = self.load_role(config["extends"])
            config = self._merge_configs(parent, config)

        self._validate_schema(config)
        return RoleConfig(**config)

    def _merge_configs(self, parent: dict, child: dict) -> dict:
        """Apply merge semantics per field type."""
        ...
```

### Project Auto-Detection

**File:** `core/project_detector.py`

On `supervisor init`, scan project files and suggest relevant role packs:

| Detection | Suggested Roles |
|-----------|-----------------|
| `package.json` + `react`/`vue` | `ui_designer`, `component_builder`, `state_architect` |
| `requirements.txt` + `pandas`/`airflow` | `data_engineer`, `schema_guard`, `etl_architect` |
| `docker-compose.yml` + `postgres` | `db_optimizer`, `backend_dev` |
| `*.sol` files | `smart_contract_auditor` |
| `Cargo.toml` | `rust_dev`, `memory_safety_reviewer` |
| `go.mod` | `go_dev`, `concurrency_reviewer` |

**Workflow:**
```bash
$ supervisor init
Scanning project...
Detected: Python + React + PostgreSQL

Suggested role packs:
  [x] Backend API (backend_dev, db_optimizer)
  [x] Frontend (ui_designer, component_builder)
  [ ] Data Pipeline (disabled - no pandas/airflow detected)

Enable selected packs? [Y/n]
```

### Domain-Specific Role Examples

#### Trading/Finance Platform

```yaml
# .supervisor/roles/risk_manager.yaml
extends: reviewer
description: "Validates trading logic against risk rules"

system_prompt_additions: |
  You are a risk management specialist.
  REJECT any code that:
  - Removes or weakens stop-loss logic
  - Exceeds position size limits
  - Bypasses circuit breaker checks
  - Uses market orders without slippage protection

context:
  always_include:
    - "config/risk_limits.py"
    - "core/circuit_breaker.py"

gates:
  - name: backtest_risk
    command: "python -m backtest.risk_scenarios"
```

```yaml
# .supervisor/roles/quant_dev.yaml
extends: implementer
description: "Quantitative strategy development"

system_prompt_additions: |
  Focus on:
  - Vectorized operations (numpy/pandas)
  - Minimizing latency in hot paths
  - Proper handling of market data edge cases (gaps, halts)

context:
  always_include:
    - "libs/strategy/**"
    - "libs/signals/**"
```

#### Web Application

```yaml
# .supervisor/roles/ui_designer.yaml
extends: implementer
description: "Frontend UI/UX implementation"

cli: "claude"
system_prompt_additions: |
  You are a UI/UX specialist.
  Focus on:
  - Accessibility (WCAG compliance)
  - Responsive design
  - Component reusability
  - State management patterns

context:
  include:
    - "src/components/**"
    - "src/styles/**"
    - "docs/STANDARDS/UI.md"
  exclude:
    - "**/*.test.*"
    - "**/api/**"
```

```yaml
# .supervisor/roles/state_architect.yaml
extends: reviewer
description: "Reviews data flow and state management"

system_prompt_additions: |
  Focus on:
  - Redux/Zustand/Context usage patterns
  - Avoiding prop drilling
  - Proper memoization
  - Race condition prevention in async state
```

#### CLI Tool Development

```yaml
# .supervisor/roles/cli_ux_dev.yaml
extends: implementer
description: "CLI user experience specialist"

system_prompt_additions: |
  Focus on Developer Experience (DX):
  - Clear, helpful error messages
  - Intuitive argument naming
  - Comprehensive --help output
  - Progress indicators for long operations
  - Consistent exit codes
```

#### Data Pipeline / ETL

```yaml
# .supervisor/roles/schema_guard.yaml
extends: reviewer
description: "Data schema and quality validation"

system_prompt_additions: |
  Validate:
  - Schema compatibility (no breaking changes)
  - NULL handling in all paths
  - Idempotency of transformations
  - Proper partitioning strategies
```

```yaml
# .supervisor/roles/etl_architect.yaml
extends: planner
description: "ETL pipeline design"

context:
  always_include:
    - "dags/**"
    - "models/**"
    - "migrations/**"
```

#### Backend API Development

```yaml
# .supervisor/roles/api_dev.yaml
extends: implementer
description: "REST/GraphQL API development"

system_prompt_additions: |
  Follow:
  - RESTful conventions (proper HTTP methods, status codes)
  - Input validation at boundaries
  - Consistent error response format
  - Rate limiting awareness
```

#### Security Auditing

```yaml
# .supervisor/roles/security_auditor.yaml
extends: reviewer
description: "Security-focused code review"

cli: "claude"  # Best for security reasoning
system_prompt_additions: |
  OWASP Top 10 focus:
  - Injection vulnerabilities (SQL, command, XSS)
  - Authentication/authorization flaws
  - Sensitive data exposure
  - Security misconfiguration

context:
  always_include:
    - "**/*auth*"
    - "**/*security*"
    - "**/middleware/**"
    - "requirements.txt"
    - "package-lock.json"
```

#### Infrastructure / DevOps

```yaml
# .supervisor/roles/infra_reviewer.yaml
extends: reviewer
description: "Infrastructure and deployment review"

system_prompt_additions: |
  Check:
  - No hardcoded secrets
  - Proper resource limits
  - Health check endpoints
  - Graceful shutdown handling
  - Rollback strategies

context:
  always_include:
    - "docker-compose*.yml"
    - "Dockerfile*"
    - "k8s/**"
    - ".github/workflows/**"
```

---

## Pluggable Context Strategies

### Strategy Interface

```yaml
# config/context_strategies/implementer_targeted.yaml
name: implementer_targeted
description: "Target file + resolved imports"

inputs:
  - target_file        # Required: file being modified
  - git_diff           # Optional: recent changes

algorithm:
  1_resolve_imports:
    command: "python -m supervisor.tools.import_resolver ${target_file}"
    output: import_list

  2_pack_context:
    repomix:
      include:
        - "${target_file}"
        - "${import_list}"
      style: xml

  3_add_diff:
    if: git_diff
    command: "git diff --cached"
    prepend: true

token_budget: 25000
priority_order:
  - system_prompt
  - task_description
  - target_file
  - imports
  - git_diff
```

### Built-in Strategies

| Strategy | Use Case | What It Packs |
|----------|----------|---------------|
| `planner_docset` | Planning phase | README, architecture docs, task specs |
| `implementer_targeted` | Coding | Target file + imports + git diff |
| `reviewer_diff` | Code review | Staged changes + full new files + standards |
| `security_audit` | Security review | Auth code, deps, env handling |
| `investigator_wide` | Exploration | Broad codebase scan (high token budget) |

---

## Hierarchical Workflow Design

### The Problem: Flat vs Hierarchical

Large features have natural hierarchy that flat workflows can't express:

```
Feature: User Authentication System (P1T5)
├── Phase 1: Backend Infrastructure
│   ├── Component A: User Model & Database Schema
│   ├── Component B: Password Hashing Service
│   └── Component C: JWT Token Service
├── Phase 2: API Endpoints
│   ├── Component D: Registration Endpoint (depends on A, B)
│   ├── Component E: Login Endpoint (depends on A, B, C)
│   └── Component F: Token Refresh Endpoint (depends on C)
├── Phase 3: Frontend
│   ├── Component G: Login Form (depends on E)
│   └── Component H: Registration Form (depends on D)
└── Phase 4: Integration Testing
    └── Component I: E2E Auth Flow Tests (depends on all)
```

### Three-Tier Execution Model

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: ARCHITECT (Feature Planning)                           │
│ Role: planner                                                   │
│ Output: Feature breakdown → Phases + high-level dependencies    │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: TECH LEAD (Phase Planning)                             │
│ Role: tech_lead                                                 │
│ Output: Phase breakdown → Components + dependency graph + APIs  │
│ CRITICAL: Define interfaces BEFORE implementation               │
├─────────────────────────────────────────────────────────────────┤
│ Level 3: DEVELOPER (Component Implementation)                   │
│ Roles: implementer, test_engineer                               │
│ Output: Working code + tests for single component               │
└─────────────────────────────────────────────────────────────────┘
```

### Interface-First Locking (Critical for Parallelism)

**Problem:** If Backend and Frontend implement in parallel, they may drift (different API shapes, types, etc.)

**Solution:** Level 2 (Tech Lead) must define interfaces BEFORE parallel implementation:

```yaml
# Phase planning output includes interface definitions
phase_output:
  components:
    - id: user_model
      files: ["models/user.py"]
    - id: login_endpoint
      files: ["api/auth.py"]
      depends_on: [user_model]
    - id: login_form
      files: ["components/LoginForm.tsx"]
      depends_on: [login_endpoint]

  # CRITICAL: Lock these before parallel work
  interfaces:
    - name: LoginRequest
      type: typescript_interface
      definition: |
        interface LoginRequest {
          email: string;
          password: string;
        }
    - name: LoginResponse
      type: typescript_interface
      definition: |
        interface LoginResponse {
          token: string;
          user: { id: string; email: string; };
          expiresAt: string;
        }
    - name: login_endpoint
      type: openapi
      definition: |
        /api/auth/login:
          post:
            requestBody: { $ref: '#/components/schemas/LoginRequest' }
            responses:
              200: { $ref: '#/components/schemas/LoginResponse' }
```

### Hierarchical State Schema

Extend SQLite schema to track hierarchy:

```sql
-- Feature (top-level task)
CREATE TABLE features (
    id TEXT PRIMARY KEY,           -- "P1T5"
    title TEXT,
    status TEXT,                   -- 'planning', 'in_progress', 'review', 'complete'
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Phase (groups related components)
CREATE TABLE phases (
    id TEXT PRIMARY KEY,           -- "P1T5-PH1"
    feature_id TEXT REFERENCES features(id),
    title TEXT,
    sequence INTEGER,              -- Execution order
    status TEXT,
    interfaces JSON,               -- Locked interface definitions
    created_at TIMESTAMP
);

-- Component (atomic unit of work)
CREATE TABLE components (
    id TEXT PRIMARY KEY,           -- "P1T5-PH1-C1"
    phase_id TEXT REFERENCES phases(id),
    title TEXT,
    files JSON,                    -- Target files
    depends_on JSON,               -- ["P1T5-PH1-C0"]
    status TEXT,                   -- 'pending', 'implementing', 'testing', 'review', 'complete'
    assigned_role TEXT,
    created_at TIMESTAMP
);

-- Dependency graph edges (for DAG scheduling)
CREATE TABLE dependencies (
    component_id TEXT REFERENCES components(id),
    depends_on_id TEXT REFERENCES components(id),
    PRIMARY KEY (component_id, depends_on_id)
);
```

### DAG Scheduler Algorithm

```python
class DAGScheduler:
    """Execute components respecting dependencies AND phase ordering."""

    def __init__(self, db: Database):
        self.db = db
        self.graph = nx.DiGraph()

    def build_graph(self, feature_id: str):
        """Build dependency graph from components with phase constraints."""
        phases = self.db.get_phases(feature_id)
        components = self.db.get_components(feature_id)

        for comp in components:
            self.graph.add_node(comp.id, data=comp)

            # Explicit component dependencies
            for dep_id in comp.depends_on:
                self.graph.add_edge(dep_id, comp.id)

        # PHASE SEQUENCING: Add implicit edges between phases
        # Components in Phase N+1 depend on ALL components in Phase N
        sorted_phases = sorted(phases, key=lambda p: p.sequence)
        for i in range(1, len(sorted_phases)):
            prev_phase = sorted_phases[i - 1]
            curr_phase = sorted_phases[i]

            prev_components = [c for c in components if c.phase_id == prev_phase.id]
            curr_components = [c for c in components if c.phase_id == curr_phase.id]

            # Every component in current phase depends on all previous phase components
            for curr_comp in curr_components:
                for prev_comp in prev_components:
                    if not self.graph.has_edge(prev_comp.id, curr_comp.id):
                        self.graph.add_edge(prev_comp.id, curr_comp.id)

        # Validate no cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise CyclicDependencyError(f"Dependency cycle detected: {cycles}")

    def get_ready_components(self) -> list[Component]:
        """Get components whose dependencies AND phase gates are satisfied."""
        ready = []
        for node_id in self.graph.nodes:
            comp = self.graph.nodes[node_id]['data']
            if comp.status != 'pending':
                continue

            # Check all dependencies complete (includes phase deps now)
            deps_complete = all(
                self.graph.nodes[dep]['data'].status == 'complete'
                for dep in self.graph.predecessors(node_id)
            )
            if deps_complete:
                ready.append(comp)

        return ready

    def execute_feature(self, feature_id: str):
        """Execute all components in dependency order."""
        self.build_graph(feature_id)

        while not self._all_complete():
            ready = self.get_ready_components()

            if not ready:
                blocked = self._get_blocked_components()
                raise WorkflowBlockedError(f"No ready components. Blocked: {blocked}")

            # Execute ready components (potentially in parallel)
            if self.config.parallel_execution:
                self._execute_parallel(ready)
            else:
                for comp in ready:
                    self._execute_component(comp)
```

### Status Rollup Logic

```python
def compute_phase_status(phase_id: str) -> str:
    """Rollup component statuses to phase status."""
    components = db.get_components_for_phase(phase_id)
    statuses = [c.status for c in components]

    if all(s == 'complete' for s in statuses):
        return 'complete'
    elif any(s == 'failed' for s in statuses):
        return 'failed'
    elif any(s in ('implementing', 'testing', 'review') for s in statuses):
        return 'in_progress'
    else:
        return 'pending'

def compute_feature_status(feature_id: str) -> str:
    """Rollup phase statuses to feature status."""
    phases = db.get_phases_for_feature(feature_id)
    # Similar logic...
```

---

## Test Engineering Strategy

### The Tradeoff

| Approach | Pros | Cons |
|----------|------|------|
| **A) Implementer writes all tests** | Fast, context preserved, TDD natural | May miss edge cases, biased toward happy path |
| **B) Separate test_engineer** | Fresh perspective, catches blind spots | Extra step, needs context transfer |
| **C) Hybrid** | Best of both worlds | More complex workflow |

### Recommended: Hybrid Model

```
┌─────────────────────────────────────────────────────────────────┐
│ IMPLEMENTER (writes code + unit tests)                          │
│ - TDD style: write test → implement → verify                    │
│ - Coverage target: 80% for unit tests                           │
│ - Focus: Function-level correctness                             │
├─────────────────────────────────────────────────────────────────┤
│                              ↓                                   │
├─────────────────────────────────────────────────────────────────┤
│ TEST_ENGINEER (writes integration + e2e tests)                  │
│ - Runs AFTER implementation complete                            │
│ - Focus: Component interactions, API contracts, edge cases      │
│ - Fresh context: only sees interfaces + implementation          │
└─────────────────────────────────────────────────────────────────┘
```

### Test Engineer Role Definition

```yaml
# config/base_roles/test_engineer.yaml
name: test_engineer
description: "Integration and E2E test specialist"
extends: implementer

cli: "claude"  # Best for understanding complex interactions

system_prompt_additions: |
  You are a QA Engineer writing integration and E2E tests.

  Your job is to BREAK the implementation by finding edge cases.

  Focus on:
  - API contract violations
  - Race conditions and timing issues
  - Error handling paths
  - Boundary conditions
  - Security edge cases (auth bypass, injection)

  You have NOT seen the implementation code.
  You only know the interfaces and expected behavior.
  This gives you a fresh perspective.

  REQUIRED: Each test must have a clear description of what it's testing and why.

context:
  include:
    - "$INTERFACE_DEFINITIONS"   # API specs, type definitions
    - "$COMPONENT_PUBLIC_API"    # Public functions/methods only
    - "tests/conftest.py"        # Test fixtures
    - "docs/STANDARDS/TESTING.md"
  exclude:
    - "$IMPLEMENTATION_INTERNALS"  # Don't see private functions

gates:
  - name: coverage_check
    command: "pytest --cov --cov-fail-under=70"
    on_fail: retry

config:
  test_types:
    - integration    # Component interaction tests
    - e2e            # Full flow tests
    - contract       # API contract tests
  min_tests_per_component: 3
```

### Workflow with Test Engineering

```yaml
# config/workflow_templates/with_test_engineer.yaml
workflow:
  steps:
    # ... planning steps ...

    implement:
      role: implementer
      config:
        tdd: true
        unit_test_coverage: 80
      next: unit_test_gate
      outputs: ["code", "unit_tests"]

    unit_test_gate:
      type: orchestrator_gate
      command: "pytest tests/unit/ --cov --cov-fail-under=80"
      next: integration_tests
      on_fail: implement
      max_retries: 3

    integration_tests:
      role: test_engineer
      config:
        test_types: ["integration", "contract"]
      next: integration_test_gate
      outputs: ["integration_tests"]

    integration_test_gate:
      type: orchestrator_gate
      command: "pytest tests/integration/"
      next: review
      on_fail: implement  # Implementation bug found
      max_retries: 2

    # ... review and commit steps ...
```

---

## Workflow Templates

### Template System Design

```yaml
# config/workflow_templates/_base.yaml
# All templates inherit from this
name: _base
description: "Base workflow template"

steps:
  plan:
    role: planner
    required: true

  implement:
    role: implementer
    required: true

  commit:
    type: orchestrator_gate
    command: "git commit"
    required: true

hooks:
  pre_commit:
    - lint
    - type_check
```

### MVP Template (Startup/Prototype)

```yaml
# config/workflow_templates/mvp.yaml
name: mvp
description: "Fast iteration for prototypes and MVPs"
extends: _base

# Minimal gates for speed
settings:
  parallel_reviews: false
  require_tests: false
  human_approval: false

steps:
  plan:
    role: planner
    next: implement

  implement:
    role: implementer_fast  # Use Codex for speed
    config:
      tdd: false
      skip_edge_cases: true
    next: quick_review

  quick_review:
    role: reviewer_codex  # Fast syntax review only
    next: commit
    gate: structured_json_only  # No magic strings, must return valid ReviewOutput
    # NOTE: auto_approve removed - conflicts with security-first approach
    # Use deterministic signals only: zero diffs + passing gates + lint
    on_reject: implement

  commit:
    type: orchestrator_gate
```

### Enterprise Template (Full Gates)

```yaml
# config/workflow_templates/enterprise.yaml
name: enterprise
description: "Full quality gates for production code"
extends: _base

settings:
  parallel_reviews: true
  require_tests: true
  human_approval: true
  security_review: true

steps:
  plan:
    role: planner
    next: plan_review

  plan_review:
    role: reviewer_gemini
    next: interface_definition
    gate: "REVIEW_STATUS: APPROVED"
    human_approval: true  # Require human sign-off on architecture

  interface_definition:
    role: tech_lead
    next: implement
    outputs: ["interfaces", "dependency_graph"]
    human_approval: true  # Lock interfaces before work begins

  implement:
    role: implementer
    config:
      tdd: true
      coverage: 90
    retry_policy:
      max_attempts: 3
      on_failure: "ask_human"
    context_scope: "component_only"
    next: unit_test_gate

  unit_test_gate:
    type: orchestrator_gate
    command: "make test-unit"
    next: integration_tests
    on_fail: implement

  integration_tests:
    role: test_engineer
    next: integration_test_gate

  integration_test_gate:
    type: orchestrator_gate
    command: "make test-integration"
    next: security_review
    on_fail: implement

  security_review:
    role: security_auditor
    next: code_review
    gate: "SECURITY_STATUS: PASSED"
    on_reject: implement

  code_review:
    parallel: true
    roles: [reviewer_gemini, reviewer_codex]
    next: ci_gate
    gate: "BOTH_APPROVED"
    human_approval: true  # Final human review
    on_reject: implement

  ci_gate:
    type: orchestrator_gate
    command: "make ci-full"
    next: commit
    on_fail: implement

  commit:
    type: orchestrator_gate
    human_approval: true  # Confirm before commit
```

### Open Source Template

```yaml
# config/workflow_templates/open_source.yaml
name: open_source
description: "Community-friendly with PR focus"
extends: _base

settings:
  auto_pr: true
  require_changelog: true

steps:
  plan:
    role: planner
    next: implement
    outputs: ["plan", "changelog_entry"]

  implement:
    role: implementer
    config:
      tdd: true
      add_docstrings: true  # OSS needs good docs
    next: test_gate

  test_gate:
    type: orchestrator_gate
    command: "make test"
    next: docs_check

  docs_check:
    type: orchestrator_gate
    command: "make docs-check"  # Verify docstrings, README updates
    next: create_pr

  create_pr:
    type: orchestrator_gate
    command: |
      gh pr create \
        --title "${PR_TITLE}" \
        --body "${PR_BODY}" \
        --draft
    next: await_review
    outputs: ["pr_url"]

  await_review:
    type: human_gate
    description: "Waiting for community review"
    # Workflow pauses here until human triggers next step
```

### Security-Critical Template

```yaml
# config/workflow_templates/security_critical.yaml
name: security_critical
description: "Maximum scrutiny for security-sensitive code"
extends: enterprise

settings:
  dual_review: true
  audit_trail: true

steps:
  # Inherits enterprise steps, adds:

  threat_modeling:
    role: security_auditor
    position: after plan_review
    next: interface_definition
    outputs: ["threat_model", "security_requirements"]

  security_review:
    # Override: require TWO independent security reviews
    parallel: true
    roles: [security_auditor, security_auditor_external]
    gate: "BOTH_PASSED"
    human_approval: true

  penetration_test:
    role: security_auditor
    position: after integration_test_gate
    config:
      test_types: ["injection", "auth_bypass", "data_exposure"]
    next: security_review
```

### Template Composition

```yaml
# .supervisor/config.yaml (in project)
workflow:
  template: enterprise

  # Override specific steps
  overrides:
    implement:
      role: quant_dev  # Use domain-specific implementer
      config:
        tdd: true
        coverage: 95  # Higher coverage for trading code

    # Add custom step
    risk_validation:
      position: after integration_test_gate
      role: risk_manager
      next: security_review

  # Disable steps not needed
  disable:
    - security_review  # Already covered by risk_validation
```

---

## Dependency Graph & Resolution

### Explicit Declaration

```yaml
# Feature definition with dependencies
feature:
  id: P1T5
  title: "User Authentication System"

  phases:
    - id: backend
      sequence: 1
      components:
        - id: user_model
          files: ["models/user.py", "migrations/001_users.py"]
          depends_on: []

        - id: auth_service
          files: ["services/auth.py"]
          depends_on: [user_model]

        - id: auth_endpoints
          files: ["api/auth.py"]
          depends_on: [user_model, auth_service]

    - id: frontend
      sequence: 2  # Can start after backend interfaces defined
      components:
        - id: login_form
          files: ["components/Login.tsx"]
          depends_on: [backend.auth_endpoints]  # Cross-phase dependency
```

### Auto-Detection (Optional)

```python
class DependencyDetector:
    """Auto-detect dependencies from imports."""

    def detect(self, component_files: list[str]) -> list[str]:
        """Analyze imports and suggest dependencies."""
        dependencies = set()

        for file_path in component_files:
            imports = self._extract_imports(file_path)

            for imp in imports:
                # Map import to component
                owning_component = self._find_component_for_module(imp)
                if owning_component:
                    dependencies.add(owning_component)

        return list(dependencies)

    def _extract_imports(self, file_path: str) -> list[str]:
        """Parse Python/TypeScript imports."""
        # Use AST for Python, regex for TypeScript
        ...
```

**Policy:** Auto-detected dependencies are suggestions, require confirmation:

```bash
$ supervisor analyze-deps P1T5

Detected dependencies:
  auth_endpoints → user_model (from: "from models.user import User")
  auth_endpoints → auth_service (from: "from services.auth import AuthService")
  login_form → auth_endpoints (from: "import { login } from '@/api/auth'")

  [!] Unconfirmed:
  auth_service → redis_client (from: "import redis")
  → Is redis_client a component or external package? [component/external/skip]

Accept detected dependencies? [Y/n/edit]
```

---

## Role Configuration

**File:** `config/roles.yaml`

```yaml
roles:
  planner:
    description: "Architecture and task planning"
    cli: "claude"
    flags: ["-p", "--output-format", "json"]
    template: "planning.j2"
    context:
      include: ["README.md", "docs/TASKS/**", "docs/ADRs/**"]
      token_budget: 30000
    strengths: ["architecture", "edge cases", "refactoring"]

  implementer:
    description: "Code implementation"
    cli: "claude"
    flags: ["-p", "--output-format", "json"]
    template: "implement.j2"
    context:
      include: ["$TARGET", "$TARGET_IMPORTS"]  # Dynamic
      git_diff: true
      token_budget: 25000

  implementer_fast:
    description: "Quick isolated implementations"
    cli: "codex"
    flags: ["exec", "--json"]
    template: "implement.j2"
    context:
      include: ["$TARGET"]
      token_budget: 15000
    strengths: ["boilerplate", "single-function tasks"]

  reviewer_gemini:
    description: "Code review (Gemini)"
    cli: "gemini"
    flags: ["-o", "json", "--yolo"]
    template: "review_strict.j2"
    context:
      include: ["git diff --cached", "docs/STANDARDS/**"]
      token_budget: 15000

  reviewer_codex:
    description: "Code review (Codex)"
    cli: "codex"
    flags: ["exec", "--json"]
    template: "review_strict.j2"
    context:
      include: ["git diff --cached"]
      token_budget: 15000

  ui_designer:
    description: "NiceGUI interface design"
    cli: "claude"
    flags: ["-p", "--output-format", "json"]
    template: "ui_design.j2"
    context:
      include: ["apps/frontend/**/*.py", "docs/STANDARDS/UI.md"]
      exclude: ["**/tests/**"]
      token_budget: 20000
```

---

## Workflow State Machine

**File:** `config/workflow.yaml`

```yaml
workflow:
  initial_step: plan

  steps:
    plan:
      role: planner
      next: plan_review
      outputs: ["task_breakdown", "file_targets"]

    plan_review:
      role: reviewer_gemini
      next: implement
      gate: "REVIEW_STATUS: APPROVED"
      on_reject: plan

    implement:
      role: implementer
      next: test
      outputs: ["files_modified"]

    test:
      gate_type: orchestrator  # Orchestrator runs this, not worker
      command: "make test"
      next: review
      on_fail: implement
      max_retries: 3

    review:
      parallel: true  # Run both reviewers simultaneously
      roles: [reviewer_gemini, reviewer_codex]
      next: ci
      gate: "BOTH_APPROVED"
      on_reject: implement

    ci:
      gate_type: orchestrator
      command: "make ci-local"
      next: commit
      on_fail: implement

    commit:
      gate_type: orchestrator
      command: "git commit"
      terminal: true
```

---

## Multi-Model Strategy

| Model | Best For | Use As |
|-------|----------|--------|
| **Claude** | Complex reasoning, planning, subtle bugs, refactoring | Planner, Primary Implementer, Architecture Reviewer |
| **Gemini** | Large context analysis, pattern finding, documentation | Investigator, QA Reviewer, Doc Generator |
| **Codex** | Fast isolated tasks, boilerplate, syntax-heavy work | Fast Implementer, Syntax Reviewer |

**Parallel Review Pattern:**
```python
# Run both reviewers simultaneously
with ThreadPoolExecutor(max_workers=2) as executor:
    gemini_future = executor.submit(run_role, "reviewer_gemini", context)
    codex_future = executor.submit(run_role, "reviewer_codex", context)

    gemini_result = gemini_future.result()
    codex_result = codex_future.result()

    # Both must approve
    approved = (
        gemini_result["review_status"] == "APPROVED" and
        codex_result["review_status"] == "APPROVED"
    )
```

---

## Safety & Limits

**File:** `config/limits.yaml`

```yaml
limits:
  # Per-step limits
  step_timeout_seconds: 300
  step_max_retries: 3

  # Workflow limits
  workflow_max_steps: 50

  # Token budgets (per invocation) - for context sizing, not cost
  default_token_budget: 20000
  max_token_budget: 50000

  # Retry backoff
  retry_base_delay_seconds: 5
  retry_max_delay_seconds: 60
  retry_exponential_base: 2
```

---

## Missing Components (To Address)

| Component | Priority | Description |
|-----------|----------|-------------|
| **Sandboxing** | High | Run worker-suggested commands in Docker/firejail |
| **Circuit Breaker** | High | Prevent infinite retry loops (see error handling section) |
| **Human Interrupt TUI** | Medium | Approve/reject/edit interface for stuck workflows |
| **Checkpoint Recovery** | Medium | Resume from last successful step after crash |
| **Metrics Dashboard** | Low | Success rates, avg steps, retry rates per workflow |

---

## Implementation Phases

### Phase 1: Foundation (MVP) - SECURITY FIRST
- [ ] Project setup (pyproject.toml, folder structure)
- [ ] **Docker sandboxing with egress allowlist** (CRITICAL - no execution without this)
- [ ] SQLite state management with event sourcing (not JSON)
- [ ] `SandboxedLLMClient` - all CLI/command execution inside containers
- [ ] Git worktree isolation for workspace safety
- [ ] One role working end-to-end (planner)

### Phase 2: Core Workflow
- [ ] Role configuration loading (YAML) with schema validation
- [ ] Jinja2 prompt templating
- [ ] Context packing with Repomix integration
- [ ] Structured output parsing (JSON-only, no marker fallback)

### Phase 3: Gates & Verification
- [ ] Orchestrator-enforced gates (test, lint) - **run inside sandbox**
- [ ] Retry logic with feedback loop
- [ ] Circuit breaker integration

### Phase 4: Multi-Model & Hierarchy
- [ ] All three CLIs integrated (Claude, Codex, Gemini)
- [ ] Parallel review execution
- [ ] Role-specific context strategies
- [ ] Hierarchical workflow (Feature→Phase→Component)
- [ ] DAG scheduler with phase sequencing enforcement

### Phase 5: Polish & Advanced
- [ ] Timeout handling
- [ ] Human interrupt interface (TUI)
- [ ] Metrics dashboard
- [ ] Adaptive role assignment

---

## Resolved Design Decisions

### 1. Streaming Output

**Decision:** Stream to UI, Buffer for Logic.

- **User sees:** Real-time progress (streaming stdout)
- **Orchestrator uses:** Complete buffered output for parsing

```python
def execute_worker(self, prompt: str) -> WorkerResult:
    process = subprocess.Popen(
        [self.cli, "-p", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    buffer = []
    for line in process.stdout:
        print(line, end="")  # Stream to user
        buffer.append(line)  # Buffer for parsing

    process.wait()
    full_output = "".join(buffer)
    return self.parser.extract(full_output)
```

### 2. Context Overlap (The "Stale Reviewer" Problem)

**Decision:** Differential Context Injection.

The orchestrator tracks file state before/after each step:

```python
def run_implement_then_review(self, task: Task):
    # 1. Snapshot before implementation
    pre_hashes = self.git.get_file_hashes()

    # 2. Run implementer
    impl_result = self.run_role("implementer", task)

    # 3. Detect what changed
    post_hashes = self.git.get_file_hashes()
    changed_files = [f for f in post_hashes if post_hashes[f] != pre_hashes.get(f)]

    # 4. Build reviewer context with "blast radius"
    reviewer_context = {
        "git_diff": self.git.diff_staged(),
        "changed_files_full": [read_file(f) for f in changed_files],
        "original_plan": task.plan,  # To verify intent
    }

    # 5. Run reviewer with fresh, accurate context
    return self.run_role("reviewer", task, extra_context=reviewer_context)
```

### 3. Conversation Continuity

**Decision:** Bounded Continuations for Complex Tasks.

- **Default:** Fresh instance per step (stateless)
- **Exception:** Allow up to 3 continuation turns for complex multi-part tasks

```yaml
# config/limits.yaml
continuity:
  default_mode: fresh          # Always fresh instance
  allow_continuation: true     # Enable for specific roles
  max_continuation_turns: 3    # Hard limit
  continuation_roles:          # Only these roles can continue
    - planner                  # Complex planning may need follow-up
    - investigator             # Deep exploration benefits from memory
```

**When to use continuation:**
- Planner needs to refine based on feasibility feedback
- Investigation requires drilling down into findings
- NOT for implementation (should be atomic)

### 4. Error Handling & Recovery (LangChain/LangGraph-Inspired)

**Multi-Level Error Handling:** Handle errors at node, graph, and application levels.

#### Error Categories (6 Types from LangChain)

```python
from enum import Enum, auto

class ErrorCategory(Enum):
    """Six categories of errors in CLI orchestration."""
    NETWORK = auto()      # Timeouts, DNS, connection failures
    CLI_ERROR = auto()    # Non-zero exit codes, CLI crashes
    VALIDATION = auto()   # Output doesn't match schema
    PARSING = auto()      # Can't extract structured data from output
    LOGIC = auto()        # Worker produced wrong/incomplete result
    RESOURCE = auto()     # Out of memory, disk full, etc.

class ErrorClassifier:
    """Classify errors for appropriate handling."""

    # Retryable with same context
    TRANSIENT_PATTERNS = {
        ErrorCategory.NETWORK: [
            r"timed? ?out",
            r"connection (refused|reset|closed)",
            r"temporary failure",
            r"EAGAIN|ECONNRESET",
        ],
        ErrorCategory.CLI_ERROR: [
            r"rate limit",
            r"overloaded",
            r"try again",
        ],
    }

    # Retryable with modified context/prompt
    FIXABLE_PATTERNS = {
        ErrorCategory.VALIDATION: [
            r"Invalid JSON",
            r"validation error",
            r"missing required field",
        ],
        ErrorCategory.PARSING: [
            r"could not extract",
            r"no JSON found",
            r"malformed output",
        ],
    }

    # Not retryable - escalate immediately
    FATAL_PATTERNS = {
        ErrorCategory.LOGIC: [
            r"BLOCKED:",
            r"cannot proceed",
            r"impossible to",
        ],
        ErrorCategory.RESOURCE: [
            r"out of memory",
            r"disk full",
            r"quota exceeded",
        ],
        ErrorCategory.CLI_ERROR: [
            r"permission denied",
            r"file not found",
            r"authentication failed",
        ],
    }

    def classify(self, error: str) -> tuple[ErrorCategory, ErrorAction]:
        error_lower = error.lower()

        # Check fatal first
        for category, patterns in self.FATAL_PATTERNS.items():
            if any(re.search(p, error_lower) for p in patterns):
                return category, ErrorAction.ESCALATE

        # Check fixable (retry with modified context)
        for category, patterns in self.FIXABLE_PATTERNS.items():
            if any(re.search(p, error_lower) for p in patterns):
                return category, ErrorAction.RETRY_WITH_FEEDBACK

        # Check transient (retry same request)
        for category, patterns in self.TRANSIENT_PATTERNS.items():
            if any(re.search(p, error_lower) for p in patterns):
                return category, ErrorAction.RETRY_SAME

        # Unknown - retry once then escalate
        return ErrorCategory.LOGIC, ErrorAction.RETRY_ONCE
```

#### Retry Policy Configuration

```yaml
# config/retry_policies.yaml
policies:
  default:
    max_attempts: 3
    initial_delay_seconds: 2
    backoff_multiplier: 2.0
    max_delay_seconds: 30
    jitter: 0.1

  aggressive:  # For critical paths
    max_attempts: 5
    initial_delay_seconds: 1
    backoff_multiplier: 1.5
    max_delay_seconds: 60

  conservative:  # For expensive operations
    max_attempts: 2
    initial_delay_seconds: 5
    backoff_multiplier: 3.0
    max_delay_seconds: 120

# Per-step retry policy assignment
steps:
  plan:
    retry_policy: default
  implement:
    retry_policy: aggressive  # More attempts for complex work
  review:
    retry_policy: conservative  # Reviews are expensive
```

#### Feedback Loop for Fixable Errors

```python
class FeedbackRetryHandler:
    """Retry with contextual feedback for fixable errors."""

    FEEDBACK_TEMPLATES = {
        ErrorCategory.VALIDATION: """
Your previous output failed schema validation:
{error}

Please ensure your response ends with a valid JSON block matching this schema:
{schema}
""",
        ErrorCategory.PARSING: """
Could not extract structured output from your response.

REQUIRED: End your response with a JSON code block:
```json
{example}
```
""",
        ErrorCategory.LOGIC: """
Your implementation has issues:
{error}

Test output:
{test_output}

Please fix and try again. If you cannot proceed, respond with:
BLOCKED: <reason>
""",
    }

    def retry_with_feedback(
        self,
        role: str,
        task: Task,
        error: str,
        category: ErrorCategory,
        attempt: int
    ) -> WorkerResult:
        template = self.FEEDBACK_TEMPLATES.get(category, "Error: {error}")
        feedback = template.format(
            error=error,
            schema=task.output_schema,
            example=task.output_example,
            test_output=getattr(task, 'test_output', ''),
            attempt=attempt
        )

        task.context["feedback"] = feedback
        task.context["attempt"] = attempt
        return self.run_role(role, task)
```

#### Circuit Breaker (Prevent Infinite Loops)

```python
class CircuitBreaker:
    """Prevent runaway error loops (LangGraph best practice)."""

    def __init__(self, max_failures: int = 5, reset_timeout: int = 300):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures: dict[str, list[float]] = {}  # step_id -> timestamps

    def record_failure(self, step_id: str):
        now = time.time()
        if step_id not in self.failures:
            self.failures[step_id] = []

        # Clean old failures outside reset window
        self.failures[step_id] = [
            t for t in self.failures[step_id]
            if now - t < self.reset_timeout
        ]
        self.failures[step_id].append(now)

    def is_open(self, step_id: str) -> bool:
        """Returns True if circuit is open (should not retry)."""
        if step_id not in self.failures:
            return False
        return len(self.failures[step_id]) >= self.max_failures

    def execute_with_breaker(self, step_id: str, func: Callable) -> Any:
        if self.is_open(step_id):
            raise CircuitOpenError(
                f"Circuit breaker open for {step_id}. "
                f"Too many failures in the last {self.reset_timeout}s. "
                f"Manual intervention required."
            )

        try:
            return func()
        except Exception as e:
            self.record_failure(step_id)
            raise
```

#### Graceful Degradation

```python
class GracefulDegradation:
    """Fallback strategies when primary approach fails."""

    def execute_with_fallback(self, task: Task) -> WorkerResult:
        strategies = [
            ("primary", self.run_primary),
            ("simplified", self.run_simplified),
            ("minimal", self.run_minimal),
        ]

        last_error = None
        for name, strategy in strategies:
            try:
                result = strategy(task)
                if name != "primary":
                    result.degraded = True
                    result.degradation_level = name
                return result
            except Exception as e:
                last_error = e
                self.log(f"Strategy '{name}' failed: {e}, trying next...")

        raise AllStrategiesFailedError(last_error)

    def run_simplified(self, task: Task) -> WorkerResult:
        """Reduce scope - implement core functionality only."""
        task.context["mode"] = "simplified"
        task.context["instruction"] = (
            "Implement ONLY the core functionality. "
            "Skip edge cases, error handling, and optimizations. "
            "We'll add those in a follow-up task."
        )
        return self.run_role(task.role, task)

    def run_minimal(self, task: Task) -> WorkerResult:
        """Stub implementation - just interfaces."""
        task.context["mode"] = "minimal"
        task.context["instruction"] = (
            "Create stub implementations that satisfy the interface. "
            "Add TODO comments for actual logic. "
            "This is a placeholder for manual implementation."
        )
        return self.run_role(task.role, task)
```

### 5. Git Integration

**Decision:** Opt-in Branch Automation with Dry-Run Preview.

```yaml
# .supervisor/config.yaml
git:
  auto_branch: true              # Create branch on task start
  branch_pattern: "ai/{task_id}-{description}"
  auto_commit: false             # Require explicit commit step
  dry_run_preview: true          # Show what would happen before doing it
```

**Workflow:**
```bash
$ supervisor start-task "Add user authentication"

Git Preview (dry-run):
  Branch: ai/task-001-add-user-authentication
  Base: main (abc123)

Proceed? [Y/n]

$ supervisor complete-task

Git Preview (dry-run):
  Commit: feat(auth): Add user authentication
  Files: src/auth.py, tests/test_auth.py
  Merge target: main

Proceed? [Y/n]
```

**State-to-Git Mapping:**
| Workflow State | Git Action |
|----------------|------------|
| `start_task` | `git checkout -b ai/{task_id}` |
| `implement_complete` | Stage files, ready for review |
| `review_approved` | Commit with message |
| `task_complete` | Merge to base branch (optional) |
| `task_failed` | `git stash` or `git reset` (with confirmation)

---

## Advanced Features (Future Phases)

### 1. Multi-Project Orchestration

**Use Case:** Monorepos or microservices where a feature spans multiple packages.

```yaml
# .supervisor/config.yaml
projects:
  - name: backend-api
    path: ./packages/backend
    roles: [api_dev, db_optimizer]

  - name: frontend-web
    path: ./packages/frontend
    roles: [ui_designer, state_architect]

  - name: shared-types
    path: ./packages/types
    roles: [type_architect]

cross_project_dependencies:
  frontend-web:
    depends_on: [shared-types, backend-api]  # Must build types first
```

**Execution Model:**
1. Build dependency graph across projects
2. Execute in topological order
3. Shared artifacts (types, schemas) are published before dependents start

### 2. Remote Execution

**Use Case:** GPU-intensive tasks, large context analysis, or CI integration.

```yaml
# config/remote.yaml
workers:
  local:
    type: subprocess
    default: true

  cloud_gpu:
    type: ssh
    host: "gpu-worker.example.com"
    user: "supervisor"
    key_file: "~/.ssh/supervisor_key"
    capabilities: ["large_context", "gpu"]

  ci_runner:
    type: github_actions
    workflow: ".github/workflows/supervisor-worker.yml"
    capabilities: ["isolated", "clean_env"]

# Role assignment can specify worker preference
roles:
  investigator:
    preferred_worker: cloud_gpu  # Large context needs more RAM
    fallback_worker: local

  security_auditor:
    preferred_worker: ci_runner  # Run in isolated environment
```

**Artifact Sync:**
```python
class RemoteWorker:
    def execute(self, task: Task) -> Result:
        # 1. Sync required files to remote
        self.rsync_to_remote(task.context_files)

        # 2. Execute CLI on remote
        result = self.ssh_exec(f"supervisor-worker run {task.id}")

        # 3. Sync outputs back
        self.rsync_from_remote(task.output_files)

        return result
```

### 3. Role Marketplace

**Concept:** Community-contributed role definitions, like npm packages.

```bash
# Install community roles
$ supervisor role install trading-roles/risk-manager
$ supervisor role install security-pack/owasp-auditor

# List installed roles
$ supervisor role list
  [built-in] planner, implementer, reviewer
  [project]  quant_dev, trader
  [community] risk-manager@1.2.0, owasp-auditor@2.0.1
```

**Registry Structure:**
```yaml
# Published to supervisor-roles.io or GitHub registry
name: risk-manager
version: 1.2.0
author: trading-community
description: "Risk validation for trading systems"

extends: reviewer
system_prompt_additions: |
  ... (role prompt)

context:
  always_include:
    - "**/risk*.py"
    - "**/limits*.py"

dependencies:
  - name: base-reviewer
    version: ">=1.0.0"
```

### 4. Metrics & Adaptive Role Assignment

**Metrics Collection:**

```sql
-- Track every worker execution
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP,
    role TEXT,
    cli TEXT,               -- "claude", "codex", "gemini"
    task_type TEXT,         -- "implement", "review", "plan"
    success BOOLEAN,
    retry_count INTEGER,
    duration_seconds REAL,
    token_usage INTEGER,
    cost_estimate REAL,
    error_category TEXT     -- NULL if success
);
```

**Dashboard Output:**
```bash
$ supervisor metrics --last 30d

Role Performance (last 30 days):
┌─────────────────┬─────────┬──────────┬───────────┬──────────┐
│ Role            │ Success │ Avg Time │ Avg Cost  │ Retries  │
├─────────────────┼─────────┼──────────┼───────────┼──────────┤
│ planner         │ 94%     │ 45s      │ $0.12     │ 0.3/task │
│ implementer     │ 87%     │ 120s     │ $0.25     │ 1.2/task │
│ reviewer_gemini │ 91%     │ 30s      │ $0.08     │ 0.1/task │
│ reviewer_codex  │ 88%     │ 25s      │ $0.05     │ 0.2/task │
│ test_engineer   │ 82%     │ 90s      │ $0.18     │ 0.8/task │
└─────────────────┴─────────┴──────────┴───────────┴──────────┘

CLI Performance by Task Type:
┌───────────┬────────────┬────────────┬────────────┐
│ Task Type │ Claude     │ Codex      │ Gemini     │
├───────────┼────────────┼────────────┼────────────┤
│ Planning  │ 96% ✓ Best │ 78%        │ 89%        │
│ Implement │ 91%        │ 85%        │ 82%        │
│ Review    │ 88%        │ 90% ✓ Best │ 92% ✓ Best │
│ Testing   │ 84%        │ 79%        │ 81%        │
└───────────┴────────────┴────────────┴────────────┘
```

**Adaptive Assignment:**

```python
class AdaptiveRoleAssigner:
    """Auto-adjust CLI assignments based on historical performance."""

    def __init__(self, db: Database):
        self.db = db
        self.min_samples = 10  # Need enough data before adapting

    def get_best_cli(self, role: str, task_type: str) -> str:
        """Select CLI with best success rate for this role/task."""
        stats = self.db.get_cli_stats(role, task_type, days=30)

        if stats.total_samples < self.min_samples:
            return self._get_default_cli(role)

        # Weighted score: success_rate * 0.7 + (1 - normalized_cost) * 0.3
        scores = {}
        for cli, data in stats.items():
            success_score = data.success_rate * 0.7
            cost_score = (1 - data.normalized_cost) * 0.3
            scores[cli] = success_score + cost_score

        best_cli = max(scores, key=scores.get)

        # Guardrail: don't switch if difference is marginal (<5%)
        default_cli = self._get_default_cli(role)
        if scores[best_cli] - scores.get(default_cli, 0) < 0.05:
            return default_cli

        return best_cli

    def suggest_role_improvements(self) -> list[Suggestion]:
        """Analyze patterns and suggest config changes."""
        suggestions = []

        # Example: Detect high retry rates
        high_retry_roles = self.db.get_roles_with_high_retries(threshold=2.0)
        for role in high_retry_roles:
            suggestions.append(Suggestion(
                role=role,
                issue="High retry rate (>2 per task)",
                recommendation="Consider increasing context budget or simplifying prompts"
            ))

        return suggestions
```

**Config for Adaptive Mode:**

```yaml
# config/limits.yaml
adaptive:
  enabled: true
  min_samples_before_adapt: 10
  adaptation_interval_days: 7
  max_deviation_from_default: 0.2  # Don't stray too far from configured defaults

  # Manual overrides always win
  locked_assignments:
    planner: claude  # Always use Claude for planning, regardless of metrics
```

### 5. Human Approval Gates

**Risk-Based Approval Policy:**

```yaml
# config/approval_policy.yaml
approval_gates:
  # Low risk: auto-proceed
  low_risk:
    conditions:
      - file_count < 3
      - no_security_sensitive_files
      - test_coverage > 80
    action: auto_approve

  # Medium risk: notify, auto-proceed after timeout
  medium_risk:
    conditions:
      - file_count < 10
      - no_breaking_changes
    action: notify_and_timeout
    timeout_minutes: 30
    notification: slack  # or email, or TUI alert

  # High risk: require explicit approval
  high_risk:
    conditions:
      - changes_auth_code
      - changes_payment_code
      - breaking_api_changes
      - file_count > 20
    action: require_approval
    approvers: ["tech_lead", "security_team"]

  # Critical: multi-party approval
  critical:
    conditions:
      - changes_encryption
      - changes_key_management
      - production_config_changes
    action: require_multi_approval
    min_approvers: 2
    approvers: ["security_team", "devops_team", "tech_lead"]
```

**TUI for Approvals:**

```
┌─────────────────────────────────────────────────────────────────┐
│ SUPERVISOR - Approval Required                                  │
├─────────────────────────────────────────────────────────────────┤
│ Task: P1T5-PH2-C3 (Auth Endpoints)                              │
│ Risk Level: HIGH (changes auth code)                            │
│                                                                 │
│ Changes:                                                        │
│   + api/auth.py (new file, 150 lines)                           │
│   ~ middleware/auth.py (modified, +20/-5 lines)                 │
│   + tests/test_auth.py (new file, 80 lines)                     │
│                                                                 │
│ Review Summary:                                                 │
│   ✓ reviewer_gemini: APPROVED                                   │
│   ✓ reviewer_codex: APPROVED                                    │
│   ✓ security_auditor: PASSED (no vulnerabilities found)         │
│                                                                 │
│ [A]pprove  [R]eject  [D]iff  [C]omments  [S]kip for now         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Critical Review Findings (Gemini + Codex)

### CRITICAL: Must Fix Before Implementation

#### 1. Sandboxing is a Blocker (Move to Phase 1) ✅ FIXED

**Problem:** Auto-approve flags (`--dangerously-skip-permissions`, `--yolo`) + no sandbox = arbitrary command execution.

**Fix:** Require containerization before any automation. Use TWO container types:

```python
class SandboxedLLMClient:
    """Execute CLI in isolated container with controlled egress."""

    # CLI containers need network to reach model APIs
    ALLOWED_EGRESS = [
        "api.anthropic.com:443",
        "api.openai.com:443",
        "generativelanguage.googleapis.com:443",
    ]

    def execute_cli(self, prompt: str) -> str:
        """Run AI CLI - needs network for API calls."""
        return subprocess.run([
            "docker", "run", "--rm",
            f"--network={self.egress_network}",  # Allowlisted egress only
            f"--volume={self.workdir}:/workspace:rw",
            "--workdir=/workspace",
            "--user=nobody",
            "--read-only",
            "--tmpfs=/tmp",
            "--memory=4g",
            "--cpus=2",
            self.cli_image,
            self.cli_command, prompt
        ], capture_output=True, text=True, timeout=300).stdout


class SandboxedExecutor:
    """Execute arbitrary commands (tests, lint) - NO network."""

    def run(self, command: list[str], workdir: str) -> subprocess.CompletedProcess:
        """Run commands in fully isolated container."""
        return subprocess.run([
            "docker", "run", "--rm",
            "--network=none",  # No network - tests don't need it
            f"--volume={workdir}:/workspace:rw",
            "--workdir=/workspace",
            "--user=nobody",
            "--read-only",
            "--tmpfs=/tmp",
            "--memory=2g",
            "--cpus=1",
            self.executor_image,
            *command
        ], capture_output=True, text=True, timeout=600)
```

**Network Policy:** Create a Docker network with egress firewall rules:
```bash
# Create network with egress allowlist (iptables/nftables)
docker network create --driver bridge supervisor-egress
# Configure firewall to only allow outbound to API endpoints
```

#### 2. Remove Pub/Sub (Conflicts with Stateless Workers) ✅ FIXED

**Problem:** Pub/Sub implies long-running agents. Our workers are stateless and die after each step.

**Fix:** Removed `MessagePool` from Design Patterns section. Use DAG Scheduler + SQLite state as sole communication channel:

```python
# CORRECT: Orchestrator pulls from DAG, no pub/sub
db.update_component(component_id, status="complete", output=result)
next_components = dag_scheduler.get_ready_components()  # DAG decides
```

#### 3. Review Gate Spoofing Prevention ✅ FIXED

**Problem:** Magic string detection (`"REVIEW_STATUS: APPROVED"`) can be spoofed.

**Fix:** Removed all marker fallback from gates and parser. JSON-only validation:

```python
def parse_review_output(raw_output: str) -> ReviewOutput:
    json_str = extract_json_block(raw_output)
    if not json_str:
        raise InvalidOutputError("No JSON block found - review rejected")

    # Strict validation, no fallback - updated in gates.py and parser.py sections
    return ReviewOutput.model_validate_json(json_str)
```

#### 4. Workspace Isolation (Prevent Partial Writes) ✅ FIXED

**Problem:** Workers mutate shared working tree. Retries stack partial changes. Also: changes were applied BEFORE gates ran.

**Fix:** Run each step in clean git worktree. **Run gates IN worktree BEFORE applying to main tree:**

```python
class IsolatedWorkspace:
    """Each step gets a clean workspace. Gates run before apply."""

    def __init__(self, gates: WorkflowGates, db: Database):
        self.gates = gates
        self.db = db

    def execute_step(self, step: Step) -> StepResult:
        worktree_path = f".worktrees/{step.id}"
        subprocess.run(["git", "worktree", "add", worktree_path, "HEAD"])

        try:
            # 1. Execute worker in isolation
            result = self.run_worker(step, cwd=worktree_path)

            # 2. Run gates IN THE WORKTREE (not main tree)
            if step.gates:
                for gate in step.gates:
                    passed, output = self.gates.run(gate, worktree_path)
                    if not passed:
                        raise GateFailedError(gate, output)

            # 3. ONLY after gates pass: apply to main tree + update state
            with self.db.transaction():
                self.apply_changes(worktree_path)
                self.db.update_step(step.id, status="complete", output=result)

            return result

        except Exception as e:
            # On failure, worktree is discarded - main tree untouched
            self.db.update_step(step.id, status="failed", error=str(e))
            raise
        finally:
            subprocess.run(["git", "worktree", "remove", worktree_path, "--force"])
```

### HIGH: Should Fix

#### 5. Parallel Execution File Locking

**Problem:** DAG allows parallel components but no file conflict detection.

**Fix:** Require disjoint file sets or serialize conflicting components:

```python
def schedule_parallel(self, ready: list[Component]) -> list[list[Component]]:
    """Group components into conflict-free batches."""
    batches = []
    scheduled_files = set()

    for comp in ready:
        comp_files = set(comp.files)
        if comp_files & scheduled_files:
            # Conflict - start new batch
            batches.append([comp])
            scheduled_files = comp_files
        else:
            if not batches:
                batches.append([])
            batches[-1].append(comp)
            scheduled_files |= comp_files

    return batches
```

#### 6. Context Overflow Pruning

**Problem:** No strategy when context exceeds token limit.

**Fix:** Progressive context reduction:

```python
class ContextPacker:
    PRIORITY_ORDER = ["system_prompt", "task", "target_file", "imports", "related", "tree"]

    def pack_with_budget(self, role: str, target: str, budget: int) -> str:
        context_parts = self.gather_all_context(role, target)

        # Try full context
        full = self.combine(context_parts)
        if self.count_tokens(full) <= budget:
            return full

        # Progressive pruning by priority
        for drop_key in reversed(self.PRIORITY_ORDER):
            if drop_key in context_parts:
                del context_parts[drop_key]
                pruned = self.combine(context_parts)
                if self.count_tokens(pruned) <= budget:
                    return pruned

        raise ContextTooLargeError("Cannot fit even minimal context in budget")
```

#### 7. History Hashing (Detect Semantic Loops)

**Problem:** Worker may oscillate between two incorrect states.

**Fix:** Track content hashes to detect loops:

```python
class LoopDetector:
    def __init__(self):
        self.seen_states: dict[str, set[str]] = {}  # step_id -> hashes

    def check_and_record(self, step_id: str, files: dict[str, str]) -> None:
        content_hash = hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest()

        if step_id not in self.seen_states:
            self.seen_states[step_id] = set()

        if content_hash in self.seen_states[step_id]:
            raise LoopDetectedError(
                f"Step {step_id} produced same output as a previous attempt. "
                "Likely oscillating between two incorrect states."
            )

        self.seen_states[step_id].add(content_hash)
```

#### 8. Secret Redaction in Context

**Problem:** Broad globs may include secrets in context.

**Fix:** Add secret scanning before packing:

```python
class SecretScanner:
    PATTERNS = [
        r"(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*['\"]?[\w-]+",
        r"ghp_[a-zA-Z0-9]{36}",  # GitHub token
        r"sk-[a-zA-Z0-9]{48}",   # OpenAI key
        r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
    ]

    def redact(self, content: str) -> str:
        for pattern in self.PATTERNS:
            content = re.sub(pattern, "[REDACTED]", content)
        return content
```

### MEDIUM: Nice to Fix

#### 9. JSONL Parser for Codex

**Problem:** Codex emits JSONL events, but parser assumes single JSON.

**Fix:** Per-CLI adapter pattern:

```python
class CodexAdapter(CLIAdapter):
    def parse_output(self, stdout: str) -> StructuredOutput:
        # Parse JSONL, extract final result event
        events = [json.loads(line) for line in stdout.strip().split('\n')]
        result_event = next(e for e in reversed(events) if e.get('type') == 'result')
        return StructuredOutput(**result_event['payload'])

class ClaudeAdapter(CLIAdapter):
    def parse_output(self, stdout: str) -> StructuredOutput:
        # Extract JSON block from markdown
        json_str = extract_json_block(stdout)
        return StructuredOutput.model_validate_json(json_str)
```

#### 10. Interface-First with Machine Validation

**Problem:** Interface locking relies on LLM following instructions.

**Fix:** Generate real artifacts and validate:

```python
def lock_interfaces(self, interfaces: list[Interface]) -> None:
    for interface in interfaces:
        # Write actual files
        if interface.type == "typescript":
            path = f"types/{interface.name}.d.ts"
            self.write_file(path, interface.definition)

        elif interface.type == "openapi":
            path = f"api/{interface.name}.yaml"
            self.write_file(path, interface.definition)

        # Make read-only during implementation
        os.chmod(path, 0o444)

def validate_implementation(self, component: Component) -> None:
    # Run actual compiler/linter to verify compliance
    subprocess.run(["tsc", "--noEmit"], check=True)  # TypeScript check
    subprocess.run(["openapi-validator", "api/"], check=True)  # API check
```

### Patterns to KEEP

1. **Orchestrator-Enforced Gates** - Workers never verify themselves
2. **SQLite Event Sourcing** - Full audit trail, reproducible state
3. **Differential Context for Reviewers** - Only "blast radius" of changes
4. **Circuit Breakers** - Prevent infinite retry loops
5. **Hierarchical Workflows** - Feature → Phase → Component
6. **DAG Scheduler** - Respects dependencies, enables parallelism

### Patterns to DISCARD

1. **MessagePool / Pub-Sub** - Conflicts with stateless worker model
2. **Magic String Detection** - Use schema validation only
3. **Auto-approve Flags without Sandbox** - Security risk

---

## Open Questions (For Discussion)

1. **Workflow Versioning:** Should workflow definitions be versioned? What happens when you update a workflow mid-feature?

2. **Rollback Strategy:** If a feature fails at Phase 3, how do we rollback Phases 1-2? Git revert? Branch deletion?

3. **Inter-AI Communication:** Should workers be able to leave notes for subsequent workers in the same feature (beyond structured outputs)?

4. **Subgraph Restart:** Should phases be self-contained subgraphs that can restart independently?

---

## References

### CLI Tools
- [Claude Code CLI](https://docs.anthropic.com/claude-code)
- [Codex CLI](https://github.com/openai/codex)
- [Gemini CLI](https://cloud.google.com/gemini)
- [Repomix](https://github.com/yamadashy/repomix)

### Design Inspirations
- [LangGraph - Build resilient language agents as graphs](https://github.com/langchain-ai/langgraph)
- [LangGraph Checkpointing & Human-in-the-Loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [MetaGPT - Multi-Agent Collaborative Framework](https://arxiv.org/html/2308.00352v6)
- [LangChain Agent Error Handling](https://apxml.com/courses/langchain-production-llm/chapter-2-sophisticated-agents-tools/agent-error-handling)
- [LangGraph Retry Policies](https://dev.to/aiengineering/a-beginners-guide-to-handling-errors-in-langgraph-with-retry-policies-h22)

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-01-03 | Claude + Gemini + Codex | Initial draft based on collaborative review |
| 2025-01-03 | Claude + Gemini + Codex | Added Dynamic Role System, domain-specific roles, pluggable context strategies, resolved open questions |
| 2025-01-03 | Claude + Gemini + Codex | Added Hierarchical Workflow (Feature→Phase→Component), Test Engineering strategy, Workflow Templates (MVP/Enterprise/OSS/Security), DAG Scheduler, Dependency Resolution, Advanced Features (Remote Execution, Metrics, Adaptive Assignment, Human Approval Gates) |
| 2025-01-03 | Claude | Added Design Patterns from LangGraph & MetaGPT (checkpointing, retry policies, publish-subscribe, schema-enforced outputs, executable feedback loops). Enhanced error handling with 6 error categories, circuit breakers, graceful degradation. Removed budget tracking (subscription model). |
| 2025-01-03 | Gemini + Codex Review | Critical review findings: (1) Remove Pub/Sub - conflicts with stateless workers, (2) Sandboxing is blocker - move to Phase 1, (3) Review gate spoofing prevention, (4) Workspace isolation for retries, (5) File locking for parallel execution, (6) Context overflow pruning, (7) Semantic loop detection, (8) Secret redaction, (9) Per-CLI output adapters, (10) Machine-validated interfaces. |
| 2025-01-04 | Claude + Gemini + Codex | **Final review fixes:** (1) Updated Implementation Phases - sandboxing + SQLite in Phase 1, (2) Marked Pub/Sub as DISCARDED in Design Patterns, (3) Removed all magic string fallbacks from gates/parser, (4) Split sandbox into SandboxedLLMClient (egress allowlist) + SandboxedExecutor (no network), (5) Fixed worktree atomicity - gates run before apply with DB transaction, (6) Added phase sequencing enforcement to DAG scheduler, (7) Removed MVP auto-approve threshold. All critical issues now addressed. |
