# Auto Coder (Supervisor) Architecture

## System Overview

Auto Coder is an AI CLI orchestrator that treats AI tools (Claude, Codex, Gemini) as **stateless workers** rather than conversational partners. This architectural document describes how Supervisor manages AI workers with the same rigor as an operating system manages processes, memory, and I/O.

### Core Philosophy

**Problem**: Context dilution - when AI CLIs maintain long conversations, their context windows grow and they progressively ignore earlier instructions.

**Solution**: For every step of a workflow:
1. Spin up a fresh, short-lived AI instance
2. Feed it only relevant context (not full conversation history)
3. Get structured output
4. Immediately terminate the instance

Think of it as **process management for AI** - each AI execution is a process with:
- Isolated execution environment (Docker + git worktrees)
- Defined inputs and outputs (schema-enforced)
- Resource limits (timeouts, token budgets)
- Verification gates (tests, linting)
- Full audit trail (event sourcing)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / CLI                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ExecutionEngine                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Context    │  │    Parser    │  │  Workspace   │          │
│  │   Packer     │  │              │  │  Manager     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   Sandbox    │  │   Database   │  │    Gates     │
    │  (Docker)    │  │ (Event Log)  │  │  Executor    │
    └──────────────┘  └──────────────┘  └──────────────┘
              │              │              │
              ▼              ▼              ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ AI CLI       │  │  SQLite DB   │  │  Git         │
    │ (Claude,     │  │  Projections │  │  Worktrees   │
    │  Codex, etc) │  │              │  │              │
    └──────────────┘  └──────────────┘  └──────────────┘
```

## Declarative Graph Workflows & Studio

Supervisor includes a **declarative graph engine** (`supervisor/core/graph_engine.py`) and
schema (`supervisor/core/graph_schema.py`) that power Supervisor Studio, the web console.

Key characteristics:
- Graphs are defined in YAML/JSON with typed nodes (TASK, GATE, BRANCH, PARALLEL, HUMAN, etc.)
- Execution is **stateless** and persisted in SQLite tables (`graph_workflows`, `graph_executions`, `node_executions`)
- Real-time updates use WebSockets from the Studio backend (`supervisor/studio/server.py`)
- Live updates are **in-memory callbacks**; only executions started by the same Studio instance stream updates

Studio is a local-only UI intended for development and inspection (no auth).

---

## Core Components

### 1. ExecutionEngine

**Purpose**: Central coordinator for all workflow operations.

**Location**: `supervisor/core/engine.py`

**Responsibilities**:
- Context packing: Assemble relevant files and instructions
- Worker invocation: Execute AI CLIs in sandboxed environments
- Output parsing: Extract and validate structured responses
- Gate verification: Run tests, linting, security scans
- State updates: Record events to the database
- Retry logic: Handle transient failures with exponential backoff
- Circuit breaking: Prevent infinite retry loops

**Key Classes**:
- `ExecutionEngine` - Main orchestrator
- `RetryPolicy` - Exponential backoff configuration
- `CircuitBreaker` - Failure tracking (simple, thread-safe)
- `EnhancedCircuitBreaker` - Advanced failure tracking with metrics
- `ErrorClassifier` - Categorize errors for appropriate handling

**Design Patterns**:
- **Coordinator Pattern**: Orchestrates multiple subsystems
- **Retry with Backoff**: Exponential backoff with jitter
- **Circuit Breaker**: Prevent cascading failures
- **Strategy Pattern**: Error classification determines retry strategy

**Example Flow**:
```python
engine = ExecutionEngine(repo_path)

# 1. Load role configuration
role = role_loader.load_role("implementer")

# 2. Pack context (files, instructions)
context = context_packer.pack_context(role, task)

# 3. Create isolated worktree
with IsolatedWorkspace(repo_path) as workspace:
    # 4. Execute in sandbox
    result = sandbox.execute(["claude", "-p", context])

    # 5. Parse output
    parsed = parser.parse(result.stdout)

    # 6. Run verification gates
    gate_results = gate_executor.run_all(workspace.worktree_path)

    # 7. Apply changes if gates pass
    workspace.apply_changes()

# 8. Record events to database
db.append_event(Event(...))
```

---

### 2. State Management (Database)

**Purpose**: Event-sourced state with SQLite backend.

**Location**: `supervisor/core/state.py`

**Architecture**:
```
Events Table (Write Model - Source of Truth)
    │
    ├─> Workflows Projection (Read Model)
    ├─> Steps Projection (Read Model)
    ├─> Features Projection (Read Model)
    ├─> Phases Projection (Read Model)
    ├─> Components Projection (Read Model)
    └─> Metrics Table (Performance data)
```

**Event Types**:
- Workflow: `STARTED`, `COMPLETED`, `FAILED`
- Step: `STARTED`, `APPLYING`, `COMPLETED`, `FAILED`, `RETRIED`
- Gate: `PASSED`, `FAILED`
- Feature/Phase/Component: `CREATED`, `STARTED`, `COMPLETED`, `FAILED`
- Approval: `REQUESTED`, `GRANTED`, `DENIED`, `SKIPPED`
- Checkpoint: `CREATED`, `RESTORED`

**Key Design Decisions**:

1. **Event Sourcing**: All state changes are recorded as immutable events
   - **Benefit**: Full audit trail, reproducible state
   - **Benefit**: Can replay events to reconstruct state
   - **Benefit**: Time-travel debugging (see state at any point)

2. **Projection Tables**: Read models derived from events
   - **Benefit**: Fast queries without replaying events
   - **Benefit**: Can rebuild projections if corrupted
   - **Benefit**: Can add new projections without migrating events

3. **WAL Mode**: Write-Ahead Logging for concurrency
   - **Benefit**: Readers and writers don't block each other
   - **Benefit**: Better performance under concurrent access

**Example Event Sourcing**:
```python
# Record events (write model)
db.append_event(Event(
    workflow_id="wf-001",
    event_type=EventType.STEP_STARTED,
    role="implementer",
    step_id="step-001",
    payload={"task": "Add login endpoint"}
))

db.append_event(Event(
    workflow_id="wf-001",
    event_type=EventType.STEP_COMPLETED,
    role="implementer",
    step_id="step-001",
    status="success",
    payload={"files_modified": ["src/api/auth.py"]}
))

# Query projections (read model)
step = db.get_step("step-001")
assert step.status == StepStatus.SUCCESS

# Reconstruct state from events (crash recovery)
events = db.get_events("wf-001")
# Replay events to rebuild projections if needed
```

---

### 3. Sandbox Execution

**Purpose**: Isolated execution environments for AI CLIs and commands.

**Location**: `supervisor/sandbox/executor.py`

**Two Container Types**:

1. **SandboxedLLMClient** - For AI CLI calls
   - Network egress: Allowed (needs to reach AI APIs)
   - Egress allowlist: `api.anthropic.com`, `api.openai.com`, etc.
   - Use case: Running Claude, Codex, Gemini

2. **SandboxedExecutor** - For tests/commands
   - Network: **None** (fully isolated)
   - Read-only mounts: Repository files
   - Limited writes: `.pytest_cache`, `__pycache__`, etc.
   - Use case: Running pytest, ruff, mypy

**Security Model**:

```
┌────────────────────────────────────────────┐
│          Host Machine                       │
│  ┌──────────────────────────────────────┐  │
│  │      Docker Container                │  │
│  │  ┌────────────────────────────────┐  │  │
│  │  │  AI CLI Process                │  │  │
│  │  │  - Limited memory               │  │  │
│  │  │  - Timeout enforced             │  │  │
│  │  │  - Read-only mounts             │  │  │
│  │  │  - Network policy               │  │  │
│  │  └────────────────────────────────┘  │  │
│  │                                        │  │
│  │  Volume Mounts:                        │  │
│  │  - /workspace (read-only)              │  │
│  │  - /tmp (read-write, specific paths)   │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

**Security Features**:
- **Docker required**: No local execution in production
- **Path validation**: Reject path traversal attempts (`../etc/passwd`)
- **Symlink rejection**: Prevent symlink attacks
- **Network isolation**: Configurable per container type
- **Resource limits**: CPU, memory, timeout
- **Output truncation**: Prevent memory exhaustion (10MB limit)
- **Command sanitization**: Prevent shell injection

**Example**:
```python
# AI CLI execution (with network)
client = SandboxedLLMClient(
    repo_path=repo_path,
    workdir=worktree_path,
    config=SandboxConfig(network_mode="egress-only")
)
result = client.execute(["claude", "-p", prompt])

# Gate execution (no network)
executor = SandboxedExecutor(
    repo_path=repo_path,
    workdir=worktree_path,
    config=SandboxConfig(network_mode="none")
)
result = executor.execute(["pytest", "-q"])
```

---

### 4. Workspace Management

**Purpose**: Isolated git worktrees for each execution step.

**Location**: `supervisor/core/workspace.py`

**Key Concept**: **Isolation**

Every AI execution gets its own worktree:
- Separate filesystem (not affecting main branch)
- Can make changes safely
- Changes only applied after gates pass
- Atomic commit or rollback

**Worktree Lifecycle**:
```
1. Create worktree from HEAD
   git worktree add /tmp/supervisor_abc123 HEAD

2. AI makes changes in worktree
   (Sandboxed execution modifies files)

3. Run verification gates
   pytest, ruff, mypy (in worktree)

4. If gates pass:
   - Copy changes to main tree
   - Update database
   - Cleanup worktree

   If gates fail:
   - Discard worktree
   - Record failure
   - Optionally retry
```

**Security Features**:
- **Symlink rejection**: No symlinks in worktree paths
- **Path validation**: Worktree must be under `.worktrees/`
- **HEAD conflict detection**: Detect concurrent modifications
- **Atomic operations**: Either all changes apply or none

**Example**:
```python
with IsolatedWorkspace(repo_path) as workspace:
    # Worktree created automatically
    print(workspace.worktree_path)  # /repo/.worktrees/abc123

    # AI makes changes
    sandbox.execute(["claude", "..."], cwd=workspace.worktree_path)

    # Run gates
    gate_results = run_gates(workspace.worktree_path)

    if all(g.passed for g in gate_results):
        # Apply changes to main tree
        workspace.apply_changes()
    else:
        # Discard worktree (automatic on context exit)
        raise GateFailedError()

# Worktree cleaned up automatically
```

---

### 5. Context Packing

**Purpose**: Intelligently select which files/content to send to AI.

**Location**: `supervisor/core/context.py`

**Challenge**: AI models have token limits (e.g., 200k tokens for Claude Opus). Can't send entire codebase.

**Solution**: Smart file selection + priority-based pruning

**Context Packing Strategy**:
```
1. Always include:
   - System prompt (role instructions)
   - Task description
   - Target files (user-specified)

2. Include based on role config:
   - Role-specific includes (e.g., "tests/**/*.py")
   - Git diff (recent changes)
   - Related files (imports, dependencies)

3. Token budget management:
   - Default budget: 25k-30k tokens per role
   - Calculate token usage (1 token ≈ 4 chars)
   - If over budget: prune lowest priority files
   - Protected keys: Never prune system_prompt, task

4. Use Repomix (optional):
   - AI-powered file selection
   - Understands project structure
   - Falls back to simple packing if unavailable
```

**Jinja2 Templating**:
```jinja
{# supervisor/prompts/implement.j2 #}
{% block system_prompt %}
{{ system_prompt }}
{% endblock %}

{% block task %}
Task: {{ task_description }}
{% endblock %}

{% block target_files %}
{% if target_files %}
Target Files:
{% for file in target_files %}
## {{ file.path }}
{{ file.content }}
{% endfor %}
{% endif %}
{% endblock %}

{% block context %}
{# Additional context based on role configuration #}
{% endblock %}
```

**Example**:
```python
packer = ContextPacker(repo_path)

context = packer.pack_context(
    role=implementer_role,
    task_description="Add /api/login endpoint",
    target_files=["src/api/auth.py"],
)

# Context includes:
# - System prompt from role
# - Task description
# - Content of src/api/auth.py
# - Related files (imports from auth.py)
# - Recent git diff
# Total: ~25k tokens (within budget)
```

---

### 6. Role System

**Purpose**: Define how AI workers behave for different tasks.

**Location**: `supervisor/core/roles.py`

**Architecture**: **Base + Overlay** model

```
Base Role (implementer.yaml)
    ├─ Default CLI: claude
    ├─ System prompt: General implementation instructions
    ├─ Context: Include src/**, tests/**
    ├─ Gates: [test, lint]
    └─ Config: {max_retries: 3, timeout: 300}
         │
         ▼
Domain Overlay (frontend_implementer.yaml)
    ├─ Extends: implementer
    ├─ Override CLI: claude:opus (higher quality for frontend)
    ├─ Additional prompt: "Follow React best practices..."
    ├─ Additional context: Include *.tsx, *.css
    └─ Additional gates: [format_check]
         │
         ▼
Merged Role (runtime)
    ├─ CLI: claude:opus
    ├─ System prompt: Base + overlay prompts
    ├─ Context: src/**, tests/**, *.tsx, *.css
    ├─ Gates: [test, lint, format_check]
    └─ Config: {max_retries: 3, timeout: 300}
```

**Merge Semantics**:
- Lists: Append (context includes, gates)
- Dicts: Deep merge (config)
- Scalars: Override (cli, description)

**Base Roles** (shipped with Supervisor):
- `planner`: Break features into phases/components
- `implementer`: Write code following TDD principles
- `reviewer`: Review code for quality/correctness

**Example Role**:
```yaml
# .supervisor/roles/security_reviewer.yaml
name: security_reviewer
description: "Security-focused code reviewer"
extends: reviewer

cli: claude:opus  # Use highest quality model

system_prompt: |
  Review code with a security focus:
  - SQL injection vulnerabilities
  - XSS attacks
  - Authentication bypass
  - Sensitive data exposure
  - Command injection
  - Path traversal

context:
  include:
    - "src/api/**/*.py"  # API endpoints
    - "src/auth/**/*.py"  # Auth logic
  token_budget: 40000  # Larger budget for security review

gates:
  - test
  - lint
  - security  # Run bandit security scanner

config:
  max_retries: 2
  timeout: 600  # 10 minutes for thorough review
```

---

### 7. Gate System

**Purpose**: Automated verification enforced by orchestrator (not AI).

**Location**: `supervisor/core/gates.py`, `gate_executor.py`, `gate_loader.py`

**Key Principle**: **Don't trust AI output**

Gates are verification steps that run **after** AI generates code but **before** changes are applied:
- AI cannot skip gates
- AI cannot modify gate results
- Gates run in isolated worktrees
- Gates have dependencies (e.g., type_check requires lint)

**Gate Configuration**:
```yaml
# .supervisor/gates.yaml
gates:
  lint:
    command: ["ruff", "check", "."]
    timeout: 60
    description: "Run linter"
    severity: error
    fail_action: fail
    parallel_safe: true

  test:
    command: ["pytest", "-q", "--maxfail=5"]
    timeout: 300
    description: "Run test suite"
    severity: error
    fail_action: fail
    depends_on: [lint]  # Run after lint passes
    cache: true  # Cache results based on file content

  type_check:
    command: ["mypy", "src/"]
    timeout: 120
    description: "Type checking"
    severity: warning  # Don't fail on type errors
    fail_action: warn
    depends_on: [lint]

  security:
    command: ["bandit", "-r", "src/", "-f", "json"]
    timeout: 60
    description: "Security scan"
    severity: error
    fail_action: fail
```

**Gate Execution Flow**:
```
1. Load gate configs
2. Build dependency DAG
3. Detect cycles (fail if found)
4. Topological sort for execution order
5. Execute gates (parallel where possible)
6. Record results
7. Determine overall status (pass/fail)
```

**Caching**:
- Gates can be cached based on input files
- Cache key: hash of (gate config + input file contents)
- Saves time on repeated executions

**Example**:
```python
loader = GateLoader()
gates = loader.load_all()

executor = GateExecutor(gates)
results = executor.run_all(worktree_path)

for result in results:
    print(f"{result.gate_name}: {result.status}")
    if result.status == GateStatus.FAILED:
        print(f"  Output: {result.output}")

if all(r.passed for r in results):
    workspace.apply_changes()
else:
    raise GateFailedError()
```

---

### 8. Multi-Model Routing

**Purpose**: Intelligently select which AI model to use for each task.

**Location**: `supervisor/core/routing.py`

**Problem**: Different AI models have different strengths:
- Claude Opus: Best for architecture, complex reasoning
- Claude Sonnet: Good balance of speed/quality
- Claude Haiku: Fast, cost-effective for simple tasks
- Codex: Specialized for code generation
- Gemini: Strong at refactoring

**Solution**: Task-aware model selection

**Routing Strategies**:

1. **Static Routing** (Role configuration):
   ```yaml
   # Use specific model for role
   cli: claude:opus
   ```

2. **Task-Type Routing** (Adaptive):
   ```python
   task_type = infer_task_type(task_description)
   # "architecture" → claude:opus
   # "code_gen" → claude:sonnet
   # "debugging" → gemini:pro
   ```

3. **Performance-Based** (Adaptive):
   ```python
   # Track historical performance
   # Route to best-performing model for task type
   best_model = get_best_model_for_task(
       task_type="refactoring",
       success_rate_threshold=0.8
   )
   ```

**Adaptive Configuration**:
```yaml
# .supervisor/adaptive.yaml
adaptive:
  enabled: true
  min_samples_before_adapt: 10
  exploration_rate: 0.1

  routing:
    architecture:
      - claude:opus
      - gemini:pro

    code_gen:
      - claude:sonnet
      - codex:gpt-4

    refactoring:
      - claude:sonnet
      - gemini:pro
```

---

## Data Flow: End-to-End Example

Let's trace a complete execution from user command to applied changes.

### Command
```bash
supervisor run implementer "Add /api/login endpoint" -t src/api/auth.py
```

### Flow

**1. CLI Parsing** (`supervisor/cli.py`)
```python
# Parse command
role_name = "implementer"
task = "Add /api/login endpoint"
target_files = ["src/api/auth.py"]
workflow_id = generate_id()  # "wf-abc123"
```

**2. Engine Initialization** (`supervisor/core/engine.py`)
```python
engine = ExecutionEngine(repo_path)

# Load role
role = engine.role_loader.load_role("implementer")
# role.cli = "claude:sonnet"
# role.gates = ["test", "lint"]
```

**3. Context Packing** (`supervisor/core/context.py`)
```python
context = engine.context_packer.pack_context(
    role=role,
    task_description=task,
    target_files=target_files
)

# context includes:
# - System prompt: "You are a Python developer..."
# - Task: "Add /api/login endpoint"
# - Target file: src/api/auth.py content
# - Related files: src/models/user.py, src/utils/jwt.py
# - Recent git diff
# Total: ~22k tokens
```

**4. Worktree Creation** (`supervisor/core/workspace.py`)
```python
with IsolatedWorkspace(repo_path) as workspace:
    worktree_path = workspace.worktree_path
    # /repo/.worktrees/wf-abc123-step-001
```

**5. Database Event** (`supervisor/core/state.py`)
```python
db.append_event(Event(
    workflow_id=workflow_id,
    event_type=EventType.STEP_STARTED,
    role="implementer",
    step_id="step-001",
    payload={"task": task}
))
```

**6. Sandbox Execution** (`supervisor/sandbox/executor.py`)
```python
client = SandboxedLLMClient(
    repo_path=repo_path,
    workdir=worktree_path,
    config=SandboxConfig(network_mode="egress-only")
)

result = client.execute([
    "claude",
    "-p", context
])

# Claude generates code and returns structured output
# {
#   "status": "success",
#   "files_modified": ["src/api/auth.py"],
#   "changes": [...],
#   "next_steps": [...]
# }
```

**7. Output Parsing** (`supervisor/core/parser.py`)
```python
parsed = parse_role_output(
    output=result.stdout,
    role_name="implementer",
    cli="claude"
)

# Validate schema
assert parsed.status == "success"
assert "src/api/auth.py" in parsed.files_modified
```

**8. Gate Execution** (`supervisor/core/gates.py`)
```python
gate_executor = GateExecutor(gates)
gate_results = gate_executor.run_all(worktree_path)

# Run in worktree:
# 1. lint (ruff check .) → PASSED
# 2. test (pytest -q) → PASSED

all_passed = all(r.passed for r in gate_results)
```

**9. Apply Changes** (if gates pass)
```python
if all_passed:
    workspace.apply_changes()
    # Copy files from worktree to main tree

    db.append_event(Event(
        workflow_id=workflow_id,
        event_type=EventType.STEP_COMPLETED,
        role="implementer",
        step_id="step-001",
        status="success",
        payload={"files_modified": parsed.files_modified}
    ))
else:
    # Retry or fail
    db.append_event(Event(
        workflow_id=workflow_id,
        event_type=EventType.STEP_FAILED,
        role="implementer",
        step_id="step-001",
        payload={"gate_failures": gate_results}
    ))
```

**10. Cleanup**
```python
# Worktree automatically removed on context exit
# workspace.__exit__() calls git worktree remove
```

---

## Security Model

### Threat Model

**Threats**:
1. **Malicious AI output**: AI generates harmful code
2. **Prompt injection**: User input manipulates AI behavior
3. **Path traversal**: AI attempts to access/modify system files
4. **Command injection**: AI generates shell commands
5. **Network attacks**: AI attempts to access internal services
6. **Resource exhaustion**: AI generates infinite loops or huge files

**Mitigations**:

| Threat | Mitigation | Implementation |
|--------|------------|----------------|
| Malicious code | Gates (tests, linting, security scan) | `supervisor/core/gates.py` |
| Prompt injection | Schema validation, output parsing | `supervisor/core/parser.py` |
| Path traversal | Path validation, symlink rejection | `supervisor/core/workspace.py`, `supervisor/sandbox/executor.py` |
| Command injection | Docker isolation, no shell execution | `supervisor/sandbox/executor.py` |
| Network attacks | Network isolation (egress allowlist) | `supervisor/sandbox/executor.py` |
| Resource exhaustion | Timeouts, output truncation, memory limits | `supervisor/core/engine.py`, `supervisor/sandbox/executor.py` |

### Isolation Layers

```
┌─────────────────────────────────────────────────┐
│ Layer 1: Docker Container                       │
│  - Process isolation                            │
│  - Network isolation                            │
│  - Filesystem isolation (read-only mounts)      │
└─────────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────────┐
│ Layer 2: Git Worktree                           │
│  - Separate filesystem branch                   │
│  - Changes don't affect main tree               │
│  - Can discard entire worktree atomically       │
└─────────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────────┐
│ Layer 3: Verification Gates                     │
│  - Tests must pass                              │
│  - Linting must pass                            │
│  - Security scan must pass                      │
│  - Changes only applied if all gates pass       │
└─────────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────────┐
│ Layer 4: Event Sourcing                         │
│  - Full audit trail                             │
│  - Can rollback to any previous state           │
│  - Immutable event log                          │
└─────────────────────────────────────────────────┘
```

---

## Extension Points

### Custom Roles

Create specialized roles for your domain:

```yaml
# .supervisor/roles/api_implementer.yaml
name: api_implementer
extends: implementer

system_prompt: |
  You are an API developer specializing in RESTful design.
  Follow these patterns:
  - Use FastAPI or Flask
  - OpenAPI documentation
  - Pydantic models for validation
  - JWT authentication

context:
  include:
    - "src/api/**/*.py"
    - "src/models/**/*.py"
    - "tests/api/**/*.py"
  token_budget: 30000

gates: [test, lint, security, openapi_validate]
```

### Custom Gates

Add project-specific verification:

```yaml
# .supervisor/gates.yaml
gates:
  custom_validate:
    command: ["python", "scripts/validate_api.py"]
    timeout: 60
    description: "Validate API contracts"
    severity: error
    fail_action: fail

  database_migration_check:
    command: ["python", "manage.py", "makemigrations", "--check"]
    timeout: 30
    description: "Check for missing migrations"
    severity: warning
    fail_action: warn
```

### Custom CLI Adapters

Support new AI tools:

```python
# supervisor/core/parser.py
class NewAIAdapter:
    """Adapter for NewAI CLI."""

    def parse(self, output: str) -> GenericOutput:
        # Extract structured output from NewAI format
        # Validate against schema
        # Return GenericOutput
        pass

# Register adapter
ADAPTERS["newai"] = NewAIAdapter()
```

---

## Performance Considerations

### Optimization Strategies

1. **Gate Caching**
   - Cache gate results based on file content hashes
   - Skip running gates if inputs haven't changed
   - Saves ~50% execution time on repeated runs

2. **Parallel Gate Execution**
   - Run independent gates in parallel
   - Use dependency DAG to maximize parallelism
   - Reduces gate execution time by ~40%

3. **Context Packing Optimization**
   - Use Repomix for intelligent file selection
   - Cache packed contexts for identical requests
   - Reduces token usage by ~30%

4. **Worktree Reuse**
   - Reuse worktrees for sequential steps (if safe)
   - Reduces git worktree overhead
   - Saves ~2-3 seconds per step

5. **Database Indexing**
   - Indexes on workflow_id, step_id, timestamp
   - WAL mode for concurrent access
   - Query performance: O(log n) for lookups

### Scaling

**Horizontal Scaling** (multiple machines):
- Each machine runs independent workflows
- Shared database (SQLite → PostgreSQL for multi-machine)
- Shared Docker registry for images

**Vertical Scaling** (single machine):
- Parallel workflow execution (--parallel flag)
- Multiple concurrent Docker containers
- Resource limits per container (CPU, memory)

---

## Monitoring and Observability

### Metrics Collected

- **Execution metrics**: Duration, success rate, retry count
- **Token usage**: Input/output tokens per role
- **Gate results**: Pass/fail rates per gate
- **Model performance**: Success rate by model and task type
- **Circuit breaker state**: Open/closed/half-open counts

### Viewing Metrics

```bash
# Dashboard with last 7 days
supervisor metrics --days 7

# Specific role
supervisor metrics --role implementer

# Export for analysis
supervisor metrics --format json > metrics.json
```

### Event Log Analysis

```sql
-- Query events directly
SELECT * FROM events
WHERE workflow_id = 'wf-abc123'
ORDER BY timestamp;

-- Aggregate statistics
SELECT
  role,
  COUNT(*) as executions,
  SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes
FROM events
WHERE event_type = 'step_completed'
GROUP BY role;
```

---

## Troubleshooting

### Common Issues

**Issue**: Docker not available
```bash
# Check Docker status
docker --version
docker ps

# Fix: Install Docker or start daemon
```

**Issue**: Gates failing repeatedly
```bash
# Check gate output
supervisor status --workflow-id wf-abc123

# Run gates manually in worktree
cd .worktrees/wf-abc123-step-001
pytest -v
ruff check .
```

**Issue**: Context too large
```yaml
# Reduce token budget in role config
context:
  token_budget: 15000  # Reduced from 25000
```

**Issue**: Circuit breaker open
```bash
# Check circuit state
# (View via database or metrics)

# Reset circuit manually
# Delete .supervisor/circuits.json
```

---

## Architectural Decisions

### Why Event Sourcing?

**Pros**:
- Full audit trail (who did what when)
- Can replay events for debugging
- Can add new projections without data migration
- Time-travel debugging

**Cons**:
- More complex than simple CRUD
- Requires careful event schema design
- Projection rebuilding can be slow for large logs

**Decision**: Pros outweigh cons for an orchestration system where auditability is critical.

### Why Docker?

**Pros**:
- Strong isolation guarantees
- Consistent environment across machines
- Network policy enforcement
- Wide tooling support

**Cons**:
- Requires Docker installation
- Adds overhead (~500ms per container start)
- Complexity in CI/CD environments

**Decision**: Security benefits justify the overhead. Local execution only for unit tests.

### Why Git Worktrees?

**Pros**:
- True filesystem isolation
- Atomic discard (just delete worktree)
- No impact on main working tree
- Git native (no custom tooling)

**Cons**:
- Disk space overhead
- Worktree creation takes ~1-2 seconds
- Cleanup required

**Decision**: Isolation is worth the cost. Cleanup is automated.

---

## Future Directions

### Planned Improvements

1. **Distributed Execution**
   - Multiple machines sharing work
   - PostgreSQL instead of SQLite
   - Kubernetes for container orchestration

2. **Streaming Output**
   - Real-time AI response streaming
   - Progress indicators during execution
   - Live TUI updates

3. **Advanced Caching**
   - Semantic caching (similar prompts)
   - Cross-workflow cache sharing
   - Cache warming strategies

4. **Enhanced Observability**
   - Prometheus metrics export
   - OpenTelemetry tracing
   - Web dashboard (Supervisor Studio)

5. **Plugin System**
   - Third-party role marketplace
   - Custom gate plugins
   - Webhook integrations

---

## Conclusion

Supervisor's architecture treats AI as managed infrastructure:
- **Stateless workers** prevent context dilution
- **Docker isolation** provides security guarantees
- **Event sourcing** ensures auditability
- **Verification gates** enforce quality standards
- **Git worktrees** enable safe experimentation

This design enables **reliable**, **auditable**, and **scalable** AI-assisted development workflows.

For implementation details, see:
- [Getting Started](GETTING_STARTED.md) - User guide
- [CLI Reference](CLI_REFERENCE.md) - Command documentation
- [Operations Guide](OPERATIONS.md) - Production deployment
- [Contributing](../CONTRIBUTING.md) - Development guide
