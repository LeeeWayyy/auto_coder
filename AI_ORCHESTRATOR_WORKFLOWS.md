# AI Orchestrator: Hierarchical Workflows & Optimization

## 1. Hierarchical Workflow Design
**Strategy:** Recursive Decomposition with Scoped Context.

We treat "Feature", "Phase", and "Component" as nested **Units of Work**.

*   **Level 1: Feature (The Architect)**
    *   **Role:** `SystemArchitect`
    *   **Input:** User Request (e.g., "Add Auth System").
    *   **Action:** Defines the *Data Models*, *API Contracts*, and *Security Boundaries*.
    *   **Output:** `design_doc.md` and a list of **Phases** (logical groupings).
    *   **No Redundancy:** Does *not* write code or implementation steps. Focuses on "What" and "Why".

*   **Level 2: Phase (The Manager)**
    *   **Role:** `TechLead`
    *   **Input:** Phase definition from Level 1 + `design_doc.md`.
    *   **Action:** Breaks Phase into parallelizable **Tasks** (Components). Identifies dependencies.
    *   **Output:** A **Dependency Graph** (JSON) for the Orchestrator.
    *   **Context:** Inherits Architect's design but strictly filtered to relevant domain.

*   **Level 3: Component (The Worker)**
    *   **Role:** `Squad` (Planner -> Implementer -> Reviewer).
    *   **Input:** Single Task definition + Interfaces defined by Architect.
    *   **Action:** The standard "Plan -> Code -> Test" loop.

## 2. Test Engineer Role Strategy
**Strategy:** Hybrid Model (TDD + Independent QA).

*   **The Implementer (TDD):**
    *   *Responsibility:* Must write "Happy Path" unit tests *before* or *during* coding.
    *   *Why:* Ensures the code is testable and meets basic requirements. "If it doesn't build, it doesn't ship."

*   **The QA Engineer (Dedicated Role):**
    *   *Responsibility:* Runs *after* the Implementer is "done".
    *   *Focus:* Integration tests, Edge cases, Security probing, Regression testing.
    *   *Why:* AI Implementers often "hallucinate" tests that pass their specific buggy code. An independent QA role (different system prompt, higher temperature) acts as an adversary.

## 3. Configurable Workflows (Pluggable Pipelines)
**Strategy:** YAML-based Pipeline Definitions.

Define "Tracks" in `.supervisor/workflows.yaml`.

```yaml
# .supervisor/workflows.yaml
pipelines:
  # Fast prototyping
  startup_mvp:
    stages:
      - role: planner
      - role: implementer
        config: { tdd: false, style: "loose" }
      - role: committer

  # High assurance
  enterprise_grade:
    stages:
      - role: architect
        output: "design.md"
        gate: human_approval  # Checkpoint
      - role: tech_lead
        output: "task_graph.json"
      - parallel:
          iterator: "task_graph.tasks"
          pipeline:
            - role: planner
            - role: implementer
              config: { tdd: true, coverage: 90 }
            - role: peer_reviewer
      - role: security_auditor
      - role: qa_engineer
      - role: committer
```

## 4. Optimization & Parallelism
**Strategy:** DAG Execution with "Interface-First" Locking.

*   **Parallelism:**
    *   The `TechLead` output (Task Graph) drives the Orchestrator's scheduler.
    *   Tasks with no dependencies run in parallel subprocesses.
    *   *Example:* "Frontend Login Form" (Task C) and "Backend User Model" (Task A) *might* depend on each other.
*   **Dependency Handling (Interface-First):**
    *   To maximize parallelism, we introduce an **"Interface Definition" Step** before parallel work.
    *   *Step 1:* Define API Spec (OpenAPI/Swagger) & Types.
    *   *Step 2:* Backend implements API (validating against Spec).
    *   *Step 3:* Frontend implements UI (mocking against Spec).
    *   *Result:* Backend and Frontend run in parallel, meeting in the middle.

## 5. Checkpoints & Human-in-the-Loop
**Strategy:** Risk-Based Gating.

*   **Auto-Proceed:**
    *   Linter failures (Fix loop).
    *   Unit Test failures (Fix loop).
    *   formatting/style changes.
*   **Human Checkpoint Required:**
    *   **Architecture Approval:** Before any code is written. (High cost to revert).
    *   **Security/Auth Changes:** High risk.
    *   **Destructive Actions:** Database migrations, deleting files.
    *   **Final Merge:** The "Acceptance" gate.

## 6. Addressing Open Questions
*   **Monorepo vs Polyrepo:** The `repomix` context packer needs `root_dir` awareness. For monorepos, context is scoped to `packages/service-A`.
*   **Role Marketplace:** A simple Git repo of YAML files (`github.com/supervisor-roles/standard`) that users can import/submodule into `.supervisor/roles`.
*   **Metrics:** Track `(Tokens Used / Lines of Code Produced)` and `(Rejection Rate by Reviewer)`. Auto-demote models that fail often.
