# Supervisor Orchestrator Specifications

This directory contains detailed specifications for each module in the Supervisor orchestrator system.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ExecutionEngine                          │
│                    (engine.py)                              │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ContextPacker  │  │ RoleLoader    │  │ CircuitBreaker│   │
│  │ (context.py)  │  │ (roles.py)    │  │ (engine.py)   │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                  IsolatedWorkspace                          │
│                  (workspace.py)                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Git Worktrees  │  Gates  │  File Lock  │  Apply    │   │
│   └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                  Docker Sandbox                             │
│                  (executor.py)                              │
│   ┌────────────────────┐  ┌────────────────────┐           │
│   │ SandboxedLLMClient │  │ SandboxedExecutor  │           │
│   │ (with network)     │  │ (no network)       │           │
│   └────────────────────┘  └────────────────────┘           │
├─────────────────────────────────────────────────────────────┤
│                     Database                                │
│                     (state.py)                              │
│   ┌─────────────┐  ┌────────────────────────────────────┐  │
│   │   Events    │→ │  Projections (workflows, steps...) │  │
│   │  (source)   │  │           (derived)                │  │
│   └─────────────┘  └────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Module Specifications

| Module | File | Description |
|--------|------|-------------|
| [Engine](engine.md) | `supervisor/core/engine.py` | Central execution coordinator |
| [Workspace](workspace.md) | `supervisor/core/workspace.py` | Git worktree isolation |
| [Executor](executor.md) | `supervisor/sandbox/executor.py` | Docker sandboxing |
| [State](state.md) | `supervisor/core/state.py` | SQLite event sourcing |
| [Parser](parser.md) | `supervisor/core/parser.py` | Output parsing |
| [Models](models.md) | `supervisor/core/models.py` | Data models |
| [Roles](roles.md) | `supervisor/core/roles.py` | Role configuration |
| [Context](context.md) | `supervisor/core/context.py` | Context packing |

## Security Model

### Isolation Layers

1. **Docker Containers**: All CLI execution in isolated containers
2. **Git Worktrees**: Each step has its own filesystem
3. **Gates**: Changes only applied after verification passes
4. **Network Control**: CLI containers have egress only; executors have no network

### Key Security Features

- Symlink rejection at multiple layers
- Path traversal prevention
- Atomic file operations
- HEAD conflict detection
- Required workdir validation

## Execution Flow

1. **Context Packing**: Assemble relevant files and context
2. **Worktree Creation**: Create isolated git worktree from HEAD
3. **CLI Execution**: Run AI CLI in Docker container
4. **Output Parsing**: Extract and validate structured output
5. **Gate Verification**: Run tests/lint in worktree
6. **Apply Changes**: Copy changes to main tree (if gates pass)
7. **State Update**: Record events and update projections

## Error Handling

- **Retry with Backoff**: Transient errors get exponential backoff
- **Retry with Feedback**: Parsing errors include correction guidance
- **Circuit Breaker**: Prevents infinite retry loops
- **Worktree Reset Warning**: Feedback includes notice about discarded changes

## Dependencies

- `pydantic`: Schema validation
- `filelock`: Concurrent access protection
- `jinja2`: Prompt templating
- `pyyaml`: Role configuration
- Docker: Container isolation
- Git: Worktree management
