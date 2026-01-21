# LangGraph Studio-Style Enhancements for Supervisor Frontend (Revised v2)

## Overview

Transform the supervisor frontend into a more capable workflow editor with:
- Drag-and-drop node palette (all node types including subgraph)
- Extracted properties panel (right sidebar, preserved layout)
- Execution trace timeline with documented limitations
- Node output inspection
- Human-in-the-loop approval UI
- Animated edges during execution

**Key Design Principles:**
- Keep existing routing: WorkflowEditorPage for editing, ExecutionPage for monitoring
- Keep existing layout patterns: right sidebar for properties (no bottom panel)
- Avoid over-engineering: no tabs, no virtualization, no canvas freezing
- Document limitations clearly: timeline is live-only
- Build incrementally: simple implementations first, enhance based on real needs

---

## Architecture Decision: No Mode Switching

**Rationale:** The codebase already has clear separation:
- `WorkflowEditorPage` → workflow editing (`/workflows/:id`)
- `ExecutionPage` → execution monitoring (`/executions/:id`)

Adding mode-switching would:
1. Duplicate ExecutionPage functionality
2. Create two sources of truth for execution state
3. Complicate routing and state management

**Decision:** Keep this separation. Enhance each page independently:
- Phase 1: Enhance WorkflowEditorPage with palette and properties panel
- Phase 2: Enhance ExecutionPage with timeline and state inspector
- Phase 3: Add HITL support
- Phase 4: Visual polish (animated edges only)

---

## Phase 1: Visual Editor Enhancements

### 1.1 Create NodePalette.tsx
**File:** `supervisor/studio/frontend/src/components/NodePalette.tsx`

A left sidebar with draggable node types (includes ALL supported types):

```typescript
interface NodePaletteProps {
  onNodeAdd: (nodeType: NodeType, position: { x: number; y: number }) => void;
}

// NOTE: No icon fields - using text badges instead (see NODE_TYPE_BADGES below)
const NODE_CATEGORIES = {
  'Tasks': [
    { type: 'task', label: 'Task', description: 'Execute an AI agent task' },
    { type: 'subgraph', label: 'Subgraph', description: 'Embed another workflow' },
  ],
  'Control Flow': [
    { type: 'gate', label: 'Gate', description: 'Quality gate checkpoint' },
    { type: 'branch', label: 'Branch', description: 'Conditional branching' },
    { type: 'merge', label: 'Merge', description: 'Merge parallel paths' },
    { type: 'parallel', label: 'Parallel', description: 'Execute nodes in parallel' },
  ],
  'Human-in-Loop': [
    { type: 'human', label: 'Human', description: 'Human approval/input' },
  ],
};
```

**Type-Specific Defaults** (all 7 node types):
```typescript
function getDefaultConfig(nodeType: NodeType): Partial<Node> {
  switch (nodeType) {
    case 'task':
      return { task_config: { role: 'implementer' } };
    case 'gate':
      return { gate_config: { gate_type: 'test', auto_approve: false } };
    case 'human':
      return { human_config: { title: 'Review Required' } };
    case 'branch':
      return {
        branch_config: {
          condition: { field: 'status', operator: '==', value: 'success' },
          on_true: '',
          on_false: ''
        }
      };
    case 'merge':
      return { merge_config: { wait_for: 'all', merge_strategy: 'union' } };
    case 'parallel':
      return { parallel_config: { branches: [], wait_for: 'all' } };
    case 'subgraph':
      return {
        subgraph_config: {
          workflow_name: '',
          input_mapping: {},
          output_mapping: {}
        }
      };
    default:
      return {};
  }
}
```

- Fixed-width sidebar (w-56)
- HTML5 drag-and-drop with `dataTransfer.setData('application/reactflow-node-type', nodeType)`
- Visual categories with **text labels** (no icon library dependency - keep simple)
- Tooltips showing node descriptions

**Note:** No icon library (Lucide, etc.) is currently in package.json. Use simple text badges or Unicode symbols to avoid adding dependencies:
```typescript
const NODE_TYPE_BADGES: Record<NodeType, string> = {
  task: 'T',
  gate: 'G',
  human: 'H',
  branch: 'B',
  merge: 'M',
  parallel: 'P',
  subgraph: 'S',
};
```

### 1.2 Create PropertiesPanel.tsx
**File:** `supervisor/studio/frontend/src/components/PropertiesPanel.tsx`

Extract the existing sidebar form into a reusable component (**no tabs**, single panel, **keep right sidebar layout**):

```typescript
interface PropertiesPanelProps {
  selectedNode: Node | null;
  workflow: WorkflowGraph;
  onNodeUpdate: (patch: Partial<Node>) => void;
  onNodeDelete: () => void;
  onEdgeAdd: (source: string, target: string) => void;
  // Run inputs (keep in same panel, collapsible section)
  runGoal: string;
  onRunGoalChange: (goal: string) => void;
  runInputs: string;
  onRunInputsChange: (inputs: string) => void;
  runLabel: string;
  onRunLabelChange: (label: string) => void;
}
```

**Structure:**
- Node info section: ID (read-only), Type badge, Label input, Description input
- Type-specific config section (rendered based on node.type):
  - **task**: role (select with datalist), task_template (textarea)
  - **gate**: gate_type (input), auto_approve (checkbox)
  - **human**: title (input), description (textarea)
  - **branch**: See "Branch Config Constraints" below
  - **merge**: wait_for (select: all/any), merge_strategy (select)
  - **parallel**: branches (multi-select from existing nodes)
  - **subgraph**: workflow_name (input), input_mapping (JSON textarea), output_mapping (JSON textarea)

**Branch Config Constraints (Required):**

Branch nodes require special handling for operator, target node selection, **and value type coercion**.

**Value Type Coercion:**

The `condition.value` field is typed as `string | number | boolean | Array<...>`. The input must coerce user strings to appropriate types:

```typescript
type ConditionValue = string | number | boolean | Array<string | number | boolean>;

// Serialize value for display in text input
function serializeConditionValue(value: ConditionValue | undefined): string {
  if (value === undefined || value === null) return '';
  if (Array.isArray(value)) return JSON.stringify(value);
  return String(value);
}

// Parse user input to typed value
function parseConditionValue(input: string): ConditionValue {
  const trimmed = input.trim();

  // Empty → empty string
  if (!trimmed) return '';

  // Boolean detection
  if (trimmed === 'true') return true;
  if (trimmed === 'false') return false;

  // Number detection (integers and floats)
  const num = Number(trimmed);
  if (!isNaN(num) && trimmed !== '') return num;

  // Array detection (JSON array syntax)
  if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) return parsed;
    } catch {
      // Invalid JSON array - fall through to string
    }
  }

  // Default: string
  return trimmed;
}
```

**Operator and Node Selection:**

```typescript
// Valid operators from LoopCondition type
const BRANCH_OPERATORS = [
  { value: '==', label: 'equals (==)' },
  { value: '!=', label: 'not equals (!=)' },
  { value: '>', label: 'greater than (>)' },
  { value: '<', label: 'less than (<)' },
  { value: '>=', label: 'greater or equal (>=)' },
  { value: '<=', label: 'less or equal (<=)' },
  { value: 'in', label: 'in array (in)' },
  { value: 'not_in', label: 'not in array (not_in)' },
  { value: 'contains', label: 'contains' },
  { value: 'starts_with', label: 'starts with' },
  { value: 'ends_with', label: 'ends with' },
] as const;

// In PropertiesPanel for branch nodes:
{selectedNode.type === 'branch' && (
  <div className="space-y-3">
    {/* Condition field */}
    <label className="block text-sm">
      <span className="text-gray-700">Field</span>
      <input
        type="text"
        value={selectedNode.branch_config?.condition?.field || ''}
        onChange={(e) => onNodeUpdate({
          branch_config: {
            ...selectedNode.branch_config,
            condition: { ...selectedNode.branch_config?.condition, field: e.target.value }
          }
        })}
        className="mt-1 block w-full rounded border-gray-300"
        placeholder="e.g., status, result.success"
      />
    </label>

    {/* Operator dropdown - constrained to valid operators */}
    <label className="block text-sm">
      <span className="text-gray-700">Operator</span>
      <select
        value={selectedNode.branch_config?.condition?.operator || '=='}
        onChange={(e) => onNodeUpdate({
          branch_config: {
            ...selectedNode.branch_config,
            condition: { ...selectedNode.branch_config?.condition, operator: e.target.value }
          }
        })}
        className="mt-1 block w-full rounded border-gray-300"
      >
        {BRANCH_OPERATORS.map(op => (
          <option key={op.value} value={op.value}>{op.label}</option>
        ))}
      </select>
    </label>

    {/* Value input with type coercion */}
    <label className="block text-sm">
      <span className="text-gray-700">Value</span>
      <input
        type="text"
        value={serializeConditionValue(selectedNode.branch_config?.condition?.value)}
        onChange={(e) => onNodeUpdate({
          branch_config: {
            ...selectedNode.branch_config,
            condition: {
              ...selectedNode.branch_config?.condition,
              value: parseConditionValue(e.target.value)
            }
          }
        })}
        className="mt-1 block w-full rounded border-gray-300"
        placeholder="e.g., success, true, 100, [1,2,3]"
      />
      <span className="text-xs text-gray-500">
        Supports: strings, numbers, booleans (true/false), JSON arrays
      </span>
    </label>

    {/* On True - node selector dropdown */}
    <label className="block text-sm">
      <span className="text-gray-700">On True (target node)</span>
      <select
        value={selectedNode.branch_config?.on_true || ''}
        onChange={(e) => onNodeUpdate({
          branch_config: { ...selectedNode.branch_config, on_true: e.target.value }
        })}
        className="mt-1 block w-full rounded border-gray-300"
      >
        <option value="">-- Select node --</option>
        {workflow.nodes
          .filter(n => n.id !== selectedNode.id)
          .map(n => (
            <option key={n.id} value={n.id}>{n.label || n.id}</option>
          ))}
      </select>
    </label>

    {/* On False - node selector dropdown */}
    <label className="block text-sm">
      <span className="text-gray-700">On False (target node)</span>
      <select
        value={selectedNode.branch_config?.on_false || ''}
        onChange={(e) => onNodeUpdate({
          branch_config: { ...selectedNode.branch_config, on_false: e.target.value }
        })}
        className="mt-1 block w-full rounded border-gray-300"
      >
        <option value="">-- Select node --</option>
        {workflow.nodes
          .filter(n => n.id !== selectedNode.id)
          .map(n => (
            <option key={n.id} value={n.id}>{n.label || n.id}</option>
          ))}
      </select>
    </label>
  </div>
)}
- Edge management section (existing from WorkflowEditorPage)
- Delete node button
- Collapsible "Run Inputs" section (Goal, Label, JSON inputs) - **NOT a modal**

**JSON Validation and Parsing (Required):**

JSON text areas MUST:
1. **Parse and coerce** JSON into typed objects (not just validate strings)
2. **Validate on blur AND on Save/Run** (blur-only misses when user clicks without blurring)
3. **Scope gating properly** (Save only checks node fields, Run checks run inputs)
4. **Clear errors when node selection changes**

```typescript
// ===== Per-field error state =====
// Separate node field errors from run input errors for proper gating
const [nodeFieldErrors, setNodeFieldErrors] = useState<Record<string, string | null>>({});
const [runInputError, setRunInputError] = useState<string | null>(null);

// Clear node field errors when selected node changes
useEffect(() => {
  setNodeFieldErrors({});
}, [selectedNode?.id]);

// ===== Parse and validate JSON, returning typed object or error =====
function parseJson<T>(value: string, fieldName: string): { data: T | null; error: string | null } {
  if (!value.trim()) {
    return { data: null, error: null };
  }
  try {
    const parsed = JSON.parse(value) as T;
    return { data: parsed, error: null };
  } catch (e) {
    return {
      data: null,
      error: `Invalid JSON in ${fieldName}: ${e instanceof Error ? e.message : 'Parse error'}`
    };
  }
}

// ===== Validate and parse on blur - update error state =====
const handleJsonBlur = (value: string, field: string) => {
  const { error } = parseJson(value, field);
  setNodeFieldErrors(prev => ({ ...prev, [field]: error }));
};

// ===== Validate and parse on Save - writes typed object to node config =====
const handleSave = () => {
  // Re-validate all JSON fields before save (catches unblurred inputs)
  let hasErrors = false;
  const errors: Record<string, string | null> = {};

  // Validate subgraph input_mapping
  if (selectedNode?.type === 'subgraph' && inputMappingText) {
    const { data, error } = parseJson<Record<string, string>>(inputMappingText, 'input_mapping');
    errors['input_mapping'] = error;
    if (error) hasErrors = true;
    // If valid, write parsed object to node config
    if (data && !error) {
      onNodeUpdate({
        subgraph_config: { ...selectedNode.subgraph_config, input_mapping: data }
      });
    }
  }

  // Validate subgraph output_mapping
  if (selectedNode?.type === 'subgraph' && outputMappingText) {
    const { data, error } = parseJson<Record<string, string>>(outputMappingText, 'output_mapping');
    errors['output_mapping'] = error;
    if (error) hasErrors = true;
    if (data && !error) {
      onNodeUpdate({
        subgraph_config: { ...selectedNode.subgraph_config, output_mapping: data }
      });
    }
  }

  setNodeFieldErrors(errors);
  if (hasErrors) return; // Block save

  // Proceed with save...
  onSave();
};

// ===== Validate run inputs separately on Run =====
const handleRun = () => {
  // Validate run inputs JSON
  const { data: inputData, error } = parseJson<Record<string, unknown>>(runInputsText, 'run inputs');
  setRunInputError(error);
  if (error) return; // Block run

  // Proceed with run using parsed inputData...
  onRun(inputData);
};

// ===== Scoped gating =====
// Save button: only check node field errors (NOT run input errors)
const canSave = !Object.values(nodeFieldErrors).some(e => e !== null);

// Run button: only check run input error (NOT node field errors)
const canRun = !runInputError;

// In JSX:
<button onClick={handleSave} disabled={!canSave || isSaving}>Save</button>
<button onClick={handleRun} disabled={!canRun || isRunning}>Run</button>
```

This applies to:
- `subgraph_config.input_mapping` (Record<string, string>)
- `subgraph_config.output_mapping` (Record<string, string>)
- Run Inputs JSON textarea (Record<string, unknown>)

**Note:** `branch_config` uses structured dropdowns (not JSON textarea), so no JSON parsing needed there.

### 1.3 Modify WorkflowCanvas.tsx
**File:** `supervisor/studio/frontend/src/components/WorkflowCanvas.tsx`

Add drop zone support:

```typescript
interface WorkflowCanvasProps {
  // ... existing props
  onNodeAdd?: (nodeType: NodeType, position: { x: number; y: number }) => void;
}

// Inside component:
const onDrop = useCallback((event: React.DragEvent) => {
  event.preventDefault();
  // IMPORTANT: Respect read-only mode - don't allow drops when readOnly
  if (readOnly) return;

  const nodeType = event.dataTransfer.getData('application/reactflow-node-type');
  if (!nodeType || !onNodeAdd) return;

  const position = reactFlow.screenToFlowPosition({
    x: event.clientX,
    y: event.clientY,
  });
  onNodeAdd(nodeType as NodeType, position);
}, [onNodeAdd, reactFlow, readOnly]);

const onDragOver = useCallback((event: React.DragEvent) => {
  // IMPORTANT: Block drag-over visual feedback when readOnly
  if (readOnly) {
    event.dataTransfer.dropEffect = 'none';
    return;
  }
  event.preventDefault();
  event.dataTransfer.dropEffect = 'move';
}, [readOnly]);

// Add to ReactFlow container div:
// onDrop={onDrop}
// onDragOver={onDragOver}
```

### 1.4 Restructure WorkflowEditorPage.tsx to 3-Column Layout
**File:** `supervisor/studio/frontend/src/pages/WorkflowEditorPage.tsx`

**Keep right sidebar pattern** (simpler than bottom panel + modal):

```
┌─────────────┬────────────────────┬─────────────────┐
│ NodePalette │   WorkflowCanvas   │ PropertiesPanel │
│   (w-56)    │     (flex-1)       │    (w-80)       │
│  min-w-48   │    min-w-[400px]   │   min-w-64      │
│ collapsible │                    │  collapsible    │
└─────────────┴────────────────────┴─────────────────┘
```

- Left: NodePalette (new) - **collapsible with min-w-48**
- Center: WorkflowCanvas (existing) - **min-w-[400px] to preserve usability**
- Right: PropertiesPanel (extracted from current inline form) - **collapsible with min-w-64**
- Run Inputs: Collapsible section in PropertiesPanel (NOT a separate modal)

**Responsive Layout:** Add collapse buttons to sidebars for smaller screens. Use CSS `min-width` to prevent canvas from becoming unusable.

**Node ID Generation (Centralized in Parent):** Keep in WorkflowEditorPage (parent), pass `onNodeAdd` callback to WorkflowCanvas that:
1. Generates unique ID with collision prevention:
   ```typescript
   // Counter to prevent collision within same millisecond
   let nodeIdCounter = 0;

   function generateNodeId(nodeType: NodeType): string {
     const timestamp = Date.now().toString(36);
     const suffix = (nodeIdCounter++).toString(36);
     return `${nodeType}-${timestamp}-${suffix}`;
   }
   ```
2. Creates node with type-specific defaults from `getDefaultConfig()`
3. Calls `handleChange()` to trigger `hasChanges` state

**CRITICAL: Dirty State Management**
The `onNodeUpdate` callback from PropertiesPanel MUST call the parent's `handleChange()` function to properly trigger `hasChanges` state:

```typescript
// In WorkflowEditorPage.tsx
const handleNodeUpdate = useCallback((patch: Partial<Node>) => {
  if (!currentWorkflow || !selectedNode) return;
  const updatedNodes = currentWorkflow.nodes.map((node) =>
    node.id === selectedNode.id ? { ...node, ...patch } : node
  );
  handleChange({ ...currentWorkflow, nodes: updatedNodes }); // This triggers hasChanges
}, [currentWorkflow, selectedNode, handleChange]);

// Pass to PropertiesPanel
<PropertiesPanel
  selectedNode={selectedNode}
  onNodeUpdate={handleNodeUpdate}  // Bubbles up to trigger hasChanges
  ...
/>
```

---

## Phase 2: Execution Timeline

### 2.1 Create TraceTimeline.tsx
**File:** `supervisor/studio/frontend/src/components/TraceTimeline.tsx`

A simple chronological event list:

```typescript
interface TraceTimelineProps {
  events: TraceEvent[];
  selectedNodeId: string | null;  // UNIFIED: Use nodeId, not eventId
  onNodeSelect: (nodeId: string) => void;  // Emits nodeId to reuse existing selection state
  isLive: boolean;
}
```

**IMPORTANT:** Timeline emits `nodeId` (not `eventId`) so we can reuse the existing `selectedNodeId` state from ExecutionMonitor. This avoids having two sources of truth for selection.

**Features:**
- Vertical timeline with status indicators
- Each event shows: relative time, node label, status badge
- Click to select and show details in StateInspector
- "Live" indicator with pulse animation when execution is running
- Auto-scroll to latest event during live execution
- Simple CSS overflow-y-auto (no virtualization)

**IMPORTANT LIMITATION (Documented):**
> The timeline shows events from the current WebSocket session only. Events are NOT persisted -
> refreshing the page will show an empty timeline for in-progress executions. This is a known
> limitation. To see full history, check the StateInspector which loads from the REST API.

**Empty Timeline UI:** When timeline is empty (e.g., after refresh), show helpful message:
```typescript
{events.length === 0 && (
  <div className="p-4 text-sm text-gray-500 text-center">
    <div className="font-medium">No events yet</div>
    <div className="text-xs mt-1">
      Events appear here during live execution.
      {isLive ? ' Waiting for updates...' : ' Refresh to reconnect.'}
    </div>
    <div className="text-xs mt-2 text-gray-400">
      For persisted data, see Node Outputs below.
    </div>
  </div>
)}
```

### 2.2 Create StateInspector.tsx (Replaces NodeOutputPanel)
**File:** `supervisor/studio/frontend/src/components/StateInspector.tsx`

**IMPORTANT:** StateInspector REPLACES the existing `NodeOutputPanel` in ExecutionMonitor - do NOT add both. This avoids UI duplication.

Enhanced node output viewer:

```typescript
interface StateInspectorProps {
  selectedNodeId: string | null;
  nodeOutputs: Map<string, NodeExecutionStatus>;
  workflow: WorkflowGraph;
  globalState: Record<string, unknown>; // Aggregated from all completed nodes
}
```

**Sections (not tabs):**
- **Selected Node Output**: JSON tree for selected node's output (from REST API, persisted)
- **Global State**: Aggregated outputs from all completed nodes (computed on frontend)
- Copy-to-clipboard button for each section

**Global State Definition:**
```typescript
// Computed in ExecutionMonitor
const globalState = useMemo(() => {
  const state: Record<string, unknown> = {};
  for (const [nodeId, nodeStatus] of nodeOutputs) {
    if (nodeStatus.status === 'completed' && nodeStatus.output) {
      state[nodeId] = nodeStatus.output;
    }
  }
  return state;
}, [nodeOutputs]);
```

### 2.3 Timeline State (Local in ExecutionMonitor - NO separate hook)
**File:** `supervisor/studio/frontend/src/components/ExecutionMonitor.tsx`

**SIMPLIFICATION:** Keep timeline state LOCAL in ExecutionMonitor. Do NOT create a separate `useTraceStore` hook until there's a clear need for reuse across multiple components.

**IMPORTANT:** Timeline events MUST be deduplicated and sorted to handle:
- Out-of-order WebSocket messages
- Reconnections that may replay events
- `initial_state` messages that should reset the timeline

```typescript
// In ExecutionMonitor.tsx - simple local state
const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);

// Clear events when executionId changes
useEffect(() => {
  setTraceEvents([]);
}, [executionId]);

// Reset events when initial_state is received (reconnect scenario)
const handleInitialState = useCallback(() => {
  // Clear timeline on reconnect - fresh start from WebSocket
  setTraceEvents([]);
}, []);

// Add event handler with DEDUPLICATION and SORTING
const handleNodeUpdate = useCallback((
  nodeId: string,
  status: NodeStatus,
  _output: Record<string, unknown>,  // Not stored in event - use nodeOutputs map
  timestamp: string  // IMPORTANT: Must receive timestamp from WebSocket
) => {
  const nodeInfo = workflow.nodes.find(n => n.id === nodeId);
  const eventId = `${nodeId}-${timestamp}`;

  setTraceEvents(prev => {
    // DEDUPLICATE: Skip if event already exists
    if (prev.some(e => e.id === eventId)) {
      return prev;
    }

    const newEvent: TraceEvent = {
      id: eventId,
      timestamp,
      nodeId,
      nodeLabel: nodeInfo?.label || nodeId,
      nodeType: nodeInfo?.type || 'task',
      status,
      // NOTE: output is NOT stored here - read from nodeOutputs.get(nodeId) when needed
    };

    // SORT by timestamp (stable sort preserves order for same timestamp)
    const updated = [...prev, newEvent];
    updated.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    return updated;
  });
}, [workflow.nodes]);
```

Pass `handleInitialState` to `useExecutionWebSocket`:
```typescript
useExecutionWebSocket({
  // ... other options
  onInitialState: handleInitialState,  // Reset timeline on reconnect
  onNodeUpdate: handleNodeUpdate,
});
```

**Note:** The `onInitialState` callback should be called in the `initial_state` case of `useExecutionWebSocket` before processing nodes. This ensures reconnects start with a clean timeline.

**CRITICAL: WebSocket Timestamp Exposure**

The current `useExecutionWebSocket` hook discards the timestamp from `node_update` messages. You MUST update the callback signature:

```typescript
// BEFORE (current - missing timestamp):
onNodeUpdate?: (nodeId: string, status: NodeStatus, output: Record<string, unknown>) => void;

// AFTER (required for timeline):
onNodeUpdate?: (nodeId: string, status: NodeStatus, output: Record<string, unknown>, timestamp: string) => void;

// In handleMessage for 'node_update':
onNodeUpdateRef.current?.(message.node_id, message.status, message.output, message.timestamp);
```

**Note:** Events are derived from existing `node_update` WebSocket messages. Timeline is session-only (documented limitation).

### 2.4 Update ExecutionMonitor.tsx
**File:** `supervisor/studio/frontend/src/components/ExecutionMonitor.tsx`

Integrate TraceTimeline and StateInspector:

```
┌─────────────────────────────────────────────────────┐
│                    Header                           │
├─────────────┬───────────────────────────────────────┤
│  Trace      │                                       │
│  Timeline   │          WorkflowCanvas               │
│  (w-64)     │          (read-only)                  │
│             │                                       │
│  [Live      ├───────────────────────────────────────┤
│  events     │        StateInspector (h-56)          │
│  only]      │    (loads from REST API - persisted)  │
└─────────────┴───────────────────────────────────────┘
```

### 2.5 Add Types to workflow.ts
**File:** `supervisor/studio/frontend/src/types/workflow.ts`

```typescript
export interface TraceEvent {
  id: string;
  timestamp: string;
  nodeId: string;
  nodeLabel: string;
  nodeType: NodeType;
  status: NodeStatus;
  // NOTE: Do NOT store output here - outputs are already in nodeOutputs map
  // Storing output in events would duplicate data and increase memory usage
}
```

**IMPORTANT:** TraceEvent stores metadata only. To get output for an event, look up `nodeOutputs.get(event.nodeId)`. This avoids duplicating potentially large output payloads.

---

## Phase 3: Human-in-the-Loop (HITL)

### 3.1 Update ExecutionStatus Type
**File:** `supervisor/studio/frontend/src/types/workflow.ts`

```typescript
// Before
export type ExecutionStatus = 'running' | 'completed' | 'failed' | 'cancelled';

// After
export type ExecutionStatus = 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
```

**CRITICAL: Required Updates for 'interrupted' Status**

| File | Location | Change | Reason |
|------|----------|--------|--------|
| `ExecutionMonitor.tsx` | Line 46 | Change `enabled: execution?.status === 'running'` to `enabled: execution?.status === 'running' \|\| execution?.status === 'interrupted'` | **WebSocket must stay active for interrupted status to receive human_resolved** |
| `ExecutionMonitor.tsx` | Line 91-96 | Add `interrupted: 'text-amber-600'` to statusColors | Display color |
| `ExecutionMonitor.tsx` | Line 132 | Show ApprovalBanner when status is 'interrupted' | UI |
| `useExecutions.ts` | Line 54 | Change `data?.status === 'running' ? 2000 : false` to `(data?.status === 'running' \|\| data?.status === 'interrupted') ? 2000 : false` | **Polling must stay active for interrupted status** |
| `useExecutionWebSocket.ts` | Line 91 | Keep 'interrupted' OUT of terminalStatuses (it's not terminal) | WebSocket should not close on interrupted |
| `useExecutionWebSocket.ts` | Line 99 | Change `message.status !== 'running'` to `message.status !== 'running' && message.status !== 'interrupted'` | **Initial state handler must NOT fire completion callback for interrupted status** |

**IMPORTANT:** The `useExecutionWebSocket.ts` line 99 fix is critical. The current code fires `onExecutionComplete` for ANY non-running status in the initial_state handler. This would incorrectly finalize "interrupted" executions after a page refresh or late-connect:

```typescript
// BEFORE (BROKEN for interrupted):
if (message.status !== 'running' && !didCompleteRef.current) {
  didCompleteRef.current = true;
  onExecutionCompleteRef.current?.(message.status);
}

// AFTER (FIXED):
const terminalStatuses: ExecutionStatus[] = ['completed', 'failed', 'cancelled'];
if (terminalStatuses.includes(message.status) && !didCompleteRef.current) {
  didCompleteRef.current = true;
  onExecutionCompleteRef.current?.(message.status);
}
```

### 3.2 Add WebSocket Message Types
**File:** `supervisor/studio/frontend/src/types/workflow.ts`

```typescript
export interface WSHumanWaiting {
  type: 'human_waiting';
  node_id: string;
  title: string;
  description?: string;
  current_output?: Record<string, unknown>;
}

export interface WSHumanResolved {
  type: 'human_resolved';
  node_id: string;
  action: 'approve' | 'reject' | 'edit';
  // IMPORTANT: Backend MUST include resulting status to avoid optimistic desync
  // 'running' for approve/edit, 'failed' for reject
  status: ExecutionStatus;
}

// Update WSMessage union
export type WSMessage =
  | WSNodeUpdate
  | WSExecutionComplete
  | WSInitialState
  | WSHeartbeat
  | WSPong
  | WSError
  | WSHumanWaiting
  | WSHumanResolved;
```

### 3.3 Create ApprovalBanner.tsx
**File:** `supervisor/studio/frontend/src/components/ApprovalBanner.tsx`

```typescript
interface ApprovalBannerProps {
  nodeId: string;
  title: string;
  description?: string;
  currentOutput?: Record<string, unknown>;
  onApprove: () => void;
  onReject: (reason: string) => void;
  onEdit: (data: Record<string, unknown>) => void;
  isSubmitting: boolean;
}
```

- Sticky banner at top (amber/yellow background)
- Shows: Node label, title, description
- Action buttons: "Approve" (green), "Reject" (red), "Edit & Submit" (blue)
- Reject opens inline textarea for reason
- Edit opens ApprovalModal

### 3.4 Create ApprovalModal.tsx
**File:** `supervisor/studio/frontend/src/components/ApprovalModal.tsx`

```typescript
interface ApprovalModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentOutput: Record<string, unknown>;
  onSubmit: (editedData: Record<string, unknown>) => void;
  isSubmitting: boolean;
}
```

- Simple modal with JSON textarea (NOT a complex JSON editor)
- `JSON.parse()` validation on submit with error display
- Cancel and Submit buttons

### 3.5 Add Backend Endpoint and Requirements
**File:** `supervisor/studio/server.py`

```python
from pydantic import BaseModel
from typing import Literal

class HumanResponseRequest(BaseModel):
    node_id: str
    action: Literal['approve', 'reject', 'edit']
    feedback: str | None = None
    edited_data: dict | None = None

@app.post("/api/executions/{execution_id}/respond")
async def respond_to_human_node(execution_id: str, request: HumanResponseRequest):
    """Handle human response to interrupted execution."""
    # Validate execution exists and is interrupted
    # Forward response to supervisor engine
    # Return updated status
    return {"status": "resumed"}
```

**BACKEND REQUIREMENTS (Not fully specified in frontend plan):**

The following backend changes are required but outside the scope of frontend implementation:

1. **Status Transitions:** Backend must transition execution status to `interrupted` when a HumanNode is reached
2. **WebSocket Broadcasts:** Backend must emit `human_waiting` message when execution reaches HumanNode
3. **WebSocket Broadcasts:** Backend must emit `human_resolved` message after `/respond` endpoint processes the action, **including the resulting `status` field** (e.g., 'running' for approve, 'failed' for reject)
4. **Status in DB:** ExecutionStatus enum in backend must include `interrupted`
5. **Status in API:** All execution-related endpoints must return `interrupted` as a valid status
6. **Edited Output Storage:** If action is `edit`, backend must store the `edited_data` and use it for subsequent nodes

**Implementation Note:** Coordinate with backend team to ensure these requirements are met before HITL Phase 3 can be fully tested.

### 3.6 Add API Client Function
**File:** `supervisor/studio/frontend/src/api/client.ts`

**IMPORTANT:** Use the existing `fetchApi` wrapper (not raw `fetch`) for consistent error handling via `ApiError`:

```typescript
export interface HumanResponseParams {
  executionId: string;
  nodeId: string;
  action: 'approve' | 'reject' | 'edit';
  feedback?: string;
  editedData?: Record<string, unknown>;
}

export async function respondToHumanNode(params: HumanResponseParams): Promise<{ status: string }> {
  // Use fetchApi for consistent error handling (throws ApiError on non-2xx)
  return fetchApi<{ status: string }>(
    `/executions/${encodeURIComponent(params.executionId)}/respond`,
    {
      method: 'POST',
      body: JSON.stringify({
        node_id: params.nodeId,
        action: params.action,
        feedback: params.feedback,
        edited_data: params.editedData,
      }),
    }
  );
}
```

This ensures:
- Consistent error handling via `ApiError` class (status code + detail)
- Automatic `Content-Type: application/json` header
- URL encoding for execution ID
- Same error pattern as all other API functions in the file

### 3.7 Update useExecutionWebSocket.ts
**File:** `supervisor/studio/frontend/src/hooks/useExecutionWebSocket.ts`

Add handlers for HITL messages:

```typescript
export interface UseExecutionWebSocketOptions {
  // ... existing
  onInitialState?: () => void;  // Called before processing initial_state (for timeline reset)
  onHumanWaiting?: (nodeId: string, title: string, description?: string, currentOutput?: Record<string, unknown>) => void;
  onHumanResolved?: (nodeId: string, action: 'approve' | 'reject' | 'edit', status: ExecutionStatus) => void;
}

// In handleMessage switch:
case 'human_waiting':
  // Update execution status to 'interrupted'
  queryClient.setQueryData<ExecutionResponse>(
    executionKeys.detail(executionId!),
    (old) => old ? { ...old, status: 'interrupted' } : old
  );
  onHumanWaitingRef.current?.(message.node_id, message.title, message.description, message.current_output);
  break;

case 'human_resolved':
  // Update execution status from message (NOT optimistic - use actual backend status)
  // This handles approve → 'running', reject → 'failed', etc.
  queryClient.setQueryData<ExecutionResponse>(
    executionKeys.detail(executionId!),
    (old) => old ? { ...old, status: message.status } : old
  );
  onHumanResolvedRef.current?.(message.node_id, message.action, message.status);
  break;
```

---

## Phase 4: Visual Polish

### 4.1 Animated Edges During Execution
**File:** `supervisor/studio/frontend/src/components/WorkflowCanvas.tsx`

Update edge styling based on execution state:

```typescript
interface WorkflowCanvasProps {
  // ... existing
  activeNodeIds?: Set<string>; // Nodes currently running
}

function toFlowEdges(edges: Edge[], activeNodeIds?: Set<string>): FlowEdge[] {
  const active = activeNodeIds || new Set();
  return edges.map((edge, index) => {
    const isActive = active.has(edge.source) || active.has(edge.target);
    return {
      id: edge.id || `${edge.source}-${edge.target}-${index}`,
      source: edge.source,
      target: edge.target,
      animated: isActive || edge.is_loop_edge,
      style: isActive
        ? { stroke: '#f59e0b', strokeWidth: 2 }
        : edge.is_loop_edge
        ? { strokeDasharray: '5,5' }
        : undefined,
      label: edge.condition,
      data: {
        id: edge.id,
        condition: edge.condition,
        is_loop_edge: edge.is_loop_edge,
        data_mapping: edge.data_mapping,
      },
    };
  });
}
```

---

## Files Summary

### New Files (6)
| File | Phase | Description |
|------|-------|-------------|
| `src/components/NodePalette.tsx` | 1 | Draggable node type palette (all 7 types, text badges) |
| `src/components/PropertiesPanel.tsx` | 1 | Extracted property editor (right sidebar, per-field JSON validation) |
| `src/components/TraceTimeline.tsx` | 2 | Simple execution event timeline (live-only, empty state UI) |
| `src/components/StateInspector.tsx` | 2 | Node output viewer (REPLACES NodeOutputPanel) |
| `src/components/ApprovalBanner.tsx` | 3 | HITL approval banner |
| `src/components/ApprovalModal.tsx` | 3 | HITL edit modal |

**Note:** `useTraceStore.ts` was removed - timeline state is kept local in ExecutionMonitor for simplicity.

### Modified Files (8)
| File | Phases | Changes |
|------|--------|---------|
| `src/pages/WorkflowEditorPage.tsx` | 1 | 3-column layout with palette, use PropertiesPanel, centralized node ID generation |
| `src/types/workflow.ts` | 2, 3 | TraceEvent type, HITL message types, 'interrupted' status |
| `src/components/WorkflowCanvas.tsx` | 1, 4 | Drop handlers (with readOnly check), animated edges |
| `src/components/ExecutionMonitor.tsx` | 2, 3 | TraceTimeline, StateInspector (replaces NodeOutputPanel), local timeline state, ApprovalBanner, 'interrupted' handling |
| `src/hooks/useExecutionWebSocket.ts` | 2, 3 | **Add timestamp to onNodeUpdate callback**, HITL message handlers, fix initial-state completion for 'interrupted' |
| `src/hooks/useExecutions.ts` | 3 | Add 'interrupted' to refetchInterval check |
| `src/api/client.ts` | 3 | respondToHumanNode function |
| `supervisor/studio/server.py` | 3 | /respond endpoint |

---

## Removed from Original Plan

| Item | Reason |
|------|--------|
| **Phase 2: Mode Switching** | Conflicts with existing routing |
| **EditorModeContext.tsx** | Not needed without mode switching |
| **ModeToggle.tsx** | Not needed without mode switching |
| **Tabbed PropertiesPanel** | Over-engineered |
| **Tabbed StateInspector** | Over-engineered |
| **Time-travel debugging** | Complex, needs backend support |
| **react-window virtualization** | Premature optimization |
| **TraceEventItem.tsx** | Inline in TraceTimeline is simpler |
| **useStreamingOutput.ts** | No backend support, defer entirely |
| **Bottom panel layout** | More complex than right sidebar |
| **Run Inputs modal** | Keep in sidebar as collapsible section |
| **useTraceStore.ts** | Over-engineered; local state in ExecutionMonitor is simpler |
| **Lucide icons** | Not in dependencies; use text badges instead |
| **NodeOutputPanel duplication** | StateInspector REPLACES it, doesn't add to it |

---

## Known Limitations (Documented)

| Limitation | Reason | Workaround |
|------------|--------|------------|
| **Timeline is live-only** | Events from WebSocket only, not persisted | StateInspector loads from REST API and shows all node outputs |
| **Timeline empty on refresh** | No history endpoint | Check StateInspector for persisted data |
| **No time-travel debugging** | Needs backend history endpoint | Deferred to future iteration |

---

## Phase Dependencies

```
Phase 1 (Editor Enhancements)
     │
     └─→ Phase 2 (Execution Timeline)
              │
              └─→ Phase 3 (HITL)
                       │
                       └─→ Phase 4 (Visual Polish)
```

Each phase is independently deployable and maintains backward compatibility.

---

## Verification Plan

### Phase 1 Verification
1. Open WorkflowEditorPage, verify NodePalette appears on left
2. Drag a Task node from palette to canvas, verify it creates node at drop position with default task_config
3. Drag a Gate node, verify it has default gate_config
4. Drag a Subgraph node, verify it has default subgraph_config
5. Select node, verify PropertiesPanel shows type-specific fields
6. Edit properties, verify changes persist after save
7. Verify Run Inputs section is collapsible in PropertiesPanel

### Phase 2 Verification
1. Navigate to ExecutionPage for a running execution
2. Verify TraceTimeline shows events as nodes execute
3. Verify "Live" indicator pulses during execution
4. Click a trace event, verify StateInspector shows that node's output
5. Verify Global State section shows aggregated outputs
6. Refresh page, verify timeline is empty but StateInspector shows persisted data

### Phase 3 Verification
1. Execute workflow with HumanNode
2. Verify ApprovalBanner appears when execution status becomes 'interrupted'
3. **Verify WebSocket stays connected during 'interrupted' status**
4. **Verify polling continues during 'interrupted' status**
5. Click Approve, verify execution resumes (status returns to 'running')
6. Test Reject flow, verify execution fails with reason
7. Test Edit flow, verify modified data is sent to backend

### Phase 4 Verification
1. Start execution, verify edges animate when connected nodes are running
2. Verify active edges have amber stroke color

---

## Implementation Order

1. **Phase 1.2** - Create PropertiesPanel (extract from WorkflowEditorPage, per-field JSON validation)
2. **Phase 1.1** - Create NodePalette with all 7 node types (text badges, no icons)
3. **Phase 1.3** - Add drop handlers to WorkflowCanvas (with readOnly check)
4. **Phase 1.4** - Restructure WorkflowEditorPage to 3-column layout
5. **Phase 2.5** - Add TraceEvent type to workflow.ts
6. **Phase 2.3** - Update useExecutionWebSocket to pass timestamp in onNodeUpdate callback
7. **Phase 2.1** - Create TraceTimeline component (with empty state UI)
8. **Phase 2.2** - Create StateInspector component (replaces NodeOutputPanel)
9. **Phase 2.4** - Update ExecutionMonitor layout (local timeline state, replace NodeOutputPanel)
10. **Phase 3.1** - Add 'interrupted' status and update ALL references (6 locations including line 99 fix)
11. **Phase 3.2** - Add HITL WebSocket message types
12. **Phase 3.5** - Add backend /respond endpoint (coordinate with backend team)
13. **Phase 3.6** - Add respondToHumanNode API client
14. **Phase 3.7** - Update useExecutionWebSocket with HITL handlers
15. **Phase 3.3** - Create ApprovalBanner
16. **Phase 3.4** - Create ApprovalModal
17. **Phase 4.1** - Add animated edges

---

## Future Enhancements (Deferred)

These items were identified but deferred to avoid over-engineering:

1. **Time-travel debugging**: Requires backend `GET /api/executions/{id}/history` endpoint
2. **Timeline persistence**: Requires backend event storage
3. **List virtualization**: Add react-window if timeline performance degrades with >500 events
4. **Streaming output**: Requires backend streaming support
5. **Undo/redo**: Editor state history management

---

## Questions Resolved

| Original Question | Resolution |
|------------------|------------|
| Time-Travel Display | Deferred. Timeline is live-only (documented) |
| Global State Definition | Aggregated outputs from completed nodes, computed on frontend |
| HITL Edit Scope | Edit only current node's output |
| Streaming Display | Deferred entirely (no hook created) |
| Trace Persistence | Not implemented; documented as known limitation |
| Subgraph Support | Included in NodePalette and PropertiesPanel |
| Layout Complexity | Keep right sidebar pattern, no bottom panel |
| WebSocket for Interrupted | Explicitly enabled for 'running' OR 'interrupted' |
