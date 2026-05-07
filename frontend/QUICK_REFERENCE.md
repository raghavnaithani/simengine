# Decision Graph Simulator - Quick Reference Guide
## API Endpoints & Function Signatures

---

## Table of Contents
1. [API Endpoints](#api-endpoints)
2. [Hook Signatures](#hook-signatures)
3. [TypeScript Types](#typescript-types)
4. [Store Actions](#store-actions)
5. [Component Props](#component-props)
6. [Constants & Configuration](#constants--configuration)

---

## API Endpoints

### Health & Status

#### GET /health
Check backend connectivity.
```typescript
// Request
GET http://localhost:8000/health

// Response (200)
{
  "status": "ok"
}

// Usage
const health = await api.health();
```

---

### Simulation Lifecycle

#### POST /simulate/start
Initialize a new simulation with scenario and context.
```typescript
// Request
POST /simulate/start
Content-Type: application/json

{
  "scenario": "How should we handle supply chain disruption?",
  "context_answers": {
    "timeline": "within_6_months",
    "budget": "limited",
    "stakeholders": "cross_functional"
  },
  "persona": "Risk Manager"
}

// Response (201)
{
  "job_id": "job_abc123xyz"
}

// Usage
const response = await api.startSimulation({
  scenario: string,
  context_answers?: Record<string, string>,
  persona?: string
});
// Returns: { job_id: string }
```

---

#### GET /jobs/{job_id}
Poll job status (used by useJobPollingWithDynamicBackoff).
```typescript
// Request
GET /jobs/job_abc123xyz

// Response (200)
{
  "job_id": "job_abc123xyz",
  "type": "start",
  "status": "processing",
  "created_at": "2026-04-20T10:30:00Z",
  "updated_at": "2026-04-20T10:30:05Z"
}

// When completed:
{
  "job_id": "job_abc123xyz",
  "type": "start",
  "status": "completed",
  "result": {
    "node_id": "node_root_001",
    "session_id": "sess_1234567890"
  }
}

// When failed:
{
  "job_id": "job_abc123xyz",
  "type": "start",
  "status": "failed",
  "error": "LLM API unavailable"
}

// Usage (via hook)
const { isDone, status, job, error, pollAttempts } = 
  useJobPollingWithDynamicBackoff(jobId, true);

// Manual call
const job = await api.getJob(jobId);
```

**Polling Behavior:**
- Initial interval: 1500ms
- After 10 polls: 3000ms
- After 30 polls: marks as "flaky" 
- Auto-stops on: completed, failed, or 404
- Retries network errors 3 times

---

#### GET /graph?session_id=SESSION_ID
Fetch the complete decision graph (nodes and edges).
```typescript
// Request
GET /graph?session_id=sess_1234567890

// Response (200)
{
  "nodes": [
    {
      "id": "node_root_001",
      "title": "Supply Chain Crisis Assessment",
      "description": "Initial evaluation of disruption impact...",
      "summary": "Evaluate impact scope and severity",
      "type": "scenario",
      "metadata": {
        "confidence": 0.85,
        "criticality": "high"
      },
      "source_citations": [
        "https://example.com/supply-chain-risks",
        "cache_id_12345"
      ]
    },
    {
      "id": "node_dec_001",
      "title": "Activate emergency suppliers?",
      "description": "Consider engaging backup suppliers...",
      "summary": "Decide on supplier activation",
      "type": "decision",
      "metadata": {
        "cost_impact": "high",
        "timeline": "immediate"
      },
      "source_citations": []
    }
  ],
  "edges": [
    {
      "from": "node_root_001",
      "to": "node_dec_001",
      "action": "evaluate_severity"
    },
    {
      "from": "node_dec_001",
      "to": "node_outcome_001",
      "action": "activate_suppliers"
    }
  ]
}

// Usage
const graph = await api.getGraphAfterJob(sessionId);
// Returns: { nodes: DecisionNode[], edges: GraphEdge[] }

// Or via helper (used in world-building)
const { nodes, edges } = await loadGraphAfterJobCompletion(
  sessionId,
  nodeId // optional: to filter/highlight specific node
);
```

**Status Codes:**
- 200: Success
- 404: Session not found
- 500: Server error

---

### Branching & Alternatives

#### POST /simulate/branch
Create a branch from existing node (generates alternative path).
```typescript
// Request
POST /simulate/branch
Content-Type: application/json

{
  "session_id": "sess_1234567890",
  "parent_node_id": "node_dec_001",
  "action": "activate_suppliers",
  "persona": "Risk Manager",
  "seed": null
}

// Response (201)
{
  "job_id": "job_branch_xyz789"
}

// Usage
const response = await api.branch({
  session_id: string,
  parent_node_id: string,
  action: string,           // Alternative label or custom prompt
  persona?: string,
  seed?: number | null
});
// Returns: { job_id: string }

// This starts an async job that creates a new child node
// Then poll /jobs/{job_id} to get result
```

**Important:** Branching uses the same job polling pattern as simulation start.

---

#### POST /jobs/{job_id}/retry
Retry a failed job.
```typescript
// Request
POST /jobs/job_abc123xyz/retry

// Response (201)
{
  "job_id": "job_abc123xyz_retry1"  // New job ID
}

// Usage
const retryResponse = await api.retryJob(jobId);
// Returns: { job_id: string }

// Then poll the new job ID
const { isDone, status, job } = 
  useJobPollingWithDynamicBackoff(retryResponse.job_id, true);
```

---

### Node Details

#### GET /nodes/{node_id}
Get detailed information about a specific node.
```typescript
// Request
GET /nodes/node_dec_001

// Response (200)
{
  "id": "node_dec_001",
  "title": "Activate emergency suppliers?",
  "description": "Consider engaging backup suppliers to mitigate disruption...",
  "summary": "Decide on supplier activation",
  "type": "decision",
  "metadata": {
    "cost_impact": "high",
    "timeline": "immediate",
    "probability": 0.7
  },
  "source_citations": [
    "https://example.com/supplier-activation",
    "https://example.com/risk-mitigation"
  ]
}

// Usage
const node = await api.getNode(nodeId);
// Returns: DecisionNode
```

---

### Logging & Analytics

#### GET /jobs/{job_id}/logs
Retrieve logs from a job.
```typescript
// Request
GET /jobs/job_abc123xyz/logs

// Response (200)
{
  "logs": [
    "2026-04-20T10:30:01Z - Starting job processing",
    "2026-04-20T10:30:02Z - Calling LLM API",
    "2026-04-20T10:30:04Z - LLM response received",
    "2026-04-20T10:30:05Z - Job completed"
  ]
}

// Usage
const { logs } = await api.getJobLogs(jobId);
```

---

#### POST /log
Client telemetry sink for tracking user actions and metrics.
```typescript
// Request
POST /log
Content-Type: application/json

{
  "timestamp": "2026-04-20T10:35:12.456Z",
  "event": "ui.branch.complete",
  "session_id": "sess_1234567890",
  "metadata": {
    "branch_latency_ms": 2340,
    "job_poll_attempts": 2,
    "parent_node_id": "node_dec_001",
    "new_node_id": "node_dec_001_alt1"
  }
}

// Response (202)
{
  "success": true
}

// Usage (automatic)
const { trackBranchCompletion } = useTelemetry();
await trackBranchCompletion(
  jobId,
  latencyMs,
  pollAttempts,
  sessionId
);

// Or manual
await api.postClientLog({
  timestamp: new Date().toISOString(),
  event: "custom.event",
  session_id: sessionId,
  metadata: { custom: "data" }
});
```

**Event Types:**
```
ui.simulation.start
ui.simulation.complete
ui.branch.created
ui.branch.complete
ui.branch.failed
ui.node.focused
ui.export.requested
ui.error
```

---

## Hook Signatures

### useJobPollingWithDynamicBackoff

Poll job status with adaptive backoff interval.

```typescript
function useJobPollingWithDynamicBackoff(
  jobId: string | null,
  enabled: boolean = true
): {
  isDone: boolean,              // true when polling stops
  status: JobStatus,            // queued, processing, completed, failed
  job: Job | null,             // Latest job data
  error: string | null,        // Error message if failed
  pollAttempts: number,        // Iterations completed
  isFlaky: boolean             // true if > 30 polls
}
```

**Usage:**
```typescript
const { isDone, status, job, error, pollAttempts, isFlaky } =
  useJobPollingWithDynamicBackoff(jobId, jobId !== null);

useEffect(() => {
  if (isDone && status === 'completed') {
    // Handle success
    const nodeId = job?.result?.node_id;
  } else if (isDone && status === 'failed') {
    // Handle error
    console.error(error);
  }
}, [isDone, status]);
```

**Polling Algorithm:**
```
Attempt 1-10: 1500ms interval
Attempt 11-30: 3000ms interval
Attempt 31+: Marked as "flaky"
Stops on: completed, failed, 404, or after 30 attempts
Auto-retries network errors: 3 times
```

---

### useTelemetry

Track user actions and metrics automatically.

```typescript
function useTelemetry(): {
  trackUserAction(
    event: string,
    nodeId: string,
    metadata?: Record<string, unknown>
  ): Promise<void>,
  
  trackSimulationComplete(
    jobId: string,
    durationMs: number,
    attempts: number
  ): Promise<void>,
  
  trackBranchCompletion(
    jobId: string,
    durationMs: number,
    attempts: number,
    sessionId: string
  ): Promise<void>,
  
  trackBranchFailure(
    jobId: string,
    error: string,
    attempts: number,
    sessionId: string
  ): Promise<void>,
  
  trackError(
    component: string,
    error: Error
  ): Promise<void>
}
```

**Usage:**
```typescript
const { trackBranchCompletion, trackError } = useTelemetry();

try {
  const startTime = Date.now();
  // ... do something
  await trackBranchCompletion(
    jobId,
    Date.now() - startTime,
    pollAttempts,
    sessionId
  );
} catch (error) {
  await trackError('MyComponent', error as Error);
}
```

**Features:**
- Auto-collects: timestamp, session_id, component name
- Silently fails if /log unavailable (won't break UI)
- Includes latency and attempt metrics
- Error details logged automatically

---

### useToast

Display notifications to user.

```typescript
function useToast(): {
  addToast(notification: {
    type: 'success' | 'error' | 'warning' | 'info',
    title: string,
    message: string,
    duration?: number,    // ms (default: 5000)
    action?: {
      label: string,
      action: string,    // 'retry' | 'view-logs' | custom
      jobId?: string
    }
  }): void
}
```

**Usage:**
```typescript
const { addToast } = useToast();

addToast({
  type: 'success',
  title: 'Branch Created',
  message: 'New decision node added to tree',
  duration: 4000
});

addToast({
  type: 'error',
  title: 'Branch Failed',
  message: 'Could not generate alternative path',
  action: {
    label: 'Retry',
    action: 'retry',
    jobId: failedJobId
  }
});
```

**Toast Types:**
- `success` - Green background, checkmark icon
- `error` - Red background, X icon
- `warning` - Orange background, alert icon
- `info` - Blue background, info icon

---

### useKeyboardShortcuts

Register keyboard shortcuts.

```typescript
function useKeyboardShortcuts(callbacks: {
  'Cmd+K'?: () => void,      // Quick search
  'Space'?: () => void,       // Open focus panel
  '+'?: () => void,           // Zoom in
  '-'?: () => void,           // Zoom out
  'f'?: () => void,           // Fit to view
  'Escape'?: () => void       // Close modal
}): void
```

**Usage:**
```typescript
const { fitToView, zoomIn } = useReactFlow();
const { selectedNodeId } = useSessionStore();

useKeyboardShortcuts({
  'f': () => fitToView(),
  '+': () => zoomIn(),
  'Escape': () => closeModal(),
  'Space': () => openFocusPanel(selectedNodeId)
});
```

---

### useSessionStore (Zustand)

Access and modify simulation state.

```typescript
function useSessionStore(): {
  // State
  sessionId: string | null,
  nodes: DecisionNode[],
  edges: GraphEdge[],
  selectedNodeId: string | null,
  ghostNodes: GhostNode[],
  
  // Actions
  setSessionId(id: string | null): void,
  addNode(node: DecisionNode): void,
  addEdge(edge: GraphEdge): void,
  selectNode(id: string): void,
  addGhostNode(ghost: GhostNode): void,
  removeGhostNode(tempId: string): void,
  updateGhostNode(tempId: string, updates: Partial<GhostNode>): void,
  clearSession(): void,
  
  // Computed
  nodeCount(): number,
  edgeCount(): number,
  selectedNode(): DecisionNode | undefined,
  getNodeById(id: string): DecisionNode | undefined,
  getEdgesBySourceId(id: string): GraphEdge[],
  getChildNodes(id: string): DecisionNode[]
}
```

**Usage:**
```typescript
const { nodes, addNode, selectNode } = useSessionStore();

// Read state
console.log('Current nodes:', nodes);

// Update state
addNode({
  id: 'node_new',
  title: 'New Decision',
  // ... other fields
});

// Select for UI
selectNode('node_abc');
```

---

## TypeScript Types

### Core Types

#### DecisionNode
Represents a single node in the decision tree.

```typescript
interface DecisionNode {
  id: string;                      // Unique identifier
  title: string;                   // Display name
  description: string;             // Full explanation
  summary: string;                 // Brief version
  type: 'scenario' | 'decision' | 'outcome' | 'context';
  metadata?: {
    confidence?: number;           // 0-1 confidence score
    severity?: 'low' | 'medium' | 'high';
    timeline?: string;             // e.g., "immediate", "6_months"
    cost_impact?: 'low' | 'medium' | 'high';
    [key: string]: unknown;        // Custom fields
  };
  source_citations?: string[];     // URLs or cache IDs
  created_at?: string;             // ISO 8601
  updated_at?: string;
}
```

---

#### GraphEdge
Represents a connection between nodes.

```typescript
interface GraphEdge {
  id: string;                      // Unique identifier
  from: string;                    // Source node ID (backend)
  to: string;                      // Target node ID (backend)
  action?: string;                 // Decision/action label
  created_at?: string;
}
```

**Note:** Frontend adapts `from`/`to` → `source`/`target` for ReactFlow.

---

#### Job
Represents an async job (simulation or branch).

```typescript
interface Job {
  job_id: string;
  type: 'start' | 'branch';
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress?: number;               // 0-100
  created_at?: string;             // ISO 8601
  updated_at?: string;
  result?: {
    node_id: string;               // New/root node ID
    session_id?: string;           // Session ID
  } | null;
  error?: string | null;           // Error message
  payload?: Record<string, unknown>; // Request payload
}
```

---

#### GhostNode
Optimistic UI representation of pending node.

```typescript
interface GhostNode {
  temp_id: string;                 // Temporary ID (e.g., "ghost_123")
  parent_id: string;               // Parent node ID
  status: 'pending' | 'loading' | 'error' | 'success';
  created_at: string;
  action: string;                  // Alternative description
  error?: string;                  // Error message
}
```

---

#### Request Types

```typescript
interface StartSimulationRequest {
  scenario: string;                // User's scenario description
  context_answers?: Record<string, string>;
  persona?: string;                // Decision-maker role
}

interface BranchRequest {
  session_id: string;
  parent_node_id: string;
  action: string;                  // Alternative or custom prompt
  persona?: string;
  seed?: number | null;            // For reproducibility
}

interface ClientLogEvent {
  timestamp: string;               // ISO 8601
  event: string;                   // Event type
  session_id: string;
  metadata?: Record<string, unknown>;
}
```

---

#### Response Types

```typescript
interface GraphResponse {
  nodes: DecisionNode[];
  edges: Array<{
    from: string;
    to: string;
    action?: string;
    created_at?: string;
  }>;
}

interface JobResponse {
  job_id: string;
}

interface HealthResponse {
  status: 'ok';
}
```

---

## Store Actions

All actions via `useSessionStore`:

```typescript
// Setters
setSessionId(id: string | null)
addNode(node: DecisionNode)
addEdge(edge: GraphEdge)
selectNode(id: string)
addGhostNode(ghost: GhostNode)
removeGhostNode(tempId: string)
updateGhostNode(tempId: string, updates: Partial<GhostNode>)
clearSession()
setNodes(nodes: DecisionNode[])
setEdges(edges: GraphEdge[])

// Getters
nodeCount() → number
edgeCount() → number
selectedNode() → DecisionNode | undefined
getNodeById(id) → DecisionNode | undefined
getEdgesBySourceId(id) → GraphEdge[]
getChildNodes(id) → DecisionNode[]
```

---

## Component Props

### Simulator (Main)
```typescript
interface SimulatorProps {
  // None - uses Zustand store internally
}
```

### PromptModal
```typescript
interface PromptModalProps {
  open: boolean;
  onSubmit: (data: { mission: string; scenario: string }) => void;
}
```

### Questionnaire
```typescript
interface QuestionnaireProps {
  scenario: string;
  answers: Record<string, string>;
  onAnswersChange: (answers: Record<string, string>) => void;
  onComplete: (metadata: Record<string, unknown>) => void;
}
```

### GraphCanvas
```typescript
interface GraphCanvasProps {
  nodes: DecisionNode[];
  edges: GraphEdge[];
  selectedNode: DecisionNode | null;
  onNodeSelect: (node: DecisionNode) => void;
  onNodeBranch: (nodeId: string, alternative: Alternative) => void;
}
```

### FocusPanel
```typescript
interface FocusPanelProps {
  node: DecisionNode | null;
  onClose: () => void;
}
```

### BranchDialog
```typescript
interface BranchDialogProps {
  open: boolean;
  node: DecisionNode | null;
  alternatives: Alternative[];
  onBranch: (alternative: Alternative, customPrompt?: string) => void;
  onClose: () => void;
}
```

---

## Constants & Configuration

### API Configuration
```typescript
// lib/api.ts
const API_BASE_URL = 'http://localhost:8000';
const REQUEST_TIMEOUT = 30000;  // 30 seconds
const RETRY_COUNT = 3;
const RETRY_BACKOFF = [100, 200, 400]; // ms delays
```

### Polling Configuration
```typescript
// hooks/use-job-polling.ts
const INITIAL_POLL_INTERVAL = 1500;  // ms
const EXTENDED_POLL_INTERVAL = 3000; // ms (after 10 polls)
const POLL_THRESHOLD = 10;            // When to extend interval
const MAX_POLLS = 30;                 // Max before marking flaky
const NETWORK_ERROR_RETRIES = 3;
```

### Toast Configuration
```typescript
// components/simulator/toast-provider.tsx
const DEFAULT_DURATION = 5000;  // ms (auto-dismiss)
const MAX_TOASTS = 5;          // Max visible toasts
const ANIMATION_DURATION = 300; // ms (enter/exit)
```

### UI Constants
```typescript
// Graph visualization
const NODE_WIDTH = 200;
const NODE_HEIGHT = 100;
const EDGE_STROKE_WIDTH = 2;

// Graph layout
const SPRING_FORCE = 1;
const DAMPING = 0.8;
const SPRING_LENGTH = 300;
```

---

## Error Codes

### HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Use response data |
| 201 | Created | Use response (usually job_id) |
| 202 | Accepted | Telemetry recorded |
| 400 | Bad Request | Check request format |
| 404 | Not Found | Resource deleted/expired |
| 429 | Too Many Requests | Implement backoff |
| 500 | Server Error | Retry after delay |
| 503 | Unavailable | Retry with exponential backoff |

### Polling Stop Conditions

| Condition | Action |
|-----------|--------|
| status='completed' | Success - load data |
| status='failed' | Error - show message |
| 404 response | Job lost - stop polling |
| 30+ polls | Flaky - suggest manual retry |
| Network error (3x) | Timeout - stop polling |

---

**For more details, see README.md and IMPLEMENTATION_DETAILS.md**
