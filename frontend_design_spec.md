🎨 MASTER FRONTEND SPECIFICATION: Decision Graph Simulator (UI/UX)

1. Design Philosophy & Aesthetic
Theme: "Strategic Command Center" / "Bloomberg Terminal meets Sci-Fi Strategy Game." Core Vibe: High-density data, dark mode only, trust-centric, distraction-free.

1.1 Color Palette (Semantic Risk System)
We do not use random colors. Every color has a logic mapping to the Backend Data Models.

Canvas Background: #0B1120 (Deepest Navy/Black) — Reduces eye strain for long sessions.

Surface / Panels: #1E293B (Slate 800) with backdrop-filter: blur(12px) and opacity: 0.95.

Borders: #334155 (Slate 700) — Subtle separation.

Typography:

Headings: Inter (Bold, Tracking Tight). Color: #F8FAFC (Slate 50).

Data/Citations: JetBrains Mono (Monospace). Color: #94A3B8 (Slate 400).

1.2 The "Status" Colors (Critical for Decision Nodes)
Safe / Stable: #10B981 (Emerald 500) — Used for nodes with Low Risk.

Caution: #F59E0B (Amber 500) — Used for Medium Risk.

Critical Danger: #EF4444 (Red 500) — Used for High/Critical Risk (Mandatory Backend Rule).

Speculative / Unverified: #A855F7 (Purple 500) — Used when confidence_score < 0.5.

System / AI Thinking: #3B82F6 (Blue 500) — Pulsing animations.

2. Overview — What this spec contains
- Full screen inventory: every screen, overlay, modal, and their states.
- Full component inventory: props, events, states, ARIA roles, example mock JSON.
- Data contracts: exact request/response shapes used by the app.
- Interaction patterns: optimistic UI, polling, error handling, retries.
- Accessibility, localization, testing matrix, telemetry, and performance guidance.

3. Screen Inventory (complete)
Each screen entry lists purpose, entry points, primary actions, data inputs, and exit transitions.

3.1 Mission Start (Landing Overlay)
- Purpose: Accept user scenario prompt and parameters to start a simulation session.
- Entry: App start or New Session button.
- Primary controls: `prompt` textarea, `mode` segmented control (Quick / Analytical), `persona` dropdown, advanced toggles (`simulate_steps`, `temperature`, `seed`).
- Validation: `prompt.length >= 10` required; show inline helper if invalid.
- Secondary actions: Load last session (if present), import session JSON.
- Transition: On submit, call `POST /simulate/start` → receive `session_id` & `job_id` → show World Building screen and job log.

3.2 World Building (Loading / Terminal)
- Purpose: Visualize backend work (ContextBuilder, scraping, chunking, embedding, ReasoningEngine warm-ups).
- UI: TerminalLog (animate typed lines), progress bar, estimated ETA, cancel button.
- Data: `job_id`, structured logs streamed via SSE/WebSocket or pulled via `GET /jobs/{job_id}/log`.
- Behavior: Terminal log lines are timestamped and can be filtered by level.
- Transition: On `job.status === completed`, animate root node appearing and move to Infinite Canvas.

3.3 Infinite Canvas (Graph View)
- Purpose: Primary interactive space; visualize decision graph, pan/zoom, select nodes, branch.
- Entry: After world building completes or when loading an existing session via `GET /graph/{session_id}`.
- Layout: Top-to-bottom tree (auto-laid-out via Dagre). Canvas shows grid (toggleable), rulers, and fit-to-view controls.
- Main interactions: select node, open Focus Panel, branch from node, reorder (programmatic only), zoom/pan.

3.4 Focus Panel (Node Details — Right Sidebar)
- Purpose: Show full node data: narrative, evidence, risks, alternatives, JSON inspector.
- Tabs: Narrative, Evidence, Risks, JSON
- Lazy-loading: Evidence entries fetch KnowledgeChunk preview on demand.

3.5 Branch Action Flow (Ghost Node & Job Monitor)
- Purpose: Represent in-progress branching work until the backend returns a child node.
- Steps: Add GhostNode (status=pending) locally → POST /simulate/branch → poll GET /jobs/{job_id} → on success replace GhostNode.

3.6 Session Manager (Left Sidebar)
- Purpose: Manage sessions: list, rename, export, delete, snapshot.
- Controls: Create, Import, Export, Rename, Fork.

3.7 Settings & System (Modal)
- Purpose: Controls for model settings, polling interval, telemetry opt-in, theme toggles.

3.8 Job Monitor / Developer Tools (Overlay)
- Purpose: Inspect active jobs, job logs, blocker reasons, and rerun diagnostics.

3.9 Notifications / Toasts
- Purpose: Short-lived messages for success, warnings, and errors with action buttons (Retry, View Job).

4. Component Inventory — exhaustive
Below each component is a developer-ready specification: props, events, visual states, accessibility, mock JSON.

4.1 `PromptModal` (Landing Overlay)
- Purpose: Collect the scenario and options.
- Props:
  - `open: boolean`
  - `defaultPrompt?: string`
  - `onSubmit: (payload: PromptStartPayload) => Promise<void>`
  - `onClose: () => void`
- Payload shape (TypeScript):
```ts
interface PromptStartPayload {
  prompt: string;
  mode: 'Quick' | 'Analytical';
  persona: string;
  simulate_steps: number;
  temperature?: number;
  seed?: number | null;
}
```
- Events: `onSubmit`, `onClose`
- Visual states: default, submitting (CTA disabled + spinner), error (inline message)
- Accessibility: `role="dialog"`, focus trap, `aria-labelledby` pointing to heading, ESC closes.

4.2 `TerminalLog` (World Building)
- Props:
  - `entries: TerminalEntry[]` where `TerminalEntry = { time:string; level:'info'|'warn'|'error'|'debug'; message:string; meta?:object }`
  - `filterLevel?: string`
  - `onCopy(entryId)`
- Behavior: typewriter animation per line (configurable), pause/resume, export logs
- Accessibility: each log is a `role="log"` item with `aria-live="polite"` for new lines

4.3 `Canvas` (React Flow wrapper)
- Props:
  - `nodes: Node[]`
  - `edges: Edge[]`
  - `onNodeSelect: (id:string) => void`
  - `onNodeBranch: (nodeId, actionId) => void`
  - `viewState: { zoom:number, x:number, y:number }`
- Config:
  - `nodesDraggable={false}`
  - `nodesConnectable={false}`
  - `panOnDrag={true}`
  - Auto-layout via `getLayoutedElements(nodes, edges)` calling Dagre
- Performance:
  - Virtualize node renders if node count > 250 (render placeholders until in-viewport)
  - Memoize node components and use `React.memo`

4.4 `DecisionNode` (Custom Node)
- Props: `DecisionNodeProps`
```ts
interface DecisionNodeProps {
  node: DecisionNode;
  onBranch: (nodeId:string, actionId:string) => void;
  onFocus: (nodeId:string) => void;
}
```
- `DecisionNode` structure (JSON example):
```json
{
  "id":"node_0",
  "title":"Market Entry: Dental SaaS",
  "summary":"Early traction possible via clinics with existing EMR integrations.",
  "description":"Full LLM generated narrative...",
  "time_step":0,
  "alternatives":[{"id":"alt_1","label":"Invest","action_type":"Invest"}],
  "risks":[{"id":"r1","severity":"High","description":"Regulatory delays","mitigation":"Hire compliance"}],
  "source_citations":["cache_123"],
  "confidence_score":0.78,
  "speculative":false
}
```
- Visual pieces:
  - Top status bar (4px) color by highest risk severity.
  - Title (1 line truncated), confidence badge to the right.
  - Summary (3-line clamp), risk icons row (up to 5), source chips.
  - Footer: branch handle (circular + button), more menu.
- States:
  - `default`, `hover`, `focused`, `pending` (ghost), `error`.
- Accessibility: the node is a button with `aria-label="Node: {title} Confidence {score}"` and keyboard-focusable. Branch handle is reachable via keyboard (Tab target).

4.5 `GhostNode` (Pending child)
- Visual: subdued, 0.5 opacity, spinner center, label `Pending...`
- Props: `tempId`, `createdAt`, `onCancel`
- Behavior: Clicking cancel calls `DELETE /jobs/{job_id}` (if supported) or marks GhostNode errored.

4.6 `BranchMenu` (Radial/Dropdown)
- Purpose: present alternatives from node JSON.
- Props: `alternatives: Alternative[]`, `onSelect(alternativeId)`
- Accessibility: menu should be `role="menu"` with `role="menuitem"` children and keyboard navigation.

4.7 `FocusPanel` (Right Sidebar)
- Props: `nodeId`, `isOpen`, `onClose`
- Tabs and content:
  - Narrative: full description + metadata (LLM model, prompt_tokens, vector_distance) and a `Copy Summary` action.
  - Evidence: list of citation chips — on click fetch `GET /chunks/{cache_id}` and show snippet with `Open Source` link.
  - Risks: list of risk cards with severity badge, mitigation text, and quick action `Create Task` (client-local or POST to /tasks).
  - JSON: raw, syntax-highlighted, copyable JSON viewer with schema validation errors highlighted.
- Lazy behavior: Evidence previews are fetched only when the Evidence tab is opened or when chip is hovered (prefetch).

4.8 `Toolbar` (Top or Floating)
- Controls: Zoom In, Zoom Out, Fit View, Toggle Grid, Export PNG, Export JSON, Undo/Redo
- Props: `onExport`, `onZoom`, `onToggleGrid`

4.9 `SessionList` (Left Sidebar)
- Props: `sessions[]`, `onSelect(sessionId)`, `onCreate`, `onDelete`
- Each session displays `name`, `lastModified`, `rootNodeTitle`, `nodeCount`.

4.10 `JobMonitor` and `JobCard`
- JobCard shows job_id, type (`branch`/`start`), status, progress, start_time, duration, actions (view logs, cancel, retry).

4.11 `Toast` / `Notification`
- Four severities: success, info, warn, error. Include optional `actionLabel` and callback.

4.12 Utility UI pieces
- `ConfirmModal`, `JSONEditorModal`, `CopyableField`, `Badge`, `IconButton`, `Spinner`, `EmptyState`, `ErrorState`.

5. Data Contracts & API flows (complete shapes)
These are the canonical request/response formats used by the frontend.

5.1 Start Simulation
POST /simulate/start
Request:
```json
{
  "prompt":"What if I enter the dental SaaS market?",
  "mode":"Analytical",
  "persona":"Skeptical Analyst",
  "simulate_steps":3,
  "temperature":0.2
}
```
Response (200):
```json
{
  "session_id":"sess_123",
  "job_id":"job_abc",
  "status":"started",
  "root_node_id":"node_0"
}
```

5.2 Branch
POST /simulate/branch
Request:
```json
{
  "session_id":"sess_123",
  "parent_node_id":"node_0",
  "action":"Invest",
  "persona":"Optimistic Founder"
}
```
Response (202):
```json
{ "job_id": "job_456", "status":"queued" }
```

5.3 Poll Job
GET /jobs/{job_id}
Response (completed):
```json
{
  "job_id":"job_456",
  "status":"completed",
  "result": { /* DecisionNode JSON (child node) */ }
}
```

5.4 Get Graph
GET /graph/{session_id}
Response:
```json
{ "session_id":"sess_123", "nodes":[...], "edges":[...] }
```

5.5 Get Chunk (evidence)
GET /chunks/{cache_id}
Response:
```json
{ "id":"cache_123","content":"Excerpt...","source_url":"https://...","chunk_index":0 }
```

5.6 Test endpoints (dev only)
- `POST /test/generate` and `POST /test/scrape` return predictable outputs for local dev; frontend should handle both cases gracefully.

6. Interaction Patterns & UX rules (detailed)

6.1 Optimistic UI and GhostNodes
- Immediately reflect action in UI by creating a GhostNode with `temp_id` and `status:pending`.
- Map `temp_id` to `job_id` in client store and poll `GET /jobs/{job_id}`.
- On success, replace ghost with real node (match by temp_id -> result.node.id).
- On failure, mark ghost as `error` and surface action: Retry / View Job.

6.2 Polling / Backoff
- Default polling interval: 1500ms. After 10 unsuccessful polls, backoff to 3000ms. After 30 attempts mark job as flaky and surface a diagnostic modal with logs.

6.3 Error handling
- Consistent error UI: inline for component-level errors, toast for global, full-screen error for unrecoverable.
- All network calls return `{ error?: string, code?: number }` on failure. If `code===429` show rate-limit message.

6.4 Retry semantics
- Branch retries limited to 3 attempts automatically; user may manually retry more.

7. Accessibility (every component)

7.1 General rules
- All interactive elements must be keyboard accessible, have visible focus styles, and exposed ARIA roles.
- Color is not the only cue: badges, icons, and text must accompany color-coded severity.

7.2 Component ARIA mapping (examples)
- `PromptModal`: `role=dialog`, `aria-modal=true`, `aria-labelledby=prompt-title`.
- `DecisionNode`: `role=button` with `aria-pressed=false|true` when toggled; `aria-describedby` points to summary element.
- `BranchMenu`: `role=menu`, options `role=menuitem` and keyboard arrow navigation.

7.3 Screen reader considerations
- TerminalLog: `role=log` with `aria-live="polite"` and `aria-relevant="additions"`.

8. Internationalization & content
- All visible strings must be keys in `i18n` (e.g., `prompt.modal.title`).
- Dates and numbers localized per user locale.

9. Testing Matrix (unit, integration, e2e)

9.1 Unit tests
- `DecisionNode` renders correctly given variant JSON (speculative, high risk, multiple alternatives).
- `BranchMenu` keyboard navigation and `onSelect` callback.
- `PromptModal` validation logic.

9.2 Integration tests
- Canvas renders nodes and edges populated from `GET /graph/{session_id}`.
- Branch flow: create GhostNode, poll job, replace with real node (can use mocked job responses).

9.3 E2E (Playwright)
- Scenario: Initialize simulation -> wait for world build -> branch from root -> open FocusPanel -> view Evidence chip -> open source URL.

10. Performance & Observability

10.1 Lazy loading
- Lazy-load FocusPanel heavy content (evidence previews, JSON viewer) when first opened.

10.2 Memoization
- Use `React.memo` for `DecisionNode` and avoid re-rendering nodes whose props haven't changed.

10.3 Telemetry metrics (client-side)
- `branch_latency_ms`, `job_poll_attempts`, `job_failure_count`, `session_duration_ms`.
- Emit minimal telemetry to `/telemetry` as batched POSTs or console-only if telemetry disabled.

10.4 Debugging UI
- JobMonitor includes per-job logs, structured trace, and retry actions.

11. Animations & Micro-interactions (explicit)
- Node entrance: scale from 0.95 to 1.0, opacity 0 → 1, duration 220ms, easing `cubic-bezier(.2,.9,.3,1)`.
- GhostNode pulse: subtle up/down scale 1 → 1.02 loop.
- FocusPanel slide: x: 100% → 0, fade in, duration 240ms.
- Branch success: child node pops with small confetti micro-animation for positive-high confidence nodes.

12. Keyboard Shortcuts (global)
- `Ctrl/Cmd+K`: quick search / jump
- `Space` (when node focused): open FocusPanel
- `+` / `-`: zoom in/out
- `F`: fit view
- `Esc`: close modals / deselect

13. Security & privacy notes
- Do not leak PII in logs; redact in TerminalLog previews. Evidence URLs are shown but snippet content should mask personal data.

14. Mock Data Examples (per component)
- `DecisionNode` example (short) — same as in component section.
- `KnowledgeChunk` example:
```json
{
  "id":"cache_123",
  "content":"Paragraph excerpt...",
  "source_url":"https://example.com/article",
  "chunk_index":0
}
```

15. Developer notes (concise)
- Keep component CSS isolated via CSS modules or Tailwind with component prefixes.
- Use a single `api.js` or `api.ts` client with centralized retry logic and exponential backoff.

16. Appendix — JSON schemas (compact)
- DecisionNode (core fields): `id`, `title`, `summary`, `description`, `alternatives[]`, `risks[]`, `source_citations[]`, `confidence_score`, `speculative`, `meta{prompt_tokens,vector_distance}`.

---

End of specification. This file is authoritative for the UI design and developer implementation; it intentionally omits handoff lists and generative delivery options.
🎨 MASTER FRONTEND SPECIFICATION: Decision Graph Simulator (UI/UX)
1. Design Philosophy & Aesthetic
Theme: "Strategic Command Center" / "Bloomberg Terminal meets Sci-Fi Strategy Game." Core Vibe: High-density data, dark mode only, trust-centric, distraction-free.

1.1 Color Palette (Semantic Risk System)
We do not use random colors. Every color has a logic mapping to the Backend Data Models.

Canvas Background: #0B1120 (Deepest Navy/Black) — Reduces eye strain for long sessions.

Surface / Panels: #1E293B (Slate 800) with backdrop-filter: blur(12px) and opacity: 0.95.

Borders: #334155 (Slate 700) — Subtle separation.

Typography:

Headings: Inter (Bold, Tracking Tight). Color: #F8FAFC (Slate 50).

Data/Citations: JetBrains Mono (Monospace). Color: #94A3B8 (Slate 400).

1.2 The "Status" Colors (Critical for Decision Nodes)
Safe / Stable: #10B981 (Emerald 500) — Used for nodes with Low Risk.

Caution: #F59E0B (Amber 500) — Used for Medium Risk.

Critical Danger: #EF4444 (Red 500) — Used for High/Critical Risk (Mandatory Backend Rule).

Speculative / Unverified: #A855F7 (Purple 500) — Used when confidence_score < 0.5.

System / AI Thinking: #3B82F6 (Blue 500) — Pulsing animations.

2. The User Flow (Step-by-Step Experience)
Phase 1: The "Mission Start" (Landing Overlay)
State: The Graph Canvas is visible but blurred (blur-sm) in the background.

Component: A centered, glass-morphism modal.

Input 1: The Scenario Prompt

UI: Large, transparent textarea. Text size: 24px.

Placeholder: "Input a strategic scenario (e.g., 'What if I launch a vertical SaaS in the dental market?')."

Input 2: The "Control Panel" (Toggles)

Mode: [ Segmented Control ] -> Quick (Fast) vs Analytical (Deep RAG).

Persona: [ Dropdown ] -> "Skeptical Analyst", "Optimistic Founder", "Legal Eagle".

Action: Big CTA Button "INITIALIZE SIMULATION".

Behavior: On click, the Modal dissolves. The Canvas unblurs. A "Terminal Loader" appears.

Phase 2: The "World Building" (Loading State)
Context: The backend is running ContextBuilder (Search 15 -> Filter -> Scrape). This takes 5-10 seconds. We must visualize this to prevent boredom.

UI: A "Terminal Log" window in the bottom-left corner.

Animation: Lines of text appear one by one (simulating the backend logs):

> Connecting to Global Context...

> Scraping 7 sources... [##########] 100%

> Distilling KnowledgeChunks...

> ReasoningEngine: Active

Transition: The first Root Node pops onto the center of the screen with a satisfying "Click" sound effect.

Phase 3: The "Infinite Canvas" (Main Interaction)
Navigation: Pan (Right Click + Drag), Zoom (Scroll Wheel).

Layout: Tree Graph (Top-to-Bottom).

Node Appearance: See Section 3.

3. Component Deep Dive: "The Decision Node"
This is the heart of the app. It must visualize the complex JSON data from DecisionNode.

Dimensions: Fixed Width: 320px. Height: Auto. Shape: Rounded Rectangle (rounded-xl).

Layer 1: The Header (Status Bar)
Top Border: A 4px colored line indicating Risk Level (Green/Red/Purple).

Left: Title (Truncated to 1 line, e.g., "Market Entry: Tokyo").

Right: Confidence Badge.

Logic: If confidence_score > 0.8 -> Green Dot. If speculative=true -> Purple Dot + "⚠ Speculative" Text.

Layer 2: The Content (Body)
Summary: 3 lines of text max. (The backend sends a summary field).

Risk Indicators: Small icons at the bottom of the card.

Example: If the node has 3 risks, show 3 small shield icons. Red shields for High Risk.

Layer 3: The Action Handle (The "Branch" Button)
Location: Hanging off the bottom center of the card.

UI: A small Circle Button (+).

Interaction:

Hover: It glows.

Click: Opens the Branching Menu (Radial Menu or Dropdown).

Menu Content: Lists the alternatives from the JSON (e.g., "Pivot", "Abort", "Invest").

Action: Clicking an option triggers POST /simulate/branch.

4. Component Deep Dive: "The Focus Panel" (Right Sidebar)
When a user clicks on a Node, this panel slides in from the right (width: 450px). It shows the Deep Data.

Tab 1: Narrative & Evidence
Full Description: The complete text from the LLM.

Citation Rendering (The "Trust" Feature):

Logic: Regex scan for [Source: cache:<id>].

UI: Replace the text with a clickable Blue Chip.

Hover Behavior: A tooltip appears showing the Original URL and the Snippet Text (retrieved from KnowledgeChunk). This proves the AI isn't lying.

Tab 2: Risk Matrix (The "Safety" Layer)
Layout: A list of cards.

Critical Risks: Highlighted with a red background/border.

Content:

Headline: The Risk Description.

Severity: Badge (Critical/High/Medium).

Mitigation: (Italicized) "Suggested mitigation: [text from JSON]".

Tab 3: JSON Inspector (Debug Mode)
A raw view of the DecisionNode JSON for developers to check vector_distance and prompt_tokens.

5. Technical Specifications (Tech Stack)
Core Libraries
React Flow (@xyflow/react):

Usage: The canvas engine.

Configuration: nodesDraggable={false}, nodesConnectable={false} (The graph is auto-generated, not drawn by hand).

Dagre (dagre):

Usage: Auto-layout algorithm. We need a function getLayoutedElements that runs every time a new node is added to keep the tree organized (Top-Down).

Framer Motion (framer-motion):

Usage: All animations.

Key Animation: AnimatePresence for the "Ghost Node" (when waiting for the backend).

Zustand (zustand):

Usage: Global State Management.

Store: useSessionStore -> holds sessionId, nodes, edges, isSimulating.

Tailwind CSS:

Usage: Styling. Use clsx or tailwind-merge for dynamic classes based on risk levels.

API Interaction Logic (The "Async" Flow)
Optimistic UI:

When the user clicks "Branch", IMMEDIATELY add a grey "Ghost Node" to the canvas with a spinner.

Why: The backend Scraper might take 10 seconds. The Ghost Node tells the user "I am working on it."

Polling Strategy:

Do not just wait for one request.

Use a Polling Hook (useInterval) that checks GET /jobs/{job_id} every 2 seconds.

Once the job status is completed, replace the Ghost Node with the real data.

6. Implementation Checklist for the AI Coder
[ ] Setup: npx create-react-app (or Vite) + Install React Flow, Tailwind, Framer Motion.

[ ] State: Create useSessionStore with Zustand.

[ ] Canvas: Build the empty React Flow wrapper with Dark Mode styles.

[ ] Node: Create CustomNode.jsx with the Conditional Border Logic (Red for Risk).

[ ] Sidebar: Create FocusPanel.jsx with the Tab System (Narrative/Risks).

[ ] API: Create api.js using axios to hit http://localhost:8000.

[ ] Polish: Add the "Terminal Loading" animation for the initial world build.

This specification covers every single pixel and behavior required for a professional, high-end Simulator UI.