import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { 
  DecisionNode, 
  GraphEdge, 
  Session, 
  Job, 
  GhostNode, 
  TerminalEntry,
  AppPhase 
} from './types'

interface SessionState {
  // App Phase
  phase: AppPhase
  setPhase: (phase: AppPhase) => void
  
  // Session
  sessionId: string | null
  setSessionId: (id: string | null) => void
  sessions: Session[]
  setSessions: (sessions: Session[]) => void
  
  // Graph Data
  nodes: DecisionNode[]
  edges: GraphEdge[]
  ghostNodes: GhostNode[]
  setNodes: (nodes: DecisionNode[]) => void
  setEdges: (edges: GraphEdge[]) => void
  addNode: (node: DecisionNode) => void
  addEdge: (edge: GraphEdge) => void
  addGhostNode: (ghost: GhostNode) => void
  removeGhostNode: (tempId: string) => void
  updateGhostNode: (tempId: string, updates: Partial<GhostNode>) => void
  
  // Selection
  selectedNodeId: string | null
  setSelectedNodeId: (id: string | null) => void
  
  // Jobs
  activeJobs: Job[]
  addJob: (job: Job) => void
  updateJob: (jobId: string, updates: Partial<Job>) => void
  removeJob: (jobId: string) => void
  
  // Terminal Logs
  terminalEntries: TerminalEntry[]
  addTerminalEntry: (entry: TerminalEntry) => void
  clearTerminalEntries: () => void
  
  // UI State
  focusPanelOpen: boolean
  setFocusPanelOpen: (open: boolean) => void
  sessionSidebarOpen: boolean
  setSessionSidebarOpen: (open: boolean) => void
  
  // Canvas View
  canvasView: { zoom: number; x: number; y: number }
  setCanvasView: (view: { zoom: number; x: number; y: number }) => void
  
  // Reset
  resetSession: () => void
}

const initialState = {
  phase: 'landing' as AppPhase,
  sessionId: null,
  sessions: [],
  nodes: [],
  edges: [],
  ghostNodes: [],
  selectedNodeId: null,
  activeJobs: [],
  terminalEntries: [],
  focusPanelOpen: false,
  sessionSidebarOpen: false,
  canvasView: { zoom: 1, x: 0, y: 0 },
}

function dedupeById<T extends { id: string }>(items: T[]): T[] {
  const map = new Map<string, T>()
  for (const item of items) {
    map.set(item.id, item)
  }
  return Array.from(map.values())
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set) => ({
      ...initialState,
      
      setPhase: (phase) => set({ phase }),
      setSessionId: (sessionId) => set({ sessionId }),
      setSessions: (sessions) => set({ sessions }),
      
      setNodes: (nodes) => set({ nodes: dedupeById(nodes) }),
      setEdges: (edges) => set({ edges: dedupeById(edges) }),
      addNode: (node) => set((state) => {
        const existing = state.nodes.findIndex(n => n.id === node.id)
        if (existing >= 0) {
          const next = [...state.nodes]
          next[existing] = node
          return { nodes: next }
        }
        return { nodes: [...state.nodes, node] }
      }),
      addEdge: (edge) => set((state) => {
        const existing = state.edges.findIndex(e => e.id === edge.id)
        if (existing >= 0) {
          const next = [...state.edges]
          next[existing] = edge
          return { edges: next }
        }
        return { edges: [...state.edges, edge] }
      }),
      
      addGhostNode: (ghost) => set((state) => ({ ghostNodes: [...state.ghostNodes, ghost] })),
      removeGhostNode: (tempId) => set((state) => ({ 
        ghostNodes: state.ghostNodes.filter(g => g.temp_id !== tempId) 
      })),
      updateGhostNode: (tempId, updates) => set((state) => ({
        ghostNodes: state.ghostNodes.map(g => 
          g.temp_id === tempId ? { ...g, ...updates } : g
        )
      })),
      
      setSelectedNodeId: (selectedNodeId) => set({ selectedNodeId }),
      
      addJob: (job) => set((state) => ({ activeJobs: [...state.activeJobs, job] })),
      updateJob: (jobId, updates) => set((state) => ({
        activeJobs: state.activeJobs.map(j => 
          j.job_id === jobId ? { ...j, ...updates } : j
        )
      })),
      removeJob: (jobId) => set((state) => ({
        activeJobs: state.activeJobs.filter(j => j.job_id !== jobId)
      })),
      
      addTerminalEntry: (entry) => set((state) => ({ 
        terminalEntries: [...state.terminalEntries, entry] 
      })),
      clearTerminalEntries: () => set({ terminalEntries: [] }),
      
      setFocusPanelOpen: (focusPanelOpen) => set({ focusPanelOpen }),
      setSessionSidebarOpen: (sessionSidebarOpen) => set({ sessionSidebarOpen }),
      
      setCanvasView: (canvasView) => set({ canvasView }),
      
      resetSession: () => set(initialState),
    }),
    {
      name: 'simengine-session-store-v1',
      partialize: (state) => ({
        phase: state.phase,
        sessionId: state.sessionId,
        sessions: state.sessions,
        nodes: state.nodes,
        edges: state.edges,
        selectedNodeId: state.selectedNodeId,
        canvasView: state.canvasView,
      }),
    }
  )
)
