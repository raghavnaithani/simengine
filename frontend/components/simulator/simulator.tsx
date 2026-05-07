'use client'

import { useCallback, useEffect, useState } from 'react'
import { useReactFlow, ReactFlowProvider } from '@xyflow/react'
import { AnimatePresence } from 'framer-motion'
import { PromptModal } from './prompt-modal'
import { Questionnaire } from './questionnaire'
import { WorldBuilding } from './world-building'
import { GraphCanvas } from './graph-canvas'
import { FocusPanel } from './focus-panel'
import { SessionSidebar } from './session-sidebar'
import { Toolbar } from './toolbar'
import { BranchDialog } from './branch-dialog'
import { Ws5DashboardPanel } from './ws5-dashboard'
import { ToastProvider, useToast } from './toast-provider'
import { useSessionStore } from '@/lib/store'
import { api, loadGraphAfterJobCompletion } from '@/lib/api'
import { useTelemetry } from '@/hooks/use-telemetry'
import { useJobPollingWithDynamicBackoff } from '@/hooks/use-job-polling'
import type { StartSimulationRequest, DecisionNode, ScenarioConfig, Alternative, Session } from '@/lib/types'

const BRANCH_PERSONA_LABELS: Record<string, string> = {
  'skeptical-analyst': 'Skeptical Analyst',
  'optimistic-founder': 'Optimistic Founder',
  'cautious-regulator': 'Cautious Regulator',
  'aggressive-founder': 'Aggressive Founder',
  'pessimistic-analyst': 'Pessimistic Analyst',
}

function getAlternativeLabel(alternative: Alternative): string {
  return alternative.label || alternative.action_type || alternative.description || 'Option'
}

function normalizeBranchPersona(persona: string): string {
  return BRANCH_PERSONA_LABELS[persona] || persona || 'Optimistic Founder'
}

function SimulatorInner() {
  const {
    phase,
    setPhase,
    sessionId,
    setSessionId,
    sessions,
    setSessions,
    selectedNodeId,
    setSelectedNodeId,
    focusPanelOpen,
    setFocusPanelOpen,
    sessionSidebarOpen,
    setSessionSidebarOpen,
    nodes,
    addNode,
    setNodes,
    setEdges,
    addGhostNode,
    removeGhostNode,
    resetSession,
  } = useSessionStore()

  const { addToast } = useToast()
  const { trackBranchCompletion, trackBranchFailure, trackUserAction } = useTelemetry()
  
  const [showGrid, setShowGrid] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [pendingPrompt, setPendingPrompt] = useState<string>('')
  const [scenarioConfig, setScenarioConfig] = useState<ScenarioConfig | null>(null)
  const [ws5DashboardOpen, setWs5DashboardOpen] = useState(false)
  
  // Branch dialog state
  const [branchDialogOpen, setBranchDialogOpen] = useState(false)
  const [branchParentNode, setBranchParentNode] = useState<DecisionNode | null>(null)
  const [activeBranchJobId, setActiveBranchJobId] = useState<string | null>(null)
  const [branchStartTime, setBranchStartTime] = useState<number>(0)
  
  // Poll active branch job
  const branchPoll = useJobPollingWithDynamicBackoff(activeBranchJobId, !!activeBranchJobId)
  
  const reactFlowInstance = useReactFlow()

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      switch (e.key) {
        case 'Escape':
          setFocusPanelOpen(false)
          setSelectedNodeId(null)
          setBranchDialogOpen(false)
          break
        case ' ':
          if (selectedNodeId) {
            e.preventDefault()
            setFocusPanelOpen(true)
          }
          break
        case '+':
        case '=':
          handleZoomIn()
          break
        case '-':
          handleZoomOut()
          break
        case 'f':
        case 'F':
          if (!e.metaKey && !e.ctrlKey) {
            handleFitView()
          }
          break
        case 'k':
          if (e.metaKey || e.ctrlKey) {
            e.preventDefault()
            // Quick search - could open a command palette
          }
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedNodeId, setFocusPanelOpen, setSelectedNodeId])

  // Zoom handlers
  const handleZoomIn = useCallback(() => {
    reactFlowInstance?.zoomIn()
    setZoom(prev => Math.min(prev * 1.2, 2))
  }, [reactFlowInstance])

  const handleZoomOut = useCallback(() => {
    reactFlowInstance?.zoomOut()
    setZoom(prev => Math.max(prev / 1.2, 0.1))
  }, [reactFlowInstance])

  const handleFitView = useCallback(() => {
    reactFlowInstance?.fitView({ padding: 0.2 })
    setZoom(1)
  }, [reactFlowInstance])

  // Handle initial prompt submission - go to questionnaire
  const handlePromptSubmit = async (payload: StartSimulationRequest) => {
    setPendingPrompt(payload.prompt)
    setPhase('questionnaire')
  }

  // Handle questionnaire completion - go to world building with full config
  const handleQuestionnaireComplete = (config: ScenarioConfig) => {
    setScenarioConfig(config)
    setPhase('world-building')
  }

  // Handle going back from questionnaire to prompt
  const handleQuestionnaireBack = () => {
    setPhase('landing')
  }

  // Handle world building completion with full tree
  const handleWorldBuildingComplete = (
    rootNode: DecisionNode,
    backendSessionId: string,
    allNodes?: DecisionNode[],
    allEdges?: { id: string; source: string; target: string; action?: string }[]
  ) => {
    // If we have a full tree, set all nodes and edges
    if (allNodes && allEdges) {
      setNodes(allNodes)
      setEdges(allEdges)
    } else {
      // Just add the root node
      addNode(rootNode)
    }
    
    setSessionId(backendSessionId)

    const now = new Date().toISOString()
    const nodeCount = allNodes?.length || 1
    const existingSession = sessions.find(s => s.id === backendSessionId)
    const updatedSession: Session = {
      id: backendSessionId,
      name: existingSession?.name || rootNode.title || 'Simulation Session',
      created_at: existingSession?.created_at || now,
      last_modified: now,
      root_node_id: rootNode.id,
      node_count: nodeCount,
    }
    setSessions([
      updatedSession,
      ...sessions.filter(s => s.id !== backendSessionId),
    ])
    
    // Move to canvas phase
    setPhase('canvas')
    
    // Show success toast
    const edgeCount = allEdges?.length || 0
    addToast({
      type: 'success',
      title: 'Simulation Initialized',
      message: `Generated ${nodeCount} nodes and ${edgeCount} branches`,
    })
  }

  // Handle new session
  const handleNewSession = () => {
    resetSession()
    setPendingPrompt('')
    setScenarioConfig(null)
    setWs5DashboardOpen(false)
    setPhase('landing')
  }

  // Handle cancel world building
  const handleCancelWorldBuilding = () => {
    setPendingPrompt('')
    setScenarioConfig(null)
    setWs5DashboardOpen(false)
    setPhase('landing')
  }

  // Handle node selection
  const handleNodeSelect = (nodeId: string) => {
    setSelectedNodeId(nodeId)
    setFocusPanelOpen(true)
  }

  // Handle branching from a node - opens branch dialog
  const handleNodeBranch = (nodeId: string, _alternative: Alternative) => {
    const node = nodes.find(n => n.id === nodeId)
    if (node) {
      setBranchParentNode(node)
      setBranchDialogOpen(true)
    }
  }

  // Handle branch creation from dialog
  const handleBranchCreate = async (alternative: Alternative, persona: string, customPrompt?: string) => {
    if (!branchParentNode || !sessionId) {
      addToast({
        type: 'error',
        title: 'Branch Error',
        message: 'Missing session or parent node',
      })
      return
    }

    const tempId = `ghost_${Date.now()}`
    const parentId = branchParentNode.id

    // Track user action
    await trackUserAction('branch.initiated', parentId, {
      alternative: alternative.action_type,
    })

    // Add ghost node immediately for optimistic UI
    addGhostNode({
      temp_id: tempId,
      parent_id: parentId,
      status: 'pending',
      created_at: new Date().toISOString(),
      action: customPrompt || getAlternativeLabel(alternative),
    })

    try {
      // Call POST /simulate/branch
      setBranchStartTime(Date.now())

      const response = await api.branch({
        session_id: sessionId,
        parent_node_id: parentId,
        action: customPrompt || alternative.action_type,
        persona: normalizeBranchPersona(persona),
        seed: null,
      })

      // Start polling the job
      setActiveBranchJobId(response.job_id)

      addToast({
        type: 'info',
        title: 'Branching...',
        message: `Creating branch from "${getAlternativeLabel(alternative)}"`,
      })
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error'
      
      // Update ghost to error state
      useSessionStore.getState().updateGhostNode(tempId, {
        status: 'error',
        error: errorMsg,
      })

      await trackBranchFailure(activeBranchJobId || '', errorMsg, 0, sessionId)

      addToast({
        type: 'error',
        title: 'Branch Failed',
        message: errorMsg,
        action: {
          label: 'Retry',
          action: 'retry',
        },
      })
    }
  }

  // Monitor branch job completion
  useEffect(() => {
    if (!branchPoll.isDone || !sessionId) return

    let isMounted = true

    const completeBranch = async () => {
      try {
        const elapsed = Date.now() - branchStartTime
        const ghostNode = useSessionStore.getState().ghostNodes[0]

        if (branchPoll.status === 'completed' && branchPoll.job?.result?.node_id) {
          // Load updated graph
          const { nodes: updatedNodes, edges: updatedEdges } = await loadGraphAfterJobCompletion(
            sessionId,
            branchPoll.job.result.node_id
          )

          if (!isMounted) return

          if (ghostNode) {
            removeGhostNode(ghostNode.temp_id)
          }

          // Replace graph state with backend truth to avoid duplicate nodes/edges.
          setNodes(updatedNodes)
          setEdges(updatedEdges)

          const newNode = updatedNodes.find(n => n.id === branchPoll.job?.result?.node_id)
          const now = new Date().toISOString()
          setSessions(
            useSessionStore.getState().sessions.map(s =>
              s.id === sessionId
                ? { ...s, last_modified: now, node_count: updatedNodes.length }
                : s
            )
          )

          addToast({
            type: 'success',
            title: 'Branch Created',
            message: newNode?.title || 'Branch generated successfully',
          })

          await trackBranchCompletion(
            branchPoll.job.job_id,
            elapsed,
            branchPoll.pollAttempts,
            sessionId
          )
        } else if (branchPoll.status === 'failed') {
          const errorMsg = branchPoll.error || 'Branch job failed'
          
          if (ghostNode) {
            useSessionStore.getState().updateGhostNode(ghostNode.temp_id, {
              status: 'error',
              error: errorMsg,
            })
          }

          await trackBranchFailure(
            branchPoll.job?.job_id || '',
            errorMsg,
            branchPoll.pollAttempts,
            sessionId
          )

          addToast({
            type: 'error',
            title: 'Branch Failed',
            message: errorMsg,
            action: {
              label: 'Retry',
              action: 'retry',
              jobId: branchPoll.job?.job_id,
            },
          })
        }

        setActiveBranchJobId(null)
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error'
        console.error('[Simulator] Branch completion error:', error)

        await trackBranchFailure(
          branchPoll.job?.job_id || '',
          errorMsg,
          branchPoll.pollAttempts,
          sessionId
        )

        addToast({
          type: 'error',
          title: 'Error',
          message: errorMsg,
        })
      }
    }

    completeBranch()

    return () => {
      isMounted = false
    }
  }, [branchPoll.isDone, branchPoll.status, branchPoll.job, branchPoll.error, branchPoll.pollAttempts, sessionId, branchStartTime, addToast, trackBranchCompletion, trackBranchFailure, removeGhostNode, setNodes, setEdges, setSessions])

  // Export handlers
  const handleExportPng = async () => {
    addToast({
      type: 'info',
      title: 'Export PNG Disabled',
      message: 'PNG export is not implemented yet. Use JSON export for now.',
    })
  }

  const handleExportJson = () => {
    const data = {
      session_id: sessionId,
      nodes: nodes,
      edges: useSessionStore.getState().edges,
      config: scenarioConfig,
      exported_at: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `dgs_session_${sessionId || 'export'}.json`
    a.click()
    URL.revokeObjectURL(url)
    
    addToast({
      type: 'success',
      title: 'Exported',
      message: 'Session exported as JSON.',
    })
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-background">
      {/* Landing Overlay / Prompt Modal */}
      <PromptModal
        open={phase === 'landing'}
        onSubmit={handlePromptSubmit}
        onClose={() => {}}
      />

      {/* Questionnaire Phase */}
      <AnimatePresence>
        {phase === 'questionnaire' && pendingPrompt && (
          <Questionnaire
            prompt={pendingPrompt}
            onComplete={handleQuestionnaireComplete}
            onBack={handleQuestionnaireBack}
          />
        )}
      </AnimatePresence>

      {/* World Building Phase */}
      <AnimatePresence>
        {phase === 'world-building' && scenarioConfig && (
          <WorldBuilding
            prompt={scenarioConfig.prompt}
            mode={scenarioConfig.mode}
            scenarioConfig={scenarioConfig}
            onComplete={handleWorldBuildingComplete}
            onCancel={handleCancelWorldBuilding}
          />
        )}
      </AnimatePresence>

      {/* Main Canvas Phase */}
      <AnimatePresence>
        {phase === 'canvas' && (
          <>
            {/* Toolbar */}
            <Toolbar
              onZoomIn={handleZoomIn}
              onZoomOut={handleZoomOut}
              onFitView={handleFitView}
              onToggleGrid={() => setShowGrid(!showGrid)}
              onToggleDashboard={() => setWs5DashboardOpen(!ws5DashboardOpen)}
              onExportPng={handleExportPng}
              canExportPng={false}
              onExportJson={handleExportJson}
              onToggleSessionSidebar={() => setSessionSidebarOpen(!sessionSidebarOpen)}
              onNewSession={handleNewSession}
              showGrid={showGrid}
              dashboardOpen={ws5DashboardOpen}
              zoom={zoom}
            />

            <Ws5DashboardPanel
              isOpen={ws5DashboardOpen}
              onClose={() => setWs5DashboardOpen(false)}
            />

            {/* Graph Canvas */}
            <GraphCanvas
              onNodeSelect={handleNodeSelect}
              onNodeBranch={handleNodeBranch}
              showGrid={showGrid}
              showMiniMap={true}
            />

            {/* Focus Panel */}
            <FocusPanel
              nodeId={selectedNodeId}
              isOpen={focusPanelOpen}
              onClose={() => {
                setFocusPanelOpen(false)
                setSelectedNodeId(null)
              }}
            />

            {/* Session Sidebar */}
            <SessionSidebar
              isOpen={sessionSidebarOpen}
              onClose={() => setSessionSidebarOpen(false)}
              onNewSession={handleNewSession}
            />

            {/* Branch Dialog */}
            <BranchDialog
              open={branchDialogOpen}
              onClose={() => {
                setBranchDialogOpen(false)
                setBranchParentNode(null)
              }}
              onBranch={handleBranchCreate}
              parentNode={branchParentNode}
            />
          </>
        )}
      </AnimatePresence>

      {/* Background grid pattern (visible during landing) */}
      {phase === 'landing' && (
        <div className="absolute inset-0 canvas-grid opacity-30" />
      )}
    </div>
  )
}

// Wrap with ReactFlowProvider and ToastProvider
export function Simulator() {
  return (
    <ToastProvider>
      <ReactFlowProvider>
        <SimulatorInner />
      </ReactFlowProvider>
    </ToastProvider>
  )
}
