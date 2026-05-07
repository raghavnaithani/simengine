'use client'

import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { TerminalLog, useTerminalLogger } from './terminal-log'
import { api, loadGraphAfterJobCompletion } from '@/lib/api'
import { useJobPollingWithDynamicBackoff } from '@/hooks/use-job-polling'
import { useTelemetry } from '@/hooks/use-telemetry'
import { useToast } from './toast-provider'
import type { DecisionNode, ScenarioConfig, GraphEdge, StartSimulationResponse } from '@/lib/types'

interface WorldBuildingProps {
  onComplete: (rootNode: DecisionNode, backendSessionId: string, allNodes?: DecisionNode[], allEdges?: GraphEdge[]) => void
  onCancel?: () => void
  prompt: string
  mode: string
  scenarioConfig?: ScenarioConfig
}

export function WorldBuilding({ onComplete, onCancel, prompt, mode, scenarioConfig }: WorldBuildingProps) {
  const { entries, addEntry, clear } = useTerminalLogger()
  const { trackSimulationStart, trackSimulationCompletion, trackError } = useTelemetry()
  const { addToast } = useToast()
  
  const [jobId, setJobId] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [eta, setEta] = useState<number | null>(null)
  const [isInitializing, setIsInitializing] = useState(true)
  const [startTime] = useState(Date.now())
  const lastLoggedPollAttemptRef = useRef<number>(0)
  const loggedFlakyWarningRef = useRef(false)

  // Use job polling hook
  const jobPoll = useJobPollingWithDynamicBackoff(jobId, !!jobId)

  // Start simulation (initialize job)
  useEffect(() => {
    let isMounted = true

    const initializeSimulation = async () => {
      try {
        clear()
        lastLoggedPollAttemptRef.current = 0
        loggedFlakyWarningRef.current = false
        addEntry('Initializing simulation...', 'info')

        if (!scenarioConfig) {
          throw new Error('Scenario config required')
        }

        // Call POST /simulate/start
        addEntry(`Mode: ${scenarioConfig.mode}`, 'info')
        addEntry(`Timeline: ${scenarioConfig.timeline}`, 'info')
        addEntry(`Persona: ${scenarioConfig.persona || 'Default'}`, 'info')

        const response: StartSimulationResponse = await api.startSimulation({
          prompt: scenarioConfig.prompt,
          mode: scenarioConfig.mode as 'Analytical' | 'Quick',
          persona: scenarioConfig.persona,
          simulate_steps: scenarioConfig.simulate_steps,
          seed: null,
        })

        if (!isMounted) return

        setSessionId(response.session_id)
        setJobId(response.job_id)

        addEntry(`Session created: ${response.session_id}`, 'info')
        addEntry(`Job started: ${response.job_id}`, 'info')
        addEntry('Connecting to backend engines...', 'info')
        addEntry('ContextBuilder (Deep RAG): Active', 'info')
        addEntry('ReasoningEngine: Initializing', 'info')

        // Track simulation start
        await trackSimulationStart(response.session_id, scenarioConfig.mode, scenarioConfig.prompt)

        setIsInitializing(false)
      } catch (error) {
        if (!isMounted) return

        const errorMsg = error instanceof Error ? error.message : 'Unknown error'
        console.error('[WorldBuilding] Init error:', error)
        addEntry(`Error: ${errorMsg}`, 'error')
        
        await trackError('simulation.start.failed', errorMsg, {
          prompt_length: prompt.length,
        })

        addToast({
          type: 'error',
          title: 'Simulation Failed',
          message: errorMsg,
        })
      }
    }

    initializeSimulation()

    return () => {
      isMounted = false
    }
  }, [scenarioConfig, clear, addEntry, trackSimulationStart, trackError, addToast, prompt.length])

  // Monitor job polling progress
  useEffect(() => {
    if (!jobPoll.job) return

    // Update progress based on job status
    if (jobPoll.isPolling) {
      // Estimate progress based on poll attempts
      const pollProgress = Math.min(90, 10 + (jobPoll.pollAttempts * 3))
      setProgress(pollProgress)

      // Calculate ETA
      const elapsed = jobPoll.duration
      if (jobPoll.pollAttempts > 0) {
        const avgTimePerPoll = elapsed / jobPoll.pollAttempts
        const estimatedRemaining = (100 - pollProgress) / 3 * avgTimePerPoll
        setEta(Math.max(0, Math.ceil(estimatedRemaining / 1000)))
      }

      // Log status updates every 3 polls
      if (jobPoll.pollAttempts > 0 && jobPoll.pollAttempts % 3 === 0 && lastLoggedPollAttemptRef.current !== jobPoll.pollAttempts) {
        addEntry(`[POLL] Attempt ${jobPoll.pollAttempts} | Status: ${jobPoll.job.status}`, 'info')
        lastLoggedPollAttemptRef.current = jobPoll.pollAttempts
      }

      // Mark as flaky if too many polls
      if (jobPoll.isFlaky && !loggedFlakyWarningRef.current) {
        addEntry(`[WARNING] Job appears to be flaky (${jobPoll.pollAttempts} polls)`, 'warn')
        loggedFlakyWarningRef.current = true
      }
    }
  }, [jobPoll.job, jobPoll.isPolling, jobPoll.pollAttempts, jobPoll.duration, jobPoll.isFlaky, addEntry])

  // Handle job completion
  useEffect(() => {
    let isMounted = true

    if (jobPoll.status !== 'completed' || !sessionId) return

    const completeSimulation = async () => {
      try {
        setProgress(95)
        addEntry('Job completed, loading graph...', 'info')

        // Load full graph using adapter
        const { nodes, edges } = await loadGraphAfterJobCompletion(
          sessionId,
          jobPoll.job?.result?.node_id
        )

        if (!isMounted) return

        setProgress(99)
        addEntry(`Loaded ${nodes.length} nodes and ${edges.length} edges`, 'info')

        // Find root node
        const rootNode = nodes.find(n => n.id === 'node_0') || nodes[0]
        if (!rootNode) {
          throw new Error('No root node found in graph')
        }

        // Finish progress
        setProgress(100)
        setEta(0)
        addEntry('World building complete!', 'info')

        const duration = Date.now() - startTime
        await trackSimulationCompletion(sessionId, nodes.length, duration)

        // Brief delay for animation
        await new Promise(r => setTimeout(r, 300))

        if (!isMounted) return

        onComplete(rootNode, sessionId, nodes, edges)
      } catch (error) {
        if (!isMounted) return

        const errorMsg = error instanceof Error ? error.message : 'Unknown error'
        console.error('[WorldBuilding] Completion error:', error)
        addEntry(`Error loading graph: ${errorMsg}`, 'error')

        await trackError('graph.load.failed', errorMsg)

        addToast({
          type: 'error',
          title: 'Graph Load Failed',
          message: errorMsg,
          action: {
            label: 'Retry',
            action: 'retry',
          },
        })
      }
    }

    completeSimulation()

    return () => {
      isMounted = false
    }
  }, [jobPoll.status, sessionId, jobPoll.job, onComplete, addEntry, trackSimulationCompletion, trackError, addToast, startTime])

  // Handle job failure
  useEffect(() => {
    if (jobPoll.status !== 'failed' || !sessionId) return

    addEntry(`Job failed: ${jobPoll.error || 'Unknown error'}`, 'error')

    addToast({
      type: 'error',
      title: 'Job Failed',
      message: jobPoll.error || 'Unknown error',
      action: {
        label: 'Retry',
        action: 'retry',
        jobId: jobId || undefined,
      },
    })

    trackError('job.failed', jobPoll.error || 'Unknown', {
      job_id: jobId,
      poll_attempts: jobPoll.pollAttempts,
    })
  }, [jobPoll.status, jobPoll.error, sessionId, jobId, jobPoll.pollAttempts, addEntry, addToast, trackError])

  const isLoading = isInitializing || (jobPoll.isPolling && progress < 100)
  const displayProgress = Math.round(progress)

  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center">
      {/* Background with grid pattern */}
      <div className="absolute inset-0 bg-background canvas-grid opacity-50" />

      {/* Center content */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="relative z-10 flex flex-col items-center gap-8 px-4"
      >
        {/* Animated orb */}
        <div className="relative">
          <motion.div
            className="h-32 w-32 rounded-full bg-gradient-to-br from-primary/30 to-primary/10"
            animate={{
              scale: [1, 1.1, 1],
              opacity: [0.5, 0.8, 0.5],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
          <motion.div
            className="absolute inset-4 rounded-full bg-gradient-to-br from-primary/50 to-primary/20"
            animate={{
              scale: [1, 1.15, 1],
              opacity: [0.6, 1, 0.6],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: 'easeInOut',
              delay: 0.3,
            }}
          />
          <motion.div
            className="absolute inset-8 rounded-full bg-primary pulse-glow"
            animate={{
              scale: [1, 1.05, 1],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
        </div>

        {/* Status text */}
        <div className="text-center">
          <h2 className="text-2xl font-bold tracking-tight text-foreground">
            {progress === 100 ? 'Complete!' : 'Building Your World'}
          </h2>
          <p className="mt-2 text-muted-foreground text-sm">
            {isInitializing
              ? 'Initializing engines...'
              : jobPoll.isFlaky
              ? `Long-running operation (${jobPoll.pollAttempts} checks)`
              : scenarioConfig
              ? `Generating ${scenarioConfig.timeline} decision tree`
              : 'Deep RAG pipeline active'}
          </p>
        </div>

        {/* Progress bar */}
        <div className="w-80">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Progress</span>
            <span className="font-mono text-foreground">{displayProgress}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-border">
            <motion.div
              className="h-full bg-primary"
              initial={{ width: '0%' }}
              animate={{ width: `${displayProgress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          {eta !== null && eta > 0 && (
            <p className="mt-2 text-center text-sm text-muted-foreground">
              Estimated time: {eta}s
            </p>
          )}
        </div>

        {/* Terminal logs */}
        {entries.length > 0 && (
          <div className="w-full max-w-2xl max-h-48 overflow-hidden">
            <TerminalLog entries={entries} isOpen={true} className="!relative !bottom-0 !left-0 !w-full" />
          </div>
        )}

        {/* Cancel button */}
        {onCancel && isLoading && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2 }}
            onClick={onCancel}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Cancel
          </motion.button>
        )}
      </motion.div>
    </div>
  )
}
