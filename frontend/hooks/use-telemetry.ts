import { useCallback } from 'react'
import { api } from '@/lib/api'

export interface TelemetryDetails {
  [key: string]: unknown
}

/**
 * Hook for tracking frontend events and sending telemetry to backend
 * Sends to POST /log endpoint
 */
export function useTelemetry() {
  const track = useCallback(
    async (
      action: string,
      message: string,
      details?: TelemetryDetails,
      level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG' = 'INFO'
    ) => {
      try {
        await api.postClientLog({
          level,
          action,
          message,
          details,
        })
      } catch (error) {
        // Silently fail - don't break UI if telemetry fails
        console.error('[Telemetry] Failed to send:', error)
      }
    },
    []
  )

  // Track branch completion with metrics
  const trackBranchCompletion = useCallback(
    async (jobId: string, latencyMs: number, pollAttempts: number, sessionId: string) => {
      await track('ui.branch.complete', 'Branch job completed', {
        job_id: jobId,
        latency_ms: latencyMs,
        poll_attempts: pollAttempts,
        session_id: sessionId,
      })
    },
    [track]
  )

  // Track branch failure
  const trackBranchFailure = useCallback(
    async (jobId: string, error: string, pollAttempts: number, sessionId: string) => {
      await track(
        'ui.branch.failed',
        'Branch job failed',
        {
          job_id: jobId,
          error,
          poll_attempts: pollAttempts,
          session_id: sessionId,
        },
        'ERROR'
      )
    },
    [track]
  )

  // Track simulation start
  const trackSimulationStart = useCallback(
    async (sessionId: string, mode: string, prompt: string) => {
      await track('ui.simulation.start', 'Simulation started', {
        session_id: sessionId,
        mode,
        prompt_length: prompt.length,
      })
    },
    [track]
  )

  // Track simulation completion
  const trackSimulationCompletion = useCallback(
    async (sessionId: string, nodeCount: number, duration: number) => {
      await track('ui.simulation.complete', 'Simulation tree generated', {
        session_id: sessionId,
        node_count: nodeCount,
        duration_ms: duration,
      })
    },
    [track]
  )

  // Track user action (node clicked, branch selected, etc.)
  const trackUserAction = useCallback(
    async (actionType: string, target: string, context?: TelemetryDetails) => {
      await track('ui.user.action', `User performed: ${actionType}`, {
        action_type: actionType,
        target,
        ...context,
      })
    },
    [track]
  )

  // Track error
  const trackError = useCallback(
    async (errorType: string, message: string, context?: TelemetryDetails) => {
      await track('ui.error', message, {
        error_type: errorType,
        ...context,
      }, 'ERROR')
    },
    [track]
  )

  return {
    track,
    trackBranchCompletion,
    trackBranchFailure,
    trackSimulationStart,
    trackSimulationCompletion,
    trackUserAction,
    trackError,
  }
}
