import { useState, useEffect, useCallback, useRef } from 'react'
import { api, ApiError } from '@/lib/api'
import type { Job } from '@/lib/types'

export interface JobPollState {
  job: Job | null
  status: 'idle' | 'polling' | 'completed' | 'failed'
  error: string | null
  pollAttempts: number
  isFlaky: boolean
}

/**
 * Canonical hook for polling a job with dynamic backoff strategy.
 * - Initial cadence: 1500ms
 * - After 10 polls: 3000ms
 * - After 30 polls: marks as flaky
 */
export function useJobPollingWithDynamicBackoff(jobId: string | null, enabled = true) {
  const [state, setState] = useState<JobPollState>({
    job: null,
    status: 'idle',
    error: null,
    pollAttempts: 0,
    isFlaky: false,
  })

  const timeoutRef = useRef<NodeJS.Timeout | null>(null)
  const pollCountRef = useRef(0)
  const startTimeRef = useRef<number>(0)

  const schedulePoll = useCallback(async () => {
    if (!jobId) return

    try {
      const job = await api.getJob(jobId)
      pollCountRef.current++
      const isDone = job.status === 'completed' || job.status === 'failed'

      setState(prev => ({
        ...prev,
        job,
        status: job.status === 'completed' ? 'completed' : 
                job.status === 'failed' ? 'failed' : 'polling',
        error: job.error || null,
        pollAttempts: pollCountRef.current,
        isFlaky: pollCountRef.current > 30,
      }))

      if (!isDone) {
        const delayMs = pollCountRef.current < 10 ? 1500 : 3000
        timeoutRef.current = setTimeout(() => {
          schedulePoll()
        }, delayMs)
      }
    } catch (error) {
      const errorMsg = error instanceof ApiError 
        ? error.message 
        : 'Failed to poll job status'
      
      setState(prev => ({
        ...prev,
        error: errorMsg,
        status: 'failed',
        isFlaky: pollCountRef.current > 30,
        pollAttempts: pollCountRef.current,
      }))

      if (!(error instanceof ApiError && error.code === 404)) {
        const delayMs = pollCountRef.current < 10 ? 1500 : 3000
        timeoutRef.current = setTimeout(() => {
          schedulePoll()
        }, delayMs)
      }
    }
  }, [jobId])

  useEffect(() => {
    if (!enabled || !jobId) return

    setState(prev => ({
      ...prev,
      status: 'polling',
      error: null,
      job: null,
      pollAttempts: 0,
      isFlaky: false,
    }))
    pollCountRef.current = 0
    startTimeRef.current = Date.now()

    schedulePoll()

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [enabled, jobId, schedulePoll])

  return {
    ...state,
    isPolling: state.status === 'polling',
    isDone: state.status === 'completed' || state.status === 'failed',
    duration: startTimeRef.current ? Date.now() - startTimeRef.current : 0,
  }
}

/**
 * Backward-compatible alias to the canonical hook.
 */
export function useJobPolling(jobId: string | null, enabled = true) {
  return useJobPollingWithDynamicBackoff(jobId, enabled)
}
