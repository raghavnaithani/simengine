import type {
  StartSimulationRequest,
  StartSimulationResponse,
  BranchRequest,
  BranchResponse,
  GraphResponse,
  Job,
  JobLogsResponse,
  DecisionNode,
  GraphEdge,
  CuratorReviewListResponse,
  CuratorReviewPayload,
  MetricsDashboardResponse,
} from './types'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export class ApiError extends Error {
  code?: number
  details?: unknown
  
  constructor(message: string, code?: number, details?: unknown) {
    super(message)
    this.code = code
    this.details = details
    this.name = 'ApiError'
  }
}

async function fetchWithRetry<T>(
  url: string,
  options?: RequestInit,
  retries = 3,
  baseBackoff = 1000
): Promise<T> {
  let lastError: Error | null = null

  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }))
        
        if (response.status === 429) {
          throw new ApiError('Rate limit exceeded. Retrying...', 429, errorData)
        }

        if (response.status >= 500) {
          throw new ApiError(`Server error: ${response.status}`, response.status, errorData)
        }

        if (response.status === 404) {
          throw new ApiError(`Not found: ${url}`, 404, errorData)
        }

        throw new ApiError(
          errorData.error || `HTTP ${response.status}`,
          response.status,
          errorData
        )
      }

      return await response.json()
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error))

      // Don't retry 404s or validation errors
      if (
        error instanceof ApiError &&
        (error.code === 404 || (error.code && error.code >= 400 && error.code < 500))
      ) {
        throw error
      }

      // If this was the last retry, throw
      if (i === retries - 1) {
        throw lastError
      }

      // Exponential backoff
      const delay = baseBackoff * Math.pow(2, i)
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }

  throw lastError || new Error('Max retries exceeded')
}

// ============= CORRECT API ROUTES (Per V0_FRONTEND_IMPLEMENTATION_MASTER.md) =============

export const api = {
  // POST /simulate/start - Start a new simulation
  startSimulation: async (payload: StartSimulationRequest): Promise<StartSimulationResponse> => {
    return fetchWithRetry(`${API_BASE}/simulate/start`, {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  },

  // POST /simulate/branch - Branch from a node
  branch: async (payload: BranchRequest): Promise<BranchResponse> => {
    return fetchWithRetry(`${API_BASE}/simulate/branch`, {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  },

  // GET /graph?session_id=<id> - Get full graph (QUERY PARAM, not path param)
  getGraph: async (sessionId: string): Promise<GraphResponse> => {
    return fetchWithRetry(`${API_BASE}/graph?session_id=${sessionId}`)
  },

  // GET /nodes/{node_id} - Get single node (PLURAL "nodes")
  getNode: async (nodeId: string): Promise<DecisionNode> => {
    return fetchWithRetry(`${API_BASE}/nodes/${nodeId}`)
  },

  // GET /jobs/{job_id} - Poll job status
  getJob: async (jobId: string): Promise<Job> => {
    return fetchWithRetry(`${API_BASE}/jobs/${jobId}`)
  },

  // GET /jobs/{job_id}/logs - Get job logs (PLURAL "logs")
  getJobLogs: async (jobId: string): Promise<JobLogsResponse> => {
    return fetchWithRetry(`${API_BASE}/jobs/${jobId}/logs`)
  },

  // POST /jobs/{job_id}/retry - Retry a failed job (MISSING ENDPOINT from old impl)
  retryJob: async (jobId: string): Promise<{ job_id: string; status: string }> => {
    return fetchWithRetry(`${API_BASE}/jobs/${jobId}/retry`, {
      method: 'POST',
      body: JSON.stringify({}),
    })
  },

  // GET /metrics/dashboard - Combined quality, grounding, performance, and alert dashboard
  getMetricsDashboard: async (limit = 20): Promise<MetricsDashboardResponse> => {
    return fetchWithRetry(`${API_BASE}/metrics/dashboard?limit=${limit}`)
  },

  // GET /curator/reviews - Curator audit log for dashboard and node review history
  getCuratorReviews: async (limit = 20, nodeId?: string): Promise<CuratorReviewListResponse> => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (nodeId) {
      params.set('node_id', nodeId)
    }
    return fetchWithRetry(`${API_BASE}/curator/reviews?${params.toString()}`)
  },

  // POST /curator/review - Approve/reject/edit a node with audit trail
  recordCuratorReview: async (payload: CuratorReviewPayload): Promise<{
    status: string
    review: Record<string, unknown>
  }> => {
    return fetchWithRetry(`${API_BASE}/curator/review`, {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  },

  // POST /log - Client telemetry sink (MISSING ENDPOINT)
  postClientLog: async (payload: {
    level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG'
    action: string
    message: string
    details?: Record<string, unknown>
  }): Promise<{ status: string }> => {
    return fetchWithRetry(`${API_BASE}/log`, {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  },

  // GET /health - Backend health check (MISSING ENDPOINT)
  healthCheck: async (): Promise<{
    status: string
    backend?: string
    ollama?: string
  }> => {
    return fetchWithRetry(`${API_BASE}/health`)
  },
}

// ============= ADAPTER LAYER (Normalize backend responses for frontend) =============

/**
 * Adapter A: Normalize graph edges (backend uses "from"/"to", frontend expects "source"/"target")
 */
export function adaptGraphEdges(
  backendEdges: Array<{
    _id?: string
    from: string
    to: string
    action?: string
    [key: string]: unknown
  }>
): GraphEdge[] {
  return backendEdges.map((edge, index) => ({
    id: edge._id ? String(edge._id) : `${edge.from}-${edge.to}-${edge.action || 'edge'}-${index}`,
    source: edge.from,
    target: edge.to,
    action: edge.action || '',
  }))
}

/**
 * Adapter E: Complete flow after job finishes
 * Call this after a branch/start job completes to reload graph with new nodes
 */
export async function loadGraphAfterJobCompletion(
  sessionId: string,
  completedNodeId?: string
): Promise<{ nodes: DecisionNode[]; edges: GraphEdge[] }> {
  const graph = await api.getGraph(sessionId)

  // Adapt edges from backend format to UI format
  const adaptedEdges = adaptGraphEdges(
    (graph.edges || []) as Array<{
      from: string
      to: string
      action?: string
      [key: string]: unknown
    }>
  )

  // If we have the completed node ID, optionally fetch it for immediate details.
  // The graph response remains the source of truth for the returned structure.
  if (completedNodeId) {
    try {
      await api.getNode(completedNodeId)
    } catch {
      // If node fetch fails, we'll just use the graph version.
    }
  }

  // Enrich nodes with topology signals for UI clarity
  const enrichedNodes = enrichNodesWithTopology(graph.nodes || [], adaptedEdges)

  return {
    nodes: enrichedNodes,
    edges: adaptedEdges,
  }
}

// Enrich nodes with frontend-only signals: `node_depth` and `branch_breadth` so the UI can
// surface graph quality signals (breadth, depth) without additional server calls.
function enrichNodesWithTopology(nodes: DecisionNode[], edges: GraphEdge[]): DecisionNode[] {
  const nodesById: Record<string, DecisionNode> = {}
  nodes.forEach(n => { nodesById[n.id] = { ...n } })

  const childrenMap: Record<string, string[]> = {}
  const parentsMap: Record<string, string[]> = {}
  edges.forEach(e => {
    childrenMap[e.source] = childrenMap[e.source] || []
    childrenMap[e.source].push(e.target)
    parentsMap[e.target] = parentsMap[e.target] || []
    parentsMap[e.target].push(e.source)
  })

  // Find root candidates: nodes with no parents or time_step === 0
  const roots = nodes.filter(n => (parentsMap[n.id] || []).length === 0 || n.time_step === 0).map(n => n.id)

  // BFS to compute minimal depth for each node
  const depthMap: Record<string, number> = {}
  const queue: string[] = []
  roots.forEach(r => { depthMap[r] = 0; queue.push(r) })

  while (queue.length) {
    const cur = queue.shift() as string
    const children = childrenMap[cur] || []
    children.forEach(childId => {
      const nextDepth = (depthMap[cur] ?? 0) + 1
      if (depthMap[childId] === undefined || nextDepth < depthMap[childId]) {
        depthMap[childId] = nextDepth
        queue.push(childId)
      }
    })
  }

  // Compute branch breadth: number of direct children for each node's parent grouping
  const breadthMap: Record<string, number> = {}
  Object.keys(childrenMap).forEach(parentId => {
    const count = childrenMap[parentId].length
    breadthMap[parentId] = count
  })

  // Attach computed signals onto a shallow copy of the nodes
  return Object.values(nodesById).map(n => ({
    ...n,
    node_depth: depthMap[n.id] ?? 0,
    branch_breadth: breadthMap[n.id] ?? 0,
  }))
}

/**
 * Adapter F: Evidence tab graceful degradation
 * Backend doesn't have GET /chunks/{cache_id}, so show citations as chips instead
 */
export function formatSourceCitations(citations: string[]): Array<{
  text: string
  isUrl: boolean
  href?: string
}> {
  return (citations || []).map(citation => {
    const isUrl = citation.startsWith('http')
    return {
      text: isUrl ? new URL(citation).hostname : citation,
      isUrl,
      href: isUrl ? citation : undefined,
    }
  })
}

/**
 * Adapter G: Ghost node handling when no cancel endpoint exists
 * Mark ghost as cancelled or show retry button
 */
export function getGhostNodeErrorActions(jobId: string): Array<{
  label: string
  action: 'retry' | 'dismiss'
  jobId?: string
}> {
  return [
    { label: 'Retry', action: 'retry', jobId },
    { label: 'Dismiss', action: 'dismiss' },
  ]
}
