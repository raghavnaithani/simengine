// Core Data Models for Decision Graph Simulator

export type RiskSeverity = 'Low' | 'Medium' | 'High' | 'Critical'
export type RiskLikelihood = 'Low' | 'Medium' | 'High'
export type VerificationStatus = 'verified' | 'unverified' | 'failed'
export type SimulationMode = 'Quick' | 'Analytical'
export type JobStatus = 'queued' | 'started' | 'running' | 'completed' | 'failed'
export type CuratorReviewStatus = 'pending' | 'approve' | 'reject' | 'edit' | 'approved' | 'rejected' | 'edited'

export interface Risk {
  id: string
  description: string
  severity: RiskSeverity
  likelihood?: RiskLikelihood
  mitigation_strategy?: string
  citation?: string
}

export interface Alternative {
  id: string
  label?: string
  description?: string
  action_type: string
  expected_outcome_summary?: string
}

export interface DecisionNode {
  id: string
  title: string
  summary: string
  description: string
  time_step: number
  alternatives: Alternative[]
  risks: Risk[]
  source_citations: string[]
  citation_provenance?: Array<Record<string, unknown>>
  citation_quality_score?: number
  citation_coverage?: number
  confidence_score: number
  speculative: boolean
  created_at?: string
  curator_review_status?: CuratorReviewStatus
  curator_review_reason?: string
  curator_reviewed_by?: string
  curator_reviewed_at?: string
  meta?: {
    prompt_tokens?: number
    vector_distance?: number
    llm_model?: string
  }
}

export interface KnowledgeChunk {
  id: string
  content: string
  source_url: string
  source_title?: string
  chunk_index: number
  embedding?: number[]
  created_at?: string
  ttl_days?: number
  verification_status: VerificationStatus
  similarity_score?: number
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  action?: string
}

export interface Session {
  id: string
  name: string
  created_at: string
  last_modified: string
  root_node_id?: string
  node_count: number
}

export interface Job {
  job_id: string
  type: 'start' | 'branch'
  status: JobStatus
  progress?: number
  created_at?: string
  updated_at?: string
  result?: {
    node_id: string
    session_id?: string
  } | null
  error?: string | null
  payload?: Record<string, unknown>
}

export interface JobLogsResponse {
  count: number
  returned_count: number
  logs: Array<{
    job_id: string
    raw?: string | null
    clean?: unknown | null
    node?: unknown | null
    prompt?: string | null
    created_at: string
    success: boolean
    error?: string | null
  }>
}

export type CuratorReviewAction = 'approve' | 'reject' | 'edit'

export interface CuratorReviewPayload {
  node_id: string
  session_id?: string | null
  curator: string
  action: CuratorReviewAction
  reason: string
  updates?: Record<string, unknown>
}

export interface CuratorReviewRecord {
  id: string
  node_id: string
  session_id?: string | null
  curator: string
  action: CuratorReviewAction
  reason: string
  before: Record<string, unknown>
  after: Record<string, unknown>
  created_at: string
}

export interface CuratorReviewListResponse {
  count: number
  reviews: CuratorReviewRecord[]
}

export interface DashboardAlert {
  type: string
  metric: string
  latest: number
  baseline: number
}

export interface FailureTaxonomyEntry {
  reason: string
  count: number
}

export interface MetricsDashboardResponse {
  total_jobs_sampled: number
  status: string
  quality: {
    citation_rate: number
    diversity_score: number
    quality_score: number
  }
  grounding: {
    citation_rate: number
    error_rate: number
    alternatives_count: number
    title_novelty_score: number
    risk_specificity_score: number
  }
  performance: {
    latency_ms: number
  }
  averages: Record<string, number>
  alerts: DashboardAlert[]
  failure_taxonomy: FailureTaxonomyEntry[]
  latest_job_id?: string | null
  latest_timestamp?: string | null
}

export interface TerminalEntry {
  id: string
  time: string
  level: 'info' | 'warn' | 'error' | 'debug'
  message: string
  meta?: Record<string, unknown>
}

// API Request/Response Types
export interface StartSimulationRequest {
  prompt: string
  mode: SimulationMode
  persona: string
  simulate_steps?: number
  seed?: number | null
}

export interface StartSimulationResponse {
  session_id: string
  job_id: string
  status: JobStatus
  root_node_id?: string
}

export interface BranchRequest {
  session_id: string
  parent_node_id: string
  action: string
  persona?: string
  seed?: number | null
}

export interface BranchResponse {
  job_id: string
  status: JobStatus
}

export interface GraphResponse {
  nodes: DecisionNode[]
  edges: Array<{
    from: string
    to: string
    action?: string
    session_id?: string
    created_at?: string
  }>
}

// UI State Types
export interface GhostNode {
  temp_id: string
  parent_id: string
  job_id?: string
  status: 'pending' | 'error'
  created_at: string
  action?: string
  error?: string
}

export type AppPhase = 'landing' | 'questionnaire' | 'world-building' | 'canvas'

export type TimelineHorizon = '3-months' | '6-months' | '12-months'

export interface ClarifyingQuestion {
  id: string
  question: string
  type: 'single' | 'multiple' | 'text' | 'scale'
  options?: string[]
  answer?: string | string[] | number
}

export interface ScenarioConfig {
  prompt: string
  mode: SimulationMode
  persona: string
  timeline: TimelineHorizon
  simulate_steps: number
  temperature: number
  clarifications: ClarifyingQuestion[]
}

export interface BranchDialogState {
  open: boolean
  parentNodeId: string | null
  defaultAction: string
  customPrompt: string
}
