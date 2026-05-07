'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  FileText, 
  AlertTriangle, 
  Code2, 
  ExternalLink, 
  Copy, 
  Check,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  Sparkles,
  BookOpen,
  Loader2,
  ThumbsUp,
  ThumbsDown,
  PencilLine,
  UserRound,
  MessageSquareText,
  Save
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { useSessionStore } from '@/lib/store'
import { api } from '@/lib/api'
import { useToast } from './toast-provider'
import type { CuratorReviewAction, DecisionNode, Risk } from '@/lib/types'

interface FocusPanelProps {
  nodeId: string | null
  isOpen: boolean
  onClose: () => void
}

function formatCuratorStatus(status?: string): string {
  if (!status) {
    return 'Not reviewed'
  }

  switch (status) {
    case 'approve':
    case 'approved':
      return 'Approved'
    case 'reject':
    case 'rejected':
      return 'Rejected'
    case 'edit':
    case 'edited':
      return 'Edited'
    case 'pending':
      return 'Pending'
    default:
      return status
  }
}

// Risk severity badge component
function RiskBadge({ severity }: { severity: Risk['severity'] }) {
  const styles = {
    Critical: 'bg-danger/20 text-danger border-danger/30',
    High: 'bg-danger/20 text-danger border-danger/30',
    Medium: 'bg-caution/20 text-caution border-caution/30',
    Low: 'bg-safe/20 text-safe border-safe/30',
  }
  
  return (
    <Badge variant="outline" className={`${styles[severity]} font-medium`}>
      {severity}
    </Badge>
  )
}

// Risk icon component
function RiskIcon({ severity }: { severity: Risk['severity'] }) {
  switch (severity) {
    case 'Critical':
    case 'High':
      return <ShieldAlert className="h-4 w-4 text-danger" />
    case 'Medium':
      return <ShieldQuestion className="h-4 w-4 text-caution" />
    default:
      return <ShieldCheck className="h-4 w-4 text-safe" />
  }
}

// Citation chip component - handles graceful degradation when chunk API not available
function CitationChip({ citation }: { citation: string }) {
  const [copied, setCopied] = useState(false)
  
  // Check if citation is a URL or cache ID
  const isUrl = citation.startsWith('http')
  const displayText = isUrl ? new URL(citation).hostname : citation.slice(0, 12) + '...'

  const copyUrl = async () => {
    if (isUrl) {
      await navigator.clipboard.writeText(citation)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <TooltipProvider>
      <Tooltip delayDuration={300}>
        <TooltipTrigger asChild>
          {isUrl ? (
            <a
              href={citation}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 rounded-full bg-primary/20 px-2 py-0.5 text-xs font-medium text-primary hover:bg-primary/30 transition-colors"
            >
              <ExternalLink className="h-3 w-3" />
              <span className="font-mono">{displayText}</span>
            </a>
          ) : (
            <button
              className="inline-flex items-center gap-1 rounded-full bg-primary/20 px-2 py-0.5 text-xs font-medium text-primary hover:bg-primary/30 transition-colors cursor-help"
            >
              <BookOpen className="h-3 w-3" />
              <span className="font-mono">{displayText}</span>
            </button>
          )}
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-sm p-0">
          <div className="p-3 space-y-2">
            {isUrl ? (
              <>
                <p className="text-sm font-medium">External Source</p>
                <p className="text-xs text-muted-foreground font-mono break-all">
                  {citation}
                </p>
                <div className="flex items-center justify-between pt-2 border-t border-border">
                  <a
                    href={citation}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-xs text-primary hover:underline"
                  >
                    <ExternalLink className="h-3 w-3" />
                    Open Source
                  </a>
                  <button
                    onClick={copyUrl}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                  >
                    {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                    {copied ? 'Copied' : 'Copy URL'}
                  </button>
                </div>
              </>
            ) : (
              <>
                <p className="text-sm font-medium">Reference ID (preview unavailable)</p>
                <p className="text-xs text-muted-foreground">
                  This citation maps to backend knowledge storage, but direct preview is not exposed.
                </p>
              </>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export function FocusPanel({ nodeId, isOpen, onClose }: FocusPanelProps) {
  const { nodes, addNode, sessionId } = useSessionStore()
  const { addToast } = useToast()
  const [copiedSummary, setCopiedSummary] = useState(false)
  const [copiedJson, setCopiedJson] = useState(false)
  const [activeTab, setActiveTab] = useState('narrative')
  const [curatorName, setCuratorName] = useState('curator')
  const [reviewReason, setReviewReason] = useState('')
  const [editDraft, setEditDraft] = useState({
    title: '',
    summary: '',
    description: '',
  })
  const [reviewBusy, setReviewBusy] = useState(false)
  const [reviewNotice, setReviewNotice] = useState<string | null>(null)

  const node = nodes.find(n => n.id === nodeId)

  useEffect(() => {
    // Reset to narrative tab when node changes
    setActiveTab('narrative')
    if (node) {
      setCuratorName(node.curator_reviewed_by || 'curator')
      setReviewReason(node.curator_review_reason || '')
      setEditDraft({
        title: node.title,
        summary: node.summary,
        description: node.description,
      })
      setReviewNotice(
        node.curator_review_status
          ? `Existing curator review: ${node.curator_review_status}`
          : null
      )
    } else {
      setCuratorName('curator')
      setReviewReason('')
      setEditDraft({ title: '', summary: '', description: '' })
      setReviewNotice(null)
    }
  }, [nodeId])

  const copySummary = async () => {
    if (node) {
      await navigator.clipboard.writeText(`${node.title}\n\n${node.summary}\n\n${node.description}`)
      setCopiedSummary(true)
      setTimeout(() => setCopiedSummary(false), 2000)
    }
  }

  const copyJson = async () => {
    if (node) {
      await navigator.clipboard.writeText(JSON.stringify(node, null, 2))
      setCopiedJson(true)
      setTimeout(() => setCopiedJson(false), 2000)
    }
  }

  const submitCuratorReview = async (action: CuratorReviewAction) => {
    if (!node) {
      return
    }

    const trimmedReason = reviewReason.trim()
    if (!trimmedReason) {
      setReviewNotice('Add a review reason before submitting.')
      return
    }

    if (
      action === 'edit' &&
      (!editDraft.title.trim() || !editDraft.summary.trim() || !editDraft.description.trim())
    ) {
      setReviewNotice('Edit reviews need title, summary, and description values.')
      return
    }

    setReviewBusy(true)
    setReviewNotice(null)

    try {
      await api.recordCuratorReview({
        node_id: node.id,
        session_id: sessionId,
        curator: curatorName.trim() || 'curator',
        action,
        reason: trimmedReason,
        updates:
          action === 'edit'
            ? {
                title: editDraft.title.trim(),
                summary: editDraft.summary.trim(),
                description: editDraft.description.trim(),
              }
            : undefined,
      })

      const refreshedNode = await api.getNode(node.id)
      addNode(refreshedNode)

      const statusText = action === 'edit' ? 'edited' : action
      setReviewNotice(`Curator review ${statusText} successfully saved.`)
      addToast({
        type: 'success',
        title: 'Curator review saved',
        message: `${statusText.toUpperCase()} review recorded for ${node.title}`,
      })
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unable to save curator review'
      setReviewNotice(errorMessage)
      addToast({
        type: 'error',
        title: 'Curator review failed',
        message: errorMessage,
      })
    } finally {
      setReviewBusy(false)
    }
  }

  return (
    <AnimatePresence>
      {isOpen && node && (
        <motion.div
          initial={{ x: '100%', opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: '100%', opacity: 0 }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="fixed right-0 top-0 z-40 h-full w-[450px] border-l border-border bg-card shadow-2xl"
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-border px-6 py-4">
            <div className="flex-1 min-w-0">
              <h2 className="truncate text-lg font-semibold text-foreground">
                {node.title}
              </h2>
              <div className="mt-1 flex items-center gap-2">
                <span className="text-xs text-muted-foreground font-mono">
                  t={node.time_step}
                </span>
                {node.speculative && (
                  <Badge variant="outline" className="bg-speculative/20 text-speculative border-speculative/30">
                    <AlertTriangle className="mr-1 h-3 w-3" />
                    Speculative
                  </Badge>
                )}
                <div className="flex items-center gap-1 text-xs">
                  <Sparkles className="h-3 w-3 text-primary" />
                  <span className={`font-medium ${
                    node.confidence_score >= 0.8 ? 'text-safe' :
                    node.confidence_score >= 0.5 ? 'text-caution' : 'text-danger'
                  }`}>
                    {Math.round(node.confidence_score * 100)}%
                  </span>
                </div>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="shrink-0 text-muted-foreground hover:text-foreground"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          {/* Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="flex h-[calc(100%-80px)] flex-col">
            <TabsList className="mx-6 mt-4 grid w-auto grid-cols-5">
              <TabsTrigger value="narrative" className="gap-1">
                <FileText className="h-3 w-3" />
                <span className="hidden sm:inline">Narrative</span>
              </TabsTrigger>
              <TabsTrigger value="evidence" className="gap-1">
                <BookOpen className="h-3 w-3" />
                <span className="hidden sm:inline">Evidence</span>
              </TabsTrigger>
              <TabsTrigger value="risks" className="gap-1">
                <AlertTriangle className="h-3 w-3" />
                <span className="hidden sm:inline">Risks</span>
                {node.risks.length > 0 && (
                  <span className="ml-1 rounded-full bg-danger/20 px-1.5 text-xs text-danger">
                    {node.risks.length}
                  </span>
                )}
              </TabsTrigger>
              <TabsTrigger value="json" className="gap-1">
                <Code2 className="h-3 w-3" />
                <span className="hidden sm:inline">JSON</span>
              </TabsTrigger>
              <TabsTrigger value="curator" className="gap-1">
                <PencilLine className="h-3 w-3" />
                <span className="hidden sm:inline">Curator</span>
              </TabsTrigger>
            </TabsList>

            <ScrollArea className="flex-1 px-6 py-4">
              {/* Narrative Tab */}
              <TabsContent value="narrative" className="mt-0 space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-medium text-muted-foreground">Description</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={copySummary}
                      className="h-7 gap-1 text-xs text-muted-foreground hover:text-foreground"
                    >
                      {copiedSummary ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                      {copiedSummary ? 'Copied' : 'Copy'}
                    </Button>
                  </div>
                  <p className="text-sm text-foreground leading-relaxed">
                    {node.description}
                  </p>
                </div>

                {/* Meta information */}
                {node.meta && (
                  <div className="rounded-lg border border-border bg-muted/30 p-4 space-y-2">
                    <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      Generation Metadata
                    </h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      {node.meta.llm_model && (
                        <div>
                          <span className="text-muted-foreground">Model:</span>
                          <span className="ml-2 font-mono text-foreground">{node.meta.llm_model}</span>
                        </div>
                      )}
                      {node.meta.prompt_tokens && (
                        <div>
                          <span className="text-muted-foreground">Tokens:</span>
                          <span className="ml-2 font-mono text-foreground">{node.meta.prompt_tokens}</span>
                        </div>
                      )}
                      {node.meta.vector_distance !== undefined && (
                        <div>
                          <span className="text-muted-foreground">Vector Dist:</span>
                          <span className="ml-2 font-mono text-foreground">
                            {node.meta.vector_distance.toFixed(4)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </TabsContent>

              {/* Evidence Tab */}
              <TabsContent value="evidence" className="mt-0 space-y-4">
                <p className="text-sm text-muted-foreground">
                  Sources used to ground this decision node:
                </p>
                {node.source_citations.length === 0 ? (
                  <div className="rounded-lg border border-dashed border-border p-6 text-center">
                    <BookOpen className="mx-auto h-8 w-8 text-muted-foreground/50" />
                    <p className="mt-2 text-sm text-muted-foreground">
                      No citations available
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="flex flex-wrap gap-2">
                      {node.source_citations.map((citation) => (
                        <CitationChip 
                          key={citation} 
                          citation={citation}
                        />
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground italic">
                      Hover over citations to view details. External URLs open in a new tab.
                    </p>
                  </div>
                )}
              </TabsContent>

              {/* Risks Tab */}
              <TabsContent value="risks" className="mt-0 space-y-4">
                {node.risks.length === 0 ? (
                  <div className="rounded-lg border border-dashed border-border p-6 text-center">
                    <ShieldCheck className="mx-auto h-8 w-8 text-safe/50" />
                    <p className="mt-2 text-sm text-muted-foreground">
                      No identified risks
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {node.risks.map((risk) => (
                      <div
                        key={risk.id}
                        className={`rounded-lg border p-4 ${
                          risk.severity === 'Critical' || risk.severity === 'High'
                            ? 'border-danger/30 bg-danger/5'
                            : risk.severity === 'Medium'
                              ? 'border-caution/30 bg-caution/5'
                              : 'border-safe/30 bg-safe/5'
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          <RiskIcon severity={risk.severity} />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <RiskBadge severity={risk.severity} />
                              {risk.likelihood && (
                                <span className="text-xs text-muted-foreground">
                                  Likelihood: {risk.likelihood}
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-foreground">
                              {risk.description}
                            </p>
                            {risk.mitigation_strategy && (
                              <div className="mt-2 pl-3 border-l-2 border-muted">
                                <p className="text-xs text-muted-foreground italic">
                                  Mitigation: {risk.mitigation_strategy}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>

              {/* JSON Tab */}
              <TabsContent value="json" className="mt-0">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-muted-foreground">Raw JSON</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={copyJson}
                    className="h-7 gap-1 text-xs text-muted-foreground hover:text-foreground"
                  >
                    {copiedJson ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                    {copiedJson ? 'Copied' : 'Copy'}
                  </Button>
                </div>
                <pre className="rounded-lg border border-border bg-background p-4 text-xs font-mono text-muted-foreground overflow-x-auto">
                  {JSON.stringify(node, null, 2)}
                </pre>
              </TabsContent>

              {/* Curator Tab */}
              <TabsContent value="curator" className="mt-0 space-y-4">
                <div className="rounded-lg border border-border bg-muted/20 p-4 space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <MessageSquareText className="h-4 w-4 text-primary" />
                    Curator review status
                  </div>
                  <div className="grid gap-2 text-sm sm:grid-cols-2">
                    <div>
                      <span className="text-muted-foreground">Current status:</span>
                      <span className="ml-2 font-medium text-foreground">
                        {formatCuratorStatus(node.curator_review_status)}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Reviewed by:</span>
                      <span className="ml-2 font-medium text-foreground">
                        {node.curator_reviewed_by || 'n/a'}
                      </span>
                    </div>
                  </div>
                  {node.curator_review_reason && (
                    <div className="rounded-md border border-border bg-card p-3 text-sm text-muted-foreground">
                      {node.curator_review_reason}
                    </div>
                  )}
                  {reviewNotice && (
                    <div className="rounded-md border border-border bg-background p-3 text-sm text-muted-foreground">
                      {reviewNotice}
                    </div>
                  )}
                </div>

                <div className="space-y-3 rounded-lg border border-border bg-card p-4">
                  <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <UserRound className="h-4 w-4 text-safe" />
                    Review metadata
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Curator name
                    </label>
                    <Input
                      value={curatorName}
                      onChange={(event) => setCuratorName(event.target.value)}
                      placeholder="curator"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Reason
                    </label>
                    <Textarea
                      value={reviewReason}
                      onChange={(event) => setReviewReason(event.target.value)}
                      placeholder="Explain why this node should be approved, rejected, or edited."
                      className="min-h-24"
                    />
                  </div>
                </div>

                <div className="space-y-3 rounded-lg border border-border bg-card p-4">
                  <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <Save className="h-4 w-4 text-primary" />
                    Optional edit payload
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Title
                    </label>
                    <Input
                      value={editDraft.title}
                      onChange={(event) => setEditDraft(prev => ({ ...prev, title: event.target.value }))}
                      placeholder="Revised title"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Summary
                    </label>
                    <Textarea
                      value={editDraft.summary}
                      onChange={(event) => setEditDraft(prev => ({ ...prev, summary: event.target.value }))}
                      className="min-h-20"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Description
                    </label>
                    <Textarea
                      value={editDraft.description}
                      onChange={(event) => setEditDraft(prev => ({ ...prev, description: event.target.value }))}
                      className="min-h-28"
                    />
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    onClick={() => void submitCuratorReview('approve')}
                    disabled={reviewBusy}
                    className="gap-2"
                  >
                    {reviewBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ThumbsUp className="h-4 w-4" />}
                    Approve
                  </Button>
                  <Button
                    type="button"
                    variant="destructive"
                    onClick={() => void submitCuratorReview('reject')}
                    disabled={reviewBusy}
                    className="gap-2"
                  >
                    {reviewBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ThumbsDown className="h-4 w-4" />}
                    Reject
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => void submitCuratorReview('edit')}
                    disabled={reviewBusy}
                    className="gap-2"
                  >
                    {reviewBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <PencilLine className="h-4 w-4" />}
                    Save edit review
                  </Button>
                </div>
              </TabsContent>
            </ScrollArea>
          </Tabs>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
