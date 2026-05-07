'use client'

import { memo, useState } from 'react'
import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import { motion } from 'framer-motion'
import { 
  ShieldAlert, 
  ShieldCheck, 
  ShieldQuestion, 
  Plus, 
  MoreHorizontal,
  AlertTriangle,
  Sparkles,
  ExternalLink
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import type { DecisionNode as DecisionNodeType, Alternative, Risk } from '@/lib/types'

interface DecisionNodeData extends DecisionNodeType, Record<string, unknown> {
  onBranch: (nodeId: string, alternative: Alternative) => void
  onFocus: (nodeId: string) => void
  isSelected?: boolean
}

type DecisionFlowNode = Node<DecisionNodeData, 'decision'>
type DecisionNodeProps = NodeProps<DecisionFlowNode>

function getAlternativeLabel(alt: Alternative): string {
  return alt.label || alt.action_type || alt.description || 'Option'
}

// Get the highest risk severity color
function getRiskColor(risks: Risk[]): string {
  if (risks.some(r => r.severity === 'Critical')) return 'bg-danger'
  if (risks.some(r => r.severity === 'High')) return 'bg-danger'
  if (risks.some(r => r.severity === 'Medium')) return 'bg-caution'
  return 'bg-safe'
}

// Get risk icon based on severity
function RiskIcon({ severity }: { severity: Risk['severity'] }) {
  switch (severity) {
    case 'Critical':
    case 'High':
      return <ShieldAlert className="h-3 w-3 text-danger" />
    case 'Medium':
      return <ShieldQuestion className="h-3 w-3 text-caution" />
    default:
      return <ShieldCheck className="h-3 w-3 text-safe" />
  }
}

// Confidence badge component
function ConfidenceBadge({ score, speculative, hasCitations }: { score: number; speculative: boolean; hasCitations: boolean }) {
  // Determine status type
  if (speculative) {
    return (
      <div 
        title="Speculative: Low-confidence claim, may lack evidence"
        className="flex items-center gap-1 rounded-full bg-amber-500/20 px-2 py-0.5 text-xs font-medium text-amber-600 dark:text-amber-400"
      >
        <AlertTriangle className="h-3 w-3" />
        <span>Speculative</span>
      </div>
    )
  }

  if (!hasCitations) {
    return (
      <div 
        title="Fallback: Generated without source verification"
        className="flex items-center gap-1 rounded-full bg-red-500/20 px-2 py-0.5 text-xs font-medium text-red-600 dark:text-red-400"
      >
        <AlertTriangle className="h-3 w-3" />
        <span>Unverified</span>
      </div>
    )
  }

  const color = score >= 0.8 
    ? 'bg-safe/20 text-safe' 
    : score >= 0.5 
      ? 'bg-caution/20 text-caution' 
      : 'bg-danger/20 text-danger'

  return (
    <div 
      title={`Confidence: ${Math.round(score * 100)}%`}
      className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${color}`}
    >
      <Sparkles className="h-3 w-3" />
      <span>{Math.round(score * 100)}%</span>
    </div>
  )
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

export const DecisionNodeComponent = memo(function DecisionNodeComponent({ 
  data,
  selected,
}: DecisionNodeProps) {
  const [branchMenuOpen, setBranchMenuOpen] = useState(false)
  
  const {
    id,
    title,
    summary,
    risks = [],
    alternatives = [],
    source_citations = [],
    confidence_score = 0,
    speculative = false,
    curator_review_status,
    onBranch,
    onFocus,
    isSelected,
  } = data

  const handleNodeClick = () => {
    onFocus(id)
  }

  const handleBranch = (alternative: Alternative) => {
    setBranchMenuOpen(false)
    onBranch(id, alternative)
  }

  const isActive = selected || isSelected
  const riskColor = getRiskColor(risks)
  const displayedRisks = risks.slice(0, 5)
  const hasMoreRisks = risks.length > 5

  return (
    <>
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="!h-3 !w-3 !border-2 !border-border !bg-card"
      />

      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.22, ease: [0.2, 0.9, 0.3, 1] }}
        onClick={handleNodeClick}
        className={`group w-80 cursor-pointer overflow-hidden rounded-xl border transition-all duration-200 ${
          isActive 
            ? 'border-primary shadow-lg shadow-primary/20' 
            : 'border-border hover:border-primary/50'
        }`}
        role="button"
        aria-label={`Node: ${title}. Confidence ${Math.round(confidence_score * 100)}%`}
        aria-pressed={isActive}
        tabIndex={0}
        data-testid={`decision-node-${id}`}
      >
        {/* Status Bar */}
        <div className={`h-1 ${riskColor}`} />

        {/* Card Content */}
        <div className="bg-card p-4">
          {/* Header */}
          <div className="flex items-start justify-between gap-2">
            <h3 className="line-clamp-1 flex-1 font-semibold text-card-foreground">
              {title}
            </h3>
            <ConfidenceBadge 
              score={confidence_score} 
              speculative={speculative}
              hasCitations={source_citations.length > 0}
            />
          </div>

          {/* Summary */}
          <p className="mt-2 line-clamp-3 text-sm text-muted-foreground leading-relaxed">
            {summary}
          </p>

          {/* Risk Indicators */}
          {displayedRisks.length > 0 && (
            <div className="mt-3 flex items-center gap-1">
              {displayedRisks.map((risk) => (
                <div
                  key={risk.id}
                  title={`${risk.severity}: ${risk.description}`}
                  className="rounded p-1 hover:bg-muted transition-colors"
                >
                  <RiskIcon severity={risk.severity} />
                </div>
              ))}
              {hasMoreRisks && (
                <span className="text-xs text-muted-foreground">
                  +{risks.length - 5} more
                </span>
              )}
            </div>
          )}

          {/* Source Citations */}
          {source_citations.length > 0 && (
            <div className="mt-2 flex items-center gap-1">
              <ExternalLink className="h-3 w-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground font-mono">
                {source_citations.length} sources
              </span>
            </div>
          )}

          {curator_review_status && (
            <div
              className="mt-2 inline-flex items-center gap-1 rounded-full border border-border bg-muted/40 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground"
              data-testid={`curator-status-${id}`}
            >
              <span>Curator {formatCuratorStatus(curator_review_status)}</span>
            </div>
          )}
        </div>

        {/* Footer with Actions */}
        <div className="flex items-center justify-between border-t border-border bg-muted/30 px-4 py-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-7 px-2 text-muted-foreground hover:text-foreground"
              >
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              <DropdownMenuItem onClick={() => onFocus(id)}>
                View Details
              </DropdownMenuItem>
              <DropdownMenuItem>
                Export Node
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Branch Menu */}
          <DropdownMenu open={branchMenuOpen} onOpenChange={setBranchMenuOpen}>
            <DropdownMenuTrigger asChild>
              <Button
                size="sm"
                className="h-7 gap-1 bg-primary/10 text-primary hover:bg-primary/20"
                aria-label="Branch from this node"
              >
                <Plus className="h-3 w-3" />
                <span className="text-xs">Branch</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent 
              align="end" 
              className="w-64"
              role="menu"
            >
              <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
                Choose an action
              </div>
              {alternatives.map((alt) => (
                <DropdownMenuItem
                  key={alt.id}
                  onClick={() => handleBranch(alt)}
                  className="flex flex-col items-start gap-0.5"
                  role="menuitem"
                >
                  <span className="font-medium">{getAlternativeLabel(alt)}</span>
                  {alt.description && (
                    <span className="text-xs text-muted-foreground">
                      {alt.description}
                    </span>
                  )}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </motion.div>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-3 !w-3 !border-2 !border-primary !bg-primary"
      />
    </>
  )
})

// Ghost Node for pending operations
interface GhostNodeData extends Record<string, unknown> {
  status: 'pending' | 'error'
  action?: string
  error?: string
  onCancel?: () => void
}

type GhostFlowNode = Node<GhostNodeData, 'ghost'>
type GhostNodeProps = NodeProps<GhostFlowNode>

export const GhostNodeComponent = memo(function GhostNodeComponent({ 
  data 
}: GhostNodeProps) {
  const { status, action, error, onCancel } = data

  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        className="!h-3 !w-3 !border-2 !border-border !bg-card"
      />

      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ 
          scale: status === 'pending' ? [1, 1.02, 1] : 1, 
          opacity: status === 'pending' ? 0.6 : 1 
        }}
        transition={status === 'pending' ? {
          scale: { duration: 1.5, repeat: Infinity, ease: 'easeInOut' },
          opacity: { duration: 0.3 }
        } : { duration: 0.3 }}
        className={`w-80 overflow-hidden rounded-xl border ${
          status === 'error' 
            ? 'border-danger/50 bg-danger/10' 
            : 'border-border/50 bg-card/50'
        }`}
      >
        <div className="h-1 bg-system" />
        
        <div className="flex flex-col items-center justify-center p-8">
          {status === 'pending' ? (
            <>
              <div className="relative">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                {action ? `Processing: ${action}...` : 'Generating node...'}
              </p>
            </>
          ) : (
            <>
              <ShieldAlert className="h-8 w-8 text-danger" />
              <p className="mt-4 text-sm text-danger">
                {error || 'Failed to generate node'}
              </p>
              {onCancel && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onCancel}
                  className="mt-2 text-muted-foreground hover:text-foreground"
                >
                  Dismiss
                </Button>
              )}
            </>
          )}
        </div>
      </motion.div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-3 !w-3 !border-2 !border-border !bg-muted"
      />
    </>
  )
})
