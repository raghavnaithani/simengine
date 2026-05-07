'use client'

import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertTriangle,
  BarChart3,
  Clock3,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  ClipboardList,
  Users,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { api } from '@/lib/api'
import type {
  CuratorReviewListResponse,
  CuratorReviewRecord,
  MetricsDashboardResponse,
} from '@/lib/types'

interface Ws5DashboardPanelProps {
  isOpen: boolean
  onClose: () => void
}

function formatTimestamp(timestamp?: string | null): string {
  if (!timestamp) {
    return 'Unknown'
  }

  const parsed = new Date(timestamp)
  if (Number.isNaN(parsed.getTime())) {
    return timestamp
  }

  return parsed.toLocaleString()
}

function actionBadgeClass(action: CuratorReviewRecord['action']): string {
  switch (action) {
    case 'approve':
      return 'border-safe/30 bg-safe/20 text-safe'
    case 'reject':
      return 'border-danger/30 bg-danger/20 text-danger'
    default:
      return 'border-caution/30 bg-caution/20 text-caution'
  }
}

function MetricCard({
  title,
  value,
  note,
}: {
  title: string
  value: string
  note?: string
}) {
  return (
    <div className="rounded-xl border border-border bg-muted/30 p-4 shadow-sm">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{title}</div>
      <div className="mt-2 text-2xl font-semibold text-foreground">{value}</div>
      {note && <div className="mt-1 text-xs text-muted-foreground">{note}</div>}
    </div>
  )
}

function ReviewRow({ review }: { review: CuratorReviewRecord }) {
  return (
    <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className={actionBadgeClass(review.action)}>
              {review.action}
            </Badge>
            <span className="truncate text-sm font-medium text-foreground">Node {review.node_id}</span>
          </div>
          <p className="mt-2 text-sm text-muted-foreground">{review.reason}</p>
        </div>
        <div className="text-right text-xs text-muted-foreground">
          <div>{review.curator}</div>
          <div className="mt-1 font-mono">{formatTimestamp(review.created_at)}</div>
        </div>
      </div>
    </div>
  )
}

export function Ws5DashboardPanel({ isOpen, onClose }: Ws5DashboardPanelProps) {
  const [dashboard, setDashboard] = useState<MetricsDashboardResponse | null>(null)
  const [reviews, setReviews] = useState<CuratorReviewRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadDashboard = async () => {
    setLoading(true)
    setError(null)

    try {
      const [dashboardResult, reviewsResult] = await Promise.all([
        api.getMetricsDashboard(20),
        api.getCuratorReviews(10),
      ])

      setDashboard(dashboardResult)
      setReviews(reviewsResult.reviews || [])
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Unable to load dashboard data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      void loadDashboard()
    }
  }, [isOpen])

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.aside
          initial={{ x: '-100%', opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: '-100%', opacity: 0 }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="fixed left-0 top-0 z-40 h-full w-[520px] max-w-[calc(100vw-1rem)] border-r border-border bg-card shadow-2xl"
        >
          <div className="flex items-center justify-between border-b border-border px-6 py-4">
            <div>
              <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                <BarChart3 className="h-4 w-4" />
                WS5 Dashboard
              </div>
              <h2 className="mt-1 text-lg font-semibold text-foreground">Grounding, quality, and review health</h2>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" onClick={loadDashboard} disabled={loading} className="h-9 w-9">
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              </Button>
              <Button variant="ghost" size="icon" onClick={onClose} className="h-9 w-9">
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <ScrollArea className="h-[calc(100%-72px)] px-6 py-4">
            {loading && !dashboard ? (
              <div className="flex min-h-48 items-center justify-center text-sm text-muted-foreground">
                Loading dashboard data...
              </div>
            ) : (
              <div className="space-y-6 pb-6">
                {error && (
                  <div className="rounded-xl border border-danger/30 bg-danger/5 p-4 text-sm text-danger">
                    {error}
                  </div>
                )}

                <div className="grid gap-3 sm:grid-cols-2">
                  <MetricCard
                    title="Quality"
                    value={dashboard ? `${Math.round(dashboard.quality.quality_score * 100)}%` : '0%'}
                    note={`Citation rate ${dashboard ? Math.round(dashboard.quality.citation_rate * 100) : 0}%`}
                  />
                  <MetricCard
                    title="Grounding"
                    value={dashboard ? `${Math.round(dashboard.grounding.citation_rate * 100)}%` : '0%'}
                    note={`Fallback rate ${dashboard ? Math.round(dashboard.grounding.error_rate * 100) : 0}%`}
                  />
                  <MetricCard
                    title="Performance"
                    value={dashboard ? `${Math.round(dashboard.performance.latency_ms)} ms` : '0 ms'}
                    note={`Latest job ${dashboard?.latest_job_id || 'n/a'}`}
                  />
                  <MetricCard
                    title="Sample Size"
                    value={dashboard ? String(dashboard.total_jobs_sampled) : '0'}
                    note={`Latest ${dashboard ? formatTimestamp(dashboard.latest_timestamp) : 'n/a'}`}
                  />
                </div>

                <section className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <ShieldAlert className="h-4 w-4 text-caution" />
                    Alerts
                  </div>
                  {dashboard?.alerts?.length ? (
                    <div className="space-y-2">
                      {dashboard.alerts.map((alert) => (
                        <div key={`${alert.type}-${alert.metric}`} className="rounded-xl border border-border bg-muted/20 p-3 text-sm">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <span className="font-medium text-foreground">{alert.type}</span>
                            <Badge variant="outline">{alert.metric}</Badge>
                          </div>
                          <div className="mt-2 text-xs text-muted-foreground">
                            Latest {alert.latest} vs baseline {alert.baseline}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-xl border border-dashed border-border p-4 text-sm text-muted-foreground">
                      No active dashboard alerts.
                    </div>
                  )}
                </section>

                <section className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <ClipboardList className="h-4 w-4 text-primary" />
                    Failure taxonomy
                  </div>
                  {dashboard?.failure_taxonomy?.length ? (
                    <div className="space-y-2">
                      {dashboard.failure_taxonomy.map((entry) => (
                        <div key={entry.reason} className="flex items-center justify-between rounded-xl border border-border bg-card px-4 py-3 text-sm">
                          <span className="text-foreground">{entry.reason}</span>
                          <span className="rounded-full bg-muted px-2 py-0.5 font-mono text-xs text-muted-foreground">
                            {entry.count}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-xl border border-dashed border-border p-4 text-sm text-muted-foreground">
                      No failures recorded in the sampled jobs.
                    </div>
                  )}
                </section>

                <section className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <Users className="h-4 w-4 text-safe" />
                    Recent curator reviews
                  </div>
                  {reviews.length ? (
                    <div className="space-y-2">
                      {reviews.map((review) => (
                        <ReviewRow key={review.id} review={review} />
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-xl border border-dashed border-border p-4 text-sm text-muted-foreground">
                      No curator reviews recorded yet.
                    </div>
                  )}
                </section>

                <section className="rounded-xl border border-border bg-muted/20 p-4 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2 font-semibold text-foreground">
                    <Clock3 className="h-4 w-4" />
                    Dashboard scope
                  </div>
                  <p className="mt-2 leading-relaxed">
                    This panel reflects backend telemetry for citation quality, grounding, latency, and curator audit history.
                  </p>
                </section>
              </div>
            )}
          </ScrollArea>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}
