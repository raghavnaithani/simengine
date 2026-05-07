'use client'

import { useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Plus, 
  GitBranch,
  Clock,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useSessionStore } from '@/lib/store'

interface SessionSidebarProps {
  isOpen: boolean
  onClose: () => void
  onNewSession: () => void
}

function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return date.toLocaleDateString()
}

export function SessionSidebar({ isOpen, onClose, onNewSession }: SessionSidebarProps) {
  const { sessionId, sessions, nodes } = useSessionStore()
  const activeSession = useMemo(() => sessions.find((s) => s.id === sessionId), [sessions, sessionId])
  const displayedNodeCount = activeSession ? Math.max(activeSession.node_count, nodes.length) : nodes.length

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ x: '-100%', opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: '-100%', opacity: 0 }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="fixed left-0 top-0 z-40 h-full w-80 border-r border-border bg-sidebar shadow-2xl"
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-sidebar-border px-4 py-4">
            <h2 className="text-lg font-semibold text-sidebar-foreground">
              Sessions
            </h2>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="text-sidebar-foreground/70 hover:text-sidebar-foreground"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          {/* Actions */}
          <div className="flex border-b border-sidebar-border p-4">
            <Button 
              onClick={onNewSession}
              className="w-full gap-2"
            >
              <Plus className="h-4 w-4" />
              New Session
            </Button>
          </div>

          {/* Session Details */}
          <ScrollArea className="h-[calc(100%-132px)] px-4 py-4">
            {!activeSession ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <GitBranch className="h-12 w-12 text-muted-foreground/30" />
                <p className="mt-4 text-sm text-muted-foreground">
                  No active session yet
                </p>
                <Button 
                  variant="link" 
                  onClick={onNewSession}
                  className="mt-2"
                >
                  Start your first simulation
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="rounded-lg border border-primary/30 bg-primary/5 p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Current Session</p>
                  <h3 className="mt-1 text-sm font-semibold text-foreground">{activeSession.name}</h3>
                  <p className="mt-2 break-all font-mono text-xs text-muted-foreground">{activeSession.id}</p>
                  <div className="mt-3 flex items-center gap-3 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <GitBranch className="h-3 w-3" />
                      {displayedNodeCount} nodes
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {formatRelativeTime(activeSession.last_modified)}
                    </span>
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Local Session History</p>
                  {sessions.length === 0 ? (
                    <p className="text-xs text-muted-foreground">No persisted sessions yet.</p>
                  ) : (
                    <div className="space-y-2">
                      {sessions.map((session) => (
                        <div key={session.id} className="rounded-lg border border-border bg-card p-3">
                          <p className="truncate text-sm font-medium text-foreground">{session.name}</p>
                          <p className="mt-1 text-xs text-muted-foreground">{session.id}</p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            {session.node_count} nodes • {formatRelativeTime(session.last_modified)}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <p className="text-xs text-muted-foreground">
                  Session switching across backend graphs is pending dedicated backend endpoints.
                </p>
              </div>
            )}
          </ScrollArea>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
