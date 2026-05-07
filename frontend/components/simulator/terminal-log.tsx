'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Terminal, X, Minimize2, Maximize2, Copy, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { TerminalEntry } from '@/lib/types'

interface TerminalLogProps {
  entries: TerminalEntry[]
  isOpen: boolean
  onClose?: () => void
  onMinimize?: () => void
  minimized?: boolean
  className?: string
}

const levelColors = {
  info: 'text-foreground',
  warn: 'text-caution',
  error: 'text-danger',
  debug: 'text-muted-foreground',
}

const levelBadgeColors = {
  info: 'bg-primary/20 text-primary',
  warn: 'bg-caution/20 text-caution',
  error: 'bg-danger/20 text-danger',
  debug: 'bg-muted text-muted-foreground',
}

export function TerminalLog({ 
  entries, 
  isOpen, 
  onClose, 
  onMinimize,
  minimized = false,
  className = '' 
}: TerminalLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [displayedEntries, setDisplayedEntries] = useState<TerminalEntry[]>([])
  const [currentTypingIndex, setCurrentTypingIndex] = useState(0)

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [displayedEntries])

  // Typewriter effect for new entries
  useEffect(() => {
    if (currentTypingIndex < entries.length) {
      const timer = setTimeout(() => {
        setDisplayedEntries(prev => [...prev, entries[currentTypingIndex]])
        setCurrentTypingIndex(prev => prev + 1)
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [entries, currentTypingIndex])

  // Reset when entries change significantly (new session)
  useEffect(() => {
    if (entries.length === 0) {
      setDisplayedEntries([])
      setCurrentTypingIndex(0)
    }
  }, [entries.length])

  const copyEntry = async (entry: TerminalEntry) => {
    await navigator.clipboard.writeText(entry.message)
    setCopiedId(entry.id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.95 }}
          animate={{ 
            opacity: 1, 
            y: 0, 
            scale: 1,
            height: minimized ? 48 : 'auto',
          }}
          exit={{ opacity: 0, y: 20, scale: 0.95 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className={`glass fixed bottom-4 left-4 z-40 w-[480px] rounded-xl shadow-2xl ${className}`}
          role="log"
          aria-label="System Terminal"
          aria-live="polite"
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-border/50 px-4 py-3">
            <div className="flex items-center gap-2">
              <Terminal className="h-4 w-4 text-primary" />
              <span className="font-mono text-sm font-medium text-foreground">
                System Terminal
              </span>
              {entries.length > 0 && (
                <span className="rounded-full bg-primary/20 px-2 py-0.5 text-xs font-mono text-primary">
                  {displayedEntries.length}/{entries.length}
                </span>
              )}
            </div>
            <div className="flex items-center gap-1">
              {onMinimize && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 text-muted-foreground hover:text-foreground"
                  onClick={onMinimize}
                >
                  {minimized ? <Maximize2 className="h-3 w-3" /> : <Minimize2 className="h-3 w-3" />}
                </Button>
              )}
              {onClose && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 text-muted-foreground hover:text-foreground"
                  onClick={onClose}
                >
                  <X className="h-3 w-3" />
                </Button>
              )}
            </div>
          </div>

          {/* Log Content */}
          <AnimatePresence>
            {!minimized && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                ref={scrollRef}
                className="max-h-64 overflow-y-auto p-4 font-mono text-sm"
              >
                {displayedEntries.length === 0 ? (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <span>Awaiting system messages...</span>
                    <span className="typewriter-cursor" />
                  </div>
                ) : (
                  <div className="space-y-2">
                    {displayedEntries.map((entry, index) => (
                      <motion.div
                        key={entry.id}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.02 }}
                        className="group flex items-start gap-2"
                      >
                        <span className="shrink-0 text-muted-foreground/50">
                          {entry.time}
                        </span>
                        <span className={`shrink-0 rounded px-1.5 py-0.5 text-xs ${levelBadgeColors[entry.level]}`}>
                          {entry.level.toUpperCase()}
                        </span>
                        <span className={`flex-1 ${levelColors[entry.level]}`}>
                          {entry.message}
                          {index === displayedEntries.length - 1 && currentTypingIndex < entries.length && (
                            <span className="typewriter-cursor" />
                          )}
                        </span>
                        <button
                          onClick={() => copyEntry(entry)}
                          className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                          aria-label="Copy log entry"
                        >
                          {copiedId === entry.id ? (
                            <Check className="h-3 w-3 text-safe" />
                          ) : (
                            <Copy className="h-3 w-3 text-muted-foreground hover:text-foreground" />
                          )}
                        </button>
                      </motion.div>
                    ))}
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Progress indicator */}
          {currentTypingIndex < entries.length && !minimized && (
            <div className="border-t border-border/50 px-4 py-2">
              <div className="h-1 overflow-hidden rounded-full bg-border">
                <motion.div
                  className="h-full bg-primary"
                  initial={{ width: '0%' }}
                  animate={{ width: `${(currentTypingIndex / entries.length) * 100}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// Hook for generating terminal entries
export function useTerminalLogger() {
  const [entries, setEntries] = useState<TerminalEntry[]>([])

  const addEntry = useCallback((message: string, level: TerminalEntry['level'] = 'info', meta?: Record<string, unknown>) => {
    const entry: TerminalEntry = {
      id: `log_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      time: new Date().toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
      }),
      level,
      message,
      meta,
    }
    setEntries(prev => [...prev, entry])
    return entry
  }, [])

  const clear = useCallback(() => setEntries([]), [])

  return { entries, addEntry, clear }
}
