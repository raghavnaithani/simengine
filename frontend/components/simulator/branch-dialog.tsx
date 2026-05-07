'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  GitBranch, 
  Sparkles, 
  X,
  ArrowRight,
  Loader2 
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { Alternative, DecisionNode } from '@/lib/types'

interface BranchDialogProps {
  open: boolean
  onClose: () => void
  onBranch: (alternative: Alternative, persona: string, customPrompt?: string) => Promise<void> | void
  parentNode: DecisionNode | null
}

const PERSONAS = [
  { value: 'skeptical-analyst', label: 'Skeptical Analyst' },
  { value: 'optimistic-founder', label: 'Optimistic Founder' },
  { value: 'cautious-regulator', label: 'Cautious Regulator' },
  { value: 'aggressive-founder', label: 'Aggressive Founder' },
  { value: 'pessimistic-analyst', label: 'Pessimistic Analyst' },
]

export function BranchDialog({ open, onClose, onBranch, parentNode }: BranchDialogProps) {
  const [mode, setMode] = useState<'preset' | 'custom'>('preset')
  const [selectedAlternative, setSelectedAlternative] = useState<string>('')
  const [customPrompt, setCustomPrompt] = useState('')
  const [persona, setPersona] = useState(PERSONAS[0].value)
  const [isSubmitting, setIsSubmitting] = useState(false)

  if (!parentNode) return null

  const handleSubmit = async () => {
    setIsSubmitting(true)
    
    let alternative: Alternative
    
    if (mode === 'custom') {
      // Create custom alternative from user prompt
      alternative = {
        id: `custom_${Date.now()}`,
        label: 'Custom Path',
        action_type: 'Custom',
        description: customPrompt,
      }
    } else {
      const found = parentNode.alternatives.find(a => a.id === selectedAlternative)
      if (!found) {
        setIsSubmitting(false)
        return
      }
      alternative = found
    }

    try {
      await onBranch(alternative, persona, mode === 'custom' ? customPrompt : undefined)
      // Reset state
      setSelectedAlternative('')
      setCustomPrompt('')
      setMode('preset')
      onClose()
    } finally {
      setIsSubmitting(false)
    }
  }

  const canSubmit = mode === 'custom' 
    ? customPrompt.length >= 10 
    : !!selectedAlternative

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
        >
          {/* Backdrop */}
          <motion.div 
            className="absolute inset-0 bg-background/80 backdrop-blur-sm"
            onClick={onClose}
          />
          
          {/* Dialog */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="glass relative z-10 w-full max-w-lg rounded-xl p-6 shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                  <GitBranch className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-foreground">
                    Branch from Node
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    {parentNode.title}
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="h-8 w-8 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Mode Selection */}
            <div className="mt-6 flex gap-2">
              <button
                onClick={() => setMode('preset')}
                className={`flex-1 rounded-lg border p-3 text-left transition-all ${
                  mode === 'preset'
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:border-primary/50'
                }`}
              >
                <span className="font-medium text-foreground">Preset Actions</span>
                <p className="mt-1 text-xs text-muted-foreground">
                  Choose from suggested alternatives
                </p>
              </button>
              <button
                onClick={() => setMode('custom')}
                className={`flex-1 rounded-lg border p-3 text-left transition-all ${
                  mode === 'custom'
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:border-primary/50'
                }`}
              >
                <span className="flex items-center gap-1 font-medium text-foreground">
                  <Sparkles className="h-3 w-3" />
                  Custom Path
                </span>
                <p className="mt-1 text-xs text-muted-foreground">
                  Define your own scenario
                </p>
              </button>
            </div>

            {/* Content based on mode */}
            <div className="mt-6 space-y-4">
              {mode === 'preset' ? (
                <div className="space-y-3">
                  <Label className="text-sm font-medium">Select an action</Label>
                  <RadioGroup
                    value={selectedAlternative}
                    onValueChange={setSelectedAlternative}
                    className="space-y-2"
                  >
                    {parentNode.alternatives.map((alt) => (
                      <Label
                        key={alt.id}
                        htmlFor={alt.id}
                        className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-all ${
                          selectedAlternative === alt.id
                            ? 'border-primary bg-primary/5'
                            : 'border-border hover:border-primary/50'
                        }`}
                      >
                        <RadioGroupItem value={alt.id} id={alt.id} className="mt-0.5" />
                        <div className="flex-1">
                          <span className="font-medium text-foreground">{getAlternativeLabel(alt)}</span>
                          {alt.description && (
                            <p className="mt-0.5 text-xs text-muted-foreground">
                              {alt.description}
                            </p>
                          )}
                          {alt.expected_outcome_summary && (
                            <p className="mt-1 text-xs text-primary/80">
                              Expected: {alt.expected_outcome_summary}
                            </p>
                          )}
                        </div>
                      </Label>
                    ))}
                  </RadioGroup>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="custom-prompt" className="text-sm font-medium">
                      Describe your custom path
                    </Label>
                    <Textarea
                      id="custom-prompt"
                      value={customPrompt}
                      onChange={(e) => setCustomPrompt(e.target.value)}
                      placeholder="What specific action or decision would you like to explore? Be specific about the approach, constraints, or variations you want to simulate..."
                      className="min-h-28 resize-none"
                    />
                    <p className="text-xs text-muted-foreground">
                      {customPrompt.length}/10 characters minimum
                    </p>
                  </div>

                </div>
              )}

              <div className="space-y-2">
                <Label className="text-sm font-medium">Analysis Persona</Label>
                <Select value={persona} onValueChange={setPersona}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {PERSONAS.map((p) => (
                      <SelectItem key={p.value} value={p.value}>
                        {p.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Actions */}
            <div className="mt-6 flex items-center justify-end gap-3">
              <Button variant="ghost" onClick={onClose}>
                Cancel
              </Button>
              <Button
                onClick={handleSubmit}
                disabled={!canSubmit || isSubmitting}
                className="gap-2"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Branching...
                  </>
                ) : (
                  <>
                    <ArrowRight className="h-4 w-4" />
                    Create Branch
                  </>
                )}
              </Button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function getAlternativeLabel(alt: Alternative): string {
  return alt.label || alt.action_type || alt.description || 'Option'
}
