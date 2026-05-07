'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, ChevronDown, Settings2, Zap, Brain, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import type { StartSimulationRequest, SimulationMode } from '@/lib/types'

interface PromptModalProps {
  open: boolean
  defaultPrompt?: string
  onSubmit: (payload: StartSimulationRequest) => Promise<void>
  onClose: () => void
}

const PERSONAS = [
  { value: 'Skeptical Analyst', label: 'Skeptical Analyst', description: 'Questions assumptions, focuses on risks' },
  { value: 'Optimistic Founder', label: 'Optimistic Founder', description: 'Sees opportunities, growth-oriented' },
  { value: 'Cautious Regulator', label: 'Cautious Regulator', description: 'Compliance and risk mitigation focus' },
  { value: 'Aggressive Founder', label: 'Aggressive Founder', description: 'Speed-first, decisive market capture' },
  { value: 'Pessimistic Analyst', label: 'Pessimistic Analyst', description: 'Prioritizes failure modes and downside risk' },
]

export function PromptModal({ open, defaultPrompt = '', onSubmit, onClose }: PromptModalProps) {
  const [prompt, setPrompt] = useState(defaultPrompt)
  const [mode, setMode] = useState<SimulationMode>('Analytical')
  const [persona, setPersona] = useState(PERSONAS[0].value)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [simulateSteps, setSimulateSteps] = useState(3)
  const [temperature, setTemperature] = useState(0.6)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const isValid = prompt.length >= 10

  const handleSubmit = async () => {
    if (!isValid || isSubmitting) return
    
    setError(null)
    setIsSubmitting(true)
    
    try {
      await onSubmit({
        prompt,
        mode,
        persona,
        simulate_steps: simulateSteps,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start simulation')
      setIsSubmitting(false)
    }
  }

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
          aria-labelledby="prompt-modal-title"
        >
          {/* Backdrop with blur */}
          <motion.div 
            className="absolute inset-0 bg-background/80 backdrop-blur-sm"
            onClick={onClose}
          />
          
          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="glass relative z-10 w-full max-w-2xl rounded-2xl p-8 shadow-2xl"
          >
            {/* Header */}
            <div className="mb-6 text-center">
              <h1 
                id="prompt-modal-title" 
                className="text-3xl font-bold tracking-tight text-foreground"
              >
                Initialize Simulation
              </h1>
              <p className="mt-2 text-muted-foreground">
                Enter a strategic scenario to explore branching futures
              </p>
            </div>

            {/* Scenario Prompt */}
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="scenario-prompt" className="text-sm font-medium text-foreground">
                  Strategic Scenario
                </Label>
                <Textarea
                  id="scenario-prompt"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="What if I launch a vertical SaaS in the dental market?"
                  className="min-h-32 resize-none bg-background/50 text-lg placeholder:text-muted-foreground/50 focus:ring-2 focus:ring-primary"
                  aria-describedby="prompt-helper"
                />
                <p 
                  id="prompt-helper" 
                  className={`text-sm ${prompt.length > 0 && !isValid ? 'text-danger' : 'text-muted-foreground'}`}
                >
                  {prompt.length > 0 && !isValid 
                    ? `At least 10 characters required (${prompt.length}/10)`
                    : 'Describe a decision, market entry, or strategic question'}
                </p>
              </div>

              {/* Mode Selection */}
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setMode('Quick')}
                  className={`flex flex-1 items-center justify-center gap-2 rounded-lg border px-4 py-3 transition-all ${
                    mode === 'Quick' 
                      ? 'border-primary bg-primary/10 text-primary' 
                      : 'border-border bg-background/50 text-muted-foreground hover:border-primary/50'
                  }`}
                >
                  <Zap className="h-4 w-4" />
                  <span className="font-medium">Quick</span>
                  <span className="text-xs opacity-70">Fast exploration</span>
                </button>
                <button
                  type="button"
                  onClick={() => setMode('Analytical')}
                  className={`flex flex-1 items-center justify-center gap-2 rounded-lg border px-4 py-3 transition-all ${
                    mode === 'Analytical' 
                      ? 'border-primary bg-primary/10 text-primary' 
                      : 'border-border bg-background/50 text-muted-foreground hover:border-primary/50'
                  }`}
                >
                  <Brain className="h-4 w-4" />
                  <span className="font-medium">Analytical</span>
                  <span className="text-xs opacity-70">Deep RAG</span>
                </button>
              </div>

              {/* Persona Selection */}
              <div className="space-y-2">
                <Label htmlFor="persona-select" className="text-sm font-medium text-foreground">
                  Analysis Persona
                </Label>
                <Select value={persona} onValueChange={setPersona}>
                  <SelectTrigger id="persona-select" className="bg-background/50">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {PERSONAS.map((p) => (
                      <SelectItem key={p.value} value={p.value}>
                        <div className="flex flex-col">
                          <span>{p.label}</span>
                          <span className="text-xs text-muted-foreground">{p.description}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Advanced Settings */}
              <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
                <CollapsibleTrigger asChild>
                  <Button 
                    variant="ghost" 
                    className="flex w-full items-center justify-between text-muted-foreground hover:text-foreground"
                  >
                    <span className="flex items-center gap-2">
                      <Settings2 className="h-4 w-4" />
                      Advanced Settings
                    </span>
                    <ChevronDown className={`h-4 w-4 transition-transform ${advancedOpen ? 'rotate-180' : ''}`} />
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-4 pt-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Simulation Steps</Label>
                      <span className="text-sm font-mono text-muted-foreground">{simulateSteps}</span>
                    </div>
                    <Slider
                      value={[simulateSteps]}
                      onValueChange={([val]) => setSimulateSteps(val)}
                      min={1}
                      max={10}
                      step={1}
                      className="py-2"
                    />
                    <p className="text-xs text-muted-foreground">
                      Number of initial time steps to simulate
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Temperature</Label>
                      <span className="text-sm font-mono text-muted-foreground">{temperature.toFixed(2)}</span>
                    </div>
                    <Slider
                      value={[temperature]}
                      onValueChange={([val]) => setTemperature(val)}
                      min={0.1}
                      max={1.0}
                      step={0.05}
                      className="py-2"
                    />
                    <p className="text-xs text-muted-foreground">
                      Local setting only. Backend currently ignores temperature.
                    </p>
                  </div>
                </CollapsibleContent>
              </Collapsible>

              {/* Error Message */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="rounded-lg bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 p-4 space-y-2"
                >
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-semibold text-red-900 dark:text-red-200">Simulation Failed</p>
                      <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
                      <p className="text-xs text-red-600 dark:text-red-400 mt-2">
                        This may indicate the backend service is offline or Ollama timed out. 
                        Please check your backend logs or retry in a moment.
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Submit Button */}
              <Button
                onClick={handleSubmit}
                disabled={!isValid || isSubmitting}
                className="w-full bg-primary py-6 text-lg font-semibold text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                {isSubmitting ? (
                  <span className="flex items-center gap-2">
                    <span className="h-5 w-5 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                    Initializing...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <Play className="h-5 w-5" />
                    Initialize Simulation
                  </span>
                )}
              </Button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
