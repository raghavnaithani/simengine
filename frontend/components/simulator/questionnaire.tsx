'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ArrowRight, 
  ArrowLeft, 
  Check, 
  Loader2,
  MessageSquare,
  Sparkles 
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Checkbox } from '@/components/ui/checkbox'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'
import type { ClarifyingQuestion, ScenarioConfig, TimelineHorizon } from '@/lib/types'

interface QuestionnaireProps {
  prompt: string
  onComplete: (config: ScenarioConfig) => void
  onBack: () => void
}

// Entity extraction from prompt - extract key information user already provided
interface PromptEntities {
  target_users: string | null
  target_market: string | null
  product_type: string | null
  key_features: string[]
  mentioned_constraints: string[]
  business_stage: string
  problem_statement: string
}

function extractPromptEntities(prompt: string): PromptEntities {
  const promptLower = prompt.toLowerCase()
  
  // Extract what was mentioned
  const targetMatch = prompt.match(/(?:for|target|serve|help)\s+([^.,;]+?)(?:\s+(?:to|who|that|in)|\.|,|$)/i)
  const featureMatch = prompt.match(/(?:with|using|featuring|includes?|provides?)\s+([^.,;]+?)(?:\.|,|$)/i)
  const problemMatch = prompt.match(/(?:problem|challenge|help|solve|address)\s+([^.,;]+?)(?:\.|,|$)/i)
  
  return {
    target_users: targetMatch ? targetMatch[1].trim() : null,
    target_market: extractMarketReference(prompt),
    product_type: extractProductType(prompt),
    key_features: extractKeywords(prompt, ['api', 'integration', 'automation', 'analytics', 'dashboard', 'platform', 'tool', 'service']),
    mentioned_constraints: extractConstraints(prompt),
    business_stage: inferBusinessStage(prompt),
    problem_statement: problemMatch ? problemMatch[1].trim() : prompt.slice(0, 100),
  }
}

function extractMarketReference(prompt: string): string | null {
  const marketPatterns = [
    /(?:market|industry|space|sector)\s+(?:is|for|in)\s+([^.,;]+)/i,
    /\b(healthcare|fintech|real estate|retail|education|logistics|insurance|saas|b2b|b2c|enterprise)\b/i,
    /(?:targeting|serving|aimed at)\s+([^.,;]+?)(?:\.|,|$)/i,
  ]
  
  for (const pattern of marketPatterns) {
    const match = prompt.match(pattern)
    if (match && match[1]) return match[1].trim()
  }
  return null
}

function extractProductType(prompt: string): string | null {
  const typePatterns = [
    /(?:build|create|develop|launch)\s+(?:a|an|the)\s+([^.,;]+?)(?:\s+(?:for|to|that)|\.|,|$)/i,
    /\b(software|platform|app|tool|service|solution|saas|mobile app|web app|desktop app)\b/i,
  ]
  
  for (const pattern of typePatterns) {
    const match = prompt.match(pattern)
    if (match && match[1]) return match[1].trim()
  }
  return null
}

function extractKeywords(prompt: string, keywords: string[]): string[] {
  return keywords.filter(kw => prompt.toLowerCase().includes(kw))
}

function extractConstraints(prompt: string): string[] {
  const constraintPatterns = [
    /(?:no|limited|tight|budget|funding|resources|expertise|experience|team|runway)\s+([^.,;]+)/gi,
  ]
  
  const constraints: string[] = []
  for (const pattern of constraintPatterns) {
    const matches = prompt.matchAll(pattern)
    for (const match of matches) {
      if (match[1]) constraints.push(match[1].trim())
    }
  }
  return constraints.slice(0, 3)
}

function inferBusinessStage(prompt: string): string {
  const promptLower = prompt.toLowerCase()
  if (promptLower.includes('mvp') || promptLower.includes('prototype') || promptLower.includes('validate')) return 'validation'
  if (promptLower.includes('launch') || promptLower.includes('go to market') || promptLower.includes('initial users')) return 'launch'
  if (promptLower.includes('scale') || promptLower.includes('grow') || promptLower.includes('expand')) return 'scaling'
  if (promptLower.includes('optimize') || promptLower.includes('improve') || promptLower.includes('mature')) return 'optimization'
  return 'idea/exploration'
}

// Generate targeted clarifying questions based on what's MISSING from the prompt
function generateSmartQuestions(prompt: string): ClarifyingQuestion[] {
  const entities = extractPromptEntities(prompt)
  const questions: ClarifyingQuestion[] = []
  
  // Question 1: Timeline (always needed for decision tree)
  questions.push({
    id: 'q_timeline',
    question: 'What is your decision timeline?',
    type: 'single',
    options: ['3 months (Quick validation)', '6 months (Balanced exploration)', '12 months (Deep strategic planning)'],
  })
  
  // Question 2: Clarify target users if not mentioned
  if (!entities.target_users) {
    questions.push({
      id: 'q_target_users',
      question: `Who specifically are you targeting with "${entities.product_type || 'your solution'}"?`,
      type: 'text',
    })
  } else {
    questions.push({
      id: 'q_target_clarity',
      question: `You mentioned targeting ${entities.target_users}. What's the primary pain point you're solving for them?`,
      type: 'text',
    })
  }
  
  // Question 3: Market validation status
  questions.push({
    id: 'q_validation',
    question: `What's your current validation status for the ${entities.business_stage} stage?`,
    type: 'single',
    options: [
      'Haven\'t validated yet',
      'Preliminary customer feedback received',
      'Paying customers or strong commitment signals',
      'Proven demand with traction'
    ],
  })
  
  // Question 4: Key uncertainty based on what's missing
  const uncertaintyOptions = getUncertaintyOptions(entities)
  if (uncertaintyOptions.length > 0) {
    questions.push({
      id: 'q_uncertainty',
      question: `What's your biggest uncertainty right now?`,
      type: 'single',
      options: uncertaintyOptions,
    })
  }
  
  // Question 5: Go-to-market / distribution approach
  questions.push({
    id: 'q_distribution',
    question: `How are you planning to reach and acquire your ${entities.target_users || 'customers'}?`,
    type: 'text',
  })
  
  return questions
}

function getUncertaintyOptions(entities: PromptEntities): string[] {
  const options = []
  
  if (!entities.target_market) {
    options.push('Which market segment to target')
  }
  if (!entities.target_users) {
    options.push('Who exactly will use this')
  }
  if (entities.mentioned_constraints.length === 0) {
    options.push('Resource constraints / funding')
  }
  if (entities.key_features.length === 0) {
    options.push('Core features and MVP scope')
  }
  
  // Always include these general options
  options.push('Competitive landscape / differentiation', 'Team/execution capability')
  
  return options.slice(0, 4)
}

export function Questionnaire({ prompt, onComplete, onBack }: QuestionnaireProps) {
  const [questions, setQuestions] = useState<ClarifyingQuestion[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [answers, setAnswers] = useState<Record<string, string | string[] | number>>({})
  const [isGenerating, setIsGenerating] = useState(true)

  useEffect(() => {
    // Generate smart questions based on prompt analysis
    const timer = setTimeout(() => {
      const generatedQuestions = generateSmartQuestions(prompt)
      console.log('[v0] Generated smart questions:', generatedQuestions.length, 'questions')
      console.log('[v0] First question:', generatedQuestions[0]?.question)
      setQuestions(generatedQuestions)
      setIsGenerating(false)
    }, 1500)
    return () => clearTimeout(timer)
  }, [prompt])

  const currentQuestion = questions[currentIndex]
  const isLastQuestion = currentIndex === questions.length - 1
  const canProceed = currentQuestion?.type === 'text' || answers[currentQuestion?.id]

  const handleAnswer = (value: string | string[] | number) => {
    setAnswers(prev => ({
      ...prev,
      [currentQuestion.id]: value,
    }))
  }

  const handleNext = () => {
    if (isLastQuestion) {
      // Determine timeline from answer
      const timelineAnswer = answers['q_timeline'] as string || ''
      let timeline: TimelineHorizon = '6-months'
      let simulateSteps = 6
      
      if (timelineAnswer.includes('3 months')) {
        timeline = '3-months'
        simulateSteps = 4
      } else if (timelineAnswer.includes('12 months')) {
        timeline = '12-months'
        simulateSteps = 10
      }

      // Build config with clarifications
      const config: ScenarioConfig = {
        prompt,
        mode: 'Analytical',
        persona: 'Skeptical Analyst',
        timeline,
        simulate_steps: simulateSteps,
        temperature: 0.6,
        clarifications: questions.map(q => ({
          ...q,
          answer: answers[q.id],
        })),
      }
      onComplete(config)
    } else {
      setCurrentIndex(prev => prev + 1)
    }
  }

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1)
    } else {
      onBack()
    }
  }

  if (isGenerating) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-background"
      >
        <div className="flex flex-col items-center gap-4">
          <div className="relative">
            <div className="absolute inset-0 animate-ping rounded-full bg-primary/20" />
            <div className="relative flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
              <Sparkles className="h-8 w-8 text-primary animate-pulse" />
            </div>
          </div>
          <p className="text-lg font-medium text-foreground">Analyzing your scenario...</p>
          <p className="text-sm text-muted-foreground">Generating personalization questions</p>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex flex-col bg-background"
    >
      {/* Header */}
      <div className="border-b border-border bg-card/50 px-6 py-4">
        <div className="mx-auto max-w-2xl">
          <div className="flex items-center gap-3">
            <MessageSquare className="h-5 w-5 text-primary" />
            <h1 className="text-lg font-semibold text-foreground">Scenario Refinement</h1>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Help us understand your context for more accurate simulations
          </p>
        </div>
      </div>

      {/* Progress */}
      <div className="border-b border-border bg-card/30 px-6 py-3">
        <div className="mx-auto max-w-2xl">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>Question {currentIndex + 1} of {questions.length}</span>
            <span>{Math.round(((currentIndex + 1) / questions.length) * 100)}% complete</span>
          </div>
          <div className="mt-2 h-1 rounded-full bg-muted">
            <motion.div
              className="h-1 rounded-full bg-primary"
              initial={{ width: 0 }}
              animate={{ width: `${((currentIndex + 1) / questions.length) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>
      </div>

      {/* Question Content */}
      <div className="flex-1 overflow-auto px-6 py-8">
        <div className="mx-auto max-w-2xl">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentQuestion.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              <div className="flex items-start gap-3">
                <div className="flex-1">
                  <div className="mb-2 flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-amber-500" />
                    <span className="text-xs font-medium text-amber-600 dark:text-amber-400">AI-ANALYZED QUESTION</span>
                  </div>
                  <h2 className="text-2xl font-semibold text-foreground">
                    {currentQuestion.question}
                  </h2>
                </div>
              </div>

              {currentQuestion.type === 'single' && currentQuestion.options && (
                <RadioGroup
                  value={answers[currentQuestion.id] as string || ''}
                  onValueChange={handleAnswer}
                  className="space-y-3"
                >
                  {currentQuestion.options.map((option, i) => (
                    <motion.div
                      key={option}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.05 }}
                    >
                      <Label
                        htmlFor={`option-${i}`}
                        className={`flex cursor-pointer items-center gap-3 rounded-lg border p-4 transition-all ${
                          answers[currentQuestion.id] === option
                            ? 'border-primary bg-primary/5'
                            : 'border-border hover:border-primary/50'
                        }`}
                      >
                        <RadioGroupItem value={option} id={`option-${i}`} />
                        <span className="text-foreground">{option}</span>
                      </Label>
                    </motion.div>
                  ))}
                </RadioGroup>
              )}

              {currentQuestion.type === 'multiple' && currentQuestion.options && (
                <div className="space-y-3">
                  {currentQuestion.options.map((option, i) => {
                    const selectedOptions = (answers[currentQuestion.id] as string[]) || []
                    const isSelected = selectedOptions.includes(option)
                    
                    return (
                      <motion.div
                        key={option}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.05 }}
                      >
                        <Label
                          htmlFor={`checkbox-${i}`}
                          className={`flex cursor-pointer items-center gap-3 rounded-lg border p-4 transition-all ${
                            isSelected
                              ? 'border-primary bg-primary/5'
                              : 'border-border hover:border-primary/50'
                          }`}
                        >
                          <Checkbox
                            id={`checkbox-${i}`}
                            checked={isSelected}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                handleAnswer([...selectedOptions, option])
                              } else {
                                handleAnswer(selectedOptions.filter(o => o !== option))
                              }
                            }}
                          />
                          <span className="text-foreground">{option}</span>
                        </Label>
                      </motion.div>
                    )
                  })}
                </div>
              )}

              {currentQuestion.type === 'text' && (
                <Textarea
                  value={(answers[currentQuestion.id] as string) || ''}
                  onChange={(e) => handleAnswer(e.target.value)}
                  placeholder="Share any additional context that might help..."
                  className="min-h-32 resize-none"
                />
              )}

              {currentQuestion.type === 'scale' && (
                <div className="space-y-4">
                  <Slider
                    value={[(answers[currentQuestion.id] as number) || 5]}
                    onValueChange={([val]) => handleAnswer(val)}
                    min={1}
                    max={10}
                    step={1}
                    className="py-4"
                  />
                  <div className="flex justify-between text-sm text-muted-foreground">
                    <span>1 - Low</span>
                    <span className="font-mono text-foreground">
                      {(answers[currentQuestion.id] as number) || 5}
                    </span>
                    <span>10 - High</span>
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* Navigation */}
      <div className="border-t border-border bg-card/50 px-6 py-4">
        <div className="mx-auto flex max-w-2xl items-center justify-between">
          <Button
            variant="ghost"
            onClick={handlePrevious}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            {currentIndex === 0 ? 'Edit Prompt' : 'Previous'}
          </Button>

          <Button
            onClick={handleNext}
            disabled={!canProceed && currentQuestion.type !== 'text'}
            className="gap-2"
          >
            {isLastQuestion ? (
              <>
                <Check className="h-4 w-4" />
                Start Simulation
              </>
            ) : (
              <>
                Next
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </Button>
        </div>
      </div>
    </motion.div>
  )
}
