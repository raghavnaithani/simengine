'use client'

import React, { createContext, useContext, useState, useCallback } from 'react'
import { X, AlertCircle, CheckCircle, Info, AlertTriangle } from 'lucide-react'

export type ToastType = 'success' | 'error' | 'info' | 'warning'
export type ToastAction = 'retry' | 'view-logs' | 'dismiss' | 'view-details'

export interface Toast {
  id: string
  type: ToastType
  title: string
  message: string
  duration?: number // ms, 0 = persistent
  action?: {
    label: string
    action: ToastAction
    jobId?: string
  }
}

interface ToastContextType {
  toasts: Toast[]
  addToast: (toast: Omit<Toast, 'id'>) => string
  removeToast: (id: string) => void
  clearAll: () => void
}

const ToastContext = createContext<ToastContextType | null>(null)

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const addToast = useCallback((toast: Omit<Toast, 'id'>): string => {
    const id = `toast-${Date.now()}-${Math.random()}`
    const fullToast: Toast = { ...toast, id }

    setToasts(prev => [...prev, fullToast])

    // Auto-remove after duration
    if (toast.duration !== 0) {
      const delay = toast.duration || 5000
      setTimeout(() => removeToast(id), delay)
    }

    return id
  }, [])

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const clearAll = useCallback(() => {
    setToasts([])
  }, [])

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast, clearAll }}>
      {children}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </ToastContext.Provider>
  )
}

export function useToast() {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within ToastProvider')
  }
  return context
}

// Toast Container - renders all toasts
function ToastContainer({
  toasts,
  onRemove,
}: {
  toasts: Toast[]
  onRemove: (id: string) => void
}) {
  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-md">
      {toasts.map(toast => (
        <ToastItem key={toast.id} toast={toast} onRemove={onRemove} />
      ))}
    </div>
  )
}

// Individual Toast Item
function ToastItem({
  toast,
  onRemove,
}: {
  toast: Toast
  onRemove: (id: string) => void
}) {
  const bgClass = {
    success: 'bg-green-900/20 border-green-500/30',
    error: 'bg-red-900/20 border-red-500/30',
    warning: 'bg-amber-900/20 border-amber-500/30',
    info: 'bg-blue-900/20 border-blue-500/30',
  }[toast.type]

  const textClass = {
    success: 'text-green-400',
    error: 'text-red-400',
    warning: 'text-amber-400',
    info: 'text-blue-400',
  }[toast.type]

  const Icon = {
    success: CheckCircle,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
  }[toast.type]

  return (
    <div
      className={`${bgClass} border rounded-lg p-4 backdrop-blur-sm animate-in slide-in-from-right fade-in duration-300`}
    >
      <div className="flex items-start gap-3">
        <Icon className={`${textClass} w-5 h-5 flex-shrink-0 mt-0.5`} />
        <div className="flex-1 min-w-0">
          <h3 className={`${textClass} font-semibold text-sm`}>{toast.title}</h3>
          <p className="text-gray-300 text-sm mt-1">{toast.message}</p>
          {toast.action && (
            <button
              onClick={() => {
                // Handle action
                onRemove(toast.id)
              }}
              className="mt-2 text-xs font-semibold px-2 py-1 rounded bg-white/10 hover:bg-white/20 transition-colors"
            >
              {toast.action.label}
            </button>
          )}
        </div>
        <button
          onClick={() => onRemove(toast.id)}
          className="flex-shrink-0 text-gray-400 hover:text-gray-200 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
