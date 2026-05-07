import { useEffect } from 'react'

export interface KeyboardShortcut {
  keys: string[] // e.g., ['Control', 'k'] or ['Meta', 'k']
  label: string
  handler: () => void
}

/**
 * Hook for registering keyboard shortcuts
 * Supports:
 * - Cmd/Ctrl+K: Quick search/command palette
 * - Space: Open focus panel
 * - +/-: Zoom in/out
 * - F: Fit view
 * - Esc: Close modals
 */
export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[]) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Skip if typing in input/textarea
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        // Allow Esc to close
        if (event.key === 'Escape') {
          const handler = shortcuts.find(s => s.keys.includes('Escape'))?.handler
          if (handler) {
            event.preventDefault()
            handler()
          }
        }
        return
      }

      for (const shortcut of shortcuts) {
        const keys = shortcut.keys.map(k => k.toLowerCase())
        const pressedKeys = [
          event.ctrlKey || event.metaKey ? 'control' : null,
          event.shiftKey ? 'shift' : null,
          event.altKey ? 'alt' : null,
          event.key.toLowerCase(),
        ].filter(Boolean)

        // Check if this shortcut matches
        const isMatch = 
          keys.length === pressedKeys.length &&
          keys.every(k => {
            // Handle special cases
            if (k === 'control') return pressedKeys.includes('control')
            if (k === 'shift') return pressedKeys.includes('shift')
            if (k === 'alt') return pressedKeys.includes('alt')
            if (k === 'meta') return pressedKeys.includes('control')
            return pressedKeys.includes(k)
          })

        if (isMatch) {
          event.preventDefault()
          shortcut.handler()
          break
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [shortcuts])
}

/**
 * Utility to create common shortcuts
 */
export const createCommonShortcuts = (handlers: {
  onSearch?: () => void
  onOpenFocus?: () => void
  onZoomIn?: () => void
  onZoomOut?: () => void
  onFitView?: () => void
  onEscape?: () => void
}): KeyboardShortcut[] => {
  const shortcuts: KeyboardShortcut[] = []

  if (handlers.onSearch) {
    shortcuts.push(
      {
        keys: ['Control', 'k'],
        label: 'Quick Search',
        handler: handlers.onSearch,
      },
      {
        keys: ['Meta', 'k'],
        label: 'Quick Search (Mac)',
        handler: handlers.onSearch,
      }
    )
  }

  if (handlers.onOpenFocus) {
    shortcuts.push({
      keys: [' '],
      label: 'Open Focus Panel',
      handler: handlers.onOpenFocus,
    })
  }

  if (handlers.onZoomIn) {
    shortcuts.push(
      {
        keys: ['+'],
        label: 'Zoom In',
        handler: handlers.onZoomIn,
      },
      {
        keys: ['='],
        label: 'Zoom In (=)',
        handler: handlers.onZoomIn,
      }
    )
  }

  if (handlers.onZoomOut) {
    shortcuts.push({
      keys: ['-'],
      label: 'Zoom Out',
      handler: handlers.onZoomOut,
    })
  }

  if (handlers.onFitView) {
    shortcuts.push({
      keys: ['f'],
      label: 'Fit View',
      handler: handlers.onFitView,
    })
  }

  if (handlers.onEscape) {
    shortcuts.push({
      keys: ['Escape'],
      label: 'Close',
      handler: handlers.onEscape,
    })
  }

  return shortcuts
}
