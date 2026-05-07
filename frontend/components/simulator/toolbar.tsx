'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  ZoomIn, 
  ZoomOut, 
  Maximize2, 
  Grid3X3, 
  Download, 
  Image as ImageIcon,
  Menu,
  Plus,
  Settings,
  Undo2,
  Redo2,
  Keyboard,
  LayoutDashboard
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Kbd } from '@/components/ui/kbd'

interface ToolbarProps {
  onZoomIn: () => void
  onZoomOut: () => void
  onFitView: () => void
  onToggleGrid: () => void
  onToggleDashboard: () => void
  onExportPng?: () => void
  onExportJson?: () => void
  canExportPng?: boolean
  onToggleSessionSidebar: () => void
  onNewSession: () => void
  showGrid: boolean
  dashboardOpen: boolean
  zoom?: number
}

export function Toolbar({
  onZoomIn,
  onZoomOut,
  onFitView,
  onToggleGrid,
  onToggleDashboard,
  onExportPng,
  onExportJson,
  canExportPng = false,
  onToggleSessionSidebar,
  onNewSession,
  showGrid,
  dashboardOpen,
  zoom = 1,
}: ToolbarProps) {
  const [showKeyboardShortcuts, setShowKeyboardShortcuts] = useState(false)

  return (
    <TooltipProvider>
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="fixed left-1/2 top-4 z-30 flex -translate-x-1/2 items-center gap-1 rounded-xl border border-border bg-card/95 p-1.5 shadow-lg backdrop-blur-sm"
      >
        {/* Menu Button */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggleSessionSidebar}
              className="h-8 w-8"
              data-testid="toolbar-sessions-toggle"
            >
              <Menu className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Sessions</TooltipContent>
        </Tooltip>

        <div className="mx-1 h-6 w-px bg-border" />

        {/* New Session */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={onNewSession}
              className="h-8 w-8"
              data-testid="toolbar-new-session"
            >
              <Plus className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>New Session</TooltipContent>
        </Tooltip>

        <div className="mx-1 h-6 w-px bg-border" />

        {/* Zoom Controls */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={onZoomOut}
              className="h-8 w-8"
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            Zoom Out <Kbd>-</Kbd>
          </TooltipContent>
        </Tooltip>

        <span className="min-w-12 text-center text-xs font-mono text-muted-foreground">
          {Math.round(zoom * 100)}%
        </span>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={onZoomIn}
              className="h-8 w-8"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            Zoom In <Kbd>+</Kbd>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={onFitView}
              className="h-8 w-8"
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            Fit View <Kbd>F</Kbd>
          </TooltipContent>
        </Tooltip>

        <div className="mx-1 h-6 w-px bg-border" />

        {/* Grid Toggle */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={showGrid ? 'secondary' : 'ghost'}
              size="icon"
              onClick={onToggleGrid}
              className="h-8 w-8"
            >
              <Grid3X3 className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Toggle Grid</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={dashboardOpen ? 'secondary' : 'ghost'}
              size="icon"
              onClick={onToggleDashboard}
              className="h-8 w-8"
              data-testid="toolbar-dashboard-toggle"
            >
              <LayoutDashboard className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>WS5 Dashboard</TooltipContent>
        </Tooltip>

        <div className="mx-1 h-6 w-px bg-border" />

        {/* Export Menu */}
        <DropdownMenu>
          <Tooltip>
            <TooltipTrigger asChild>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                >
                  <Download className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
            </TooltipTrigger>
            <TooltipContent>Export</TooltipContent>
          </Tooltip>
          <DropdownMenuContent align="center">
            <DropdownMenuItem onClick={onExportPng} disabled={!canExportPng}>
              <ImageIcon className="mr-2 h-4 w-4" />
              {canExportPng ? 'Export as PNG' : 'Export as PNG (coming soon)'}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={onExportJson}>
              <Download className="mr-2 h-4 w-4" />
              Export as JSON
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <div className="mx-1 h-6 w-px bg-border" />

        {/* Keyboard Shortcuts */}
        <DropdownMenu open={showKeyboardShortcuts} onOpenChange={setShowKeyboardShortcuts}>
          <Tooltip>
            <TooltipTrigger asChild>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                >
                  <Keyboard className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
            </TooltipTrigger>
            <TooltipContent>Keyboard Shortcuts</TooltipContent>
          </Tooltip>
          <DropdownMenuContent align="center" className="w-56">
            <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
              Keyboard Shortcuts
            </div>
            <DropdownMenuSeparator />
            <div className="px-2 py-2 space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span>Quick Search</span>
                <Kbd>Ctrl+K</Kbd>
              </div>
              <div className="flex items-center justify-between">
                <span>Open Focus Panel</span>
                <Kbd>Space</Kbd>
              </div>
              <div className="flex items-center justify-between">
                <span>Zoom In</span>
                <Kbd>+</Kbd>
              </div>
              <div className="flex items-center justify-between">
                <span>Zoom Out</span>
                <Kbd>-</Kbd>
              </div>
              <div className="flex items-center justify-between">
                <span>Fit View</span>
                <Kbd>F</Kbd>
              </div>
              <div className="flex items-center justify-between">
                <span>Close/Deselect</span>
                <Kbd>Esc</Kbd>
              </div>
            </div>
          </DropdownMenuContent>
        </DropdownMenu>
      </motion.div>
    </TooltipProvider>
  )
}

// Floating action button for mobile/compact view
export function FloatingNewButton({ onClick }: { onClick: () => void }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onClick}
            className="fixed bottom-6 right-6 z-30 flex h-14 w-14 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-lg hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
          >
            <Plus className="h-6 w-6" />
          </motion.button>
        </TooltipTrigger>
        <TooltipContent side="left">New Simulation</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
