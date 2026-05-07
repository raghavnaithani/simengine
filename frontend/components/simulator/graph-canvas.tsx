'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type OnConnect,
  BackgroundVariant,
  Panel,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { motion, AnimatePresence } from 'framer-motion'
import { DecisionNodeComponent, GhostNodeComponent } from './decision-node'
import { getLayoutedElements } from '@/lib/layout'
import { useSessionStore } from '@/lib/store'
import type { DecisionNode, Alternative, GhostNode } from '@/lib/types'

function dedupeById<T extends { id: string }>(items: T[]): T[] {
  const map = new Map<string, T>()
  for (const item of items) {
    map.set(item.id, item)
  }
  return Array.from(map.values())
}

const nodeTypes = {
  decision: DecisionNodeComponent,
  ghost: GhostNodeComponent,
}

interface GraphCanvasProps {
  onNodeSelect: (nodeId: string) => void
  onNodeBranch?: (nodeId: string, alternative: Alternative) => void
  showGrid?: boolean
  showMiniMap?: boolean
}

export function GraphCanvas({ 
  onNodeSelect,
  onNodeBranch,
  showGrid = true,
  showMiniMap = true 
}: GraphCanvasProps) {
  const { 
    nodes: storeNodes, 
    edges: storeEdges, 
    ghostNodes,
    selectedNodeId,
    setSelectedNodeId,
    addNode,
    addEdge,
    addGhostNode,
    removeGhostNode,
    updateGhostNode,
    setNodes: setStoreNodes,
    setEdges: setStoreEdges,
  } = useSessionStore()

  // Convert store nodes to ReactFlow nodes
  const initialNodes = useMemo(() => {
    const decisionNodes: Node[] = storeNodes.map((node) => ({
      id: node.id,
      type: 'decision',
      position: { x: 0, y: 0 },
      data: {
        ...node,
        onBranch: handleBranch,
        onFocus: handleNodeFocus,
        isSelected: node.id === selectedNodeId,
      },
    }))

    const ghosts: Node[] = ghostNodes.map((ghost) => ({
      id: ghost.temp_id,
      type: 'ghost',
      position: { x: 0, y: 0 },
      data: {
        status: ghost.status,
        action: ghost.action,
        error: ghost.error,
        onCancel: () => removeGhostNode(ghost.temp_id),
      },
    }))

    return dedupeById([...decisionNodes, ...ghosts])
  }, [storeNodes, ghostNodes, selectedNodeId])

  // Convert store edges to ReactFlow edges
  const initialEdges = useMemo(() => {
    const regularEdges: Edge[] = storeEdges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'smoothstep',
      style: { stroke: '#334155', strokeWidth: 2 },
      animated: false,
    }))

    // Add edges for ghost nodes
    const ghostEdges: Edge[] = ghostNodes.map((ghost) => ({
      id: `edge_ghost_${ghost.temp_id}`,
      source: ghost.parent_id,
      target: ghost.temp_id,
      type: 'smoothstep',
      style: { stroke: '#3B82F6', strokeWidth: 2, strokeDasharray: '5,5' },
      animated: true,
    }))

    return dedupeById([...regularEdges, ...ghostEdges])
  }, [storeEdges, ghostNodes])

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  // Apply layout when nodes change
  useEffect(() => {
    if (initialNodes.length === 0) return

    const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
      initialNodes,
      initialEdges
    )
    setNodes(layoutedNodes)
    setEdges(layoutedEdges)
  }, [initialNodes, initialEdges, setNodes, setEdges])

  function handleNodeFocus(nodeId: string) {
    setSelectedNodeId(nodeId)
    onNodeSelect(nodeId)
  }

  function handleBranch(nodeId: string, alternative: Alternative) {
    // All branch handling now goes through parent simulator via onNodeBranch callback
    // This ensures proper job polling, telemetry, and error handling
    if (onNodeBranch) {
      onNodeBranch(nodeId, alternative)
    }
  }

  const onConnect: OnConnect = useCallback((params) => {
    // Connections are not user-draggable in this app
  }, [])

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    if (node.type === 'decision') {
      handleNodeFocus(node.id)
    }
  }, [])

  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        nodesDraggable={false}
        nodesConnectable={false}
        panOnDrag={true}
        zoomOnScroll={true}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2}
        className="bg-background"
      >
        {showGrid && (
          <Background 
            variant={BackgroundVariant.Dots} 
            gap={20} 
            size={1}
            color="#334155"
          />
        )}
        
        <Controls 
          showInteractive={false}
          className="!bg-card !border-border !rounded-lg"
        />
        
        {showMiniMap && (
          <MiniMap
            nodeColor={(node) => {
              if (node.type === 'ghost') return '#3B82F6'
              return '#1E293B'
            }}
            maskColor="rgba(11, 17, 32, 0.8)"
            className="!bg-card !border-border !rounded-lg"
          />
        )}

        {/* Empty state */}
        {nodes.length === 0 && (
          <Panel position="top-center" className="mt-20">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center"
            >
              <p className="text-lg text-muted-foreground">
                No nodes yet. Start a simulation to begin.
              </p>
            </motion.div>
          </Panel>
        )}
      </ReactFlow>
    </div>
  )
}
