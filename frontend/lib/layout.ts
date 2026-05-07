import dagre from 'dagre'
import type { Node, Edge } from '@xyflow/react'

export interface LayoutOptions {
  direction?: 'TB' | 'LR' | 'BT' | 'RL'
  nodeWidth?: number
  nodeHeight?: number
  rankSep?: number
  nodeSep?: number
}

const defaultOptions: LayoutOptions = {
  direction: 'TB',
  nodeWidth: 320,
  nodeHeight: 180,
  rankSep: 100,
  nodeSep: 50,
}

export function getLayoutedElements(
  nodes: Node[],
  edges: Edge[],
  options: LayoutOptions = {}
): { nodes: Node[]; edges: Edge[] } {
  const opts = { ...defaultOptions, ...options }
  
  const dagreGraph = new dagre.graphlib.Graph()
  dagreGraph.setDefaultEdgeLabel(() => ({}))
  dagreGraph.setGraph({ 
    rankdir: opts.direction,
    ranksep: opts.rankSep,
    nodesep: opts.nodeSep,
  })

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { 
      width: opts.nodeWidth, 
      height: opts.nodeHeight 
    })
  })

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target)
  })

  dagre.layout(dagreGraph)

  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id)
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - (opts.nodeWidth ?? 320) / 2,
        y: nodeWithPosition.y - (opts.nodeHeight ?? 180) / 2,
      },
    }
  })

  return { nodes: layoutedNodes, edges }
}
