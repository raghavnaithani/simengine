"""SimulationEngine: Manages time steps, branching rules, node lifecycle, and game-over detection.

Responsibilities:
- Automated world-building (optionally run N time steps initially)
- Branch handling: lock parents, orchestrate ContextBuilder + ReasoningEngine, append child node
- State snapshots: ensure upstream immutability and allow multiple independent branches
- Terminal state detection and game-over marking
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import uuid
import random

from backend.app.database.connection import get_database
from backend.app.engines.scraper import ContextBuilder
from backend.app.engines.reasoner import ReasoningEngine
from backend.app.models.schemas import DecisionNode
from backend.app.utils.logger import append_log, record_event


class SimulationEngine:
    """Manages simulation state, branching, and node lifecycle."""

    def __init__(self):
        self.context_builder = ContextBuilder()
        self.reasoning_engine = ReasoningEngine()

    async def build_initial_world(
        self,
        prompt: str,
        session_id: str,
        mode: str = "Analytical",
        persona: str = "Skeptical Analyst",
        num_steps: int = 3,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build initial simulation world with N time steps.

        Args:
            prompt: Initial scenario prompt
            session_id: Session identifier
            mode: Simulation mode ('Analytical' or 'Quick')
            persona: Persona for reasoning
            num_steps: Number of initial time steps to generate
            job_id: Optional job ID for logging

        Returns:
            Dict with 'root_node_id', 'node_ids' (list), and 'status'
        """
        record_event(
            level="INFO",
            action="simulation.build_world.start",
            message=f"Building initial world for session {session_id}",
            details={"prompt": prompt[:100], "num_steps": num_steps, "job_id": job_id}
        )

        try:
            # Step 1: Build knowledge base for initial prompt
            await self.context_builder.build_knowledge_base(prompt)

            # Step 2: Get context for reasoning
            context = await self.context_builder.get_context_for_reasoner(prompt, k=5)

            # Debug log to trace fallback logic
            record_event(
                level="DEBUG",
                action="simulation.build_world.trace",
                message="Checking fallback logic",
                details={"context": context}
            )

            # Step 3: Sample temperature per session (0.5-0.8 range per spec) - reuse for all nodes in this session
            temperature = round(random.uniform(0.5, 0.8), 2)

            # Step 4: Generate root node (time_step 0)
            root_node = await self.reasoning_engine.generate_decision(
                prompt,
                context,
                job_id=job_id,
                persona=persona,
                temperature=temperature
            )
            root_node.time_step = 0

            # Persist root node
            db = await get_database()
            nodes_coll = db['decision_nodes']
            await nodes_coll.insert_one(root_node.model_dump())

            node_ids = [root_node.id]
            current_node = root_node

            # Step 5: Generate subsequent time steps (if num_steps > 1)
            for step in range(1, num_steps):
                # Use parent summary for incremental simulation
                seed_prompt = current_node.summary
                context = await self.context_builder.get_context_for_reasoner(seed_prompt, k=5)

                # Generate next node (reuse same temperature for consistency within session)
                next_node = await self.reasoning_engine.generate_decision(
                    f"Time step {step}: Continue from {seed_prompt}",
                    context,
                    job_id=job_id,
                    persona=persona,
                    temperature=temperature
                )
                next_node.time_step = step

                # Persist node and create edge
                await nodes_coll.insert_one(next_node.model_dump())
                edges_coll = db['edges']
                await edges_coll.insert_one({
                    'from': current_node.id,
                    'to': next_node.id,
                    'action': f"Time step {step}",
                    'session_id': session_id,
                    'created_at': datetime.now(timezone.utc)
                })

                node_ids.append(next_node.id)
                current_node = next_node

                # Check for terminal state
                if await self._is_terminal_state(next_node):
                    record_event(
                        level="INFO",
                        action="simulation.terminal_detected",
                        message=f"Terminal state detected at step {step}",
                        details={"node_id": next_node.id, "job_id": job_id}
                    )
                    break

            # Update session metadata
            sessions_coll = db['sessions']
            await sessions_coll.update_one(
                {'session_id': session_id},
                {'$set': {
                    'root_node_id': root_node.id,
                    'current_node_id': current_node.id,
                    'num_nodes': len(node_ids),
                    'game_over': await self._is_terminal_state(current_node),
                    'updated_at': datetime.now(timezone.utc)
                }}
            )

            record_event(
                level="INFO",
                action="simulation.build_world.complete",
                message=f"Initial world built with {len(node_ids)} nodes",
                details={"session_id": session_id, "root_node_id": root_node.id, "job_id": job_id}
            )

            return {
                'root_node_id': root_node.id,
                'node_ids': node_ids,
                'current_node_id': current_node.id,
                'status': 'completed',
                'game_over': await self._is_terminal_state(current_node)
            }

        except Exception as e:
            record_event(
                level="ERROR",
                action="simulation.build_world.failed",
                message=f"Failed to build initial world: {str(e)}",
                details={"session_id": session_id, "job_id": job_id, "error": str(e)}
            )
            raise
    
    async def create_branch(
        self,
        parent_node_id: str,
        action: str,
        session_id: str,
        persona: str = "Optimistic Founder",
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a branch from a parent node.

        This implements incremental simulation: locks parent, uses only parent summary
        and context chunks (not full history) to generate child node.

        Args:
            parent_node_id: ID of parent node to branch from
            action: Action description for the branch
            session_id: Session identifier
            persona: Persona for reasoning
            job_id: Optional job ID for logging

        Returns:
            Dict with 'node_id', 'edge_id', and 'status'
        """
        record_event(
            level="INFO",
            action="simulation.branch.start",
            message=f"Creating branch from parent {parent_node_id}",
            details={"action": action, "session_id": session_id, "job_id": job_id}
        )

        try:
            db = await get_database()
            nodes_coll = db['decision_nodes']

            # Step 1: Lock parent node (fetch and verify it exists)
            parent = await nodes_coll.find_one({'id': parent_node_id})
            if not parent:
                raise ValueError(f"Parent node {parent_node_id} not found")

            # Mark parent as locked (immutable snapshot)
            await nodes_coll.update_one(
                {'id': parent_node_id},
                {'$set': {'locked': True, 'locked_at': datetime.now(timezone.utc)}}
            )

            # Step 2: Get parent summary for incremental simulation
            parent_summary = parent.get('summary', '')
            seed_prompt = f"Action: {action}\nContext: {parent_summary}"

            # Step 3: Get context for reasoning (only recent context, not full history)
            context = await self.context_builder.get_context_for_reasoner(seed_prompt, k=5)

            # Step 4: Sample temperature for this branch (0.5-0.8 range per spec)
            temperature = round(random.uniform(0.5, 0.8), 2)

            # Step 5: Generate child node
            child_node = await self.reasoning_engine.generate_decision(
                seed_prompt,
                context,
                job_id=job_id,
                persona=persona,
                temperature=temperature
            )

            # Set time step (increment from parent)
            parent_time_step = parent.get('time_step', 0)
            child_node.time_step = parent_time_step + 1

            # Step 6: Persist child node
            await nodes_coll.insert_one(child_node.model_dump())

            # Step 7: Create edge
            edges_coll = db['edges']
            edge_doc = {
                'from': parent_node_id,
                'to': child_node.id,
                'action': action,
                'session_id': session_id,
                'created_at': datetime.now(timezone.utc)
            }
            edge_result = await edges_coll.insert_one(edge_doc)

            # Step 7: Check for terminal state
            is_terminal = await self._is_terminal_state(child_node)
            if is_terminal:
                await nodes_coll.update_one(
                    {'id': child_node.id},
                    {'$set': {'game_over': True, 'game_over_reason': 'Terminal state detected'}}
                )

            # Step 8: Update session metadata
            sessions_coll = db['sessions']
            await sessions_coll.update_one(
                {'session_id': session_id},
                {'$set': {
                    'current_node_id': child_node.id,
                    'updated_at': datetime.now(timezone.utc),
                    'game_over': is_terminal
                }},
                upsert=False
            )

            record_event(
                level="INFO",
                action="simulation.branch.complete",
                message=f"Branch created: {child_node.id}",
                details={
                    "parent_id": parent_node_id,
                    "child_id": child_node.id,
                    "session_id": session_id,
                    "job_id": job_id,
                    "game_over": is_terminal
                }
            )

            return {
                'node_id': child_node.id,
                'edge_id': str(edge_result.inserted_id),
                'status': 'completed',
                'game_over': is_terminal
            }

        except Exception as e:
            record_event(
                level="ERROR",
                action="simulation.branch.failed",
                message=f"Failed to create branch: {str(e)}",
                details={"parent_node_id": parent_node_id, "job_id": job_id, "error": str(e)}
            )
            raise
    
    async def _is_terminal_state(self, node: DecisionNode) -> bool:
        """Detect if a node represents a terminal state (game over).
        
        Terminal states are detected by:
        - High severity risks with high likelihood
        - Specific keywords in description (e.g., "failure", "abandon", "terminate")
        - Confidence score below threshold
        
        Args:
            node: DecisionNode to check
            
        Returns:
            True if terminal state detected, False otherwise
        """
        # Check for high-risk scenarios
        high_risk_high_likelihood = any(
            r.severity in ['High', 'Critical'] and r.likelihood == 'High'
            for r in node.risks
        )
        
        # Check for terminal keywords in description
        terminal_keywords = ['failure', 'abandon', 'terminate', 'end', 'stop', 'game over']
        description_lower = node.description.lower()
        has_terminal_keyword = any(keyword in description_lower for keyword in terminal_keywords)
        
        # Check confidence threshold
        low_confidence = node.confidence_score < 0.3
        
        return high_risk_high_likelihood or (has_terminal_keyword and low_confidence)
    
    async def get_session_graph(self, session_id: str) -> Dict[str, Any]:
        """Get full graph for a session

        Args:
            session_id: Session identifier

        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        db = await get_database()
        nodes_coll = db['decision_nodes']
        edges_coll = db['edges']

        # Get all edges for this session
        edges = await edges_coll.find({'session_id': session_id}).to_list(length=1000)

        # Collect all node IDs from edges
        node_ids = set()
        for edge in edges:
            node_ids.add(edge.get('from'))
            node_ids.add(edge.get('to'))

        # Also get root node if session exists
        sessions_coll = db['sessions']
        session = await sessions_coll.find_one({'session_id': session_id})
        if session and session.get('root_node_id'):
            node_ids.add(session['root_node_id'])

        # Fetch all nodes
        nodes = []
        for node_id in node_ids:
            node = await nodes_coll.find_one({'id': node_id})
            if node:
                if '_id' in node:
                    node['_id'] = str(node['_id'])
                nodes.append(node)

        # Clean edge IDs
        for edge in edges:
            if '_id' in edge:
                edge['_id'] = str(edge['_id'])

        return {'nodes': nodes, 'edges': edges}
