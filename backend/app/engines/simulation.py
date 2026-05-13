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
import hashlib

from backend.app.database.connection import get_database
from backend.app.engines.scraper import ContextBuilder
from backend.app.engines.reasoner import ReasoningEngine
from backend.app.models.schemas import DecisionNode
from backend.app.utils.logger import append_log, record_event
from backend.app.utils.quality import annotate_node_quality


class SimulationEngine:
    """Manages simulation state, branching, and node lifecycle."""

    def __init__(self):
        self.context_builder = ContextBuilder()
        self.reasoning_engine = ReasoningEngine()

    def _compute_content_hash(self, title: str, summary: str, description: str) -> str:
        """Compute SHA-256 hash of node content for idempotency detection.
        
        Per project guide: "branch creation must be safe under retries 
        (check for duplicate child nodes by content hash)."
        
        Args:
            title: Node title
            summary: Node summary
            description: Node description
            
        Returns:
            Hex SHA-256 hash of concatenated content
        """
        content = f"{title}||{summary}||{description}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    async def _recent_session_nodes(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        db = await get_database()
        nodes_coll = db['decision_nodes']
        return await nodes_coll.find({'session_id': session_id}).sort('created_at', -1).limit(limit).to_list(length=limit)

    def _normalize_branch_action(self, branch_action: str) -> str:
        cleaned = ' '.join(str(branch_action or '').split()).strip()
        if not cleaned:
            return 'Explore a contrasting alternative path'
        return cleaned

    def _branch_context_query(self, prompt: str, parent_summary: str, branch_action: str, step: int, branch_index: int) -> str:
        return (
            f"Scenario prompt: {prompt}\n"
            f"Parent summary: {parent_summary}\n"
            f"Branch action: {branch_action}\n"
            f"Branch depth: step {step}, option {branch_index}\n"
            f"Focus on a downstream outcome that is distinct from other branch options."
        )

    def _branch_prompt(self, prompt: str, parent_summary: str, branch_action: str, step: int, branch_index: int) -> str:
        return (
            f"Branch path {step}.{branch_index}: pursue '{branch_action}' as a distinct future.\n"
            f"Parent summary: {parent_summary}\n"
            f"Original scenario: {prompt}\n"
            f"Generate a branch-specific decision node with materially different implications."
        )

    def _branch_temperature(self, base_temperature: float, branch_action: str, step: int, branch_index: int) -> float:
        branch_key = f"{branch_action}|{step}|{branch_index}"
        branch_hash = hashlib.sha256(branch_key.encode('utf-8')).hexdigest()
        offset_bucket = int(branch_hash[:6], 16) % 11 - 5
        adjusted = round(base_temperature + (offset_bucket * 0.01), 2)
        return max(0.45, min(0.85, adjusted))

    def _branch_actions_for_node(self, node: DecisionNode, max_actions: int = 2) -> List[str]:
        """Derive a small set of divergent branch actions from a node's alternatives.

        If the model does not provide alternatives, fall back to a conservative
        pair of generic branching directions so the graph still fans out.
        """
        actions: List[str] = []
        seen_actions = set()
        alternatives = getattr(node, 'alternatives', None) or []

        for alternative in alternatives:
            if len(actions) >= max_actions:
                break
            if isinstance(alternative, dict):
                description = str(alternative.get('description') or '').strip()
                action_type = str(alternative.get('action_type') or '').strip()
            else:
                description = str(getattr(alternative, 'description', '') or '').strip()
                action_type = str(getattr(alternative, 'action_type', '') or '').strip()

            if description and action_type:
                candidate = f"{action_type}: {description}"
            elif description:
                candidate = description
            elif action_type:
                candidate = action_type
            else:
                candidate = ''

            candidate = self._normalize_branch_action(candidate)
            candidate_key = candidate.lower()
            if candidate and candidate_key not in seen_actions:
                actions.append(candidate)
                seen_actions.add(candidate_key)

        if not actions:
            actions = [
                'Pursue the most aggressive growth path',
                'Pursue the most conservative validation path',
            ]

        if len(actions) == 1:
            actions.append('Explore a contrasting alternative path')

        return actions[:max_actions]

    async def build_initial_world(
        self,
        prompt: str,
        session_id: str,
        mode: str = "Analytical",
        persona: str = "Skeptical Analyst",
        num_steps: int = 3,
        job_id: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Build initial simulation world with N time steps.

        Args:
            prompt: Initial scenario prompt
            session_id: Session identifier
            mode: Simulation mode ('Analytical' or 'Quick')
            persona: Persona for reasoning
            num_steps: Number of initial time steps to generate
            job_id: Optional job ID for logging
            seed: Optional random seed for reproducible temperature sampling

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

            # Step 3: Set random seed for reproducibility (if provided)
            if seed is not None:
                random.seed(seed)
                record_event(
                    level="INFO",
                    action="simulation.seed_set",
                    message=f"Random seed set to {seed} for reproducibility",
                    details={"seed": seed, "session_id": session_id}
                )

            # Fast mode keeps initial response under typical UX SLA.
            mode_normalized = str(mode or "").strip().lower()
            if mode_normalized == "quick":
                num_steps = min(max(1, int(num_steps)), 2)

            # Step 4: Sample temperature per session (0.5-0.8 range per spec) - reuse for all nodes in this session
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
            annotate_node_quality(root_node, await self._recent_session_nodes(session_id))

            # Persist root node
            db = await get_database()
            nodes_coll = db['decision_nodes']
            root_node_doc = root_node.model_dump()
            root_node_doc['session_id'] = session_id
            if job_id:
                root_node_doc['job_id'] = job_id
            await nodes_coll.insert_one(root_node_doc)

            node_ids = [root_node.id]
            current_node = root_node

            # Step 5: Expand into branching scenarios breadth-first.
            # Each depth level fans out from every active frontier node so the
            # graph becomes a tree instead of a single linear chain.
            frontier = [root_node]
            current_leaf = root_node
            terminal_detected = await self._is_terminal_state(root_node)
            breadth_by_step: Dict[int, int] = {0: 1}

            for step in range(1, num_steps):
                next_frontier: List[DecisionNode] = []

                for parent_node in frontier:
                    record_event(
                        level="INFO",
                        action="simulation.branching.parent_node",
                        message=f"Processing parent node at step {step}",
                        details={"parent_id": parent_node.id, "frontier_size": len(frontier), "step": step, "num_steps": num_steps}
                    )
                    parent_summary = parent_node.summary
                    context = await self.context_builder.get_context_for_reasoner(parent_summary, k=5)
                    branch_actions = self._branch_actions_for_node(parent_node, max_actions=2)
                    record_event(
                        level="INFO",
                        action="simulation.branching.actions",
                        message=f"Branch actions for parent: {len(branch_actions)} actions",
                        details={"actions": branch_actions[:2]}
                    )

                    for branch_index, branch_action in enumerate(branch_actions, start=1):
                        normalized_action = self._normalize_branch_action(branch_action)
                        branch_temperature = self._branch_temperature(temperature, normalized_action, step, branch_index)
                        record_event(
                            level="DEBUG",
                            action="simulation.branching.creating_child",
                            message=f"Creating child node {branch_index}/{len(branch_actions)} for step {step}",
                            details={"branch_action": normalized_action, "step": step, "parent_id": parent_node.id}
                        )
                        branch_context_query = self._branch_context_query(
                            prompt=prompt,
                            parent_summary=parent_summary,
                            branch_action=normalized_action,
                            step=step,
                            branch_index=branch_index,
                        )
                        branch_prompt = self._branch_prompt(
                            prompt=prompt,
                            parent_summary=parent_summary,
                            branch_action=normalized_action,
                            step=step,
                            branch_index=branch_index,
                        )

                        context = await self.context_builder.get_context_for_reasoner(branch_context_query, k=5)

                        child_node = await self.reasoning_engine.generate_decision(
                            branch_prompt,
                            context,
                            job_id=job_id,
                            persona=persona,
                            temperature=branch_temperature
                        )
                        record_event(
                            level="DEBUG",
                            action="simulation.branching.child_created",
                            message=f"Child node created at step {step}",
                            details={"child_id": child_node.id, "step": step, "title": child_node.title}
                        )
                        child_node.time_step = step
                        annotate_node_quality(child_node, await self._recent_session_nodes(session_id))

                        content_hash = self._compute_content_hash(
                            child_node.title,
                            child_node.summary,
                            child_node.description
                        )
                        existing_child = await nodes_coll.find_one({
                            'parent_id': parent_node.id,
                            'content_hash': content_hash,
                            'session_id': session_id
                        })

                        if existing_child:
                            record_event(
                                level="INFO",
                                action="simulation.branching.duplicate_prevented",
                                message="Duplicate child prevented during initial world expansion",
                                details={
                                    "parent_id": parent_node.id,
                                    "existing_child_id": existing_child['id'],
                                    "content_hash": content_hash,
                                    "step": step,
                                    "branch_index": branch_index,
                                }
                            )
                            continue

                        child_node_doc = child_node.model_dump()
                        child_node_doc['session_id'] = session_id
                        child_node_doc['parent_id'] = parent_node.id
                        child_node_doc['branch_action'] = normalized_action
                        child_node_doc['branch_step'] = step
                        child_node_doc['branch_index'] = branch_index
                        child_node_doc['content_hash'] = content_hash
                        if job_id:
                            child_node_doc['job_id'] = job_id
                        await nodes_coll.insert_one(child_node_doc)

                        edges_coll = db['edges']
                        await edges_coll.insert_one({
                            'from': parent_node.id,
                            'to': child_node.id,
                            'action': normalized_action,
                            'session_id': session_id,
                            'created_at': datetime.now(timezone.utc)
                        })

                        node_ids.append(child_node.id)
                        next_frontier.append(child_node)
                        current_leaf = child_node
                        breadth_by_step[step] = breadth_by_step.get(step, 0) + 1

                        if await self._is_terminal_state(child_node):
                            terminal_detected = True
                            record_event(
                                level="INFO",
                                action="simulation.terminal_detected",
                                message=f"Terminal state detected at step {step}",
                                details={"node_id": child_node.id, "job_id": job_id}
                            )

                if not next_frontier:
                    record_event(
                        level="WARN",
                        action="simulation.branching.empty_frontier",
                        message=f"Empty frontier at step {step}, terminating branching",
                        details={"step": step, "num_steps": num_steps}
                    )
                    break

                frontier = next_frontier

            non_root_breadths = [count for depth, count in breadth_by_step.items() if depth > 0]
            if num_steps > 1 and non_root_breadths and max(non_root_breadths) <= 1:
                record_event(
                    level="ERROR",
                    action="simulation.branching.shallow_graph_rejected",
                    message="Initial world rejected because branching never exceeded one node per timestep",
                    details={"session_id": session_id, "breadth_by_step": breadth_by_step, "job_id": job_id}
                )
                raise ValueError("Shallow branching graph rejected: no non-root step achieved branching breadth")

            # Update session metadata
            sessions_coll = db['sessions']
            await sessions_coll.update_one(
                {'session_id': session_id},
                {'$set': {
                    'root_node_id': root_node.id,
                    'current_node_id': current_leaf.id,
                    'num_nodes': len(node_ids),
                    'branch_breadth_by_step': breadth_by_step,
                    'max_branch_breadth': max(non_root_breadths) if non_root_breadths else 1,
                    'game_over': terminal_detected,
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
                'current_node_id': current_leaf.id,
                'status': 'completed',
                'game_over': terminal_detected
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
        job_id: Optional[str] = None,
        seed: Optional[int] = None
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
            seed: Optional random seed for reproducible temperature sampling

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
            normalized_action = self._normalize_branch_action(action)
            seed_prompt = f"Action: {normalized_action}\nContext: {parent_summary}"

            # Step 3: Get context for reasoning (only recent context, not full history)
            context_query = self._branch_context_query(
                prompt=seed_prompt,
                parent_summary=parent_summary,
                branch_action=normalized_action,
                step=parent.get('time_step', 0) + 1,
                branch_index=1,
            )
            context = await self.context_builder.get_context_for_reasoner(context_query, k=5)

            # Step 4: Set random seed for reproducibility (if provided)
            if seed is not None:
                random.seed(seed)
                record_event(
                    level="INFO",
                    action="simulation.seed_set",
                    message=f"Random seed set to {seed} for branch reproducibility",
                    details={"seed": seed, "parent_node_id": parent_node_id}
                )

            # Step 5: Sample temperature for this branch (0.5-0.8 range per spec)
            temperature = round(random.uniform(0.5, 0.8), 2)
            temperature = self._branch_temperature(temperature, normalized_action, parent.get('time_step', 0) + 1, 1)

            # Step 6: Generate child node
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
            annotate_node_quality(child_node, await self._recent_session_nodes(session_id))

            # Step 7: IDEMPOTENCY CHECK - Prevent duplicate child nodes (project guide requirement)
            # Compute content hash to detect if this exact node was already created
            content_hash = self._compute_content_hash(
                child_node.title,
                child_node.summary,
                child_node.description
            )
            
            # Check if a child with same content hash already exists under this parent
            existing_child = await nodes_coll.find_one({
                'parent_id': parent_node_id,
                'content_hash': content_hash,
                'session_id': session_id
            })
            
            if existing_child:
                record_event(
                    level="INFO",
                    action="simulation.branch.duplicate_prevented",
                    message=f"Duplicate child prevented via content hash",
                    details={
                        "parent_id": parent_node_id,
                        "existing_child_id": existing_child['id'],
                        "content_hash": content_hash,
                        "session_id": session_id
                    }
                )
                
                # Return the existing child (idempotent behavior)
                return {
                    'node_id': existing_child['id'],
                    'edge_id': existing_child.get('edge_id', 'N/A'),
                    'status': 'already_exists',
                    'game_over': existing_child.get('game_over', False)
                }
            
            # Step 8: Persist child node with content hash for future idempotency checks
            child_node_doc = child_node.model_dump()
            child_node_doc['session_id'] = session_id
            child_node_doc['parent_id'] = parent_node_id  # Track parent for idempotency queries
            child_node_doc['branch_action'] = normalized_action
            child_node_doc['content_hash'] = content_hash  # Store hash for duplicate detection
            if job_id:
                child_node_doc['job_id'] = job_id
            await nodes_coll.insert_one(child_node_doc)

            # Step 9: Create edge
            edges_coll = db['edges']
            edge_doc = {
                'from': parent_node_id,
                'to': child_node.id,
                'action': normalized_action,
                'session_id': session_id,
                'created_at': datetime.now(timezone.utc)
            }
            edge_result = await edges_coll.insert_one(edge_doc)
            
            # Store edge ID in node doc for idempotency lookup
            await nodes_coll.update_one(
                {'id': child_node.id},
                {'$set': {'edge_id': str(edge_result.inserted_id)}}
            )

            # Step 10: Check for terminal state
            is_terminal = await self._is_terminal_state(child_node)
            if is_terminal:
                await nodes_coll.update_one(
                    {'id': child_node.id},
                    {'$set': {'game_over': True, 'game_over_reason': 'Terminal state detected'}}
                )

            # Step 11: Update session metadata
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

        # Fetch all nodes for the session so the full branching structure is visible,
        # including nodes that may not be reachable from a shallow edge walk.
        node_docs = await nodes_coll.find({'session_id': session_id}).to_list(length=2000)
        nodes = []
        for node in sorted(
            node_docs,
            key=lambda item: (
                int(item.get('time_step', 0) or 0),
                item.get('created_at') or datetime.min.replace(tzinfo=timezone.utc),
                item.get('id') or ''
            )
        ):
            if '_id' in node:
                node['_id'] = str(node['_id'])
            nodes.append(node)

        # Clean edge IDs
        for edge in edges:
            if '_id' in edge:
                edge['_id'] = str(edge['_id'])

        return {'nodes': nodes, 'edges': edges}
