import httpx
import os
import json
import asyncio
import re
import uuid
import random
from typing import Any, Dict, Optional
from datetime import datetime


from backend.app.utils.logger import append_log, record_event
from backend.app.utils.metrics import metrics, track_latency
from backend.app.models.schemas import DecisionNode, Risk, Alternative
from backend.app.database.connection import get_database

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "phi3")


class ReasoningEngine:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model
    
    def _get_persona_prompt(self, persona: str = "Skeptical Analyst") -> str:
        """Get persona-specific prompt text to inject into system prompt."""
        persona_templates = {
            "Skeptical Analyst": "You are a skeptical strategic analyst. You focus on identifying critical risks, potential failures, and worst-case scenarios. You question assumptions and demand evidence.",
            "Optimistic Founder": "You are an optimistic founder. You focus on opportunities, growth potential, and creative solutions. You see challenges as opportunities for innovation.",
            "Cautious Regulator": "You are a cautious regulator. You prioritize compliance, risk mitigation, and systematic evaluation. You require thorough documentation and evidence.",
            "Aggressive Founder": "You are an aggressive founder. You prioritize speed, market capture, and bold moves. You accept calculated risks for high rewards.",
            "Pessimistic Analyst": "You are a pessimistic analyst. You expect things to go wrong and identify failure modes early. You emphasize defensive strategies and risk avoidance.",
        }
        return persona_templates.get(persona, persona_templates["Skeptical Analyst"])

    async def _call_model(self, prompt: str, temperature: float = 0.7, timeout: float = 300.0) -> str:
        attempts = 0
        max_attempts = 3
        backoff = 1.0

        while attempts < max_attempts:
            attempts += 1
        with track_latency('llm.api_call'):
            attempts = 0
            max_attempts = 3
            backoff = 1.0
            
            while attempts < max_attempts:
                attempts += 1
                try:
                    append_log(f"ReasoningEngine: model call attempt {attempts}")
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        resp = await client.post(
                            OLLAMA_URL,
                            json={
                                "model": self.model,
                                "prompt": prompt,
                                "stream": False,
                                "format": "json",
                                "options": {
                                    "temperature": min(max(temperature, 0.0), 1.0),  # Clamp to 0.0-1.0
                                },
                            },
                        )
                        resp.raise_for_status()
                        try:
                            j = resp.json()
                            if isinstance(j, dict) and "response" in j:
                                metrics.record_metric(
                                    operation='llm.api_call',
                                    retry_count=attempts - 1,
                                    success=True
                                )
                                return j["response"]
                        except Exception:
                            pass
                        return resp.text
                
                except Exception as e:
                    append_log(f"ReasoningEngine: error on attempt {attempts}: {str(e)}")
                    record_event(level="ERROR", action="reasoner.call.error", message="model call failed", details={"attempts": attempts, "error": str(e)})
                    if attempts == max_attempts:
                        metrics.record_metric(
                            operation='llm.api_call',
                            retry_count=attempts - 1,
                            success=False,
                            details={'error': str(e)}
                        )
                        raise RuntimeError(f"Model call failed: {str(e)}")
                    await asyncio.sleep(backoff)
                    backoff *= 2
            return ""
        """Calculate confidence_score from retrieval metrics and validation success.
        
        Rules per project guide:
        - Base score: context_confidence (max similarity from retrieval, 0.0-1.0)
        - Penalty for validation retries: -0.1 per retry (max 3 retries)
        - Calibration: if context_confidence < 0.5, cap final score at 0.5
        - Minimum score: 0.0
        
        Args:
            context_confidence: Max similarity score from retrieval (0.0-1.0)
            validation_retries: Number of validation retries needed (0-3)
            
        Returns:
            Calibrated confidence score (0.0-1.0)
        """
        # Base score from retrieval similarity
        base_score = float(context_confidence)
        
        # Penalty for validation failures (each retry reduces confidence)
        retry_penalty = min(validation_retries * 0.1, 0.3)  # Max 0.3 penalty for 3+ retries
        adjusted_score = max(0.0, base_score - retry_penalty)
        
        # Calibration: if retrieval similarity was low, cap confidence
        if base_score < 0.5:
            adjusted_score = min(adjusted_score, 0.5)
        
        # Round to 2 decimal places
        return round(adjusted_score, 2)

    def _should_mark_speculative(
        self,
        confidence_score: float,
        context_confidence: float,
        has_citations: bool,
        validation_retries: int
    ) -> bool:
        """Determine if a DecisionNode should be marked as speculative.
        
        Mark as speculative if ANY of these conditions are true:
        1. Confidence score < 0.5 (low overall confidence)
        2. Context confidence < 0.8 (weak retrieval similarity per project guide section 5)
        3. No citations found AND context_confidence < 0.9 (claims without grounding)
        4. Multiple validation retries (>= 2) indicate unstable reasoning
        
        Per project guide section 9: "Low-confidence claims are flagged speculative"
        Per project guide section 5: "lacking a matching chunk similarity >= 0.8 must be flagged speculative=true"
        
        Args:
            confidence_score: Calculated confidence (0.0-1.0)
            context_confidence: Max retrieval similarity (0.0-1.0)
            has_citations: Whether node includes any source citations
            validation_retries: Number of retries needed for valid output
            
        Returns:
            True if node should be marked speculative
        """
        # Rule 1: Overall confidence too low
        if confidence_score < 0.5:
            record_event(
                level="INFO",
                action="speculative.low_confidence",
                message=f"Marking speculative: confidence {confidence_score} < 0.5"
            )
            return True
        
        # Rule 2: Retrieval similarity below threshold (project guide: 0.8)
        if context_confidence < 0.8:
            record_event(
                level="INFO",
                action="speculative.low_similarity",
                message=f"Marking speculative: context similarity {context_confidence} < 0.8"
            )
            return True
        
        # Rule 3: No citations and weak grounding
        if not has_citations and context_confidence < 0.9:
            record_event(
                level="INFO",
                action="speculative.no_citations",
                message=f"Marking speculative: no citations, similarity {context_confidence} < 0.9"
            )
            return True
        
        # Rule 4: Multiple retries indicate unstable reasoning
        if validation_retries >= 2:
            record_event(
                level="INFO",
                action="speculative.retries",
                message=f"Marking speculative: {validation_retries} validation retries"
            )
            return True
        
        return False

    def _janitor_fix_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggressively fixes messy AI output to satisfy Pydantic."""
        if not data.get("id"):
            data["id"] = str(uuid.uuid4())

        for field in ["title", "summary", "description"]:
            if not data.get(field):
                data[field] = "Content unavailable"

        try:
            val = data.get("time_step")
            if isinstance(val, float):
                data["time_step"] = int(val)
            elif val is None:
                data["time_step"] = 0
        except Exception:
            data["time_step"] = 0

        raw_risks = data.get("risks")
        if not raw_risks or not isinstance(raw_risks, list):
            data["risks"] = []

        valid_risks = []
        for r in data.get("risks", []):
            if isinstance(r, dict):
                r["description"] = r.get("description") or r.get("title") or "Generic Risk"
                r["severity"] = r.get("severity") if r.get("severity") in ["Low", "Medium", "High", "Critical"] else "Medium"
                r["likelihood"] = r.get("likelihood") if r.get("likelihood") in ["Low", "Medium", "High"] else "Medium"
                valid_risks.append(r)

        if not valid_risks:
            data["risks"] = [{"description": "General uncertainty.", "severity": "Low", "likelihood": "Low"}]
        else:
            data["risks"] = valid_risks

        raw_alts = data.get("alternatives")
        if not raw_alts or not isinstance(raw_alts, list):
            data["alternatives"] = []

        valid_alts = []
        for a in data.get("alternatives", []):
            if isinstance(a, dict):
                a["description"] = a.get("description") or "Explore option"
                a["action_type"] = a.get("action_type") or "Wait"
                valid_alts.append(a)
        data["alternatives"] = valid_alts

        # FIX 2: Map extracted citations to source_citations field (Citation Enforcement)
        # Per project guide Section 5: source_citations must be populated from extracted [Source:] tokens
        if 'citations' in data and isinstance(data['citations'], list) and data['citations']:
            # Format citations as "Source:<content>" per schemas.py validator requirements (line 110)
            formatted_citations = [f"Source:{c}" if not c.startswith('Source:') else c 
                                  for c in data['citations']]
            data['source_citations'] = formatted_citations
            append_log(f"ReasoningEngine: Mapped {len(formatted_citations)} citations to source_citations field")
        elif 'source_citations' not in data:
            data['source_citations'] = []

        return data

    def _extract_and_clean_json(self, raw_text: str) -> Dict[str, Any]:
        """Aggressively clean and extract JSON from LLM output with multiple fallback strategies, including citation enforcement."""
        if not raw_text or not isinstance(raw_text, str):
            raise ValueError("Empty or invalid input text")

        clean_text = raw_text.strip()

        # Strategy 1: Remove markdown code blocks
        clean_text = re.sub(r'```json\s*', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'```\s*', '', clean_text)
        clean_text = clean_text.strip()

        # Strategy 2: Remove ALL control characters (ASCII + Unicode) except newlines/tabs
        clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean_text)
        clean_text = re.sub(r'[\u0080-\u009f]', '', clean_text)
        clean_text = re.sub(r'[\u200b-\u200f\u2028-\u202f]', '', clean_text)

        # Strategy 3: Remove comments (both // and /* */ style)
        clean_text = re.sub(r'//.*?$', '', clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r'/\*.*?\*/', '', clean_text, flags=re.DOTALL)

        # Strategy 4: Remove trailing commas before } or ]
        while True:
            new_text = re.sub(r',\s*([}\]])', r'\1', clean_text)
            if new_text == clean_text:
                break
            clean_text = new_text

        # Extract and remove citation tokens before parsing JSON
        # Format per project guide Section 7: [Source: cache:<id> | <url>]
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        citations = re.findall(citation_pattern, raw_text)
        if citations:
            append_log(f"ReasoningEngine: Detected citations (Source format): {citations}")

        # Debug logging for citation validation
        append_log(f"Validating citations: {citations}")
        for citation in citations:
            # Validate format: should be cache:<id> or cache:<id> | url
            if not (citation.startswith('cache:') or 'http' in citation):
                append_log(f"Invalid citation detected: {citation}")
                raise ValueError(f"Invalid citation format: {citation}. Expected [Source: cache:<id> | <url>]")

        # Remove citation tokens from text
        raw_text = re.sub(citation_pattern, '', raw_text)  # Remove citation tokens from text

        # Ensure proper JSON formatting for large inputs
        if raw_text.startswith('{') and raw_text.endswith('}'):
            try:
                json.loads(raw_text)  # Validate JSON structure
            except json.JSONDecodeError as e:
                append_log(f"ReasoningEngine: Invalid JSON structure detected: {e}")
                raise ValueError("Invalid JSON structure for large input")

        # Simplified handling for large inputs
        if len(raw_text) > 1000:  # Arbitrary threshold for large input
            append_log("ReasoningEngine: Received large input for processing.")
            append_log(f"ReasoningEngine: First 500 characters of input: {raw_text[:500]}")
            append_log(f"ReasoningEngine: Last 500 characters of input: {raw_text[-500:]}")
            try:
                json.loads(raw_text)  # Validate JSON structure directly
            except json.JSONDecodeError as e:
                append_log(f"ReasoningEngine: Large input validation failed: {e}")
                raise ValueError("Large input validation failed")

        # Try parsing with progressively more aggressive extraction
        parse_attempts = []

        # Attempt 1: Direct parse
        parse_attempts.append(("direct", lambda t: json.loads(t)))

        # Attempt 2: Extract first complete JSON object (find matching braces)
        def extract_balanced_braces(text):
            start = text.find('{')
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
            return None

        parse_attempts.append(("balanced_braces", lambda t: json.loads(extract_balanced_braces(t) or "{}")))

        # Attempt 3: Regex extract (find first { ... } block)
        def regex_extract(text):
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                return match.group(0)
            return None

        parse_attempts.append(("regex_extract", lambda t: json.loads(regex_extract(t) or "{}")))

        # Attempt 4: Find outer braces (simple fallback)
        def outer_braces(text):
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return text[start:end + 1]
            return None

        parse_attempts.append(("outer_braces", lambda t: json.loads(outer_braces(t) or "{}")))

        # Try each parsing strategy
        last_error = None
        for strategy_name, parse_func in parse_attempts:
            try:
                data = parse_func(clean_text)
                if isinstance(data, dict) and data:  # Ensure we got a non-empty dict
                    data['citations'] = citations  # Include citations in the parsed JSON
                    append_log(f"ReasoningEngine: JSON parsed successfully using strategy: {strategy_name}")
                    return data
            except Exception as e:
                last_error = e
                continue

        # If all attempts failed, try one more time with the original text (maybe it's already clean)
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict):
                data['citations'] = citations  # Include citations in the parsed JSON
                append_log(f"ReasoningEngine: JSON parsed from raw text")
                return data
        except Exception:
            pass

        raise ValueError(f"JSON parsing failed after all strategies. Last error: {last_error}")

    def _validate_citation(self, citation: str) -> bool:
        """Validate a citation token. Example logic: ensure it matches a predefined schema."""
        # Placeholder validation logic; replace with actual rules
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', citation))

    def _generate_retry_instruction(self, error_message: str, attempt_number: int) -> str:
        """Generate targeted retry instruction based on validation failure type.
        
        Provides specific guidance to LLM on what needs to be fixed.
        """
        error_lower = error_message.lower()
        base = f"\n\nAttempt {attempt_number} failed."
        
        # Risk-specific guidance (High severity missing)
        if 'high severity' in error_lower:
            return (
                f"{base} Your output is missing High severity risks. "
                "You MUST identify at least one High severity failure mode, challenge, or threat. "
                "Include why it's a serious concern and cite sources where applicable. "
                "Retry with at least one High severity risk in the risks array."
            )
        
        # Citation-specific guidance
        elif 'citation' in error_lower or '[source:' in error_lower.lower():
            return (
                f"{base} Your output is missing required citations. "
                "Every external claim must include [Source: cache:<id> | <url>] inline. "
                "Review your text and add citations for all factual assertions."
            )
        
        # Confidence-specific guidance
        elif 'confidence' in error_lower:
            return (
                f"{base} Your output has invalid confidence_score. "
                "Confidence must be between 0.0 and 1.0. "
                "Set to 0.5 if uncertain, 0.8+ if well-supported by evidence."
            )
        
        # Generic JSON parse error
        else:
            return (
                f"{base} JSON formatting error: {error_message[:100]}. "
                "Ensure valid JSON with: all quotes matched, all commas present, "
                "no trailing commas, all braces closed, no control characters."
            )

    async def generate_decision(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None, 
        job_id: Optional[str] = None,
        persona: str = "Skeptical Analyst",
        temperature: Optional[float] = None,
        validation_retries: int = 0  # Track retry count for confidence calculation
    ) -> DecisionNode:
        with track_latency('llm.generate'):
            # Sample temperature if not provided (0.5-0.8 range per spec)
            if temperature is None:
                temperature = round(random.uniform(0.5, 0.8), 2)
            
            # Extract context_confidence from context dict
            context_confidence = 0.0
            if context and isinstance(context, dict):
                context_confidence = context.get('context_confidence', 0.0)
            
            # Get persona prompt
            persona_text = self._get_persona_prompt(persona)
        
        # Enhanced instruction with stronger JSON emphasis
        instruction = (
            f"You are a strategic simulation engine. {persona_text}\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST respond with ONLY valid JSON. No explanatory text, no markdown, no code blocks.\n"
            "2. Start your response with {{ and end with }}. Nothing else.\n"
            "3. Schema: {id, title, summary, description, time_step (int), "
            "risks: [{description, severity (Low/Medium/High/Critical), likelihood (Low/Medium/High)}], "
            "alternatives: [{description, action_type}]}\n"
            "4. Every factual claim should ideally include [Source: cache:<id> | <url>] inline where used.\n"
            "5. If claim cannot be grounded, set speculative: true.\n"
            "6. Ensure all strings are properly quoted, all commas are correct, no trailing commas."
        )

        full_prompt = f"{instruction}\n\nSCENARIO: {prompt}\n\nCONTEXT: {json.dumps(context or {}, default=str)}\n\nJSON OUTPUT:"
        
        # Retry logic for JSON parsing with progressively clearer instructions
        max_json_retries = 3
        body = None
        last_parse_error = None
        
        for json_attempt in range(max_json_retries):
            try:
                # Call model (with retry prompt if this is a retry)
                if json_attempt > 0:
                    retry_instruction = self._generate_retry_instruction(
                        error_message=last_parse_error,
                        attempt_number=json_attempt
                    )
                    retry_prompt = f"{full_prompt}{retry_instruction}\n\nJSON OUTPUT:"
                else:
                    retry_prompt = full_prompt
                
                body = await self._call_model(retry_prompt, temperature=temperature, timeout=300.0)
                append_log(f"ReasoningEngine: raw output len={len(body)} (attempt {json_attempt + 1})")
                record_event(level="INFO", action="reasoner.raw_output", message="raw output received", details={"job_id": job_id, "length": len(body), "attempt": json_attempt + 1})

                # Extract and clean JSON
                data = self._extract_and_clean_json(body)
                
                # If we got here, JSON parsing succeeded
                clean_data = self._janitor_fix_data(data)
                
                # Calculate confidence_score from retrieval metrics and validation success
                confidence_score = self._calculate_confidence_score(
                    context_confidence=context_confidence,
                    validation_retries=validation_retries
                )
                clean_data['confidence_score'] = confidence_score
                
                # FIX 3: Enforce citation requirement per project guide Section 9
                # "ReasoningEngine must fail a node if required citations are missing for claims that exceed confidence threshold"
                has_citations = bool(clean_data.get('source_citations', []))
                if confidence_score >= 0.5 and not has_citations:
                    # High-confidence node without citations - trigger adversarial retry
                    if json_attempt < max_json_retries - 1:
                        retry_instruction = (
                            f"\n\nCRITICAL CITATION REQUIREMENT: Your claim has HIGH confidence ({confidence_score:.2f}) "
                            f"but NO CITATIONS. Per project rules:\n"
                            f"  1. You MUST include [Source: cache:<id> | <url>] for EVERY external claim\n"
                            f"  2. If you cannot ground a claim, set speculative: true and include [Source: speculative]\n"
                            f"  3. Re-examine each claim in description and alternatives\n"
                            f"Retry now with proper citations for all factual claims."
                        )
                        retry_prompt = f"{full_prompt}{retry_instruction}\n\nJSON OUTPUT:"
                        append_log(
                            f"ReasoningEngine: High-confidence node ({confidence_score:.2f}) missing citations. "
                            f"Triggering adversarial retry (attempt {json_attempt + 2}/{max_json_retries})"
                        )
                        record_event(
                            level="WARN",
                            action="reasoner.citation_retry",
                            message="Adversarial retry triggered for missing citations",
                            details={"confidence_score": confidence_score, "attempt": json_attempt + 1}
                        )
                        last_parse_error = "Missing required citations for high-confidence claim"
                        continue  # Retry with new prompt
                    else:
                        # All retries exhausted, fail the node
                        error_msg = (
                            f"Node rejected: High-confidence claim (score={confidence_score:.2f}) "
                            f"missing required citations. Failed after {max_json_retries} attempts. "
                            f"Add [Source: cache:<id> | <url>] for factual claims or set speculative: true."
                        )
                        append_log(f"ReasoningEngine: {error_msg}")
                        record_event(
                            level="ERROR",
                            action="reasoner.citation_enforcement_failed",
                            message="Node rejected: missing required citations",
                            details={"confidence_score": confidence_score, "job_id": job_id}
                        )
                        raise ValueError(error_msg)
                
                # Determine if node should be marked speculative
                should_be_speculative = self._should_mark_speculative(
                    confidence_score=confidence_score,
                    context_confidence=context_confidence,
                    has_citations=has_citations,
                    validation_retries=json_attempt  # Use current retry count
                )
                
                # Apply speculative flag if needed
                if should_be_speculative and not clean_data.get('speculative', False):
                    clean_data['speculative'] = True
                    record_event(
                        level="INFO",
                        action="speculative.flag_applied",
                        message=f"Node marked speculative",
                        details={
                            "confidence_score": confidence_score,
                            "context_confidence": context_confidence,
                            "has_citations": has_citations,
                            "validation_retries": json_attempt
                        }
                    )
                
                node = DecisionNode(**clean_data)

                # persist model response for audit
                try:
                    db = await get_database()
                    await db["model_responses"].insert_one({
                        "job_id": job_id,
                        "raw": body,
                        "clean": clean_data,
                        "node": node.model_dump(),
                        "prompt": full_prompt[:1000],
                        "created_at": datetime.now().astimezone(),
                        "success": True,
                    })
                    record_event(level="INFO", action="reasoner.persist_success", message="parsed node persisted", details={"job_id": job_id, "node_id": node.id})
                except Exception as ex_db:
                    append_log(f"ReasoningEngine: failed to persist parsed node to DB: {ex_db}")
                    record_event(level="ERROR", action="reasoner.persist_failed", message="failed to persist parsed node", details={"job_id": job_id, "error": str(ex_db)})

                append_log(f"ReasoningEngine: Successfully built node {node.id} (speculative={node.speculative})")
                record_event(level="INFO", action="reasoner.generate.success", message=f"node {node.id}", details={"job_id": job_id, "node_id": node.id, "confidence_score": confidence_score, "speculative": node.speculative})
                
                # Log generation metrics
                metrics.record_metric(
                    operation='llm.generate',
                    success=True,
                    details={
                        'confidence_score': confidence_score,
                        'persona': persona,
                        'temperature': temperature,
                        'retry_count': json_attempt,
                        'speculative': node.speculative
                    }
                )
                
                return node
                
            except Exception as e:
                last_parse_error = str(e)
                append_log(f"ReasoningEngine: JSON parsing attempt {json_attempt + 1} failed: {last_parse_error}")
                
                if json_attempt == max_json_retries - 1:
                    # Final attempt failed, log and return error node
                    append_log(f"ReasoningEngine: All JSON parsing attempts failed. Last error: {last_parse_error}")
                    record_event(level="ERROR", action="reasoner.generate.failure", message="parsing failed after all retries", details={"job_id": job_id, "error": last_parse_error, "attempts": max_json_retries})
                    
                    # persist failure for debugging
                    try:
                        db = await get_database()
                        await db["model_responses"].insert_one({
                            "job_id": job_id,
                            "raw": body if body else None,
                            "error": last_parse_error,
                            "prompt": full_prompt[:1000] if "full_prompt" in locals() else None,
                            "created_at": datetime.now().astimezone(),
                            "success": False,
                        })
                        record_event(level="INFO", action="reasoner.persist_failure", message="persisted failure", details={"job_id": job_id, "error": last_parse_error})
                    except Exception as ex_db:
                        append_log(f"ReasoningEngine: failed to persist model failure to DB: {ex_db}")

                    # On error, use low confidence
                    error_confidence = self._calculate_confidence_score(context_confidence=0.0, validation_retries=999)
                    
                    # Log failure metrics
                    metrics.record_metric(
                        operation='llm.generate',
                        success=False,
                        details={
                            'error': last_parse_error,
                            'retry_count': max_json_retries,
                            'persona': persona,
                            'temperature': temperature
                        }
                    )
                    
                    return DecisionNode(
                        id=str(uuid.uuid4()),
                        title="Simulation Error",
                        summary="The AI returned invalid data after multiple retry attempts.",
                        description=f"System recovered from error: {last_parse_error}",
                        risks=[Risk(description="System instability", severity="Low", likelihood="Low")],
                        alternatives=[],
                        confidence_score=error_confidence,
                    )
                else:
                    # Wait a bit before retry (exponential backoff)
                    await asyncio.sleep(0.5 * (json_attempt + 1))
                    continue
        
        # Should never reach here, but just in case
        error_confidence = self._calculate_confidence_score(context_confidence=0.0, validation_retries=999)
        return DecisionNode(
            id=str(uuid.uuid4()),
            title="Simulation Error",
            summary="The AI returned invalid data.",
            description="System recovered from error: Unexpected failure in JSON parsing retry loop",
            risks=[Risk(description="System instability", severity="Low", likelihood="Low")],
            alternatives=[],
            confidence_score=error_confidence,
        )