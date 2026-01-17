import httpx
import os
import json
import asyncio
import re
import uuid
import random
from typing import Any, Dict, Optional, List
from datetime import datetime
from bson import ObjectId

from app.utils.logger import append_log, record_event
from app.models.schemas import DecisionNode, Risk, Alternative
from app.database.connection import get_database

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
                            return j["response"]
                    except Exception:
                        pass
                    return resp.text

            except Exception as e:
                append_log(f"ReasoningEngine: error on attempt {attempts}: {str(e)}")
                record_event(level="ERROR", action="reasoner.call.error", message="model call failed", details={"attempts": attempts, "error": str(e)})
                if attempts == max_attempts:
                    raise RuntimeError(f"Model call failed: {str(e)}")
                await asyncio.sleep(backoff)
                backoff *= 2
        return ""

    def _calculate_confidence_score(self, context_confidence: float, validation_retries: int = 0) -> float:
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

        return data

    async def generate_decision(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None, 
        job_id: Optional[str] = None,
        persona: str = "Skeptical Analyst",
        temperature: Optional[float] = None,
        validation_retries: int = 0  # Track retry count for confidence calculation
    ) -> DecisionNode:
        # Sample temperature if not provided (0.5-0.8 range per spec)
        if temperature is None:
            temperature = round(random.uniform(0.5, 0.8), 2)
        
        # Extract context_confidence from context dict
        context_confidence = 0.0
        if context and isinstance(context, dict):
            context_confidence = context.get('context_confidence', 0.0)
        
        # Get persona prompt
        persona_text = self._get_persona_prompt(persona)
        
        instruction = (
            f"You are a strategic simulation engine. {persona_text}\n\n"
            "RULES:\n"
            "1. Respond with valid JSON only.\n"
            "2. Schema: {id, title, summary, description, time_step (int), "
            "risks: [{description, severity (Low/Medium/High), likelihood}], "
            "alternatives: [{description, action_type}]}\n"
            "3. Every factual claim should ideally include [Source: cache:<id> | <url>] inline where used.\n"
            "4. If claim cannot be grounded, set speculative: true."
        )

        full_prompt = f"{instruction}\n\nSCENARIO: {prompt}\n\nCONTEXT: {json.dumps(context or {}, default=str)}\n\nJSON OUTPUT:"
        try:
            body = await self._call_model(full_prompt, temperature=temperature, timeout=300.0)
            append_log(f"ReasoningEngine: raw output len={len(body)}")
            record_event(level="INFO", action="reasoner.raw_output", message="raw output received", details={"job_id": job_id, "length": len(body)})

            # --- THE NUCLEAR CLEANER ---
            clean_text = body.replace("", "").replace("```", "").strip()

            # 1. Strip invisible control characters (ASCII 0-31) except newlines/tabs
            clean_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", clean_text)

            # 2. Remove comments //...
            clean_text = re.sub(r"//.*", "", clean_text)

            # 3. Remove trailing commas
            clean_text = re.sub(r',(\s*})', r"\1", clean_text)
            clean_text = re.sub(r',(\s*])', r"\1", clean_text)

            try:
                data = json.loads(clean_text)
            except Exception:
                # Fallback: try finding the outer braces if garbage exists around JSON
                start, end = clean_text.find("{"), clean_text.rfind("}")
                if start != -1 and end != -1:
                    data = json.loads(clean_text[start:end+1])
                else:
                    raise ValueError("No JSON found")

            clean_data = self._janitor_fix_data(data)
            
            # Calculate confidence_score from retrieval metrics and validation success
            confidence_score = self._calculate_confidence_score(
                context_confidence=context_confidence,
                validation_retries=validation_retries
            )
            clean_data['confidence_score'] = confidence_score
            
            node = DecisionNode(**clean_data)

            # persist model response for audit
            try:
                db = await get_database()
                await db["model_responses"].insert_one({
                    "job_id": job_id,
                    "raw": body,
                    "clean": clean_data,
                    "node": node.dict(),
                    "prompt": full_prompt[:1000],
                    "created_at": datetime.now().astimezone(),
                    "success": True,
                })
                record_event(level="INFO", action="reasoner.persist_success", message="parsed node persisted", details={"job_id": job_id, "node_id": node.id})
            except Exception as ex_db:
                append_log(f"ReasoningEngine: failed to persist parsed node to DB: {ex_db}")
                record_event(level="ERROR", action="reasoner.persist_failed", message="failed to persist parsed node", details={"job_id": job_id, "error": str(ex_db)})

            append_log(f"ReasoningEngine: Successfully built node {node.id}")
            record_event(level="INFO", action="reasoner.generate.success", message=f"node {node.id}", details={"job_id": job_id, "node_id": node.id, "confidence_score": confidence_score})
            return node

        except Exception as e:
            append_log(f"ReasoningEngine: Parsing failed: {str(e)}")
            record_event(level="ERROR", action="reasoner.generate.failure", message="parsing failed", details={"job_id": job_id, "error": str(e)})
            # persist failure for debugging
            try:
                db = await get_database()
                await db["model_responses"].insert_one({
                    "job_id": job_id,
                    "raw": body if "body" in locals() else None,
                    "error": str(e),
                    "prompt": full_prompt[:1000] if "full_prompt" in locals() else None,
                    "created_at": datetime.now().astimezone(),
                    "success": False,
                })
                record_event(level="INFO", action="reasoner.persist_failure", message="persisted failure", details={"job_id": job_id, "error": str(e)})
            except Exception as ex_db:
                append_log(f"ReasoningEngine: failed to persist model failure to DB: {ex_db}")

            # On error, use low confidence
            error_confidence = self._calculate_confidence_score(context_confidence=0.0, validation_retries=999)
            return DecisionNode(
                id=str(uuid.uuid4()),
                title="Simulation Error",
                summary="The AI returned invalid data.",
                description=f"System recovered from error: {str(e)}",
                risks=[Risk(description="System instability", severity="Low", likelihood="Low")],
                alternatives=[],
                confidence_score=error_confidence,
            )