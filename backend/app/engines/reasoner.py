import httpx
import os
import json
import asyncio
import re
import uuid
import random
from typing import Any, Dict, Optional, List
from datetime import datetime


from backend.app.utils.logger import append_log, record_event
from backend.app.utils.metrics import metrics, track_latency
from backend.app.utils.token_budget import log_truncation_diagnostics, estimate_token_count
from backend.app.engines.prompt_builder import PromptBuilder
from backend.app.experiments.prompt_ab_test import build_variant_suffix, choose_prompt_variant
from backend.app.utils.quality import annotate_node_quality, compute_quality_score_for_node
from backend.app.engines.citation_extractor import (
    build_citation_provenance,
    summarize_provenance_quality,
)
from backend.app.models.schemas import DecisionNode, Risk, Alternative
from backend.app.database.connection import get_database


class ReasoningEngine:
    def __init__(self, model: Optional[str] = None):
        # Read runtime settings at engine creation time so PROFILE-based env
        # values applied during startup are honored by model calls.
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
        self.model = model or os.getenv("OLLAMA_MODEL", "phi3")
        self.ollama_timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))
        self.ollama_call_max_attempts = int(os.getenv("OLLAMA_CALL_MAX_ATTEMPTS", "2"))
        self.ollama_json_max_retries = int(os.getenv("OLLAMA_JSON_MAX_RETRIES", "3"))
        self.ollama_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
        self.context_max_chunks = int(os.getenv("CONTEXT_MAX_CHUNKS", "4"))
        self.context_chunk_char_limit = int(os.getenv("CONTEXT_CHUNK_CHAR_LIMIT", "240"))

        # V4 WS1: Profile-based backoff configuration
        self.backoff_base_seconds = float(os.getenv("BACKOFF_BASE_SECONDS", "2"))
        self.backoff_multiplier = float(os.getenv("BACKOFF_MULTIPLIER", "2.0"))
        self.backoff_max_seconds = float(os.getenv("BACKOFF_MAX_SECONDS", "32"))
        
        # V4 WS1: Token budget configuration
        self.prompt_token_budget = int(os.getenv("PROMPT_TOKEN_BUDGET", "1500"))
        self.model_output_token_budget = int(os.getenv("MODEL_OUTPUT_TOKEN_BUDGET", "250"))
        self.truncation_warning_threshold = float(os.getenv("TRUNCATION_WARNING_THRESHOLD", "0.85"))
        self.ensemble_enabled = os.getenv("OLLAMA_ENSEMBLE_ENABLED", "0") == "1"
        self.ensemble_candidates = max(1, int(os.getenv("OLLAMA_ENSEMBLE_CANDIDATES", "1")))

        append_log(
            "ReasoningEngine: runtime config "
            f"model={self.model}, timeout={self.ollama_timeout_seconds}s, "
            f"call_attempts={self.ollama_call_max_attempts}, "
            f"json_retries={self.ollama_json_max_retries}, "
            f"num_predict={self.ollama_num_predict}, "
            f"context_max_chunks={self.context_max_chunks}, "
            f"backoff=(base={self.backoff_base_seconds}s, mult={self.backoff_multiplier}, max={self.backoff_max_seconds}s), "
            f"token_budget=(prompt={self.prompt_token_budget}, output={self.model_output_token_budget})"
        )

    def _citation_body_from_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Return project-guide citation body: cache:<id> | <url>."""
        if not isinstance(chunk, dict):
            return None

        cache_id = chunk.get("id") or chunk.get("_id")
        source_url = chunk.get("source_url") or chunk.get("url")
        if not cache_id and not source_url:
            return None

        if cache_id and source_url:
            return f"cache:{cache_id} | {source_url}"
        if cache_id:
            return f"cache:{cache_id}"
        return str(source_url)

    def _normalize_citation(self, citation: Any) -> Optional[str]:
        """Normalize model/context citation values to schema-compatible Source: strings."""
        if citation is None:
            return None
        if isinstance(citation, dict):
            body = self._citation_body_from_chunk(citation)
            if body is None:
                body = citation.get("citation") or citation.get("source") or citation.get("source_url")
        else:
            body = str(citation).strip()

        if not body:
            return None
        if body.startswith("[Source:") and body.endswith("]"):
            body = body[len("[Source:"):-1].strip()
        if body.startswith("Source:"):
            body = body[len("Source:"):].strip()

        if not (body.startswith("cache:") or "http://" in body or "https://" in body):
            return None
        return f"Source: {body}"

    def _dedupe_citations(self, citations: List[Any]) -> List[str]:
        seen = set()
        normalized: List[str] = []
        for citation in citations:
            item = self._normalize_citation(citation)
            if item and item not in seen:
                seen.add(item)
                normalized.append(item)
        return normalized

    def _context_citations(self, context: Optional[Dict[str, Any]], limit: int = 3) -> List[str]:
        if not context or not isinstance(context, dict):
            return []
        chunks = context.get("chunks") or []
        return self._dedupe_citations(chunks[:limit])

    def _compact_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Keep prompt context short and high-signal for lower latency generation.

        Returns ultra-compact text representation instead of JSON to minimize token bloat.
        """
        if not context or not isinstance(context, dict):
            return "No context available."

        chunks = context.get("chunks") or []
        conf = float(context.get("context_confidence", 0.0))

        lines = [f"[Confidence: {conf:.2f}]"]
        for i, c in enumerate(chunks[:max(1, self.context_max_chunks)]):
            content = str(c.get("content", "")).strip()[:self.context_chunk_char_limit]
            citation_body = self._citation_body_from_chunk(c)
            if content:
                if citation_body:
                    lines.append(f"- [Source: {citation_body}] {content}")
                else:
                    lines.append(f"- {content}")

        return "\n".join(lines) if len(lines) > 1 else "No context available."

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

    async def _call_model(self, prompt: str, temperature: float = 0.7, timeout: float = 120.0) -> str:
        attempts = 0
        max_attempts = max(1, self.ollama_call_max_attempts)
        # V4 WS1: Use profile-based backoff configuration with capping
        backoff = self.backoff_base_seconds

        with track_latency('llm.api_call'):
            while attempts < max_attempts:
                attempts += 1
                try:
                    append_log(f"ReasoningEngine: model call attempt {attempts}/{max_attempts}, backoff_next={backoff}s")
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        resp = await client.post(
                            self.ollama_url,
                            json={
                                "model": self.model,
                                "prompt": prompt,
                                "stream": False,
                                "format": "json",
                                "options": {
                                    "temperature": min(max(temperature, 0.0), 1.0),
                                    "num_predict": max(64, self.ollama_num_predict),
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
                    err = str(e) or e.__class__.__name__
                    append_log(f"ReasoningEngine: error on attempt {attempts}/{max_attempts}: {err}")
                    record_event(level="ERROR", action="reasoner.call.error", message="model call failed", details={"attempts": attempts, "error": err, "backoff_next": backoff})
                    if attempts == max_attempts:
                        metrics.record_metric(
                            operation='llm.api_call',
                            retry_count=attempts - 1,
                            success=False,
                            details={'error': err}
                        )
                        raise RuntimeError(f"Model call failed: {err}")
                    # V4 WS1: Sleep with exponential backoff, capped at backoff_max_seconds
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * self.backoff_multiplier, self.backoff_max_seconds)

        return ""

    def _calculate_confidence_score(self, context_confidence: float, validation_retries: int) -> float:
        """Calculate confidence score from retrieval confidence and validation retries."""
        base_score = float(context_confidence)
        retry_penalty = min(validation_retries * 0.1, 0.3)
        adjusted_score = max(0.0, base_score - retry_penalty)

        if base_score < 0.5:
            adjusted_score = min(adjusted_score, 0.5)

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
        # Always assign a backend-owned UUID to prevent duplicate React/node keys
        # when the model repeats an id value across generations.
        data["id"] = str(uuid.uuid4())

        if not data.get("description"):
            data["description"] = (
                "Generated with partial structured output; details were normalized "
                "from the model response."
            )

        # FIX: Handle title as list or non-string (LLM sometimes returns structured formats)
        title = data.get("title")
        if isinstance(title, list):
            # If title is a list (e.g., [{'@type': 'string', 'name': '...'}]), extract first string or use placeholder
            title = next((t.get("name") if isinstance(t, dict) else str(t) for t in title), None) or "Strategic Scenario Analysis"
        elif not isinstance(title, str) or not title.strip():
            title = "Strategic Scenario Analysis"
        data["title"] = title.strip() if isinstance(title, str) else str(title)

        if not data.get("summary"):
            data["summary"] = str(data.get("description", "")).strip()[:180] or (
                "Model provided limited structured content; review details and branch options."
            )

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
            data["risks"] = [{"description": "General uncertainty.", "severity": "Medium", "likelihood": "Medium"}]
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

        citation_candidates: List[Any] = []
        if isinstance(data.get('source_citations'), list):
            citation_candidates.extend(data.get('source_citations') or [])
        if isinstance(data.get('citations'), list):
            citation_candidates.extend(data.get('citations') or [])

        formatted_citations = self._dedupe_citations(citation_candidates)
        data['source_citations'] = formatted_citations
        if formatted_citations:
            append_log(f"ReasoningEngine: Mapped {len(formatted_citations)} citations to source_citations field")

        return data

    def _quality_gate_issues(self, clean_data: Dict[str, Any]) -> List[str]:
        """Return a conservative list of output quality issues that should trigger regeneration."""
        issues: List[str] = []

        title = str(clean_data.get("title", "") or "").strip()
        generic_titles = {
            "strategic scenario analysis",
            "simulation error",
            "untitled decision",
            "generated node",
        }
        if not title:
            issues.append("missing title")
        elif title.lower() in generic_titles:
            issues.append("generic title")

        risks = clean_data.get("risks") or []
        if len(risks) < 2:
            issues.append("fewer than 2 risks")
        for risk in risks:
            risk_description = str(getattr(risk, "description", None) or risk.get("description", "") if isinstance(risk, dict) else "").strip()
            if risk_description and len(risk_description.split()) < 8:
                issues.append("risk description too short")
                break

        alternatives = clean_data.get("alternatives") or []
        if len(alternatives) < 2:
            issues.append("fewer than 2 alternatives")

        citations = clean_data.get("source_citations") or []
        if not citations:
            issues.append("missing citations")

        return issues

    def _extract_and_clean_json(self, raw_text: str) -> Dict[str, Any]:
        """Aggressively clean and extract JSON from LLM output with multiple fallback strategies, including citation enforcement."""
        if not raw_text or not isinstance(raw_text, str):
            raise ValueError("Empty or invalid input text")

        clean_text = raw_text.strip()

        # Remove markdown code fences and common invisible characters early
        clean_text = re.sub(r'```json\s*', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'```\s*', '', clean_text)
        clean_text = clean_text.replace('\ufeff', '')  # BOM
        clean_text = clean_text.strip()

        # Remove zero-width and known problematic unicode separators
        clean_text = re.sub(r'[\u200b-\u200f\u2028-\u202f]', '', clean_text)

        # Remove comments (both // and /* */ style) and tidy trailing commas
        clean_text = re.sub(r'//.*?$', '', clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r'/\*.*?\*/', '', clean_text, flags=re.DOTALL)
        while True:
            new_text = re.sub(r',\s*([}\]])', r'\1', clean_text)
            if new_text == clean_text:
                break
            clean_text = new_text

        # Build an "escaped" variant which replaces unescaped control characters
        # with their unicode escape sequences (\u00XX). This often repairs
        # malformed JSON where the LLM injected raw control chars.
        def _escape_control_sequences(text: str) -> str:
            def esc(m):
                ch = m.group(0)
                return "\\u%04x" % ord(ch)

            return re.sub(r'[\x00-\x1f\u0080-\u009f]', esc, text)

        escaped_text = _escape_control_sequences(clean_text)

        # Extract and remove citation tokens before parsing JSON.
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        citation_matches = re.findall(citation_pattern, raw_text)
        citations = self._dedupe_citations(citation_matches)
        if citations:
            append_log(f"ReasoningEngine: Detected citations (Source format): {citations}")

        append_log(f"Validating citations: {citations}")
        invalid_citations = [
            citation for citation in citation_matches
            if not self._normalize_citation(citation)
        ]
        if invalid_citations:
            append_log(f"Invalid citation detected: {invalid_citations[0]}")
            raise ValueError(
                f"Invalid citation format: {invalid_citations[0]}. "
                "Expected [Source: cache:<id> | <url>]"
            )

        # Remove citation tokens from both raw and cleaned text so trailing citation
        # chips after a JSON object do not break large-output validation.
        raw_text = re.sub(citation_pattern, '', raw_text).strip()
        clean_text = re.sub(citation_pattern, '', clean_text).strip()
        escaped_text = re.sub(citation_pattern, '', escaped_text).strip()

        # Quick sanity checks for very large inputs
        if len(raw_text) > 1000:
            append_log("ReasoningEngine: Received large input for processing.")
            append_log(f"ReasoningEngine: First 500 characters of input: {raw_text[:500]}")
            append_log(f"ReasoningEngine: Last 500 characters of input: {raw_text[-500:]}")

        # Parsing strategies (operate on the provided text string)
        parse_attempts = []
        parse_attempts.append(("direct", lambda t: json.loads(t)))

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

        def regex_extract(text):
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                return match.group(0)
            return None

        parse_attempts.append(("regex_extract", lambda t: json.loads(regex_extract(t) or "{}")))

        def outer_braces(text):
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return text[start:end + 1]
            return None

        parse_attempts.append(("outer_braces", lambda t: json.loads(outer_braces(t) or "{}")))

        # Try parsing in order of most-to-least repaired text:
        # 1) escaped_text (control chars replaced by \uXXXX)
        # 2) clean_text (comments/trailing commas removed)
        # 3) raw_text (original)
        last_error = None
        for text_name, text_variant in ("escaped", escaped_text), ("clean", clean_text), ("raw", raw_text):
            for strategy_name, parse_func in parse_attempts:
                try:
                    data = parse_func(text_variant)
                    if isinstance(data, dict) and data:
                        data['citations'] = citations
                        append_log(f"ReasoningEngine: JSON parsed successfully using {text_name}/{strategy_name}")
                        return data
                except Exception as e:
                    last_error = e
                    continue

            # Try decoder with relaxed strict flag as a last attempt on this variant
            try:
                decoder = json.JSONDecoder(strict=False)
                data = decoder.decode(text_variant)
                if isinstance(data, dict) and data:
                    data['citations'] = citations
                    append_log(f"ReasoningEngine: JSON decoded with relaxed decoder on {text_name}")
                    return data
            except Exception as e:
                last_error = e
                continue

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

        elif 'quality gate failed' in error_lower:
            return (
                f"{base} Your output parsed but failed quality validation: {error_message[:160]}. "
                "Produce at least 2 specific risks, at least 2 distinct alternatives, "
                "a non-generic title, and grounded Source citations."
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

    def _enforce_grounded_confidence(self, clean_data: Dict[str, Any], job_id: Optional[str] = None) -> None:
        confidence_score = float(clean_data.get('confidence_score', 0.0) or 0.0)
        has_citations = bool(clean_data.get('source_citations', []))
        citation_coverage = float(clean_data.get('citation_coverage', 0.0) or 0.0)
        citation_quality_score = float(clean_data.get('citation_quality_score', 0.0) or 0.0)

        # Only enforce strict quality checks when a quality score was computed and is low.
        quality_score_present = 'citation_quality_score' in clean_data
        if confidence_score >= 0.7 and (
            not has_citations or citation_coverage < 0.5 or (quality_score_present and citation_quality_score < 0.7)
        ):
            clean_data['confidence_score'] = min(confidence_score, 0.49)
            clean_data['speculative'] = True
            record_event(
                level="WARN",
                action="reasoner.grounding_enforced",
                message="Downgraded high-confidence node due to weak citation grounding",
                details={
                    "job_id": job_id,
                    "original_confidence": confidence_score,
                    "new_confidence": clean_data['confidence_score'],
                    "citation_coverage": citation_coverage,
                    "citation_quality_score": citation_quality_score,
                    "has_citations": has_citations,
                },
            )
            return

        # Soft cap: apply when citations exist and either coverage is low or
        # a computed quality score indicates lower trust.
        if confidence_score >= 0.5 and has_citations and (
            citation_coverage < 0.3 or (quality_score_present and citation_quality_score < 0.5)
        ):
            clean_data['confidence_score'] = min(confidence_score, 0.59)
            clean_data['speculative'] = True
            record_event(
                level="INFO",
                action="reasoner.grounding_soft_enforced",
                message="Applied soft confidence cap due to low citation coverage",
                details={
                    "job_id": job_id,
                    "original_confidence": confidence_score,
                    "new_confidence": clean_data['confidence_score'],
                    "citation_coverage": citation_coverage,
                    "citation_quality_score": citation_quality_score,
                },
            )

    def _enforce_provenance_quality(self, clean_data: Dict[str, Any], job_id: Optional[str] = None) -> None:
        provenance = list(clean_data.get('citation_provenance') or [])
        if not provenance:
            return

        summary = summarize_provenance_quality(provenance)
        clean_data['citation_coverage'] = summary['coverage']
        clean_data['citation_quality_score'] = summary['quality_score']
        clean_data['citation_provenance_completeness'] = summary['completeness']
        clean_data['citation_provenance_matched_count'] = summary['matched_count']
        clean_data['citation_provenance_unmatched_count'] = summary['unmatched_count']

        if summary['coverage'] < 0.85 or summary['quality_score'] < 0.70:
            clean_data['speculative'] = True
            record_event(
                level="WARN",
                action="reasoner.provenance_quality_warn",
                message="Citation provenance below production grounding target",
                details={
                    "job_id": job_id,
                    "coverage": summary['coverage'],
                    "quality_score": summary['quality_score'],
                    "matched_count": summary['matched_count'],
                    "unmatched_count": summary['unmatched_count'],
                },
            )

    def _rerank_citations_by_quality(self, clean_data: Dict[str, Any]) -> None:
        provenance = list(clean_data.get('citation_provenance') or [])
        if not provenance:
            return

        ranked = sorted(
            provenance,
            key=lambda item: float(item.get('citation_quality_score', 0.0) or 0.0),
            reverse=True,
        )
        clean_data['citation_provenance'] = ranked

        ranked_labels = [item.get('source_label') for item in ranked if item.get('source_label')]
        normalized = self._dedupe_citations(ranked_labels)
        if normalized:
            clean_data['source_citations'] = normalized

    def _build_ensemble_prompt_variants(self, prompt: str) -> List[str]:
        """Return prompt suffixes for optional multi-prompt ensemble reranking."""
        if not self.ensemble_enabled or self.ensemble_candidates <= 1:
            return [""]

        suffixes = [
            "Focus on grounded evidence, citation completeness, and conservative claims.",
            "Focus on concrete failure modes, operational risks, and high-signal alternatives.",
            "Focus on authority, recency, retrieval support, and provenance clarity.",
            "Focus on concise but distinct alternatives with strong evidence coverage.",
        ]

        variants = [""]
        for suffix in suffixes[: max(0, self.ensemble_candidates - 1)]:
            variants.append(suffix)
        return variants

    def _score_ensemble_candidate(self, node: DecisionNode) -> float:
        """Score a candidate node using quality and grounding signals."""
        quality_score = float(getattr(node, 'quality_score', 0.0) or 0.0)
        if quality_score <= 0.0:
            quality_score = compute_quality_score_for_node(node, [])

        citation_quality_score = float(getattr(node, 'citation_quality_score', 0.0) or 0.0)
        citation_coverage = float(getattr(node, 'citation_coverage', 0.0) or 0.0)
        confidence_score = float(getattr(node, 'confidence_score', 0.0) or 0.0)
        novelty_bonus = min(float(getattr(node, 'title_novelty_score', 0.0) or 0.0), 1.0) * 0.05
        speculative_penalty = 0.12 if getattr(node, 'speculative', False) else 0.0

        score = (
            0.40 * quality_score
            + 0.25 * citation_quality_score
            + 0.20 * citation_coverage
            + 0.15 * confidence_score
            + novelty_bonus
            - speculative_penalty
        )
        return round(max(score, 0.0), 3)

    async def generate_decision(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        persona: str = "Skeptical Analyst",
        temperature: Optional[float] = None,
        validation_retries: int = 0,
    ) -> DecisionNode:
        """Generate a decision node, optionally reranking ensemble candidates."""
        if not self.ensemble_enabled or self.ensemble_candidates <= 1:
            return await self._generate_decision_single(
                prompt=prompt,
                context=context,
                job_id=job_id,
                persona=persona,
                temperature=temperature,
                validation_retries=validation_retries,
                prompt_suffix="",
            )

        prompt_suffixes = self._build_ensemble_prompt_variants(prompt)
        candidates: List[DecisionNode] = []

        for variant_index, prompt_suffix in enumerate(prompt_suffixes, start=1):
            candidate = await self._generate_decision_single(
                prompt=prompt,
                context=context,
                job_id=job_id,
                persona=persona,
                temperature=temperature,
                validation_retries=validation_retries,
                prompt_suffix=prompt_suffix,
            )
            annotate_node_quality(candidate, [])
            candidate_score = self._score_ensemble_candidate(candidate)
            setattr(candidate, "ensemble_candidate_score", candidate_score)
            setattr(candidate, "created_by_engine", f"ensemble_candidate_{variant_index}")
            candidates.append(candidate)

        best_candidate = max(candidates, key=self._score_ensemble_candidate)
        record_event(
            level="INFO",
            action="reasoner.ensemble_rerank",
            message="selected best candidate from ensemble prompts",
            details={
                "job_id": job_id,
                "candidate_count": len(candidates),
                "best_score": self._score_ensemble_candidate(best_candidate),
            },
        )
        return best_candidate

    async def _generate_decision_single(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None, 
        job_id: Optional[str] = None,
        persona: str = "Skeptical Analyst",
        temperature: Optional[float] = None,
        validation_retries: int = 0,  # Track retry count for confidence calculation
        prompt_suffix: str = "",
    ) -> DecisionNode:
        with track_latency('llm.generate'):
            # Sample temperature if not provided (0.5-0.8 range per spec)
            if temperature is None:
                temperature = round(random.uniform(0.5, 0.8), 2)
            
            # Extract context_confidence from context dict
            context_confidence = 0.0
            if context and isinstance(context, dict):
                context_confidence = context.get('context_confidence', 0.0)
            compact_context = self._compact_context(context)
            prompt_experiment_variant = choose_prompt_variant(job_id or prompt)
            
            # Build the structured V2 prompt first, then fall back to the compact
            # inline prompt if the template loader fails for any reason.
            try:
                full_prompt = PromptBuilder.build_v2_prompt(
                    prompt=prompt,
                    context_text=compact_context,
                    persona=persona,
                    use_template=True,
                )
                full_prompt = f"{full_prompt}{build_variant_suffix(prompt_experiment_variant)}"
                if prompt_suffix:
                    full_prompt = f"{full_prompt}\n\nENSEMBLE FOCUS:\n{prompt_suffix}"
            except Exception as prompt_error:
                append_log(f"ReasoningEngine: V2 prompt builder failed: {prompt_error}")
                persona_text = self._get_persona_prompt(persona)
                instruction = (
                    f"You are a strategic simulation engine. {persona_text}\n\n"
                    "CRITICAL RULES:\n"
                    "1. You MUST respond with ONLY valid JSON. No explanatory text, no markdown, no code blocks.\n"
                    "2. Start your response with {{ and end with }}. Nothing else.\n"
                    "3. Schema: {title, summary, description, risks: [{description, severity (Low/Medium/High), likelihood (Low/Medium/High)}], alternatives: [{description, action_type}], source_citations: [string], speculative: boolean}\n"
                    "4. Use only sources shown in CONTEXT. Every factual claim should cite them with [Source: cache:<id> | <url>]. Also copy each used citation into source_citations as \"Source: cache:<id> | <url>\".\n"
                    "5. If claim cannot be grounded, set speculative: true.\n"
                    "6. Keep each text field concise (1-2 sentences max).\n"
                    "7. Ensure all strings are properly quoted, all commas are correct, no trailing commas."
                )
                full_prompt = f"{instruction}\n\nSCENARIO: {prompt}\n\nCONTEXT:\n{compact_context}\n\nJSON OUTPUT:"
                full_prompt = f"{full_prompt}{build_variant_suffix(prompt_experiment_variant)}"
        
        # V4 WS1: Log token budget diagnostics for the built prompt
        token_diags = log_truncation_diagnostics(
            full_prompt,
            self.prompt_token_budget,
            self.truncation_warning_threshold,
            context_name="full_prompt"
        )
        
        # Retry logic for JSON parsing with progressively clearer instructions
        # V2: adversarial retry logic is capped at 3 attempts to fail closed.
        max_json_retries = min(3, max(1, self.ollama_json_max_retries))
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
                
                body = await self._call_model(
                    retry_prompt,
                    temperature=temperature,
                    timeout=self.ollama_timeout_seconds,
                )
                append_log(f"ReasoningEngine: raw output len={len(body)} (attempt {json_attempt + 1})")
                record_event(level="INFO", action="reasoner.raw_output", message="raw output received", details={"job_id": job_id, "length": len(body), "attempt": json_attempt + 1})

                # V4 WS1: Log token diagnostics for model output
                output_token_diags = log_truncation_diagnostics(
                    body,
                    self.model_output_token_budget,
                    self.truncation_warning_threshold,
                    context_name="model_output"
                )

                # Extract and clean JSON
                data = self._extract_and_clean_json(body)
                
                # If we got here, JSON parsing succeeded
                clean_data = self._janitor_fix_data(data)

                if not clean_data.get('source_citations'):
                    context_citations = self._context_citations(context)
                    if context_citations:
                        clean_data['source_citations'] = context_citations
                        append_log(
                            "ReasoningEngine: Added context citations because model omitted "
                            "source_citations"
                        )
                        record_event(
                            level="WARN",
                            action="reasoner.context_citations_applied",
                            message="Model omitted citations; using retrieved context citations",
                            details={"job_id": job_id, "citation_count": len(context_citations)}
                        )

                context_chunks = []
                if context and isinstance(context, dict):
                    context_chunks = context.get('chunks') or []
                clean_data['citation_provenance'] = build_citation_provenance(
                    clean_data.get('source_citations', []),
                    context_chunks,
                )
                clean_data['prompt_experiment_variant'] = prompt_experiment_variant
                clean_data['prompt_experiment_batch_id'] = job_id
                self._rerank_citations_by_quality(clean_data)
                self._enforce_provenance_quality(clean_data=clean_data, job_id=job_id)

                quality_issues = self._quality_gate_issues(clean_data)
                if quality_issues:
                    last_parse_error = f"Quality gate failed: {', '.join(quality_issues)}"
                    append_log(f"ReasoningEngine: {last_parse_error}")
                    record_event(
                        level="WARN",
                        action="reasoner.quality_gate_failed",
                        message="parsed output failed quality gates",
                        details={"job_id": job_id, "issues": quality_issues, "attempt": json_attempt + 1},
                    )
                    if json_attempt < max_json_retries - 1:
                        await asyncio.sleep(0.5 * (json_attempt + 1))
                        continue
                    raise ValueError(last_parse_error)
                
                # Calculate confidence_score from retrieval metrics and validation success
                confidence_score = self._calculate_confidence_score(
                    context_confidence=context_confidence,
                    validation_retries=validation_retries
                )
                clean_data['confidence_score'] = confidence_score
                self._enforce_grounded_confidence(clean_data=clean_data, job_id=job_id)
                confidence_score = float(clean_data.get('confidence_score', confidence_score))
                has_citations = bool(clean_data.get('source_citations', []))
                
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
                        "prompt_experiment_variant": prompt_experiment_variant,
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
                        risks=[Risk(description="Critical simulation instability due to invalid model output", severity="High", likelihood="High")],
                        alternatives=[],
                        confidence_score=error_confidence,
                        speculative=True,
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
            risks=[Risk(description="Critical simulation instability due to invalid model output", severity="High", likelihood="High")],
            alternatives=[],
            confidence_score=error_confidence,
            speculative=True,
        )
