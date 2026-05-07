"""V2 Prompt Engineering Module

Loads structured prompts with few-shot examples and enforces output quality constraints.
"""

import os
from typing import Optional, Dict, Any
from backend.app.utils.logger import append_log
from backend.app.utils.prompt_templates import full_prompt_contract


class PromptBuilder:
    """Builds structured decision analysis prompts with persona support."""

    @staticmethod
    def quality_requirements() -> str:
        """Return the compact output contract used by V2 prompts."""
        return (
            "\n\n## QUALITY REQUIREMENTS FOR OUTPUT:\n"
            "- Return ONLY valid JSON. No markdown, no explanations, no code fences.\n"
            "- Title: 5-10 words, specific, and non-generic.\n"
            "- Summary/description: concise, evidence-based, and scenario-specific.\n"
            "- Risks: at least 2 items, each with concrete impact language and no generic filler.\n"
            "- Alternatives: at least 2 distinct options with different action_type values.\n"
            "- source_citations: include at least 1 citation as [Source: cache:<id> | <url>].\n"
            "- speculative: true when evidence is thin or unsupported.\n"
            "- Fail closed: if grounding is missing, do not invent evidence; set speculative=true.\n"
            "\n"
            f"{full_prompt_contract()}\n"
        )

    @staticmethod
    def load_template() -> str:
        """Load decision_prompt.txt template if available, else fallback."""
        template_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "templates", 
            "decision_prompt.txt"
        )
        
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r') as f:
                    return f.read()
            except Exception as e:
                append_log(f"PromptBuilder: Failed to load template: {e}")
                return PromptBuilder.default_template()
        
        return PromptBuilder.default_template()

    @staticmethod
    def default_template() -> str:
        """Fallback prompt if template file not found."""
        return """
# Decision Analysis Prompt (V2 - Structured with Few-Shot)

{persona_prompt}

## Your Task

Analyze the given scenario and provide a structured decision analysis in valid JSON format.

Your analysis MUST include:
1. **Title**: 5-10 words, specific, unique (not generic)
2. **Risks**: Array of 2+ risks, each with description (>15 words, specific), severity, likelihood
3. **Alternatives**: Array of 2+ options with description and action_type
4. **Citations**: At least 1 source, format: `[Source: cache:<id> | <url>]`

## Scenario

{prompt}

## Evidence

{context_text}

## Output Contract

Return ONLY valid JSON. No markdown. No explanations. Use concise fields and grounded claims.

JSON Output:
"""

    @staticmethod
    def build_v2_prompt(
        prompt: str,
        context_text: str,
        persona: str = "Skeptical Analyst",
        use_template: bool = True
    ) -> str:
        """Build a V2-structured prompt with quality enforcement.
        
        Args:
            prompt: Scenario/decision prompt
            context_text: Retrieved context or compact context representation
            persona: Persona name for analysis style
            use_template: Whether to use template file (V2) vs. inline (V1)
            
        Returns:
            Full structured prompt ready for LLM
        """
        
        persona_prompts = {
            "Skeptical Analyst": "You are a skeptical strategic analyst. Focus on critical risks, potential failures, worst-case scenarios. Question assumptions and demand evidence.",
            "Optimistic Founder": "You are an optimistic founder. Focus on opportunities, growth potential, creative solutions. See challenges as innovation opportunities.",
            "Cautious Regulator": "You are a cautious regulator. Prioritize compliance, risk mitigation, systematic evaluation. Require thorough documentation and evidence.",
            "Aggressive Founder": "You are an aggressive founder. Prioritize speed, market capture, bold moves. Accept calculated risks for high rewards.",
            "Pessimistic Analyst": "You are a pessimistic analyst. Expect things to go wrong. Identify failure modes early. Emphasize defensive strategies and risk avoidance.",
        }
        
        persona_text = persona_prompts.get(persona, persona_prompts["Skeptical Analyst"])
        
        if use_template:
            template = PromptBuilder.load_template()
            if "{prompt}" in template:
                full_prompt = template.format(
                    persona_prompt=persona_text,
                    context_text=context_text,
                    prompt=prompt,
                )
            else:
                full_prompt = template.format(
                    persona_prompt=persona_text,
                    context_text=context_text,
                )
        else:
            # V1 fallback
            full_prompt = f"{persona_text}\n\nScenario:\n{prompt}\n\nContext:\n{context_text}\n\nProvide JSON response."
        
        if "SCENARIO:" not in full_prompt and "Scenario:" not in full_prompt:
            full_prompt += f"\n\nSCENARIO:\n{prompt}"

        if "QUALITY REQUIREMENTS FOR OUTPUT" not in full_prompt:
            full_prompt += PromptBuilder.quality_requirements()
        
        return full_prompt
