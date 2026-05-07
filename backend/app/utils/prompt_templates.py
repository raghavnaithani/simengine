"""Prompt contract helpers for grounded decision generation.

These helpers centralize the JSON contract and hard fail-closed rules used by
reasoning prompts so enforcement remains consistent across templates.
"""

from __future__ import annotations


def decision_node_json_schema() -> str:
    """Return the exact JSON schema contract expected from the model."""
    return (
        "{\n"
        '  "title": "string",\n'
        '  "summary": "string",\n'
        '  "description": "string",\n'
        '  "time_step": 0,\n'
        '  "risks": [\n'
        "    {\n"
        '      "description": "string",\n'
        '      "severity": "Low|Medium|High|Critical",\n'
        '      "likelihood": "Low|Medium|High"\n'
        "    }\n"
        "  ],\n"
        '  "alternatives": [\n'
        "    {\n"
        '      "description": "string",\n'
        '      "action_type": "string"\n'
        "    }\n"
        "  ],\n"
        '  "source_citations": ["Source: cache:<id> | <url>"],\n'
        '  "speculative": false\n'
        "}"
    )


def citation_rules() -> str:
    """Return strict citation requirements for grounded output."""
    return (
        "Citation rules:\n"
        "- Every factual claim must be grounded in provided evidence chunks.\n"
        "- Include inline citation tokens in output text using: [Source: cache:<id> | <url>].\n"
        "- Copy each used citation into source_citations as: Source: cache:<id> | <url>.\n"
        "- Never invent cache ids or URLs that are not present in context."
    )


def risk_rules() -> str:
    """Return minimum risk requirements."""
    return (
        "Risk rules:\n"
        "- Provide at least 2 concrete risks.\n"
        "- At least one risk should be High or Critical when scenario implies material downside.\n"
        "- Avoid generic phrases like 'general uncertainty'."
    )


def hard_fail_closed_rule() -> str:
    """Return the fail-closed instruction for missing grounding."""
    return (
        "Fail-closed rule:\n"
        "- If grounding is insufficient, do not fabricate support.\n"
        "- Set speculative=true and use conservative language.\n"
        "- Output valid JSON anyway; never output free-form explanations."
    )


def full_prompt_contract() -> str:
    """Return the compact full prompt contract block."""
    return (
        "Output contract:\n"
        "- Return ONLY valid JSON, no markdown/code fences.\n"
        f"- JSON schema:\n{decision_node_json_schema()}\n"
        f"{citation_rules()}\n"
        f"{risk_rules()}\n"
        f"{hard_fail_closed_rule()}"
    )
