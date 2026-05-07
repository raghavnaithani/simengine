"""
Token budget and truncation diagnostics for V4 WS1.
Provides utilities for counting tokens, enforcing budgets, and diagnostic logging.
"""

import os
import re
from typing import Dict, Tuple

from backend.app.utils.logger import append_log


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for a given text using a simple heuristic.
    Assumes ~4 characters per token on average (standard for English).
    For more accurate counting, use a proper tokenizer library if integrated.
    """
    if not text:
        return 0
    # Remove extra whitespace and count words as a proxy
    words = len(text.split())
    # Approximate: ~1.3 tokens per word on average for English
    return max(1, int(words * 1.3))


def truncate_to_budget(text: str, token_budget: int, estimate_only: bool = False) -> Tuple[str, Dict]:
    """
    Truncate text to fit within token budget. Returns (truncated_text, diagnostics).
    
    Args:
        text: Input text to truncate
        token_budget: Maximum tokens allowed
        estimate_only: If True, only estimate; don't actually truncate
    
    Returns:
        (truncated_text, diagnostics_dict)
    """
    original_token_count = estimate_token_count(text)
    diagnostics = {
        "original_token_count": original_token_count,
        "token_budget": token_budget,
        "truncated": False,
        "truncation_ratio": 1.0,
    }
    
    if original_token_count <= token_budget:
        return text, diagnostics
    
    if estimate_only:
        diagnostics["truncated"] = True
        diagnostics["truncation_ratio"] = original_token_count / token_budget
        return text, diagnostics
    
    # Binary search to find a good truncation point
    low, high = 0, len(text)
    best_truncation = text
    
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        candidate_tokens = estimate_token_count(candidate)
        
        if candidate_tokens <= token_budget:
            best_truncation = candidate
            low = mid + 1
        else:
            high = mid - 1
    
    # Add ellipsis if truncated
    if len(best_truncation) < len(text):
        best_truncation = best_truncation.rstrip() + "..."
    
    final_token_count = estimate_token_count(best_truncation)
    diagnostics["truncated"] = True
    diagnostics["final_token_count"] = final_token_count
    diagnostics["truncation_ratio"] = original_token_count / max(1, final_token_count)
    
    return best_truncation, diagnostics


def check_truncation_warning(token_count: int, token_budget: int, warning_threshold: float = 0.85) -> Tuple[bool, str]:
    """
    Check if token count is approaching budget and return warning message.
    
    Args:
        token_count: Actual token count
        token_budget: Maximum tokens allowed
        warning_threshold: Fraction of budget (0-1) at which to warn
    
    Returns:
        (warning_triggered, message)
    """
    if token_budget <= 0:
        return False, ""
    
    usage_ratio = token_count / token_budget
    
    if usage_ratio >= warning_threshold:
        message = (
            f"⚠️  Token budget warning: {token_count}/{token_budget} tokens ({usage_ratio*100:.1f}%) - "
            f"approaching limit"
        )
        return True, message
    
    return False, ""


def log_truncation_diagnostics(
    text: str,
    token_budget: int,
    truncation_warning_threshold: float,
    context_name: str = "prompt"
):
    """
    Log comprehensive truncation diagnostics for a text/prompt.
    Called during prompt building and model response handling.
    """
    token_count = estimate_token_count(text)
    truncated_text, diags = truncate_to_budget(text, token_budget, estimate_only=True)
    warning_triggered, warning_msg = check_truncation_warning(token_count, token_budget, truncation_warning_threshold)
    
    status = "TRUNCATION" if diags["truncated"] else "OK"
    append_log(
        f"Token budget diagnostics ({context_name}): {status} - "
        f"tokens={token_count}/{token_budget}, ratio={diags['truncation_ratio']:.2f}"
    )
    
    if warning_triggered:
        append_log(warning_msg)
    
    return {
        "status": status,
        "token_count": token_count,
        "token_budget": token_budget,
        "truncation_ratio": diags["truncation_ratio"],
        "warning": warning_msg,
    }


def build_prompt_with_budget(
    base_prompt: str,
    context: str,
    token_budget: int,
    truncation_warning_threshold: float = 0.85
) -> Tuple[str, Dict]:
    """
    Build a full prompt ensuring it stays within token budget.
    Truncates context if necessary, logs diagnostics.
    
    Returns:
        (final_prompt, diagnostics)
    """
    full_prompt = f"{base_prompt}\n\n{context}" if context else base_prompt
    full_tokens = estimate_token_count(full_prompt)
    
    diagnostics = {
        "base_tokens": estimate_token_count(base_prompt),
        "context_tokens": estimate_token_count(context),
        "full_tokens": full_tokens,
        "token_budget": token_budget,
        "truncated": False,
    }
    
    if full_tokens > token_budget:
        # Truncate context to fit budget
        base_tokens = diagnostics["base_tokens"]
        available_for_context = token_budget - base_tokens
        
        if available_for_context < 50:
            append_log(f"⚠️  WARNING: Very little room for context ({available_for_context} tokens remaining)")
        
        truncated_context, ctx_diags = truncate_to_budget(context, available_for_context)
        full_prompt = f"{base_prompt}\n\n{truncated_context}" if truncated_context else base_prompt
        diagnostics["truncated"] = True
        diagnostics["truncation_ratio"] = ctx_diags["truncation_ratio"]
    
    # Log warning if approaching threshold
    warning_triggered, warning_msg = check_truncation_warning(full_tokens, token_budget, truncation_warning_threshold)
    if warning_triggered:
        append_log(warning_msg)
        diagnostics["warning"] = warning_msg
    
    append_log(
        f"Prompt built: base={diagnostics['base_tokens']} + context={diagnostics['context_tokens']} = "
        f"{estimate_token_count(full_prompt)}/{token_budget} tokens"
    )
    
    return full_prompt, diagnostics
