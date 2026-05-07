"""
Ollama configuration profiles for different deployment environments.
Controls timeout, retry logic, and resource allocation per profile.

Per user requirement: Local runs should complete in <2 minutes with real LLM output.
"""

from typing import Dict, Any
from enum import Enum
import os


class Profile(str, Enum):
    """Deployment profiles with different timeout/retry configurations."""
    LOCAL = "local"
    PROD = "prod"
    DEBUG = "debug"
    STAGING = "staging"


PROFILES: Dict[str, Dict[str, Any]] = {
    "local": {
        "description": "Local development with sufficient timeouts for phi3 LLM - V4 WS1 Tuning",
        "ollama_timeout_seconds": 60,
        "ollama_call_max_attempts": 4,
        "ollama_json_max_retries": 3,
        "ollama_num_predict": 200,
        "ollama_context_token_limit": 2048,
        "context_max_chunks": 3,
        "context_chunk_char_limit": 180,
        "vector_search_timeout_seconds": 5,
        "embedding_timeout_seconds": 10,
        "scraper_timeout_seconds": 4,
        "parallel_scrape_max_workers": 2,
        "rag_candidates_top_n": 10,
        "rag_filtered_keep": 4,
        "dgs_use_web": False,
        "resource_memory_gb": 2,
        "backoff_base_seconds": 2,
        "backoff_multiplier": 2.0,
        "backoff_max_seconds": 16,
        "prompt_token_budget": 1500,
        "model_output_token_budget": 250,
        "truncation_warning_threshold": 0.85,
        "job_concurrency_limit": 3,
        "job_poll_timeout_seconds": 120,
        "job_poll_interval_seconds": 3,
    },
    "prod": {
        "description": "Production with generous timeouts for reliability - V4 WS1 Tuning",
        "ollama_timeout_seconds": 90,
        "ollama_call_max_attempts": 5,
        "ollama_json_max_retries": 4,
        "ollama_num_predict": 250,
        "ollama_context_token_limit": 2048,
        "context_max_chunks": 4,
        "context_chunk_char_limit": 200,
        "vector_search_timeout_seconds": 30,
        "embedding_timeout_seconds": 60,
        "scraper_timeout_seconds": 8,
        "parallel_scrape_max_workers": 4,
        "rag_candidates_top_n": 12,
        "rag_filtered_keep": 5,
        "dgs_use_web": True,
        "resource_memory_gb": 8,
        "backoff_base_seconds": 2,
        "backoff_multiplier": 2.0,
        "backoff_max_seconds": 32,
        "prompt_token_budget": 2000,
        "model_output_token_budget": 300,
        "truncation_warning_threshold": 0.85,
        "job_concurrency_limit": 10,
        "job_poll_timeout_seconds": 300,
        "job_poll_interval_seconds": 2,
    },
    "debug": {
        "description": "Debug mode with extreme timeouts and verbose logging - V4 WS1 Tuning",
        "ollama_timeout_seconds": 300,
        "ollama_call_max_attempts": 6,
        "ollama_json_max_retries": 5,
        "ollama_num_predict": 150,
        "ollama_context_token_limit": 3072,
        "context_max_chunks": 3,
        "context_chunk_char_limit": 250,
        "vector_search_timeout_seconds": 60,
        "embedding_timeout_seconds": 120,
        "scraper_timeout_seconds": 120,
        "parallel_scrape_max_workers": 1,
        "rag_candidates_top_n": 15,
        "rag_filtered_keep": 7,
        "dgs_use_web": False,
        "resource_memory_gb": 4,
        "backoff_base_seconds": 1,
        "backoff_multiplier": 2.0,
        "backoff_max_seconds": 64,
        "prompt_token_budget": 3000,
        "model_output_token_budget": 400,
        "truncation_warning_threshold": 0.90,
        "job_concurrency_limit": 2,
        "job_poll_timeout_seconds": 600,
        "job_poll_interval_seconds": 1,
    },
    "staging": {
        "description": "Staging profile with ensemble reranking enabled for WS4 validation",
        "ollama_timeout_seconds": 75,
        "ollama_call_max_attempts": 4,
        "ollama_json_max_retries": 4,
        "ollama_num_predict": 220,
        "ollama_context_token_limit": 2048,
        "context_max_chunks": 4,
        "context_chunk_char_limit": 190,
        "vector_search_timeout_seconds": 20,
        "embedding_timeout_seconds": 40,
        "scraper_timeout_seconds": 6,
        "parallel_scrape_max_workers": 3,
        "rag_candidates_top_n": 12,
        "rag_filtered_keep": 5,
        "dgs_use_web": True,
        "resource_memory_gb": 6,
        "backoff_base_seconds": 2,
        "backoff_multiplier": 2.0,
        "backoff_max_seconds": 24,
        "prompt_token_budget": 2200,
        "model_output_token_budget": 300,
        "truncation_warning_threshold": 0.85,
        "job_concurrency_limit": 6,
        "job_poll_timeout_seconds": 240,
        "job_poll_interval_seconds": 2,
        "ensemble_enabled": True,
        "ensemble_candidates": 3,
    },
}


def get_profile() -> str:
    """Get current profile from environment or default to 'local'."""
    profile = os.getenv("PROFILE", "local").lower()
    if profile not in PROFILES:
        print(f"⚠️  Unknown profile '{profile}', defaulting to 'local'")
        profile = "local"
    return profile


def get_config() -> Dict[str, Any]:
    """Get configuration for current profile."""
    profile = get_profile()
    return PROFILES[profile].copy()


def apply_profile_env_vars():
    """Apply profile-based environment variables for Ollama and other services.
    
    Should be called at backend startup, before creating engine instances.
    """
    profile = get_profile()
    config = PROFILES[profile]
    
    # Only override if not already set explicitly
    os.environ.setdefault("PROFILE", profile)
    os.environ.setdefault("OLLAMA_TIMEOUT_SECONDS", str(config["ollama_timeout_seconds"]))
    os.environ.setdefault("OLLAMA_CALL_MAX_ATTEMPTS", str(config["ollama_call_max_attempts"]))
    os.environ.setdefault("OLLAMA_JSON_MAX_RETRIES", str(config["ollama_json_max_retries"]))
    os.environ.setdefault("OLLAMA_NUM_PREDICT", str(config["ollama_num_predict"]))
    os.environ.setdefault("CONTEXT_MAX_CHUNKS", str(config["context_max_chunks"]))
    os.environ.setdefault("CONTEXT_CHUNK_CHAR_LIMIT", str(config["context_chunk_char_limit"]))
    os.environ.setdefault("SCRAPER_TIMEOUT_SECONDS", str(config["scraper_timeout_seconds"]))
    os.environ.setdefault("PARALLEL_SCRAPE_MAX_WORKERS", str(config["parallel_scrape_max_workers"]))
    os.environ.setdefault("RAG_CANDIDATES_TOP_N", str(config["rag_candidates_top_n"]))
    os.environ.setdefault("RAG_FILTERED_KEEP", str(config["rag_filtered_keep"]))
    os.environ.setdefault("DGS_USE_WEB", "1" if config.get("dgs_use_web") else "0")
    os.environ.setdefault("DGS_RESPECT_ROBOTS", "1")
    
    # V4 WS1: New configuration variables for backoff, token budgets, and concurrency
    os.environ.setdefault("BACKOFF_BASE_SECONDS", str(config["backoff_base_seconds"]))
    os.environ.setdefault("BACKOFF_MULTIPLIER", str(config["backoff_multiplier"]))
    os.environ.setdefault("BACKOFF_MAX_SECONDS", str(config["backoff_max_seconds"]))
    os.environ.setdefault("PROMPT_TOKEN_BUDGET", str(config["prompt_token_budget"]))
    os.environ.setdefault("MODEL_OUTPUT_TOKEN_BUDGET", str(config["model_output_token_budget"]))
    os.environ.setdefault("TRUNCATION_WARNING_THRESHOLD", str(config["truncation_warning_threshold"]))
    os.environ.setdefault("JOB_CONCURRENCY_LIMIT", str(config["job_concurrency_limit"]))
    os.environ.setdefault("JOB_POLL_TIMEOUT_SECONDS", str(config["job_poll_timeout_seconds"]))
    os.environ.setdefault("JOB_POLL_INTERVAL_SECONDS", str(config["job_poll_interval_seconds"]))
    os.environ.setdefault("OLLAMA_ENSEMBLE_ENABLED", "1" if config.get("ensemble_enabled") else "0")
    os.environ.setdefault("OLLAMA_ENSEMBLE_CANDIDATES", str(config.get("ensemble_candidates", 1)))
    
    profile_name = Profile(profile).name
    print(f"Applied profile: {profile_name}")
    print(f"  - Ollama timeout: {config['ollama_timeout_seconds']}s")
    print(f"  - Ollama max tokens: {config['ollama_num_predict']}")
    print(f"  - Max attempts: {config['ollama_call_max_attempts']}")
    print(f"  - JSON retries: {config['ollama_json_max_retries']}")
    print(f"  - Backoff: base={config['backoff_base_seconds']}s, multiplier={config['backoff_multiplier']}, max={config['backoff_max_seconds']}s")
    print(f"  - Token budget: prompt={config['prompt_token_budget']}, output={config['model_output_token_budget']}")
    print(f"  - Concurrency: {config['job_concurrency_limit']} concurrent jobs")
    print(f"  - Real RAG web mode: {'enabled' if config.get('dgs_use_web') else 'disabled'}")
    print(f"  - Ensemble rerank: {'enabled' if config.get('ensemble_enabled') else 'disabled'} ({config.get('ensemble_candidates', 1)} candidates)")
