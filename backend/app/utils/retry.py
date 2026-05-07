"""Centralized async retry and backoff utilities for V4.

Provides a simple async retry decorator used across engines and network calls.
"""
import asyncio
import os
import math
import random
from typing import Callable, Any, Coroutine, Optional


def _get_backoff_config():
    base = float(os.getenv("BACKOFF_BASE_SECONDS", "2"))
    multiplier = float(os.getenv("BACKOFF_MULTIPLIER", "2.0"))
    maximum = float(os.getenv("BACKOFF_MAX_SECONDS", "16"))
    return base, multiplier, maximum


def async_retry(max_attempts: Optional[int] = None):
    """Decorator to retry an async function with exponential backoff.

    Usage:
        @async_retry(max_attempts=3)
        async def call():
            ...
    """
    def _decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        async def _wrapped(*args, **kwargs):
            attempts = int(max_attempts or int(os.getenv("OLLAMA_CALL_MAX_ATTEMPTS", "4")))
            base, multiplier, maximum = _get_backoff_config()

            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt >= attempts:
                        raise
                    # exponential backoff with jitter
                    backoff = min(maximum, base * (multiplier ** (attempt - 1)))
                    jitter = backoff * 0.1
                    sleep_for = backoff + (jitter * (2 * (random.random()) - 1))
                    # guard negative
                    if sleep_for < 0:
                        sleep_for = backoff
                    await asyncio.sleep(sleep_for)

        return _wrapped
    return _decorator


async def backoff_sleep(attempt: int):
    """Returnively sleep for an attempt-based backoff (helper)."""
    base, multiplier, maximum = _get_backoff_config()
    backoff = min(maximum, base * (multiplier ** max(0, attempt - 1)))
    await asyncio.sleep(backoff)
