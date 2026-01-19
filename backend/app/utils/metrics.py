"""
Local telemetry metrics collection for DGS.
All metrics stay on user's machine - no external reporting.

Per project guide section 11: "Local logs (console or file) should include:
[METRIC] Latency: Xs | CacheHit: True/False | TopSim: 0.87 | Retries: N"
"""
import time
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from backend.app.utils.logger import append_log, record_event


class MetricsCollector:
    """Singleton for collecting and logging local telemetry metrics."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.metrics_buffer = []
        return cls._instance
    
    def record_metric(
        self, 
        operation: str, 
        latency_ms: Optional[float] = None,
        cache_hit: Optional[bool] = None,
        similarity_score: Optional[float] = None,
        retry_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record a metric event for local telemetry.
        
        Per project guide section 11: Log latency, cache hits, similarity, retries
        
        Args:
            operation: Name of operation (e.g., 'rag.retrieval', 'llm.generate', 'scrape.parallel')
            latency_ms: Operation duration in milliseconds
            cache_hit: Whether cache was used (True/False/None)
            similarity_score: Top similarity score for retrieval operations
            retry_count: Number of retries for failed operations
            chunk_count: Number of chunks processed/retrieved
            success: Whether operation succeeded
            details: Additional context
        """
        metric_parts = [f"[METRIC] {operation}"]
        
        if latency_ms is not None:
            metric_parts.append(f"Latency: {latency_ms:.0f}ms")
        
        if cache_hit is not None:
            metric_parts.append(f"CacheHit: {cache_hit}")
        
        if similarity_score is not None:
            metric_parts.append(f"TopSim: {similarity_score:.2f}")
        
        if retry_count is not None:
            metric_parts.append(f"Retries: {retry_count}")
        
        if chunk_count is not None:
            metric_parts.append(f"Chunks: {chunk_count}")
        
        metric_parts.append(f"Success: {success}")
        
        metric_line = " | ".join(metric_parts)
        append_log(metric_line)
        
        # Also record as structured event for analysis
        record_event(
            level="METRIC",
            action=f"metric.{operation}",
            message=metric_line,
            details={
                "timestamp": datetime.now().isoformat(),
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "similarity_score": similarity_score,
                "retry_count": retry_count,
                "chunk_count": chunk_count,
                "success": success,
                **(details or {})
            }
        )


@contextmanager
def track_latency(operation: str, **kwargs):
    """Context manager to automatically track operation latency.
    
    Usage:
        with track_latency('rag.retrieval', cache_hit=True):
            # ... perform operation ...
            pass
    
    Args:
        operation: Name of the operation being tracked
        **kwargs: Additional metric parameters (cache_hit, similarity_score, etc.)
    """
    start_time = time.time()
    success = True
    error = None
    
    try:
        yield
    except Exception as e:
        success = False
        error = str(e)
        raise
    finally:
        latency_ms = (time.time() - start_time) * 1000
        
        # Merge any additional details
        details = kwargs.pop('details', {})
        if error:
            details['error'] = error
        
        metrics.record_metric(
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            details=details if details else None,
            **kwargs
        )


# Global singleton instance
metrics = MetricsCollector()
