"""
Job concurrency management for V4 WS1.
Provides utilities for throttling concurrent jobs and managing job queues.
"""

import os
import asyncio
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

from backend.app.utils.logger import append_log


class JobConcurrencyManager:
    """
    Manages job concurrency limits and queuing.
    Ensures no more than N concurrent jobs are running.
    """
    
    def __init__(self, max_concurrent: Optional[int] = None):
        self.max_concurrent = max_concurrent or int(os.getenv("JOB_CONCURRENCY_LIMIT", "3"))
        self.active_jobs: Dict[str, datetime] = {}  # job_id -> start_time
        self.queued_jobs: list = []  # job_ids waiting to start
        self.lock = asyncio.Lock()
    
    async def acquire_slot(self, job_id: str) -> bool:
        """
        Try to acquire a concurrency slot for a job.
        Returns True if slot acquired immediately, False if queued.
        """
        async with self.lock:
            if len(self.active_jobs) < self.max_concurrent:
                self.active_jobs[job_id] = datetime.now(timezone.utc)
                append_log(f"Job concurrency: ACQUIRED slot for {job_id} ({len(self.active_jobs)}/{self.max_concurrent})")
                return True
            else:
                if job_id not in self.queued_jobs:
                    self.queued_jobs.append(job_id)
                append_log(f"Job concurrency: QUEUED {job_id} (queue size: {len(self.queued_jobs)})")
                return False
    
    async def release_slot(self, job_id: str) -> Optional[str]:
        """
        Release concurrency slot for a job.
        Returns next job_id to start from queue, or None if queue empty.
        """
        async with self.lock:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
                append_log(f"Job concurrency: RELEASED slot for {job_id} ({len(self.active_jobs)}/{self.max_concurrent})")
            
            # Promote next job from queue
            if self.queued_jobs:
                next_job_id = self.queued_jobs.pop(0)
                self.active_jobs[next_job_id] = datetime.now(timezone.utc)
                append_log(f"Job concurrency: PROMOTED {next_job_id} from queue ({len(self.active_jobs)}/{self.max_concurrent})")
                return next_job_id
            
            return None
    
    async def wait_for_slot(self, job_id: str, timeout_seconds: int = 300) -> bool:
        """
        Wait for a concurrency slot to become available (with timeout).
        Returns True if slot acquired, False if timeout.
        """
        start_time = datetime.now(timezone.utc)

        # First attempt may acquire immediately or enqueue once.
        if await self.acquire_slot(job_id):
            return True

        while True:
            async with self.lock:
                if job_id in self.active_jobs:
                    return True

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed > timeout_seconds:
                async with self.lock:
                    if job_id in self.queued_jobs:
                        self.queued_jobs = [queued_id for queued_id in self.queued_jobs if queued_id != job_id]
                append_log(f"Job concurrency: TIMEOUT waiting for slot for {job_id} after {elapsed}s")
                return False

            # Wait a bit before checking again without re-queueing.
            await asyncio.sleep(0.25)
    
    def get_stats(self) -> Dict:
        """Get current concurrency stats."""
        return {
            "active_jobs": len(self.active_jobs),
            "queued_jobs": len(self.queued_jobs),
            "max_concurrent": self.max_concurrent,
            "utilization": len(self.active_jobs) / self.max_concurrent,
            "active_job_ids": list(self.active_jobs.keys()),
            "queued_job_ids": list(self.queued_jobs),
        }


# Global concurrency manager singleton
_concurrency_manager: Optional[JobConcurrencyManager] = None


def get_concurrency_manager() -> JobConcurrencyManager:
    """Get or create the global concurrency manager."""
    global _concurrency_manager
    if _concurrency_manager is None:
        _concurrency_manager = JobConcurrencyManager()
    return _concurrency_manager


def reset_concurrency_manager():
    """Reset the global concurrency manager (for testing)."""
    global _concurrency_manager
    _concurrency_manager = None
