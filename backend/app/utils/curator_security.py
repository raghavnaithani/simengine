"""V3 Security and curator access control utilities."""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Header, HTTPException


# Configuration
CURATOR_RETENTION_DAYS = int(os.getenv("CURATOR_RETENTION_DAYS", "60"))
CURATOR_ROLE_REQUIRED = os.getenv("CURATOR_ROLE_REQUIRED", "0").lower() in ("1", "true")


def verify_curator_role(x_curator_role: Optional[str] = Header(None)) -> str:
    """Verify that request includes curator role header (if required by config).
    
    Usage:
        @app.post("/curator/review")
        async def review(payload, role = Depends(verify_curator_role)):
            ...
    """
    if not CURATOR_ROLE_REQUIRED:
        return x_curator_role or "unknown"
    
    if not x_curator_role:
        raise HTTPException(status_code=403, detail="Curator role header required")
    
    if x_curator_role != "curator":
        raise HTTPException(status_code=403, detail="Curator role required for this operation")
    
    return x_curator_role


async def cleanup_expired_curator_reviews(db):
    """Delete curator reviews older than retention period (for data minimization).
    
    Call periodically or on-demand via admin endpoint.
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=CURATOR_RETENTION_DAYS)
    reviews_coll = db['curator_reviews']
    
    result = await reviews_coll.delete_many({'created_at': {'$lt': cutoff_date}})
    deleted_count = result.deleted_count
    
    return {
        "status": "ok",
        "deleted_count": deleted_count,
        "cutoff_date": cutoff_date.isoformat(),
        "retention_days": CURATOR_RETENTION_DAYS
    }


def log_curator_access(curator_id: str, action: str, resource_id: str):
    """Create audit log entry for curator access (optional, for future audit tables).
    
    Args:
        curator_id: Curator identity
        action: "view_export_preview", "view_nodes", "submit_review", etc.
        resource_id: job_id, node_id, etc.
    """
    # TODO: implement curator_access_log collection if needed for compliance
    pass
