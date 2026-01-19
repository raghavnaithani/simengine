from typing import Any, Dict, Optional
from datetime import datetime, timezone
import uuid
from backend.app.database.connection import get_database
from backend.app.utils.logger import append_log

COLLECTION = 'jobs'


async def create_job(job_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    db = await get_database()
    coll = db[COLLECTION]
    job = {
        'job_id': str(uuid.uuid4()),
        'type': job_type,
        'payload': payload,
        'status': 'queued',
        'created_at': datetime.now(timezone.utc),
        'updated_at': datetime.now(timezone.utc),
        'result': None,
        'error': None,
    }
    await coll.insert_one(job)
    append_log(f"create_job: created job {job['job_id']} type={job_type}")
    return job


async def update_job(job_id: str, status: str, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    db = await get_database()
    coll = db[COLLECTION]
    update = {'status': status, 'updated_at': datetime.now(timezone.utc)}
    if result is not None:
        update['result'] = result
    if error is not None:
        update['error'] = error
    await coll.update_one({'job_id': job_id}, {'$set': update})
    append_log(f"update_job: job {job_id} -> {status}")


async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    db = await get_database()
    coll = db[COLLECTION]
    job = await coll.find_one({'job_id': job_id})
    return job
