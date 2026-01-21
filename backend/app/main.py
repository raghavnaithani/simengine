from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from urllib.parse import urlparse
import os

from backend.app.utils.logger import append_log, record_event
from backend.app.database.connection import close_mongo_connection, get_database
from backend.app.engines.scraper import ContextBuilder
from backend.app.engines.reasoner import ReasoningEngine
from backend.app.engines.simulation import SimulationEngine
from backend.app.utils.jobs import create_job, update_job, get_job
import asyncio
from typing import Dict, Any
import traceback
from bson import ObjectId
from datetime import datetime
from fastapi import Request
import pytz

app = FastAPI(title="Decision Graph Simulator - Backend (v1.2)")

# Allow Frontend to talk to Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from Docker Environment
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "phi3")

# Derive a base URL from OLLAMA_URL for health checks (preserves host/port)
_parsed = urlparse(OLLAMA_URL)
OLLAMA_BASE = f"{_parsed.scheme}://{_parsed.netloc}"


class PromptRequest(BaseModel):
    prompt: str


class StartSimulationPayload(BaseModel):
    prompt: str
    mode: str = "Analytical"
    persona: str = "Skeptical Analyst"
    simulate_steps: int = 3
    seed: Optional[int] = None  # Optional seed for reproducibility


class BranchPayload(BaseModel):
    session_id: str
    parent_node_id: str
    action: str
    persona: str = "Optimistic Founder"
    seed: Optional[int] = None  # Optional seed for reproducibility


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()


@app.on_event("startup")
async def startup_event():
    record_event(level="INFO", action="startup", message=f"Backend startup (model={MODEL_NAME}, ollama_url={OLLAMA_URL})")


@app.get("/")
async def root():
    return {"status": "Decision Graph Simulator backend (v1.2)", "model": MODEL_NAME}



@app.get("/health")
async def health():
    """Real Health Check: Pings the Ollama container to see if the Brain is alive."""
    ollama_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/")
            if resp.status_code == 200:
                ollama_status = "online"
    except Exception as e:
        ollama_status = f"offline ({str(e)})"

    return {"status": "ok", "ollama": ollama_status, "model_target": MODEL_NAME}


@app.post("/test/generate")
async def test_generate(payload: PromptRequest):
    """Task A: send a short prompt to Ollama and return the raw response."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "prompt": payload.prompt, "stream": False},
            )
            # record a concise test event
            try:
                resp_json = response.json()
                preview = str(resp_json)[:400]
            except Exception:
                resp_json = None
                preview = f"status={response.status_code}"

            record_event(level="INFO", action="test.generate", message=f"prompt: {payload.prompt}", details={"status_code": response.status_code, "preview": preview})
            return resp_json
    except Exception as e:
        record_event(level="ERROR", action="test.generate.error", message=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test/scrape")
async def test_scrape(payload: PromptRequest):
    """Task B: create a test KnowledgeChunk via ContextBuilder and write to Mongo."""
    try:
        builder = ContextBuilder()
        result = await builder.build_knowledge_base(payload.prompt)
        # Log result (concise)
        record_event(level="INFO", action="test.scrape", message=f"prompt: {payload.prompt}", details={"result_preview": str(result)[:400]})
        return result
    except Exception as e:
        record_event(level="ERROR", action="test.scrape.error", message=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/start")
async def simulate_start(payload: StartSimulationPayload):
    """Start a simulation session. This creates a session job and begins world-building in background."""
    try:
        # create session record minimal
        db = await get_database()
        sessions = db['sessions']
        
        # Generate seed if not provided (for reproducibility)
        import random
        import time
        session_seed = payload.seed if payload.seed is not None else int(time.time() * 1000000) % (2**31)
        
        session = {
            'session_id': str(payload.prompt)[:8] + '_' + str(int(asyncio.get_event_loop().time())),
            'prompt': payload.prompt,
            'mode': payload.mode,
            'persona': payload.persona,
            'seed': session_seed,  # Store seed for reproducibility
            'created_at': None,
        }
        await sessions.insert_one(session)

        # Add session_id to job payload for background worker
        job_payload = payload.dict()
        job_payload['session_id'] = session['session_id']
        job = await create_job('start', job_payload)

        # schedule background worker
        asyncio.create_task(_run_start_job(job['job_id']))

        return {'session_id': session['session_id'], 'job_id': job['job_id'], 'status': 'started'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/branch")
async def simulate_branch(payload: BranchPayload):
    try:
        job = await create_job('branch', payload.dict())
        asyncio.create_task(_run_branch_job(job['job_id']))
        return {'job_id': job['job_id'], 'status': 'queued'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
async def jobs_get(job_id: str):
    try:
        job = await get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail='job not found')

        def _clean(d):
            for k, v in list(d.items()):
                if isinstance(v, ObjectId):
                    d[k] = str(v)
                elif isinstance(v, datetime):
                    d[k] = v.isoformat()
                elif isinstance(v, dict):
                    d[k] = _clean(v)
            return d

        cleaned = _clean(dict(job))
        return cleaned
    except HTTPException:
        raise
    except Exception as e:
        append_log(f"jobs_get error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
async def jobs_list(limit: int = 20):
    """Return recent jobs for debugging (most recent first)."""
    try:
        db = await get_database()
        coll = db['jobs']
        docs = await coll.find().sort('created_at', -1).to_list(length=limit)

        def _clean_doc(d):
            nd = {}
            for k, v in d.items():
                if isinstance(v, ObjectId):
                    nd[k] = str(v)
                elif isinstance(v, datetime):
                    nd[k] = v.isoformat()
                else:
                    nd[k] = v
            return nd

        cleaned = [_clean_doc(d) for d in docs]
        return {'count': len(cleaned), 'jobs': cleaned}
    except Exception as e:
        append_log(f"jobs_list error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs/{job_id}/retry")
async def jobs_retry(job_id: str):
    """Retry a failed job by re-queuing it and starting the background worker."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='job not found')
    if job.get('status') == 'running':
        raise HTTPException(status_code=400, detail='job already running')
    await update_job(job_id, 'queued')
    # re-dispatch based on type
    typ = job.get('type')
    if typ == 'start':
        asyncio.create_task(_run_start_job(job_id))
    elif typ == 'branch':
        asyncio.create_task(_run_branch_job(job_id))
    else:
        raise HTTPException(status_code=400, detail=f'unknown job type: {typ}')
    return {'job_id': job_id, 'status': 'requeued'}


@app.get('/jobs/{job_id}/logs')
async def job_logs(job_id: str, limit: int = 50):
    """Return raw model responses and stored logs for a given job."""
    try:
        db = await get_database()
        coll = db['model_responses']
        docs = await coll.find({'job_id': job_id}).sort('created_at', -1).to_list(length=limit)

        def _clean(d):
            nd = {}
            for k, v in d.items():
                if isinstance(v, ObjectId):
                    nd[k] = str(v)
                elif isinstance(v, datetime):
                    nd[k] = v.isoformat()
                else:
                    nd[k] = v
            return nd

        return {'count': len(docs), 'logs': [_clean(d) for d in docs]}
    except Exception as e:
        append_log(f"job_logs error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/nodes/{node_id}')
async def get_node(node_id: str):
    try:
        db = await get_database()
        coll = db['decision_nodes']
        node = await coll.find_one({'id': node_id})
        if not node:
            raise HTTPException(status_code=404, detail='node not found')
        # sanitize
        node['_id'] = str(node.get('_id'))
        return node
    except HTTPException:
        raise
    except Exception as e:
        append_log(f"get_node error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/log')
async def external_log(payload: Dict[str, Any]):
    """Accept external structured logs and append to project_log.txt.

    Expected JSON: { level?: str, action?: str, message?: str, details?: any }
    """
    try:
        lvl = payload.get('level', 'INFO')
        action = payload.get('action')
        message = payload.get('message')
        details = payload.get('details')
        record_event(level=lvl, action=action, message=message, details=details)
        return {'status': 'ok'}
    except Exception as e:
        record_event(level='ERROR', action='external_log.failed', message=str(e), details=payload)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/graph')
async def get_graph(session_id: str = None):
    """Get graph for a session or all nodes if no session_id provided."""
    try:
        if session_id:
            # Use SimulationEngine to get session-specific graph
            sim_engine = SimulationEngine()
            return await sim_engine.get_session_graph(session_id)
        else:
            # Fallback: return all nodes/edges (for backward compatibility)
            db = await get_database()
            nodes = await db['decision_nodes'].find().to_list(length=1000)
            edges = await db['edges'].find().to_list(length=1000)
            # sanitize ids
            for n in nodes:
                if '_id' in n:
                    n['_id'] = str(n['_id'])
            for e in edges:
                if '_id' in e:
                    e['_id'] = str(e['_id'])
            return {'nodes': nodes, 'edges': edges}
    except Exception as e:
        append_log(f"get_graph error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_start_job(job_id: str):
    append_log(f"_run_start_job: starting {job_id}")
    try:
        job = await get_job(job_id)
        payload: Dict[str, Any] = job.get('payload', {})
        prompt = payload.get('prompt', '')
        mode = payload.get('mode', 'Analytical')
        persona = payload.get('persona', 'Skeptical Analyst')
        simulate_steps = payload.get('simulate_steps', 3)
        seed = payload.get('seed')  # Get seed for reproducibility

        await update_job(job_id, 'running')

        # Get session_id from job payload (set by simulate_start endpoint)
        session_id = payload.get('session_id')
        if not session_id:
            # Fallback: try to find or create session
            db = await get_database()
            sessions_coll = db['sessions']
            session = await sessions_coll.find_one({'prompt': prompt}, sort=[('created_at', -1)])
            session_id = session['session_id'] if session else None
            
            if not session_id:
                # Create session if it doesn't exist
                session_id = str(prompt)[:8] + '_' + str(int(asyncio.get_event_loop().time()))
                await sessions_coll.insert_one({
                    'session_id': session_id,
                    'prompt': prompt,
                    'mode': mode,
                    'persona': persona,
                    'seed': seed,  # Store seed
                    'created_at': datetime.now(pytz.utc)
                })

        # Use SimulationEngine to build initial world
        sim_engine = SimulationEngine()
        result = await sim_engine.build_initial_world(
            prompt=prompt,
            session_id=session_id,
            mode=mode,
            persona=persona,
            num_steps=simulate_steps,
            job_id=job_id,
            seed=seed  # Pass seed for reproducibility
        )

        await update_job(job_id, 'completed', result={'node_id': result['root_node_id'], 'session_id': session_id})
    except Exception as e:
        err = traceback.format_exc()
        await update_job(job_id, 'failed', error=str(e) or err)
        append_log(f"_run_start_job: failed {job_id} error={err}")


async def _run_branch_job(job_id: str):
    append_log(f"_run_branch_job: starting {job_id}")
    try:
        job = await get_job(job_id)
        payload: Dict[str, Any] = job.get('payload', {})
        parent_id = payload.get('parent_node_id')
        action = payload.get('action')
        session_id = payload.get('session_id')
        persona = payload.get('persona', 'Optimistic Founder')
        seed = payload.get('seed')  # Get seed for reproducibility

        await update_job(job_id, 'running')

        # Use SimulationEngine to create branch
        sim_engine = SimulationEngine()
        result = await sim_engine.create_branch(
            parent_node_id=parent_id,
            action=action,
            session_id=session_id,
            persona=persona,
            job_id=job_id,
            seed=seed  # Pass seed for reproducibility
        )

        await update_job(job_id, 'completed', result={'node_id': result['node_id'], 'session_id': session_id})
    except Exception as e:
        err = traceback.format_exc()
        await update_job(job_id, 'failed', error=str(e) or err)
        append_log(f"_run_branch_job: failed {job_id} error={err}")