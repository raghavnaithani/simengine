from pathlib import Path
import os
import sys

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from urllib.parse import urlparse

from backend.app.utils.logger import append_log, record_event
from backend.app.database.connection import close_mongo_connection, get_database
from backend.app.engines.scraper import ContextBuilder
from backend.app.engines.reasoner import ReasoningEngine
from backend.app.engines.simulation import SimulationEngine
from backend.app.utils.jobs import create_job, update_job, get_job
from backend.app.config import apply_profile_env_vars, get_config
from backend.app.utils.quality import summarize_nodes_for_telemetry
from backend.app.utils.concurrency import get_concurrency_manager
from backend.app.models.schemas import CuratorReview
from backend.app.utils.curator_security import (
    verify_curator_role,
    cleanup_expired_curator_reviews,
    log_curator_access,
)
from backend.app.utils.ttl_manager import get_ttl_manager
import asyncio
from typing import Dict, Any, Optional, Literal
import traceback
from bson import ObjectId
from datetime import datetime, timezone
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


class IngestPayload(BaseModel):
    query: str
    top_k: int = 5


class CuratorReviewPayload(BaseModel):
    node_id: str
    session_id: Optional[str] = None
    curator: str = "curator"
    action: Literal["approve", "reject", "edit"]
    reason: str
    updates: Dict[str, Any] = Field(default_factory=dict)


def _sanitize_mongo_value(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_sanitize_mongo_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_mongo_value(item) for key, item in value.items()}
    return value


async def _normalize_job_result(job: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(job.get('result') or {})
    if result:
        return _sanitize_mongo_value(result)

    payload = job.get('payload') or {}
    session_id = payload.get('session_id')
    if not session_id:
        return {}

    normalized: Dict[str, Any] = {'session_id': session_id}

    try:
        db = await get_database()
        session_doc = await db['sessions'].find_one({'session_id': session_id})
        if session_doc:
            node_id = session_doc.get('root_node_id') or session_doc.get('current_node_id')
            if node_id:
                normalized['node_id'] = node_id
    except Exception:
        pass

    return _sanitize_mongo_value(normalized)


@app.on_event("shutdown")
async def shutdown_event():
    # Stop TTL pruning scheduler
    ttl_manager = await get_ttl_manager()
    await ttl_manager.stop_pruning_scheduler()
    
    # Close MongoDB connection
    await close_mongo_connection()


@app.on_event("startup")
async def startup_event():
    apply_profile_env_vars()
    record_event(level="INFO", action="startup", message=f"Backend startup (model={MODEL_NAME}, ollama_url={OLLAMA_URL})")
    
    # Start TTL pruning scheduler (per project guide: scheduled pruning)
    ttl_manager = await get_ttl_manager()
    await ttl_manager.start_pruning_scheduler()


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
        cleaned['result'] = await _normalize_job_result(job)
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
    elif typ == 'ingest':
        asyncio.create_task(_run_ingest_job(job_id))
    else:
        raise HTTPException(status_code=400, detail=f'unknown job type: {typ}')
    return {'job_id': job_id, 'status': 'requeued'}


@app.get('/jobs/{job_id}/logs')
async def job_logs(job_id: str, limit: int = 50):
    """Return raw model responses and stored logs for a given job."""
    try:
        db = await get_database()
        coll = db['model_responses']
        total_count = await coll.count_documents({'job_id': job_id})
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

        return {'count': total_count, 'returned_count': len(docs), 'logs': [_clean(d) for d in docs]}
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


@app.post('/ingest/start')
async def ingest_start(payload: IngestPayload, background_tasks: BackgroundTasks):
    """Start a Deep RAG ingestion job in the background."""
    try:
        job = await create_job('ingest', payload.model_dump())
        background_tasks.add_task(_run_ingest_job, job['job_id'])
        return {'job_id': job['job_id'], 'status': 'queued'}
    except Exception as e:
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


@app.get('/jobs/{job_id}/quality')
async def get_job_quality(job_id: str):
    """Get quality metrics for a specific job.
    
    Returns:
    {
        "job_id": "...",
        "quality_level": "SUCCESS|DEGRADED|FAILED",
        "metrics": {
            "total_nodes": 5,
            "valid_nodes": 4,
            "fallback_nodes": 1,
            "speculative_nodes": 1,
            "citation_rate": 0.8,
            "timeout_count": 0
        },
        "issues": ["LOW_CITATION_RATE"],
        "timestamp": "2026-01-22T..."
    }
    """
    try:
        db = await get_database()
        
        # Get the job document
        job = await db['jobs'].find_one({"job_id": job_id})
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Get result nodes from the job
        result = job.get('result') or {}
        session_id = result.get('session_id') or (job.get('payload') or {}).get('session_id')
        
        if not session_id:
            raise HTTPException(status_code=404, detail=f"No session found for job {job_id}")
        
        # Get nodes for this session
        nodes = await db['decision_nodes'].find({"session_id": session_id}).to_list(length=None)
        
        # Analyze quality metrics
        total_nodes = len(nodes)
        valid_nodes = 0
        fallback_nodes = 0
        speculative_nodes = 0
        cited_nodes = 0
        issues = []
        
        for node in nodes:
            has_citations = node.get('source_citations') and len(node['source_citations']) > 0
            is_speculative = node.get('speculative', False)
            
            if has_citations:
                valid_nodes += 1
                cited_nodes += 1
            else:
                fallback_nodes += 1
            
            if is_speculative:
                speculative_nodes += 1
        
        # Calculate metrics
        citation_rate = cited_nodes / total_nodes if total_nodes > 0 else 0
        fallback_ratio = fallback_nodes / total_nodes if total_nodes > 0 else 0
        speculative_ratio = speculative_nodes / total_nodes if total_nodes > 0 else 0
        
        # Determine quality level
        quality_level = "SUCCESS"
        
        # Check for quality issues
        if fallback_ratio == 1.0:
            issues.append("FALLBACK_ONLY")
            quality_level = "DEGRADED"
        elif citation_rate < 0.6:
            issues.append("LOW_CITATION_RATE")
            quality_level = "DEGRADED"
        
        if speculative_ratio > 0.8:
            issues.append("MOSTLY_SPECULATIVE")
            if quality_level == "SUCCESS":
                quality_level = "DEGRADED"
        
        if total_nodes == 0:
            issues.append("NO_NODES_GENERATED")
            quality_level = "FAILED"
        
        # Check for timeout
        if job.get('status') in ['timeout', 'TIMEOUT']:
            issues.append("TIMEOUT")
            if quality_level == "SUCCESS":
                quality_level = "DEGRADED"
        
        return {
            "job_id": job_id,
            "quality_level": quality_level,
            "metrics": {
                "total_nodes": total_nodes,
                "valid_nodes": valid_nodes,
                "fallback_nodes": fallback_nodes,
                "speculative_nodes": speculative_nodes,
                "citation_rate": round(citation_rate, 2),
                "fallback_ratio": round(fallback_ratio, 2),
                "speculative_ratio": round(speculative_ratio, 2),
                "timeout_count": 0
            },
            "issues": issues,
            "timestamp": job.get('created_at', datetime.now(timezone.utc)).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        append_log(f"get_job_quality error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/metrics')
async def get_system_metrics():
    """Get system-wide quality metrics aggregated from all jobs.
    
    Returns basic health metrics even if Mongo is unavailable.
    """
    # Always include basic process health
    base_metrics = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "simengine-backend",
    }
    
    try:
        db = await get_database()
        
        # Get all completed jobs
        jobs = await db['jobs'].find({"status": {"$in": ["completed", "failed"]}}).to_list(length=None)
        
        total_jobs = len(jobs)
        success_jobs = 0
        degraded_jobs = 0
        failed_jobs = 0
        total_nodes = 0
        total_cited_nodes = 0
        
        for job in jobs:
            result = job.get('result') or {}
            session_id = result.get('session_id') or (job.get('payload') or {}).get('session_id')
            
            if not session_id:
                continue
            
            nodes = await db['decision_nodes'].find({"session_id": session_id}).to_list(length=None)
            total_nodes += len(nodes)
            
            # Count cited nodes
            cited_count = sum(1 for n in nodes if n.get('source_citations') and len(n['source_citations']) > 0)
            total_cited_nodes += cited_count
            
            # Determine job quality
            if job.get('status') == 'failed':
                failed_jobs += 1
            elif len(nodes) > 0:
                citation_rate = cited_count / len(nodes)
                if citation_rate >= 0.6 and len(nodes) >= 1:
                    success_jobs += 1
                else:
                    degraded_jobs += 1
            else:
                degraded_jobs += 1
        
        overall_citation_rate = total_cited_nodes / total_nodes if total_nodes > 0 else 0
        overall_pass_rate = (success_jobs / total_jobs * 100) if total_jobs > 0 else 0
        
        return {
            **base_metrics,
            "total_jobs": total_jobs,
            "success_jobs": success_jobs,
            "degraded_jobs": degraded_jobs,
            "failed_jobs": failed_jobs,
            "overall_pass_rate": round(overall_pass_rate, 1),
            "overall_citation_rate": round(overall_citation_rate, 2),
            "total_nodes_generated": total_nodes,
            "average_nodes_per_job": round(total_nodes / total_jobs, 1) if total_jobs > 0 else 0,
            "database": "connected",
        }
    except Exception as e:
        # Mongo unavailable: return basic health metrics instead of failing
        append_log(f"get_system_metrics: Mongo unavailable, returning basic metrics: {str(e)}")
        return {
            **base_metrics,
            "database": "unavailable",
            "note": "Mongo connection failed; returning basic health status only",
            "error": str(e)[:100],
        }


@app.get('/metrics/summary')
async def get_metrics_summary(limit: int = 20):
    """Get summary of recent jobs' telemetry metrics (V2 feature).
    
    Returns averages and trends from last N jobs.
    """
    try:
        from backend.app.database.telemetry import TelemetryCollector
        summary = await TelemetryCollector.get_metrics_summary(limit)
        return summary
    except Exception as e:
        append_log(f"get_metrics_summary error: {str(e)}")
        # Graceful fallback if telemetry not available
        return {"status": "telemetry_unavailable", "error": str(e)}


@app.get('/metrics/citations')
async def get_citation_metrics():
    """Get citation-specific statistics from recent jobs (V2 feature)."""
    try:
        from backend.app.database.telemetry import TelemetryCollector
        stats = await TelemetryCollector.get_citation_stats()
        db = await get_database()
        nodes = await db['decision_nodes'].find(
            {},
            {
                'source_citations': 1,
                'citation_provenance': 1,
                'citation_quality_score': 1,
                'citation_coverage': 1,
                'created_at': 1,
            },
        ).sort('created_at', -1).limit(500).to_list(length=500)

        node_coverage_values = []
        entry_quality_values = []
        matched_entries = 0
        total_entries = 0
        completeness_values = []

        for node in nodes:
            node_coverage = node.get('citation_coverage')
            if isinstance(node_coverage, (int, float)):
                node_coverage_values.append(float(node_coverage))

            node_completeness = node.get('citation_provenance_completeness')
            if isinstance(node_completeness, (int, float)):
                completeness_values.append(float(node_completeness))

            provenance = node.get('citation_provenance') or []
            for entry in provenance:
                total_entries += 1
                if entry.get('matched'):
                    matched_entries += 1
                if isinstance(entry.get('citation_quality_score'), (int, float)):
                    entry_quality_values.append(float(entry.get('citation_quality_score')))

        stats['provenance_sample_nodes'] = len(nodes)
        stats['average_citation_coverage'] = round(
            sum(node_coverage_values) / len(node_coverage_values), 3
        ) if node_coverage_values else 0.0
        stats['average_citation_quality_score'] = round(
            sum(entry_quality_values) / len(entry_quality_values), 3
        ) if entry_quality_values else 0.0
        stats['average_provenance_completeness'] = round(
            sum(completeness_values) / len(completeness_values), 3
        ) if completeness_values else stats['provenance_match_rate']
        stats['provenance_match_rate'] = round(
            matched_entries / total_entries, 3
        ) if total_entries else 0.0
        return stats
    except Exception as e:
        append_log(f"get_citation_metrics error: {str(e)}")
        return {"status": "telemetry_unavailable", "error": str(e)}


@app.get('/export/preview')
async def export_preview(minimum_quality: float = 0.8, min_citation_rate: float = 0.8, limit: int = 20):
    """Preview exportable training records without writing files.

    Returns a count and a small sample of candidate records matching thresholds.
    """
    try:
        db = await get_database()
        # DB-side quality filter then in-process citation coverage filter
        candidates = await db['decision_nodes'].find({'quality_score': {'$gte': minimum_quality}}).sort('quality_score', -1).to_list(length=None)

        exported = []
        for node in candidates:
            citation_coverage = node.get('citation_coverage')
            if citation_coverage is None:
                citation_coverage = 1.0 if node.get('source_citations') else 0.0
            if float(citation_coverage) < float(min_citation_rate):
                continue

            exported.append({
                'id': node.get('id'),
                'title': node.get('title'),
                'quality_score': node.get('quality_score', 0.0),
                'citation_coverage': float(citation_coverage),
                'speculative': bool(node.get('speculative', False)),
            })

        sample = exported[: max(0, int(limit))]
        return {'candidates': len(exported), 'sample': sample}
    except Exception as e:
        append_log(f"export_preview error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/metrics/dashboard')
async def get_metrics_dashboard(limit: int = 20):
    """Get a combined quality, grounding, performance, and alert dashboard."""
    try:
        from backend.app.database.telemetry import TelemetryCollector

        return await TelemetryCollector.get_dashboard_summary(limit)
    except Exception as e:
        append_log(f"get_metrics_dashboard error: {str(e)}")
        return {"status": "telemetry_unavailable", "error": str(e)}


@app.get('/experiments/prompt-ab-test')
async def get_prompt_ab_experiment(limit: int = 100):
    """Summarize the prompt A/B experiment across recent decision nodes."""
    try:
        from backend.app.experiments.prompt_ab_test import summarize_prompt_experiment

        db = await get_database()
        nodes = await db['decision_nodes'].find({}).sort('created_at', -1).limit(limit).to_list(length=limit)
        return summarize_prompt_experiment(nodes, minimum_sample_size=50)
    except Exception as e:
        append_log(f"get_prompt_ab_experiment error: {str(e)}")
        return {"status": "experiment_unavailable", "error": str(e)}


@app.get('/curator/reviews')
async def list_curator_reviews(
    limit: int = 20,
    node_id: Optional[str] = None,
    role: str = Depends(verify_curator_role),
):
    """List recent curator review actions for dashboard and audit views."""
    try:
        db = await get_database()
        coll = db['curator_reviews']
        query: Dict[str, Any] = {}
        if node_id:
            query['node_id'] = node_id

        docs = await coll.find(query).sort('created_at', -1).limit(limit).to_list(length=limit)
        log_curator_access(role, 'list_reviews', node_id or 'all')
        return {
            'count': len(docs),
            'reviews': [_sanitize_mongo_value(doc) for doc in docs],
        }
    except Exception as e:
        append_log(f"list_curator_reviews error: {str(e)}")
        return {"status": "telemetry_unavailable", "error": str(e)}


@app.post('/curator/review')
async def record_curator_review(
    payload: CuratorReviewPayload,
    role: str = Depends(verify_curator_role),
):
    """Record curator approve/reject/edit actions and persist an audit trail."""
    try:
        db = await get_database()
        node_coll = db['decision_nodes']
        review_coll = db['curator_reviews']

        node = await node_coll.find_one({'id': payload.node_id})
        if not node:
            raise HTTPException(status_code=404, detail='node not found')

        node_before = _sanitize_mongo_value(node)
        node_after = dict(node_before)
        updates = dict(payload.updates or {})

        node_update: Dict[str, Any] = {
            'curator_review_status': payload.action,
            'curator_review_reason': payload.reason,
            'curator_reviewed_by': payload.curator,
            'curator_reviewed_at': datetime.now(timezone.utc),
        }

        if payload.action == 'edit':
            if not updates:
                raise HTTPException(status_code=400, detail='updates are required for edit reviews')
            node_update.update(updates)
            node_after.update(updates)

        await node_coll.update_one({'id': payload.node_id}, {'$set': node_update})

        curator_identity = payload.curator if payload.curator and payload.curator != 'curator' else role

        review = CuratorReview(
            node_id=payload.node_id,
            session_id=payload.session_id or node_before.get('session_id'),
            curator=curator_identity,
            action=payload.action,
            reason=payload.reason,
            before=node_before,
            after=node_after,
        ).model_dump()

        await review_coll.insert_one(review)
        log_curator_access(curator_identity, 'submit_review', payload.node_id)
        return {
            'status': 'ok',
            'review': _sanitize_mongo_value(review),
        }
    except HTTPException:
        raise
    except Exception as e:
        append_log(f"record_curator_review error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/curator/reviews/cleanup')
async def cleanup_curator_reviews(role: str = Depends(verify_curator_role)):
    """Run local retention cleanup for curator review audit records."""
    try:
        db = await get_database()
        result = await cleanup_expired_curator_reviews(db)
        log_curator_access(role, 'cleanup_reviews', 'curator_reviews')
        return result
    except Exception as e:
        append_log(f"cleanup_curator_reviews error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/session/{session_id}/scraping/enable')
async def enable_session_scraping(session_id: str):
    """Enable web scraping for a session (project guide: user opt-in control).
    
    Per project guide: "provide a toggle to opt out of web scraping if user prefers privacy/legal safety"
    """
    try:
        db = await get_database()
        sessions_coll = db['sessions']
        
        result = await sessions_coll.update_one(
            {'session_id': session_id},
            {'$set': {'scraping_enabled': True, 'scraping_updated_at': datetime.now(timezone.utc)}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        record_event(
            level="INFO",
            action="session.scraping_enabled",
            message=f"Web scraping enabled for session {session_id}"
        )
        
        return {'session_id': session_id, 'scraping_enabled': True, 'status': 'ok'}
    except HTTPException:
        raise
    except Exception as e:
        append_log(f"enable_session_scraping error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/session/{session_id}/scraping/disable')
async def disable_session_scraping(session_id: str):
    """Disable web scraping for a session (privacy/legal safety preference).
    
    Per project guide: "provide a toggle to opt out of web scraping if user prefers privacy/legal safety"
    """
    try:
        db = await get_database()
        sessions_coll = db['sessions']
        
        result = await sessions_coll.update_one(
            {'session_id': session_id},
            {'$set': {'scraping_enabled': False, 'scraping_updated_at': datetime.now(timezone.utc)}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        record_event(
            level="INFO",
            action="session.scraping_disabled",
            message=f"Web scraping disabled for session {session_id}"
        )
        
        return {'session_id': session_id, 'scraping_enabled': False, 'status': 'ok'}
    except HTTPException:
        raise
    except Exception as e:
        append_log(f"disable_session_scraping error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/session/{session_id}/scraping/status')
async def get_session_scraping_status(session_id: str):
    """Get web scraping status for a session."""
    try:
        db = await get_database()
        sessions_coll = db['sessions']
        
        session = await sessions_coll.find_one({'session_id': session_id})
        
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        scraping_enabled = session.get('scraping_enabled', True)  # Default to enabled
        
        return {
            'session_id': session_id,
            'scraping_enabled': scraping_enabled,
            'scraping_updated_at': session.get('scraping_updated_at')
        }
    except HTTPException:
        raise
    except Exception as e:
        append_log(f"get_session_scraping_status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _record_v2_job_telemetry(job_id: str, session_id: Optional[str], started_at: Optional[datetime] = None) -> None:
    """Persist telemetry for a completed job using its generated nodes."""
    try:
        from backend.app.database.telemetry import TelemetryCollector

        db = await get_database()
        nodes = await db['decision_nodes'].find({'job_id': job_id}).to_list(length=None)
        if not nodes and session_id:
            nodes = await db['decision_nodes'].find({'session_id': session_id}).to_list(length=None)

        metrics = summarize_nodes_for_telemetry(nodes)

        if started_at is None:
            job = await get_job(job_id)
            started_at = job.get('created_at') if job else None

        latency_ms = 0.0
        if isinstance(started_at, datetime):
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            latency_ms = max(0.0, (datetime.now(timezone.utc) - started_at).total_seconds() * 1000.0)

        await TelemetryCollector.record_job_metrics(
            job_id=job_id,
            citation_rate=metrics['citation_rate'],
            diversity_score=metrics['diversity_score'],
            quality_score=metrics['quality_score'],
            latency_ms=latency_ms,
            error_rate=metrics['error_rate'],
            alternatives_count=metrics['alternatives_count'],
            title_novelty=metrics['title_novelty'],
            risk_specificity=metrics['risk_specificity'],
        )
    except Exception as e:
        append_log(f"_record_v2_job_telemetry error: {str(e)}")


async def _run_start_job(job_id: str):
    append_log(f"_run_start_job: starting {job_id}")
    
    # V4 WS1: Wait for concurrency slot
    concurrency_mgr = get_concurrency_manager()
    job_poll_timeout = int(os.getenv("JOB_POLL_TIMEOUT_SECONDS", "120"))
    
    slot_acquired = await concurrency_mgr.wait_for_slot(job_id, timeout_seconds=job_poll_timeout)
    if not slot_acquired:
        await update_job(job_id, 'failed', error='Concurrency limit timeout: could not acquire job slot')
        append_log(f"_run_start_job: failed {job_id} - concurrency timeout")
        return
    
    try:
        job = await get_job(job_id)
        payload: Dict[str, Any] = job.get('payload', {})
        started_at = job.get('created_at')
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
                    'created_at': datetime.now(timezone.utc)
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
        await _record_v2_job_telemetry(job_id, session_id, started_at=started_at)
    except Exception as e:
        err = traceback.format_exc()
        await update_job(job_id, 'failed', error=str(e) or err)
        append_log(f"_run_start_job: failed {job_id} error={err}")
    finally:
        # V4 WS1: Release concurrency slot and promote next queued job
        next_job_id = await concurrency_mgr.release_slot(job_id)
        if next_job_id:
            append_log(f"_run_start_job: promoting queued job {next_job_id}")
            asyncio.create_task(_run_start_job(next_job_id))


async def _run_branch_job(job_id: str):
    append_log(f"_run_branch_job: starting {job_id}")
    try:
        job = await get_job(job_id)
        payload: Dict[str, Any] = job.get('payload', {})
        started_at = job.get('created_at')
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
        await _record_v2_job_telemetry(job_id, session_id, started_at=started_at)
    except Exception as e:
        err = traceback.format_exc()
        await update_job(job_id, 'failed', error=str(e) or err)
        append_log(f"_run_branch_job: failed {job_id} error={err}")


async def _run_ingest_job(job_id: str):
    append_log(f"_run_ingest_job: starting {job_id}")
    try:
        job = await get_job(job_id)
        payload: Dict[str, Any] = job.get('payload', {})
        query = payload.get('query', '').strip()
        top_k = int(payload.get('top_k', 5))

        if not query:
            await update_job(job_id, 'failed', error='query is required')
            return

        await update_job(job_id, 'running')

        builder = ContextBuilder()
        result = await builder.build_knowledge_base(query=query, top_k=top_k)
        inserted_ids = result.get('inserted_ids', [])

        await update_job(
            job_id,
            'completed',
            result={
                'query': query,
                'inserted_count': len(inserted_ids),
                'total_chunks': result.get('total_chunks', 0),
            },
        )
    except Exception as e:
        err = traceback.format_exc()
        await update_job(job_id, 'failed', error=str(e) or err)
        append_log(f"_run_ingest_job: failed {job_id} error={err}")