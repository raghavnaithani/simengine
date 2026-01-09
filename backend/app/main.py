from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

from app.utils.logger import append_log
from app.database.connection import close_mongo_connection
from app.engines.scraper import ContextBuilder

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


class PromptRequest(BaseModel):
    prompt: str


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()


@app.get("/")
async def root():
    return {"status": "Decision Graph Simulator backend (v1.2)", "model": MODEL_NAME}


@app.get("/health")
async def health():
    """Real Health Check: Pings the Ollama container to see if the Brain is alive."""
    ollama_status = "unknown"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://ollama:11434/")
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
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test/scrape")
async def test_scrape(payload: PromptRequest):
    """Task B: create a test KnowledgeChunk via ContextBuilder and write to Mongo."""
    try:
        builder = ContextBuilder()
        result = await builder.build_knowledge_base(payload.prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))