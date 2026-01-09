"""ContextBuilder (Deep RAG) minimal implementation.
Note: heavy external deps (sentence-transformers, crawl4ai) are imported lazily to avoid startup failures.
This module provides placeholders that can be extended to full Deep RAG ingestion.
"""
from typing import List, Dict, Any
import uuid
from datetime import datetime
from app.utils.logger import append_log
from app.database.connection import get_database

try:
    from sentence_transformers import SentenceTransformer
    EMB_MODEL_AVAILABLE = True
except Exception:
    EMB_MODEL_AVAILABLE = False


class ContextBuilder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._embedder = None

    async def build_knowledge_base(self, query: str):
        """Create a simulated KnowledgeChunk and insert into Mongo for verification.

        This implements a minimal ingestion path used for testing Deep RAG plumbing.
        """
        append_log(f"build_knowledge_base called for query: {query}")
        db = await get_database()
        coll = db["global_context"]

        chunk = {
            "id": str(uuid.uuid4()),
            "content": f"Simulated knowledge about '{query}': sample content for testing.",
            "source_url": "http://sim.test/1",
            "source_title": "Simulated Source",
            "chunk_index": 0,
            "verification_status": "verified",
            "created_at": datetime.utcnow()
        }

        res = await coll.insert_one(chunk)
        append_log(f"Inserted test chunk with _id={res.inserted_id}")
        return {"status": "ok", "inserted_id": str(res.inserted_id), "preview": chunk["content"]}

    async def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        db = await get_database()
        coll = db["global_context"]
        docs = await coll.find().to_list(length=k)
        return docs

    async def get_context_for_reasoner(self, query: str, k: int = 5, min_confidence: float = 0.85):
        chunks = await self.retrieve_relevant_chunks(query, k=k)
        context_confidence = 0.0
        if chunks:
            sims = [c.get('similarity_score', 0.0) for c in chunks]
            context_confidence = max(sims) if sims else 0.0
        return {"chunks": chunks, "context_confidence": context_confidence}
