from typing import List, Dict, Any, Optional
import hashlib
import math
import asyncio
from backend.app.database.connection import get_database
from backend.app.utils.logger import record_event
import os

try:
    from sentence_transformers import SentenceTransformer
    _EMB_MODEL = None
    EMBEDDER_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    _EMB_MODEL = None
    EMBEDDER_AVAILABLE = False

COLLECTION = 'global_context'
# all-MiniLM-L6-v2 produces 384 dimensions. Use this as the canonical default.
DEFAULT_DIM = 384

# Update MongoDB connection string to use localhost
MONGO_URL = os.getenv('MONGO_URL', 'mongodb://127.0.0.1:27017')


def _hash_embedding(text: str, dim: int = DEFAULT_DIM) -> List[float]:
    """Deterministic fallback embedding. Returns a list of `dim` floats in range [0,1]."""
    out: List[float] = []
    for i in range(dim):
        # salt the hash with the index to create unique values per dimension
        h = hashlib.sha256((text + f"::{i}").encode('utf-8')).digest()
        val = int.from_bytes(h[:4], 'big', signed=False)
        out.append((val / 2**32))
    return out


def _ensure_embedder():
    global _EMB_MODEL
    if not EMBEDDER_AVAILABLE:
        return None
    if _EMB_MODEL is None:
        # This model is 80MB and fast. Perfect for CPU/Edge.
        _EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMB_MODEL


def embed_text(text: str, dim: int = DEFAULT_DIM) -> List[float]:
    """Return embedding for text. Uses SentenceTransformer (384 dim) when available, else fallback."""
    if EMBEDDER_AVAILABLE:
        try:
            model = _ensure_embedder()
            if model is not None:
                vec = model.encode(text)
                return [float(x) for x in vec.tolist()] if hasattr(vec, 'tolist') else [float(x) for x in vec]
        except Exception as e:
            print(f"[WARN] Embedder failed, using fallback: {e}")

    # Fallback: creates a fake vector of the CORRECT dimension (384)
    return _hash_embedding(text, dim=dim)


def _cosine(a: List[float], b: List[float]) -> float:
    # simple cosine similarity without numpy
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


async def upsert_chunk(chunk: Dict[str, Any], embed_dim: int = DEFAULT_DIM):
    """Compute embedding and upsert chunk into the `global_context` collection."""
    text = chunk.get('content', '')
    # This will now always return DEFAULT_DIM dimensions (either real or fake)
    emb = embed_text(text, dim=embed_dim)
    chunk['embedding'] = emb
    db = await get_database()
    coll = db[COLLECTION]
    await coll.update_one({'id': chunk.get('id')}, {'$set': chunk}, upsert=True)


async def query_similar_chunks(query: str = None, query_embedding: Optional[List[float]] = None, k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k similar chunks to the query (by text or by embedding)."""
    if query_embedding is None:
        if query is None:
            return []
        query_embedding = embed_text(query, dim=DEFAULT_DIM)

    db = await get_database()
    coll = db[COLLECTION]

    # 1. Fetch candidates (Simple Scan)
    docs = await coll.find({'embedding': {'$exists': True}}).limit(200).to_list(length=200)

    scored: List[Dict[str, Any]] = []
    for d in docs:
        emb = d.get('embedding')
        if not emb:
            continue
        # Compute cosine similarity
        score = _cosine(query_embedding, emb)
        d['_similarity_score'] = float(score)
        scored.append(d)

    # Sort by highest score first
    scored.sort(key=lambda x: x.get('_similarity_score', 0.0), reverse=True)

    # Log top-k results
    record_event(level='INFO', action='vector_search_results', message=f"Vector search results: {len(scored[:k])} chunks", details={'query': query, 'results': scored[:k]})

    return scored[:k]


async def ensure_text_index():
    """Ensure the MongoDB collection has the required text index."""
    db = await get_database()
    coll = db[COLLECTION]
    indexes = await coll.index_information()
    if "content_text_index" not in indexes:
        await coll.create_index([("content", "text")], name="content_text_index")
