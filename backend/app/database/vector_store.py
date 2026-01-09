from typing import List, Dict, Any
from app.database.connection import get_database

COLLECTION = 'global_context'


async def upsert_chunk(chunk: Dict[str, Any]):
    db = await get_database()
    coll = db[COLLECTION]
    await coll.update_one({'id': chunk.get('id')}, {'$set': chunk}, upsert=True)


async def query_similar_chunks(query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
    db = await get_database()
    coll = db[COLLECTION]
    docs = await coll.find().to_list(length=k)
    return docs
