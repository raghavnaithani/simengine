import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture(autouse=True)
def mock_motor_operations():
    with patch("motor.motor_asyncio.AsyncIOMotorClient") as MockMotorClient:
        mock_client = MockMotorClient.return_value
        mock_client["simengine_db"].return_value = AsyncMock()
        yield


@pytest.fixture(autouse=True)
def patch_upsert_chunk_global():
    """FIX #15: CHUNK_EMBEDDING_REAL - Patch upsert_chunk globally so no DB embeds happen."""
    with patch("backend.app.database.vector_store.upsert_chunk") as mock_upsert:
        mock_upsert.return_value = None
        yield mock_upsert


@pytest.fixture(autouse=True)
def patch_query_similar_chunks_global():
    """Patch query_similar_chunks to return empty list by default (triggers fallback in tests)."""
    with patch("backend.app.database.vector_store.query_similar_chunks") as mock_query:
        mock_query.return_value = []
        yield mock_query