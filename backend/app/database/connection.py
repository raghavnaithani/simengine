import os
import asyncio
import contextvars
from motor.motor_asyncio import AsyncIOMotorClient
from backend.app.utils.logger import append_log

MONGO_URL = os.getenv('MONGO_URL', 'mongodb://127.0.0.1:27017')
DB_NAME = os.getenv('MONGO_DB', 'simengine_db')

# Context variable to store the MongoDB client for each event loop
client_context = contextvars.ContextVar("mongo_client", default=None)

async def get_database():
	"""Return an async Motor database instance. Connects on first call."""
	client = client_context.get()
	if client is None:
		append_log(f"[DB] Connecting to MongoDB at {MONGO_URL}")
		client = AsyncIOMotorClient(MONGO_URL)
		client_context.set(client)
		append_log("[DB] Connected to MongoDB")
	return client[DB_NAME]

async def close_mongo_connection():
	client = client_context.get()
	if client:
		client.close()
		append_log("[DB] MongoDB connection closed")
