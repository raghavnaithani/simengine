import os
import asyncio
import threading
from motor.motor_asyncio import AsyncIOMotorClient
from backend.app.utils.logger import append_log

MONGO_URL = os.getenv('MONGO_URL', 'mongodb://127.0.0.1:27017')
DB_NAME = os.getenv('MONGO_DB', 'simengine_db')

_mongo_client = None
_mongo_client_loop_id = None
_mongo_lock = threading.Lock()

async def get_database():
	"""Return a shared async Motor database instance.

	A process-wide singleton avoids creating a new client on each request,
	which can cause connection churn and intermittent topology timeouts.
	"""
	global _mongo_client
	global _mongo_client_loop_id

	current_loop = asyncio.get_running_loop()
	current_loop_id = id(current_loop)

	if _mongo_client is None or _mongo_client_loop_id != current_loop_id:
		with _mongo_lock:
			if _mongo_client is not None and _mongo_client_loop_id != current_loop_id:
				try:
					_mongo_client.close()
				except Exception:
					pass
			if _mongo_client is None or _mongo_client_loop_id != current_loop_id:
				append_log(f"[DB] Connecting to MongoDB at {MONGO_URL}")
				_mongo_client = AsyncIOMotorClient(MONGO_URL)
				_mongo_client_loop_id = current_loop_id
				append_log("[DB] Connected to MongoDB")

	return _mongo_client[DB_NAME]

async def close_mongo_connection():
	global _mongo_client
	global _mongo_client_loop_id

	if _mongo_client is not None:
		_mongo_client.close()
		_mongo_client = None
		_mongo_client_loop_id = None
		append_log("[DB] MongoDB connection closed")
