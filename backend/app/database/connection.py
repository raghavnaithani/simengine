import os
from motor.motor_asyncio import AsyncIOMotorClient
from app.utils.logger import append_log

MONGO_URL = os.getenv('MONGO_URL', 'mongodb://mongo:27017')
DB_NAME = os.getenv('MONGO_DB', 'simengine_db')

class Database:
	client: AsyncIOMotorClient = None

db = Database()

async def get_database():
	"""Return an async Motor database instance. Connects on first call."""
	if db.client is None:
		append_log(f"[DB] Connecting to MongoDB at {MONGO_URL}")
		db.client = AsyncIOMotorClient(MONGO_URL)
		append_log("[DB] Connected to MongoDB")
	return db.client[DB_NAME]

async def close_mongo_connection():
	if db.client:
		db.client.close()
		append_log("[DB] MongoDB connection closed")
