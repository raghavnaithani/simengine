"""TTL Manager: Background scheduler for MongoDB document expiration.

Per project guide: "Set TTL = 30 days by default" and "scheduled pruning"

Handles:
- Periodic deletion of expired KnowledgeChunks
- Periodic deletion of expired sessions
- Expiration tracking via created_at + ttl_days
"""
import asyncio
from datetime import datetime, timedelta, timezone
from backend.app.database.connection import get_database
from backend.app.utils.logger import append_log, record_event
from typing import Dict, Any


class TTLManager:
    """Manages TTL-based document expiration in MongoDB."""
    
    def __init__(self, check_interval_minutes: int = 60):
        """Initialize TTL manager.
        
        Args:
            check_interval_minutes: How often to run pruning job (default 60 mins)
        """
        self.check_interval_minutes = check_interval_minutes
        self.is_running = False
        self._background_task = None
    
    async def start_pruning_scheduler(self):
        """Start the background TTL pruning scheduler.
        
        Runs periodically to delete expired documents:
        - KnowledgeChunks where created_at + ttl_days < now
        - Sessions where created_at + TTL (30 days default) < now
        """
        if self.is_running:
            return
        
        self.is_running = True
        record_event(
            level="INFO",
            action="ttl_manager.scheduler_start",
            message=f"TTL pruning scheduler started (check interval: {self.check_interval_minutes} minutes)"
        )
        
        # Create background task
        self._background_task = asyncio.create_task(self._pruning_loop())
    
    async def stop_pruning_scheduler(self):
        """Stop the background TTL pruning scheduler."""
        self.is_running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        record_event(
            level="INFO",
            action="ttl_manager.scheduler_stop",
            message="TTL pruning scheduler stopped"
        )
    
    async def _pruning_loop(self):
        """Main loop for periodic pruning.
        
        Runs every check_interval_minutes to delete expired documents.
        """
        while self.is_running:
            try:
                await asyncio.sleep(self.check_interval_minutes * 60)
                if self.is_running:
                    await self.prune_expired_documents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                append_log(f"ttl_manager loop error: {str(e)}")
                record_event(
                    level="ERROR",
                    action="ttl_manager.loop_error",
                    message=f"TTL pruning loop error: {str(e)}"
                )
    
    async def prune_expired_documents(self) -> Dict[str, Any]:
        """Delete expired documents from MongoDB.
        
        Per project guide: Delete KnowledgeChunks and sessions that have exceeded their TTL.
        
        Returns:
            Summary dict with counts of deleted documents by collection
        """
        try:
            db = await get_database()
            now = datetime.now(timezone.utc)
            
            # Prune KnowledgeChunks
            chunks_result = await self._prune_knowledge_chunks(db, now)
            
            # Prune sessions
            sessions_result = await self._prune_sessions(db, now)
            
            summary = {
                'timestamp': now.isoformat(),
                'knowledge_chunks_deleted': chunks_result['deleted_count'],
                'knowledge_chunks_checked': chunks_result['checked_count'],
                'sessions_deleted': sessions_result['deleted_count'],
                'sessions_checked': sessions_result['checked_count']
            }
            
            # Log summary if anything was deleted
            if chunks_result['deleted_count'] > 0 or sessions_result['deleted_count'] > 0:
                record_event(
                    level="INFO",
                    action="ttl_manager.pruning_complete",
                    message=f"TTL pruning completed: {chunks_result['deleted_count']} chunks, {sessions_result['deleted_count']} sessions deleted",
                    details=summary
                )
                append_log(f"TTL pruning: {chunks_result['deleted_count']} chunks, {sessions_result['deleted_count']} sessions deleted")
            
            return summary
            
        except Exception as e:
            append_log(f"prune_expired_documents error: {str(e)}")
            record_event(
                level="ERROR",
                action="ttl_manager.pruning_error",
                message=f"TTL pruning error: {str(e)}"
            )
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'knowledge_chunks_deleted': 0,
                'sessions_deleted': 0
            }
    
    async def _prune_knowledge_chunks(self, db, now: datetime) -> Dict[str, Any]:
        """Delete expired KnowledgeChunks.
        
        A chunk is expired if: created_at + (ttl_days in days) < now
        """
        coll = db['global_context']  # KnowledgeChunks collection
        
        try:
            # Find all chunks to check expiration
            all_chunks = await coll.find({}).to_list(length=10000)
            checked_count = len(all_chunks)
            
            expired_ids = []
            for chunk in all_chunks:
                created_at = chunk.get('created_at')
                ttl_days = chunk.get('ttl_days', 30)  # Default 30 days per project guide
                
                if created_at:
                    # Calculate expiration date
                    expiration_date = created_at + timedelta(days=ttl_days)
                    
                    # Check if expired
                    if expiration_date < now:
                        expired_ids.append(chunk['_id'])
            
            # Delete expired chunks
            if expired_ids:
                result = await coll.delete_many({'_id': {'$in': expired_ids}})
                deleted_count = result.deleted_count
            else:
                deleted_count = 0
            
            return {
                'checked_count': checked_count,
                'deleted_count': deleted_count
            }
            
        except Exception as e:
            append_log(f"_prune_knowledge_chunks error: {str(e)}")
            return {'checked_count': 0, 'deleted_count': 0, 'error': str(e)}
    
    async def _prune_sessions(self, db, now: datetime) -> Dict[str, Any]:
        """Delete expired sessions.
        
        Sessions expire 30 days after creation per default TTL.
        """
        coll = db['sessions']
        
        try:
            # Default session TTL: 30 days
            session_ttl_days = 30
            expiration_threshold = now - timedelta(days=session_ttl_days)
            
            # Find and delete expired sessions
            result = await coll.delete_many({
                'created_at': {'$lt': expiration_threshold}
            })
            
            # Count total sessions for context
            total_sessions = await coll.count_documents({})
            
            return {
                'checked_count': total_sessions,
                'deleted_count': result.deleted_count
            }
            
        except Exception as e:
            append_log(f"_prune_sessions error: {str(e)}")
            return {'checked_count': 0, 'deleted_count': 0, 'error': str(e)}


# Global TTL manager instance
_ttl_manager = TTLManager(check_interval_minutes=60)  # Check every hour


async def get_ttl_manager() -> TTLManager:
    """Get the global TTL manager instance."""
    return _ttl_manager
