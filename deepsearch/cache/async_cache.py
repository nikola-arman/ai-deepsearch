from functools import lru_cache
import aiosqlite
import os
import time
from typing import Any, Optional
import json
import asyncio
import deepsearch.constants as const

class AsyncSQLiteCache:
    def __init__(self, db_path: str = "./storage/async_cache.db"):
        """Initialize the async SQLite cache.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self._lock = asyncio.Lock()  # For thread-safe operations
        self._initialized = False
    
    async def _init_db(self):
        """Initialize the database with required tables."""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:  # Double-check pattern
                return
                
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        expires_at REAL,
                        created_at REAL NOT NULL
                    )
                """)
                await db.commit()
                self._initialized = True
    
    async def _clean_expired(self):
        """Remove expired entries from the cache."""
        current_time = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                (current_time,)
            )
            await db.commit()
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The key to store the value under
            value: The value to store (will be JSON serialized)
            ttl: Time to live in seconds. If None, the key won't expire
        """
        await self._init_db()
        
        current_time = time.time()
        expires_at = current_time + ttl if ttl is not None else None
        
        # Convert value to JSON string
        value_json = json.dumps(value)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO cache (key, value, expires_at, created_at)
                VALUES (?, ?, ?, ?)
            """, (key, value_json, expires_at, current_time))
            await db.commit()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found or expired
        """
        await self._init_db()
        await self._clean_expired()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT value FROM cache 
                WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            """, (key, time.time())) as cursor:
                result = await cursor.fetchone()
                
                if result is None:
                    return None
                    
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    return None
    
    async def delete(self, key: str) -> None:
        """Delete a key from the cache.
        
        Args:
            key: The key to delete
        """
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM cache WHERE key = ?", (key,))
            await db.commit()
    
    async def clear(self) -> None:
        """Clear all entries from the cache."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM cache")
            await db.commit()
    
    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from the cache.
        
        Args:
            keys: List of keys to retrieve
            
        Returns:
            Dictionary of key-value pairs for found and non-expired keys
        """
        await self._init_db()
        await self._clean_expired()
        
        if not keys:
            return {}
            
        placeholders = ','.join(['?' for _ in keys])
        current_time = time.time()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(f"""
                SELECT key, value FROM cache 
                WHERE key IN ({placeholders}) 
                AND (expires_at IS NULL OR expires_at > ?)
            """, (*keys, current_time)) as cursor:
                results = {}
                async for row in cursor:
                    try:
                        results[row[0]] = json.loads(row[1])
                    except json.JSONDecodeError:
                        continue
                return results
    
    async def set_many(self, mapping: dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set multiple values in the cache.
        
        Args:
            mapping: Dictionary of key-value pairs to store
            ttl: Time to live in seconds for all keys. If None, keys won't expire
        """
        await self._init_db()
        
        if not mapping:
            return
            
        current_time = time.time()
        expires_at = current_time + ttl if ttl is not None else None
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany("""
                INSERT OR REPLACE INTO cache (key, value, expires_at, created_at)
                VALUES (?, ?, ?, ?)
            """, [
                (key, json.dumps(value), expires_at, current_time)
                for key, value in mapping.items()
            ])
            await db.commit() 

@lru_cache(maxsize=1)
def get_async_sqlite_cache(database_name: str = "cache"):
    return AsyncSQLiteCache(f"/storage/{database_name}.db")
