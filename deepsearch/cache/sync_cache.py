from functools import lru_cache
import sqlite3
import os
import time
from typing import Any, Optional
import json
import deepsearch.constants as const

class SQLiteCache:
    def __init__(self, db_path):
        """Initialize the SQLite cache.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path

    def _create_table_if_not_exists(self, table_name: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL,
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()

    def _clean_expired(self, table_name: str):
        """Remove expired entries from the cache."""
        current_time = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {table_name} WHERE expires_at IS NOT NULL AND expires_at < ?",
                (current_time,)
            )
            conn.commit()
    
    def set(self, table_name: str, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The key to store the value under
            value: The value to store (will be JSON serialized)
            ttl: Time to live in seconds. If None, the key won't expire
        """
        self._create_table_if_not_exists(table_name)

        current_time = time.time()
        expires_at = current_time + ttl if ttl is not None else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT OR REPLACE INTO {table_name} (key, value, expires_at, created_at)
                VALUES (?, ?, ?, ?)
            """, (key, value, expires_at, current_time))
            conn.commit()
    
    def get(self, table_name: str, key: str) -> Optional[str]:
        """Get a value from the cache.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found or expired
        """
        self._create_table_if_not_exists(table_name)
        self._clean_expired(table_name)  # Clean expired entries before getting
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL,
                    created_at REAL NOT NULL
                )
            """)
            cursor.execute(f"""
                SELECT value FROM {table_name} 
                WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            """, (key, time.time()))
            
            result = cursor.fetchone()
            
            if result is None:
                return None
                
            return result[0]
    
    def delete(self, table_name: str, key: str) -> None:
        """Delete a key from the cache.
        
        Args:
            key: The key to delete
        """
        self._create_table_if_not_exists(table_name)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name} WHERE key = ?", (key,))
            conn.commit()
    
    def clear(self, table_name: str) -> None:
        """Clear all entries from the cache."""
        self._create_table_if_not_exists(table_name)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name}")
            conn.commit()


@lru_cache(maxsize=1)
def get_sqlite_cache(database_name: str = "cache"):
    return SQLiteCache(os.path.join(const.CACHE_DB_FOLDER, f"{database_name}.db"))
