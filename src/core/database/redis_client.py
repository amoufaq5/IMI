"""Redis client for caching and session management"""
import json
from typing import Optional, Any, List
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from src.core.config import settings


class RedisClient:
    """Async Redis client for caching"""
    
    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Establish connection to Redis"""
        self._pool = ConnectionPool.from_url(
            settings.redis_url,
            password=settings.redis_password,
            max_connections=50,
            decode_responses=True,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        # Verify connection
        await self._client.ping()
    
    async def close(self) -> None:
        """Close Redis connection"""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
    
    @property
    def client(self) -> redis.Redis:
        """Get Redis client"""
        if not self._client:
            raise RuntimeError("Redis client not connected")
        return self._client
    
    # Basic operations
    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        return await self.client.get(key)
    
    async def set(
        self,
        key: str,
        value: str,
        expire: Optional[int] = None,
    ) -> bool:
        """Set value with optional expiration (seconds)"""
        return await self.client.set(key, value, ex=expire)
    
    async def delete(self, key: str) -> int:
        """Delete key"""
        return await self.client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        return await self.client.exists(key) > 0
    
    # JSON operations
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value"""
        value = await self.get(key)
        return json.loads(value) if value else None
    
    async def set_json(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
    ) -> bool:
        """Set JSON value"""
        return await self.set(key, json.dumps(value), expire)
    
    # Cache operations
    async def cache_get(self, namespace: str, key: str) -> Optional[Any]:
        """Get cached value with namespace"""
        full_key = f"cache:{namespace}:{key}"
        return await self.get_json(full_key)
    
    async def cache_set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
    ) -> bool:
        """Set cached value with namespace and TTL"""
        full_key = f"cache:{namespace}:{key}"
        return await self.set_json(full_key, value, ttl_seconds)
    
    async def cache_invalidate(self, namespace: str, key: Optional[str] = None) -> int:
        """Invalidate cache by namespace or specific key"""
        if key:
            return await self.delete(f"cache:{namespace}:{key}")
        else:
            # Invalidate all keys in namespace
            pattern = f"cache:{namespace}:*"
            keys = await self.client.keys(pattern)
            if keys:
                return await self.client.delete(*keys)
            return 0
    
    # Session management
    async def create_session(
        self,
        session_id: str,
        user_id: str,
        data: dict,
        ttl_hours: int = 24,
    ) -> bool:
        """Create user session"""
        session_data = {
            "user_id": user_id,
            "data": data,
        }
        return await self.set_json(
            f"session:{session_id}",
            session_data,
            ttl_hours * 3600,
        )
    
    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data"""
        return await self.get_json(f"session:{session_id}")
    
    async def delete_session(self, session_id: str) -> int:
        """Delete session"""
        return await self.delete(f"session:{session_id}")
    
    async def extend_session(self, session_id: str, ttl_hours: int = 24) -> bool:
        """Extend session TTL"""
        return await self.client.expire(
            f"session:{session_id}",
            ttl_hours * 3600,
        )
    
    # Rate limiting
    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check rate limit using sliding window
        Returns (is_allowed, remaining_requests)
        """
        rate_key = f"ratelimit:{key}"
        
        # Increment counter
        current = await self.client.incr(rate_key)
        
        # Set expiry on first request
        if current == 1:
            await self.client.expire(rate_key, window_seconds)
        
        remaining = max(0, max_requests - current)
        is_allowed = current <= max_requests
        
        return is_allowed, remaining
    
    # Pub/Sub for real-time updates
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel"""
        return await self.client.publish(channel, json.dumps(message))
    
    async def subscribe(self, *channels: str):
        """Subscribe to channels"""
        pubsub = self.client.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub


# Singleton instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """Get or create Redis client singleton"""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
        await _redis_client.connect()
    return _redis_client


async def close_redis_client() -> None:
    """Close Redis client"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
