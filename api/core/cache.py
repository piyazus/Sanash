"""
Redis Cache Utilities
=====================

Redis connection and caching helpers.
"""

from typing import Any, Optional
import json

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from .config import settings


# =============================================================================
# CONNECTION POOL
# =============================================================================

pool: Optional[ConnectionPool] = None


async def init_redis() -> redis.Redis:
    """Initialize Redis connection pool."""
    global pool
    pool = ConnectionPool.from_url(
        settings.REDIS_URL,
        max_connections=50,
        decode_responses=True,
    )
    return redis.Redis(connection_pool=pool)


async def close_redis():
    """Close Redis connection pool."""
    global pool
    if pool:
        await pool.disconnect()


async def get_redis() -> redis.Redis:
    """Get Redis client."""
    if pool is None:
        return await init_redis()
    return redis.Redis(connection_pool=pool)


# =============================================================================
# CACHE UTILITIES
# =============================================================================

class Cache:
    """
    Redis cache wrapper with JSON serialization.
    
    Usage:
        cache = Cache(await get_redis())
        await cache.set("key", {"data": "value"}, ttl=3600)
        data = await cache.get("key")
    """
    
    def __init__(self, client: redis.Redis):
        self.client = client
        self.default_ttl = settings.REDIS_CACHE_TTL
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        data = await self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set cached value with optional TTL."""
        ttl = ttl or self.default_ttl
        await self.client.setex(key, ttl, json.dumps(value))
    
    async def delete(self, key: str):
        """Delete cached value."""
        await self.client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.client.exists(key) > 0
    
    async def invalidate_pattern(self, pattern: str):
        """Delete all keys matching pattern."""
        keys = await self.client.keys(pattern)
        if keys:
            await self.client.delete(*keys)


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter using Redis.
    
    Usage:
        limiter = RateLimiter(await get_redis())
        if await limiter.is_allowed("user:123"):
            # Process request
        else:
            # Rate limited
    """
    
    def __init__(
        self,
        client: redis.Redis,
        max_requests: int = 100,
        window_seconds: int = 60
    ):
        self.client = client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit."""
        full_key = f"ratelimit:{key}"
        
        current = await self.client.get(full_key)
        if current is None:
            await self.client.setex(full_key, self.window_seconds, 1)
            return True
        
        if int(current) >= self.max_requests:
            return False
        
        await self.client.incr(full_key)
        return True
    
    async def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        full_key = f"ratelimit:{key}"
        current = await self.client.get(full_key)
        if current is None:
            return self.max_requests
        return max(0, self.max_requests - int(current))
