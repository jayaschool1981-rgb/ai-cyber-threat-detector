import time
import logging
from typing import Dict, Tuple
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
import redis

from src.core.config import settings

logger = logging.getLogger("rate_limit_middleware")

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int = 100, window: int = 60):
        super().__init__(app)
        self.limit = limit
        self.window = window
        
        # Initialize Redis client
        self.redis_client = None
        try:
            # We assume redis service name 'redis' from docker-compose, or localhost
            # We construct redis url.
            redis_host = "localhost"
            # If docker environment settings exist, we can use them.
            # Let's check environment settings.
            import os
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", 6379))
            logger.info(f"Connecting to Redis at {host}:{port} for rate limiting...")
            self.redis_client = redis.Redis(host=host, port=port, db=0, socket_timeout=1.0)
            self.redis_client.ping()
            logger.info("Connected to Redis successfully for rate limiting.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {str(e)}. Falling back to in-memory rate limiting.")
            self.redis_client = None

        # In-memory store fallback
        # Key: IP -> Value: (window_start_timestamp, count)
        self.in_memory_store: Dict[str, Tuple[float, int]] = {}

    def _is_rate_limited(self, ip: str) -> bool:
        now = time.time()
        
        # Redis implementation
        if self.redis_client is not None:
            try:
                key = f"rate_limit:{ip}"
                val = self.redis_client.incr(key)
                if val == 1:
                    self.redis_client.expire(key, self.window)
                if val > self.limit:
                    return True
                return False
            except Exception as e:
                logger.warning(f"Redis operation failed, falling back to in-memory: {str(e)}")
                # Proceed to in-memory fallback
        
        # In-memory fallback implementation
        # Clean up stale memory entries on the fly to avoid growth leaks
        stale_ips = [k for k, v in self.in_memory_store.items() if now - v[0] > self.window]
        for k in stale_ips:
            self.in_memory_store.pop(k, None)

        if ip in self.in_memory_store:
            start_time, count = self.in_memory_store[ip]
            if now - start_time > self.window:
                # Reset window
                self.in_memory_store[ip] = (now, 1)
                return False
            else:
                if count >= self.limit:
                    return True
                else:
                    self.in_memory_store[ip] = (start_time, count + 1)
                    return False
        else:
            self.in_memory_store[ip] = (now, 1)
            return False

    async def dispatch(self, request: Request, call_next) -> Response:
        # Exclude docs and health check endpoints from rate limiting
        path = request.url.path
        if path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return Response(
                content="Rate limit exceeded. Try again later.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS
            )

        return await call_next(request)
