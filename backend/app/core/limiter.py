"""
Rate Limiter
Simple in-memory rate limiting dependency for FastAPI.
"""
import time
from collections import defaultdict
from typing import Dict, List
from fastapi import Request, HTTPException, status

class RateLimiter:
    def __init__(self, requests: int = 5, window: int = 60):
        self.requests = requests
        self.window = window
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.last_cleanup = time.time()

    async def __call__(self, request: Request):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Periodic cleanup (every 10 minutes) to prevent memory leaks
        if now - self.last_cleanup > 600:
            self._cleanup(now)
            self.last_cleanup = now

        # Clean up current client's timestamps
        self.history[client_ip] = [
            t for t in self.history[client_ip]
            if now - t < self.window
        ]

        # Check limit
        if len(self.history[client_ip]) >= self.requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later.",
                headers={"Retry-After": str(self.window)}
            )

        # Add current request
        self.history[client_ip].append(now)

    def _cleanup(self, now: float):
        """Remove empty or expired entries from history"""
        # Iterate over copy of keys
        for ip in list(self.history.keys()):
            # Filter timestamps
            self.history[ip] = [t for t in self.history[ip] if now - t < self.window]
            # Remove key if empty
            if not self.history[ip]:
                del self.history[ip]
