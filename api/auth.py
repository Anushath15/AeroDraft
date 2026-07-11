import os
import hashlib
import time
from typing import Optional
from dataclasses import dataclass
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

@dataclass(frozen=True)
class AuthConfig:
    api_key: str
    rate_limit_per_minute: int = 60

def get_auth_config() -> AuthConfig:
    api_key = os.environ.get("AERODRAFT_API_KEY", "")
    if not api_key:
        api_key = hashlib.sha256(b"aerodraft-dev").hexdigest()[:32]
    return AuthConfig(api_key=api_key)

_request_counts: dict[str, list[float]] = {}

def check_rate_limit(api_key: str, limit: int) -> bool:
    now = time.time()
    minute_ago = now - 60
    if api_key not in _request_counts:
        _request_counts[api_key] = []
    _request_counts[api_key] = [t for t in _request_counts[api_key] if t > minute_ago]
    if len(_request_counts[api_key]) >= limit:
        return False
    _request_counts[api_key].append(now)
    return True

async def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    config = get_auth_config()
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required.")
    if api_key != config.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    if not check_rate_limit(api_key, config.rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    return api_key