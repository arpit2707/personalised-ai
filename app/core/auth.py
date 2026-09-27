"""Backend-only merchant authentication for catalog, memory and inbox APIs."""
from hmac import compare_digest
from fastapi import Header, HTTPException
from app.core.config import settings


def authenticated_brand(x_service_key: str = Header(default="")) -> str:
    if not settings.SERVICE_API_KEYS:
        raise HTTPException(503, "Configure SERVICE_API_KEYS before serving merchant requests")
    for key, brand in settings.SERVICE_API_KEYS.items():
        if compare_digest(x_service_key.encode(), key.encode()):
            return brand
    raise HTTPException(401, "Invalid service key")


def require_brand(requested: str, authenticated: str):
    if requested != authenticated:
        raise HTTPException(403, "Merchant access denied")
