"""Backend-only merchant authentication for catalog, memory and inbox APIs."""
from hmac import compare_digest
from fastapi import Header, HTTPException, Request
from app.core.config import settings


async def authenticated_brand(request: Request, x_service_key: str = Header(default=""),
                              x_ai_service_token: str = Header(default=""),
                              x_brand_id: str = Header(default="")) -> str:
    # Existing trusted Reel2Real backend authenticates end users itself. Bind its
    # shared credential to the brand explicitly supplied for this request.
    if settings.AI_SERVICE_TOKEN and compare_digest(x_ai_service_token.encode(), settings.AI_SERVICE_TOKEN.encode()):
        brand = x_brand_id or request.query_params.get('brand_id')
        if not brand and request.method == 'POST':
            try:
                body = await request.json()
                brand = body.get('brand_id') if isinstance(body, dict) else None
            except ValueError:
                pass
        if not isinstance(brand, str) or not brand or len(brand) > 128:
            raise HTTPException(400, 'Supply brand_id or X-Brand-ID for backend requests')
        return brand
    if not settings.SERVICE_API_KEYS and not settings.AI_SERVICE_TOKEN:
        raise HTTPException(503, "Configure SERVICE_API_KEYS before serving merchant requests")
    for key, brand in settings.SERVICE_API_KEYS.items():
        if compare_digest(x_service_key.encode(), key.encode()):
            return brand
    raise HTTPException(401, "Invalid service key")


def require_brand(requested: str, authenticated: str):
    if requested != authenticated:
        raise HTTPException(403, "Merchant access denied")
