import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.core.config import settings
from app.services.conversation_store import conversation_store
from app.services.gemini_service import gemini_service

# Without a handler our INFO lines (llm_generation ... elapsed_ms) never reach the
# Render logs; uvicorn only configures its own loggers.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app):
    async def cleanup():
        # The first purge runs alongside the start-up, not before it: on a cold
        # start the backend's 55 s wait is already counting.
        while True:
            try:
                await asyncio.to_thread(conversation_store.purge)
            except Exception:
                logging.getLogger(__name__).error("Conversation retention cleanup failed")
            await asyncio.sleep(60)

    task = asyncio.create_task(cleanup())
    # The first customer after a cold start should not also wait for the SDK import.
    await asyncio.to_thread(gemini_service.warm_up)
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

app = FastAPI(
    lifespan=lifespan,
    title="Reel2Real Personalised AI Core",
    version="1.0.0",
    description="Dedicated AI Intelligence Microservice for Reel2Real Omnichannel Automation"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {
        "message": "Reel2Real Personalised AI Engine is running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
