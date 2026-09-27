import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.core.config import settings
from app.services.conversation_store import conversation_store


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(conversation_store.purge)

    async def cleanup():
        while True:
            await asyncio.sleep(60)
            try:
                await asyncio.to_thread(conversation_store.purge)
            except Exception:
                logging.getLogger(__name__).error("Conversation retention cleanup failed")

    task = asyncio.create_task(cleanup())
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
