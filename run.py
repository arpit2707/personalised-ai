import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print(f"Starting Personalised AI on http://{settings.HOST}:{settings.PORT} (Model: {settings.DEFAULT_MODEL})")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
