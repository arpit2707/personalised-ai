from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    DEFAULT_MODEL: str = "gemini-3.6-flash"
    # Supabase Postgres (the same database the backend uses). When set, product
    # search reads the backend's "Product" table and keeps pgvector embeddings in
    # the "ai" schema. When empty, an in-memory catalog is used (local dev, tests).
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    DATABASE_URL: str = ""
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 768
    # Shared secret the backend sends as X-AI-Service-Token. Empty disables the check.
    AI_SERVICE_TOKEN: str = ""
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CONVERSATION_DB_PATH: str = "./data/conversations.sqlite3"
    MEMORY_RETENTION_DAYS: int = Field(default=20, ge=20, le=20)
    HISTORY_MAX_MESSAGES: int = Field(default=20, ge=2, le=100)
    HISTORY_MAX_CHARS: int = Field(default=12000, ge=1000, le=50000)
    SERVICE_API_KEYS: dict[str, str] = Field(default_factory=dict)
    LLM_TIMEOUT_MS: int = Field(default=15000, ge=1000, le=60000)
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
