from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    DEFAULT_MODEL: str = "gemini-3.6-flash"
    # Supabase Postgres (the same database the backend uses). When set, product
    # search reads the backend's "Product" table and keeps pgvector embeddings in
    # the "ai" schema. When empty, persistent Chroma/SQLite are used locally.
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    DATABASE_URL: str = ""
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = Field(default=768, ge=1, le=3072)
    # Shared secret from the trusted backend; per-brand SERVICE_API_KEYS also work.
    AI_SERVICE_TOKEN: str = ""
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CONVERSATION_DB_PATH: str = "./data/conversations.sqlite3"
    MEMORY_RETENTION_DAYS: int = Field(default=20, ge=20, le=20)
    HISTORY_MAX_MESSAGES: int = Field(default=20, ge=2, le=100)
    HISTORY_MAX_CHARS: int = Field(default=12000, ge=1000, le=50000)
    SERVICE_API_KEYS: dict[str, str] = Field(default_factory=dict)
    # One model attempt may take LLM_TIMEOUT_MS; all attempts together (retries and
    # the fallback model) stay within LLM_TOTAL_BUDGET_MS, so the service always
    # answers before the backend gives up on it (55 s).
    LLM_TIMEOUT_MS: int = Field(default=25000, ge=1000, le=50000)
    LLM_TOTAL_BUDGET_MS: int = Field(default=45000, ge=1000, le=50000)
    # Second attempt on Gemini (the service's key or a dashboard one). Empty
    # means the built-in fallback (gemini_service.FALLBACK_MODEL, or
    # LAST_RESORT_MODEL when the primary already is that model).
    LLM_FALLBACK_MODEL: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
