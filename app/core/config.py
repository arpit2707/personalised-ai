from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    DEFAULT_MODEL: str = "gemini-3.6-flash"
    # Supabase Postgres (the same database the backend uses). When set, product
    # search reads the backend's "Product" table and keeps pgvector embeddings in
    # the "ai" schema. When empty, an in-memory catalog is used (local dev, tests).
    DATABASE_URL: str = ""
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 768
    # Shared secret the backend sends as X-AI-Service-Token. Empty disables the check.
    AI_SERVICE_TOKEN: str = ""
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
