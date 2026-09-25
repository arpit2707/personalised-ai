"""Product catalog search for reply grounding.

Production reads products straight from the backend's Supabase "Product" table
(the dashboard's catalog is the single source of truth, so nothing has to be
synced) and keeps one pgvector embedding per product in the "ai" schema, where
Prisma never looks. Without DATABASE_URL an in-memory catalog is used instead.
"""
import hashlib
import logging
import re
import threading
import time
from typing import Dict, List, Optional, Sequence

from app.core.config import settings
from app.models.schemas import ProductInfo

logger = logging.getLogger(__name__)

# Re-check a brand's catalog for new or edited products at most this often.
SYNC_INTERVAL_SECONDS = 60
EMBED_BATCH = 50


def product_text(p: ProductInfo) -> str:
    return (
        f"{p.title}. {p.description or ''} SKU {p.sku}. "
        f"Sizes: {', '.join(p.sizes)}. Colors: {', '.join(p.colors)}."
    )


class GeminiEmbedder:
    """Returns None when embeddings are unavailable so callers fall back to text search."""

    def __init__(self):
        self._client = None
        if settings.GEMINI_API_KEY:
            try:
                from google import genai

                self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Embedding client unavailable: {e}")

    @property
    def available(self) -> bool:
        return self._client is not None

    def embed(self, texts: Sequence[str], task_type: str) -> Optional[List[List[float]]]:
        if not self._client or not texts:
            return None
        try:
            from google.genai import types

            res = self._client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=list(texts),
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=settings.EMBEDDING_DIM
                ),
            )
            vectors = []
            for e in res.embeddings:
                # Truncated Gemini embeddings are not unit length; normalise for cosine.
                norm = sum(x * x for x in e.values) ** 0.5 or 1.0
                vectors.append([x / norm for x in e.values])
            return vectors
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            return None


class MemoryCatalogStore:
    """Per-process catalog for local development and tests (bag-of-words vectors)."""

    managed_externally = False

    def __init__(self):
        self._products: Dict[str, Dict[str, ProductInfo]] = {}

    @staticmethod
    def _vec(text: str) -> List[float]:
        vec = [0.0] * 128
        for token in text.lower().split():
            vec[int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 128] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def upsert_product(self, brand_id: str, product: ProductInfo):
        self._products.setdefault(brand_id, {})[product.sku] = product

    def get_product_by_sku(self, brand_id: str, sku: str) -> Optional[ProductInfo]:
        return self._products.get(brand_id, {}).get(sku)

    def search_products(self, brand_id: str, query: str, limit: int = 2) -> List[ProductInfo]:
        products = list(self._products.get(brand_id, {}).values())
        if not products:
            return []
        q = self._vec(query)
        scored = [(sum(a * b for a, b in zip(q, self._vec(product_text(p)))), p) for p in products]
        scored.sort(key=lambda s: s[0], reverse=True)
        return [p for score, p in scored[:limit] if score > 0]

    def reindex(self, brand_id: str) -> int:
        return len(self._products.get(brand_id, {}))


PRODUCT_COLUMNS = (
    'p.sku, p.title, p.description, p.price, p.currency, p."inStock", '
    'p."stockQuantity", p.sizes, p.colors, p."checkoutUrl"'
)


def _row_to_product(row) -> ProductInfo:
    return ProductInfo(
        sku=row[0],
        title=row[1],
        description=row[2] or "",
        price=float(row[3]),
        currency=row[4] or "INR",
        in_stock=bool(row[5]),
        stock_quantity=int(row[6] or 0),
        sizes=list(row[7] or []),
        colors=list(row[8] or []),
        checkout_url=row[9] or "",
    )


def _vector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


class PgCatalogStore:
    """Supabase-backed catalog: products from "Product", embeddings in ai.product_embedding."""

    managed_externally = True

    def __init__(self, dsn: str, embedder: GeminiEmbedder):
        from psycopg_pool import ConnectionPool

        # prepare_threshold=None: Supabase's transaction pooler does not support
        # server-side prepared statements.
        self._pool = ConnectionPool(
            dsn,
            min_size=0,
            max_size=5,
            kwargs={"prepare_threshold": None, "autocommit": True},
            open=True,
        )
        self._embedder = embedder
        self._last_sync: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._vector_ready = False
        self._ensure_schema()

    def _ensure_schema(self):
        # Idempotent, and only touches the "ai" schema and the vector extension.
        try:
            with self._pool.connection() as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.execute("CREATE SCHEMA IF NOT EXISTS ai")
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS ai.product_embedding (
                        product_id text PRIMARY KEY,
                        org_id text NOT NULL,
                        product_updated_at timestamp(3) NOT NULL,
                        model text NOT NULL,
                        embedding vector({settings.EMBEDDING_DIM}) NOT NULL,
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS product_embedding_org_idx ON ai.product_embedding (org_id)"
                )
            self._vector_ready = True
        except Exception as e:
            # Keyword search on "Product" still works without pgvector.
            logger.error(f"pgvector setup failed, using keyword search only: {e}")

    def upsert_product(self, brand_id: str, product: ProductInfo):
        raise PermissionError("Products are managed in the Reel2Real dashboard, not through this service")

    def get_product_by_sku(self, brand_id: str, sku: str) -> Optional[ProductInfo]:
        with self._pool.connection() as conn:
            row = conn.execute(
                f'SELECT {PRODUCT_COLUMNS} FROM "Product" p WHERE p."orgId" = %s AND p.sku = %s',
                (brand_id, sku),
            ).fetchone()
        return _row_to_product(row) if row else None

    def reindex(self, brand_id: str) -> int:
        """Embed products that are new or changed since their last embedding."""
        if not (self._vector_ready and self._embedder.available):
            return 0
        embedded = 0
        with self._pool.connection() as conn:
            while True:
                rows = conn.execute(
                    f"""
                    SELECT p.id, p."updatedAt", {PRODUCT_COLUMNS}
                    FROM "Product" p
                    LEFT JOIN ai.product_embedding e ON e.product_id = p.id
                    WHERE p."orgId" = %s
                      AND (e.product_id IS NULL OR e.product_updated_at <> p."updatedAt" OR e.model <> %s)
                    LIMIT %s
                    """,
                    (brand_id, settings.EMBEDDING_MODEL, EMBED_BATCH),
                ).fetchall()
                if not rows:
                    break
                vectors = self._embedder.embed(
                    [product_text(_row_to_product(r[2:])) for r in rows], "RETRIEVAL_DOCUMENT"
                )
                if not vectors:
                    break
                with conn.transaction():
                    for r, vec in zip(rows, vectors):
                        conn.execute(
                            """
                            INSERT INTO ai.product_embedding (product_id, org_id, product_updated_at, model, embedding)
                            VALUES (%s, %s, %s, %s, %s::vector)
                            ON CONFLICT (product_id) DO UPDATE SET
                                product_updated_at = EXCLUDED.product_updated_at,
                                model = EXCLUDED.model,
                                embedding = EXCLUDED.embedding,
                                updated_at = now()
                            """,
                            (r[0], brand_id, r[1], settings.EMBEDDING_MODEL, _vector_literal(vec)),
                        )
                embedded += len(rows)
                if len(rows) < EMBED_BATCH:
                    break
            # Embeddings of products deleted from the dashboard.
            conn.execute(
                'DELETE FROM ai.product_embedding e WHERE e.org_id = %s '
                'AND NOT EXISTS (SELECT 1 FROM "Product" p WHERE p.id = e.product_id)',
                (brand_id,),
            )
        return embedded

    def _maybe_sync(self, brand_id: str):
        now = time.monotonic()
        with self._lock:
            if now - self._last_sync.get(brand_id, 0) < SYNC_INTERVAL_SECONDS:
                return
            self._last_sync[brand_id] = now
        try:
            self.reindex(brand_id)
        except Exception as e:
            logger.error(f"Catalog embedding sync failed for {brand_id}: {e}")

    def search_products(self, brand_id: str, query: str, limit: int = 2) -> List[ProductInfo]:
        limit = max(1, min(limit, 20))
        if self._vector_ready and self._embedder.available:
            self._maybe_sync(brand_id)
            qvec = self._embedder.embed([query], "RETRIEVAL_QUERY")
            if qvec:
                with self._pool.connection() as conn:
                    rows = conn.execute(
                        f"""
                        SELECT {PRODUCT_COLUMNS}
                        FROM ai.product_embedding e
                        JOIN "Product" p ON p.id = e.product_id
                        WHERE e.org_id = %s AND e.model = %s
                        ORDER BY e.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (brand_id, settings.EMBEDDING_MODEL, _vector_literal(qvec[0]), limit),
                    ).fetchall()
                if rows:
                    return [_row_to_product(r) for r in rows]
        return self._keyword_search(brand_id, query, limit)

    def _keyword_search(self, brand_id: str, query: str, limit: int) -> List[ProductInfo]:
        tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
        if not tokens:
            return []
        tsquery = " | ".join(tokens)
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT {PRODUCT_COLUMNS}
                FROM "Product" p,
                     to_tsvector('simple', coalesce(p.title, '') || ' ' || coalesce(p.description, '') || ' ' || p.sku) doc,
                     to_tsquery('simple', %s) q
                WHERE p."orgId" = %s AND doc @@ q
                ORDER BY ts_rank(doc, q) DESC
                LIMIT %s
                """,
                (tsquery, brand_id, limit),
            ).fetchall()
        return [_row_to_product(r) for r in rows]


def build_catalog_store():
    if settings.DATABASE_URL:
        return PgCatalogStore(settings.DATABASE_URL, GeminiEmbedder())
    logger.warning("DATABASE_URL not set: using an in-memory product catalog")
    return MemoryCatalogStore()


catalog_store = build_catalog_store()
