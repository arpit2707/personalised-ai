"""PgCatalogStore against a real Postgres with pgvector.

Needs TEST_DATABASE_URL pointing at a database that has the backend's "Product"
table (run `prisma migrate deploy` from ReelToRealAuto first); skipped otherwise.
"""
import os

import pytest

from app.services.vector_store import MemoryCatalogStore, PgCatalogStore, product_text

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")


class FakeEmbedder:
    """Deterministic stand-in for Gemini, reusing the bag-of-words vectors (padded to 768)."""

    available = True

    def __init__(self):
        self.calls = 0

    def embed(self, texts, task_type):
        self.calls += 1
        return [MemoryCatalogStore._vec(t) + [0.0] * (768 - 128) for t in texts]


PRODUCTS = [
    ("p1", "KURTA-101", "Royal Silk Blue Kurta", "Hand-woven silk kurta for weddings", 1499),
    ("p2", "SAREE-7", "Banarasi Red Saree", "Classic banarasi saree with zari border", 3999),
]


@pytest.fixture()
def store():
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute('DELETE FROM "Product" WHERE "orgId" IN (%s, %s)', ("org_a", "org_b"))
        for org in ("org_a", "org_b"):
            conn.execute(
                'INSERT INTO "Organization" (id, name, slug, "updatedAt") VALUES (%s, %s, %s, now()) '
                "ON CONFLICT (id) DO NOTHING",
                (org, org, org),
            )
        for pid, sku, title, desc, price in PRODUCTS:
            conn.execute(
                'INSERT INTO "Product" (id, "orgId", sku, title, description, price, sizes, colors, "checkoutUrl", "updatedAt") '
                "VALUES (%s, 'org_a', %s, %s, %s, %s, ARRAY['M','L'], ARRAY['Red'], %s, now())",
                (pid, sku, title, desc, price, f"https://shop/{sku}"),
            )
        conn.execute("DROP TABLE IF EXISTS ai.product_embedding")
    return PgCatalogStore(DSN, FakeEmbedder())


def test_vector_search_reads_products_from_backend_table(store):
    assert store.reindex("org_a") == 2
    hits = store.search_products("org_a", "silk kurta", limit=1)
    assert [h.sku for h in hits] == ["KURTA-101"]
    assert hits[0].checkout_url == "https://shop/KURTA-101"
    assert hits[0].sizes == ["M", "L"]


def test_brands_are_isolated(store):
    store.reindex("org_a")
    assert store.search_products("org_b", "silk kurta") == []


def test_only_changed_products_are_re_embedded(store):
    import psycopg

    assert store.reindex("org_a") == 2
    assert store.reindex("org_a") == 0
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("""UPDATE "Product" SET title = 'Royal Silk Green Kurta', "updatedAt" = now() WHERE id = 'p1'""")
        conn.execute("""DELETE FROM "Product" WHERE id = 'p2'""")
    assert store.reindex("org_a") == 1
    with psycopg.connect(DSN) as conn:
        assert conn.execute("SELECT count(*) FROM ai.product_embedding WHERE org_id = 'org_a'").fetchone()[0] == 1


def test_keyword_search_without_embeddings(store):
    store._embedder.available = False
    hits = store.search_products("org_a", "banarasi saree price?", limit=2)
    assert [h.sku for h in hits] == ["SAREE-7"]


def test_sku_lookup_and_read_only_catalog(store):
    assert store.get_product_by_sku("org_a", "SAREE-7").title == "Banarasi Red Saree"
    assert store.get_product_by_sku("org_b", "SAREE-7") is None
    with pytest.raises(PermissionError):
        store.upsert_product("org_a", None)


def test_product_text_mentions_searchable_fields():
    from app.models.schemas import ProductInfo

    text = product_text(ProductInfo(sku="X1", title="Kurta", price=1, sizes=["M"], colors=["Blue"]))
    assert "Kurta" in text and "X1" in text and "Blue" in text
