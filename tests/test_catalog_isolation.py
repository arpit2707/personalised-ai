from fastapi.testclient import TestClient
from app.main import app
from app.api.v1 import endpoints
from app.models.schemas import ProductInfo
from app.services.guardrails import SAFETY_REFUSAL


def product(sku='one', description='Cotton shirt'):
    return ProductInfo(sku=sku, title='Shirt', price=100, description=description)


def test_colliding_legacy_brand_names_are_isolated():
    store = endpoints.catalog_store
    store.upsert_product('a-b', product())
    assert store.get_product_by_sku('a_b', 'one') is None
    assert store.get_product_by_sku('A-B', 'one') is None
    assert store.get_product_by_sku('a-b', 'one').description == 'Cotton shirt'


def test_upsert_preserves_description_and_replaces_price():
    store = endpoints.catalog_store
    item = product()
    store.upsert_product('test_brand', item)
    item.price = 120
    item.description = 'Updated cotton shirt'
    store.upsert_product('test_brand', item)
    result = store.search_products('test_brand', 'cotton shirt')[0]
    assert result.price == 120
    assert result.description == 'Updated cotton shirt'


def test_seller_catalog_text_does_not_trigger_the_safety_filter():
    # Only the customer's words and the model's output are filtered; a seller's
    # description must not silence every reply about that item.
    endpoints.catalog_store.upsert_product('test_brand', product(description='toxic dye free fabric, no chemicals'))
    response = TestClient(app).post('/api/v1/generate-reply', json={
        'brand_id': 'test_brand', 'sender_id': 'u', 'message_text': 'Details please',
        'post_context': {'post_id': 'p', 'tagged_product_sku': 'one'},
    })
    assert response.json()['private_dm'] != SAFETY_REFUSAL
