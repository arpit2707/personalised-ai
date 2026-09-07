import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "personalised-ai"

def test_catalog_upsert_and_search():
    product_payload = {
        "sku": "KURTA-101",
        "title": "Royal Silk Blue Kurta",
        "description": "Premium hand-woven royal silk kurta for weddings and festive wear",
        "price": 1499.0,
        "currency": "INR",
        "in_stock": True,
        "stock_quantity": 25,
        "sizes": ["S", "M", "L", "XL"],
        "colors": ["Royal Blue"],
        "checkout_url": "https://brand.com/products/kurta-101"
    }
    
    # Upsert
    res = client.post("/api/v1/catalog/product?brand_id=test_brand", json=product_payload)
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # Search
    search_res = client.get("/api/v1/catalog/search?brand_id=test_brand&query=silk%20kurta")
    assert search_res.status_code == 200
    products = search_res.json()
    assert len(products) > 0
    assert products[0]["sku"] == "KURTA-101"

def test_negative_sentiment_guardrail():
    payload = {
        "brand_id": "test_brand",
        "channel_type": "instagram",
        "event_type": "comment",
        "message_text": "Yeh fraud scam page hai, mera refund abhi tak nahi aaya!",
        "sender_id": "user_999"
    }
    res = client.post("/api/v1/generate-reply", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["requires_human_attention"] is True
    assert data["sentiment"] == "negative"
    assert "apologize" in data["private_dm"].lower()

def test_price_inquiry_reply_generation():
    payload = {
        "brand_id": "test_brand",
        "channel_type": "instagram",
        "event_type": "comment",
        "message_text": "Bhai price kya hai iska?",
        "sender_id": "user_123",
        "post_context": {
            "post_id": "post_789",
            "caption": "New Festive Collection Silk Kurta launching today!",
            "tagged_product_sku": "KURTA-101"
        },
        "brand_persona": {
            "brand_name": "Royal Styles",
            "tone": "friendly",
            "language_mode": "hinglish",
            "emoji_density": "moderate"
        }
    }
    res = client.post("/api/v1/generate-reply", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["requires_human_attention"] is False
    assert data["public_reply"] is not None
    assert data["private_dm"] is not None
    assert data["detected_product_sku"] == "KURTA-101"
