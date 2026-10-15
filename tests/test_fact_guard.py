"""Facts other than prices: delivery times, policies, stock and links."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.gemini_service import gemini_service

client = TestClient(app)

SHOP = {
    "brand_id": "test_brand",
    "channel_type": "instagram",
    "event_type": "dm",
    "message_text": "COD hai? delivery kitne din me?",
    "sender_id": "fact-u1",
    "business": {"industry": "APPAREL",
                 "description": "Handmade kurtas. Order form: forms.gle/kurta-order",
                 "policies": {"delivery": "Delivery in 4-6 days across India", "returns": "Exchange within 7 days"}},
    "offerings": [{"id": "k1", "title": "Silk Kurta", "price_label": "₹1999", "price_min": 1999,
                   "action_url": "https://shop.example.in/kurta",
                   "variants": [{"label": "M", "in_stock": False}, {"label": "L", "in_stock": True}]}],
}


def reply(dm, monkeypatch, public=None, **extra):
    monkeypatch.setattr(gemini_service, "generate", lambda *_: {
        "private_dm": dm, "public_reply": public, "action": "ANSWER", "offering_ids": ["k1"]})
    payload = {**SHOP, "sender_id": f"fact-{dm[:30]}-{extra.get('event_type', 'dm')}", **extra}
    res = client.post("/api/v1/generate-reply", json=payload)
    assert res.status_code == 200, res.text
    return res.json()


@pytest.mark.parametrize("dm", [
    "Haan COD available hai!",                  # no COD in the policies
    "Delivery 2-3 din me ho jayegi.",           # the policy says 4-6 days
    "Free delivery hai!",
    "Same day delivery bhi hai.",
    "M size available hai.",                    # M is out of stock
    "Sirf 2 pieces bache hain, jaldi karo!",    # nobody said so
    "1 saal ki warranty hai.",
])
def test_unsupported_fact_is_not_sent(dm, monkeypatch):
    data = reply(dm, monkeypatch)
    assert data["action"] == "HANDOFF"
    assert data["intent"] == "fact_blocked"
    assert dm not in (data["private_dm"] or "")


@pytest.mark.parametrize("dm", [
    "Delivery 4-6 din me ho jayegi.",
    "7 din me exchange ho jata hai.",
    "M size abhi available nahi hai, L available hai.",
    "Silk Kurta ₹1999 ka hai. Order: https://shop.example.in/kurta",
    "Order form yahan bhar dijiye: forms.gle/kurta-order",
])
def test_fact_from_the_seller_is_sent(dm, monkeypatch):
    data = reply(dm, monkeypatch)
    assert data["action"] == "ANSWER", data
    assert data["private_dm"] == dm


def test_customer_asking_is_not_a_source(monkeypatch):
    # "COD hai?" from the customer does not make "Haan COD hai" true.
    data = reply("Haan ji, COD hai.", monkeypatch, message_text="COD hai kya?")
    assert data["intent"] == "fact_blocked"


def test_customer_count_may_be_repeated(monkeypatch):
    data = reply("Theek hai, 3 pieces ka order note kar liya, team confirm karegi.", monkeypatch,
                 message_text="mujhe 3 pieces chahiye")
    assert data["action"] == "ANSWER"


@pytest.mark.parametrize("dm", [
    "Order yahan karo: myshop.in/kurta",
    "Details: www.other-shop.com",
    "WhatsApp karo wa.me/919999999999",
    "Link: https://evil.example.com/x",
])
def test_links_without_scheme_are_checked(dm, monkeypatch):
    data = reply(dm, monkeypatch)
    assert data["action"] == "HANDOFF"
    assert dm not in (data["private_dm"] or "")


def test_public_comment_reply_carries_no_link(monkeypatch):
    data = reply("Silk Kurta ₹1999 ka hai. Order: https://shop.example.in/kurta", monkeypatch,
                 public="DM check karo! https://shop.example.in/kurta", event_type="comment",
                 message_text="price?")
    assert data["action"] == "ANSWER"
    assert "http" not in data["public_reply"]
    assert data["public_reply"].startswith("DM check karo")
