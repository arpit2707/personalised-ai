import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import catalog_reply
from app.services.gemini_service import gemini_service

client = TestClient(app)

MAKEUP = {
    "brand_id": "test_brand",
    "channel_type": "instagram",
    "event_type": "comment",
    "message_text": "Is look ka kitna?",
    "sender_id": "u1",
    "post_context": {"post_id": "ig_1"},
    "brand_persona": {"brand_name": "Glam by Riya"},
    "business": {"industry": "BEAUTY_SERVICE", "industry_label": "Makeup artist / salon", "city": "Patna"},
    "playbook": {
        "goal": "BOOKING",
        "lead_fields": [
            {"key": "date", "label": "Event date", "ask": "Aapka function kis date ko hai?"},
            {"key": "city", "label": "City", "ask": "Aap kis city me ho?"},
        ],
        "rules": ["Prices are starting from."],
    },
    "offerings": [
        {
            "id": "off_bridal",
            "type": "PACKAGE",
            "title": "Bridal full look",
            "price_label": "₹18,000 se start",
            "price_mode": "STARTING_FROM",
            "price_min": 18000,
            "includes": ["Face makeup", "Hair styling", "Draping"],
            "linked_to_post": True,
        }
    ],
    "goal_state": {"fields": {"city": "Patna"}},
    "recent_messages": [{"from": "customer", "text": "hi"}],
}


def post(payload, raw, monkeypatch):
    monkeypatch.setattr(gemini_service, "generate", lambda *_: raw)
    res = client.post("/api/v1/generate-reply", json=payload)
    assert res.status_code == 200, res.text
    return res.json()


def test_catalog_price_passes_and_fields_are_filtered(monkeypatch):
    data = post(MAKEUP, {
        "public_reply": "DM check karo!",
        "private_dm": "Bridal full look ₹18,000 se start. Function kis date ko hai?",
        "intent": "price_inquiry",
        "action": "ASK_FIELD",
        "offering_ids": ["off_bridal", "made_up_id"],
        "collected_fields": {"city": "Patna", "favourite_colour": "red"},
    }, monkeypatch)
    assert data["action"] == "ASK_FIELD"
    assert data["offering_ids"] == ["off_bridal"]
    assert data["collected_fields"] == {}  # Not supplied in the current message.
    assert "18,000" in data["private_dm"]


def test_invented_price_is_blocked(monkeypatch):
    data = post(MAKEUP, {
        "public_reply": "Sirf ₹12,000!",
        "private_dm": "Ye look ₹12,000 me ho jayega",
        "action": "ANSWER",
    }, monkeypatch)
    assert data["action"] == "HANDOFF"
    assert data["requires_human_attention"] is True
    assert "12,000" not in data["private_dm"]
    assert data["intent"] == "price_blocked"


def test_model_down_queues_even_with_linked_offering(monkeypatch):
    data = post(MAKEUP, None, monkeypatch)
    assert data["action"] == "HANDOFF"
    assert data["handoff_reason"] == "generation_unavailable"


def test_model_down_with_no_catalog_match_hands_over(monkeypatch):
    payload = {**MAKEUP, "offerings": []}
    data = post(payload, None, monkeypatch)
    assert data["action"] == "HANDOFF"
    assert "₹" not in data["private_dm"]


def test_empty_catalog_still_answers_about_the_business(monkeypatch):
    payload = {
        **MAKEUP,
        "event_type": "dm",
        "message_text": "May I know something about your page?",
        "offerings": [],
        "business": {**MAKEUP["business"], "description": "Bridal and party makeup artist in Patna, home visits too."},
    }
    data = post(payload, {
        "private_dm": "Hum Patna me bridal aur party makeup karte hain, home visit bhi.",
        "intent": "general",
        "action": "ANSWER",
    }, monkeypatch)
    assert data["action"] == "ANSWER"
    assert data["requires_human_attention"] is False
    assert "Patna" in data["private_dm"]


def test_empty_catalog_without_business_info_asks_an_open_question(monkeypatch):
    seen = {}

    def model(system, prompt, *_):
        seen["prompt"] = prompt
        return {"private_dm": "Hi! Aap kya dhoondh rahe hain?", "intent": "general", "action": "ANSWER"}
    monkeypatch.setattr(gemini_service, "generate", model)
    data = client.post("/api/v1/generate-reply", json={**MAKEUP, "event_type": "dm", "offerings": []}).json()
    assert data["action"] == "ANSWER"
    assert data["private_dm"] == "Hi! Aap kya dhoondh rahe hain?"


def test_prompt_answers_greetings_from_business_info():
    from app.models.schemas import BrandPersona, GenerateReplyRequest
    req = GenerateReplyRequest.model_validate(MAKEUP)
    prompt = catalog_reply.build_system_prompt(BrandPersona(brand_name="Glam by Riya"), req)
    assert "Greetings" in prompt


def test_prompt_lists_only_missing_fields():
    from app.models.schemas import GenerateReplyRequest

    req = GenerateReplyRequest.model_validate(MAKEUP)
    prompt = catalog_reply.build_user_prompt(req)
    assert "Aapka function kis date ko hai?" in prompt
    assert "Aap kis city me ho?" not in prompt
    assert '"shown_in_this_post": true' in prompt


def test_amount_parsing():
    assert catalog_reply.extract_amounts("₹18k se start, budget 45 lakh, 1499/-") == [18000, 1499, 4500000]


def test_greeting_with_empty_catalog_is_answered(monkeypatch):
    payload = {**MAKEUP, "event_type": "dm", "message_text": "Hi how are you?", "offerings": [],
               "business": {**MAKEUP["business"], "description": "Bridal and party makeup in Patna"}}
    data = post(payload, {"private_dm": "Hi! Main theek hoon. Hum Patna me bridal aur party makeup karte hain.",
                          "intent": "general", "action": "ANSWER"}, monkeypatch)
    assert data["action"] == "ANSWER"
    assert data["requires_human_attention"] is False
    assert "Patna" in data["private_dm"]


def test_prompt_includes_post_caption_and_seller_note():
    from app.models.schemas import GenerateReplyRequest
    req = GenerateReplyRequest.model_validate({
        **MAKEUP,
        "post_context": {"post_id": "ig_1", "caption": "Wedding season offer!", "note": "Offer valid till Sunday"},
    })
    prompt = catalog_reply.build_user_prompt(req)
    assert "POST the customer is reacting to" in prompt
    assert "Wedding season offer!" in prompt
    assert '"seller_note": "Offer valid till Sunday"' in prompt


def test_prompt_has_no_post_section_without_caption_or_note():
    from app.models.schemas import GenerateReplyRequest
    prompt = catalog_reply.build_user_prompt(GenerateReplyRequest.model_validate(MAKEUP))
    assert "POST the customer is reacting to" not in prompt


def test_price_in_a_caption_is_still_blocked(monkeypatch):
    payload = {**MAKEUP, "post_context": {"post_id": "ig_1", "caption": "Sirf ₹9,999 me bridal look"}}
    data = post(payload, {"public_reply": "DM check karo", "private_dm": "Ye look ₹9,999 ka hai",
                          "intent": "price_inquiry", "action": "ANSWER", "offering_ids": ["off_bridal"]}, monkeypatch)
    assert "9,999" not in (data["private_dm"] or "")
    assert data["requires_human_attention"] is True


def test_prompt_answers_page_questions_instead_of_handing_over():
    from app.models.schemas import BrandPersona, GenerateReplyRequest
    req = GenerateReplyRequest(**{**MAKEUP, "offerings": []})
    system = catalog_reply.build_system_prompt(BrandPersona(brand_name="Glam"), req)
    assert "Greetings, small talk and questions about the page" in system


def test_public_comment_reply_loses_model_mentions(monkeypatch):
    data = post({**MAKEUP, "comment_author": "priya_sharma"}, {
        "public_reply": "@priya_sharma Haan, hair styling included hai!",
        "private_dm": "Bridal full look ₹18,000 se start.",
        "action": "ANSWER",
    }, monkeypatch)
    assert data["public_reply"] == "Haan, hair styling included hai!"


def test_prompt_names_the_commenter_and_asks_for_no_tag():
    from app.models.schemas import BrandPersona, GenerateReplyRequest
    req = GenerateReplyRequest.model_validate({**MAKEUP, "comment_author": "priya_sharma"})
    assert "Commenter: priya_sharma" in catalog_reply.build_user_prompt(req)
    assert "no @mentions" in catalog_reply.build_system_prompt(BrandPersona(brand_name="G"), req)


def test_post_item_rule_and_match_reach_the_prompt():
    from app.models.schemas import BrandPersona, GenerateReplyRequest
    offering = {**MAKEUP["offerings"][0], "linked_to_post": False, "match": "post"}
    req = GenerateReplyRequest.model_validate({**MAKEUP, "offerings": [offering]})
    assert '"shown_in_this_post": true' in catalog_reply.build_user_prompt(req)
    assert "mean that item" in catalog_reply.build_system_prompt(BrandPersona(brand_name="G"), req)
    chat = GenerateReplyRequest.model_validate({**MAKEUP, "offerings": [{**offering, "match": "chat"}]})
    assert '"why_listed": "chat"' in catalog_reply.build_user_prompt(chat)


def test_discovery_and_offer_type_rules():
    from app.models.schemas import BrandPersona, GenerateReplyRequest
    both = GenerateReplyRequest.model_validate({
        **MAKEUP, "event_type": "dm",
        "business": {**MAKEUP["business"], "offer_type": "BOTH", "categories": ["Bridal makeup", "Lehenga"]},
        "goal_state": {"stage": "DISCOVER", "fields": {}},
    })
    system = catalog_reply.build_system_prompt(BrandPersona(brand_name="G"), both)
    assert "product chahiye ya service?" in system
    assert "No prices yet" in system
    prompt = catalog_reply.build_user_prompt(both)
    assert '"categories": ["Bridal makeup", "Lehenga"]' in prompt
    assert '"stage": "DISCOVER"' in prompt
    services = GenerateReplyRequest.model_validate({**MAKEUP, "business": {**MAKEUP["business"], "offer_type": "SERVICES"}})
    assert "Hum services dete hain" in catalog_reply.build_system_prompt(BrandPersona(brand_name="G"), services)


def test_offering_type_comes_back(monkeypatch):
    data = post({**MAKEUP, "event_type": "dm"}, {
        "private_dm": "Bridal full look ₹18,000 se start.", "action": "ANSWER", "offering_type": "SERVICES",
    }, monkeypatch)
    assert data["offering_type"] == "SERVICES"


SPOT = {"post_id": "ig_9", "label": "Diwali bridal offer", "permalink": "https://instagram.com/p/abc",
        "caption": "Bridal looks for Diwali", "offering_ids": ["off_bridal"]}


def test_spotlight_reaches_the_prompt():
    from app.models.schemas import BrandPersona, GenerateReplyRequest
    req = GenerateReplyRequest.model_validate({**MAKEUP, "event_type": "dm", "spotlight": [SPOT]})
    prompt = catalog_reply.build_user_prompt(req)
    assert "Diwali bridal offer" in prompt
    assert "https://instagram.com/p/abc" in prompt
    assert "SPOTLIGHT lists the posts" in catalog_reply.build_system_prompt(BrandPersona(brand_name="G"), req)


def test_spotlight_link_is_allowed_other_links_are_not(monkeypatch):
    ok = post({**MAKEUP, "event_type": "dm", "spotlight": [SPOT]}, {
        "private_dm": "Ye dekhiye: https://instagram.com/p/abc", "action": "ANSWER"}, monkeypatch)
    assert "https://instagram.com/p/abc" in ok["private_dm"]
    bad = post({**MAKEUP, "event_type": "dm", "sender_id": "u2", "spotlight": [SPOT]}, {
        "private_dm": "Ye dekhiye: https://instagram.com/p/other", "action": "ANSWER"}, monkeypatch)
    assert "other" not in (bad["private_dm"] or "")


# ---------------------------------------------------------------- price guard gaps

@pytest.mark.parametrize("text,amounts", [
    ("Ye 2500 ka hai", [2500]), ("sirf 1499 me", [1499]), ("2.5k only", [2500]), ("2500 rupaye", [2500]),
    ("price 1499", [1499]), ("1800 per night", [1800]), ("flat 500 off", [500]), ("2 hazar ka", [2000]),
    # Not prices: sizes, days, years, phone numbers, counts, opening hours.
    ("size 38 me available", []), ("Delivery 3-5 din me", []), ("Dec 2026 me", []),
    ("98765 43210 pe call karo", []), ("only 3 left", []), ("24 hrs 7 days open", []),
])
def test_bare_prices_are_read(text, amounts):
    assert catalog_reply.extract_amounts(text) == amounts


@pytest.mark.parametrize("dm", [
    "Bridal full look 25000 ka hai.", "Bridal look sirf 15000 me!", "Bridal look 20k se start.",
    "Abhi 20% off mil raha hai.", "Flat 2000 off hai aaj.",
])
def test_made_up_price_or_discount_is_blocked(dm, monkeypatch):
    data = post({**MAKEUP, "event_type": "dm"}, {"private_dm": dm, "action": "ANSWER"}, monkeypatch)
    assert data["intent"] == "price_blocked"
    assert data["action"] == "HANDOFF"
    assert dm not in (data["private_dm"] or "")


def test_business_and_customer_figures_are_allowed(monkeypatch):
    payload = {**MAKEUP, "event_type": "dm", "message_text": "advance kitna? mera budget 20000 hai",
               "business": {**MAKEUP["business"], "policies": {"advance": "50% advance, travel charge ₹500 outside Patna"}}}
    dm = "Bridal look ₹18,000 se start, aapke 20000 budget me ho jayega. 50% advance hota hai, Patna ke bahar ₹500 travel."
    data = post(payload, {"private_dm": dm, "action": "ANSWER"}, monkeypatch)
    assert data["intent"] != "price_blocked"
    assert data["private_dm"] == dm


def test_offer_percent_from_the_post_is_allowed_but_not_its_price(monkeypatch):
    payload = {**MAKEUP, "event_type": "dm", "post_context": {"post_id": "ig_1", "caption": "Diwali offer 10% off, ₹15000 only"}}
    ok = post(payload, {"private_dm": "Diwali pe 10% off hai, bridal ₹18,000 se start.", "action": "ANSWER"}, monkeypatch)
    assert ok["intent"] != "price_blocked"
    blocked = post({**payload, "sender_id": "u2"}, {"private_dm": "Bridal sirf ₹15000 only.", "action": "ANSWER"}, monkeypatch)
    assert blocked["intent"] == "price_blocked"


def test_amount_in_item_description_is_a_catalog_price(monkeypatch):
    offering = {**MAKEUP["offerings"][0], "description": "Engagement add-on ₹6,500"}
    data = post({**MAKEUP, "event_type": "dm", "offerings": [offering]},
                {"private_dm": "Engagement add-on ₹6,500 ka hai.", "action": "ANSWER"}, monkeypatch)
    assert data["intent"] != "price_blocked"
