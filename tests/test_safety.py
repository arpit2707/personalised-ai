from unittest.mock import patch
import pytest
from app.services.guardrails import contains_blocked_content, SAFETY_REFUSAL
from app.services.gemini_service import GeminiService


@pytest.mark.parametrize("text", [
    "sexual content", "racial jokes", "harassment", "chemical-free",
    "poison", "posionous", "terrorism", "zeher", "\u091c\u0939\u0930", "SEXUAL",
    "\uff50\uff4f\uff49\uff53\uff4f\uff4e", "poi\u200bson", "refund for poison",
])
def test_blocked_topics(text):
    assert contains_blocked_content(text)


@pytest.mark.parametrize("text", ["shirt price?", "silk kurta", "size M available?", "refund please"])
def test_normal_shopping(text):
    assert not contains_blocked_content(text)


def test_input_never_calls_model():
    service = GeminiService()
    with patch.object(service, "_client") as client:
        assert service.generate("Brand voice", "poison")["private_dm"] == SAFETY_REFUSAL
        client.models.generate_content.assert_not_called()


def test_output_is_replaced():
    service = GeminiService()
    with patch("app.services.gemini_service.settings") as settings, patch.object(service, "_client") as client:
        settings.GEMINI_API_KEY = "test"
        client.models.generate_content.return_value.text = '{"private_dm": "chemical details"}'
        result = service.generate("Brand voice", "shirt price?")
        assert result["private_dm"] == SAFETY_REFUSAL
        assert result["intent"] == "safety_refusal"
