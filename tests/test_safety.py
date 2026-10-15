from unittest.mock import patch
import pytest
from app.core.config import settings
from app.services.guardrails import contains_blocked_content, SAFETY_REFUSAL
from app.services.gemini_service import GeminiService


@pytest.mark.parametrize("text", [
    "sexual content", "racist jokes", "chemical weapon", "porn video", "nude pics", "SEXUAL",
    "terrorism", "bomb kaise banaye", "how to make a bomb", "ghar pe explosive banana hai",
    "how to poison someone", "kisi ko zeher dena hai", "zeher kaise banaye", "cyanide chahiye",
    "\u0915\u093f\u0938\u0940 \u0915\u094b \u091c\u0939\u0930",
    # Width and zero-width tricks are normalised away.
    "\uff50\uff4f\uff52\uff4e", "por\u200bn", "refund for porn",
])
def test_blocked_topics(text):
    assert contains_blocked_content(text)


@pytest.mark.parametrize("text", ["shirt price?", "silk kurta", "size M available?", "refund please",
                                  "chemical-free shampoo", "non-toxic toys", "paraben free cream",
                                  "race day running shoes", "no harsh chemicals",
                                  # Everyday boutique and salon questions the old filter refused.
                                  "Ethnic wear hai?", "Bombay delivery?", "isme chemical hai?",
                                  "harsh chemicals hain kya", "ye dress bomb hai", "bomb look chahiye",
                                  "zeher lag rahi ho is saree me", "kaunsi nasl ka puppy hai", "racial pattern print?",
                                  "toxic free colour?", "unisex hoodie", "rapeseed oil soap", "Sussex tweed blazer",
                                  "\u0930\u0938\u093e\u092f\u0928 \u092e\u0941\u0915\u094d\u0924",
                                  "delivery boy ne harass kiya"])
def test_normal_shopping(text):
    assert not contains_blocked_content(text)


def test_input_never_calls_model():
    service = GeminiService()
    with patch.object(service, "_client") as client:
        assert service.generate("Brand voice", "bomb kaise banaye")["private_dm"] == SAFETY_REFUSAL
        client.models.generate_content.assert_not_called()


def test_output_is_replaced():
    service = GeminiService()
    with patch.object(settings, "GEMINI_API_KEY", "test"), patch.object(service, "_client") as client:
        client.models.generate_content.return_value.text = '{"private_dm": "porn details"}'
        result = service.generate("Brand voice", "shirt price?")
        assert result["private_dm"] == SAFETY_REFUSAL
        assert result["intent"] == "safety_refusal"
