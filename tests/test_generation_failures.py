import json
from unittest.mock import Mock
import pytest
from app.core.config import settings
from app.models.schemas import ModelReplyWire
from app.services.gemini_service import GeminiService, LLMUnavailable


def test_unconfigured_model_does_not_fabricate():
    with pytest.raises(LLMUnavailable):
        GeminiService().generate('Brand voice', 'size?')


@pytest.mark.parametrize('text', ['not json', 'null', '[]', '{"private_dm": ""}', '{"private_dm": 7}'])
def test_invalid_provider_responses(text, monkeypatch):
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'test')
    service = GeminiService()
    service._client = Mock()
    service._client.models.generate_content.return_value.text = text
    with pytest.raises(LLMUnavailable):
        service.generate('Brand voice', 'Hello')


def test_provider_exception_is_not_a_sales_reply(monkeypatch):
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'test')
    service = GeminiService()
    service._client = Mock()
    service._client.models.generate_content.side_effect = TimeoutError()
    with pytest.raises(LLMUnavailable):
        service.generate('Brand voice', 'Hello')


def test_reply_schema_has_no_free_form_objects():
    # The Gemini Developer API rejects additionalProperties; every reply would fail.
    assert "additionalProperties" not in json.dumps(ModelReplyWire.model_json_schema())


def test_collected_fields_come_back_as_a_dict(monkeypatch):
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'test')
    service = GeminiService()
    service._client = Mock()
    service._client.models.generate_content.return_value.text = json.dumps({
        "private_dm": "Noted!", "action": "ASK_FIELD",
        "collected_fields": [{"key": "city", "value": "Patna"}, {"key": "", "value": "x"}],
    })
    service._client.models.generate_content.return_value.usage_metadata = None
    result = service.generate('Brand voice', 'Patna')
    assert result["collected_fields"] == {"city": "Patna"}
    config = service._client.models.generate_content.call_args.kwargs["config"]
    assert config.response_schema is ModelReplyWire
