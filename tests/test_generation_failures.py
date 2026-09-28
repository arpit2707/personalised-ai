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


REPLY = {'public_reply': None, 'private_dm': 'Ji, available hai.', 'intent': 'product_query',
         'reasoning': None, 'collected_fields': [{'key': 'city', 'value': 'Patna'}]}


def _llm(provider, model):
    from app.models.schemas import LLMConfig
    return LLMConfig(provider=provider, api_key='sk-test-123456', model=model)


def test_openai_provider_from_the_backend(monkeypatch):
    import httpx
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    calls = []

    def post(url, **kw):
        calls.append((url, kw))
        return Mock(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': json.dumps(REPLY)}}], 'usage': {}})
    monkeypatch.setattr(httpx, 'post', post)
    out = GeminiService().generate('Brand voice', 'Hello', None, _llm('OPENAI', 'gpt-5-mini'))
    assert out['private_dm'] == 'Ji, available hai.'
    assert out['collected_fields'] == {'city': 'Patna'}
    url, kw = calls[0]
    assert url.endswith('/chat/completions')
    assert kw['headers']['Authorization'] == 'Bearer sk-test-123456'
    assert kw['json']['model'] == 'gpt-5-mini'
    assert '$ref' not in json.dumps(kw['json']['response_format'])


def test_claude_provider_answers_through_a_forced_tool(monkeypatch):
    import httpx
    calls = []

    def post(url, **kw):
        calls.append((url, kw))
        return Mock(raise_for_status=lambda: None, json=lambda: {
            'content': [{'type': 'tool_use', 'name': 'reply', 'input': REPLY}], 'usage': {}})
    monkeypatch.setattr(httpx, 'post', post)
    out = GeminiService().generate('Brand voice', 'Hello', None, _llm('CLAUDE', 'claude-sonnet-5-5'))
    assert out['private_dm'] == 'Ji, available hai.'
    url, kw = calls[0]
    assert url.endswith('/v1/messages')
    assert kw['headers']['x-api-key'] == 'sk-test-123456'
    assert kw['json']['tool_choice'] == {'type': 'tool', 'name': 'reply'}


def test_provider_http_error_is_unavailable(monkeypatch):
    import httpx

    def post(url, **kw):
        raise httpx.HTTPStatusError('401', request=Mock(), response=Mock())
    monkeypatch.setattr(httpx, 'post', post)
    with pytest.raises(LLMUnavailable):
        GeminiService().generate('Brand voice', 'Hello', None, _llm('OPENAI', 'gpt-5-mini'))


def test_api_key_is_not_in_the_request_repr():
    assert 'sk-test-123456' not in repr(_llm('OPENAI', 'gpt-5-mini'))
