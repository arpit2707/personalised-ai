"""The model call answers within its time budget: retries, fallback model,
thinking settings, temperature and the small output repairs before validation."""
import json
import logging
from unittest.mock import Mock

import httpx
import pytest
from google.genai import errors, types

from app.core.config import settings
from app.models.schemas import LLMConfig, ModelReply
from app.services import gemini_service as gs
from app.services.gemini_service import GeminiService, LLMUnavailable, _normalize

GOOD = {"private_dm": "Haan ji, available hai.", "intent": "availability"}


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(gs, "_clock", c)
    monkeypatch.setattr(gs, "_sleep", c.sleep)
    return c


def scripted(monkeypatch, clock, *steps, model="gemini-3.6-flash"):
    """A service on its own Gemini key whose client plays (seconds, result) steps in turn."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test")
    monkeypatch.setattr(settings, "DEFAULT_MODEL", model)
    monkeypatch.setattr(settings, "LLM_FALLBACK_MODEL", "")
    service = GeminiService()
    service._client = Mock()
    calls = []
    plan = list(steps)

    def generate_content(*, model, contents, config):
        calls.append((model, config))
        seconds, result = plan.pop(0)
        clock.now += seconds
        if isinstance(result, Exception):
            raise result
        text = result if result is None or isinstance(result, str) else json.dumps(result)
        return Mock(text=text, usage_metadata=None, candidates=[Mock(finish_reason=None)])
    service._client.models.generate_content.side_effect = generate_content
    return service, calls


def api_error(cls, code, message):
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}})


# ---------------------------------------------------------------- budget and fallback

def test_timeout_falls_back_to_the_second_model_within_the_budget(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (25, httpx.ReadTimeout("slow")), (3, GOOD), model="gemini-3.5-pro")
    out = service.generate("Brand voice", "Hello")
    assert out["private_dm"] == GOOD["private_dm"]
    assert [m for m, _ in calls] == ["gemini-3.5-pro", "gemini-3.6-flash"]
    # Each attempt gets its own timeout, and the second only what is left of the budget.
    assert [c.http_options.timeout for _, c in calls] == [25000, 20000]


def test_the_budget_ends_the_attempts(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (25, httpx.ReadTimeout("slow")), (20, httpx.ReadTimeout("slow")),
                              (1, GOOD))
    with pytest.raises(LLMUnavailable) as err:
        service.generate("Brand voice", "Hello")
    assert err.value.code == "llm_timeout"
    assert len(calls) == 2
    assert clock.now - 1000 <= settings.LLM_TOTAL_BUDGET_MS / 1000


def test_default_fallback_when_the_primary_is_already_the_fallback(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (1, api_error(errors.ServerError, 503, "high demand")), (1, GOOD))
    service.generate("Brand voice", "Hello")
    assert [m for m, _ in calls] == ["gemini-3.6-flash", "gemini-flash-lite-latest"]
    # A short pause between the two, well inside the budget.
    assert clock.slept and max(clock.slept) <= 3


def test_configured_fallback_model(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (1, api_error(errors.ClientError, 429, "quota")), (1, GOOD))
    monkeypatch.setattr(settings, "LLM_FALLBACK_MODEL", "gemini-2.5-flash")
    service.generate("Brand voice", "Hello")
    assert [m for m, _ in calls] == ["gemini-3.6-flash", "gemini-2.5-flash"]


def test_dashboard_gemini_key_also_falls_back(monkeypatch, clock):
    # A 503 burst on the workspace's model: the same key still reaches the fallback.
    service, calls = scripted(monkeypatch, clock, (1, api_error(errors.ServerError, 503, "high demand")), (2, GOOD))
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    service._clients["AIza-dashboard-key"] = service._client
    llm = LLMConfig(provider="GEMINI", api_key="AIza-dashboard-key", model="gemini-3.6-flash")
    assert service.generate("Brand voice", "Hello", None, llm)["private_dm"] == GOOD["private_dm"]
    assert [m for m, _ in calls] == ["gemini-3.6-flash", "gemini-flash-lite-latest"]
    assert list(service._clients) == ["AIza-dashboard-key"]


def test_dashboard_gemini_key_uses_the_configured_fallback(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (25, httpx.ReadTimeout("slow")), (2, GOOD))
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_FALLBACK_MODEL", "gemini-2.5-flash")
    service._clients["AIza-dashboard-key"] = service._client
    llm = LLMConfig(provider="GEMINI", api_key="AIza-dashboard-key", model="gemini-3.5-pro")
    service.generate("Brand voice", "Hello", None, llm)
    assert [m for m, _ in calls] == ["gemini-3.5-pro", "gemini-2.5-flash"]


@pytest.mark.parametrize("first,code", [
    (None, "empty_output"),
    ("not json", "invalid_output"),
    ({"private_dm": ["a list"]}, "invalid_output"),
    ({"private_dm": "  "}, "empty_output"),
])
def test_empty_or_invalid_output_is_tried_again(first, code, monkeypatch, clock, caplog):
    service, calls = scripted(monkeypatch, clock, (1, first), (1, GOOD))
    with caplog.at_level(logging.WARNING, logger="app.services.gemini_service"):
        assert service.generate("Brand voice", "Hello")["private_dm"] == GOOD["private_dm"]
    assert len(calls) == 2
    assert f"code={code}" in caplog.text


def test_failure_code_after_every_attempt_failed(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, *[(1, None)] * 3)
    with pytest.raises(LLMUnavailable) as err:
        service.generate("Brand voice", "Hello")
    assert err.value.code == "empty_output"
    assert len(calls) == gs.MAX_ATTEMPTS


def test_a_bad_key_is_not_retried(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (1, api_error(errors.ClientError, 403, "API key not valid")), (1, GOOD))
    with pytest.raises(LLMUnavailable) as err:
        service.generate("Brand voice", "Hello")
    assert err.value.code == "provider_error:403"
    assert len(calls) == 1


def test_unknown_model_switches_to_the_fallback(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (1, api_error(errors.ClientError, 404, "model not found")), (1, GOOD),
                              model="gemini-9-typo")
    service.generate("Brand voice", "Hello")
    assert [m for m, _ in calls] == ["gemini-9-typo", "gemini-3.6-flash"]


def test_not_configured_is_logged(monkeypatch, caplog):
    with caplog.at_level(logging.ERROR, logger="app.services.gemini_service"):
        with pytest.raises(LLMUnavailable) as err:
            GeminiService().generate("Brand voice", "Hello")
    assert err.value.code == "not_configured"
    assert "code=not_configured" in caplog.text


def test_an_sdk_that_refuses_our_request_is_not_called_invalid_output(monkeypatch, clock, caplog):
    # An SDK too old for a config field fails while the request is built, before
    # the model ever answers: not the model's fault and not worth two more tries.
    from pydantic import ValidationError
    try:
        types.GenerateContentConfig(no_such_field=1)
    except ValidationError as exc:
        refused = exc

    def config(**_):
        raise refused
    service, calls = scripted(monkeypatch, clock, (1, GOOD))
    monkeypatch.setattr(types, "GenerateContentConfig", config)
    with caplog.at_level(logging.WARNING, logger="app.services.gemini_service"):
        with pytest.raises(LLMUnavailable) as err:
            service.generate("Brand voice", "Hello")
    assert err.value.code == "provider_error:ValidationError"
    assert calls == []
    assert "detail=sdk_request:no_such_field" in caplog.text
    assert caplog.text.count("llm_generation_failed") == 1


def test_failure_logs_carry_no_customer_text(monkeypatch, clock, caplog):
    service, _ = scripted(monkeypatch, clock, *[(1, {"private_dm": 7, "reasoning": "meri shaadi 15 Nov"})] * 3)
    with caplog.at_level(logging.WARNING, logger="app.services.gemini_service"):
        with pytest.raises(LLMUnavailable):
            service.generate("Brand voice", "meri shaadi 15 Nov ko hai", "meri shaadi 15 Nov ko hai")
    assert "detail=private_dm" in caplog.text
    assert "status=None" in caplog.text and "elapsed_ms=" in caplog.text
    assert "shaadi" not in caplog.text


# ---------------------------------------------------------------- request settings

def test_gemini_3_request_settings(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (1, GOOD))
    service.generate("Brand voice", "Hello", None, None, "dm")
    config = calls[0][1]
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert config.temperature is None  # Gemini 3 keeps its default
    assert config.max_output_tokens == 2048
    assert config.http_options.timeout == settings.LLM_TIMEOUT_MS


@pytest.mark.parametrize("event,temperature", [("dm", 0.6), ("comment", 0.2)])
def test_gemini_2_5_request_settings(event, temperature, monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (1, GOOD), model="gemini-2.5-flash")
    service.generate("Brand voice", "Hello", None, None, event)
    config = calls[0][1]
    assert config.thinking_config.thinking_budget == 0
    assert config.thinking_config.thinking_level is None
    assert config.temperature == temperature


def test_thinking_settings_per_model():
    assert gs._thinking_config("gemini-3.6-flash").thinking_level == types.ThinkingLevel.LOW
    assert gs._thinking_config("models/gemini-3-pro-preview").thinking_level == types.ThinkingLevel.LOW
    assert gs._thinking_config("gemini-2.5-pro").thinking_budget == 256
    assert gs._thinking_config("gemini-2.5-flash-lite").thinking_budget == 0
    assert gs._thinking_config("gemini-2.0-flash") is None
    assert gs._thinking_config("gemini-flash-lite-latest") is None


def test_rejected_thinking_setting_is_retried_without_it(monkeypatch, clock):
    rejected = api_error(errors.ClientError, 400, "Thinking level is not supported for this model.")
    service, calls = scripted(monkeypatch, clock, (1, rejected), (1, GOOD), (1, GOOD))
    assert service.generate("Brand voice", "Hello")["private_dm"] == GOOD["private_dm"]
    assert [m for m, _ in calls] == ["gemini-3.6-flash", "gemini-3.6-flash"]
    assert calls[0][1].thinking_config is not None and calls[1][1].thinking_config is None
    # Remembered: the next reply does not pay for the rejected call again.
    service.generate("Brand voice", "Hello")
    assert calls[2][1].thinking_config is None


def test_other_bad_requests_are_not_retried_without_thinking(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (1, api_error(errors.ClientError, 400, "Invalid schema")), (1, GOOD))
    with pytest.raises(LLMUnavailable) as err:
        service.generate("Brand voice", "Hello")
    assert err.value.code == "provider_error:400"
    assert len(calls) == 1


def test_sdk_does_not_retry_on_its_own_and_clients_are_cached(monkeypatch):
    from google import genai
    made = []
    monkeypatch.setattr(genai, "Client", lambda **kw: made.append(kw) or Mock())
    service = GeminiService()
    first = service._gemini_client("key-one")
    assert service._gemini_client("key-one") is first
    assert service._gemini_client("key-two") is not first
    assert len(made) == 2
    assert made[0]["http_options"].retry_options.attempts == 1


def test_warm_up_survives_a_missing_sdk(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "google.genai", None)
    GeminiService().warm_up()


def test_startup_runs_purge_and_warm_up():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200


# ---------------------------------------------------------------- other providers

REPLY = {"private_dm": "Ji, available hai.", "intent": "availability"}


def _openai_post(calls, fail_first=None):
    def post(url, **kw):
        calls.append(kw["json"].copy())
        if fail_first and len(calls) == 1:
            raise fail_first
        return Mock(raise_for_status=lambda: None, json=lambda: {
            "choices": [{"message": {"content": json.dumps(REPLY)}}], "usage": {}})
    return post


@pytest.mark.parametrize("model,effort", [("gpt-5-mini", "minimal"), ("o4-mini", "low"), ("gpt-4.1", None)])
def test_openai_reasoning_effort(model, effort, monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "post", _openai_post(calls))
    GeminiService().generate("Brand voice", "Hello", None, LLMConfig(provider="OPENAI", api_key="sk-test-123456", model=model))
    assert calls[0].get("reasoning_effort") == effort
    assert "temperature" not in calls[0]


def test_openai_rejected_reasoning_effort_is_dropped(monkeypatch):
    calls = []
    rejected = httpx.HTTPStatusError("400", request=Mock(), response=Mock(
        status_code=400, text='{"error": "Unsupported value: reasoning_effort"}'))
    monkeypatch.setattr(httpx, "post", _openai_post(calls, rejected))
    out = GeminiService().generate("Brand voice", "Hello", None,
                                   LLMConfig(provider="OPENAI", api_key="sk-test-123456", model="gpt-5.9"))
    assert out["private_dm"] == REPLY["private_dm"]
    assert "reasoning_effort" in calls[0] and "reasoning_effort" not in calls[1]


@pytest.mark.parametrize("event,temperature", [("dm", 0.6), ("comment", 0.2)])
def test_claude_temperature_and_retry(event, temperature, monkeypatch, clock):
    calls = []

    def post(url, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise httpx.ConnectError("reset")
        return Mock(raise_for_status=lambda: None, json=lambda: {
            "content": [{"type": "tool_use", "name": "reply", "input": REPLY}], "usage": {}})
    monkeypatch.setattr(httpx, "post", post)
    out = GeminiService().generate("Brand voice", "Hello", None,
                                   LLMConfig(provider="CLAUDE", api_key="sk-test-123456", model="claude-model"), event)
    assert out["private_dm"] == REPLY["private_dm"]
    assert len(calls) == 2
    # A Claude (or OpenAI) model is retried as it is: no Gemini fallback for that key.
    assert calls[0]["json"]["model"] == calls[1]["json"]["model"] == "claude-model"
    assert calls[1]["json"]["temperature"] == temperature
    assert calls[1]["timeout"] <= settings.LLM_TIMEOUT_MS / 1000


# ---------------------------------------------------------------- output repairs

def test_normalize_keeps_a_good_answer_with_small_slips():
    raw = {
        "private_dm": "50 log ke liye ho jayega!", "handoff_reason": "none", "action": "answer",
        "offering_ids": None, "preferences": None, "collected_fields": {"guests": 50, "date": None, "ok": True},
        "intent": "price_inquiry | availability | details | booking | general | complaint",
        "previous_answer_unresolved": None, "buying_interest": "true", "offering_type": "null",
    }
    out = ModelReply.model_validate(_normalize(raw))
    assert out.handoff_reason is None
    assert out.action == "ANSWER"
    assert out.offering_ids == [] and out.preferences == []
    assert out.collected_fields == {"guests": "50"}
    assert out.intent == "general"
    assert out.previous_answer_unresolved is False and out.buying_interest is True
    assert out.offering_type is None


@pytest.mark.parametrize("field,value,expected", [
    ("handoff_reason", "Missing_Information", "missing_information"),
    ("handoff_reason", "", None),
    ("handoff_reason", "because", None),
    ("action", "create_lead", "CREATE_LEAD"),
    ("action", "SEND_LINK | ASK_FIELD", None),
    ("offering_type", "service", "SERVICES"),
    ("intent", "x" * 70, "x" * 64),
    ("intent", None, "general"),
])
def test_normalize_single_fields(field, value, expected):
    out = ModelReply.model_validate(_normalize({"private_dm": "Hi", field: value}))
    assert getattr(out, field) == expected


def test_normalize_trims_lists_and_text():
    prefs = [{"key": "Size", "value": "XL", "evidence": "I wear XL"}, {"key": "mood", "value": "x", "evidence": "x"},
             {"key": "color", "value": "red", "evidence": "red please"}, {"key": "language", "value": "hindi", "evidence": "hindi me"},
             {"key": "size", "value": "M", "evidence": "M bhi"}]
    out = ModelReply.model_validate(_normalize({
        "private_dm": "Noted", "preferences": prefs, "reasoning": "r" * 600, "offering_ids": "o1",
        "public_reply": "p" * 2500}))
    assert [p.key for p in out.preferences] == ["size", "color", "language"]
    assert len(out.reasoning) == 500 and len(out.public_reply) == 2000
    assert out.offering_ids == ["o1"]


def test_normalize_uses_the_public_reply_for_an_empty_dm():
    out = ModelReply.model_validate(_normalize({"private_dm": "", "public_reply": "Haan, available hai!"}))
    assert out.private_dm == "Haan, available hai!"
    # A DM that is not text at all stays invalid.
    with pytest.raises(ValueError):
        ModelReply.model_validate(_normalize({"private_dm": 7}))


def test_wire_reply_with_slips_reaches_the_caller(monkeypatch, clock):
    service, calls = scripted(monkeypatch, clock, (1, {
        "private_dm": "Ji!", "action": "ask_field", "handoff_reason": "null",
        "collected_fields": [{"key": "guests", "value": 50}], "preferences": None}))
    out = service.generate("Brand voice", "50 guests")
    assert len(calls) == 1
    assert out["action"] == "ASK_FIELD" and out["collected_fields"] == {"guests": "50"}
