import json
import logging
import re
import threading
import time
from app.core.config import settings
from app.models.schemas import ModelReply, ModelReplyWire, PreferenceEvidence
from app.services.guardrails import SAFETY_INSTRUCTION, SAFETY_REFUSAL, contains_blocked_content

logger = logging.getLogger(__name__)

# The attempt loop reads the time only through these, so tests can move it.
_clock = time.monotonic
_sleep = time.sleep

# No new attempt starts with less budget than this: a model rarely answers faster.
MIN_ATTEMPT_MS = 6000
MAX_ATTEMPTS = 3
# A repetition loop otherwise runs until the timeout. Thinking tokens count
# against this too, so 1024 could cut off a normal reply after a long think.
GEMINI_MAX_OUTPUT_TOKENS = 2048
FALLBACK_MODEL = "gemini-3.6-flash"
LAST_RESORT_MODEL = "gemini-flash-lite-latest"


class LLMUnavailable(Exception):
    """No usable reply. code is what the backend shows the seller: llm_timeout,
    provider_error:<status or type>, invalid_output, empty_output or not_configured."""

    def __init__(self, message: str = "Generation failed", code: str = "provider_error"):
        super().__init__(message)
        self.code = code


class _EmptyOutput(Exception):
    """No reply text (safety block, or thinking used up the output tokens)."""

    def __init__(self, detail=None):
        super().__init__("Empty reply")
        self.detail = detail


class _InvalidOutput(Exception):
    """The model's answer does not fit the reply schema (detail: field paths only)."""

    def __init__(self, message="Invalid reply", detail=None):
        super().__init__(message)
        self.detail = detail


def _from_wire(result):
    """Turns the [{key, value}] list Gemini returns back into a dict."""
    if isinstance(result, dict) and isinstance(result.get("collected_fields"), list):
        result["collected_fields"] = {
            str(f["key"]): str(f["value"])
            for f in result["collected_fields"]
            if isinstance(f, dict) and f.get("key") and f.get("value") is not None
        }
    return result


_ACTIONS = {"SEND_LINK", "ASK_FIELD", "CREATE_LEAD", "HANDOFF", "ANSWER"}
_HANDOFF_REASONS = {"human_request", "complaint", "order_support", "purchase_assistance",
                    "missing_information", "conflicting_information"}
_OFFERING_TYPES = {"PRODUCTS": "PRODUCTS", "PRODUCT": "PRODUCTS", "SERVICES": "SERVICES", "SERVICE": "SERVICES"}
_TEXT_LIMITS = {"public_reply": 2000, "private_dm": 4000, "reasoning": 500}
_TRUE = {"true", "yes", "1"}


def _normalize(raw):
    """Fixes the small slips a model makes in an otherwise good answer, so the
    reply is not thrown away over them: "none" for no hand-off, a lowercase
    action, null lists, a number as a field value, the whole intent option list.

    Only the shape is repaired; the text itself still meets every guard.
    """
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    action = str(out.get("action") or "").strip().upper()
    out["action"] = action if action in _ACTIONS else None
    reason = str(out.get("handoff_reason") or "").strip().lower()
    out["handoff_reason"] = reason if reason in _HANDOFF_REASONS else None
    out["offering_type"] = _OFFERING_TYPES.get(str(out.get("offering_type") or "").strip().upper())
    for key in ("previous_answer_unresolved", "buying_interest"):
        value = out.get(key)
        out[key] = value.strip().lower() in _TRUE if isinstance(value, str) else bool(value)
    intent = str(out.get("intent") or "").strip()
    # "price_inquiry | availability | ..." means the model copied the options, not chose one.
    out["intent"] = "general" if not intent or "|" in intent else intent[:64]
    ids = out.get("offering_ids")
    ids = [ids] if isinstance(ids, str) else ids if isinstance(ids, list) else []
    out["offering_ids"] = [str(i) for i in ids if isinstance(i, (str, int)) and str(i).strip()][:30]
    preferences = []
    for pref in out.get("preferences") if isinstance(out.get("preferences"), list) else []:
        try:
            if isinstance(pref, dict):
                pref = {**pref, "key": str(pref.get("key") or "").strip().lower()}
            preferences.append(PreferenceEvidence.model_validate(pref).model_dump())
        except ValueError:
            continue
    out["preferences"] = preferences[:3]
    fields = out.get("collected_fields")
    out["collected_fields"] = {
        str(k): str(v) for k, v in fields.items()
        if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip()
    } if isinstance(fields, dict) else {}
    if out.get("public_reply") is not None and not isinstance(out["public_reply"], str):
        out["public_reply"] = None
    if out.get("reasoning") is not None:
        out["reasoning"] = str(out["reasoning"])
    if out.get("private_dm") is None:
        out["private_dm"] = ""
    # A DM must never be a number or a list; that stays invalid. An empty DM
    # with a usable public reply takes the public reply.
    if isinstance(out["private_dm"], str) and not out["private_dm"].strip() and (out.get("public_reply") or "").strip():
        out["private_dm"] = out["public_reply"]
    for key, limit in _TEXT_LIMITS.items():
        if isinstance(out.get(key), str) and len(out[key]) > limit:
            out[key] = out[key][:limit]
    return out


OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _inline_refs(node, defs):
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline_refs(defs[node["$ref"].split("/")[-1]], defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _reply_schema() -> dict:
    """ModelReplyWire as plain JSON Schema (no $refs) for OpenAI and Claude."""
    schema = ModelReplyWire.model_json_schema()
    return _inline_refs(schema, schema.get("$defs", {}))


def _gemini_major(model: str):
    match = re.match(r"(?:models/)?gemini-(\d+)", model or "")
    return int(match.group(1)) if match else None


def _thinking_config(model: str):
    """Low thinking: Gemini 3 thinks at length by default, which a long prompt turns into a timeout."""
    from google.genai import types
    name = (model or "").split("/")[-1]
    major = _gemini_major(name)
    if major is not None and major >= 3:
        level = getattr(types, "ThinkingLevel", None)
        # An SDK older than thinking levels sends none rather than failing every reply.
        return types.ThinkingConfig(thinking_level=level.LOW) if level else None
    if name.startswith("gemini-2.5"):
        # Pro cannot switch thinking off; flash and flash-lite can.
        return types.ThinkingConfig(thinking_budget=256 if "pro" in name else 0)
    return None


def _temperature(provider: str, model: str, event_type: str):
    """None leaves the provider default: OpenAI's reasoning models accept only that,
    and Gemini 3 is tuned for it (a low temperature can make it loop)."""
    if provider == "OPENAI" or (provider == "GEMINI" and _gemini_major(model) != 2):
        return None
    # A DM is a conversation; a public comment reply stays close to the facts.
    return 0.2 if event_type == "comment" else 0.6


def _reasoning_effort(model: str):
    name = (model or "").lower()
    if name.startswith("gpt-5"):
        return "minimal"
    if re.match(r"o\d", name):
        return "low"  # the o-series has no "minimal"
    return None


def _fallback_model(primary: str) -> str:
    if settings.LLM_FALLBACK_MODEL:
        return settings.LLM_FALLBACK_MODEL
    return FALLBACK_MODEL if primary != FALLBACK_MODEL else LAST_RESORT_MODEL


def _is_timeout(exc) -> bool:
    import httpx
    return isinstance(exc, (TimeoutError, httpx.TimeoutException))


def _status_of(exc):
    import httpx
    status = None
    try:
        from google.genai import errors
        if isinstance(exc, errors.APIError):
            status = exc.code
    except ImportError:
        pass
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", None)
    return status if isinstance(status, int) else None


def _error_paths(exc) -> str:
    # Field paths only: the values are the model's words about the customer.
    return ",".join(".".join(str(p) for p in e["loc"]) for e in exc.errors()[:5])


def _checked_reply(result) -> dict:
    """ModelReply.model_validate, with a schema miss named as the model's fault.

    Only this validation is "invalid_output": a pydantic error while building the
    SDK request (an SDK too old for a config field) is ours and is not retried.
    """
    from pydantic import ValidationError
    try:
        return ModelReply.model_validate(result).model_dump()
    except ValidationError as exc:
        raise _InvalidOutput(detail=_error_paths(exc)) from exc


def _failure(exc):
    """(code, provider HTTP status, detail) for one failed attempt; never customer text."""
    from pydantic import ValidationError
    if isinstance(exc, _EmptyOutput):
        return "empty_output", None, exc.detail
    if isinstance(exc, _InvalidOutput):
        return "invalid_output", None, exc.detail
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_output", None, None
    if isinstance(exc, ValidationError):
        # Not the model's answer: the SDK refused the request we built.
        return "provider_error:ValidationError", None, "sdk_request:" + _error_paths(exc)
    if _is_timeout(exc):
        return "llm_timeout", None, None
    status = _status_of(exc)
    if status:
        return f"provider_error:{status}", status, None
    return f"provider_error:{type(exc).__name__}", None, None


def _retryable(exc, code: str, status) -> bool:
    import httpx
    if code in ("llm_timeout", "empty_output", "invalid_output"):
        return True
    if status is not None:
        return status in (408, 429) or status >= 500
    # A dropped connection; a 4xx (bad key, bad request) would only fail again.
    return isinstance(exc, (httpx.TransportError, ConnectionError))


def _finish_reason(response):
    try:
        candidate = (getattr(response, "candidates", None) or [None])[0]
        reason = getattr(candidate, "finish_reason", None) or getattr(
            getattr(response, "prompt_feedback", None), "block_reason", None)
        return getattr(reason, "name", None) or (str(reason) if reason else None)
    except Exception:
        return None


class GeminiService:
    """Generates the reply JSON. Gemini with the service's own key by default;
    the backend can name another provider (OpenAI, Claude) or key per request."""

    MAX_CACHED_CLIENTS = 32

    def __init__(self):
        self._client = None
        # Per dashboard key: a new client per request repeats the TLS setup every time.
        self._clients = {}
        self._lock = threading.Lock()
        # Models that rejected a thinking or reasoning setting go without it from then on.
        self._no_thinking = set()
        self._no_reasoning = set()

    def warm_up(self):
        """Imports the SDK and builds the default client before the first customer waits for it."""
        try:
            from google import genai  # noqa: F401
            if settings.GEMINI_API_KEY:
                self._gemini_client()
        except Exception as exc:
            logger.warning("llm_warm_up_failed type=%s", type(exc).__name__)

    def _gemini_client(self, api_key: str = None):
        from google import genai
        from google.genai import types
        # The SDK's own retries would stack their timeouts on ours; generate()
        # retries within one overall budget instead.
        options = types.HttpOptions(timeout=settings.LLM_TIMEOUT_MS,
                                    retry_options=types.HttpRetryOptions(attempts=1))
        if not api_key:
            if self._client is None:
                self._client = genai.Client(api_key=settings.GEMINI_API_KEY, http_options=options)
            return self._client
        with self._lock:
            client = self._clients.pop(api_key, None) or genai.Client(api_key=api_key, http_options=options)
            self._clients[api_key] = client  # most recently used last
            while len(self._clients) > self.MAX_CACHED_CLIENTS:
                self._clients.pop(next(iter(self._clients)))
            return client

    def _call_gemini(self, system: str, user_prompt: str, model: str, api_key: str = None,
                     temperature: float = None, timeout_ms: int = None):
        from google.genai import errors
        thinking = None if model in self._no_thinking else _thinking_config(model)
        try:
            return self._gemini_request(system, user_prompt, model, api_key, temperature, timeout_ms, thinking)
        except errors.ClientError as exc:
            if thinking is None or exc.code != 400 or "thinking" not in str(exc).lower():
                raise
        self._no_thinking.add(model)
        logger.warning("llm_thinking_rejected provider=GEMINI model=%s", model)
        return self._gemini_request(system, user_prompt, model, api_key, temperature, timeout_ms, None)

    def _gemini_request(self, system, user_prompt, model, api_key, temperature, timeout_ms, thinking):
        from google.genai import types
        response = self._gemini_client(api_key).models.generate_content(
            model=model, contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=ModelReplyWire,
                temperature=temperature,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                thinking_config=thinking,
                http_options=types.HttpOptions(timeout=timeout_ms) if timeout_ms else None,
            ),
        )
        if not response.text:
            raise _EmptyOutput(_finish_reason(response))
        usage = getattr(response, "usage_metadata", None)
        return (json.loads(response.text),
                getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None))

    def _call_openai(self, system: str, user_prompt: str, model: str, api_key: str, timeout_ms: int = None):
        import httpx
        # Temperature is left out: OpenAI's reasoning models only accept the default.
        body = {"model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user_prompt}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "reply", "schema": _reply_schema(), "strict": False}}}
        effort = _reasoning_effort(model)
        if effort and model not in self._no_reasoning:
            # A short DM needs no long private reasoning before it.
            body["reasoning_effort"] = effort

        def post():
            res = httpx.post(OPENAI_URL, timeout=(timeout_ms or settings.LLM_TIMEOUT_MS) / 1000,
                             headers={"Authorization": f"Bearer {api_key}"}, json=body)
            res.raise_for_status()
            return res
        try:
            res = post()
        except httpx.HTTPStatusError as exc:
            if "reasoning_effort" not in body or getattr(exc.response, "status_code", None) != 400 \
                    or "reasoning" not in str(getattr(exc.response, "text", "")).lower():
                raise
            self._no_reasoning.add(model)
            body.pop("reasoning_effort")
            res = post()
        body = res.json()
        usage = body.get("usage") or {}
        text = body["choices"][0]["message"]["content"]
        if not text:
            raise _EmptyOutput(body["choices"][0].get("finish_reason"))
        return json.loads(text), usage.get("prompt_tokens"), usage.get("completion_tokens")

    def _call_claude(self, system: str, user_prompt: str, model: str, api_key: str,
                     temperature: float = None, timeout_ms: int = None):
        import httpx
        # A forced tool call makes Claude answer in the reply schema.
        body = {"model": model, "max_tokens": 2048, "system": system,
                "tools": [{"name": "reply", "description": "Send the reply.",
                           "input_schema": _reply_schema()}],
                "tool_choice": {"type": "tool", "name": "reply"},
                "messages": [{"role": "user", "content": user_prompt}]}
        if temperature is not None:
            body["temperature"] = temperature
        res = httpx.post(ANTHROPIC_URL, timeout=(timeout_ms or settings.LLM_TIMEOUT_MS) / 1000,
                         headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}, json=body)
        res.raise_for_status()
        body = res.json()
        block = next((b for b in body.get("content", []) if b.get("type") == "tool_use"), None)
        if block is None:
            raise _InvalidOutput("No structured reply")
        usage = body.get("usage") or {}
        return block.get("input"), usage.get("input_tokens"), usage.get("output_tokens")

    def _call(self, provider, system, user_prompt, model, api_key, temperature, timeout_ms):
        if provider == "OPENAI":
            return self._call_openai(system, user_prompt, model, api_key, timeout_ms)
        if provider == "CLAUDE":
            return self._call_claude(system, user_prompt, model, api_key, temperature, timeout_ms)
        return self._call_gemini(system, user_prompt, model, api_key, temperature, timeout_ms)

    def generate(self, system_instruction: str, user_prompt: str, customer_text: str = None, llm=None,
                 event_type: str = "dm") -> dict:
        """The checked reply, or LLMUnavailable within LLM_TOTAL_BUDGET_MS.

        A timeout, a 429/5xx, an empty answer or JSON that does not fit the
        schema is tried again while the budget lasts; on Gemini (the service's
        key or a dashboard one) the second try uses the fallback model.
        """
        refusal = {"public_reply": SAFETY_REFUSAL, "private_dm": SAFETY_REFUSAL,
                   "intent": "safety_refusal", "reasoning": None}
        # Callers that pass the customer's own words have the filter applied to
        # those only, so seller text such as a "chemical-free" catalog item or
        # caption does not block every reply on that page.
        checked = customer_text if customer_text is not None else [system_instruction, user_prompt]
        if contains_blocked_content(checked):
            return refusal
        provider = llm.provider if llm else "GEMINI"
        model = llm.model if llm else settings.DEFAULT_MODEL
        api_key = llm.api_key.get_secret_value() if llm else None
        if not llm and not settings.GEMINI_API_KEY:
            logger.error("llm_generation_failed provider=GEMINI model=%s code=not_configured", model)
            raise LLMUnavailable("Generation is not configured", "not_configured")
        system = system_instruction + "\n\n" + SAFETY_INSTRUCTION
        # Any Gemini key can call the fallback model, which often still answers
        # during a 503 "high demand" burst on the primary. An OpenAI or Claude
        # model the backend chose is retried as it is.
        models = [model, _fallback_model(model)] if provider == "GEMINI" else [model]
        started = _clock()
        deadline = started + settings.LLM_TOTAL_BUDGET_MS / 1000
        code, last = "provider_error", None
        for attempt in range(MAX_ATTEMPTS):
            remaining_ms = (deadline - _clock()) * 1000
            if attempt and remaining_ms < MIN_ATTEMPT_MS:
                break
            current = models[attempt % len(models)]
            timeout_ms = int(max(1000, min(settings.LLM_TIMEOUT_MS, remaining_ms)))
            try:
                raw, tokens_in, tokens_out = self._call(provider, system, user_prompt, current, api_key,
                                                        _temperature(provider, current, event_type), timeout_ms)
                result = _normalize(_from_wire(raw))
                if contains_blocked_content(result) or (isinstance(result, dict) and result.get("intent") == "safety_refusal"):
                    return refusal
                result = _checked_reply(result)
                if not result['private_dm'].strip() and not result['handoff_reason']:
                    raise _EmptyOutput()
                logger.info("llm_generation provider=%s model=%s attempt=%d elapsed_ms=%d input_tokens=%s "
                            "output_tokens=%s", provider, current, attempt + 1, (_clock() - started) * 1000,
                            tokens_in, tokens_out)
                return result
            except Exception as exc:
                last = exc
                code, status, detail = _failure(exc)
                # Do not log customer text or provider exceptions containing request bodies.
                logger.warning("llm_generation_failed provider=%s model=%s attempt=%d code=%s status=%s "
                               "elapsed_ms=%d%s", provider, current, attempt + 1, code, status,
                               (_clock() - started) * 1000, f" detail={detail}" if detail else "")
                switching = models[(attempt + 1) % len(models)] != current
                if not (_retryable(exc, code, status) or (status == 404 and switching)):
                    break
                if attempt + 1 < MAX_ATTEMPTS and (status == 429 or (status or 0) >= 500):
                    # A short pause: these come in bursts. Never eat the next attempt's budget.
                    _sleep(max(0.0, min(attempt + 1.0, deadline - _clock() - MIN_ATTEMPT_MS / 1000)))
        raise LLMUnavailable("Generation failed", code) from last


gemini_service = GeminiService()
