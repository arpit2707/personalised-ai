import json
import logging
import time
from app.core.config import settings
from app.models.schemas import ModelReply, ModelReplyWire
from app.services.guardrails import SAFETY_INSTRUCTION, SAFETY_REFUSAL, contains_blocked_content

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    pass


def _from_wire(result):
    """Turns the [{key, value}] list Gemini returns back into a dict."""
    if isinstance(result, dict) and isinstance(result.get("collected_fields"), list):
        result["collected_fields"] = {
            str(f["key"]): str(f["value"])
            for f in result["collected_fields"]
            if isinstance(f, dict) and f.get("key") and f.get("value") is not None
        }
    return result


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


class GeminiService:
    """Generates the reply JSON. Gemini with the service's own key by default;
    the backend can name another provider (OpenAI, Claude) or key per request."""

    def __init__(self):
        self._client = None

    def _gemini_client(self, api_key: str = None):
        from google import genai
        from google.genai import types
        options = types.HttpOptions(timeout=settings.LLM_TIMEOUT_MS,
                                    retry_options=types.HttpRetryOptions(attempts=2))
        if api_key:
            return genai.Client(api_key=api_key, http_options=options)
        if self._client is None:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY, http_options=options)
        return self._client

    def _call_gemini(self, system: str, user_prompt: str, model: str, api_key: str = None):
        from google.genai import types
        response = self._gemini_client(api_key).models.generate_content(
            model=model, contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=ModelReplyWire,
                temperature=0.2,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        return (json.loads(response.text or "null"),
                getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None))

    def _call_openai(self, system: str, user_prompt: str, model: str, api_key: str):
        import httpx
        # Temperature is left out: OpenAI's reasoning models only accept the default.
        res = httpx.post(OPENAI_URL, timeout=settings.LLM_TIMEOUT_MS / 1000,
                         headers={"Authorization": f"Bearer {api_key}"},
                         json={"model": model,
                               "messages": [{"role": "system", "content": system},
                                            {"role": "user", "content": user_prompt}],
                               "response_format": {"type": "json_schema", "json_schema": {
                                   "name": "reply", "schema": _reply_schema(), "strict": False}}})
        res.raise_for_status()
        body = res.json()
        usage = body.get("usage") or {}
        text = body["choices"][0]["message"]["content"] or "null"
        return json.loads(text), usage.get("prompt_tokens"), usage.get("completion_tokens")

    def _call_claude(self, system: str, user_prompt: str, model: str, api_key: str):
        import httpx
        # A forced tool call makes Claude answer in the reply schema.
        res = httpx.post(ANTHROPIC_URL, timeout=settings.LLM_TIMEOUT_MS / 1000,
                         headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                         json={"model": model, "max_tokens": 2048, "temperature": 0.2, "system": system,
                               "tools": [{"name": "reply", "description": "Send the reply.",
                                          "input_schema": _reply_schema()}],
                               "tool_choice": {"type": "tool", "name": "reply"},
                               "messages": [{"role": "user", "content": user_prompt}]})
        res.raise_for_status()
        body = res.json()
        block = next((b for b in body.get("content", []) if b.get("type") == "tool_use"), None)
        if block is None:
            raise ValueError("No structured reply")
        usage = body.get("usage") or {}
        return block.get("input"), usage.get("input_tokens"), usage.get("output_tokens")

    def generate(self, system_instruction: str, user_prompt: str, customer_text: str = None, llm=None) -> dict:
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
            raise LLMUnavailable("Generation is not configured")
        started = time.monotonic()
        try:
            system = system_instruction + "\n\n" + SAFETY_INSTRUCTION
            if provider == "OPENAI":
                raw, tokens_in, tokens_out = self._call_openai(system, user_prompt, model, api_key)
            elif provider == "CLAUDE":
                raw, tokens_in, tokens_out = self._call_claude(system, user_prompt, model, api_key)
            else:
                raw, tokens_in, tokens_out = self._call_gemini(system, user_prompt, model, api_key)
            result = _from_wire(raw)
            if contains_blocked_content(result) or (isinstance(result, dict) and result.get("intent") == "safety_refusal"):
                return refusal
            result = ModelReply.model_validate(result).model_dump()
            if not result['private_dm'].strip() and not result['handoff_reason']:
                raise ValueError("Empty reply")
            logger.info("llm_generation provider=%s model=%s elapsed_ms=%d input_tokens=%s output_tokens=%s",
                        provider, model, (time.monotonic() - started) * 1000, tokens_in, tokens_out)
            return result
        except Exception as exc:
            # Do not log customer text or provider exceptions containing request bodies.
            logger.warning("llm_generation_failed provider=%s type=%s elapsed_ms=%d", provider, type(exc).__name__,
                           (time.monotonic() - started) * 1000)
            raise LLMUnavailable("Generation failed") from exc


gemini_service = GeminiService()
