import json
import logging
import time
from app.core.config import settings
from app.models.schemas import ModelReply
from app.services.guardrails import SAFETY_INSTRUCTION, SAFETY_REFUSAL, contains_blocked_content

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    pass


class GeminiService:
    def __init__(self):
        self._client = None

    def generate(self, system_instruction: str, user_prompt: str) -> dict:
        refusal = {"public_reply": SAFETY_REFUSAL, "private_dm": SAFETY_REFUSAL,
                   "intent": "safety_refusal", "reasoning": None}
        if contains_blocked_content([system_instruction, user_prompt]):
            return refusal
        if not settings.GEMINI_API_KEY:
            raise LLMUnavailable("Generation is not configured")
        started = time.monotonic()
        try:
            from google import genai
            from google.genai import types
            if self._client is None:
                self._client = genai.Client(
                    api_key=settings.GEMINI_API_KEY,
                    http_options=types.HttpOptions(timeout=settings.LLM_TIMEOUT_MS,
                        retry_options=types.HttpRetryOptions(attempts=2)),
                )
            response = self._client.models.generate_content(
                model=settings.DEFAULT_MODEL, contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction + "\n\n" + SAFETY_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=ModelReply,
                    temperature=0.2,
                ),
            )
            result = json.loads(response.text or "null")
            if contains_blocked_content(result) or (isinstance(result, dict) and result.get("intent") == "safety_refusal"):
                return refusal
            result = ModelReply.model_validate(result).model_dump()
            if not result['private_dm'].strip() and not result['handoff_reason']:
                raise ValueError("Empty reply")
            usage = getattr(response, "usage_metadata", None)
            logger.info("llm_generation model=%s elapsed_ms=%d input_tokens=%s output_tokens=%s",
                        settings.DEFAULT_MODEL, (time.monotonic() - started) * 1000,
                        getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None))
            return result
        except Exception as exc:
            # Do not log customer text or provider exceptions containing request bodies.
            logger.warning("llm_generation_failed type=%s elapsed_ms=%d", type(exc).__name__,
                           (time.monotonic() - started) * 1000)
            raise LLMUnavailable("Generation failed") from exc


gemini_service = GeminiService()
