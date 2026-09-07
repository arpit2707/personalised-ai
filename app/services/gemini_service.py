import json
import logging
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self._client = None
        if settings.GEMINI_API_KEY:
            try:
                from google import genai
                self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize Google GenAI Client: {e}")

    def generate(self, system_instruction: str, user_prompt: str) -> Dict[str, Any]:
        if self._client and settings.GEMINI_API_KEY:
            try:
                from google.genai import types
                response = self._client.models.generate_content(
                    model=settings.DEFAULT_MODEL,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.7,
                    )
                )
                if response.text:
                    return json.loads(response.text)
            except Exception as e:
                logger.error(f"Gemini API invocation error: {e}")

        # Intelligent local fallback when GEMINI_API_KEY is not yet populated
        return self._local_fallback(user_prompt)

    def _local_fallback(self, user_prompt: str) -> Dict[str, Any]:
        lower_prompt = user_prompt.lower()
        if "price" in lower_prompt or "kitne" in lower_prompt or "cost" in lower_prompt:
            return {
                "public_reply": "Hey! Sent you the exclusive price and direct order link in your DM! 🛍️✨",
                "private_dm": "Hey there! Thanks for your interest. You can check the complete product details, price, and place your order directly here: https://brand.com/checkout. Let us know if you need any help with sizing! 💖",
                "intent": "price_inquiry",
                "reasoning": "Detected price inquiry in customer message."
            }
        elif "size" in lower_prompt or "m available" in lower_prompt or "large" in lower_prompt:
            return {
                "public_reply": "Yes, sizes are currently in stock! Dropped the size chart in your DM! 👗",
                "private_dm": "Hello! Yes, standard sizes (S, M, L, XL) are currently in stock and shipping within 24 hours. Check out the size guide and place your order here: https://brand.com/checkout 🛍️",
                "intent": "size_availability",
                "reasoning": "Detected size question in customer message."
            }
        else:
            return {
                "public_reply": "Hey! Sent you all the details in your DM, please check! ✨",
                "private_dm": "Hi! Thanks for reaching out to us. We have shared the complete details with you. Feel free to ask if you have any questions! 🙌",
                "intent": "general",
                "reasoning": "General query response."
            }

gemini_service = GeminiService()
