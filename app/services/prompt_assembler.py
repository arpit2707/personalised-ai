from typing import Optional, List
from app.models.schemas import BrandPersona, ProductInfo, PostContext, ToneEnum, LanguageModeEnum, EmojiDensityEnum

class PromptAssembler:
    @staticmethod
    def build_system_prompt(persona: BrandPersona) -> str:
        tone_descriptions = {
            ToneEnum.FORMAL: "Formal, respectful, polished, and professional.",
            ToneEnum.CASUAL: "Casual, modern, approachable, and friendly.",
            ToneEnum.FRIENDLY: "Warm, super helpful, vibrant, and welcoming.",
            ToneEnum.PLAYFUL: "Fun, witty, upbeat, and humorous.",
            ToneEnum.GEN_Z: "Trendy, energetic, youthful, using modern conversational cues."
        }
        
        lang_instructions = {
            LanguageModeEnum.ENGLISH: "Always respond in clean, fluent English.",
            LanguageModeEnum.HINGLISH: "Respond naturally in contemporary Hinglish (Hindi written in Roman English script, e.g., 'Bhai bilkul available hai!', 'Check your DM for special price'). Match the user's vibe.",
            LanguageModeEnum.HINDI: "Respond in clear Hindi (Devanagari or Romanized according to user input).",
            LanguageModeEnum.AUTO: "Automatically match the language and slang used by the customer in their message."
        }
        
        emoji_guide = {
            EmojiDensityEnum.NONE: "Do not use any emojis.",
            EmojiDensityEnum.LOW: "Use maximum 1 subtle emoji per message.",
            EmojiDensityEnum.MODERATE: "Use 2-3 engaging emojis naturally placed.",
            EmojiDensityEnum.HIGH: "Use lively emojis (3-5) to keep it vibrant and energetic."
        }

        prompt = f"""You are the AI Sales & Customer Engagement Specialist for '{persona.brand_name}'.
Brand Tone: {tone_descriptions.get(persona.tone, 'Friendly and helpful')}
Language Guide: {lang_instructions.get(persona.language_mode, 'Contemporary Hinglish')}
Emoji Usage: {emoji_guide.get(persona.emoji_density, 'Moderate')}

CORE OBJECTIVES:
1. PUBLIC COMMENT REPLY: Short, snappy, and engaging (under 15-20 words). Never reveal complete checkout links in public comments to avoid spam. Tease the benefit and let them know you've DM'ed them with the secret link or size details.
2. PRIVATE DM REPLY: Complete, clear, and high-converting. Detail product features, exact price in INR, available sizes, stock status, and direct checkout link.
3. CONVERSION FOCUS: Gently guide the customer toward buying without being aggressive or pushy.
4. HONESTY: If a size is out of stock, politely inform them and offer alternatives or notify-me options.

{persona.custom_instructions if persona.custom_instructions else ''}
"""
        return prompt

    @staticmethod
    def build_user_prompt(
        message_text: str,
        channel_type: str,
        event_type: str,
        post_context: Optional[PostContext] = None,
        product: Optional[ProductInfo] = None
    ) -> str:
        product_section = "NO SPECIFIC PRODUCT LINKED"
        if product:
            sizes_str = ", ".join(product.sizes) if product.sizes else "Standard / Free Size"
            colors_str = ", ".join(product.colors) if product.colors else "As pictured"
            stock_status = "In Stock" if product.in_stock else "OUT OF STOCK"
            
            product_section = f"""
TARGET PRODUCT CONTEXT:
- SKU: {product.sku}
- Title: {product.title}
- Price: {product.currency} {product.price}
- Stock Status: {stock_status} (Available qty: {product.stock_quantity})
- Sizes: {sizes_str}
- Colors: {colors_str}
- Checkout Link: {product.checkout_url}
- Description: {product.description}
"""
        post_section = ""
        if post_context and post_context.caption:
            post_section = f"\nPOST CAPTION: {post_context.caption}\n"

        prompt = f"""Incoming Channel: {channel_type.upper()}
Event Type: {event_type.upper()}
Customer Message: "{message_text}"
{post_section}
{product_section}

Format your response strictly as JSON with the following keys:
{{
  "public_reply": "Short engaging reply for public comment (if event_type == 'comment', else null)",
  "private_dm": "Helpful conversion-focused private DM message with details & checkout link",
  "intent": "price_inquiry | size_availability | product_details | general | out_of_stock",
  "reasoning": "brief 1-line reason for chosen reply"
}}
"""
        return prompt

prompt_assembler = PromptAssembler()
