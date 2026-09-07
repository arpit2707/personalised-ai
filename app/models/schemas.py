from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ToneEnum(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PLAYFUL = "playful"
    GEN_Z = "gen_z"

class LanguageModeEnum(str, Enum):
    AUTO = "auto"
    ENGLISH = "english"
    HINGLISH = "hinglish"
    HINDI = "hindi"

class EmojiDensityEnum(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

class BrandPersona(BaseModel):
    brand_name: str
    tone: ToneEnum = ToneEnum.FRIENDLY
    language_mode: LanguageModeEnum = LanguageModeEnum.HINGLISH
    emoji_density: EmojiDensityEnum = EmojiDensityEnum.MODERATE
    custom_instructions: Optional[str] = None
    dm_cta_text: Optional[str] = "Check your DM for exclusive checkout link! 🛍️"

class ProductInfo(BaseModel):
    sku: str
    title: str
    description: Optional[str] = ""
    price: float
    currency: str = "INR"
    in_stock: bool = True
    stock_quantity: int = 10
    sizes: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    checkout_url: str = ""

class PostContext(BaseModel):
    post_id: str
    caption: Optional[str] = ""
    tagged_product_sku: Optional[str] = None

class GenerateReplyRequest(BaseModel):
    brand_id: str
    channel_type: str = "instagram"  # instagram, facebook, whatsapp
    event_type: str = "comment"      # comment, dm
    message_text: str
    sender_id: str
    post_context: Optional[PostContext] = None
    brand_persona: Optional[BrandPersona] = None

class GenerateReplyResponse(BaseModel):
    public_reply: Optional[str] = None
    private_dm: Optional[str] = None
    intent: str = "general"
    sentiment: str = "neutral"
    requires_human_attention: bool = False
    detected_product_sku: Optional[str] = None
    reasoning: Optional[str] = None
