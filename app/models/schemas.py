from pydantic import BaseModel, Field
from typing import Optional, List, Literal
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
    brand_id: str = Field(min_length=1, max_length=128)
    channel_type: Literal["instagram", "facebook", "whatsapp"] = "instagram"
    event_type: Literal["comment", "dm"] = "comment"
    message_text: str = Field(min_length=1, max_length=8000)
    sender_id: str = Field(min_length=1, max_length=256)
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
    conversation_id: Optional[str] = None
    conversation_status: Literal["ai", "pending", "active"] = "ai"
    handoff_reason: Optional[str] = None
    lead_interested: bool = False


class PreferenceEvidence(BaseModel):
    key: Literal["size", "color", "language"]
    value: str = Field(min_length=1, max_length=80)
    evidence: str = Field(min_length=1, max_length=200)


class ModelReply(BaseModel):
    public_reply: Optional[str] = Field(default=None, max_length=2000)
    private_dm: str = Field(max_length=4000)
    intent: str = Field(default="general", max_length=64)
    reasoning: Optional[str] = Field(default=None, max_length=500)
    handoff_reason: Optional[Literal[
        "human_request", "complaint", "order_support", "purchase_assistance",
        "missing_information", "conflicting_information",
    ]] = None
    previous_answer_unresolved: bool = False
    buying_interest: bool = False
    preferences: List[PreferenceEvidence] = Field(default_factory=list, max_length=3)


class AgentAction(BaseModel):
    agent_id: str = Field(min_length=1, max_length=128)


class AgentMessage(AgentAction):
    message_text: str = Field(min_length=1, max_length=4000)
