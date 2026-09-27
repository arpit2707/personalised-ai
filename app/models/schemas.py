from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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

class OfferingVariant(BaseModel):
    label: str
    price: Optional[float] = None
    stock: Optional[int] = None
    in_stock: bool = True

class OfferingAvailability(BaseModel):
    date: str
    status: str

class OfferingContext(BaseModel):
    """One catalog item the backend picked for this message (product, service, room…)."""
    id: str
    type: str = "PRODUCT"
    title: str
    description: Optional[str] = None
    price_label: str = ""
    price_mode: str = "FIXED"
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    currency: str = "INR"
    action_url: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    variants: List[OfferingVariant] = Field(default_factory=list)
    includes: List[str] = Field(default_factory=list)
    linked_to_post: bool = False
    availability: Optional[List[OfferingAvailability]] = None

class LeadField(BaseModel):
    key: str
    label: str
    ask: str = ""

class Playbook(BaseModel):
    goal: str = "ORDER"  # ORDER | LEAD | BOOKING
    lead_fields: List[LeadField] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)

class BusinessContext(BaseModel):
    industry: str = "APPAREL"
    industry_label: str = ""
    description: Optional[str] = None
    city: Optional[str] = None
    service_areas: List[str] = Field(default_factory=list)
    hours: Optional[str] = None
    policies: Optional[Dict[str, str]] = None
    faqs: List[Dict[str, str]] = Field(default_factory=list)

class RecentMessage(BaseModel):
    # "from" is a Python keyword, so the attribute is `sender`.
    sender: str = Field(alias="from")
    text: str

    model_config = {"populate_by_name": True}

class GenerateReplyRequest(BaseModel):
    brand_id: str
    channel_type: str = "instagram"  # instagram, facebook, whatsapp
    event_type: str = "comment"      # comment, dm
    message_text: str
    sender_id: str
    post_context: Optional[PostContext] = None
    brand_persona: Optional[BrandPersona] = None
    # Sent by the Reel2Real backend's reply engine. When `offerings` is present
    # the backend owns the catalog and this service does not search its own.
    business: Optional[BusinessContext] = None
    playbook: Optional[Playbook] = None
    offerings: Optional[List[OfferingContext]] = None
    goal_state: Optional[Dict[str, Any]] = None
    recent_messages: List[RecentMessage] = Field(default_factory=list)

class GenerateReplyResponse(BaseModel):
    public_reply: Optional[str] = None
    private_dm: Optional[str] = None
    intent: str = "general"
    sentiment: str = "neutral"
    requires_human_attention: bool = False
    detected_product_sku: Optional[str] = None
    reasoning: Optional[str] = None
    # SEND_LINK | ASK_FIELD | CREATE_LEAD | HANDOFF | ANSWER
    action: Optional[str] = None
    offering_ids: List[str] = Field(default_factory=list)
    collected_fields: Dict[str, str] = Field(default_factory=dict)
