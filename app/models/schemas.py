from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
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
    caption: Optional[str] = Field(default="", max_length=2000)
    # The seller's own note about the post ("offer valid till Sunday").
    note: Optional[str] = Field(default=None, max_length=1000)
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
    brand_id: str = Field(min_length=1, max_length=128)
    channel_type: Literal["instagram", "facebook", "whatsapp"] = "instagram"
    event_type: Literal["comment", "dm"] = "comment"
    message_text: str = Field(min_length=1, max_length=8000)
    sender_id: str = Field(min_length=1, max_length=256)
    post_context: Optional[PostContext] = None
    brand_persona: Optional[BrandPersona] = None
    business: Optional[BusinessContext] = None
    playbook: Optional[Playbook] = None
    offerings: Optional[List[OfferingContext]] = Field(default=None, max_length=30)
    goal_state: Optional[Dict[str, Any]] = None
    recent_messages: List[RecentMessage] = Field(default_factory=list, max_length=20)
    # The Reel2Real backend keeps its own hand-off pause and only calls when the
    # seller wants the AI to answer, so it asks to reopen a chat no agent has
    # claimed. Other callers keep the queue until an agent releases it.
    resume_if_pending: bool = False

class GenerateReplyResponse(BaseModel):
    public_reply: Optional[str] = None
    private_dm: Optional[str] = None
    intent: str = "general"
    sentiment: str = "neutral"
    requires_human_attention: bool = False
    detected_product_sku: Optional[str] = None
    reasoning: Optional[str] = None
    action: Optional[str] = None
    offering_ids: List[str] = Field(default_factory=list)
    collected_fields: Dict[str, str] = Field(default_factory=dict)
    conversation_id: Optional[str] = None
    conversation_status: Literal["ai", "pending", "active"] = "ai"
    handoff_reason: Optional[str] = None
    lead_interested: bool = False
    # PRODUCTS or SERVICES once the customer has made clear which they want.
    offering_type: Optional[str] = None


class PreferenceEvidence(BaseModel):
    key: Literal["size", "color", "language"]
    value: str = Field(min_length=1, max_length=80)
    evidence: str = Field(min_length=1, max_length=200)


class ModelReply(BaseModel):
    action: Optional[Literal["SEND_LINK", "ASK_FIELD", "CREATE_LEAD", "HANDOFF", "ANSWER"]] = None
    offering_ids: List[str] = Field(default_factory=list, max_length=30)
    collected_fields: Dict[str, str] = Field(default_factory=dict)
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


class CollectedField(BaseModel):
    key: str = Field(max_length=64)
    value: str = Field(max_length=200)


class ModelReplyWire(ModelReply):
    """The schema Gemini is asked to fill. The Developer API rejects free-form
    objects (dict-typed fields), so collected fields come back as a list of
    key/value pairs and are turned into ModelReply's dict afterwards."""
    collected_fields: List[CollectedField] = Field(default_factory=list, max_length=20)


class AgentAction(BaseModel):
    agent_id: str = Field(min_length=1, max_length=128)


class AgentMessage(AgentAction):
    message_text: str = Field(min_length=1, max_length=4000)
