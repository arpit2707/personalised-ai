"""What the customer reads when the AI steps back.

Pausing reasons hand the chat to the seller: one friendly message, then the AI
stays quiet until the seller (or the backend's pause timer) reopens it.
Soft reasons ("the team will confirm this") keep the AI answering the next
message, and the same soft line is never sent twice in a row.
"""
from app.services.crisis import crisis_reply, language_of

PAUSING_REASONS = {
    "human_request", "complaint", "order_support", "purchase_assistance",
    "unresolved_query", "crisis",
}
SOFT_REASONS = {"missing_information", "conflicting_information", "generation_unavailable", "price_blocked"}

_TEXTS = {
    "human_request": (
        "Sure! I've asked our team to reply to you here personally. They'll get back to you soon 🙏",
        "Zaroor! Hamari team ko bata diya hai, woh aapko yahin personally reply karenge 🙏",
    ),
    "complaint": (
        "We're really sorry about this. Our team has your message and will personally sort it out with you here soon 🙏",
        "Iske liye humein sach mein afsos hai. Hamari team ne aapka message dekh liya hai aur jaldi yahin aapse baat karke ise theek karegi 🙏",
    ),
    "order_support": (
        "Sorry for the trouble! Our team will check your order and reply here shortly 🙏",
        "Takleef ke liye sorry! Hamari team aapka order check karke yahin jaldi reply karegi 🙏",
    ),
    "purchase_assistance": (
        "Great! Our team will help you with this personally and reply here soon 😊",
        "Badhiya! Hamari team is mein aapki personally help karegi aur jaldi yahin reply karegi 😊",
    ),
    "unresolved_query": (
        "Sorry I couldn't answer that properly. I've asked our team to reply to you here 🙏",
        "Sorry, main theek se jawab nahi de paaya. Hamari team ko bata diya hai, woh aapko yahin reply karenge 🙏",
    ),
    "missing_information": (
        "Good question! Our team will confirm this and get back to you here.",
        "Accha sawaal! Team ise confirm karke aapko yahin batayegi.",
    ),
    "generation_unavailable": (
        "Got your message 🙏 Our team will get back to you shortly.",
        "Aapka message mil gaya hai 🙏 Team jaldi aapko reply karegi.",
    ),
}
_TEXTS["conflicting_information"] = _TEXTS["missing_information"]
_TEXTS["price_blocked"] = _TEXTS["missing_information"]

LEAD_CONFIRMATION = (
    "Thank you! Our team will contact you to confirm the details.",
    "Thank you! Team aapse contact karke confirm karegi.",
)


def _pick(pair, customer_text: str) -> str:
    return pair[0] if language_of(customer_text) == "english" else pair[1]


def handoff_text(reason: str, customer_text: str = "") -> str:
    if reason == "crisis":
        return crisis_reply(customer_text)
    return _pick(_TEXTS.get(reason, _TEXTS["missing_information"]), customer_text)


def soft_texts(reason: str) -> set:
    """Every wording of a soft reason's line (English and Hinglish)."""
    return set(_TEXTS.get(reason, _TEXTS["missing_information"]))


def lead_confirmation(customer_text: str = "") -> str:
    return _pick(LEAD_CONFIRMATION, customer_text)


def is_handoff_line(text: str) -> bool:
    """True for any canned hand-off line, so a model echoing one is not a confirmation."""
    lines = {t for pair in _TEXTS.values() for t in pair}
    return (text or "").strip() in lines or "queue" in (text or "").casefold()
