"""Deterministic high-confidence rules, supplemented by model judgment."""
import re
from app.services.guardrails import guardrails


def handoff_reason(text):
    text = text.casefold()
    if re.search(r'\b(human|agent|representative|real person|insaan)\b|kisi se baat|team se baat', text):
        return 'human_request'
    if re.search(r'\b(refund|cancel(?:lation)?|payment|tracking|track my order|where is my order)\b|order (?:status|issue|nahi|kab)|paise kat', text):
        return 'order_support'
    if guardrails.evaluate_sentiment_and_safety(text)[1]:
        return 'complaint'
    if re.search(r'\b(bulk|wholesale|negotiate|negotiation)\b|order (?:place )?kar(?:wa|va)|help (?:me )?(?:order|buy|purchase)|place (?:my|an) order|discount (?:do|chahiye)|best price', text):
        return 'purchase_assistance'
    return None


def unresolved_feedback(text):
    return bool(re.search(r"still (?:not|wrong|confused)|didn.t (?:answer|help)|not (?:helpful|resolved)|same (?:question|issue)|samajh nahi|jawab nahi|galat (?:jawab|answer)|phir se", text.casefold()))


def interested(text):
    return bool(re.search(r'\b(price|size|buy|purchase|interested|cost)\b|kitne|kharid', text.casefold()))
