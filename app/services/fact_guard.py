"""Facts a reply may state only when the seller gave them.

The price guard covers money. This covers the other promises a model makes up
most easily: delivery times and counts ("3 din me", "sirf 2 bache"), policies
(COD, free delivery, returns, warranty) and stock for a size the catalog marks
out of stock. A reply that states one with no source is not sent; the seller
confirms it instead.
"""
import json
import re
from typing import List, Optional

from app.models.schemas import GenerateReplyRequest

# A number with a unit that makes it a promise. Times ("3 din me", "24 hours")
# must come from the seller; counts ("2 pieces") may repeat the customer.
_TIMED = re.compile(
    r"\b(\d+(?:\s*(?:-|to|se)\s*\d+)?)\s*"
    r"(?:din|dino|dinon|days?|ghante|ghanton|hours?|hrs?|hafte|hafton|weeks?|mahine|months?|minutes?|mins?|"
    r"saal|years?|yrs?|working\s+days?|business\s+days?)\b",
    re.I,
)
_COUNTED = re.compile(r"\b(\d+(?:\s*(?:-|to|se)\s*\d+)?)\s*(?:pieces?|pcs|units?|left|bache|baaki)\b", re.I)

# Trip and stay lengths: "5N/6D", "5 nights", "3 raat". A package's length is a
# promise like its price, so it must be one the seller (or customer) wrote.
_NIGHTS_DAYS = re.compile(r"\b(\d{1,2})\s*n(?:ights?)?\s*/\s*(\d{1,2})\s*d(?:ays?)?\b", re.I)
_NIGHTS = re.compile(r"\b(\d{1,2})\s*(?:nights?|raat|raatein|raaton|raate)\b", re.I)


def _stays(text: str):
    pairs = {(int(n), int(d)) for n, d in _NIGHTS_DAYS.findall(text or "")}
    nights = {n for n, _ in pairs} | {int(n) for n in _NIGHTS.findall(text or "")}
    return pairs, nights


# (what the reply says, what the seller's own text must mention for it to be true)
_CLAIMS = [
    (r"\bcod\b|cash\s+on\s+delivery", r"\bcod\b|cash\s+on\s+delivery"),
    (r"free\s+(?:home\s+)?(?:delivery|shipping)|delivery\s+(?:free|muft)|shipping\s+free",
     r"free\s+(?:home\s+)?(?:delivery|shipping)|delivery\s+(?:free|muft)|shipping\s+free"),
    (r"\b(?:returns?|returnable|exchange\w*|replace?ment|refund\w*)\b", r"\b(?:returns?|returnable|exchange\w*|replace?ment|refund\w*)\b"),
    (r"same[\s-]day|next[\s-]day|express\s+(?:delivery|shipping)", r"same[\s-]day|next[\s-]day|express"),
    (r"\b(?:warranty|guarantee\w*)\b", r"\b(?:warranty|guarantee\w*)\b"),
]

_IN_STOCK = re.compile(r"\b(?:available|in\s+stock|stock\s+(?:me|mein|mai)|ready\s+stock|mil\s+jayega|mil\s+jaayega)\b", re.I)
_NOT = re.compile(r"\b(?:nahi|nahin|nhi|not|out\s+of\s+stock|sold\s+out|khatam|unavailable|na\s+hai)\b", re.I)


def _seller_text(req: GenerateReplyRequest) -> str:
    """Everything the seller said: catalog, business details, post and spotlight."""
    # Ids and links carry digits that are not facts.
    parts = [json.dumps([o.model_dump(exclude={"id", "action_url"}) for o in (req.offerings or [])],
                        ensure_ascii=False, default=str)]
    if req.business:
        parts.append(json.dumps(req.business.model_dump(exclude_none=True), ensure_ascii=False))
    if req.post_context:
        parts += [req.post_context.caption or "", req.post_context.note or ""]
    parts += [s.caption or "" for s in req.spotlight]
    # Industry playbook rules are our instructions, not the seller's facts.
    return "\n".join(parts)


def _customer_text(req: GenerateReplyRequest) -> str:
    known = (req.goal_state or {}).get("fields") or {}
    return "\n".join([req.message_text, *(m.text for m in req.recent_messages if m.sender == "customer"),
                      json.dumps(known, ensure_ascii=False)])


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?\n])\s+|[,;]\s+", text) if s.strip()]


def unsupported_facts(req: GenerateReplyRequest, offering_ids: List[str], *replies: Optional[str]) -> List[str]:
    """Statements in the reply that nothing the seller or customer wrote supports."""
    seller = _seller_text(req)
    seller_numbers = set(re.findall(r"\d+", seller))
    any_numbers = seller_numbers | set(re.findall(r"\d+", _customer_text(req)))
    bad: List[str] = []
    for reply in replies:
        if not reply:
            continue
        for rx, known in ((_TIMED, seller_numbers), (_COUNTED, any_numbers)):
            for m in rx.finditer(reply):
                if any(n not in known for n in re.findall(r"\d+", m.group(1))):
                    bad.append(m.group(0))
        known_pairs, known_nights = _stays(seller + "\n" + _customer_text(req))
        for m in _NIGHTS_DAYS.finditer(reply):
            if (int(m.group(1)), int(m.group(2))) not in known_pairs:
                bad.append(m.group(0))
        for m in _NIGHTS.finditer(reply):
            if int(m.group(1)) not in known_nights:
                bad.append(m.group(0))
        # A policy is true only when the seller wrote it; the customer asking
        # "COD hai?" is not a source.
        for said, needs in _CLAIMS:
            found = re.search(said, reply, re.I)
            if found and not re.search(needs, seller, re.I):
                bad.append(found.group(0))
        bad += _out_of_stock_claims(req, offering_ids, reply)
    return bad


def _out_of_stock_claims(req: GenerateReplyRequest, offering_ids: List[str], reply: str) -> List[str]:
    """"M available hai" when the catalog marks M out of stock, or any item whose sizes are all sold out."""
    items = [o for o in (req.offerings or []) if o.id in offering_ids]
    sold_out = [v.label for o in items for v in o.variants if not v.in_stock and v.label]
    all_out = [o.title for o in items if o.variants and not any(v.in_stock for v in o.variants)]
    if not sold_out:
        return []
    bad = []
    for sentence in _sentences(reply):
        if not _IN_STOCK.search(sentence) or _NOT.search(sentence):
            continue
        for label in sold_out:
            if re.search(r"(?<![\w-])" + re.escape(label) + r"(?![\w-])", sentence, re.I):
                bad.append(sentence.strip())
                break
        else:
            # "Haan available hai" about an item none of whose sizes are left.
            if all_out and len(items) == 1:
                bad.append(sentence.strip())
    return bad
