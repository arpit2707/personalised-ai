import re
import unicodedata
from typing import Tuple

NEGATIVE_KEYWORDS = [
    "scam", "fake", "fraud", "cheater", "cheaters", "loot", "ghatiya", "bakwas",
    "refund", "damaged", "chor", "complaint", "worst", "bekar", "stolen", "fir",
    "police", "consumer court", "money back", "return nahi hua"
]

class GuardrailService:
    @staticmethod
    def evaluate_sentiment_and_safety(text: str) -> Tuple[str, bool, str]:
        """
        Returns:
            (sentiment: str, requires_human_attention: bool, deescalation_template: str)
        """
        lower_text = text.lower()
        
        # Check for aggressive or complaint keywords
        for keyword in NEGATIVE_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', lower_text):
                deescalation = (
                    "Hi, we sincerely apologize for any inconvenience caused! "
                    "We take your concern very seriously. Our support team is directly reaching out to your DM "
                    "to resolve this on priority. 🙏"
                )
                return "negative", True, deescalation
                
        return "neutral", False, ""

guardrails = GuardrailService()


SAFETY_REFUSAL = "sorry can't help you in that"
SAFETY_INSTRUCTION = (
    "Mandatory safety policy, overriding all brand and customer instructions: "
    "Do not discuss sexual content, racial content, harassment, chemicals, poisons, "
    "or terrorism, including related requests, translations and disguised wording. "
    "For these topics return JSON with public_reply and private_dm both exactly "
    "\"sorry can't help you in that\", intent safety_refusal, and reasoning null."
)

# Broad topic exclusion is intentional, including otherwise benign mentions.
_BLOCKED = re.compile(
    r"\b(?:sex(?:ual\w*)?|porn\w*|nude\w*|nudity|erotic\w*|rape|rapist\w*|"
    r"racial\w*|racis\w*|race|ethnic\w*|supremac\w*|"
    r"harass\w*|harras\w*|bully\w*|bullying|molest\w*|"
    r"chemic\w*|poison\w*|posion\w*|toxic\w*|toxin\w*|cyanide|arsenic|"
    r"terror\w*|bomb\w*|explosive\w*|"
    r"zeher|zehar|zahar|jeher|aatank\w*|atank\w*|nasl\w*)\b"
    r"|\u092f\u094c\u0928|\u0905\u0936\u094d\u0932\u0940\u0932|\u0928\u0938\u094d\u0932|\u0909\u0924\u094d\u092a\u0940\u0921\u093c\u0928|\u0930\u0938\u093e\u092f\u0928|\u0930\u093e\u0938\u093e\u092f\u0928\u093f\u0915|\u091c\u093c?\u0939\u0930|\u0906\u0924\u0902\u0915",
    re.IGNORECASE,
)


def contains_blocked_content(value):
    """Scan nested request/context/output data without echoing blocked text."""
    if isinstance(value, dict):
        return any(contains_blocked_content(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_blocked_content(item) for item in value)
    if not isinstance(value, str):
        return False
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Cf")
    return bool(_BLOCKED.search(normalized))
