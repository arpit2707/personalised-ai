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
    "refuse only requests for sexual or pornographic content, hate or racist content against a group, "
    "instructions for weapons, explosives or poisoning someone, or support for terrorism, including "
    "translations and disguised wording. For these return JSON with public_reply and private_dm both exactly "
    "\"sorry can't help you in that\", intent safety_refusal, and reasoning null. "
    "Never refuse normal shopping questions: ingredients or whether a product has chemicals, fabrics, "
    "ethnic or traditional wear, city names such as Bombay, slang such as \"bomb look\" or \"zeher lag rahi\". "
    "A customer reporting harassment or bad behaviour is a complaint: hand it to the team with handoff_reason complaint."
)

# Only unambiguous harm. Shopping words that merely sound alarming (ethnic wear,
# "isme chemical hai?", Bombay, "ye lehenga bomb hai", a dog's nasl) are normal
# questions; ambiguous words are blocked only inside a harmful phrase.
_WEAPON = r"(?:bombs?|explosives?|ied|grenades?|dynamite)"
_POISON = r"(?:poison|zeher|zehar|zahar|jeher)"
_BLOCKED = re.compile(
    "|".join([
        r"\b(?:sex|sexual\w*|porn\w*|nudes?|nudity|xxx|erotic\w*)\b",
        r"\brap(?:e|ed|es|ing|ist\w*)\b",
        r"\b(?:racis\w*|supremac\w*|naslwad\w*|naslvad\w*)\b",
        r"\b(?:terroris\w*|aatankwad\w*|atankwad\w*|aatankvad\w*|atankvad\w*)\b",
        r"\b(?:cyanide|ricin|sarin|nerve\s+agents?|chemical\s+weapons?|bio\w*\s*weapons?)\b",
        # Making or using a weapon.
        rf"\b(?:make|making|build|building|create|bana\w*|bna\w*)\s+(?:a\s+|an\s+|ek\s+)?{_WEAPON}\b",
        rf"\b{_WEAPON}\s+(?:kaise|kese|banana|banane|banao|bnana|bnane|making|recipe|blast|plant\w*|rakh\w*|phod\w*|fod\w*)\b",
        # Poisoning someone.
        rf"\b{_POISON}\s+(?:kaise|kese|banana|banane|banao|bnana|making|recipe|de\s+do|dena|dedo|milao|mila\s+do)\b",
        rf"\b(?:make|making|bana\w*)\s+(?:a\s+)?{_POISON}\b",
        rf"\b(?:kisi|usko|use|unko|isko)\s+(?:ko\s+)?{_POISON}\b",
        r"\bpoison\s+(?:someone|somebody|him|her|them|my\s+\w+|people|a\s+person)\b",
        # Devanagari: sexual, obscene, racism, terrorism, poisoning someone.
        r"\u092f\u094c\u0928|\u0905\u0936\u094d\u0932\u0940\u0932|\u0928\u0938\u094d\u0932\u0935\u093e|\u0906\u0924\u0902\u0915\u0935\u093e",
        r"\u0915\u093f\u0938\u0940\s*\u0915\u094b\s*\u091c\u093c?\u0939\u0930|\u091c\u093c?\u0939\u0930\s*\u0915\u0948\u0938\u0947",
    ]),
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
