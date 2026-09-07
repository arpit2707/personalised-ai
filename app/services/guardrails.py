import re
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
