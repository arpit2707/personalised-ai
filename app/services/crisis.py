"""Self-harm and suicide messages get helplines, never a sales reply.

This runs before every other check (the safety filter included), because a
message like "zeher kha lungi" must reach a helpline, not a refusal. The
backend runs the same gate before calling us; this is the second line.
"""
import re
import unicodedata

HELPLINES_EN = (
    "We're really sorry you're going through this. You don't have to face it alone. "
    "Please talk to someone right now:\n"
    "• Tele-MANAS: 14416 (free, 24x7)\n"
    "• KIRAN: 1800-599-0019\n"
    "• Emergency: 112\n"
    "If you are in immediate danger, please call 112."
)

HELPLINES_HINGLISH = (
    "Humein bahut afsos hai ki aap itna mushkil waqt dekh rahe hain. Aap akele nahi hain. "
    "Please abhi kisi se baat kariye:\n"
    "• Tele-MANAS: 14416 (free, 24x7)\n"
    "• KIRAN: 1800-599-0019\n"
    "• Emergency: 112\n"
    "Agar aap abhi khatre mein hain to turant 112 par call kariye."
)

HELPLINES_HINDI = (
    "हमें बहुत अफ़सोस है कि आप इतने मुश्किल समय से गुज़र रहे हैं। आप अकेले नहीं हैं। "
    "कृपया अभी किसी से बात कीजिए:\n"
    "• टेली-मानस: 14416 (मुफ़्त, 24x7)\n"
    "• किरण: 1800-599-0019\n"
    "• आपातकाल: 112\n"
    "अगर आप अभी ख़तरे में हैं तो तुरंत 112 पर कॉल कीजिए।"
)

# Posted under a comment: the helpline itself goes privately.
CRISIS_PUBLIC = "We've sent you a message 🙏"

_PATTERNS = [
    # English
    r"\b(?:kill|killing|hang|hanging|hurt|hurting|harm|harming|cut|cutting)\s+(?:my\s*self|myself)\b",
    r"\bsuicid\w*",
    r"\bend(?:ing)?\s+(?:my|this)\s+life\b",
    r"\btake\s+my\s+(?:own\s+)?life\b",
    r"\b(?:want|wanna|going|ready|plan(?:ning)?)\s+to\s+die\b",
    r"\bdon'?t\s+want\s+to\s+(?:live|be\s+alive|wake\s+up)\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bself[\s-]?harm\w*",
    # Hinglish (Roman script)
    r"\b(?:khud\s*ko|apne\s*aap\s*ko)\s+(?:maar|mar|khatam|nuksan|hurt)",
    r"\b(?:mar|marr)\s*(?:jaana|jana|jaunga|jaungi|jaaunga|jaaungi|jaun|jaau)\b",
    r"\bmarna\s+(?:hai|chahta|chahti|chahata)\b",
    r"\bjeena\s+nahi\s+(?:hai|chahta|chahti)\b",
    r"\bjeene\s+ka\s+(?:mann|man)\s+nahi\b",
    r"\bzindagi\s+(?:khatam|se\s+tang)\b",
    r"\b(?:atmhatya|aatmhatya|aatmahatya|atmahatya|khudkushi|khudkhushi)\b",
    r"\b(?:zeher|zehar|zahar|jeher)\s+(?:kha|pee|pi)\w*",
    r"\bfaansi\b|\bphansi\s+(?:laga|lagaa)",
    # Devanagari
    r"आत्महत्या|ख़ुदकुशी|खुदकुशी|मरना\s*(?:है|चाहता|चाहती)|जीना\s*नहीं|ज़हर\s*खा|जहर\s*खा|फाँसी|फांसी|ख़ुद\s*को\s*मार|खुद\s*को\s*मार",
]
_CRISIS = re.compile("|".join(_PATTERNS), re.IGNORECASE)

_HINGLISH_MARKERS = re.compile(
    r"\b(?:hai|hain|nahi|nahin|kya|mujhe|mera|meri|main|mai|hoon|hu|kar|karna|chahta|chahti|"
    r"zindagi|jeena|marna|jaana|kuch|koi|aap|tum|yaar|bhai|kaise|kyun|ab|abhi)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def is_crisis(text: str) -> bool:
    return bool(_CRISIS.search(_normalize(text)))


def language_of(text: str) -> str:
    """'hindi' for Devanagari, 'hinglish' for Roman Hindi, else 'english'."""
    text = text or ""
    if re.search(r"[ऀ-ॿ]", text):
        return "hindi"
    if len(_HINGLISH_MARKERS.findall(text)) >= 1:
        return "hinglish"
    return "english"


def crisis_reply(text: str) -> str:
    return {"hindi": HELPLINES_HINDI, "hinglish": HELPLINES_HINGLISH}.get(language_of(text), HELPLINES_EN)
