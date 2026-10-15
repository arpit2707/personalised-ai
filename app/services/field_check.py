"""Checks a lead detail the model says the customer gave, against what they wrote.

The model returns dates as YYYY-MM-DD, phone numbers without spaces and cities
by their current name, so a plain "is it in the message" test dropped "15 Nov"
(stored as 2026-11-15), "98765 43210" and "Bombay". Each kind of field is
compared in its own way; a value the message does not support is still dropped.
"""
import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Set

IST = timezone(timedelta(hours=5, minutes=30))

_MONTHS = {
    "jan": 1, "january": 1, "janvari": 1, "janwari": 1,
    "feb": 2, "february": 2, "farvari": 2, "farwari": 2,
    "mar": 3, "march": 3, "maarch": 3,
    "apr": 4, "april": 4, "aprail": 4, "apreil": 4,
    "may": 5, "mai": 5,
    "jun": 6, "june": 6, "joon": 6,
    "jul": 7, "july": 7, "julai": 7, "jolai": 7,
    "aug": 8, "august": 8, "agast": 8, "agust": 8,
    "sep": 9, "sept": 9, "september": 9, "sitambar": 9, "sitamber": 9,
    "oct": 10, "october": 10, "aktubar": 10, "aktoobar": 10,
    "nov": 11, "november": 11, "navambar": 11, "navamber": 11,
    "dec": 12, "december": 12, "disambar": 12, "disamber": 12,
}
_WEEKDAYS = {
    "monday": 0, "mon": 0, "somvar": 0, "somwar": 0,
    "tuesday": 1, "tue": 1, "mangalvar": 1, "mangalwar": 1,
    "wednesday": 2, "wed": 2, "budhvar": 2, "budhwar": 2,
    "thursday": 3, "thu": 3, "guruvar": 3, "guruwar": 3, "brihaspativar": 3,
    "friday": 4, "fri": 4, "shukravar": 4, "shukrawar": 4,
    "saturday": 5, "shanivar": 5, "shaniwar": 5,
    "sunday": 6, "ravivar": 6, "raviwar": 6, "itvaar": 6, "itwar": 6, "itvar": 6,
}
_MONTH_WORD = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))
_WEEKDAY_WORD = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))

_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?\b")
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({_MONTH_WORD})\.?(?:\s*,?\s*(\d{{4}}))?\b", re.I)
_MONTH_DAY = re.compile(rf"\b({_MONTH_WORD})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{{4}}))?\b", re.I)
_MONTH_ONLY = re.compile(rf"\b({_MONTH_WORD})\b(?:\s+(\d{{4}}))?", re.I)
_WEEKDAY = re.compile(rf"\b(agle|agla|next|is|iss|this|coming)?\s*({_WEEKDAY_WORD})\b", re.I)

_CITY_ALIASES = {
    "mumbai": {"bombay", "bambai"}, "kolkata": {"calcutta"}, "chennai": {"madras"},
    "bengaluru": {"bangalore", "banglore"}, "gurugram": {"gurgaon"}, "pune": {"poona"},
    "varanasi": {"banaras", "benaras", "kashi"}, "prayagraj": {"allahabad"},
    "thiruvananthapuram": {"trivandrum"}, "kochi": {"cochin"}, "mysuru": {"mysore"},
    "vadodara": {"baroda"}, "puducherry": {"pondicherry", "pondy"}, "delhi": {"dilli", "new delhi"},
}


def today_in_india(value: Optional[str] = None) -> date:
    """The request's `today` (the seller's local date) or today in India."""
    if value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    return datetime.now(IST).date()


def _safe(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _upcoming(m: int, d: int, year: Optional[str], today: date) -> Optional[date]:
    """A day and month without a year is the next one to come."""
    if year:
        y = int(year)
        return _safe(y + 2000 if y < 100 else y, m, d)
    found = _safe(today.year, m, d)
    if found and found < today:
        found = _safe(today.year + 1, m, d)
    return found


def dates_in(text: str, today: date) -> Set[date]:
    """Every calendar date the customer's words name: "15 Nov", "15/11", "kal", "agle Sunday"."""
    text = (text or "").casefold()
    found: Set[date] = set()
    for y, m, d in _ISO.findall(text):
        found.add(_safe(int(y), int(m), int(d)))
    for d, m, y in _NUMERIC.findall(text):
        # Indian order: day first.
        found.add(_upcoming(int(m), int(d), y or None, today))
    for d, mon, y in _DAY_MONTH.findall(text):
        found.add(_upcoming(_MONTHS[mon.rstrip(".")], int(d), y or None, today))
    for mon, d, y in _MONTH_DAY.findall(text):
        found.add(_upcoming(_MONTHS[mon.rstrip(".")], int(d), y or None, today))
    if re.search(r"\b(?:aaj|today)\b", text):
        found.add(today)
    if re.search(r"\b(?:kal|tomorrow|tmrw|tmr)\b", text):
        found.add(today + timedelta(days=1))
    if re.search(r"\b(?:parso|parson|parsoon|day\s+after\s+tomorrow)\b", text):
        found.add(today + timedelta(days=2))
    for prefix, name in _WEEKDAY.findall(text):
        ahead = (_WEEKDAYS[name] - today.weekday()) % 7
        if prefix in ("agle", "agla", "next"):
            ahead = ahead or 7
        found.add(today + timedelta(days=ahead))
        if prefix in ("agle", "agla", "next") and ahead < 7:
            # "next Sunday" said on a Thursday means this coming or the one after.
            found.add(today + timedelta(days=ahead + 7))
    found.discard(None)
    return found


def months_in(text: str, today: date) -> Set[str]:
    """YYYY-MM for every month named ("December me", "Jan 2027")."""
    out: Set[str] = set()
    for mon, y in _MONTH_ONLY.findall((text or "").casefold()):
        m = _MONTHS[mon.rstrip(".")]
        if mon in ("may", "mai") and not y:
            continue  # "Patna mai" is "in Patna", not May
        year = int(y) if y else (today.year if m >= today.month else today.year + 1)
        out.add(f"{year:04d}-{m:02d}")
    for d in dates_in(text, today):
        out.add(f"{d.year:04d}-{d.month:02d}")
    return out


def _compact(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").casefold())


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def checked_value(key: str, label: str, value: str, message: str, today: date) -> Optional[str]:
    """The value to store when the customer's message supports it, else None."""
    from app.services.catalog_reply import extract_amounts  # avoid an import cycle

    value = str(value).strip()[:120]
    if not value:
        return None
    kind = f"{key} {label}".casefold()
    text = message or ""
    if re.search(r"date|check.?in|check.?out|visit|month|\bday\b|kab", kind):
        iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
        if iso:
            return value if _safe(*map(int, iso.groups())) in dates_in(text, today) else None
        ym = re.fullmatch(r"(\d{4})-(\d{2})", value)
        if ym:
            return value if value in months_in(text, today) else None
        if value.casefold() in text.casefold():
            named = dates_in(value, today)
            return named.pop().isoformat() if len(named) == 1 else value
        return None
    if re.search(r"phone|mobile|whatsapp|contact|number", kind):
        digits = _digits(value)
        digits = digits[-10:] if len(digits) > 10 else digits
        return value if len(digits) == 10 and digits in _digits(text) else None
    if re.search(r"budget|price|amount", kind):
        wanted = extract_amounts(value) or ([float(_digits(value))] if _digits(value) else [])
        said = extract_amounts(text) + [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text)]
        if wanted and any(abs(w - s) <= max(1, s * 0.01) for w in wanted for s in said):
            return value
    if value.casefold() in text.casefold() or (_compact(value) and _compact(value) in _compact(text)):
        return value
    lowered = text.casefold()
    for name, aliases in _CITY_ALIASES.items():
        if value.casefold() == name and any(re.search(rf"\b{re.escape(a)}\b", lowered) for a in aliases):
            return value
    # "50 people" for "50 log aayenge": every number the value has, the customer said.
    numbers = re.findall(r"\d+", value)
    if numbers and all(re.search(rf"(?<!\d){n}(?!\d)", text) for n in numbers) and not re.search(r"[a-z]{4,}", _strip_units(value)):
        return value
    return None


def _strip_units(value: str) -> str:
    return re.sub(r"\b(?:people|persons?|guests?|log|nights?|raat|raatein|days?|din|adults?|kids?|children|bachche|bhk)\b",
                  "", value.casefold())
