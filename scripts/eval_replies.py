"""Runs real-model replies over scripts/eval_cases.json and reports what failed.

The unit tests stub the model; this asks the real one, through the same
endpoint the backend calls (safety filter, hand-off rules, price and fact
guards included), so it shows how often the model gets it right and how
often a guard had to stop a made-up fact.

    GEMINI_API_KEY=... python scripts/eval_replies.py
    python scripts/eval_replies.py --provider CLAUDE --model claude-sonnet-5-5 --api-key sk-...
    python scripts/eval_replies.py --only salon-date-written --verbose

Each case uses a fresh sender, a throwaway SQLite memory and no Supabase.
Exit code 1 when any case fails, so it can gate a prompt change.
"""
import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GUARD_INTENTS = {"price_blocked", "fact_blocked", "link_blocked"}


def build_payload(case: dict, fixtures: dict, today: str) -> dict:
    payload = copy.deepcopy(fixtures[case["fixture"]])
    patches = case.get("offerings_patch") or {}
    for offering in payload.get("offerings") or []:
        offering.update(patches.get(offering["id"], {}))
    for key, value in case.items():
        if key not in ("id", "fixture", "expect", "offerings_patch"):
            payload[key] = value
    payload.setdefault("today", today)
    payload["sender_id"] = f"eval-{case['id']}"
    if payload.get("event_type") == "comment":
        payload.setdefault("comment_author", "eval_user")
    return payload


def check(case: dict, res: dict) -> list:
    """Every expectation the reply misses, as a readable line."""
    from app.services.catalog_reply import extract_amounts

    exp = case.get("expect", {})
    dm = (res.get("private_dm") or "").casefold()
    public = (res.get("public_reply") or "").casefold()
    intent = res.get("intent")
    action = res.get("action")
    fails = []
    if "intent" in exp and intent != exp["intent"]:
        fails.append(f"intent {intent!r}, wanted {exp['intent']!r}")
    for bad in exp.get("intent_not", []) + ([] if exp.get("intent") == "safety_refusal" else ["safety_refusal"]):
        if intent == bad:
            fails.append(f"intent is {bad!r}")
    if "action" in exp and action not in exp["action"]:
        fails.append(f"action {action!r}, wanted one of {exp['action']}")
    if exp.get("not_handoff") and (action == "HANDOFF" or res.get("requires_human_attention")):
        fails.append(f"handed over ({intent}, {res.get('handoff_reason')}) instead of answering")
    if exp.get("dm_contains_any") and not any(w.casefold() in dm for w in exp["dm_contains_any"]):
        fails.append(f"DM has none of {exp['dm_contains_any']}")
    for w in exp.get("dm_contains_all", []):
        if w.casefold() not in dm:
            fails.append(f"DM misses {w!r}")
    for w in exp.get("dm_not_contains", []):
        if w.casefold() in dm:
            fails.append(f"DM says {w!r}")
    if exp.get("no_price") and (extract_amounts(dm) or extract_amounts(public)):
        fails.append("quoted a price")
    if exp.get("public_no_price") and extract_amounts(public):
        fails.append("public reply quoted a price")
    collected = res.get("collected_fields") or {}
    for key, value in (exp.get("collected") or {}).items():
        if collected.get(key) != value:
            fails.append(f"collected {key}={collected.get(key)!r}, wanted {value!r}")
    for key in exp.get("collected_keys", []):
        if not collected.get(key):
            fails.append(f"did not collect {key}")
    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(ROOT / "scripts" / "eval_cases.json"))
    parser.add_argument("--only", action="append", help="case id (repeatable)")
    parser.add_argument("--provider", choices=["GEMINI", "OPENAI", "CLAUDE"])
    parser.add_argument("--model")
    parser.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY"))
    parser.add_argument("--out", help="write the full results as JSON here")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = [c for c in data["cases"] if not args.only or c["id"] in args.only]

    from app.core.config import settings

    tmp = tempfile.mkdtemp(prefix="reel2real-eval-")
    settings.DATABASE_URL = ""
    settings.CONVERSATION_DB_PATH = os.path.join(tmp, "conversations.sqlite3")
    settings.CHROMA_PERSIST_DIR = os.path.join(tmp, "catalog")
    if not args.provider and not settings.GEMINI_API_KEY:
        print("Set GEMINI_API_KEY, or pass --provider/--model/--api-key.", file=sys.stderr)
        return 2

    from fastapi.testclient import TestClient
    from app.api.v1 import endpoints
    from app.core.auth import authenticated_brand
    from app.main import app
    from app.services.conversation_store import ConversationStore

    endpoints.conversation_store = ConversationStore()
    client = TestClient(app)
    llm = None
    if args.provider:
        llm = {"provider": args.provider, "model": args.model or settings.DEFAULT_MODEL, "api_key": args.api_key}

    results, failed, guarded = [], 0, 0
    for case in cases:
        payload = build_payload(case, data["fixtures"], data.get("today"))
        if llm:
            payload["llm"] = llm
        app.dependency_overrides[authenticated_brand] = lambda brand=payload["brand_id"]: brand
        res = client.post("/api/v1/generate-reply", json=payload)
        body = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
        fails = [f"HTTP {res.status_code}: {res.text[:200]}"] if res.status_code != 200 else check(case, body)
        if body.get("intent") in GUARD_INTENTS:
            guarded += 1
        failed += bool(fails)
        results.append({"id": case["id"], "ok": not fails, "fails": fails, "response": body})
        mark = "PASS" if not fails else "FAIL"
        print(f"{mark}  {case['id']:<32} {'; '.join(fails)}")
        if args.verbose or fails:
            print(f"      customer: {payload['message_text']}")
            if body.get("public_reply"):
                print(f"      public:   {body['public_reply']}")
            print(f"      dm:       {body.get('private_dm')}")
            print(f"      action={body.get('action')} intent={body.get('intent')} "
                  f"collected={body.get('collected_fields')} reasoning={body.get('reasoning')}")
    app.dependency_overrides.clear()

    total = len(cases)
    print(f"\n{total - failed}/{total} passed ({(total - failed) * 100 // max(total, 1)}%). "
          f"A guard stopped a made-up price, fact or link in {guarded} replies.")
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
