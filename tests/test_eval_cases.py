"""The real-model eval set stays runnable: every case is a valid request."""
import json
from pathlib import Path

from app.models.schemas import GenerateReplyRequest
from scripts.eval_replies import build_payload, check

DATA = json.loads((Path(__file__).resolve().parents[1] / "scripts" / "eval_cases.json").read_text(encoding="utf-8"))


def test_every_case_builds_a_valid_request():
    ids = [c["id"] for c in DATA["cases"]]
    assert len(ids) == len(set(ids))
    for case in DATA["cases"]:
        GenerateReplyRequest.model_validate(build_payload(case, DATA["fixtures"], DATA["today"]))


def test_checks_read_the_response():
    case = {"expect": {"action": ["ASK_FIELD"], "collected": {"date": "2026-11-15"}, "dm_not_contains": ["mil gaye"]}}
    assert check(case, {"action": "ASK_FIELD", "collected_fields": {"date": "2026-11-15"}, "private_dm": "City?"}) == []
    fails = check(case, {"action": "CREATE_LEAD", "collected_fields": {}, "private_dm": "Details mil gaye",
                         "intent": "safety_refusal"})
    assert len(fails) == 4
