import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governance.models import Action, PolicyMatch, Redaction, Verdict
from governance.resolver import resolve

HI = 0.9
LO = 0.2


def v(matches=None, redactions=None, conf=HI):
    return Verdict(
        policy_matches=[PolicyMatch(**m) for m in (matches or [])],
        redactions=[Redaction(**r) for r in (redactions or [])],
        confidence=conf,
    )


def test_commit_when_nothing_matches():
    assert resolve(v()).action == Action.COMMIT


def test_drop_compensation():
    assert resolve(v([{"policy_id": 1, "matched": True}])).action == Action.DROP


def test_drop_codename():
    assert resolve(v([{"policy_id": 2, "matched": True}])).action == Action.DROP


def test_redact_financial():
    r = resolve(v([{"policy_id": 4, "matched": True}],
                  [{"value": "4012-8888-3320-7741"}]))
    assert r.action == Action.REDACT


def test_flag_legal_exposure():
    assert resolve(v([{"policy_id": 3, "matched": True}])).action == Action.FLAG_FOR_REVIEW


def test_precedence_drop_beats_redact():
    # Compensation + a card number in the same utterance -> DROP wins (no copy).
    r = resolve(v([{"policy_id": 1, "matched": True}, {"policy_id": 4, "matched": True}],
                  [{"value": "4012-8888-3320-7741"}]))
    assert r.action == Action.DROP


def test_none_verdict_is_contained():
    r = resolve(None)
    assert r.action == Action.DROP and r.unsure


def test_low_confidence_drop_suspect_contained():
    # Unsure AND touches a drop-class policy -> contain.
    r = resolve(v([{"policy_id": 1, "matched": False, "span": "comp philosophy"}], conf=LO))
    assert r.action == Action.DROP and r.unsure


def test_low_confidence_benign_escalates():
    # Unsure, no drop-class signal -> keep but flag for review.
    r = resolve(v(conf=LO))
    assert r.action == Action.FLAG_FOR_REVIEW and r.unsure
