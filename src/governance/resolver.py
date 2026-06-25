"""Precedence resolver — turns a model Verdict into a final Action in plain,
testable code. The model reports evidence (per-policy matches + redactions); the
resolver applies the precedence ladder and the conservative "unsure" rule.

Precedence:  CONSENT_GATE  >  DROP  >  REDACT  >  FLAG_FOR_REVIEW  >  COMMIT
(CONSENT_GATE is applied earlier, in the orchestrator, before the LLM runs.)

Unsure rule (liability product): if confidence < threshold OR the verdict is
missing (refusal / unparseable), do NOT commit sensitive content in the clear.
If the uncertainty could plausibly be a DROP-class concern, contain (DROP-equiv);
otherwise keep-but-FLAG for human review. Either way it is escalated."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import config
from .models import Action, Verdict

# Policies whose presence forces a no-copy DROP.
_DROP_POLICIES = {1, 2}      # compensation, deal codename
_FLAG_POLICY = 3             # legal/regulatory exposure


@dataclass
class Resolution:
    action: Action
    policy_ids: list[int]
    confidence: float
    unsure: bool
    reason: str               # generic, non-leaking


def resolve(verdict: Optional[Verdict],
            threshold: float | None = None) -> Resolution:
    threshold = config.CONFIDENCE_THRESHOLD if threshold is None else threshold

    # No verdict at all (refusal / unparseable) -> unsure, contain.
    if verdict is None:
        return Resolution(Action.DROP, [], 0.0, unsure=True,
                          reason="no verdict (refusal/parse-fail) — contained")

    matched = verdict.matched_ids()
    confident = verdict.confidence >= threshold

    # Confident, clear verdict: apply the ladder.
    drop_hit = bool(matched & _DROP_POLICIES)
    redact_hit = bool(verdict.redactions)
    flag_hit = _FLAG_POLICY in matched

    if confident:
        if drop_hit:
            # Contain either way; escalate for review if not highly confident.
            escalate = verdict.confidence < config.HIGH_CONFIDENCE
            reason = "policy match — contained"
            if escalate:
                reason += " (low confidence — escalated for human review)"
            return Resolution(Action.DROP, sorted(matched & _DROP_POLICIES),
                              verdict.confidence, escalate, reason)
        if redact_hit:
            return Resolution(Action.REDACT, [4] if 4 in matched else [4],
                              verdict.confidence, False, "financial identifier — redacted")
        if flag_hit:
            return Resolution(Action.FLAG_FOR_REVIEW, [_FLAG_POLICY],
                              verdict.confidence, False, "possible legal exposure — flagged")
        return Resolution(Action.COMMIT, [], verdict.confidence, False, "no policy match")

    # --- Unsure path -------------------------------------------------------
    # Low confidence. Never commit in the clear. Choose containment vs flag by
    # whether the uncertainty touches a DROP-class concern.
    if drop_hit or (matched & _DROP_POLICIES) or _suspects_drop(verdict):
        return Resolution(Action.DROP, sorted(matched), verdict.confidence, True,
                          "unsure about sensitive (drop-class) content — contained")
    # Otherwise keep but escalate to a human.
    pol = sorted(matched) or [_FLAG_POLICY]
    return Resolution(Action.FLAG_FOR_REVIEW, pol, verdict.confidence, True,
                      "unsure — escalated to human review")


def _suspects_drop(verdict: Verdict) -> bool:
    """A near-miss on a drop-class policy (matched=False but rationale signals
    compensation/codename) should still be contained when we're unsure."""
    for m in verdict.policy_matches:
        if m.policy_id in _DROP_POLICIES and (m.matched or (m.span and m.span.strip())):
            return True
    return False
