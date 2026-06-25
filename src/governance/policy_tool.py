"""The policy-check tool the agent invokes for inference.

It runs REAL LLM inference (via OpenRouter) over the plain-English policies and
returns a structured Verdict. It reasons over MEANING — there are no keyword
lists or regexes anywhere in this path. The model reports per-policy *evidence*;
the action is derived later by resolver.py.

The system prompt is NOT hardcoded: it is assembled at runtime from
  (a) policies/policies.md            — the authoritative policy text
  (b) scenario/policy_guidance.json   — per-policy reasoning guidance DERIVED from
                                        the policies (see governance.guidance)
  (c) the Verdict Pydantic schema     — the output contract, rendered from code
so that changing the policies (and re-deriving guidance) changes the prompt with
no code edits."""
from __future__ import annotations

import json
from typing import Optional

from . import config
from .llm import LLM
from .models import PolicyMatch, Redaction, Verdict


def _load_guidance_block() -> str:
    """Render the derived per-policy guidance, or a generic instruction if the
    guidance file is absent (the policies text below still carries the nuances)."""
    if config.GUIDANCE_PATH.exists():
        try:
            entries = json.loads(config.GUIDANCE_PATH.read_text()).get("guidance", [])
        except (json.JSONDecodeError, OSError):
            entries = []
        lines = []
        for g in sorted(entries, key=lambda e: e.get("policy_id", 99)):
            parts = [f"- Policy {g.get('policy_id')} ({g.get('name','')}): "
                     f"applies when {g.get('triggers_when','').rstrip('.')}."]
            if g.get("does_not_cover"):
                parts.append(f" Does NOT cover: {g['does_not_cover'].rstrip('.')}.")
            if g.get("confidence_note"):
                parts.append(f" Confidence: {g['confidence_note'].rstrip('.')}.")
            lines.append("".join(parts))
        if lines:
            return "\n".join(lines)
    return ("- Reason over the policy text above. Honor every parenthetical "
            "clarification. For borderline or meta-references to a sensitive topic, "
            "report LOW confidence so the system can escalate.")


def _output_contract() -> str:
    """Render the required JSON shape from the Pydantic schema (not a hand-fixed
    literal — stays in sync with models.py)."""
    pm = ", ".join(f'"{n}"' for n in PolicyMatch.model_fields)
    red = ", ".join(f'"{n}"' for n in Redaction.model_fields)
    return (
        "Return ONLY a JSON object with these keys:\n"
        f'  "policy_matches": a list of objects with keys [{pm}] '
        "(policy_id is 1-4; consent/Policy 5 is handled in code, do not report it),\n"
        f'  "redactions": a list of objects with keys [{red}] '
        "(the exact spoken value to mask, for financial identifiers only),\n"
        '  "confidence": a number 0.0-1.0,\n'
        '  "overall_reasoning": one or two sentences.\n'
        "Include a policy_matches entry only for policies you considered relevant. "
        "If nothing applies, return empty lists and high confidence."
    )


def build_system_prompt() -> str:
    policies = config.POLICIES_PATH.read_text()
    return (
        "You are a real-time meeting-governance classifier for high-stakes M&A "
        "diligence calls. You decide, utterance by utterance, how each piece of "
        "speech relates to the governance policies below. You reason over the "
        "MEANING of what is said — never by keyword or pattern matching.\n\n"
        "=== GOVERNANCE POLICIES (authoritative) ===\n"
        f"{policies}\n"
        "=== END POLICIES ===\n\n"
        "REASONING GUIDANCE (derived from the policies above):\n"
        f"{_load_guidance_block()}\n\n"
        "YOUR TASK\n"
        "For the CURRENT UTTERANCE (using the trailing context only to "
        "disambiguate), report which of Policies 1-4 it implicates, any financial "
        "identifiers to redact, and your confidence. Do NOT decide a final action.\n\n"
        "CONFIDENCE\n"
        "Use the full 0.0-1.0 range honestly. Be confident on clear cases; for "
        "borderline or meta-references (talking ABOUT a sensitive topic without "
        "disclosing the sensitive content itself), report LOW confidence.\n\n"
        "OUTPUT\n"
        f"{_output_contract()}"
    )


# Built once at import from the policy/guidance files (dynamic, not hardcoded).
SYSTEM_PROMPT = build_system_prompt()


def check_against_policies(
    utterance_text: str,
    context_text: str,
    llm: LLM,
) -> Optional[Verdict]:
    """Run one real LLM inference. Returns a validated Verdict, or None on
    refusal / unparseable output (the caller treats None as unsure -> contain)."""
    if not utterance_text.strip():
        return Verdict(confidence=1.0, overall_reasoning="empty utterance")

    user = (
        f"TRAILING CONTEXT (for disambiguation only):\n{context_text}\n\n"
        f'CURRENT UTTERANCE TO CLASSIFY:\n"{utterance_text}"\n\n'
        "Return the JSON verdict."
    )
    return llm.structured(SYSTEM_PROMPT, user, Verdict)
