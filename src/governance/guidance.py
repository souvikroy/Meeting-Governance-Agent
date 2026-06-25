"""Dynamic policy-reasoning guidance.

The policy-check system prompt is NOT hand-written — it is assembled at runtime
from the plain-English policies (policies/policies.md) plus the per-policy
reasoning guidance derived here. Regenerate the guidance whenever the policies
change:

  uv run python -m governance.guidance
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from . import config
from .llm import LLM


class PolicyGuidance(BaseModel):
    policy_id: int = Field(description="policy number 1-5")
    name: str = Field(description="short policy name")
    triggers_when: str = Field(description="what meaning makes this policy apply")
    does_not_cover: str = Field(default="", description="look-alikes that do NOT apply")
    confidence_note: str = Field(default="", description="how to calibrate confidence / borderline handling")


class GuidanceSet(BaseModel):
    guidance: list[PolicyGuidance] = Field(default_factory=list)


_SYSTEM = (
    "You write the reasoning guide for a meeting-governance classifier. Given the "
    "plain-English governance policies, produce concise per-policy guidance the "
    "classifier will use to reason over MEANING (never keywords). For each policy: "
    "what meaning makes it apply (triggers_when); the look-alikes that do NOT apply "
    "(does_not_cover) — honor every parenthetical clarification in the policies; and "
    "a confidence note for borderline/ambiguous or meta-reference cases "
    "(confidence_note). Output ONLY JSON of the form "
    '{"guidance":[{"policy_id":N,"name":"...","triggers_when":"...","does_not_cover":"...","confidence_note":"..."}]}.'
)


def derive_guidance(llm: Optional[LLM] = None) -> list[dict]:
    llm = llm or LLM()
    policies = config.POLICIES_PATH.read_text()
    user = f"GOVERNANCE POLICIES:\n{policies}\n\nReturn the guidance JSON."
    result = llm.structured(_SYSTEM, user, GuidanceSet)
    if not result or not result.guidance:
        raise SystemExit("LLM returned no guidance; leaving the existing file unchanged.")
    return [g.model_dump() for g in result.guidance]


def main() -> None:
    config.require("OPENROUTER_API_KEY")
    guidance = derive_guidance()
    config.GUIDANCE_PATH.write_text(json.dumps({"guidance": guidance}, indent=2))
    print(f"Wrote {len(guidance)} policy-guidance entries to {config.GUIDANCE_PATH}:")
    for g in guidance:
        print(f"  - Policy {g['policy_id']} ({g['name']})")


if __name__ == "__main__":
    main()
