"""Dynamic topic derivation.

The knowledge-graph topics are NOT hardcoded — they live in the config-controlled
ontology (scenario/ontology.json) and can be regenerated from the plain-English
policies with the LLM, so they stay in sync when the policies change.

  uv run python -m governance.topics          # re-derive topics from policies
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from . import config
from .llm import LLM


class TopicSpec(BaseModel):
    name: str = Field(description="short lowercase topic name")
    note: str = Field(description="one line: what it covers + the governance action")
    policy_id: Optional[int] = Field(default=None, description="mapped policy 1-5, if any")


class TopicSet(BaseModel):
    topics: list[TopicSpec] = Field(default_factory=list)


_SYSTEM = (
    "You design the topic taxonomy for a meeting-governance knowledge graph. "
    "Given the plain-English governance policies, extract the distinct governance "
    "TOPICS the system should track. For each topic return: a short lowercase "
    "name; a one-line note describing what it covers AND the governance action "
    "(drop / redact / flag / gate); and the policy_id (1-5) it maps to when "
    "applicable. Reason over meaning, not keywords. Output ONLY JSON of the form "
    '{"topics":[{"name":"...","note":"...","policy_id":N}]}.'
)


def derive_topics(llm: Optional[LLM] = None) -> list[dict]:
    """Use the LLM to derive topics from the current policies. Returns a list of
    {name, note, policy_id} dicts."""
    llm = llm or LLM()
    policies = config.POLICIES_PATH.read_text()
    user = f"GOVERNANCE POLICIES:\n{policies}\n\nReturn the topics JSON."
    result = llm.structured(_SYSTEM, user, TopicSet)
    if not result or not result.topics:
        raise SystemExit("LLM returned no topics; leaving the existing ontology unchanged.")
    return [t.model_dump() for t in result.topics]


def main() -> None:
    config.require("OPENROUTER_API_KEY")
    topics = derive_topics()
    ontology = config.load_ontology()
    ontology["topics"] = topics
    ontology["_comment"] = (
        "Config-controlled governance ontology. Topics regenerated from the "
        "policies via: uv run python -m governance.topics"
    )
    config.ONTOLOGY_PATH.write_text(json.dumps(ontology, indent=2))
    print(f"Wrote {len(topics)} topics to {config.ONTOLOGY_PATH}:")
    for t in topics:
        print(f"  - {t['name']} (policy {t.get('policy_id')})")


if __name__ == "__main__":
    main()
