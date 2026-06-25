import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governance.kg import KnowledgeGraph

SPEAKERS = {
    "Maya Okafor": {"role": "Partner", "side": "Northwind", "consent": True},
    "Tomas Herrera": {"role": "Counsel", "side": "Cendara law", "consent": False},
}


def _kg():
    kg = KnowledgeGraph()
    kg.seed(SPEAKERS)
    return kg


def test_codename_seeded_a_priori():
    kg = _kg()
    facts = kg.neighbors_text("let's align on Project Atlas timeline", depth=1)
    joined = "\n".join(facts)
    assert "codename" in joined.lower()


def test_non_consent_modeled():
    kg = _kg()
    assert kg.g.has_edge("Tomas Herrera", "consent gate")


def test_withheld_utterance_has_no_content():
    kg = _kg()
    kg.add_withheld_utterance("u07", policy_ids=[5])
    note = kg.g.nodes["u07"]["note"]
    # provenance only — never the content or the entity involved
    assert "removal" in note.lower()
    assert "atlas" not in note.lower()
    assert kg.g.out_degree("u07") == 0   # no entity links from a withheld utterance


def test_committed_utterance_links_entities():
    kg = _kg()
    kg.add_committed_utterance("u01", "Maya Okafor", [], "intro from Northwind Capital")
    assert kg.g.has_edge("u01", "Maya Okafor")
    assert kg.g.has_edge("u01", "Northwind Capital")
