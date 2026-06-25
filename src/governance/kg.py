"""Governance knowledge graph (networkx).

Seeded a priori from the policies + manifest (NOT from meeting content): the
protected codename, company roles, per-speaker consent, and topic semantics.
Grown during the run with SANITIZED facts only:

  * COMMIT/REDACT/FLAG utterances may link to non-sensitive entities/topics.
  * DROP/CONSENT_GATE utterances add ONLY a content-free provenance node — no
    entity links, no text — so the graph can never surface dropped content.

The in-session graph is discarded when the process exits. Only non-sensitive,
abstracted lessons persist across sessions (see selfimprove.py)."""
from __future__ import annotations

import networkx as nx

from . import config


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.DiGraph()

    # --- seeding -----------------------------------------------------------
    def seed(self, speakers: dict[str, dict]) -> None:
        """Seed the graph from config (the ontology + codename) and the manifest
        (speakers/consent). Nothing domain-specific is hardcoded here — topics and
        entities come from scenario/ontology.json (see config.load_ontology)."""
        g = self.g
        deal = "this acquisition"
        g.add_node(deal, type="deal", note="the M&A transaction under diligence")

        # Codename — known a priori from Policy 2 (config), never learned content.
        g.add_node(config.DEAL_CODENAME, type="codename", note=(
            f"'{config.DEAL_CODENAME}' is the protected deal codename (Policy 2). "
            "A use that ties it to this transaction must not be retained; an "
            "unrelated use (e.g. a title that merely contains the phrase) is not "
            "covered."
        ))
        g.add_edge(config.DEAL_CODENAME, deal, rel="codename_for")

        ontology = config.load_ontology()

        # Entities (companies, etc.) — config-driven.
        for ent in ontology.get("entities", []):
            name = ent.get("name")
            if not name:
                continue
            g.add_node(name, type=ent.get("type", "entity"), note=ent.get("note", ""))
            if ent.get("party_to_deal"):
                g.add_edge(name, deal, rel="party_to")

        # Topics — config-driven and dynamically (re)derivable from the policies.
        for t in ontology.get("topics", []):
            name = t.get("name")
            if not name:
                continue
            g.add_node(name, type="topic", note=t.get("note", ""))
            pid = t.get("policy_id")
            if pid:
                pnode = f"Policy {pid}"
                if pnode not in g:
                    g.add_node(pnode, type="policy", note=f"governance policy {pid}")
                g.add_edge(name, pnode, rel="governed_by")

        for name, meta in speakers.items():
            consent = bool(meta.get("consent", True))
            g.add_node(name, type="person", note=(
                f"{meta.get('role','')} ({meta.get('side','')}); "
                f"consent_to_record={'yes' if consent else 'NO'}"
            ))
            if not consent:
                g.add_node("consent gate", type="policy",
                           note="Policy 5: non-consented speech is never recorded")
                g.add_edge(name, "consent gate", rel="blocked_by")

    # --- growth (sanitized) ------------------------------------------------
    def add_committed_utterance(self, utt_id: str, speaker: str,
                                policy_ids: list[int], text: str) -> None:
        """Link a kept utterance to the seeded entities/topics it mentions.
        No raw text is stored on the node."""
        self.g.add_node(utt_id, type="utterance", note=f"kept; policies={policy_ids}")
        if speaker in self.g:
            self.g.add_edge(utt_id, speaker, rel="spoken_by")
        for ent in self._mentioned_entities(text):
            self.g.add_edge(utt_id, ent, rel="mentions")

    def add_withheld_utterance(self, utt_id: str, policy_ids: list[int]) -> None:
        """Content-free provenance only — records THAT a governed removal happened,
        never what was said or which entity was involved."""
        self.g.add_node(utt_id, type="provenance",
                        note=f"governed removal under policies={policy_ids}")

    # --- retrieval (GraphRAG expansion) -----------------------------------
    def _seed_nodes(self) -> list[str]:
        return [n for n, d in self.g.nodes(data=True)
                if d.get("type") in {"codename", "company", "topic", "person", "deal"}]

    def _mentioned_entities(self, text: str) -> list[str]:
        low = text.lower()
        return [n for n in self._seed_nodes() if n.lower() in low]

    def neighbors_text(self, text: str, depth: int | None = None) -> list[str]:
        """Return short fact strings for entities mentioned in `text`, expanded
        to `depth` hops in the graph (the recursive/GraphRAG expansion)."""
        depth = config.GRAPH_DEPTH if depth is None else depth
        seeds = self._mentioned_entities(text)
        if not seeds:
            return []
        seen: set[str] = set()
        facts: list[str] = []
        frontier = list(seeds)
        for _ in range(max(1, depth)):
            nxt: list[str] = []
            for node in frontier:
                if node in seen or node not in self.g:
                    continue
                seen.add(node)
                d = self.g.nodes[node]
                if d.get("note"):
                    facts.append(f"- {node}: {d['note']}")
                for _, tgt, ed in self.g.out_edges(node, data=True):
                    rel = ed.get("rel", "related_to")
                    facts.append(f"- {node} —{rel}→ {tgt}")
                    if tgt not in seen:
                        nxt.append(tgt)
            frontier = nxt
        # De-dupe, preserve order.
        out, s = [], set()
        for f in facts:
            if f not in s:
                s.add(f)
                out.append(f)
        return out
