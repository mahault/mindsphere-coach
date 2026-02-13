"""
UserProfile + ProfileBayesNet: semantic user model with causal reasoning.

Two layers:
1. ProfileFact — raw facts extracted from conversation (flat list)
2. ProfileBayesNet — a DAG of ProfileNodes connected by CausalEdges,
   where observed facts have P=1.0 and inferred states have probabilities
   computed via forward belief propagation.

The LLM is the PRIMARY fact extractor — it catches arbitrary personal
information, not just predefined patterns. Heuristic extraction is a
fallback for when the LLM is unavailable.

The Bayesian network allows the system to reason causally about the user:
    breakup (observed, P=1.0)
        → emotional_stress (inferred, P=0.85)
            → focus_impaired (inferred, P=0.6)
            → sleep_disrupted (inferred, P=0.5)
        → social_withdrawal (inferred, P=0.4)

These inferred states feed into the POMDP belief updates and enrich
the LLM system prompt with causal context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Fact categories ──────────────────────────────────────────────────────
CATEGORY_LIFE_EVENT = "life_event"
CATEGORY_GOAL = "goal"
CATEGORY_CHALLENGE = "challenge"
CATEGORY_INTEREST = "interest"
CATEGORY_CONTEXT = "context"
CATEGORY_STRENGTH = "strength"
CATEGORY_DECISION = "decision"
CATEGORY_SELF_INSIGHT = "self_insight"
CATEGORY_INFERRED = "inferred"  # latent states from BN propagation


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class ProfileFact:
    """A single fact about the user."""
    content: str
    category: str
    turn: int
    source: str = "explicit"       # "explicit" | "inferred" | "llm"
    significance: str = "medium"   # "low" | "medium" | "high"
    valence: str = "neutral"       # "positive" | "negative" | "neutral"
    related_skills: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "category": self.category,
            "turn": self.turn,
            "source": self.source,
            "significance": self.significance,
            "valence": self.valence,
            "related_skills": self.related_skills,
        }


@dataclass
class CausalEdge:
    """A directed causal link in the Bayesian network."""
    source_id: str              # Node ID of the cause
    target_id: str              # Node ID of the effect
    strength: float             # Conditional probability influence (0-1)
    relationship: str           # "increases" | "decreases" | "triggers" | "blocks"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "strength": self.strength,
            "relationship": self.relationship,
        }


@dataclass
class ProfileNode:
    """A node in the Bayesian network — an observed fact or inferred state."""
    id: str                         # Unique identifier (e.g., "breakup_001")
    content: str                    # Human-readable description
    category: str                   # Fact category
    observed: bool                  # True = user directly stated it
    probability: float              # P(this state is active) — 0 to 1
    turn: int                       # When first observed/inferred
    valence: str = "neutral"
    significance: str = "medium"
    related_skills: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "observed": self.observed,
            "probability": round(self.probability, 3),
            "turn": self.turn,
            "valence": self.valence,
            "significance": self.significance,
            "related_skills": self.related_skills,
        }


# ── Bayesian Network ────────────────────────────────────────────────────

class ProfileBayesNet:
    """
    Bayesian network over user profile facts and inferred states.

    Nodes are observed facts (P=1.0) and inferred latent states
    (P computed via forward belief propagation). Edges are causal
    links with conditional probability strengths.

    Propagation:
        For each inferred node with parents:
        P(node) = clamp(base_rate + sum(sign_i * strength_i * P(parent_i)))
        where sign is +1 for increases/triggers, -1 for decreases/blocks.
    """

    def __init__(self):
        self.nodes: Dict[str, ProfileNode] = {}
        self.edges: List[CausalEdge] = []
        self._node_counter: int = 0

    def _next_id(self, prefix: str = "n") -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter:03d}"

    def add_observed_fact(self, fact: ProfileFact) -> str:
        """Add a directly observed fact as a node with P=1.0. Returns node ID."""
        node_id = self._next_id("obs")
        self.nodes[node_id] = ProfileNode(
            id=node_id,
            content=fact.content,
            category=fact.category,
            observed=True,
            probability=1.0,
            turn=fact.turn,
            valence=fact.valence,
            significance=fact.significance,
            related_skills=list(fact.related_skills),
        )
        return node_id

    def add_inferred_state(
        self,
        content: str,
        base_probability: float = 0.3,
        turn: int = 0,
        valence: str = "neutral",
        significance: str = "medium",
        related_skills: Optional[List[str]] = None,
    ) -> str:
        """Add a latent inferred state (e.g., 'emotional_stress'). Returns node ID."""
        # Check for existing node with similar content
        for nid, node in self.nodes.items():
            if self._content_similar(node.content, content):
                return nid  # Reuse existing node

        node_id = self._next_id("inf")
        self.nodes[node_id] = ProfileNode(
            id=node_id,
            content=content,
            category=CATEGORY_INFERRED,
            observed=False,
            probability=base_probability,
            turn=turn,
            valence=valence,
            significance=significance,
            related_skills=related_skills or [],
        )
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        strength: float = 0.5,
        relationship: str = "increases",
    ) -> None:
        """Add a causal edge from source to target."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        # Don't add duplicate edges
        for e in self.edges:
            if e.source_id == source_id and e.target_id == target_id:
                e.strength = max(e.strength, strength)  # strengthen existing
                return
        self.edges.append(CausalEdge(
            source_id=source_id,
            target_id=target_id,
            strength=min(max(strength, 0.0), 1.0),
            relationship=relationship,
        ))

    def propagate(self) -> None:
        """
        Forward belief propagation through the DAG.

        For each inferred node, compute:
            P(node) = clamp(base + sum(sign_i * strength_i * P(parent_i)))

        Uses topological ordering to ensure parents are computed first.
        """
        order = self._topological_sort()
        for node_id in order:
            node = self.nodes[node_id]
            if node.observed:
                continue  # Observed facts stay at P=1.0

            # Find all incoming edges
            parent_effects = []
            for edge in self.edges:
                if edge.target_id == node_id and edge.source_id in self.nodes:
                    parent = self.nodes[edge.source_id]
                    sign = 1.0 if edge.relationship in ("increases", "triggers") else -1.0
                    parent_effects.append(sign * edge.strength * parent.probability)

            if parent_effects:
                # Base rate + causal influence
                base = 0.1  # small base rate for unobserved states
                node.probability = max(0.0, min(1.0, base + sum(parent_effects)))

    def _topological_sort(self) -> List[str]:
        """Topological sort of nodes for forward propagation."""
        # Build adjacency
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        children: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for edge in self.edges:
            if edge.source_id in self.nodes and edge.target_id in self.nodes:
                in_degree[edge.target_id] += 1
                children[edge.source_id].append(edge.target_id)

        # Kahn's algorithm
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for child in children.get(nid, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Add any remaining (cycles — shouldn't happen but be safe)
        for nid in self.nodes:
            if nid not in order:
                order.append(nid)
        return order

    def get_active_states(self, threshold: float = 0.3) -> List[ProfileNode]:
        """Get all nodes with probability above threshold, sorted by P descending."""
        active = [n for n in self.nodes.values() if n.probability >= threshold]
        active.sort(key=lambda n: n.probability, reverse=True)
        return active

    def get_skill_impacts(self) -> Dict[str, float]:
        """
        Compute net impact on POMDP skill factors from the Bayesian network.

        Returns a dict mapping skill names to impact scores (-1 to +1).
        Positive = this network state suggests skill is stronger than baseline.
        Negative = this network state suggests skill is impaired.
        """
        impacts: Dict[str, float] = {}
        for node in self.nodes.values():
            if node.probability < 0.2:
                continue
            for skill in node.related_skills:
                if skill not in impacts:
                    impacts[skill] = 0.0
                sign = -1.0 if node.valence == "negative" else 1.0
                weight = node.probability * (0.15 if node.significance == "high" else 0.08)
                impacts[skill] += sign * weight
        return impacts

    def format_for_prompt(self) -> str:
        """Format the Bayesian network as a readable section for the LLM prompt."""
        active = self.get_active_states(threshold=0.25)
        if not active:
            return ""

        lines = ["\n\n## Causal Model of This Person"]

        observed = [n for n in active if n.observed]
        inferred = [n for n in active if not n.observed]

        if observed:
            lines.append("  **Known facts:**")
            for node in observed:
                lines.append(f"    - {node.content}")

        if inferred:
            lines.append("  **Inferred states (from causal reasoning):**")
            for node in inferred:
                pct = int(node.probability * 100)
                lines.append(f"    - {node.content} ({pct}% likely)")

        # Show key causal chains
        chains = self._get_causal_chains()
        if chains:
            lines.append("  **Causal connections:**")
            for chain in chains[:5]:  # Top 5 chains
                lines.append(f"    - {chain}")

        lines.append(
            "\nUse these causal insights to be genuinely perceptive. If their breakup "
            "is causing emotional stress which is impairing focus, connect those dots "
            "naturally in conversation — don't recite the model."
        )
        return "\n".join(lines)

    def _get_causal_chains(self) -> List[str]:
        """Get human-readable causal chain descriptions."""
        chains = []
        for edge in self.edges:
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            if not src or not tgt:
                continue
            if src.probability < 0.3:
                continue
            arrow = "→" if edge.relationship in ("increases", "triggers") else "⊣"
            pct = int(edge.strength * 100)
            chains.append(f"{src.content} {arrow} {tgt.content} ({pct}%)")
        return chains

    def _content_similar(self, a: str, b: str) -> bool:
        """Check if two content strings are similar enough to be the same node."""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return False
        overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
        return overlap > 0.6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "skill_impacts": self.get_skill_impacts(),
        }


# ── Skill-to-impact mapping for LLM-extracted causal links ──────────────
# Maps common inferred state keywords to POMDP skill factors
LATENT_STATE_SKILLS = {
    "emotional_stress": ["emotional_reg", "focus"],
    "emotional stress": ["emotional_reg", "focus"],
    "anxiety": ["emotional_reg", "focus", "self_trust"],
    "low_confidence": ["self_trust", "social_courage"],
    "low confidence": ["self_trust", "social_courage"],
    "grief": ["emotional_reg", "consistency"],
    "loneliness": ["social_courage", "emotional_reg"],
    "overwhelm": ["focus", "consistency", "emotional_reg"],
    "motivation_loss": ["follow_through", "consistency"],
    "motivation loss": ["follow_through", "consistency"],
    "focus_impaired": ["focus", "systems_thinking"],
    "focus impaired": ["focus", "systems_thinking"],
    "sleep_disrupted": ["focus", "emotional_reg", "consistency"],
    "sleep disrupted": ["focus", "emotional_reg", "consistency"],
    "social_withdrawal": ["social_courage"],
    "social withdrawal": ["social_courage"],
    "identity_crisis": ["self_trust", "task_clarity"],
    "identity crisis": ["self_trust", "task_clarity"],
    "burnout": ["emotional_reg", "consistency", "focus"],
    "financial_stress": ["emotional_reg", "focus"],
    "financial stress": ["emotional_reg", "focus"],
    "self_doubt": ["self_trust", "follow_through"],
    "self doubt": ["self_trust", "follow_through"],
    "procrastination": ["follow_through", "task_clarity", "focus"],
    "perfectionism": ["follow_through", "self_trust"],
    "imposter_syndrome": ["self_trust", "social_courage"],
    "imposter syndrome": ["self_trust", "social_courage"],
}


# ── Main UserProfile class ──────────────────────────────────────────────

class UserProfile:
    """
    Accumulates structured facts about the user and builds a causal
    Bayesian network connecting them.

    The LLM is the primary extractor — it catches arbitrary personal
    information and infers causal connections. Keyword heuristics are
    a fallback for when the LLM is unavailable.

    Usage:
        profile = UserProfile()
        new_facts = profile.extract_and_store(
            user_text="I just broke up with my boyfriend",
            turn=10,
            classifier=llm_classifier,
        )
        # Causal model for system prompt
        bn_section = profile.bayes_net.format_for_prompt()
        # Skill impacts for POMDP
        impacts = profile.bayes_net.get_skill_impacts()
    """

    def __init__(self):
        self.facts: List[ProfileFact] = []
        self.bayes_net = ProfileBayesNet()

    def extract_and_store(
        self,
        user_text: str,
        turn: int,
        classifier=None,
        context: str = "",
    ) -> List[ProfileFact]:
        """
        Extract facts from user text, store them, and update the Bayesian network.

        LLM extraction is primary — it catches ANY meaningful personal information
        and infers causal connections. Heuristics are fallback only.
        """
        new_facts = []
        causal_links = []

        # PRIMARY: LLM extraction (flexible, catches anything)
        if classifier is not None:
            try:
                llm_result = self._extract_via_llm(user_text, turn, classifier, context)
                if llm_result:
                    new_facts.extend(llm_result["facts"])
                    causal_links.extend(llm_result.get("causal_links", []))
            except Exception as e:
                logger.debug(f"LLM profile extraction failed: {e}")

        # FALLBACK: Heuristic extraction (when LLM unavailable)
        if not new_facts:
            heuristic_facts = self._extract_heuristic(user_text, turn)
            for hf in heuristic_facts:
                if not self._is_duplicate(hf, new_facts):
                    new_facts.append(hf)

        # Deduplicate against existing profile
        truly_new = []
        for fact in new_facts:
            if not self._is_duplicate(fact, self.facts):
                self.facts.append(fact)
                truly_new.append(fact)

        # Update Bayesian network with new facts
        if truly_new:
            self._update_bayes_net(truly_new, causal_links, turn)
            logger.info(
                f"[Profile] Extracted {len(truly_new)} new facts: "
                + ", ".join(f"'{f.content}' ({f.category})" for f in truly_new)
            )
            if causal_links:
                logger.info(
                    f"[Profile] {len(causal_links)} causal links: "
                    + ", ".join(f"'{l['from']}' → '{l['to']}'" for l in causal_links)
                )

        return truly_new

    def _extract_via_llm(
        self,
        user_text: str,
        turn: int,
        classifier,
        context: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract profile facts AND causal connections using LLM.

        Returns:
            {"facts": [ProfileFact, ...], "causal_links": [{"from": ..., "to": ..., ...}]}
        """
        # Build context about what we already know
        existing_summary = ""
        if self.facts:
            existing_items = [f.content for f in self.facts[-5:]]
            existing_summary = f"\nAlready known about this person: {'; '.join(existing_items)}"

        prompt = (
            "Analyze this message for personal information. Extract TWO things:\n\n"
            "1. FACTS: Any meaningful personal information — life events, emotions, "
            "goals, challenges, relationships, work, health, interests, decisions, "
            "realizations, or context about their life. Be flexible — extract "
            "ANYTHING personally significant, not just predefined categories.\n\n"
            "2. CAUSAL LINKS: How do the new facts connect causally to each other "
            "or to existing facts? What does this fact likely CAUSE or AFFECT?\n"
            "For example: 'breakup' → 'emotional_stress' (strength 0.8)\n"
            "             'emotional_stress' → 'focus_impaired' (strength 0.6)\n\n"
            f"{existing_summary}\n"
            f"Message: \"{user_text}\"\n\n"
            "Return JSON with this exact structure:\n"
            "{\n"
            '  "facts": [\n'
            '    {"content": "...", "category": "life_event|goal|challenge|interest'
            '|context|strength|decision|self_insight", '
            '"significance": "low|medium|high", "valence": "positive|negative|neutral", '
            '"related_skills": ["focus", "emotional_reg", ...]}\n'
            "  ],\n"
            '  "causal_links": [\n'
            '    {"from": "fact content or existing fact", "to": "inferred state or effect", '
            '"strength": 0.0-1.0, "relationship": "increases|decreases|triggers|blocks", '
            '"to_valence": "positive|negative|neutral"}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Only extract genuinely meaningful information, not trivial filler\n"
            "- related_skills should map to: focus, follow_through, social_courage, "
            "emotional_reg, systems_thinking, self_trust, task_clarity, consistency\n"
            "- Causal links can point to latent states the user didn't explicitly say "
            "(like 'emotional_stress' inferred from 'breakup')\n"
            "- Return empty arrays if the message has no significant personal content\n"
            "- Keep fact content concise (under 100 chars)"
        )

        try:
            result = classifier.client.chat_completion(
                messages=[
                    {"role": "system", "content": (
                        "You are a structured information extractor for a coaching system. "
                        "Extract personal facts and causal connections from conversation. "
                        "Return valid JSON only."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            if not result:
                return None

            data = json.loads(result)
            facts_data = data.get("facts", [])
            links_data = data.get("causal_links", [])

            # Parse facts
            facts = []
            for item in facts_data:
                if isinstance(item, dict) and item.get("content"):
                    skills = item.get("related_skills", [])
                    if isinstance(skills, str):
                        skills = [skills]
                    facts.append(ProfileFact(
                        content=item["content"][:120],
                        category=item.get("category", "context"),
                        turn=turn,
                        source="llm",
                        significance=item.get("significance", "medium"),
                        valence=item.get("valence", "neutral"),
                        related_skills=skills,
                    ))

            # Parse causal links
            causal_links = []
            for link in links_data:
                if isinstance(link, dict) and link.get("from") and link.get("to"):
                    causal_links.append({
                        "from": link["from"],
                        "to": link["to"],
                        "strength": float(link.get("strength", 0.5)),
                        "relationship": link.get("relationship", "increases"),
                        "to_valence": link.get("to_valence", "neutral"),
                    })

            return {"facts": facts, "causal_links": causal_links}

        except Exception as e:
            logger.debug(f"LLM profile extraction parse error: {e}")
            return None

    def _update_bayes_net(
        self,
        new_facts: List[ProfileFact],
        causal_links: List[Dict[str, Any]],
        turn: int,
    ) -> None:
        """Update the Bayesian network with new facts and causal connections."""
        # 1. Add observed facts as nodes
        fact_node_ids = {}
        for fact in new_facts:
            node_id = self.bayes_net.add_observed_fact(fact)
            fact_node_ids[fact.content.lower()] = node_id

        # 2. Process causal links from LLM
        for link in causal_links:
            from_content = link["from"]
            to_content = link["to"]
            strength = link.get("strength", 0.5)
            relationship = link.get("relationship", "increases")
            to_valence = link.get("to_valence", "neutral")

            # Find or create source node
            source_id = self._find_node_by_content(from_content)
            if not source_id:
                # Check if it matches a new fact
                source_id = fact_node_ids.get(from_content.lower())
            if not source_id:
                continue  # Can't find source — skip

            # Find or create target node (may be an inferred state)
            target_id = self._find_node_by_content(to_content)
            if not target_id:
                # Create as inferred state
                related = self._skills_for_state(to_content)
                target_id = self.bayes_net.add_inferred_state(
                    content=to_content,
                    base_probability=0.1,
                    turn=turn,
                    valence=to_valence,
                    significance="medium",
                    related_skills=related,
                )

            self.bayes_net.add_edge(source_id, target_id, strength, relationship)

        # 3. Add heuristic causal links for facts that the LLM didn't connect
        if not causal_links:
            self._add_heuristic_causal_links(new_facts, turn)

        # 4. Propagate beliefs through the network
        self.bayes_net.propagate()

    def _find_node_by_content(self, content: str) -> Optional[str]:
        """Find an existing node that matches this content."""
        content_lower = content.lower()
        for nid, node in self.bayes_net.nodes.items():
            if self.bayes_net._content_similar(node.content.lower(), content_lower):
                return nid
        return None

    def _skills_for_state(self, state_content: str) -> List[str]:
        """Look up which POMDP skills a latent state affects."""
        lower = state_content.lower().replace(" ", "_")
        # Direct match
        if lower in LATENT_STATE_SKILLS:
            return LATENT_STATE_SKILLS[lower]
        # Partial match
        lower_space = state_content.lower()
        if lower_space in LATENT_STATE_SKILLS:
            return LATENT_STATE_SKILLS[lower_space]
        # Fuzzy match on keywords
        for key, skills in LATENT_STATE_SKILLS.items():
            if key.replace("_", " ") in lower_space or lower_space in key.replace("_", " "):
                return skills
        return []

    def _add_heuristic_causal_links(self, facts: List[ProfileFact], turn: int) -> None:
        """Add common-sense causal links when LLM doesn't provide them."""
        for fact in facts:
            node_id = self._find_node_by_content(fact.content)
            if not node_id:
                continue

            lower = fact.content.lower()

            # Breakup / relationship loss → emotional stress
            if any(w in lower for w in ["broke up", "breakup", "break up", "divorced"]):
                stress_id = self.bayes_net.add_inferred_state(
                    "emotional stress", 0.1, turn, "negative", "high",
                    ["emotional_reg", "focus"],
                )
                self.bayes_net.add_edge(node_id, stress_id, 0.8, "triggers")
                focus_id = self.bayes_net.add_inferred_state(
                    "focus impaired", 0.1, turn, "negative", "medium",
                    ["focus"],
                )
                self.bayes_net.add_edge(stress_id, focus_id, 0.5, "increases")

            # Job loss → anxiety + self-doubt
            elif any(w in lower for w in ["fired", "lost my job", "laid off"]):
                anxiety_id = self.bayes_net.add_inferred_state(
                    "financial anxiety", 0.1, turn, "negative", "high",
                    ["emotional_reg", "focus"],
                )
                self.bayes_net.add_edge(node_id, anxiety_id, 0.7, "triggers")
                doubt_id = self.bayes_net.add_inferred_state(
                    "self-doubt", 0.1, turn, "negative", "medium",
                    ["self_trust", "follow_through"],
                )
                self.bayes_net.add_edge(node_id, doubt_id, 0.5, "increases")

            # Bereavement → grief + social withdrawal
            elif any(w in lower for w in ["passed away", "lost someone", "died"]):
                grief_id = self.bayes_net.add_inferred_state(
                    "grief", 0.1, turn, "negative", "high",
                    ["emotional_reg", "consistency"],
                )
                self.bayes_net.add_edge(node_id, grief_id, 0.9, "triggers")

            # Overwhelm → focus + consistency impaired
            elif any(w in lower for w in ["overwhelmed", "too much", "drowning"]):
                focus_id = self.bayes_net.add_inferred_state(
                    "focus impaired", 0.1, turn, "negative", "medium",
                    ["focus"],
                )
                self.bayes_net.add_edge(node_id, focus_id, 0.6, "increases")

    def _extract_heuristic(self, user_text: str, turn: int) -> List[ProfileFact]:
        """Fallback: extract profile facts using keyword/pattern matching."""
        facts = []
        lower = user_text.lower()

        # --- Life events ---
        life_event_patterns = {
            "broke up": ("relationship breakup", "negative", "high",
                         ["emotional_reg", "social_courage"]),
            "break up": ("relationship breakup", "negative", "high",
                         ["emotional_reg", "social_courage"]),
            "breakup": ("relationship breakup", "negative", "high",
                        ["emotional_reg", "social_courage"]),
            "divorced": ("divorce", "negative", "high", ["emotional_reg"]),
            "got fired": ("job loss", "negative", "high",
                          ["self_trust", "emotional_reg"]),
            "lost my job": ("job loss", "negative", "high",
                            ["self_trust", "emotional_reg"]),
            "laid off": ("job loss", "negative", "high",
                         ["self_trust", "emotional_reg"]),
            "got promoted": ("promotion", "positive", "high",
                             ["self_trust", "follow_through"]),
            "new job": ("started new job", "positive", "high",
                        ["self_trust", "focus"]),
            "moved to": ("relocated", "neutral", "medium", ["consistency"]),
            "graduated": ("graduated", "positive", "high",
                          ["follow_through", "consistency"]),
            "had a baby": ("new parent", "positive", "high",
                           ["consistency", "emotional_reg"]),
            "got married": ("married", "positive", "high",
                            ["emotional_reg", "social_courage"]),
            "lost someone": ("bereavement", "negative", "high",
                             ["emotional_reg"]),
            "passed away": ("bereavement", "negative", "high",
                            ["emotional_reg"]),
            "diagnosed": ("health diagnosis", "negative", "high",
                          ["emotional_reg", "self_trust"]),
        }

        for pattern, (label, valence, significance, skills) in life_event_patterns.items():
            if pattern in lower:
                facts.append(ProfileFact(
                    content=user_text.strip()[:120],
                    category=CATEGORY_LIFE_EVENT,
                    turn=turn,
                    significance=significance,
                    valence=valence,
                    related_skills=skills,
                ))
                break

        # --- Goals ---
        goal_patterns = [
            "i want to", "i need to", "my goal", "i'm trying to",
            "i hope to", "i'd like to", "i wish i could",
            "looking for", "searching for", "working toward",
        ]
        for pattern in goal_patterns:
            if pattern in lower:
                idx = lower.index(pattern)
                goal_text = user_text[idx:].strip()[:120]
                facts.append(ProfileFact(
                    content=goal_text, category=CATEGORY_GOAL, turn=turn,
                    significance="high", valence="neutral",
                ))
                break

        # --- Challenges ---
        challenge_patterns = [
            "i struggle with", "i can't seem to", "my problem is",
            "what stops me", "i keep failing", "i'm stuck on",
            "the hardest part", "what's blocking me",
        ]
        for pattern in challenge_patterns:
            if pattern in lower:
                idx = lower.index(pattern)
                facts.append(ProfileFact(
                    content=user_text[idx:].strip()[:120],
                    category=CATEGORY_CHALLENGE, turn=turn,
                    significance="high", valence="negative",
                ))
                break

        # --- Context ---
        context_patterns = [
            "i work as", "i'm a ", "my job is", "i work in",
            "i study", "i'm studying", "i live in",
            "i have kids", "i have children",
            "my partner", "my husband", "my wife",
        ]
        for pattern in context_patterns:
            if pattern in lower:
                idx = lower.index(pattern)
                facts.append(ProfileFact(
                    content=user_text[idx:].strip()[:120],
                    category=CATEGORY_CONTEXT, turn=turn,
                    significance="medium", valence="neutral",
                ))
                break

        # --- Interests ---
        interest_patterns = [
            "i love", "i enjoy", "i'm passionate about",
            "i'm interested in", "my hobby", "i like to",
        ]
        for pattern in interest_patterns:
            if pattern in lower:
                idx = lower.index(pattern)
                facts.append(ProfileFact(
                    content=user_text[idx:].strip()[:120],
                    category=CATEGORY_INTEREST, turn=turn,
                    significance="medium", valence="positive",
                ))
                break

        # --- Decisions ---
        decision_patterns = [
            "i'll try", "i'm going to", "i will", "i've decided",
            "i commit to", "let me try", "i'll do",
        ]
        for pattern in decision_patterns:
            if pattern in lower:
                idx = lower.index(pattern)
                facts.append(ProfileFact(
                    content=user_text[idx:].strip()[:120],
                    category=CATEGORY_DECISION, turn=turn,
                    significance="medium", valence="positive",
                    related_skills=["follow_through"],
                ))
                break

        # --- Self-insights ---
        insight_patterns = [
            "i realize", "i never thought", "i just noticed",
            "it hit me that", "i'm starting to see",
            "i think the real issue", "i never do this",
        ]
        for pattern in insight_patterns:
            if pattern in lower:
                idx = lower.index(pattern)
                facts.append(ProfileFact(
                    content=user_text[idx:].strip()[:120],
                    category=CATEGORY_SELF_INSIGHT, turn=turn,
                    significance="high", valence="neutral",
                    related_skills=["self_trust"],
                ))
                break

        return facts

    def _is_duplicate(self, new_fact: ProfileFact, existing: List[ProfileFact]) -> bool:
        """Check if a fact is a duplicate of an existing one."""
        for ef in existing:
            if ef.category == new_fact.category and self._content_overlap(ef.content, new_fact.content):
                return True
        return False

    def _content_overlap(self, a: str, b: str) -> bool:
        """Check if two fact contents overlap significantly."""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return False
        overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
        return overlap > 0.6

    def format_for_prompt(self) -> str:
        """
        Format the full profile for the LLM system prompt.
        Includes both flat facts and the Bayesian network causal model.
        """
        sections = []

        # Flat facts section (backward-compatible)
        if self.facts:
            lines = ["\n\n## What I Know About This Person"]
            by_category: Dict[str, List[ProfileFact]] = {}
            for fact in self.facts:
                by_category.setdefault(fact.category, []).append(fact)

            category_labels = {
                CATEGORY_LIFE_EVENT: "Life Events",
                CATEGORY_GOAL: "Goals",
                CATEGORY_CHALLENGE: "Challenges",
                CATEGORY_INTEREST: "Interests",
                CATEGORY_CONTEXT: "Context",
                CATEGORY_STRENGTH: "Observed Strengths",
                CATEGORY_DECISION: "Decisions Made",
                CATEGORY_SELF_INSIGHT: "Self-Insights",
            }
            for category, label in category_labels.items():
                if category in by_category:
                    lines.append(f"  **{label}:**")
                    for fact in by_category[category]:
                        sig = " (!)" if fact.significance == "high" else ""
                        lines.append(f"    - {fact.content}{sig}")
            sections.append("\n".join(lines))

        # Bayesian network causal model
        bn_section = self.bayes_net.format_for_prompt()
        if bn_section:
            sections.append(bn_section)

        if not sections:
            return ""

        combined = "\n".join(sections)
        combined += (
            "\n\nUse this information naturally in conversation. Reference specific "
            "things they've shared. Connect coaching insights to their real situation. "
            "Never recite this list — weave it in like a friend who remembers."
        )
        return combined

    def get_belief_signals(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get signals that should influence POMDP belief updates.
        Combines direct fact signals with Bayesian network skill impacts.
        """
        signals: Dict[str, List[Dict[str, Any]]] = {}

        # From direct facts
        for fact in self.facts:
            for skill in fact.related_skills:
                if skill not in signals:
                    signals[skill] = []
                if fact.valence == "negative":
                    signals[skill].append({
                        "direction": "stress",
                        "strength": 0.15 if fact.significance == "high" else 0.08,
                        "source": fact.content[:50],
                    })
                elif fact.valence == "positive":
                    signals[skill].append({
                        "direction": "boost",
                        "strength": 0.10 if fact.significance == "high" else 0.05,
                        "source": fact.content[:50],
                    })

        # From Bayesian network inferred states
        for node in self.bayes_net.nodes.values():
            if node.observed or node.probability < 0.3:
                continue
            for skill in node.related_skills:
                if skill not in signals:
                    signals[skill] = []
                direction = "stress" if node.valence == "negative" else "boost"
                signals[skill].append({
                    "direction": direction,
                    "strength": node.probability * 0.1,
                    "source": f"[inferred] {node.content[:40]}",
                })

        return signals

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the profile for the API."""
        return {
            "n_facts": len(self.facts),
            "categories": {
                cat: len([f for f in self.facts if f.category == cat])
                for cat in set(f.category for f in self.facts)
            },
            "facts": [f.to_dict() for f in self.facts],
            "bayes_net": self.bayes_net.to_dict(),
        }

    def reset(self):
        """Clear all facts and the Bayesian network."""
        self.facts = []
        self.bayes_net = ProfileBayesNet()
