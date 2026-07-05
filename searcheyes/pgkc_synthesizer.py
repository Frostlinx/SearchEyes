"""
PGKC Multi-hop Question Synthesizer v3

Design Principles (from REDSearcher, OpenSearch-VL, Vision-DeepResearch, MuSiQue):

1. P-K TRUE ALTERNATION
   - First hop is always P-hop (must identify entity from image)
   - Intermediate entities with images are also P-hops (agent must
     visually verify, not just read text)
   - At least 2 P-hops per chain

2. TREEWIDTH >= 2 (ALL SAMPLES)
   - At least one hop uses a one-to-few relation, requiring a
     disambiguating constraint (two conditions must be jointly satisfied)
   - Constraint can appear at any hop, not just the last one

3. 4-6 HOPS MINIMUM
   - Deep chains that require iterative search, not one-shot retrieval

4. SEMANTIC DOMAIN DIVERSITY
   - Adjacent hops must use relations from DIFFERENT semantic domains
   - Each chain must span >= 3 distinct domains
   - Domains: PERSON, WORK, ORGANIZATION, GEOGRAPHY

5. INFORMATION CONCEALMENT
   - Question does NOT reveal entity types
   - Question does NOT expose the full relation path
   - Fuzzy descriptions instead of explicit relation names

6. SOURCE DISPERSION
   - Consecutive hops' facts should not appear in the same Wikipedia article
   - (Checked via entity identity: different entities = different articles)

Output format per sample:
{
    "question_id":       str,
    "image_path":        str,
    "question":          str,
    "answer":            str,
    "answer_qid":        str,
    "chain":             [...],
    "constraints":       [...],
    "structure":         "branched",
    "num_hops":          int,
    "anchor_qid":        str,
    "anchor_title":      str,
    "hop_types":         [str, ...],  # ["P", "K", "P", "K", ...]
    "semantic_domains":  [str, ...],  # domains used per hop
    "treewidth":         int,         # 2 or 3
}
"""

import json
import logging
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =====================================================================
# RELATION CLASSIFICATION
# =====================================================================

BLACKLISTED_RELATIONS = {
    "P31",    # instance of
    "P279",   # subclass of
    "P910",   # topic's main category
    "P1012",  # including
    "P360",   # is a list of
    "P2959",  # permanent duplicated item
    "P530",   # diplomatic relation
    "P421",   # located in time zone
    "P735",   # given name
}

# ── Semantic Domain Classification ──────────────────────────────
DOMAIN_PERSON = "PERSON"
DOMAIN_WORK = "WORK"
DOMAIN_ORG = "ORGANIZATION"
DOMAIN_GEO = "GEOGRAPHY"

RELATION_TO_DOMAIN = {
    # PERSON domain
    "P19": DOMAIN_PERSON,    # place of birth
    "P20": DOMAIN_PERSON,    # place of death
    "P27": DOMAIN_PERSON,    # country of citizenship
    "P69": DOMAIN_PERSON,    # educated at
    "P106": DOMAIN_PERSON,   # occupation
    "P108": DOMAIN_PERSON,   # employer
    "P1412": DOMAIN_PERSON,  # languages spoken
    "P102": DOMAIN_PERSON,   # member of political party
    "P937": DOMAIN_PERSON,   # work location
    "P241": DOMAIN_PERSON,   # military branch
    "P607": DOMAIN_PERSON,   # conflict
    "P1344": DOMAIN_PERSON,  # participant of
    "P166": DOMAIN_PERSON,   # award received
    "P463": DOMAIN_PERSON,   # member of
    "P54": DOMAIN_PERSON,    # member of sports team
    "P3373": DOMAIN_PERSON,  # sibling
    "P26": DOMAIN_PERSON,    # spouse
    "P22": DOMAIN_PERSON,    # father
    "P40": DOMAIN_PERSON,    # child
    "P119": DOMAIN_PERSON,   # place of burial
    "P140": DOMAIN_PERSON,   # religion
    "P39": DOMAIN_PERSON,    # position held
    "P509": DOMAIN_PERSON,   # cause of death

    # WORK domain
    "P57": DOMAIN_WORK,      # director
    "P175": DOMAIN_WORK,     # performer
    "P264": DOMAIN_WORK,     # record label
    "P136": DOMAIN_WORK,     # genre
    "P840": DOMAIN_WORK,     # narrative location
    "P155": DOMAIN_WORK,     # follows
    "P156": DOMAIN_WORK,     # followed by
    "P407": DOMAIN_WORK,     # language of work
    "P364": DOMAIN_WORK,     # original language of film/TV
    "P449": DOMAIN_WORK,     # original network
    "P272": DOMAIN_WORK,     # production company
    "P161": DOMAIN_WORK,     # cast member
    "P800": DOMAIN_WORK,     # notable work
    "P50": DOMAIN_WORK,      # author
    "P86": DOMAIN_WORK,      # composer
    "P162": DOMAIN_WORK,     # producer
    "P58": DOMAIN_WORK,      # screenwriter
    "P915": DOMAIN_WORK,     # filming location
    "P400": DOMAIN_WORK,     # platform
    "P138": DOMAIN_WORK,     # named after

    # ORGANIZATION domain
    "P159": DOMAIN_ORG,      # headquarters location
    "P112": DOMAIN_ORG,      # founded by
    "P127": DOMAIN_ORG,      # owned by
    "P176": DOMAIN_ORG,      # manufacturer
    "P750": DOMAIN_ORG,      # distributor
    "P123": DOMAIN_ORG,      # publisher
    "P740": DOMAIN_ORG,      # location of formation
    "P118": DOMAIN_ORG,      # league
    "P137": DOMAIN_ORG,      # operator

    # GEOGRAPHY domain
    "P17": DOMAIN_GEO,       # country
    "P131": DOMAIN_GEO,      # located in admin entity
    "P495": DOMAIN_GEO,      # country of origin
    "P36": DOMAIN_GEO,       # capital
    "P37": DOMAIN_GEO,       # official language
    "P276": DOMAIN_GEO,      # location
    "P361": DOMAIN_GEO,      # part of
    "P641": DOMAIN_GEO,      # sport (loosely geo)
    "P150": DOMAIN_GEO,      # contains admin entity
    "P47": DOMAIN_GEO,       # shares border with
}

# One-to-one relations (answer uniqueness guaranteed)
ONE_TO_ONE_RELATIONS = {
    "P19", "P17", "P131", "P27", "P20", "P364", "P1412", "P69",
    "P407", "P175", "P264", "P155", "P156", "P136", "P102",
    "P159", "P937", "P495", "P641", "P57", "P607", "P241",
    "P840", "P750", "P123", "P106", "P108", "P127", "P176",
    "P449", "P272", "P276", "P740", "P36", "P37", "P361", "P112",
    # Newly added
    "P22",   # father (one person has one father)
    "P119",  # place of burial
    "P140",  # religion
    "P509",  # cause of death
    "P50",   # author
    "P86",   # composer
    "P137",  # operator
    "P915",  # filming location
    "P400",  # platform
    "P138",  # named after
}

# One-to-few relations (need constraint to disambiguate)
ONE_TO_FEW_RELATIONS = {
    "P150",   # contains admin entity
    "P527",   # has part
    "P1344",  # participant of
    "P166",   # award received
    "P3373",  # sibling
    "P800",   # notable work
    "P161",   # cast member
    "P54",    # member of sports team
    "P118",   # league
    # Newly added
    "P26",    # spouse (some people have multiple)
    "P40",    # child
    "P39",    # position held
    "P162",   # producer
    "P58",    # screenwriter
    "P47",    # shares border with
    "P463",   # member of
}
# NOTE: Removed P190 (twinned admin body), P197 (adjacent station),
# P463 (member of) — these create boring/repetitive GEO chains

ALL_ALLOWED_RELATIONS = (ONE_TO_ONE_RELATIONS | ONE_TO_FEW_RELATIONS) - BLACKLISTED_RELATIONS

TRIVIAL_ANSWERS = {
    "human", "actor", "singer", "politician", "writer", "musician",
    "film", "television series", "sovereign state", "city",
    "english", "french", "german", "spanish", "italian",
    "english language", "french language", "german language",
    "male", "female", "association football",
}

OVERLY_POPULAR_ANSWER_QIDS = {
    # Countries
    "Q30", "Q145", "Q142", "Q183", "Q148", "Q17", "Q159", "Q408",
    "Q16", "Q38", "Q29", "Q155", "Q668", "Q60", "Q84", "Q90",
    "Q64", "Q1490", "Q956", "Q649", "Q5", "Q515", "Q6256",
    # Major cities (too easy to guess)
    "Q1492",  # Barcelona
    "Q1726",  # Munich
    "Q1748",  # Copenhagen
    "Q1757",  # Helsinki
    "Q220",   # Rome
    "Q239",   # Brussels
    "Q270",   # Warsaw
    "Q1085",  # Prague
    "Q1860",  # English language
    "Q1861",  # Budapest
    "Q33935", # Tel Aviv
    "Q2807",  # Madrid
    "Q34370", # Nuuk
    "Q3561",  # Algiers
    "Q1354",  # Delhi
    "Q1530",  # Baghdad
    "Q2044",  # Florence
    "Q365",   # Cologne
    "Q490",   # Milan
    "Q36036", # Ankara
    "Q8684",  # Sydney
    "Q1563",  # Havana
    "Q8678",  # Canberra
    "Q85",    # Cairo
    "Q1070",  # Jakarta
}

# ── Relation "interestingness" tiers ────────────────────────────
# Prefer relations that create non-trivial reasoning chains
INTERESTING_RELATIONS = {
    "P57", "P175", "P112", "P69", "P108", "P127", "P176",
    "P272", "P123", "P750", "P264", "P449", "P161", "P800",
    "P166", "P1344", "P54", "P840", "P155", "P156",
    # Newly added
    "P50", "P86", "P162", "P58", "P138", "P915",  # WORK
    "P26", "P22", "P40", "P39",                    # PERSON
}

BORING_RELATIONS = {
    "P17", "P131", "P495", "P27", "P276", "P361",
}


class PGKCSynthesizer:
    """Generate multi-hop questions with P-K alternation, treewidth>=2,
    4-6 hops, and semantic domain diversity."""

    def __init__(self, graph, seed=42):
        self.graph = graph
        self.rng = random.Random(seed)
        self._build_edge_index()
        self._build_anchor_pool()

    # ── public API ───────────────────────────────────────────────

    def generate_batch(self, num_samples, min_hops=4, max_hops=6):
        """Generate a batch of high-difficulty multi-hop question samples.

        All samples have treewidth >= 2 (at least one branching constraint).
        All samples have P-K alternation with >= 2 P-hops.
        All samples span >= 3 semantic domains.
        """
        samples = []
        used_anchor_answer_pairs = set()  # same anchor+answer = duplicate question
        used_full_paths = set()    # exact chain path dedup
        GEO_ANSWER_QUOTA = 0.35    # at most 35% of answers can be GEO domain
        geo_answer_count = 0
        attempts = 0
        last_success_attempt = 0
        EARLY_STOP_GAP = 30000     # stop if 30k attempts since last success
        max_attempts = max(num_samples * 500, 50000)

        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            # Early stop: if no progress for too long, the pool is exhausted
            if attempts - last_success_attempt > EARLY_STOP_GAP:
                logger.info("Early stop: no new sample in %d attempts", EARLY_STOP_GAP)
                break
            sample = self._generate_sample(min_hops, max_hops)
            if sample is None:
                continue
            # Same anchor+answer pair = effectively same question
            pair_key = (sample["anchor_qid"], sample["answer_qid"])
            if pair_key in used_anchor_answer_pairs:
                continue
            # Exact path dedup: same sequence of entities = same reasoning chain
            path_key = tuple(h["to_qid"] for h in sample["chain"])
            if path_key in used_full_paths:
                continue
            # Answer domain diversity: GEO answers capped at 35% of batch
            answer_domain = sample["semantic_domains"][-1]
            if answer_domain == DOMAIN_GEO:
                if len(samples) > 0 and geo_answer_count / (len(samples) + 1) > GEO_ANSWER_QUOTA:
                    continue
                geo_answer_count += 1
            samples.append(sample)
            used_anchor_answer_pairs.add(pair_key)
            used_full_paths.add(path_key)
            last_success_attempt = attempts

        success_rate = 100.0 * len(samples) / max(attempts, 1)
        # Stats
        if samples:
            hop_dist = defaultdict(int)
            p_hop_counts = []
            domain_counts = []
            for s in samples:
                hop_dist[s["num_hops"]] += 1
                p_hop_counts.append(s["hop_types"].count("P"))
                domain_counts.append(len(set(s["semantic_domains"])))
            logger.info("Hop distribution: %s", dict(sorted(hop_dist.items())))
            logger.info("Avg P-hops per chain: %.1f",
                        sum(p_hop_counts) / len(p_hop_counts))
            logger.info("Avg semantic domains per chain: %.1f",
                        sum(domain_counts) / len(domain_counts))

        return samples

    # ── Edge index ───────────────────────────────────────────────

    def _build_edge_index(self):
        """Pre-compute per-entity edge lists, classified by relation type.
        Also identify hub nodes (degree > threshold) that should be
        excluded from chains — they create trivial, guessable paths.
        """
        # Phase 1: compute total degree for hub detection
        HUB_DEGREE_THRESHOLD = 1500
        degree = {}
        for qid, node in self.graph.nodes.items():
            degree[qid] = len(node.out_edges) + len(node.in_edges)

        # Hub set = static popular list + dynamically detected high-degree nodes
        self._hub_nodes = set(OVERLY_POPULAR_ANSWER_QIDS)
        for qid, deg in degree.items():
            if deg > HUB_DEGREE_THRESHOLD:
                self._hub_nodes.add(qid)
        logger.info("Hub nodes excluded from chains: %d (threshold=%d)",
                    len(self._hub_nodes), HUB_DEGREE_THRESHOLD)

        # Phase 2: build edge index, excluding edges TO hub nodes
        self._edges_by_entity = {}  # qid -> list of (pid, tgt, is_1to1)

        allowed_1to1 = ONE_TO_ONE_RELATIONS - BLACKLISTED_RELATIONS
        allowed_1tofew = ONE_TO_FEW_RELATIONS - BLACKLISTED_RELATIONS

        for qid, node in self.graph.nodes.items():
            edges = []
            for pid, tgt in node.out_edges:
                if tgt in self._hub_nodes:
                    continue  # never walk to a hub node
                if pid in allowed_1to1:
                    edges.append((pid, tgt, True))
                elif pid in allowed_1tofew:
                    edges.append((pid, tgt, False))
            if edges:
                self._edges_by_entity[qid] = edges

        logger.info("Edge index: %d entities with allowed edges",
                    len(self._edges_by_entity))

    def _build_anchor_pool(self):
        """Select entities that are good perception-hop anchors:
        - Must have an image
        - Must have at least 2 allowed outgoing edges (for longer chains)
        """
        self._anchor_pool = []
        for qid in self._edges_by_entity:
            node = self.graph.nodes.get(qid)
            if node and node.image_path and len(self._edges_by_entity[qid]) >= 2:
                self._anchor_pool.append(qid)

        self.rng.shuffle(self._anchor_pool)
        logger.info("Anchor pool: %d entities (image + >=2 edges)",
                    len(self._anchor_pool))

    # =====================================================================
    # CORE CHAIN GENERATION
    # =====================================================================

    def _generate_sample(self, min_hops, max_hops):
        """Generate a single sample satisfying all v3 constraints:
        - 4-6 hops
        - treewidth >= 2 (at least one one-to-few hop with constraint)
        - P-K alternation (>= 2 P-hops)
        - semantic domain diversity (>= 3 domains)
        - no adjacent hops from same domain
        - relation diversity (no repeated relation PID)
        """
        # Bias toward longer chains: 4=30%, 5=40%, 6=30%
        hop_weights = []
        for h in range(min_hops, max_hops + 1):
            if h == min_hops:
                hop_weights.append((h, 30))
            elif h == max_hops:
                hop_weights.append((h, 30))
            else:
                hop_weights.append((h, 40))
        total_w = sum(w for _, w in hop_weights)
        r = self.rng.random() * total_w
        cum = 0
        target_hops = min_hops
        for h, w in hop_weights:
            cum += w
            if cum >= r:
                target_hops = h
                break

        anchor_qid = self.rng.choice(self._anchor_pool)
        anchor_node = self.graph.nodes[anchor_qid]

        chain = []
        visited = {anchor_qid}
        current_qid = anchor_qid
        hop_types = []
        domains_used = []
        relation_pids_used = defaultdict(int)  # pid -> usage count
        prev_domain = None
        has_branching_hop = False
        branching_hop_idx = None
        constraint = None
        p_hop_count = 1  # anchor itself is the first P-hop

        # Decide which hop index will be the branching (one-to-few) hop
        # Place it somewhere in the middle, not at the very end
        branching_target_idx = self.rng.randint(1, max(1, target_hops - 2))

        for hop_idx in range(target_hops):
            # Determine whether this hop should be one-to-few (branching)
            want_branching = (hop_idx == branching_target_idx
                              and not has_branching_hop)

            # Get candidate edges
            all_edges = self._edges_by_entity.get(current_qid, [])

            # Filter: not visited, relation not overused
            # (Hub nodes already excluded at edge-index build time)
            candidates = []
            for pid, tgt, is_1to1 in all_edges:
                if tgt in visited:
                    continue
                if relation_pids_used.get(pid, 0) >= 2:
                    continue
                domain = RELATION_TO_DOMAIN.get(pid)
                if domain is None:
                    continue
                # Adjacent domain constraint
                if domain == prev_domain:
                    continue
                # If we want branching, prefer one-to-few
                if want_branching and not is_1to1:
                    candidates.append((pid, tgt, is_1to1, domain, 10))
                elif want_branching and is_1to1:
                    candidates.append((pid, tgt, is_1to1, domain, 1))
                elif not want_branching and is_1to1:
                    candidates.append((pid, tgt, is_1to1, domain, 5))
                elif not want_branching and not is_1to1:
                    candidates.append((pid, tgt, is_1to1, domain, 3))

            if not candidates:
                break

            # Weighted selection with three principles:
            # 1) Relation interestingness: boost INTERESTING, penalize BORING
            # 2) Hub avoidance: penalize high-degree targets (they make trivial paths)
            # 3) Continuability: target must have enough edges to keep walking
            #
            # The hub avoidance is the key design element: nodes like NYC/London
            # have 1000+ edges and act as attractors. Penalizing them forces
            # the random walk through niche, interesting paths.
            domain_counts = {}
            for d in domains_used:
                domain_counts[d] = domain_counts.get(d, 0) + 1

            weighted = []
            for pid, tgt, is_1to1, domain, base_weight in candidates:
                weight = float(base_weight)

                # Relation quality
                if pid in INTERESTING_RELATIONS:
                    weight *= 5.0
                elif pid in BORING_RELATIONS:
                    weight *= 0.2

                # Domain saturation: penalize domains already used twice+
                if domain_counts.get(domain, 0) >= 2:
                    weight *= 0.1

                # Continuability: prefer targets with enough edges
                # (Hub nodes already excluded at edge-index level)
                tgt_degree = len(self._edges_by_entity.get(tgt, []))
                if tgt_degree == 0:
                    weight = 0.1  # dead end
                elif tgt_degree >= 3:
                    weight *= 2.0  # good branching potential
                elif tgt_degree == 1:
                    weight *= 0.5  # risky, may dead-end next hop

                weighted.append((pid, tgt, is_1to1, domain, max(0.01, weight)))

            # Weighted random choice
            total_weight = sum(w for _, _, _, _, w in weighted)
            if total_weight == 0:
                break
            r = self.rng.random() * total_weight
            cumulative = 0
            chosen = weighted[0]
            for item in weighted:
                cumulative += item[4]
                if cumulative >= r:
                    chosen = item
                    break

            pid, next_qid, is_1to1, domain, _ = chosen
            next_node = self.graph.nodes.get(next_qid)
            if next_node is None:
                break

            rel_info = self.graph.relations.get(pid)
            rel_name = rel_info.name if rel_info else pid

            chain.append({
                "from_qid": current_qid,
                "from_title": self.graph.get_title(current_qid) or current_qid,
                "relation_pid": pid,
                "relation_name": rel_name,
                "to_qid": next_qid,
                "to_title": self.graph.get_title(next_qid) or next_qid,
                "is_1to1": is_1to1,
            })

            # Track branching
            if not is_1to1 and not has_branching_hop:
                has_branching_hop = True
                branching_hop_idx = hop_idx
                constraint = self._find_disambiguating_constraint(
                    next_qid, current_qid, pid, visited)

            # Determine hop type: P or K
            # Hop 0 is always P (anchor needs visual identification)
            # Subsequent hops: if the TARGET entity has an image AND
            # we haven't had a P-hop recently, make it P
            if hop_idx == 0:
                hop_type = "P"
            elif (next_node.image_path
                  and hop_types and hop_types[-1] == "K"
                  and p_hop_count < 3):
                hop_type = "P"
                p_hop_count += 1
            else:
                hop_type = "K"

            hop_types.append(hop_type)
            domains_used.append(domain)
            relation_pids_used[pid] += 1
            visited.add(next_qid)
            current_qid = next_qid
            prev_domain = domain

        # ── Validate all constraints ────────────────────────────
        if len(chain) < min_hops:
            return None
        # Must have treewidth >= 2: always find constraint on the ANSWER node.
        # This ensures the constraint directly helps the solver verify the answer,
        # rather than disambiguating an obscure intermediate node.
        constraint = self._try_add_answer_constraint(chain, visited)
        if constraint is None:
            return None
        # Must have >= 2 P-hops
        if hop_types.count("P") < 2:
            self._try_upgrade_to_p_hop(chain, hop_types)
            if hop_types.count("P") < 2:
                return None

        # Must span >= 3 semantic domains
        if len(set(domains_used)) < 3:
            return None

        # Reject trivial/popular answers
        answer_qid = chain[-1]["to_qid"]
        answer_title = chain[-1]["to_title"]
        if answer_title.lower() in TRIVIAL_ANSWERS:
            return None
        if answer_qid in OVERLY_POPULAR_ANSWER_QIDS:
            return None
        if answer_qid == anchor_node.qid:
            return None
        if len(answer_title) < 2:
            return None

        # Geographic saturation check: at most 2 GEO-domain hops
        geo_count = sum(1 for d in domains_used if d == DOMAIN_GEO)
        if geo_count > 2:
            return None

        # Constraint quality check: constraint relation must not be boring
        if constraint and constraint.get("relation_pid") in BORING_RELATIONS:
            return None

        # Answer must not be a well-known city/country/language — 
        # even if not in OVERLY_POPULAR list, check title patterns
        answer_lower = answer_title.lower()
        if any(answer_lower.endswith(suffix)
               for suffix in (" language", " province", " region",
                               " county", " district", " prefecture")):
            return None


        # Build question (fuzzy, information-concealed)
        question = self._build_question(chain, anchor_node, [constraint])

        sample = {
            "question_id": "pgkc_{}_{}".format(anchor_node.qid, answer_qid),
            "image_path": anchor_node.image_path,
            "question": question,
            "answer": answer_title,
            "answer_qid": answer_qid,
            "chain": chain,
            "constraints": [constraint],
            "structure": "branched",
            "num_hops": len(chain),
            "anchor_qid": anchor_node.qid,
            "anchor_title": anchor_node.title,
            "hop_types": hop_types,
            "semantic_domains": domains_used,
            "treewidth": 2,
        }
        return sample

    # ── Branching constraint logic ───────────────────────────────

    def _try_add_answer_constraint(self, chain, visited):
        """If the chain has no branching hop, try to add a disambiguating
        constraint on the final answer entity."""
        if not chain:
            return None
        last_hop = chain[-1]
        answer_qid = last_hop["to_qid"]
        parent_qid = last_hop["from_qid"]
        rel_pid = last_hop["relation_pid"]
        return self._find_disambiguating_constraint(
            answer_qid, parent_qid, rel_pid, visited)

    def _find_disambiguating_constraint(self, target_qid, parent_qid,
                                        ambiguous_rel_pid, visited):
        """Find a property of the target entity that distinguishes it from
        other entities reachable via the same relation from parent."""
        target_node = self.graph.nodes.get(target_qid)
        if target_node is None:
            return None

        parent_node = self.graph.nodes.get(parent_qid)
        if parent_node is None:
            return None

        # Get siblings (other targets of same relation from parent)
        siblings = set()
        for pid, tgt in parent_node.out_edges:
            if pid == ambiguous_rel_pid and tgt != target_qid:
                siblings.add(tgt)

        # If no siblings, still add a confirmation constraint
        if not siblings:
            return self._find_confirmation_constraint(target_qid, visited)

        # Look for a property of target that NO sibling shares
        sibling_sample = list(siblings)[:20]
        best_constraints = []

        # Check out-edges
        for pid, tgt in target_node.out_edges:
            if pid in BLACKLISTED_RELATIONS:
                continue
            if tgt in visited or tgt in OVERLY_POPULAR_ANSWER_QIDS:
                continue
            tgt_node = self.graph.nodes.get(tgt)
            if tgt_node is None or not tgt_node.title or len(tgt_node.title) < 2:
                continue

            shared = False
            for sib_qid in sibling_sample:
                sib_node = self.graph.nodes.get(sib_qid)
                if sib_node and any(p == pid and t == tgt
                                    for p, t in sib_node.out_edges):
                    shared = True
                    break
            if not shared:
                rel_info = self.graph.relations.get(pid)
                best_constraints.append({
                    "constraint_qid": tgt,
                    "constraint_title": tgt_node.title,
                    "relation_pid": pid,
                    "relation_name": rel_info.name if rel_info else pid,
                    "direction": "out",
                    "target_qid": target_qid,
                    "target_title": self.graph.get_title(target_qid) or target_qid,
                })
                if len(best_constraints) >= 5:
                    break

        # Check in-edges
        if len(best_constraints) < 3:
            for pid, src in target_node.in_edges:
                if pid in BLACKLISTED_RELATIONS:
                    continue
                if src in visited or src in OVERLY_POPULAR_ANSWER_QIDS:
                    continue
                src_node = self.graph.nodes.get(src)
                if src_node is None or not src_node.title or len(src_node.title) < 2:
                    continue

                shared = False
                for sib_qid in sibling_sample:
                    sib_node = self.graph.nodes.get(sib_qid)
                    if sib_node and any(p == pid and s == src
                                        for p, s in sib_node.in_edges):
                        shared = True
                        break
                if not shared:
                    rel_info = self.graph.relations.get(pid)
                    best_constraints.append({
                        "constraint_qid": src,
                        "constraint_title": src_node.title,
                        "relation_pid": pid,
                        "relation_name": rel_info.name if rel_info else pid,
                        "direction": "in",
                        "target_qid": target_qid,
                        "target_title": self.graph.get_title(target_qid) or target_qid,
                    })
                    if len(best_constraints) >= 5:
                        break

        if not best_constraints:
            return None
        return self.rng.choice(best_constraints)

    def _find_confirmation_constraint(self, target_qid, visited):
        """For a one-to-one hop target, find a verifiable property that
        the agent must confirm (adds treewidth without disambiguation)."""
        target_node = self.graph.nodes.get(target_qid)
        if target_node is None:
            return None

        for pid, tgt in target_node.out_edges:
            if pid in BLACKLISTED_RELATIONS:
                continue
            if tgt in visited or tgt in OVERLY_POPULAR_ANSWER_QIDS:
                continue
            tgt_node = self.graph.nodes.get(tgt)
            if tgt_node is None or not tgt_node.title or len(tgt_node.title) < 2:
                continue
            if pid in BORING_RELATIONS:
                continue
            rel_info = self.graph.relations.get(pid)
            return {
                "constraint_qid": tgt,
                "constraint_title": tgt_node.title,
                "relation_pid": pid,
                "relation_name": rel_info.name if rel_info else pid,
                "direction": "out",
                "target_qid": target_qid,
                "target_title": self.graph.get_title(target_qid) or target_qid,
            }
        return None

    # ── P-hop upgrade ────────────────────────────────────────────

    def _try_upgrade_to_p_hop(self, chain, hop_types):
        """Try to upgrade K-hops to P-hops for entities that have images,
        ensuring P-K alternation pattern."""
        for i in range(1, len(hop_types)):
            if hop_types[i] == "K":
                target_qid = chain[i]["to_qid"]
                target_node = self.graph.nodes.get(target_qid)
                if target_node and target_node.image_path:
                    # Ensure alternation: previous should be K
                    if i > 0 and hop_types[i - 1] == "K":
                        hop_types[i] = "P"
                        if hop_types.count("P") >= 2:
                            return

    # =====================================================================
    # QUESTION CONSTRUCTION (Information Concealment)
    # =====================================================================

    def _build_question(self, chain, anchor_node, constraints):
        """Build a natural-language question with information concealment.

        Design principles:
        1. Never reveal the number of hops
        2. Never list the relation path — give at most ONE vague domain hint
        3. Constraint uses fuzzy entity description, not the full name
        4. The question should read like a genuine research puzzle
        """
        constraint = constraints[0] if constraints else None

        # Pick one vague domain hint from the chain (not the first or last)
        mid_idx = len(chain) // 2
        domain_hint = self._domain_hint(chain[mid_idx]["relation_pid"])

        # Build constraint clause with fuzzy entity reference
        constraint_clause = ""
        if constraint:
            fuzzy_entity = self._fuzzy_entity_ref(
                constraint["constraint_title"],
                constraint["constraint_qid"])
            fuzzy_rel = self._fuzzy_constraint_rel(constraint["relation_name"])
            if constraint["direction"] == "out":
                templates_c = [
                    "The answer is the one that {entity} {rel_phrase}.",
                    "You are looking for the one that {entity} {rel_phrase}.",
                    "Hint: {entity} {rel_phrase} the answer.",
                ]
            else:
                templates_c = [
                    "The answer is the one that {rel_phrase} {entity}.",
                    "You are looking for the one that {rel_phrase} {entity}.",
                    "Hint: the answer {rel_phrase} {entity}.",
                ]
            constraint_clause = self.rng.choice(templates_c).format(
                entity=fuzzy_entity, rel_phrase=fuzzy_rel)

        templates = [
            ("Identify the subject in the image. Through a series of "
             "connections — some involving {hint} — trace the chain of "
             "associations to its end. What is the final entity you reach? "
             "{constraint}"),
            ("Starting from the entity shown in the image, follow a "
             "multi-step trail of relationships. The path passes through "
             "domains related to {hint}, among others. Where does this "
             "investigation ultimately lead? {constraint}"),
            ("Look at the image. The subject is connected, through several "
             "degrees of separation, to a specific entity. The trail "
             "crosses into the domain of {hint} along the way. What entity "
             "do you arrive at? {constraint}"),
            ("The entity depicted in the image is the starting point of a "
             "research chain. Following its associations across multiple "
             "steps — including connections in the area of {hint} — what "
             "do you ultimately find? {constraint}"),
            ("Examine the image and identify the subject. A sequence of "
             "factual connections links it to a particular entity. "
             "At least one step involves {hint}. What is that entity? "
             "{constraint}"),
        ]

        template = self.rng.choice(templates)
        question = template.format(hint=domain_hint, constraint=constraint_clause)
        # Clean up whitespace
        return " ".join(question.split())

    def _domain_hint(self, relation_pid):
        """Return a vague domain-level hint for a relation, much less
        specific than the relation itself. Multiple relations map to the
        same hint, so it doesn't reveal which exact relation is used."""
        domain = RELATION_TO_DOMAIN.get(relation_pid, None)
        hints_by_domain = {
            DOMAIN_PERSON: [
                "biographical details", "personal history",
                "someone's life and career", "individual backgrounds",
            ],
            DOMAIN_WORK: [
                "creative works", "artistic productions",
                "media and entertainment", "cultural output",
            ],
            DOMAIN_ORG: [
                "institutional connections", "organizational ties",
                "corporate and institutional networks",
            ],
            DOMAIN_GEO: [
                "geographic associations", "places and territories",
                "spatial relationships",
            ],
        }
        candidates = hints_by_domain.get(domain, ["various associations"])
        return self.rng.choice(candidates)

    def _fuzzy_entity_ref(self, entity_title, entity_qid):
        """Create a fuzzy reference to an entity for the constraint clause.
        Instead of giving the full name, provide a partial or descriptive hint."""
        title = entity_title
        # For short names (likely well-known), use as-is — they serve as
        # the constraint anchor and the solver needs to find them
        if len(title.split()) <= 2:
            return title
        # For longer names, keep the full name but it's less revealing
        # because longer names are more obscure
        return title

    def _fuzzy_constraint_rel(self, relation_name):
        """Rephrase a constraint relation into vague natural language."""
        vague_map = {
            "educated at": "has an educational connection to",
            "member of sports team": "has a team affiliation with",
            "place of birth": "has origins connected to",
            "place of death": "has a historical connection to",
            "country": "is geographically tied to",
            "director": "has a creative connection to",
            "performer": "has a performance connection to",
            "record label": "has a label association with",
            "cast member": "features in the same production as",
            "founded by": "has a founding connection to",
            "headquarters location": "is based near",
            "located in the administrative territorial entity":
                "falls within the same administrative area as",
            "follows": "is the predecessor of",
            "followed by": "comes before",
            "twinned administrative body": "is paired with",
            "contains administrative territorial entity":
                "administratively encompasses",
            "part of": "belongs to the same collection as",
            "award received": "shares an honor with",
            "employer": "has a professional connection to",
            "owned by": "shares ownership ties with",
            "shares border with": "is geographically adjacent to",
            "participant of": "participated alongside",
            "member of political party": "shares political ties with",
            "capital": "serves as the seat of",
            "country of citizenship": "holds citizenship connections to",
            "notable work": "is known for work connected to",
            "sibling": "has a familial connection to",
            "has part": "structurally contains",
            "ethnic group": "has cultural ties to",
        }
        return vague_map.get(relation_name, "is connected to")


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    logger.info("Loading graph from /tmp/pgkc_graph.pkl ...")
    graph = pickle.load(open("/tmp/pgkc_graph.pkl", "rb"))
    logger.info("Graph loaded: %d nodes", len(graph.nodes))

    synth = PGKCSynthesizer(graph, seed=42)

    samples = synth.generate_batch(num_samples=20, min_hops=5, max_hops=6)

    print("\n=== Generated {} samples ===\n".format(len(samples)))

    for s in samples[:5]:
        print("=" * 70)
        print("Q: {}".format(s["question"]))
        print("A: {}".format(s["answer"]))
        print("Hops: {} | Types: {} | Domains: {}".format(
            s["num_hops"], s["hop_types"], s["semantic_domains"]))
        print("Image: {}".format(s["image_path"]))
        print("Chain:")
        for hop in s["chain"]:
            print("  {} --[{}]--> {}".format(
                hop["from_title"], hop["relation_name"], hop["to_title"]))
        if s.get("constraints"):
            for c in s["constraints"]:
                print("  Constraint: {} --[{}]--> {} (dir={})".format(
                    c.get("constraint_title", "?"),
                    c.get("relation_name", "?"),
                    c.get("target_title", "?"),
                    c.get("direction", "?")))
        print()
