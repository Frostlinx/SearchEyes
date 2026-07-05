"""
PGKC Entity Graph — Wikidata5M × Wiki6M × Wikipedia Images

Builds a queryable entity graph by intersecting three data sources:
1. Wikidata5M triples  (head, relation, tail)  — structured relations
2. Wiki6M              (wikidata_id → title, content, summary) — text knowledge
3. Wikipedia images    (wikidata_id.jpg)       — visual grounding

The resulting sub-graph only keeps entities that live in Wiki6M AND
have a corresponding image on disk, so every node is both
"searchable via RAG" and "visually groundable".
"""

import json
import os
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Default paths ────────────────────────────────────────────────
WIKI6M_PATH = "/dev/shm/oven_wiki/Wiki6M_ver_1_0.jsonl"
TRIPLET_PATH = "/dev/shm/oven_wiki/wikidata5m_transductive_train.txt"
ENTITY_ALIAS_PATH = "/dev/shm/oven_wiki/wikidata5m_entity.txt"
RELATION_ALIAS_PATH = "/dev/shm/oven_wiki/wikidata5m_relation.txt"
IMAGE_ROOT = "/tmp/oven_wiki_images/wikipedia_images_full"


# ── Data classes ─────────────────────────────────────────────────
class EntityNode:
    __slots__ = ("qid", "title", "summary", "image_path", "out_edges", "in_edges")

    def __init__(self, qid, title, summary="", image_path=None,
                 out_edges=None, in_edges=None):
        self.qid = qid
        self.title = title
        self.summary = summary
        self.image_path = image_path
        self.out_edges = out_edges if out_edges is not None else []
        self.in_edges = in_edges if in_edges is not None else []


class RelationInfo:
    __slots__ = ("pid", "name", "aliases")

    def __init__(self, pid, name, aliases=None):
        self.pid = pid
        self.name = name
        self.aliases = aliases if aliases is not None else []


class PGKCGraph:
    """
    Perception-Grounded Knowledge Chain graph.

    Nodes = Wiki6M entities that have a Wikipedia image.
    Edges = Wikidata5M triples restricted to those nodes.
    """

    def __init__(
        self,
        wiki6m_path: str = WIKI6M_PATH,
        triplet_path: str = TRIPLET_PATH,
        entity_alias_path: str = ENTITY_ALIAS_PATH,
        relation_alias_path: str = RELATION_ALIAS_PATH,
        image_root: str = IMAGE_ROOT,
        require_image: bool = True,
    ):
        self.wiki6m_path = wiki6m_path
        self.triplet_path = triplet_path
        self.entity_alias_path = entity_alias_path
        self.relation_alias_path = relation_alias_path
        self.image_root = image_root
        self.require_image = require_image

        self.nodes: Dict[str, EntityNode] = {}
        self.relations: Dict[str, RelationInfo] = {}

        # Indices built during load
        self._wiki6m_qids: Set[str] = set()
        self._image_qids: Set[str] = set()
        self._eligible_qids: Set[str] = set()

    # ── public API ───────────────────────────────────────────────

    def build(self) -> "PGKCGraph":
        """Load all data sources and build the sub-graph."""
        self._load_relation_aliases()
        self._load_entity_aliases()
        self._scan_wiki6m()
        if self.require_image:
            self._scan_images()
        self._compute_eligible()
        self._load_triples()
        self._log_stats()
        return self

    def neighbors(self, qid: str, direction: str = "out") -> List[Tuple[str, str, str]]:
        """Return list of (relation_pid, relation_name, neighbor_qid)."""
        node = self.nodes.get(qid)
        if node is None:
            return []
        edges = node.out_edges if direction == "out" else node.in_edges
        results = []
        for rel_pid, nbr_qid in edges:
            rel_name = self.relations[rel_pid].name if rel_pid in self.relations else rel_pid
            results.append((rel_pid, rel_name, nbr_qid))
        return results

    def get_image_path(self, qid: str) -> Optional[str]:
        """Return the on-disk image path for an entity, or None."""
        node = self.nodes.get(qid)
        return node.image_path if node else None

    def get_title(self, qid: str) -> Optional[str]:
        node = self.nodes.get(qid)
        return node.title if node else None

    def degree(self, qid: str) -> int:
        node = self.nodes.get(qid)
        if node is None:
            return 0
        return len(node.out_edges) + len(node.in_edges)

    def rich_nodes(self, min_degree: int = 3) -> List[str]:
        """Return QIDs of nodes with at least `min_degree` edges."""
        return [qid for qid, node in self.nodes.items()
                if len(node.out_edges) + len(node.in_edges) >= min_degree]

    # ── internals ────────────────────────────────────────────────

    def _load_relation_aliases(self):
        logger.info("Loading relation aliases from %s", self.relation_alias_path)
        with open(self.relation_alias_path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                pid = parts[0]
                aliases = parts[1:] if len(parts) > 1 else []
                name = aliases[0] if aliases else pid
                self.relations[pid] = RelationInfo(pid=pid, name=name, aliases=aliases)
        logger.info("  %d relations loaded", len(self.relations))

    def _load_entity_aliases(self):
        """Load entity alias file to get human-readable names for entities
        that may not appear in Wiki6M."""
        logger.info("Loading entity aliases from %s", self.entity_alias_path)
        self._entity_aliases: Dict[str, str] = {}
        with open(self.entity_alias_path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                qid = parts[0]
                if len(parts) > 1:
                    self._entity_aliases[qid] = parts[1]
        logger.info("  %d entity aliases loaded", len(self._entity_aliases))

    def _scan_wiki6m(self):
        """Scan Wiki6M to collect QIDs, titles, and summaries."""
        logger.info("Scanning Wiki6M from %s", self.wiki6m_path)
        self._wiki6m_data: Dict[str, Tuple[str, str]] = {}
        count = 0
        with open(self.wiki6m_path) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = obj["wikidata_id"]
                title = obj["wikipedia_title"]
                summary = obj.get("wikipedia_summary", "")
                self._wiki6m_data[qid] = (title, summary)
                self._wiki6m_qids.add(qid)
                count += 1
                if count % 500_000 == 0:
                    logger.info("  scanned %d entities...", count)
        logger.info("  Wiki6M total: %d entities", count)

    def _scan_images(self):
        """Walk image directory to collect QIDs that have a .jpg file."""
        logger.info("Scanning images under %s", self.image_root)
        for subdir in os.listdir(self.image_root):
            subdir_path = os.path.join(self.image_root, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in os.listdir(subdir_path):
                if fname.endswith(".jpg"):
                    qid = fname[:-4]
                    self._image_qids.add(qid)
        logger.info("  %d image files found", len(self._image_qids))

    def _compute_eligible(self):
        """Eligible = in Wiki6M ∩ has image (if required)."""
        if self.require_image:
            self._eligible_qids = self._wiki6m_qids & self._image_qids
        else:
            self._eligible_qids = self._wiki6m_qids.copy()
        logger.info("Eligible entities (Wiki6M %s image): %d",
                     "∩" if self.require_image else "only", len(self._eligible_qids))

    def _qid_to_image_path(self, qid: str) -> Optional[str]:
        """Resolve QID → image file path.

        Directory rule: if len(QID) <= 4, sub-dir = QID itself;
        otherwise sub-dir = first 4 chars of QID.
        E.g.  Q5 → Q5/Q5.jpg,  Q517545 → Q517/Q517545.jpg,
              Q27978968 → Q279/Q27978968.jpg
        """
        if qid not in self._image_qids:
            return None
        prefix = qid if len(qid) <= 4 else qid[:4]
        return os.path.join(self.image_root, prefix, qid + ".jpg")

    def _ensure_node(self, qid: str) -> EntityNode:
        if qid in self.nodes:
            return self.nodes[qid]
        title, summary = self._wiki6m_data.get(qid, (None, ""))
        if title is None:
            title = self._entity_aliases.get(qid, qid)
        image_path = self._qid_to_image_path(qid) if self.require_image else None
        node = EntityNode(qid=qid, title=title, summary=summary, image_path=image_path)
        self.nodes[qid] = node
        return node

    def _load_triples(self):
        """Load Wikidata5M triples, keeping only edges where BOTH endpoints
        are eligible entities."""
        logger.info("Loading triples from %s", self.triplet_path)
        total = 0
        kept = 0
        with open(self.triplet_path) as fh:
            for line in fh:
                total += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                head_qid, rel_pid, tail_qid = parts
                if head_qid not in self._eligible_qids or tail_qid not in self._eligible_qids:
                    continue
                head_node = self._ensure_node(head_qid)
                tail_node = self._ensure_node(tail_qid)
                head_node.out_edges.append((rel_pid, tail_qid))
                tail_node.in_edges.append((rel_pid, head_qid))
                kept += 1
                if total % 5_000_000 == 0:
                    logger.info("  processed %dM triples, kept %d...", total // 1_000_000, kept)
        logger.info("  Triples: %d total → %d kept (both endpoints eligible)", total, kept)

    def _log_stats(self):
        total_edges = sum(len(n.out_edges) for n in self.nodes.values())
        has_image = sum(1 for n in self.nodes.values() if n.image_path)
        degrees = [len(n.out_edges) + len(n.in_edges) for n in self.nodes.values()]
        avg_deg = sum(degrees) / max(len(degrees), 1)
        max_deg = max(degrees) if degrees else 0
        rich = sum(1 for d in degrees if d >= 3)
        logger.info("═══ PGKC Graph Stats ═══")
        logger.info("  Nodes:       %d", len(self.nodes))
        logger.info("  With image:  %d", has_image)
        logger.info("  Edges:       %d", total_edges)
        logger.info("  Avg degree:  %.1f", avg_deg)
        logger.info("  Max degree:  %d", max_deg)
        logger.info("  Rich (≥3):   %d", rich)
        logger.info("  Relations:   %d", len(self.relations))


# ── CLI for quick testing ────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    graph = PGKCGraph().build()

    rich = graph.rich_nodes(min_degree=5)
    logger.info("Sample rich nodes (degree≥5): %d", len(rich))
    for qid in rich[:5]:
        node = graph.nodes[qid]
        out_neighbors = graph.neighbors(qid, "out")
        logger.info("  %s (%s) — %d out-edges, image=%s",
                     node.title, qid, len(out_neighbors),
                     "yes" if node.image_path else "no")
        for pid, pname, nbr in out_neighbors[:3]:
            nbr_title = graph.get_title(nbr) or nbr
            logger.info("    → %s → %s (%s)", pname, nbr_title, nbr)
