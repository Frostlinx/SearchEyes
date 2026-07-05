"""
SearchEyes KB Search Tool for verl RL training.

Provides two tools for the agent to interact with the knowledge base:
  1. search_entity: Search KB entities by text query
  2. read_entity: Read a specific entity by its Wikidata QID

The KB is loaded from a JSON file into memory at initialization.
"""

import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)

# Global KB cache (loaded once, shared across tool instances)
_KB_CACHE = {}
_KB_TITLE_INDEX = {}


def load_kb(kb_path: str) -> tuple[dict, dict]:
    """Load KB from JSON file. Returns (qid_to_entity, title_to_qid)."""
    global _KB_CACHE, _KB_TITLE_INDEX
    if _KB_CACHE:
        return _KB_CACHE, _KB_TITLE_INDEX

    logger.info(f"Loading KB from {kb_path} ...")
    with open(kb_path) as f:
        raw = json.load(f)

    _KB_CACHE = raw  # qid -> {title, content, image}
    _KB_TITLE_INDEX = {}
    for qid, entity in raw.items():
        title_lower = entity.get("title", "").lower().strip()
        if title_lower:
            _KB_TITLE_INDEX[title_lower] = qid

    logger.info(f"KB loaded: {len(_KB_CACHE)} entities, {len(_KB_TITLE_INDEX)} titles indexed")
    return _KB_CACHE, _KB_TITLE_INDEX


def search_kb(query: str, kb: dict, title_index: dict, topk: int = 5) -> list[dict]:
    """Simple keyword-based search over KB entities.

    Matches query tokens against entity titles and content.
    Returns top-k results sorted by relevance score.
    """
    query_lower = query.lower().strip()
    query_tokens = set(re.findall(r'\w+', query_lower))

    if not query_tokens:
        return []

    # Exact title match first
    if query_lower in title_index:
        qid = title_index[query_lower]
        entity = kb[qid]
        return [{
            "qid": qid,
            "title": entity.get("title", ""),
            "snippet": entity.get("content", "")[:300],
            "score": 1.0,
        }]

    # Token-based scoring
    scored_results = []
    for qid, entity in kb.items():
        title = entity.get("title", "").lower()
        content = entity.get("content", "").lower()

        title_tokens = set(re.findall(r'\w+', title))
        content_tokens = set(re.findall(r'\w+', content[:500]))

        # Title match weight: 3x, content match weight: 1x
        title_overlap = len(query_tokens & title_tokens)
        content_overlap = len(query_tokens & content_tokens)
        score = title_overlap * 3.0 + content_overlap * 1.0

        if score > 0:
            scored_results.append((score, qid, entity))

    # Sort by score descending
    scored_results.sort(key=lambda x: -x[0])

    results = []
    for score, qid, entity in scored_results[:topk]:
        results.append({
            "qid": qid,
            "title": entity.get("title", ""),
            "snippet": entity.get("content", "")[:300],
            "score": round(score, 2),
        })

    return results


def format_search_results(results: list[dict]) -> str:
    """Format search results as a readable string for the agent."""
    if not results:
        return "No results found."

    lines = []
    for i, result in enumerate(results, 1):
        lines.append(
            f"[{i}] {result['title']} (ID: {result['qid']})\n"
            f"    {result['snippet']}"
        )
    return "\n\n".join(lines)


def format_entity_detail(qid: str, entity: dict) -> str:
    """Format a single entity's full details."""
    title = entity.get("title", "Unknown")
    content = entity.get("content", "No content available.")
    return f"Entity: {title} (ID: {qid})\n\n{content}"


# ═══════════════════════════════════════════════════════════
# Tool: search_entity
# ═══════════════════════════════════════════════════════════

SEARCH_SCHEMA = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the knowledge base for entities matching a query. "
            "Returns a list of matching entities with their titles, IDs, and snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
            },
            "required": ["query"],
        },
    },
})


class KBSearchTool(BaseTool):
    """Knowledge base search tool for SearchEyes agent."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        if tool_schema is None:
            tool_schema = SEARCH_SCHEMA
        super().__init__(config, tool_schema)

        kb_path = config.get("kb_path", "/tmp/pgkc_full_kb.json")
        self.topk = config.get("topk", 5)
        self.kb, self.title_index = load_kb(kb_path)
        self._instances = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instances[instance_id] = {"search_history": []}
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        if not query:
            return ToolResponse(text="Error: empty search query"), 0.0, {}

        results = search_kb(query, self.kb, self.title_index, topk=self.topk)
        result_text = format_search_results(results)

        # Track search history for this instance
        if instance_id in self._instances:
            self._instances[instance_id]["search_history"].append({
                "query": query,
                "num_results": len(results),
                "result_qids": [r["qid"] for r in results],
            })

        metrics = {
            "num_results": len(results),
            "query_length": len(query),
        }

        return ToolResponse(text=result_text), 0.0, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


# ═══════════════════════════════════════════════════════════
# Tool: read_entity
# ═══════════════════════════════════════════════════════════

READ_SCHEMA = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "lookup",
        "description": (
            "Look up a specific entity by its Wikidata ID (e.g., Q12345)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The Wikidata entity ID (e.g., Q12345)",
                },
            },
            "required": ["entity_id"],
        },
    },
})


class KBReadTool(BaseTool):
    """Knowledge base entity reader tool for SearchEyes agent."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        if tool_schema is None:
            tool_schema = READ_SCHEMA
        super().__init__(config, tool_schema)

        kb_path = config.get("kb_path", "/tmp/pgkc_full_kb.json")
        self.kb, self.title_index = load_kb(kb_path)

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        entity_id = parameters.get("entity_id", "").strip()
        if not entity_id:
            return ToolResponse(text="Error: empty entity_id"), 0.0, {}

        entity = self.kb.get(entity_id)
        if entity is None:
            return (
                ToolResponse(text=f"Entity {entity_id} not found in knowledge base."),
                0.0,
                {"found": False},
            )

        result_text = format_entity_detail(entity_id, entity)
        return ToolResponse(text=result_text), 0.0, {"found": True}


# ═══════════════════════════════════════════════════════════
# Tool: crop_and_search
# ═══════════════════════════════════════════════════════════

CROP_SEARCH_SCHEMA = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "crop_and_search",
        "description": (
            "Crop a region of the image and search for the entity depicted. "
            "Returns entity information if found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file",
                },
                "bbox": {
                    "type": "string",
                    "description": "Bounding box as 'x1, y1, x2, y2'",
                },
            },
            "required": ["image_path", "bbox"],
        },
    },
})


class KBCropAndSearchTool(BaseTool):
    """Visual entity search tool - identifies entity from image path and looks up in KB."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        if tool_schema is None:
            tool_schema = CROP_SEARCH_SCHEMA
        super().__init__(config, tool_schema)

        kb_path = config.get("kb_path", "/tmp/pgkc_full_kb.json")
        self.kb, self.title_index = load_kb(kb_path)

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        image_path = parameters.get("image_path", "")
        if not image_path:
            return ToolResponse(text="Error: empty image_path"), 0.0, {}

        # Extract QID from image path (format: .../Q123/Q12345678.jpg)
        qid_matches = re.findall(r'(Q\d+)\.\w+$', image_path)
        if not qid_matches:
            qid_matches = re.findall(r'(Q\d+)', image_path)

        if not qid_matches:
            return ToolResponse(text="Could not identify entity in image."), 0.0, {}

        entity_id = qid_matches[-1]
        entity = self.kb.get(entity_id)
        if entity is None:
            return (
                ToolResponse(text=f"Entity from image not found in knowledge base."),
                0.0,
                {"found": False},
            )

        result_text = format_entity_detail(entity_id, entity)
        return ToolResponse(text=result_text), 0.0, {"found": True}
