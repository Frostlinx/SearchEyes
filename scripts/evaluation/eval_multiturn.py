#!/usr/bin/env python3
"""
eval_multiturn.py — Multi-Turn Agentic Evaluation for SearchEyes
=================================================================
Evaluates SearchEyes models on visual search benchmarks using a ReAct-style
agent loop with tool calling (search, read_entity, answer).

Supports two search backends:
  1. Local KB  — uses the PKC knowledge base (for internal eval)
  2. Serper API — uses Google Search via Serper (for standard benchmarks)

Supported benchmarks:
  - SimpleVQA   (HuggingFace: m-a-p/SimpleVQA)
  - InfoSeek    (HuggingFace: google/infoseek)
  - FVQA        (local JSONL)
  - PKC-Test    (local JSONL, uses local KB)

Usage:
    # Local KB eval on PKC test set
    python eval_multiturn.py \
        --model-path ./Searcheyes-9b-sft \
        --benchmark pkc-test \
        --search-backend local \
        --kb-path /tmp/pgkc_full_kb.json \
        --max-samples 100

    # Web search eval on SimpleVQA
    python eval_multiturn.py \
        --model-path ./Searcheyes-9b-sft \
        --benchmark simplevqa \
        --search-backend serper \
        --serper-api-key YOUR_KEY \
        --max-samples 200

    # Use already-running vLLM server
    python eval_multiturn.py \
        --api-base http://localhost:8000/v1 \
        --model-name Searcheyes-9b-sft \
        --benchmark simplevqa \
        --search-backend serper
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import base64
import string
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. Search Backends
# ═══════════════════════════════════════════════════════════

class SearchBackend:
    """Base class for search backends."""

    def search(self, query: str, top_k: int = 5) -> str:
        raise NotImplementedError

    def read_entity(self, entity_id: str) -> str:
        raise NotImplementedError


class LocalKBBackend(SearchBackend):
    """Search backend using a local JSON knowledge base (token-overlap, legacy)."""

    def __init__(self, kb_path: str, top_k: int = 5):
        logger.info(f"Loading local KB from {kb_path} ...")
        with open(kb_path) as f:
            self.kb: dict[str, dict] = json.load(f)
        self.top_k = top_k

        self.title_index: dict[str, str] = {}
        for qid, entity in self.kb.items():
            title_lower = entity.get("title", "").lower().strip()
            if title_lower:
                self.title_index[title_lower] = qid
        logger.info(f"KB loaded: {len(self.kb)} entities")

    def search(self, query: str, top_k: int = 5) -> str:
        query_lower = query.lower().strip()
        query_tokens = set(re.findall(r"\w+", query_lower))
        if not query_tokens:
            return "No results found."

        # Exact title match
        if query_lower in self.title_index:
            qid = self.title_index[query_lower]
            entity = self.kb[qid]
            return (
                f"[1] {entity.get('title', '')} (ID: {qid})\n"
                f"    {entity.get('content', '')[:300]}"
            )

        # Token-based scoring
        scored = []
        for qid, entity in self.kb.items():
            title = entity.get("title", "").lower()
            content = entity.get("content", "").lower()
            title_tokens = set(re.findall(r"\w+", title))
            content_tokens = set(re.findall(r"\w+", content[:500]))
            score = len(query_tokens & title_tokens) * 3.0 + len(query_tokens & content_tokens)
            if score > 0:
                scored.append((score, qid, entity))

        scored.sort(key=lambda x: -x[0])
        if not scored:
            return "No results found."

        lines = []
        for i, (score, qid, entity) in enumerate(scored[: top_k or self.top_k], 1):
            lines.append(
                f"[{i}] {entity.get('title', '')} (ID: {qid})\n"
                f"    {entity.get('content', '')[:300]}"
            )
        return "\n\n".join(lines)

    def read_entity(self, entity_id: str) -> str:
        entity_id = entity_id.strip()
        entity = self.kb.get(entity_id)
        if entity is None:
            return f"Entity {entity_id} not found in the knowledge base."
        title = entity.get("title", "Unknown")
        content = entity.get("content", "No content available.")
        return f"Entity: {title} (ID: {entity_id})\n\n{content}"


class RagEngineBackend(SearchBackend):
    """Search backend using RagEngine (embedding + BM25 + RRF hybrid retrieval).

    This is the SAME retrieval engine used during SearchEyes RL training,
    ensuring train-eval consistency.
    """

    def __init__(
        self,
        wiki6m_path: str = "/dev/shm/oven_wiki/Wiki6M_ver_1_0.jsonl",
        embeddings_dir: str | None = "/dev/shm/oven_wiki/embeddings",
        bm25_index_path: str | None = "/dev/shm/oven_wiki/bm25_index.json",
        top_k: int = 5,
    ):
        import sys
        searcheyes_dir = str(Path(__file__).parent / "searcheyes")
        if searcheyes_dir not in sys.path:
            sys.path.insert(0, searcheyes_dir)

        from rag_engine import RagEngine

        logger.info("Initializing RagEngineBackend (same as training)...")
        self.engine = RagEngine(
            wiki6m_path=wiki6m_path,
            embeddings_dir=embeddings_dir,
            bm25_index_path=bm25_index_path,
        )
        self.engine.load()
        self.top_k = top_k
        logger.info("RagEngineBackend ready")

    def search(self, query: str, top_k: int = 5) -> str:
        effective_top_k = top_k or self.top_k
        facts = self.engine.get_rag_facts_combined(
            text=query, top_k=effective_top_k, use_hybrid=True
        )
        if not facts:
            return "No results found."

        lines = []
        for i, fact in enumerate(facts, 1):
            snippet = fact.caption or fact.summary or ""
            lines.append(
                f"[{i}] {fact.title} (ID: {fact.wit_id})\n"
                f"    {snippet[:300]}"
            )
        return "\n\n".join(lines)

    def read_entity(self, entity_id: str) -> str:
        entity_id = entity_id.strip()
        result = self.engine.read_full_entity(entity_id)
        if result is None:
            return f"Entity {entity_id} not found in the knowledge base."
        title = result.get("title", "Unknown")
        content = result.get("content", "") or result.get("summary", "No content available.")
        return f"Entity: {title} (ID: {entity_id})\n\n{content}"


class SerperBackend(SearchBackend):
    """Search backend using Serper.dev Google Search API."""

    SEARCH_URL = "https://google.serper.dev/search"
    SCRAPE_URL = "https://scrape.serper.dev"

    def __init__(self, api_key: str, top_k: int = 5):
        self.api_key = api_key
        self.top_k = top_k
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        logger.info("Serper search backend initialized")

    def search(self, query: str, top_k: int = 5) -> str:
        try:
            response = requests.post(
                self.SEARCH_URL,
                headers=self.headers,
                json={"q": query, "num": top_k or self.top_k},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return f"Search error: {exc}"

        lines = []
        # Knowledge graph snippet
        knowledge_graph = data.get("knowledgeGraph", {})
        if knowledge_graph:
            title = knowledge_graph.get("title", "")
            description = knowledge_graph.get("description", "")
            if title:
                lines.append(f"[Knowledge Graph] {title}\n    {description}")

        # Organic results
        for i, result in enumerate(data.get("organic", [])[: top_k or self.top_k], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            lines.append(f"[{i}] {title}\n    {snippet}\n    URL: {link}")

        return "\n\n".join(lines) if lines else "No results found."

    def read_entity(self, entity_id: str) -> str:
        """For web search, read_entity fetches a URL or searches for the entity."""
        # If entity_id looks like a URL, scrape it
        if entity_id.startswith("http"):
            return self._scrape_url(entity_id)
        # Otherwise, treat as a search query
        return self.search(entity_id, top_k=3)

    def _scrape_url(self, url: str) -> str:
        try:
            response = requests.post(
                self.SCRAPE_URL,
                headers=self.headers,
                json={"url": url},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("text", "")[:2000]
            return text if text else "Could not extract content from URL."
        except Exception as exc:
            return f"Scrape error: {exc}"


# ═══════════════════════════════════════════════════════════
# 2. Answer Extraction & Matching
# ═══════════════════════════════════════════════════════════

def normalize_answer(text: str) -> str:
    """Lowercase, remove articles, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def extract_final_answer(response: str) -> str | None:
    """Extract the final answer from the agent's last message.

    Supports multiple formats:
      - Action: answer(text="...")
      - Answer: ...
      - <answer>...</answer>
    """
    # Format 1: Action: answer(text="...")
    match = re.search(r'answer\(text=["\'](.+?)["\']\)', response)
    if match:
        return match.group(1).strip()

    # Format 2: <answer>...</answer>
    matches = list(re.finditer(r"<answer>(.*?)</answer>", response, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()

    # Format 3: Answer: ... (at end of message)
    match = re.search(r"Answer:\s*(.+?)$", response, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return None


def exact_match(prediction: str, golden_answers: str | list[str]) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized = normalize_answer(prediction)
    return any(normalize_answer(ga) == normalized for ga in golden_answers)


def substring_match(prediction: str, golden_answers: str | list[str]) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized = normalize_answer(prediction)
    return any(normalize_answer(ga) in normalized for ga in golden_answers)


# ═══════════════════════════════════════════════════════════
# 3. Agent Loop
# ═══════════════════════════════════════════════════════════

MAX_REPEAT_TURN = 3  # consecutive repetition limit before forced answer

SYSTEM_PROMPT = """You are a visual research assistant. Given an image and a complex multi-hop question, \
you must find the answer by searching through a knowledge base step by step.

When you have gathered sufficient information and are ready to provide the definitive response, \
you must enclose the entire final answer within <answer></answer> tags.  \
The answer inside <answer> tags should be a short, precise entity name — NOT a long sentence.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Search the knowledge base with a text query. Returns top results with titles, IDs, and summaries.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "read_entity", "description": "Read the full Wikipedia content of a specific entity by its ID.", "parameters": {"type": "object", "properties": {"entity_id": {"type": "string", "description": "The entity ID (e.g. Q12345) to read."}}, "required": ["entity_id"]}}}
</tools>

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Strategy

1. First, carefully examine the image to identify the entity shown.
2. Use search to find relevant entities and verify your identification.
3. Follow the chain of relationships described in the question step by step.
4. At each hop, search for or read the relevant entity and verify the connection before moving on.
5. Only provide your final answer when you have traced the COMPLETE chain and have sufficient evidence.
6. Your final answer must be a precise entity name enclosed in <answer></answer> tags.

Think step by step. At each step, explain your reasoning, then call a tool or provide your answer."""


@dataclass
class AgentTurn:
    """One turn in the agent loop."""
    thought: str = ""
    action_raw: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    observation: str = ""
    is_final_answer: bool = False
    final_answer: str = ""


def analyze_repetition_ngram(text: str, n: int = 30, threshold: float = 0.5) -> bool:
    """Detect repetition via N-gram distinct ratio (from VisionDR)."""
    if not text or len(text) < n:
        return False
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    total_count = len(ngrams)
    if total_count == 0:
        return False
    from collections import Counter
    unique_count = len(Counter(ngrams))
    return (unique_count / total_count) < threshold


def count_words(text: str) -> int:
    """Count English words in text (from VisionDR)."""
    return len(re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*").findall(text))


def parse_action(assistant_text: str) -> tuple[str, dict, bool, str]:
    """Parse the assistant's action from its response.

    Supports VisionDR-style XML format:
      - <tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>
      - <answer>...</answer>

    Returns: (tool_name, tool_args, is_final_answer, final_answer_text)
    """
    # Priority 1: <tool_call> XML format (VisionDR / OpenSearch-VL standard)
    if "<tool_call>" in assistant_text and "</tool_call>" in assistant_text:
        tool_call_text = assistant_text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        try:
            tool_call = json.loads(tool_call_text)
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            if tool_name == "answer":
                return "answer", {}, True, tool_args.get("text", "")
            return tool_name, tool_args, False, ""
        except json.JSONDecodeError:
            pass

    # Priority 2: <answer>...</answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", assistant_text, re.DOTALL)
    if answer_match:
        return "answer", {}, True, answer_match.group(1).strip()

    # Fallback 3: answer(text="...") function-call style
    answer_match = re.search(r'answer\(text=["\'](.+?)["\']\)', assistant_text)
    if answer_match:
        return "answer", {}, True, answer_match.group(1).strip()

    # Fallback 4: Answer: ... line format
    answer_match = re.search(r"Answer:\s*(.+?)$", assistant_text, re.MULTILINE)
    if answer_match:
        return "answer", {}, True, answer_match.group(1).strip()

    # Fallback 5: Legacy JSON tool call {"tool": "search", "args": {...}}
    json_match = re.search(r'\{["\']tool["\']\s*:\s*["\'](\w+)["\'].*?\}', assistant_text, re.DOTALL)
    if json_match:
        try:
            json_str = _extract_json_object(assistant_text, json_match.start())
            tool_call = json.loads(json_str)
            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})
            if tool_name == "answer":
                return "answer", {}, True, tool_args.get("text", "")
            return tool_name, tool_args, False, ""
        except (json.JSONDecodeError, ValueError):
            pass

    # Nothing parseable
    return "", {}, False, ""


def _extract_json_object(text: str, start: int) -> str:
    """Extract a complete JSON object from text starting at the given position."""
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        char = text[i]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


class VLLMClient:
    """Client for vLLM / OpenAI-compatible API."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model_name: str = "default",
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.95,
        api_key: str | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send a chat completion request and return assistant content."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stop": kwargs.get("stop", None),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            logger.error("API request timed out after 300s")
            return "[API Error: timeout]"
        except Exception as exc:
            logger.error(f"API error: {type(exc).__name__}: {exc}")
            return f"[API Error: {exc}]"


def _force_final_answer(client: VLLMClient, messages: list[dict], max_turns: int) -> str:
    """Send a final instruction to force the model to produce an <answer>."""
    final_instruction = {
        "role": "user",
        "content": (
            f"You have reached the maximum number of reasoning rounds ({max_turns}). "
            "Based on all the information gathered so far, please provide your best "
            "final answer now in the format: <answer>your answer</answer>"
        ),
    }
    messages.append(final_instruction)
    final_content = client.chat(messages)
    messages.append({"role": "assistant", "content": final_content.strip()})

    if "<answer>" in final_content and "</answer>" in final_content:
        return final_content.split("<answer>")[1].split("</answer>")[0].strip()
    return final_content.strip() if final_content.strip() else "No answer found."


def run_agent_episode(
    client: VLLMClient,
    search_backend: SearchBackend,
    question: str,
    image_path: str | None = None,
    max_turns: int = 50,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """Run one multi-turn agent episode (VisionDR-aligned agent loop).

    Key design choices aligned with VisionDR:
      - Tool calls use <tool_call>{"name":"...","arguments":{...}}</tool_call> XML
      - Tool results returned as <tool_response>...</tool_response> (role=user)
      - Final answers use <answer>...</answer>
      - Format errors prompt the model to retry
      - Repetition detection forces early answer
      - Round limit forces final answer

    Returns dict with:
      - turns: list of turn dicts
      - final_answer: str or None
      - num_search_calls: int
      - num_read_calls: int
      - total_turns: int
      - termination: str (reason for stopping)
      - raw_messages: list of message dicts
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Build user message with image (support vision API via base64)
    if image_path and Path(image_path).is_file():
        with open(image_path, "rb") as img_f:
            img_b64 = base64.b64encode(img_f.read()).decode("utf-8")
        suffix = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{img_b64}"}},
                {"type": "text", "text": f"Question: {question}"},
            ],
        }
    else:
        text_content = ""
        if image_path:
            text_content += f"Image: {image_path}\n\n"
        text_content += f"Question: {question}"
        user_message = {"role": "user", "content": text_content}
    messages.append(user_message)

    turns: list[dict] = []
    final_answer = None
    termination = None
    num_search_calls = 0
    num_read_calls = 0
    repetition_count = 0

    for turn_idx in range(max_turns):
        # Get model response
        assistant_text = client.chat(messages)

        if not assistant_text or assistant_text.startswith("[API Error"):
            logger.warning(f"Turn {turn_idx}: API error: {assistant_text}")
            termination = "api_error"
            break

        # Strip any hallucinated <tool_response> from model output
        if "<tool_response>" in assistant_text:
            assistant_text = assistant_text[: assistant_text.find("<tool_response>")]

        turn_record = {
            "turn_index": turn_idx,
            "assistant_text": assistant_text,
            "tool_name": "",
            "tool_args": {},
            "is_final_answer": False,
            "observation": "",
        }

        # ── Priority 1: <tool_call> present → execute tool ──
        if "<tool_call>" in assistant_text and "</tool_call>" in assistant_text:
            messages.append({"role": "assistant", "content": assistant_text.strip()})

            tool_name, tool_args, is_final, answer_text = parse_action(assistant_text)
            turn_record["tool_name"] = tool_name
            turn_record["tool_args"] = tool_args

            if is_final:
                final_answer = answer_text
                turn_record["is_final_answer"] = True
                turn_record["final_answer"] = answer_text
                turns.append(turn_record)
                termination = "answer"
                break

            # Execute tool
            observation = ""
            if tool_name == "search":
                query = tool_args.get("query", "")
                if query:
                    observation = search_backend.search(query)
                    num_search_calls += 1
                else:
                    observation = "Error: empty search query."
            elif tool_name == "read_entity":
                entity_id = tool_args.get("entity_id", "")
                if entity_id:
                    observation = search_backend.read_entity(entity_id)
                    num_read_calls += 1
                else:
                    observation = "Error: empty entity_id."
            elif tool_name:
                observation = f"Unknown tool: {tool_name}. Available tools: search, read_entity."
            else:
                observation = "[Json Parse Error]: Tool call is not a valid JSON."

            # Truncate long observations
            if len(observation) > 3000:
                observation = observation[:3000] + "\n[...truncated]"

            turn_record["observation"] = observation
            turns.append(turn_record)

            # Add tool response in VisionDR format
            tool_response = f"<tool_response>\n{observation}\n</tool_response>"
            messages.append({"role": "user", "content": tool_response})
            repetition_count = 0  # reset on successful tool use
            continue

        # ── Priority 2: <answer> present → extract final answer ──
        if "<answer>" in assistant_text and "</answer>" in assistant_text:
            messages.append({"role": "assistant", "content": assistant_text.strip()})
            final_answer = assistant_text.split("<answer>")[1].split("</answer>")[0].strip()
            turn_record["is_final_answer"] = True
            turn_record["final_answer"] = final_answer
            turns.append(turn_record)
            termination = "answer"
            break

        # ── Priority 3: No tool_call / answer → format error ──
        is_repetitive = analyze_repetition_ngram(assistant_text)
        is_overlong = count_words(assistant_text) > 2500

        if is_repetitive and is_overlong:
            repetition_count += 1
            logger.warning(
                f"Turn {turn_idx}: Content repetition detected "
                f"(count: {repetition_count}/{MAX_REPEAT_TURN})"
            )
            if repetition_count >= MAX_REPEAT_TURN:
                # Force final answer
                messages.append({"role": "assistant", "content": assistant_text.strip()})
                final_answer = _force_final_answer(client, messages, max_turns)
                turn_record["is_final_answer"] = True
                turn_record["final_answer"] = final_answer
                turns.append(turn_record)
                termination = "repetition_limit"
                break

        # Send format error prompt (like VisionDR)
        messages.append({"role": "assistant", "content": assistant_text.strip()})
        format_error_msg = (
            "Error: Invalid content format. Content must contain "
            "<tool_call> or <answer> tags. Please try again with the correct format.\n"
            'To call a tool: <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>\n'
            "To answer: <answer>your answer</answer>"
        )
        messages.append({"role": "user", "content": format_error_msg})
        turn_record["observation"] = format_error_msg
        turns.append(turn_record)

    # ── Round limit reached without answer → force final answer ──
    if final_answer is None:
        if turns:
            final_answer = _force_final_answer(client, messages, max_turns)
            termination = "round_limit"
        else:
            termination = "no_output"

    return {
        "turns": turns,
        "final_answer": final_answer,
        "num_search_calls": num_search_calls,
        "num_read_calls": num_read_calls,
        "total_turns": len(turns),
        "termination": termination or "answer",
        "raw_messages": messages,
    }


# ═══════════════════════════════════════════════════════════
# 4. Benchmark Data Loaders
# ═══════════════════════════════════════════════════════════

@dataclass
class EvalSample:
    sample_id: str
    question: str
    image_path: str | None
    golden_answers: list[str]
    metadata: dict = field(default_factory=dict)


def load_simplevqa(data_path: str, max_samples: int | None = None) -> list[EvalSample]:
    """Load SimpleVQA dataset from HuggingFace-downloaded JSONL or parquet."""
    samples = []
    path = Path(data_path)

    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                samples.append(EvalSample(
                    sample_id=str(obj.get("id", len(samples))),
                    question=obj["question"],
                    image_path=obj.get("image_path", obj.get("image", None)),
                    golden_answers=[obj["answer"]] if isinstance(obj["answer"], str) else obj["answer"],
                    metadata={"source": "simplevqa"},
                ))
                if max_samples and len(samples) >= max_samples:
                    break
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        for obj in data:
            samples.append(EvalSample(
                sample_id=str(obj.get("id", len(samples))),
                question=obj["question"],
                image_path=obj.get("image_path", obj.get("image", None)),
                golden_answers=[obj["answer"]] if isinstance(obj["answer"], str) else obj["answer"],
                metadata={"source": "simplevqa"},
            ))
            if max_samples and len(samples) >= max_samples:
                break
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .jsonl or .json")

    logger.info(f"Loaded {len(samples)} SimpleVQA samples from {data_path}")
    return samples


def load_infoseek(data_path: str, max_samples: int | None = None) -> list[EvalSample]:
    """Load InfoSeek dataset from JSONL."""
    samples = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            answer = obj.get("answer", obj.get("answers", ""))
            if isinstance(answer, str):
                answer = [answer]
            elif isinstance(answer, list) and answer and isinstance(answer[0], dict):
                answer = [a.get("answer", "") for a in answer]
            samples.append(EvalSample(
                sample_id=str(obj.get("data_id", obj.get("id", len(samples)))),
                question=obj.get("question", ""),
                image_path=obj.get("image_path", obj.get("image", None)),
                golden_answers=answer,
                metadata={"source": "infoseek", "entity": obj.get("entity", "")},
            ))
            if max_samples and len(samples) >= max_samples:
                break
    logger.info(f"Loaded {len(samples)} InfoSeek samples from {data_path}")
    return samples


def load_fvqa(data_path: str, max_samples: int | None = None) -> list[EvalSample]:
    """Load FVQA dataset from JSONL."""
    samples = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            answer = obj.get("answer", obj.get("answers", ""))
            if isinstance(answer, str):
                answer = [answer]
            samples.append(EvalSample(
                sample_id=str(obj.get("id", len(samples))),
                question=obj.get("question", ""),
                image_path=obj.get("image_path", obj.get("image", None)),
                golden_answers=answer,
                metadata={"source": "fvqa"},
            ))
            if max_samples and len(samples) >= max_samples:
                break
    logger.info(f"Loaded {len(samples)} FVQA samples from {data_path}")
    return samples


def load_pkc_test(data_path: str, max_samples: int | None = None) -> list[EvalSample]:
    """Load PKC test set from JSONL (same format as trajectories)."""
    samples = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            # Support both trajectory format and flat format
            if "messages" in obj:
                # Trajectory format: extract question from user message
                user_msgs = [m for m in obj["messages"] if m["role"] == "user"]
                if user_msgs:
                    user_content = user_msgs[0]["content"]
                    question_match = re.search(r"Question:\s*(.+)", user_content, re.DOTALL)
                    question = question_match.group(1).strip() if question_match else user_content
                    image_match = re.search(r"Image:\s*(\S+)", user_content)
                    image_path = image_match.group(1) if image_match else None
                else:
                    continue
                gold_answer = obj.get("gold_answer", obj.get("answer", ""))
            else:
                question = obj.get("question", "")
                image_path = obj.get("image_path", obj.get("image", None))
                gold_answer = obj.get("answer", obj.get("gold_answer", ""))

            if isinstance(gold_answer, str):
                gold_answer = [gold_answer]

            samples.append(EvalSample(
                sample_id=obj.get("question_id", obj.get("id", str(len(samples)))),
                question=question,
                image_path=image_path,
                golden_answers=gold_answer,
                metadata={
                    "source": "pkc-test",
                    "gold_entities": obj.get("gold_entities", []),
                },
            ))
            if max_samples and len(samples) >= max_samples:
                break
    logger.info(f"Loaded {len(samples)} PKC-test samples from {data_path}")
    return samples


BENCHMARK_LOADERS = {
    "simplevqa": load_simplevqa,
    "infoseek": load_infoseek,
    "fvqa": load_fvqa,
    "pkc-test": load_pkc_test,
}


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Runner
# ═══════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    sample_id: str
    question: str
    golden_answers: list[str]
    prediction: str | None
    exact_match: bool
    substring_match: bool
    num_turns: int
    num_search_calls: int
    num_read_calls: int
    termination: str = ""
    error: str = ""


def run_evaluation(
    client: VLLMClient,
    search_backend: SearchBackend,
    samples: list[EvalSample],
    max_turns: int = 10,
    output_path: str | None = None,
) -> dict:
    """Run evaluation on a list of samples and compute metrics."""
    results: list[EvalResult] = []
    total_em = 0
    total_sub = 0
    total_answered = 0

    for idx, sample in enumerate(samples):
        logger.info(
            f"[{idx + 1}/{len(samples)}] Evaluating: {sample.sample_id} "
            f"| Q: {sample.question[:80]}..."
        )

        try:
            episode = run_agent_episode(
                client=client,
                search_backend=search_backend,
                question=sample.question,
                image_path=sample.image_path,
                max_turns=max_turns,
            )
            prediction = episode["final_answer"]
            error = ""
        except Exception as exc:
            logger.error(f"Error on sample {sample.sample_id}: {exc}")
            prediction = None
            error = str(exc)
            episode = {"num_search_calls": 0, "num_read_calls": 0, "total_turns": 0}

        is_em = False
        is_sub = False
        if prediction:
            total_answered += 1
            is_em = exact_match(prediction, sample.golden_answers)
            is_sub = substring_match(prediction, sample.golden_answers)
            if is_em:
                total_em += 1
            if is_sub:
                total_sub += 1

        result = EvalResult(
            sample_id=sample.sample_id,
            question=sample.question,
            golden_answers=sample.golden_answers,
            prediction=prediction,
            exact_match=is_em,
            substring_match=is_sub,
            num_turns=episode["total_turns"],
            num_search_calls=episode["num_search_calls"],
            num_read_calls=episode["num_read_calls"],
            termination=episode.get("termination", ""),
            error=error,
        )
        results.append(result)

        # Log progress
        progress_em = total_em / (idx + 1) * 100
        progress_sub = total_sub / (idx + 1) * 100
        logger.info(
            f"  → Pred: {prediction or 'None'} | Gold: {sample.golden_answers[0]} "
            f"| EM: {'✓' if is_em else '✗'} | Running EM: {progress_em:.1f}%"
        )

        # Save intermediate results
        if output_path and (idx + 1) % 10 == 0:
            _save_results(results, output_path, len(samples))

    # Final metrics
    total = len(samples)
    metrics = {
        "total_samples": total,
        "total_answered": total_answered,
        "exact_match_accuracy": total_em / total * 100 if total else 0,
        "substring_match_accuracy": total_sub / total * 100 if total else 0,
        "answer_rate": total_answered / total * 100 if total else 0,
        "avg_turns": sum(r.num_turns for r in results) / total if total else 0,
        "avg_search_calls": sum(r.num_search_calls for r in results) / total if total else 0,
        "avg_read_calls": sum(r.num_read_calls for r in results) / total if total else 0,
    }

    output = {
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }

    if output_path:
        _save_results(results, output_path, total, metrics)

    return output


def _save_results(
    results: list[EvalResult],
    output_path: str,
    total: int,
    metrics: dict | None = None,
):
    """Save evaluation results to JSON file."""
    if metrics is None:
        answered = sum(1 for r in results if r.prediction)
        em = sum(1 for r in results if r.exact_match)
        sub = sum(1 for r in results if r.substring_match)
        n = len(results)
        metrics = {
            "total_samples": total,
            "evaluated_so_far": n,
            "exact_match_accuracy": em / n * 100 if n else 0,
            "substring_match_accuracy": sub / n * 100 if n else 0,
            "answer_rate": answered / n * 100 if n else 0,
        }

    output = {
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


# ═══════════════════════════════════════════════════════════
# 6. vLLM Server Launcher
# ═══════════════════════════════════════════════════════════

def start_vllm_server(
    model_path: str,
    port: int = 8000,
    tensor_parallel: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 16384,
) -> None:
    """Start a vLLM server as a subprocess (blocking until ready)."""
    import subprocess
    import time as _time

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--enable-auto-tool-choice",
    ]

    logger.info(f"Starting vLLM server: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for server to be ready
    api_base = f"http://localhost:{port}/v1"
    for attempt in range(120):
        try:
            resp = requests.get(f"{api_base}/models", timeout=2)
            if resp.status_code == 200:
                logger.info(f"vLLM server ready at {api_base}")
                return
        except requests.ConnectionError:
            pass
        _time.sleep(2)

    raise RuntimeError("vLLM server failed to start within 240 seconds")


# ═══════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Turn Agentic Evaluation for SearchEyes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model / API
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-path", type=str, default=None,
        help="Path to local model checkpoint (will auto-start vLLM server)",
    )
    model_group.add_argument(
        "--api-base", type=str, default="http://localhost:8000/v1",
        help="vLLM API base URL (if server already running)",
    )
    model_group.add_argument(
        "--model-name", type=str, default="default",
        help="Model name for vLLM API",
    )
    model_group.add_argument(
        "--api-key", type=str, default=None,
        help="API key for remote OpenAI-compatible endpoints (e.g. Routify)",
    )
    model_group.add_argument(
        "--tensor-parallel", type=int, default=1,
        help="Tensor parallel size for vLLM",
    )

    # Benchmark
    bench_group = parser.add_argument_group("Benchmark")
    bench_group.add_argument(
        "--benchmark", type=str, required=True,
        choices=list(BENCHMARK_LOADERS.keys()),
        help="Benchmark to evaluate on",
    )
    bench_group.add_argument(
        "--data-path", type=str, default=None,
        help="Path to benchmark data file (auto-detected if not set)",
    )
    bench_group.add_argument(
        "--max-samples", type=int, default=None,
        help="Max number of samples to evaluate",
    )
    bench_group.add_argument(
        "--start-index", type=int, default=0,
        help="Start from this sample index (for parallel sharding)",
    )

    # Search backend
    search_group = parser.add_argument_group("Search Backend")
    search_group.add_argument(
        "--search-backend", type=str, default="local",
        choices=["local", "serper", "rag-engine"],
        help="Search backend: 'local' (token-overlap KB), 'serper' (web), or 'rag-engine' (embedding+BM25 hybrid, same as training)",
    )
    search_group.add_argument(
        "--kb-path", type=str, default="/tmp/pgkc_full_kb.json",
        help="Path to local KB JSON file (for local backend)",
    )
    search_group.add_argument(
        "--serper-api-key", type=str, default=None,
        help="Serper API key (or set SERPER_API_KEY env var)",
    )

    # Agent config
    agent_group = parser.add_argument_group("Agent Configuration")
    agent_group.add_argument(
        "--max-turns", type=int, default=50,
        help="Maximum turns per episode (VisionDR default: 50)",
    )
    agent_group.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Max tokens per model generation (VisionDR default: 8192)",
    )
    agent_group.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature (VisionDR default: 0.6)",
    )
    agent_group.add_argument(
        "--top-p", type=float, default=0.95,
        help="Top-p sampling (VisionDR default: 0.95)",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (auto-generated if not set)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Search backend ──
    if args.search_backend == "local":
        search_backend = LocalKBBackend(args.kb_path)
    elif args.search_backend == "rag-engine":
        search_backend = RagEngineBackend()
    elif args.search_backend == "serper":
        api_key = args.serper_api_key or os.environ.get("SERPER_API_KEY", "")
        if not api_key:
            logger.error(
                "Serper API key required. Set --serper-api-key or SERPER_API_KEY env var.\n"
                "Get one at https://serper.dev (free tier: 2500 queries)"
            )
            sys.exit(1)
        search_backend = SerperBackend(api_key)
    else:
        raise ValueError(f"Unknown search backend: {args.search_backend}")

    # ── vLLM client ──
    if args.model_path:
        # Auto-start vLLM server
        start_vllm_server(
            model_path=args.model_path,
            tensor_parallel=args.tensor_parallel,
        )
        model_name = args.model_path
    else:
        model_name = args.model_name

    client = VLLMClient(
        api_base=args.api_base,
        model_name=model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        api_key=args.api_key,
    )

    # ── Load benchmark data ──
    loader = BENCHMARK_LOADERS[args.benchmark]
    if args.data_path:
        data_path = args.data_path
    else:
        # Default paths
        default_paths = {
            "simplevqa": "data/simplevqa/test.jsonl",
            "infoseek": "data/infoseek/test.jsonl",
            "fvqa": "data/fvqa/test.jsonl",
            "pkc-test": "/tmp/sft_output/trajectories_correct.jsonl",
        }
        data_path = default_paths.get(args.benchmark, "")
        if not Path(data_path).exists():
            logger.error(
                f"Data file not found: {data_path}\n"
                f"Please specify --data-path or download the dataset first."
            )
            sys.exit(1)

    samples = loader(data_path, max_samples=None)  # load all first for sharding

    # Apply start-index sharding for parallel evaluation
    if args.start_index > 0:
        samples = samples[args.start_index:]
        logger.info(f"Sharding: starting from index {args.start_index}, {len(samples)} samples remaining")
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    # ── Output path ──
    output_path = args.output
    if not output_path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/eval_multiturn")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(
            output_dir / f"{args.benchmark}_{args.search_backend}_{timestamp}.json"
        )

    # ── Run evaluation ──
    logger.info(f"=" * 60)
    logger.info(f"Multi-Turn Evaluation")
    logger.info(f"  Benchmark:      {args.benchmark}")
    logger.info(f"  Search backend: {args.search_backend}")
    logger.info(f"  Samples:        {len(samples)}")
    logger.info(f"  Max turns:      {args.max_turns}")
    logger.info(f"  Output:         {output_path}")
    logger.info(f"=" * 60)

    output = run_evaluation(
        client=client,
        search_backend=search_backend,
        samples=samples,
        max_turns=args.max_turns,
        output_path=output_path,
    )

    # ── Print summary ──
    metrics = output["metrics"]
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total samples:       {metrics['total_samples']}")
    print(f"  Answered:            {metrics['total_answered']} ({metrics['answer_rate']:.1f}%)")
    print(f"  Exact Match:         {metrics['exact_match_accuracy']:.2f}%")
    print(f"  Substring Match:     {metrics['substring_match_accuracy']:.2f}%")
    print(f"  Avg turns/episode:   {metrics['avg_turns']:.1f}")
    print(f"  Avg search calls:    {metrics['avg_search_calls']:.1f}")
    print(f"  Avg read calls:      {metrics['avg_read_calls']:.1f}")
    print(f"  Results saved to:    {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
