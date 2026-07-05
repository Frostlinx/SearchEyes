"""
sft_synthesis.py — SFT Trajectory Synthesis Pipeline
=====================================================
融合 OpenResearcher + OpenSeeker + SWE-Bench++ 的思路：

Core Design:
  1. Corpus-Level Hint (OpenResearcher):
     - 从 PGKC knowledge graph 提取 gold entities
     - 在检索环境中 boost gold pages 的分数
     - Agent trajectory 完全自然——它只是"运气好"搜到了
     
  2. Observation Denoising (OpenSeeker):
     - 生成时: LLM 对 search results 做摘要 → teacher 看到 clean observation
     - 训练时: student SFT 数据保留 raw observation → 迫使学会去噪
     
  3. Rejection Sampling:
     - Multi-sampling (N=16-32 per question)
     - 只保留最终答案正确的 trajectory
     
  4. Clean Export:
     - 训练数据 = clean system prompt + raw observations + correct actions
     - 没有任何 hint/denoising 痕迹

Pipeline Steps:
  A. build_hint_environment(sample) → HintedRagEngine
  B. generate_trajectory(sample, hint_engine, denoising=True) → Trajectory
  C. verify_answer(trajectory, gold_answer) → bool
  D. export_clean_sft_data(trajectory) → SFTSample (with raw observations)
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. Data Structures
# ═══════════════════════════════════════════════════════════

@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    thought: str
    action: str            # tool call (function name + args)
    raw_observation: str   # 原始搜索结果（用于 SFT 数据）
    denoised_observation: str = ""  # 去噪后的摘要（仅 teacher 生成时可见）
    step_index: int = 0

@dataclass
class Trajectory:
    """Complete agent trajectory for one question."""
    question_id: str
    question: str
    image_path: str
    gold_answer: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str = ""
    is_correct: bool = False
    total_tokens: int = 0
    generation_time: float = 0.0
    sampling_index: int = 0  # which sample (for multi-sampling)

@dataclass
class SFTSample:
    """Clean SFT training sample (no hints, raw observations)."""
    question_id: str
    messages: list[dict]  # OpenAI-format messages
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
# 2. Hinted RAG Environment
# ═══════════════════════════════════════════════════════════

class HintedRagEngine:
    """Wraps the real RagEngine with corpus-level hints.
    
    Strategy: Boost gold entity scores by a multiplier so they
    rank higher in search results. The agent doesn't know about
    the boost — it just happens to find relevant results more easily.
    
    This is analogous to OpenResearcher's "gold documents mixed into corpus"
    approach, but more fine-grained (score boosting vs. inclusion).
    """

    def __init__(
        self,
        base_engine,  # RagEngine instance
        gold_entity_ids: set[str],
        boost_factor: float = 2.0,
        reduced_corpus_mode: bool = False,
        reduced_corpus_size: int = 500,
    ):
        """
        Args:
            base_engine: The real RagEngine (fully loaded)
            gold_entity_ids: Set of wikidata_ids that are on the knowledge path
            boost_factor: Score multiplier for gold entities in search results
            reduced_corpus_mode: If True, restrict search to a small subset
            reduced_corpus_size: Number of distractor entities to include
        """
        self.base_engine = base_engine
        self.gold_entity_ids = gold_entity_ids
        self.boost_factor = boost_factor
        self.reduced_corpus_mode = reduced_corpus_mode
        self.reduced_corpus_size = reduced_corpus_size
        
        # For reduced corpus mode, precompute allowed indices
        self._allowed_indices: Optional[set[int]] = None
        if reduced_corpus_mode:
            self._build_reduced_corpus()

    def _build_reduced_corpus(self):
        """Build a reduced corpus: gold entities + random distractors."""
        engine = self.base_engine
        gold_indices = set()
        for qid in self.gold_entity_ids:
            idx = engine._id_to_idx.get(qid)
            if idx is not None:
                gold_indices.add(idx)
        
        # Add random distractors
        all_indices = set(range(len(engine._entities)))
        non_gold = list(all_indices - gold_indices)
        np.random.shuffle(non_gold)
        distractor_count = min(self.reduced_corpus_size, len(non_gold))
        distractor_indices = set(non_gold[:distractor_count])
        
        self._allowed_indices = gold_indices | distractor_indices
        logger.info(
            f"Reduced corpus: {len(gold_indices)} gold + "
            f"{distractor_count} distractors = {len(self._allowed_indices)} total"
        )

    def search(
        self,
        text: str = "",
        image_path: str = "",
        top_k: int = 10,
    ) -> list[dict]:
        """Search with corpus-level hints (boost gold entities).
        
        Returns results in the same format as vdr_tools expects.
        """
        # Get raw results from base engine (request more to account for filtering)
        request_k = top_k * 3 if self.reduced_corpus_mode else top_k * 2
        raw_facts = self.base_engine.get_rag_facts_combined(
            text=text, image_path=image_path, top_k=request_k, use_hybrid=True
        )
        
        results = []
        for fact in raw_facts:
            # Reduced corpus filtering
            if self.reduced_corpus_mode and self._allowed_indices is not None:
                idx = self.base_engine._id_to_idx.get(fact.wit_id)
                if idx is not None and idx not in self._allowed_indices:
                    continue
            
            # Score boosting for gold entities
            score = fact.score
            if fact.wit_id in self.gold_entity_ids:
                score *= self.boost_factor
            
            results.append({
                "entity_id": fact.wit_id,
                "title": fact.title,
                "summary": fact.caption,
                "score": score,
                "image_url": fact.image_url,
            })
        
        # Re-sort by boosted score and truncate
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_entity_content(self, entity_id: str) -> Optional[str]:
        """Get full content of an entity (for read_full tool)."""
        fact = self.base_engine.get_fact_by_id(entity_id)
        if fact is None:
            return None
        # Try to get full content from _contents array
        id_to_idx = getattr(self.base_engine, '_id_to_idx', None)
        contents = getattr(self.base_engine, '_contents', None)
        if id_to_idx and contents:
            idx = id_to_idx.get(entity_id)
            if idx is not None and idx < len(contents):
                return contents[idx]
        # Fallback: use content attribute from fact object
        return getattr(fact, 'content', None) or getattr(fact, 'caption', '')


# ═══════════════════════════════════════════════════════════
# 3. Observation Denoiser (OpenSeeker-style)
# ═══════════════════════════════════════════════════════════

class ObservationDenoiser:
    """Summarizes raw search observations for the teacher model.
    
    During generation: teacher sees denoised (summarized) observations
    → makes better decisions.
    
    During training: student learns from raw observations
    → forced to develop its own denoising ability.
    
    This is the key insight from OpenSeeker's "retrospective summarization".
    """

    SUMMARIZE_PROMPT = """You are a research assistant. Given a search query and its raw results, 
provide a concise, factual summary of the most relevant information found.

Search Query: {query}

Raw Search Results:
{raw_results}

Instructions:
- Extract only the most relevant facts that could help answer the research question
- Remove irrelevant noise, ads, boilerplate text
- Keep entity names, dates, numbers, and relationships accurate
- Be concise (max 200 words)
- If no relevant information found, say "No relevant results found."

Summary:"""

    def __init__(self, llm_client, max_summary_length: int = 500):
        """
        Args:
            llm_client: LLM client for summarization (can be same as teacher)
            max_summary_length: Max chars for summary
        """
        self.llm_client = llm_client
        self.max_summary_length = max_summary_length

    async def denoise(self, query: str, raw_observation: str) -> str:
        """Summarize a raw observation for cleaner teacher reasoning.
        
        Args:
            query: The search query that produced this observation
            raw_observation: Raw search results text
            
        Returns:
            Concise summary of relevant information
        """
        if len(raw_observation) < 200:
            # Already short enough, no need to summarize
            return raw_observation

        prompt = self.SUMMARIZE_PROMPT.format(
            query=query,
            raw_results=raw_observation[:3000],  # Truncate very long results
        )

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=256,
                temperature=0.3,
            )
            summary = response.strip()
            return summary[:self.max_summary_length]
        except Exception as exc:
            logger.warning(f"Denoising failed: {exc}, using truncated raw")
            return raw_observation[:self.max_summary_length]


# ═══════════════════════════════════════════════════════════
# 4. Trajectory Generator
# ═══════════════════════════════════════════════════════════

# System prompt for the teacher (same as normal agent — NO hints here)
TEACHER_SYSTEM_PROMPT = """You are a visual research assistant. Given an image and a complex multi-hop question, you must find the answer by searching through a knowledge base step by step.

You have access to these tools. Call exactly ONE tool per turn using the exact format shown:

1. search(query="your search query")
   Searches the knowledge base by text. Returns titles and summaries.

2. crop_and_search(image_path="path", bbox=[x1,y1,x2,y2])
   Crops image region and searches by visual similarity.

3. read_entity(entity_id="Q12345")
   Reads the full Wikipedia article for an entity.

4. answer(text="your final answer")
   Provides your final answer. Use this only when you are confident.

FORMAT: Each turn you must output:
Thought: <your reasoning about what to do next>
Action: <exactly one tool call, e.g. search(query="Barack Obama")>

Do NOT use JSON format. Do NOT use code blocks. Just plain text with Thought and Action.

Strategy:
1. First identify who/what is in the image
2. Follow the chain of relationships in the question step by step
3. Search for each entity along the path — do NOT skip steps or guess
4. You MUST call search() or read_entity() to verify each connection before moving on
5. Answer ONLY when you have concrete evidence for EVERY hop in the chain
6. NEVER answer with vague phrases like "one of the..." — give a specific entity name"""


class TrajectoryGenerator:
    """Generates trajectories using teacher model with hinted environment.
    
    Combines:
    - HintedRagEngine for corpus-level hints
    - ObservationDenoiser for cleaner teacher observations
    - Multi-sampling for rejection sampling
    """

    def __init__(
        self,
        llm_client,
        hinted_engine,
        denoiser: Optional[ObservationDenoiser] = None,
        max_steps: int = 50,
        temperature: float = 0.7,
        num_samples: int = 16,
        max_response_tokens: int = 4096,
        max_observation_chars: int = 1024,
        context_budget_tokens: int = 28000,
        enable_sliding_window: bool = True,
    ):
        """
        Args:
            llm_client: Teacher LLM client (Qwen3.6-27B)
            hinted_engine: Hinted retrieval environment (HintedRagEngine or HintedMiniRagEngine)
            denoiser: Optional observation denoiser
            max_steps: Maximum steps per trajectory (VDR uses 50)
            temperature: Sampling temperature
            num_samples: Number of samples per question (for rejection sampling)
            max_response_tokens: Max tokens per LLM response (VDR uses 4096)
            max_observation_chars: Max chars for each observation (truncated)
            context_budget_tokens: Approximate token budget before sliding window kicks in
            enable_sliding_window: Whether to drop early turns when context exceeds budget
        """
        self.llm_client = llm_client
        self.hinted_engine = hinted_engine
        self.denoiser = denoiser
        self.max_steps = max_steps
        self.temperature = temperature
        self.num_samples = num_samples
        self.max_response_tokens = max_response_tokens
        self.max_observation_chars = max_observation_chars
        self.context_budget_tokens = context_budget_tokens
        self.enable_sliding_window = enable_sliding_window

    async def generate_single(
        self,
        question: str,
        image_path: str,
        question_id: str,
        gold_answer: str,
        sampling_index: int = 0,
    ) -> Trajectory:
        """Generate one trajectory for a question.
        
        The teacher model interacts with the hinted environment,
        optionally seeing denoised observations, but the raw observations
        are always recorded for the final SFT data.
        """
        trajectory = Trajectory(
            question_id=question_id,
            question=question,
            image_path=image_path,
            gold_answer=gold_answer,
            sampling_index=sampling_index,
        )
        
        start_time = time.time()
        
        # Build conversation history (what teacher sees)
        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Image: {image_path}\n\nQuestion: {question}"},
        ]
        
        for step_idx in range(self.max_steps):
            # Sliding window: trim early turns if context too long
            if self.enable_sliding_window:
                messages = self._apply_sliding_window(messages)

            # Generate thought + action
            response = await self.llm_client.generate_chat(
                messages=messages,
                max_tokens=self.max_response_tokens,
                temperature=self.temperature,
                stop=["Observation:"],
            )
            
            # Parse thought and action from response
            thought, action_call = self._parse_response(response)
            
            if not action_call:
                # Model wants to answer or got confused
                final_ans = self._extract_answer(response)
                if final_ans:
                    trajectory.final_answer = final_ans
                break
            
            # If it's an answer action, extract and stop
            if action_call.get("tool") == "answer":
                trajectory.final_answer = action_call.get("args", {}).get("text", "")
                break
            
            # Execute action in hinted environment
            raw_observation = await self._execute_action(action_call, image_path)

            # Truncate observation to budget (VDR-style: keep it concise)
            if len(raw_observation) > self.max_observation_chars:
                raw_observation = raw_observation[:self.max_observation_chars] + "\n[...truncated]"

            # Denoise for teacher (if denoiser available)
            denoised_obs = raw_observation
            if self.denoiser and action_call.get("tool") in ("search", "crop_and_search"):
                query_text = action_call.get("args", {}).get("query", "")
                denoised_obs = await self.denoiser.denoise(query_text, raw_observation)

            # Record step (both raw and denoised)
            step = TrajectoryStep(
                thought=thought,
                action=json.dumps(action_call),
                raw_observation=raw_observation,
                denoised_observation=denoised_obs,
                step_index=step_idx,
            )
            trajectory.steps.append(step)

            # Teacher sees denoised observation in its context
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation:\n{denoised_obs}"})
        
        # If no explicit answer, try to extract from last response
        if not trajectory.final_answer:
            trajectory.final_answer = self._extract_answer(response)
        
        trajectory.generation_time = time.time() - start_time
        trajectory.is_correct = self._verify_answer(
            trajectory.final_answer, gold_answer
        )
        
        return trajectory

    async def generate_multi_sample(
        self,
        question: str,
        image_path: str,
        question_id: str,
        gold_answer: str,
    ) -> list[Trajectory]:
        """Generate multiple trajectory samples for rejection sampling."""
        tasks = [
            self.generate_single(
                question=question,
                image_path=image_path,
                question_id=question_id,
                gold_answer=gold_answer,
                sampling_index=i,
            )
            for i in range(self.num_samples)
        ]
        
        # Run concurrently (but respect rate limits)
        results = []
        batch_size = 4  # Concurrent batch size
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start:batch_start + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, Trajectory):
                    results.append(result)
                else:
                    logger.warning(f"Trajectory generation failed: {result}")
        
        return results

    def _apply_sliding_window(self, messages: list[dict]) -> list[dict]:
        """Drop early turns when context exceeds token budget (DeepMiner-style).

        Keeps: system prompt + first user message + most recent N turns.
        Approximation: 1 token ≈ 4 chars.
        """
        total_chars = sum(len(m.get("content", "")) for m in messages)
        approx_tokens = total_chars // 4

        if approx_tokens <= self.context_budget_tokens:
            return messages

        # Keep system (idx 0) + first user (idx 1) + drop oldest assistant/user pairs
        preserved_head = messages[:2]
        tail = messages[2:]

        # Remove pairs from the front of tail until under budget
        while len(tail) >= 2:
            tail_chars = sum(len(m.get("content", "")) for m in tail)
            head_chars = sum(len(m.get("content", "")) for m in preserved_head)
            if (head_chars + tail_chars) // 4 <= self.context_budget_tokens:
                break
            # Remove oldest assistant + user pair
            tail = tail[2:]

        return preserved_head + tail

    async def _execute_action(self, action_call: dict, image_path: str) -> str:
        """Execute a tool call in the hinted environment."""
        tool_name = action_call.get("tool", "")
        args = action_call.get("args", {})
        
        if tool_name == "search":
            query = args.get("query", "")
            results = self.hinted_engine.search(text=query, top_k=10)
            return self._format_search_results(results)
        
        elif tool_name == "crop_and_search":
            # For crop_and_search, use image path
            img = args.get("image_path", image_path)
            results = self.hinted_engine.search(image_path=img, top_k=10)
            return self._format_search_results(results)
        
        elif tool_name == "read_entity":
            entity_id = args.get("entity_id", "")
            content = self.hinted_engine.get_entity_content(entity_id)
            if content:
                # Return structured content with section headers preserved
                # Limit to 4000 chars but try to include section boundaries
                if len(content) <= 4000:
                    return content
                # Include first 2000 + last 2000 to capture both intro and later sections
                return content[:2000] + "\n\n[...content truncated...]\n\n" + content[-2000:]
            return "Entity not found."
        
        elif tool_name == "answer":
            return ""  # Answer doesn't produce observation
        
        else:
            return f"Unknown tool: {tool_name}"

    def _format_search_results(self, results: list[dict]) -> str:
        """Format search results as text observation."""
        if not results:
            return "No results found."
        
        lines = []
        for i, r in enumerate(results[:10], 1):
            lines.append(
                f"[{i}] {r['title']} (ID: {r['entity_id']})\n"
                f"    {r['summary'][:200]}"
            )
        return "\n\n".join(lines)

    def _parse_response(self, response: str) -> tuple[str, Optional[dict]]:
        """Parse thought and tool call from model response.
        
        Handles multiple output formats:
        1. "Thought: ...\nAction: tool_name(...)"
        2. "Some reasoning text\ntool_name(...)"  (model's natural format)
        3. "```python\ntool_name(...)\n```"  (code block format)
        4. JSON tool call format
        """
        thought = ""
        action_call = None
        
        # Pre-clean: remove </think> tags and code block markers
        cleaned = re.sub(r'</think>', '', response)
        cleaned = re.sub(r'```(?:python|json)?\s*', '', cleaned)
        cleaned = re.sub(r'```', '', cleaned)
        cleaned = cleaned.strip()

        # Format 5: Qwen3 tool calling format "call:default_api:tool_name{...json...}"
        qwen_tool_pattern = re.compile(
            r'call:(?:\w+:)?(search|crop_and_search|read_entity|answer)\s*(\{.*?\})',
            re.DOTALL | re.IGNORECASE,
        )
        qwen_match = qwen_tool_pattern.search(cleaned)
        if qwen_match:
            tool_name = qwen_match.group(1).lower()
            try:
                args = json.loads(qwen_match.group(2))
            except json.JSONDecodeError:
                args = {}
            thought = cleaned[: qwen_match.start()].strip()
            for prefix in ("Thought:", "Think:", "Reasoning:"):
                if thought.lower().startswith(prefix.lower()):
                    thought = thought[len(prefix):].strip()
            if tool_name == "answer":
                ans_text = args.get("text", args.get("answer", ""))
                if ans_text:
                    return thought, {"tool": "answer", "args": {"text": ans_text}}
                return thought, None
            return thought, {"tool": tool_name, "args": args}

        # Try to find a tool call anywhere in the cleaned response
        # Matches: search(...), crop_and_search(...), read_entity(...), answer(...)
        tool_pattern = re.compile(
            r'\b(search|crop_and_search|read_entity|answer)\s*\((.*?)\)',
            re.DOTALL | re.IGNORECASE
        )
        tool_match = tool_pattern.search(cleaned)
        
        if tool_match:
            tool_name = tool_match.group(1).lower()
            args_str = tool_match.group(2)
            
            # Everything before the tool call is the thought
            thought = cleaned[:tool_match.start()].strip()
            # Clean up thought prefixes
            for prefix in ("Thought:", "Think:", "Reasoning:", "Action:"):
                if thought.lower().startswith(prefix.lower()):
                    thought = thought[len(prefix):].strip()
            
            if tool_name == "answer":
                # Extract answer text from answer(text="...") or answer("...")
                ans_match = re.search(
                    r'(?:text\s*=\s*)?["\'](.+?)["\']', args_str, re.DOTALL
                )
                if ans_match:
                    return thought, {"tool": "answer", "args": {"text": ans_match.group(1)}}
                # Try unquoted: answer(some text)
                if args_str.strip():
                    return thought, {"tool": "answer", "args": {"text": args_str.strip()}}
                return thought, None
            
            # Parse keyword arguments: key="value" or key='value'
            args = {}
            for kwarg_match in re.finditer(
                r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|\[([^\]]*?)\])',
                args_str
            ):
                key = kwarg_match.group(1)
                value = (
                    kwarg_match.group(2) if kwarg_match.group(2) is not None
                    else kwarg_match.group(3) if kwarg_match.group(3) is not None
                    else kwarg_match.group(4) or ""
                )
                args[key] = value
            
            # If no kwargs found, try positional: search("query text")
            if not args:
                pos_match = re.search(r'["\'](.+?)["\']', args_str, re.DOTALL)
                if pos_match:
                    if tool_name in ("search", "crop_and_search"):
                        args["query"] = pos_match.group(1)
                    elif tool_name == "read_entity":
                        args["entity_id"] = pos_match.group(1)
            
            action_call = {"tool": tool_name, "args": args}
        else:
            # No tool call found — entire response is thought
            thought = cleaned
            for prefix in ("Thought:", "Think:", "Reasoning:"):
                if thought.lower().startswith(prefix.lower()):
                    thought = thought[len(prefix):].strip()
        
        return thought, action_call

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response text."""
        patterns = [
            r'answer\s*\(\s*(?:text\s*=\s*)?["\'](.+?)["\']\s*\)',
            r'(?:Final\s+)?[Aa]nswer:\s*(.+?)(?:\n|$)',
            r'\*\*Answer\*\*:\s*(.+?)(?:\n|$)',
            r'(?:The answer is|the answer is)\s+(.+?)(?:\.|$)',
            r'(?:Therefore|So|Thus),?\s+(?:the answer is\s+)?(.+?)(?:\.|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"').strip("'")
        return ""

    @staticmethod
    def _verify_answer(predicted: str, gold: str) -> bool:
        """Verify if predicted answer matches gold answer.
        
        Uses normalized string matching (case-insensitive, stripped).
        """
        if not predicted or not gold:
            return False
        
        pred_norm = predicted.lower().strip().rstrip(".")
        gold_norm = gold.lower().strip().rstrip(".")
        
        # Exact match
        if pred_norm == gold_norm:
            return True
        
        # Containment (either direction)
        if gold_norm in pred_norm or pred_norm in gold_norm:
            return True
        
        # Token overlap (Jaccard > 0.7)
        pred_tokens = set(pred_norm.split())
        gold_tokens = set(gold_norm.split())
        if gold_tokens and pred_tokens:
            jaccard = len(pred_tokens & gold_tokens) / len(pred_tokens | gold_tokens)
            if jaccard > 0.7:
                return True
        
        return False


# ═══════════════════════════════════════════════════════════
# 5. Clean SFT Data Exporter
# ═══════════════════════════════════════════════════════════

class SFTExporter:
    """Export correct trajectories as clean SFT data.
    
    Key principle: The exported data contains:
    - Clean system prompt (NO hints, NO denoising instructions)
    - Raw observations (what the student will actually see at inference)
    - Teacher's correct actions (learned from hinted environment)
    
    This is the "decoupling" trick from OpenSeeker: 
    train on raw data, but the actions were generated under easier conditions.
    """

    CLEAN_SYSTEM_PROMPT = """You are a visual research assistant. Given an image and a complex multi-hop question, 
you must find the answer by searching through a knowledge base step by step.

Available tools:
- search(query: str) → Search the knowledge base with a text query. Returns top results with titles and summaries.
- crop_and_search(image_path: str, bbox: list[int]) → Crop a region from the image and search by visual similarity.
- read_entity(entity_id: str) → Read the full Wikipedia content of an entity.
- answer(text: str) → Provide your final answer.

Strategy:
1. First, identify what/who is in the image using crop_and_search or search
2. Follow the chain of relationships described in the question
3. At each hop, search for the relevant entity and verify the connection
4. Only answer when you have sufficient evidence

Think step by step. At each step, explain your reasoning, then call a tool."""

    def export_trajectory(self, trajectory: Trajectory) -> Optional[SFTSample]:
        """Convert a correct trajectory to clean SFT format.
        
        Only exports trajectories where is_correct=True.
        Uses RAW observations (not denoised) in the training data.
        """
        if not trajectory.is_correct:
            return None
        
        messages = [
            {"role": "system", "content": self.CLEAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Image: {trajectory.image_path}\n\nQuestion: {trajectory.question}",
            },
        ]
        
        for step in trajectory.steps:
            # Assistant turn: thought + action
            assistant_content = f"Thought: {step.thought}\nAction: {step.action}"
            messages.append({"role": "assistant", "content": assistant_content})
            
            # User turn: RAW observation (NOT denoised!)
            # This is the key: student learns to handle noisy observations
            messages.append({
                "role": "user",
                "content": f"Observation:\n{step.raw_observation}",
            })
        
        # Final answer
        if trajectory.final_answer:
            messages.append({
                "role": "assistant",
                "content": f"Thought: Based on my research, I have found the answer.\nAction: answer(text=\"{trajectory.final_answer}\")",
            })
        
        return SFTSample(
            question_id=trajectory.question_id,
            messages=messages,
            metadata={
                "num_steps": len(trajectory.steps),
                "generation_time": trajectory.generation_time,
                "sampling_index": trajectory.sampling_index,
                "gold_answer": trajectory.gold_answer,
            },
        )

    def export_batch(
        self,
        trajectories: list[Trajectory],
        output_path: str | Path,
        best_only: bool = True,
    ) -> int:
        """Export a batch of trajectories, applying rejection sampling.
        
        Args:
            trajectories: All generated trajectories (multiple samples per question)
            output_path: Path to write JSONL output
            best_only: If True, keep only the shortest correct trajectory per question
            
        Returns:
            Number of exported samples
        """
        # Group by question_id
        by_question: dict[str, list[Trajectory]] = {}
        for traj in trajectories:
            if traj.is_correct:
                by_question.setdefault(traj.question_id, []).append(traj)
        
        samples = []
        for question_id, correct_trajs in by_question.items():
            if best_only:
                # Pick the shortest correct trajectory (fewer steps = more efficient)
                best = min(correct_trajs, key=lambda t: len(t.steps))
                sample = self.export_trajectory(best)
                if sample:
                    samples.append(sample)
            else:
                # Export all correct trajectories
                for traj in correct_trajs:
                    sample = self.export_trajectory(traj)
                    if sample:
                        samples.append(sample)
        
        # Write to JSONL
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "question_id": sample.question_id,
                    "messages": sample.messages,
                    "metadata": sample.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(
            f"Exported {len(samples)} SFT samples to {output_path} "
            f"({len(by_question)} questions answered correctly)"
        )
        return len(samples)


# ═══════════════════════════════════════════════════════════
# 5b. Mini RAG Engine (JSON KB, BM25-style word overlap)
# ═══════════════════════════════════════════════════════════

class MiniRagEngine:
    """Lightweight retrieval engine backed by a JSON KB file.

    Uses BM25-style tf-idf word overlap for search — no embedding model needed.
    Designed for trajectory synthesis where gold entities get score boost anyway.
    """

    def __init__(self, kb_path: str):
        """Load KB from JSON: {entity_id: {title, content, image}}."""
        with open(kb_path) as f:
            self._kb: dict[str, dict] = json.load(f)
        # Build inverted index for fast BM25-style retrieval
        self._inv_index: dict[str, list[tuple[str, float]]] = {}  # token → [(qid, tf)]
        self._doc_lens: dict[str, int] = {}
        self._avg_dl = 0.0
        self._build_index()
        logger.info(f"MiniRagEngine loaded: {len(self._kb)} entities")

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer, lowercased."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def _build_index(self):
        """Build inverted index for BM25 scoring."""
        import math
        doc_freq: dict[str, int] = {}  # token → num docs containing it
        total_len = 0

        for qid, entity in self._kb.items():
            text = (entity.get("title", "") + " " + entity.get("content", ""))
            tokens = self._tokenize(text)
            self._doc_lens[qid] = len(tokens)
            total_len += len(tokens)
            # Term frequency
            tf_map: dict[str, int] = {}
            for tok in tokens:
                tf_map[tok] = tf_map.get(tok, 0) + 1
            # Update inverted index
            for tok, count in tf_map.items():
                if tok not in self._inv_index:
                    self._inv_index[tok] = []
                self._inv_index[tok].append((qid, count))
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        num_docs = len(self._kb)
        self._avg_dl = total_len / max(num_docs, 1)

        # Precompute IDF
        self._idf: dict[str, float] = {}
        for tok, df in doc_freq.items():
            self._idf[tok] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, text: str = "", top_k: int = 10) -> list[dict]:
        """BM25 search over KB. Returns list of {entity_id, title, summary, score, image_url}."""
        query_tokens = self._tokenize(text)
        if not query_tokens:
            return []

        k1, b = 1.5, 0.75
        scores: dict[str, float] = {}
        for tok in query_tokens:
            if tok not in self._inv_index:
                continue
            idf = self._idf.get(tok, 0.0)
            for qid, tf in self._inv_index[tok]:
                dl = self._doc_lens.get(qid, 1)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._avg_dl))
                scores[qid] = scores.get(qid, 0.0) + idf * tf_norm

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]
        results = []
        for qid, score in ranked[:top_k]:
            entity = self._kb[qid]
            results.append({
                "entity_id": qid,
                "title": entity.get("title", qid),
                "summary": entity.get("content", "")[:300],
                "score": score,
                "image_url": entity.get("image", ""),
            })
        return results

    def get_entity_content(self, entity_id: str) -> Optional[str]:
        """Get full content of an entity."""
        entity = self._kb.get(entity_id)
        if entity is None:
            return None
        return entity.get("content", "")


class HintedMiniRagEngine:
    """HintedRagEngine variant backed by MiniRagEngine (no embeddings needed).

    Boosts gold entity scores in search results — same concept as HintedRagEngine
    but works with the lightweight MiniRagEngine.
    """

    def __init__(
        self,
        mini_engine: MiniRagEngine,
        gold_entity_ids: set[str],
        boost_factor: float = 3.0,
    ):
        self.mini_engine = mini_engine
        self.gold_entity_ids = gold_entity_ids
        self.boost_factor = boost_factor

    def search(self, text: str = "", image_path: str = "", top_k: int = 10) -> list[dict]:
        """Search with gold boost. image_path ignored (text-only search)."""
        # Request more results to ensure gold entities make it after re-ranking
        results = self.mini_engine.search(text=text, top_k=top_k * 3)

        # Boost gold entities
        for result in results:
            if result["entity_id"] in self.gold_entity_ids:
                result["score"] *= self.boost_factor

        # Also inject gold entities that might have been missed by text search
        found_ids = {r["entity_id"] for r in results}
        for qid in self.gold_entity_ids:
            if qid not in found_ids:
                entity = self.mini_engine._kb.get(qid)
                if entity:
                    # Give a moderate base score so it appears in results
                    results.append({
                        "entity_id": qid,
                        "title": entity.get("title", qid),
                        "summary": entity.get("content", "")[:300],
                        "score": 0.5 * self.boost_factor,
                        "image_url": entity.get("image", ""),
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_entity_content(self, entity_id: str) -> Optional[str]:
        """Get full content of an entity."""
        return self.mini_engine.get_entity_content(entity_id)


# ═══════════════════════════════════════════════════════════
# 6. LLM Client (vLLM Backend)
# ═══════════════════════════════════════════════════════════

class VLLMClient:
    """Sync/async client for local vLLM server using urllib (no extra deps)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "/dev/shm/Qwen3.6-27B",
        enable_thinking: bool = False,
    ):
        self.base_url = base_url
        self.model = model
        self.enable_thinking = enable_thinking

    def _post(self, endpoint: str, payload: dict, timeout: int = 600) -> dict:
        """Sync HTTP POST to vLLM server."""
        import urllib.request

        url = f"{self.base_url}/{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate text completion (runs sync in executor for async compat)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, prompt, max_tokens, temperature, stop
        )

    def _generate_sync(
        self, prompt: str, max_tokens: int, temperature: float, stop: Optional[list[str]]
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
        }
        data = self._post("completions", payload)
        return data["choices"][0]["text"]

    async def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate chat completion (runs sync in executor for async compat)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_chat_sync, messages, max_tokens, temperature, stop
        )

    def _generate_chat_sync(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]],
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        data = self._post("chat/completions", payload)
        return data["choices"][0]["message"]["content"]


# ═══════════════════════════════════════════════════════════
# 7. Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════

class SFTPipeline:
    """End-to-end SFT trajectory synthesis pipeline.
    
    Usage:
        pipeline = SFTPipeline(rag_engine=engine)
        await pipeline.run(
            pgkc_data_path="/tmp/pgkc_output/pgkc_v3_final.json",
            output_path="/tmp/sft_output/trajectories.jsonl",
        )
    """

    def __init__(
        self,
        rag_engine,
        llm_base_url: str = "http://localhost:8000/v1",
        llm_model: str = "/dev/shm/Qwen3.6-27B",
        boost_factor: float = 3.0,
        use_reduced_corpus: bool = False,
        use_denoising: bool = True,
        num_samples: int = 16,
        max_steps: int = 50,
        temperature: float = 0.7,
        max_response_tokens: int = 4096,
        max_observation_chars: int = 1024,
        enable_sliding_window: bool = True,
    ):
        self.rag_engine = rag_engine
        self.llm_client = VLLMClient(base_url=llm_base_url, model=llm_model)
        self.boost_factor = boost_factor
        self.use_reduced_corpus = use_reduced_corpus
        self.use_denoising = use_denoising
        self.num_samples = num_samples
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_response_tokens = max_response_tokens
        self.max_observation_chars = max_observation_chars
        self.enable_sliding_window = enable_sliding_window
        self.exporter = SFTExporter()

    def _extract_gold_entities(self, sample: dict) -> set[str]:
        """Extract all entity QIDs on the knowledge path for a PGKC sample."""
        gold_ids = set()
        gold_ids.add(sample["anchor_qid"])
        gold_ids.add(sample["answer_qid"])
        
        for hop in sample.get("chain", []):
            gold_ids.add(hop["from_qid"])
            gold_ids.add(hop["to_qid"])
        
        for constraint in sample.get("constraints", []):
            gold_ids.add(constraint["constraint_qid"])
            gold_ids.add(constraint.get("target_qid", ""))
        
        gold_ids.discard("")
        return gold_ids

    def _build_hinted_engine(self, sample: dict) -> HintedRagEngine:
        """Build a hinted RAG engine for a specific sample."""
        gold_ids = self._extract_gold_entities(sample)
        return HintedRagEngine(
            base_engine=self.rag_engine,
            gold_entity_ids=gold_ids,
            boost_factor=self.boost_factor,
            reduced_corpus_mode=self.use_reduced_corpus,
        )

    async def run_single(self, sample: dict) -> list[Trajectory]:
        """Run pipeline for a single PGKC sample."""
        question_id = sample["question_id"]
        question = sample["question"]
        image_path = sample["image_path"]
        gold_answer = sample["answer"]
        
        logger.info(f"Processing {question_id}: {question[:60]}...")
        
        # Build hinted environment
        hinted_engine = self._build_hinted_engine(sample)
        
        # Build denoiser (optional)
        denoiser = None
        if self.use_denoising:
            denoiser = ObservationDenoiser(llm_client=self.llm_client)
        
        # Build trajectory generator
        generator = TrajectoryGenerator(
            llm_client=self.llm_client,
            hinted_engine=hinted_engine,
            denoiser=denoiser,
            max_steps=self.max_steps,
            temperature=self.temperature,
            num_samples=self.num_samples,
            max_response_tokens=self.max_response_tokens,
            max_observation_chars=self.max_observation_chars,
            context_budget_tokens=28000,
            enable_sliding_window=self.enable_sliding_window,
        )
        
        # Generate trajectories (multi-sample)
        trajectories = await generator.generate_multi_sample(
            question=question,
            image_path=image_path,
            question_id=question_id,
            gold_answer=gold_answer,
        )
        
        # Stats
        correct_count = sum(1 for t in trajectories if t.is_correct)
        logger.info(
            f"  {question_id}: {correct_count}/{len(trajectories)} correct "
            f"(pass@{self.num_samples} = {correct_count > 0})"
        )
        
        return trajectories

    async def run(
        self,
        pgkc_data_path: str = "/tmp/pgkc_output/pgkc_v3_final.json",
        output_path: str = "/tmp/sft_output/trajectories.jsonl",
        max_questions: Optional[int] = None,
    ) -> dict:
        """Run full pipeline on PGKC dataset.
        
        Returns:
            Stats dict with pass rates, sample counts, etc.
        """
        # Load PGKC data
        with open(pgkc_data_path) as f:
            samples = json.load(f)
        
        if max_questions:
            samples = samples[:max_questions]
        
        logger.info(f"Starting SFT pipeline: {len(samples)} questions, "
                    f"num_samples={self.num_samples}, boost={self.boost_factor}")
        
        # Process all questions
        all_trajectories = []
        for sample in samples:
            trajectories = await self.run_single(sample)
            all_trajectories.extend(trajectories)
        
        # Export clean SFT data (rejection sampling)
        num_exported = self.exporter.export_batch(
            all_trajectories, output_path, best_only=True
        )
        
        # Compute stats
        total_generated = len(all_trajectories)
        total_correct = sum(1 for t in all_trajectories if t.is_correct)
        questions_solved = len(set(
            t.question_id for t in all_trajectories if t.is_correct
        ))
        
        stats = {
            "total_questions": len(samples),
            "total_trajectories_generated": total_generated,
            "total_correct": total_correct,
            "questions_solved": questions_solved,
            "pass_rate": questions_solved / max(len(samples), 1),
            "sample_accuracy": total_correct / max(total_generated, 1),
            "num_exported_sft_samples": num_exported,
            "output_path": str(output_path),
        }
        
        logger.info(f"Pipeline complete: {json.dumps(stats, indent=2)}")
        
        # Also save stats
        stats_path = Path(output_path).parent / "pipeline_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        return stats


# ═══════════════════════════════════════════════════════════
# 8. CLI Entry Point
# ═══════════════════════════════════════════════════════════

async def main():
    """Main entry point for running the SFT synthesis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Trajectory Synthesis Pipeline")
    parser.add_argument("--pgkc-data", default="/tmp/pgkc_output/pgkc_v3_final.json")
    parser.add_argument("--output", default="/tmp/sft_output/trajectories.jsonl")
    parser.add_argument("--llm-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-model", default="/dev/shm/Qwen3.6-27B")
    parser.add_argument("--boost-factor", type=float, default=2.0)
    parser.add_argument("--reduced-corpus", action="store_true")
    parser.add_argument("--no-denoising", action="store_true")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--wiki6m-path", default="/dev/shm/oven_wiki/Wiki6M_ver_1_0.jsonl")
    parser.add_argument("--embeddings-dir", default="/dev/shm/oven_wiki/embeddings")
    parser.add_argument("--bm25-index", default="/dev/shm/oven_wiki/bm25_index.json")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    
    # Load RAG engine
    from searcheyes.rag_engine import RagEngine
    
    logger.info("Loading RAG engine...")
    engine = RagEngine(
        wiki6m_path=args.wiki6m_path,
        embeddings_dir=args.embeddings_dir,
        bm25_index_path=args.bm25_index,
    )
    engine.load()
    
    # Run pipeline
    pipeline = SFTPipeline(
        rag_engine=engine,
        llm_base_url=args.llm_url,
        llm_model=args.llm_model,
        boost_factor=args.boost_factor,
        use_reduced_corpus=args.reduced_corpus,
        use_denoising=not args.no_denoising,
        num_samples=args.num_samples,
        max_steps=args.max_steps,
        temperature=args.temperature,
    )
    
    stats = await pipeline.run(
        pgkc_data_path=args.pgkc_data,
        output_path=args.output,
        max_questions=args.max_questions,
    )
    
    print(f"\n{'='*60}")
    print("SFT Synthesis Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Questions solved: {stats['questions_solved']}/{stats['total_questions']} "
          f"({stats['pass_rate']*100:.1f}%)")
    print(f"Sample accuracy: {stats['sample_accuracy']*100:.1f}%")
    print(f"Exported SFT samples: {stats['num_exported_sft_samples']}")
    print(f"Output: {stats['output_path']}")


if __name__ == "__main__":
    asyncio.run(main())
