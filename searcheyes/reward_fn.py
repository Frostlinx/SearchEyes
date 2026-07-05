"""
SearchEyes Reward Function for verl RL training.

Computes outcome reward (EM match) and extracts hop-anchor metadata
for HaPO advantage estimation.

Usage in verl config:
    reward_model:
        custom_reward_function:
            path: ./reward_fn.py
            name: searcheyes_compute_score
"""

import json
import re
import string
from typing import Any


# ── Answer extraction & matching ──

def normalize_answer(text: str) -> str:
    """Lowercase, remove articles, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def extract_answer(response: str) -> str | None:
    """Extract content from the last <answer>...</answer> tag."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", response, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def exact_match(prediction: str, golden_answers: str | list[str]) -> bool:
    """Check if normalized prediction matches any golden answer."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized = normalize_answer(prediction)
    return any(normalize_answer(ga) == normalized for ga in golden_answers)


def substring_match(prediction: str, golden_answers: str | list[str]) -> bool:
    """Check if any golden answer is a substring of normalized prediction."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized = normalize_answer(prediction)
    return any(normalize_answer(ga) in normalized for ga in golden_answers)


# ── Step parsing & entity extraction ──

def extract_entity_ids_from_text(text: str) -> set[str]:
    """Extract Wikidata entity IDs (Q-numbers) from text.

    Looks for patterns like:
    - (ID: Q12345)
    - entity_id: Q12345
    - "qid": "Q12345"
    """
    return set(re.findall(r"Q\d+", text))


def parse_steps_from_response(response: str) -> list[dict]:
    """Parse multi-turn agent response into steps.

    Each step is delimited by tool-call / observation boundaries.
    We look for patterns like:
    - <tool_call>...</tool_call> or ```tool_call...```
    - <observation>...</observation> or <tool_response>...</tool_response>

    Returns list of dicts with:
        - step_index: int
        - action_text: str (the tool call)
        - observation_text: str (the tool response / observation)
        - retrieved_entity_ids: set of Q-IDs found in observation
        - token_start: approximate char offset (will be converted to token offset later)
        - token_end: approximate char offset
    """
    steps = []

    # Pattern 1: <search>query</search> ... <information>...</information>
    # Pattern 2: tool calls with observations
    # We split by observation/information blocks
    observation_patterns = [
        r"<(?:observation|information|tool_response|results?)>(.*?)</(?:observation|information|tool_response|results?)>",
        r"```(?:observation|output|result)\n(.*?)```",
    ]

    # Find all observations
    obs_matches = []
    for pattern in observation_patterns:
        for match in re.finditer(pattern, response, re.DOTALL):
            obs_matches.append((match.start(), match.end(), match.group(1)))

    # Also find search/tool_call blocks
    action_patterns = [
        r"<(?:search|tool_call)>(.*?)</(?:search|tool_call)>",
        r"```(?:tool_call|json)\n(.*?)```",
    ]

    action_matches = []
    for pattern in action_patterns:
        for match in re.finditer(pattern, response, re.DOTALL):
            action_matches.append((match.start(), match.end(), match.group(1)))

    if not obs_matches and not action_matches:
        # No structured steps found; treat entire response as single step
        return [{
            "step_index": 0,
            "action_text": "",
            "observation_text": "",
            "retrieved_entity_ids": set(),
            "char_start": 0,
            "char_end": len(response),
        }]

    # Merge and sort all matches by position
    all_blocks = []
    for start, end, text in action_matches:
        all_blocks.append(("action", start, end, text))
    for start, end, text in obs_matches:
        all_blocks.append(("observation", start, end, text))
    all_blocks.sort(key=lambda x: x[1])

    # Group into steps: each action+observation pair is one step
    current_step = {"step_index": 0, "action_text": "", "observation_text": "",
                    "retrieved_entity_ids": set(), "char_start": 0, "char_end": 0}
    step_idx = 0

    for block_type, start, end, text in all_blocks:
        if block_type == "action":
            if current_step["action_text"] and current_step["observation_text"]:
                # Previous step is complete, save and start new
                steps.append(current_step)
                step_idx += 1
                current_step = {"step_index": step_idx, "action_text": "", "observation_text": "",
                                "retrieved_entity_ids": set(), "char_start": start, "char_end": 0}
            current_step["action_text"] = text
            if current_step["char_start"] == 0 and step_idx == 0:
                current_step["char_start"] = start
        elif block_type == "observation":
            current_step["observation_text"] = text
            current_step["retrieved_entity_ids"] = extract_entity_ids_from_text(text)
            current_step["char_end"] = end

    # Don't forget the last step
    if current_step["action_text"] or current_step["observation_text"]:
        if current_step["char_end"] == 0:
            current_step["char_end"] = len(response)
        steps.append(current_step)

    # If no steps were created, add a default one
    if not steps:
        steps.append({
            "step_index": 0,
            "action_text": "",
            "observation_text": "",
            "retrieved_entity_ids": set(),
            "char_start": 0,
            "char_end": len(response),
        })

    return steps


def extract_gold_entities_from_chain(chain: list[dict]) -> list[str]:
    """Extract ordered gold entity IDs from the reasoning chain.

    Args:
        chain: List of hop dicts with from_qid, to_qid fields

    Returns:
        Ordered list of unique entity IDs in the chain
    """
    entities = []
    seen = set()
    if chain:
        first_qid = chain[0].get("from_qid", "")
        if first_qid and first_qid not in seen:
            entities.append(first_qid)
            seen.add(first_qid)
    for hop in chain:
        to_qid = hop.get("to_qid", "")
        if to_qid and to_qid not in seen:
            entities.append(to_qid)
            seen.add(to_qid)
    return entities


# ── Main reward function ──

def searcheyes_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Compute SearchEyes reward score with HaPO metadata.

    Args:
        data_source: dataset identifier (e.g., "searcheyes_pgkc")
        solution_str: model's full response text
        ground_truth: dict with "target" (answer) and "chain" (gold reasoning chain)
        extra_info: additional info (unused)

    Returns:
        dict with:
            - score: float (0.0 or 1.0)
            - hapo_gold_entities: list of gold entity QIDs
            - hapo_step_entity_hits: dict mapping step_idx -> set of entity IDs
            - hapo_step_boundaries: None (computed from char offsets, actual token
              boundaries will be approximated by the advantage estimator)
    """
    # ── Outcome reward ──
    answer = extract_answer(solution_str)
    target = ground_truth.get("target", "") if isinstance(ground_truth, dict) else ground_truth

    if isinstance(target, list):
        target_list = target
    else:
        target_list = [target]

    if answer is not None and exact_match(answer, target_list):
        score = 1.0
    elif answer is not None and substring_match(answer, target_list):
        score = 0.5
    else:
        score = 0.0

    # Format penalty: too many answer tags
    open_count = solution_str.count("<answer>")
    close_count = solution_str.count("</answer>")
    if open_count > 10 or close_count > 10:
        score = score / 4.0

    # ── HaPO hop anchor metadata ──
    chain = []
    if isinstance(ground_truth, dict):
        chain = ground_truth.get("chain", [])

    gold_entities = extract_gold_entities_from_chain(chain)

    # Parse steps and extract per-step entity hits
    steps = parse_steps_from_response(solution_str)
    step_entity_hits = {}
    for step in steps:
        entity_ids = step["retrieved_entity_ids"]
        if entity_ids:
            step_entity_hits[step["step_index"]] = list(entity_ids)

    # Step boundaries (char-level, will be None for token-level)
    # In the advantage estimator, if step_boundaries is None,
    # we fall back to uniform assignment across the response
    step_boundaries = None

    result = {
        "score": score,
        "hapo_gold_entities": json.dumps(gold_entities),
        "hapo_step_entity_hits": json.dumps(step_entity_hits),
    }

    return result
