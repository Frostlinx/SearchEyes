"""
PGKC Anti-shortcut Filter v2

Five-level filtering to ensure questions genuinely require multi-step
visual + knowledge reasoning:

Level 1 (Rule-based, fast):
  - Answer-leak: answer text in question
  - Anchor-answer shortcut: answer is direct neighbor of anchor
  - Popularity bias: answer is too-popular entity
  - Chain quality: intermediates must have Wiki6M context
  - Image grounding: image exists and is informative

Level 2 (VLM direct-answer):
  - Show VLM the image + question, NO search allowed
  - If VLM answers correctly → question is too easy → reject

Level 3 (Text-only):
  - Give LLM the question text WITHOUT the image
  - If LLM answers correctly → image is not needed → reject

Level 4 (Vision-only):
  - Show VLM the image WITHOUT the question context
  - Ask "What is the answer?" — if correct → question leaks too much

Level 5 (Shortcut detection):
  - Check if intermediate hops can be skipped
  - Verify the question truly needs each hop
"""

import json
import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Very popular entities that VLMs might guess without reasoning
# Kept in sync with pgkc_synthesizer.py OVERLY_POPULAR_ANSWER_QIDS
OVERLY_POPULAR_ENTITIES = {
    # Countries
    "Q30", "Q145", "Q142", "Q183", "Q148", "Q17", "Q159", "Q408",
    "Q16", "Q38", "Q29", "Q155", "Q668", "Q60", "Q84", "Q90",
    "Q64", "Q1490", "Q956", "Q649", "Q5", "Q515", "Q6256",
    # Major cities
    "Q1492", "Q1726", "Q1748", "Q1757", "Q220", "Q239", "Q270",
    "Q1085", "Q1860", "Q1861", "Q33935", "Q2807", "Q34370",
    "Q3561", "Q1354", "Q1530", "Q2044", "Q365", "Q490",
    "Q36036", "Q8684", "Q1563", "Q8678", "Q85", "Q1070",
}

# Minimum image file size (bytes) to consider it informative
MIN_IMAGE_SIZE = 5000


class PGKCRuleFilter:
    """Level 1: Fast rule-based filtering (no model calls)."""

    def __init__(self, graph):
        self.graph = graph

    def check(self, sample):
        """Return None if passed, or rejection reason string."""
        answer_lower = sample["answer"].lower()
        question_lower = sample["question"].lower()

        # 1. Answer-leak
        if answer_lower in question_lower:
            return "answer_leak"

        # 2. Anchor-answer shortcut
        anchor_qid = sample["anchor_qid"]
        answer_qid = sample["answer_qid"]
        anchor_node = self.graph.nodes.get(anchor_qid)
        if anchor_node and sample["num_hops"] > 1:
            direct_neighbors = set(tgt for _, tgt in anchor_node.out_edges)
            direct_neighbors.update(src for _, src in anchor_node.in_edges)
            if answer_qid in direct_neighbors:
                return "anchor_answer_shortcut"

        # 3. Popularity bias
        if answer_qid in OVERLY_POPULAR_ENTITIES:
            return "popularity_bias"

        # 4. Chain quality
        for hop in sample["chain"]:
            to_node = self.graph.nodes.get(hop["to_qid"])
            if to_node is None:
                return "missing_intermediate"
            if not to_node.summary and not to_node.title:
                return "no_context_intermediate"

        # 5. Image grounding
        image_path = sample.get("image_path")
        if not image_path or not os.path.exists(image_path):
            return "no_image"
        try:
            if os.path.getsize(image_path) < MIN_IMAGE_SIZE:
                return "tiny_image"
        except OSError:
            return "image_error"

        # 6. Answer too short
        if len(sample["answer"]) < 2:
            return "answer_too_short"

        # 7. Question too short
        if len(sample["question"]) < 30:
            return "question_too_short"

        # 8. Constraint leak — if branched, constraint entity name in answer
        if sample.get("constraints"):
            for c in sample["constraints"]:
                if c["constraint_title"].lower() == answer_lower:
                    return "constraint_is_answer"

        return None


class PGKCModelFilter:
    """Level 2-5: Model-based anti-shortcut filtering.

    Uses a VLM (for visual) and LLM (for text-only) to verify that
    the question genuinely requires multi-modal multi-hop reasoning.

    Requires Python 3.10 environment with openai/httpx.
    """

    def __init__(
        self,
        vlm_base_url="http://localhost:8000/v1",
        vlm_model="/dev/shm/Qwen3.6-27B",
        llm_api_key="sk-9eba1adb38fa4cb1af5dca05f58f8472",
        llm_base_url="https://routify.alibaba-inc.com/protocol/openai/v1",
        llm_model="claude-sonnet-4-6-20260217",
    ):
        self.vlm_base_url = vlm_base_url
        self.vlm_model = vlm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self._vlm_client = None
        self._llm_client = None

    def _get_vlm_client(self):
        if self._vlm_client is None:
            import openai
            self._vlm_client = openai.OpenAI(
                api_key="EMPTY",
                base_url=self.vlm_base_url,
            )
        return self._vlm_client

    def _get_llm_client(self):
        if self._llm_client is None:
            import openai
            import httpx
            self._llm_client = openai.OpenAI(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                http_client=httpx.Client(verify=False),
            )
        return self._llm_client

    # ── Level 2: VLM Direct Answer ──────────────────────────────

    def check_vlm_direct(self, sample):
        """Show VLM image + question, no search. If correct → too easy.

        Returns:
            (passed: bool, vlm_answer: str)
        """
        import base64
        image_path = sample["image_path"]
        if not image_path or not os.path.exists(image_path):
            return True, ""

        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except (IOError, OSError):
            return True, ""

        prompt = (
            "Answer this question directly based ONLY on what you see in "
            "the image. Do NOT search for additional information. "
            "Give a short, specific answer.\n\n"
            "Question: {}\n\nAnswer:".format(sample["question"])
        )

        try:
            client = self._get_vlm_client()
            resp = client.chat.completions.create(
                model=self.vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": "data:image/jpeg;base64," + image_data}},
                        {"type": "text", "text": prompt},
                    ]
                }],
                max_tokens=100,
                temperature=0.0,
            )
            vlm_answer = resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("VLM direct-answer call failed: %s", exc)
            return True, ""

        # Check if VLM got it right
        is_correct = self._answer_matches(vlm_answer, sample["answer"])
        return (not is_correct), vlm_answer

    # ── Level 3: Text-only (no image) ───────────────────────────

    def check_text_only(self, sample):
        """Give LLM the question WITHOUT the image. If correct → image not needed.

        Returns:
            (passed: bool, llm_answer: str)
        """
        prompt = (
            "Answer the following question. You do NOT have access to any "
            "image. Try your best to answer based on your knowledge.\n\n"
            "Question: {}\n\n"
            "Give a short, specific answer. If you cannot answer, say "
            "'CANNOT_ANSWER'.".format(sample["question"])
        )

        try:
            client = self._get_llm_client()
            resp = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            llm_answer = resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Text-only LLM call failed: %s", exc)
            return True, ""

        if "CANNOT_ANSWER" in llm_answer.upper():
            return True, llm_answer

        is_correct = self._answer_matches(llm_answer, sample["answer"])
        return (not is_correct), llm_answer

    # ── Level 4: Vision-only (image, no question context) ────────

    def check_vision_only(self, sample):
        """Show VLM the image + a generic 'what is the answer?' prompt.
        If correct → the question leaks too much visual info.

        Returns:
            (passed: bool, vlm_answer: str)
        """
        import base64
        image_path = sample["image_path"]
        if not image_path or not os.path.exists(image_path):
            return True, ""

        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except (IOError, OSError):
            return True, ""

        # Give a hint about what type of answer is expected, but not the chain
        last_rel = sample["chain"][-1]["relation_name"]
        prompt = (
            "Look at this image. Based only on what you see, what would "
            "you guess as the '{}' associated with this? "
            "Give a short, specific answer.".format(last_rel)
        )

        try:
            client = self._get_vlm_client()
            resp = client.chat.completions.create(
                model=self.vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": "data:image/jpeg;base64," + image_data}},
                        {"type": "text", "text": prompt},
                    ]
                }],
                max_tokens=100,
                temperature=0.0,
            )
            vlm_answer = resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Vision-only VLM call failed: %s", exc)
            return True, ""

        is_correct = self._answer_matches(vlm_answer, sample["answer"])
        return (not is_correct), vlm_answer

    # ── Level 5: Shortcut detection ─────────────────────────────

    def check_shortcut(self, sample, graph):
        """Verify that each hop in the chain is actually needed.

        Check: can you reach the answer from the anchor in fewer hops
        than the chain specifies? If yes → shortcut exists.

        Returns:
            (passed: bool, reason: str)
        """
        if sample["num_hops"] <= 1:
            return True, ""

        anchor_qid = sample["anchor_qid"]
        answer_qid = sample["answer_qid"]

        # BFS from anchor to answer, limited to num_hops - 1
        max_depth = sample["num_hops"] - 1
        frontier = {anchor_qid}
        visited = {anchor_qid}

        for depth in range(max_depth):
            next_frontier = set()
            for qid in frontier:
                node = graph.nodes.get(qid)
                if node is None:
                    continue
                for _, tgt in node.out_edges:
                    if tgt == answer_qid:
                        return False, "reachable_in_{}_hops".format(depth + 1)
                    if tgt not in visited:
                        visited.add(tgt)
                        next_frontier.add(tgt)
            frontier = next_frontier

        return True, ""

    # ── Utility ──────────────────────────────────────────────────

    def _answer_matches(self, predicted, ground_truth):
        """Fuzzy answer matching: exact, containment, or normalized."""
        pred = predicted.lower().strip().rstrip(".")
        gt = ground_truth.lower().strip()

        if pred == gt:
            return True
        if gt in pred or pred in gt:
            return True
        # Normalize: remove articles
        pred_norm = pred.replace("the ", "").replace("a ", "").replace("an ", "")
        gt_norm = gt.replace("the ", "").replace("a ", "").replace("an ", "")
        if pred_norm == gt_norm:
            return True
        if gt_norm in pred_norm:
            return True
        return False


class PGKCFilter:
    """Unified filter: combines rule-based + model-based filtering."""

    def __init__(self, graph, enable_model_filter=False, **model_kwargs):
        """
        Args:
            graph:               PGKCGraph instance
            enable_model_filter: if True, run Level 2-5 (requires VLM/LLM)
            **model_kwargs:      passed to PGKCModelFilter
        """
        self.graph = graph
        self.rule_filter = PGKCRuleFilter(graph)
        self.model_filter = None
        if enable_model_filter:
            self.model_filter = PGKCModelFilter(**model_kwargs)

    def filter_batch(self, samples, skip_model=False):
        """Apply all filters.

        Returns:
            passed:   list of samples that pass
            rejected: list of (sample, reason) tuples
        """
        passed = []
        rejected = []

        # Level 1: Rule-based (fast, no API calls)
        for sample in samples:
            reason = self.rule_filter.check(sample)
            if reason:
                rejected.append((sample, "rule:" + reason))
            else:
                passed.append(sample)

        logger.info("Rule filter: %d/%d passed (%.1f%%)",
                    len(passed), len(samples),
                    100.0 * len(passed) / max(len(samples), 1))
        self._log_rejection_stats(rejected)

        if skip_model or self.model_filter is None:
            return passed, rejected

        # Level 2-5: Model-based (slower, requires API)
        model_passed = []
        for sample in passed:
            model_reason = self._run_model_checks(sample)
            if model_reason:
                rejected.append((sample, "model:" + model_reason))
            else:
                model_passed.append(sample)

        logger.info("Model filter: %d/%d passed (%.1f%% of rule-passed)",
                    len(model_passed), len(passed),
                    100.0 * len(model_passed) / max(len(passed), 1))
        self._log_rejection_stats(
            [(s, r) for s, r in rejected if r.startswith("model:")])

        return model_passed, rejected

    def _run_model_checks(self, sample):
        """Run Level 2-5 checks. Return None if all pass, else reason."""
        mf = self.model_filter

        # Level 5: Shortcut (no API needed, just graph BFS)
        passed, reason = mf.check_shortcut(sample, self.graph)
        if not passed:
            return "shortcut:" + reason

        # Level 3: Text-only (cheapest API call — just LLM, no image)
        passed, llm_answer = mf.check_text_only(sample)
        if not passed:
            return "text_only_solvable"
        time.sleep(0.3)

        # Level 2: VLM direct answer (needs VLM with image)
        passed, vlm_answer = mf.check_vlm_direct(sample)
        if not passed:
            return "vlm_direct_solvable"
        time.sleep(0.3)

        # Level 4: Vision-only
        passed, vlm_answer = mf.check_vision_only(sample)
        if not passed:
            return "vision_only_solvable"

        return None

    def _log_rejection_stats(self, rejected):
        reasons = {}
        for _, reason in rejected:
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            logger.info("  Rejected [%s]: %d", reason, count)


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logger.info("Loading graph...")
    graph = pickle.load(open("/tmp/pgkc_graph.pkl", "rb"))

    # Try loading v2 samples first
    sample_path = "/tmp/pgkc_samples_v2.json"
    if not os.path.exists(sample_path):
        sample_path = "/tmp/pgkc_samples.json"
    logger.info("Loading samples from %s ...", sample_path)
    samples = json.load(open(sample_path))
    logger.info("  %d samples loaded", len(samples))

    # Rule-only filter (fast)
    filt = PGKCFilter(graph, enable_model_filter=False)
    passed, rejected = filt.filter_batch(samples)

    print("\n=== PASSED ({}) ===".format(len(passed)))
    for s in passed[:10]:
        print("  Q: {}".format(s["question"][:90]))
        print("  A: {}".format(s["answer"]))
        print("  Structure: {} | Hops: {}".format(
            s.get("structure", "linear"), s["num_hops"]))
        print()

    print("=== REJECTED ({}) ===".format(len(rejected)))
    for s, reason in rejected[:10]:
        print("  [{}] A: {}".format(reason, s["answer"]))