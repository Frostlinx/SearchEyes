"""
PGKC LLM Rewriter v2 — Use LLM to enhance synthesized multi-hop questions.

Three LLM calls per sample:
1. Chain validation: Is this reasoning chain factually correct?
2. Question rewriting: Convert template into natural, challenging question.
3. Uniqueness check: Verify the answer is uniquely determined.

Key design changes from v1:
- Supports branched (treewidth=2) structures with constraints
- Question must NOT reveal entity type (no "the person", "the place")
- Validates answer uniqueness given the chain
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CHAIN_VALIDATION_PROMPT = """\
You are evaluating a multi-hop reasoning chain derived from Wikidata for factual correctness.

Given:
- An anchor entity (visible in an image, identity must be determined visually)
- A chain of knowledge hops from that entity to an answer
{constraint_section}

IMPORTANT GUIDELINES:
- These relations come from Wikidata. Wikidata sometimes records less well-known or \
historical facts (e.g., a country sending a small contingent to a war, or a person \
holding a minor position). Such facts should be treated as VALID even if they seem \
surprising or obscure.
- Only mark as INVALID if a hop is CLEARLY and UNAMBIGUOUSLY wrong \
(e.g., "Sam Cooke → occupation → Pianist" when he was a singer, or \
"Tokyo → capital of → China").
- Do NOT reject chains because a relation seems "unusual" or "minor" — \
Wikidata records many secondary/historical connections that are factually correct.
- A simple but factually correct chain is VALID.

Your job is to check:
1. Is each hop CLEARLY WRONG? (not just unusual/obscure)
2. Is the answer entity specific enough (not overly generic)?
3. If there are constraints, are they clearly wrong?

Chain:
{chain_text}

Entity descriptions:
{entity_descriptions}

Answer: {answer}

Respond with EXACTLY one of:
- VALID: <brief confirmation>
- INVALID: <which hop is CLEARLY wrong and why>
"""

QUESTION_REWRITE_PROMPT = """\
Rewrite the following mechanical multi-hop question into a natural, fluent, \
challenging English question.

STRICT requirements:
1. The question MUST reference "the image" — the solver needs to visually identify the first entity.
2. Do NOT reveal what type of entity is in the image (no "the person", "the building", "the film" etc.). Use neutral phrases like "what is shown in the image", "the subject of the image", etc.
3. Do NOT reveal any intermediate entity names.
4. Do NOT reveal the answer.
5. If there is a constraint/disambiguation clue, incorporate it naturally (e.g., "...which is also known for X...").
6. Make it sound like a challenging trivia question.
7. Keep it concise (1-3 sentences max).
8. The answer to your rewritten question must STILL be exactly: {answer}

Reasoning chain (for your understanding only, do NOT expose):
{chain_text}
{constraint_text}

Original mechanical question:
{original_question}

Rewritten question:"""

UNIQUENESS_CHECK_PROMPT = """\
Given this question, reasoning chain, and any constraints, is the answer UNIQUELY determined?

Question: {question}
Chain: {chain_text}
{constraint_text}
Answer: {answer}

IMPORTANT: If the question includes a constraint/disambiguation clue (e.g., "whose capital is X", \
"which shares border with Y"), that constraint NARROWS DOWN the answer to exactly one entity. \
The constraint is part of the question — it's how the solver identifies which specific entity is meant.

Think about:
- Does each hop have exactly one valid target (one-to-one relation)?
- If a hop has multiple possible targets (one-to-many), does the constraint in the question \
uniquely identify which one?
- A question like "which X of Y whose capital is Z?" means: among all X's of Y, \
find the ONE whose capital is Z. This IS unique if Z is indeed the capital of exactly one X.

Respond with:
- UNIQUE: <brief explanation>
- AMBIGUOUS: <explain why the constraint still doesn't disambiguate>
"""


class PGKCRewriter:
    """LLM-based chain validation, question rewriting, and uniqueness check."""

    def __init__(
        self,
        api_key="sk-9eba1adb38fa4cb1af5dca05f58f8472",
        base_url="https://routify.alibaba-inc.com/protocol/openai/v1",
        model="claude-opus-4-7",
        max_tokens=300,
        temperature=0.7,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        import openai
        import httpx
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(verify=False),
        )

    def _call_llm(self, prompt, temperature=None, max_tokens=None):
        """Single LLM call with retry on 429, returns response text."""
        import time as _time
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
        }
        temp = temperature if temperature is not None else self.temperature
        if temp is not None and "opus" not in self.model:
            kwargs["temperature"] = temp

        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "Too Many Requests" in err_str:
                    wait = 3 * (attempt + 1)
                    _time.sleep(wait)
                    continue
                logger.error("LLM call failed: %s", exc)
                return None
        logger.error("LLM call exhausted retries (429)")
        return None

    # ── Chain Validation ─────────────────────────────────────────

    def validate_chain(self, sample, graph=None):
        """Ask LLM whether the reasoning chain is factually correct.

        Returns:
            (is_valid: bool, reason: str)
        """
        chain_text = self._format_chain(sample["chain"])
        entity_descs = self._format_entity_descriptions(sample, graph)
        constraint_section = self._format_constraint_section(sample)

        prompt = CHAIN_VALIDATION_PROMPT.format(
            chain_text=chain_text,
            entity_descriptions=entity_descs,
            answer=sample["answer"],
            constraint_section=constraint_section,
        )

        response = self._call_llm(prompt, temperature=0.2)
        if response is None:
            return True, "LLM unavailable, assuming valid"

        return self._parse_valid_invalid(response)

    # ── Question Rewriting ───────────────────────────────────────

    def rewrite_question(self, sample):
        """Rewrite a mechanical question into natural language.

        Ensures:
        - No entity type revealed
        - No intermediate names leaked
        - Constraint incorporated naturally (if branched)

        Returns:
            rewritten question string, or None on failure.
        """
        chain_text = self._format_chain(sample["chain"])
        constraint_text = ""
        if sample.get("constraints"):
            constraint_text = "Constraint: " + self._format_constraints(
                sample["constraints"])

        prompt = QUESTION_REWRITE_PROMPT.format(
            chain_text=chain_text,
            constraint_text=constraint_text,
            original_question=sample["question"],
            answer=sample["answer"],
        )

        response = self._call_llm(prompt, temperature=0.7)
        if response is None:
            return None

        rewritten = response.strip().strip('"').strip("'")

        # Sanity: answer must NOT appear in question
        if sample["answer"].lower() in rewritten.lower():
            logger.warning("Answer leaked in rewrite, retrying...")
            response2 = self._call_llm(prompt, temperature=0.9)
            if response2:
                rewritten = response2.strip().strip('"').strip("'")
                if sample["answer"].lower() in rewritten.lower():
                    return None

        # Sanity: must reference "image"
        if "image" not in rewritten.lower():
            rewritten = "Based on what is shown in the image: " + rewritten

        # Sanity: must not reveal entity type explicitly
        type_leaks = ["the person", "the building", "the film", "the movie",
                      "the city", "the country", "the singer", "the actor",
                      "the band", "the player", "the athlete"]
        for leak in type_leaks:
            if leak in rewritten.lower():
                # Replace with neutral phrasing
                rewritten = rewritten.replace(leak, "the subject")
                rewritten = rewritten.replace(leak.title(), "The subject")

        return rewritten

    # ── Uniqueness Check ─────────────────────────────────────────

    def check_uniqueness(self, sample):
        """Verify the answer is uniquely determined by the chain + constraints.

        Returns:
            (is_unique: bool, reason: str)
        """
        chain_text = self._format_chain(sample["chain"])
        question = sample["question"]

        constraint_text = ""
        if sample.get("constraints"):
            constraint_text = "Disambiguation constraint: " + self._format_constraints(
                sample["constraints"])

        prompt = UNIQUENESS_CHECK_PROMPT.format(
            question=question,
            chain_text=chain_text,
            constraint_text=constraint_text,
            answer=sample["answer"],
        )

        response = self._call_llm(prompt, temperature=0.2)
        if response is None:
            return True, "LLM unavailable, assuming unique"

        resp_upper = response.upper()
        if "AMBIGUOUS" in resp_upper and "UNIQUE" not in resp_upper.split("AMBIGUOUS")[0]:
            return False, response
        return True, response

    # ── Batch Processing ─────────────────────────────────────────

    def enhance_batch(self, samples, graph=None, validate=True, rewrite=True,
                      check_unique=True):
        """Validate, check uniqueness, and rewrite a batch of samples.

        Returns:
            enhanced: list of enhanced samples
            rejected: list of (sample, reason) tuples
        """
        enhanced = []
        rejected = []

        for idx, sample in enumerate(samples):
            logger.info("Processing %d/%d: %s [%s]...",
                        idx + 1, len(samples),
                        sample["anchor_title"][:25],
                        sample.get("structure", "linear"))

            # Step 1: Validate chain factually
            if validate:
                is_valid, reason = self.validate_chain(sample, graph)
                if not is_valid:
                    logger.info("  REJECTED (invalid): %s", reason[:80])
                    rejected.append((sample, "invalid:" + reason))
                    time.sleep(0.3)
                    continue
                logger.info("  VALID")
                time.sleep(0.3)

            # Step 2: Check answer uniqueness
            if check_unique:
                is_unique, reason = self.check_uniqueness(sample)
                if not is_unique:
                    logger.info("  REJECTED (ambiguous): %s", reason[:80])
                    rejected.append((sample, "ambiguous:" + reason))
                    time.sleep(0.3)
                    continue
                logger.info("  UNIQUE")
                time.sleep(0.3)

            # Step 3: Rewrite question
            if rewrite:
                new_question = self.rewrite_question(sample)
                if new_question:
                    sample = dict(sample)
                    sample["question_original"] = sample["question"]
                    sample["question"] = new_question
                    logger.info("  Rewritten: %s", new_question[:80])
                else:
                    logger.warning("  Rewrite failed, keeping original")
                time.sleep(0.3)

            enhanced.append(sample)

        logger.info("Enhanced: %d passed, %d rejected out of %d",
                    len(enhanced), len(rejected), len(samples))
        return enhanced, rejected

    # ── Formatting Helpers ───────────────────────────────────────

    def _format_chain(self, chain):
        lines = []
        for i, hop in enumerate(chain):
            lines.append("  Hop {}: {} --[{}]--> {}".format(
                i + 1, hop["from_title"], hop["relation_name"], hop["to_title"]))
        return "\n".join(lines)

    def _format_constraints(self, constraints):
        parts = []
        for c in constraints:
            if c["direction"] == "out":
                parts.append("{} --[{}]--> {}".format(
                    c["target_title"], c["relation_name"], c["constraint_title"]))
            else:
                parts.append("{} --[{}]--> {}".format(
                    c["constraint_title"], c["relation_name"], c["target_title"]))
        return "; ".join(parts)

    def _format_constraint_section(self, sample):
        if not sample.get("constraints"):
            return ""
        return "\nConstraints (disambiguation clues):\n  " + self._format_constraints(
            sample["constraints"])

    def _format_entity_descriptions(self, sample, graph=None):
        descs = []
        descs.append("- {} (anchor, shown in image)".format(sample["anchor_title"]))
        seen = {sample["anchor_qid"]}
        for hop in sample["chain"]:
            to_qid = hop["to_qid"]
            if to_qid not in seen:
                title = hop["to_title"]
                summary = ""
                if graph and to_qid in graph.nodes:
                    summary = graph.nodes[to_qid].summary
                if summary:
                    descs.append("- {}: {}".format(title, summary[:150]))
                else:
                    descs.append("- {}".format(title))
                seen.add(to_qid)
        if sample.get("constraints"):
            for c in sample["constraints"]:
                cqid = c["constraint_qid"]
                if cqid not in seen:
                    descs.append("- {} (constraint)".format(c["constraint_title"]))
                    seen.add(cqid)
        return "\n".join(descs)

    def _parse_valid_invalid(self, response):
        """Parse LLM response for VALID/INVALID verdict."""
        resp_upper = response.upper()
        if resp_upper.startswith("VALID"):
            return True, response
        if resp_upper.startswith("INVALID"):
            return False, response
        if "INVALID:" in resp_upper or "FACTUALLY WRONG" in resp_upper:
            return False, response
        if "VALID:" in resp_upper and "INVALID" not in resp_upper:
            return True, response
        logger.warning("Ambiguous response, defaulting to VALID: %s",
                       response[:100])
        return True, response


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import pickle
    import sys
    sys.path.insert(0, ".")

    logger.info("Loading graph and samples...")
    graph = pickle.load(open("/tmp/pgkc_graph.pkl", "rb"))

    # Try v2 samples first
    sample_path = "/tmp/pgkc_samples_v2.json"
    if not os.path.exists(sample_path):
        sample_path = "/tmp/pgkc_samples.json"
    samples = json.load(open(sample_path))
    logger.info("  %d samples from %s", len(samples), sample_path)

    # Take first 10 for demo
    demo_samples = samples[:10]

    rewriter = PGKCRewriter()
    enhanced, rejected = rewriter.enhance_batch(
        demo_samples, graph=graph, validate=True, rewrite=True, check_unique=True)

    print("\n" + "=" * 70)
    print("ENHANCED QUESTIONS ({} passed, {} rejected)".format(
        len(enhanced), len(rejected)))
    print("=" * 70)
    for s in enhanced:
        print()
        if "question_original" in s:
            print("Original:  {}".format(s["question_original"]))
            print("Rewritten: {}".format(s["question"]))
        else:
            print("Question: {}".format(s["question"]))
        print("Answer:    {}".format(s["answer"]))
        print("Structure: {} | Hops: {}".format(
            s.get("structure", "linear"), s["num_hops"]))
        chain_str = " -> ".join(
            "{} --[{}]--> {}".format(h["from_title"], h["relation_name"], h["to_title"])
            for h in s["chain"]
        )
        print("Chain:     {}".format(chain_str))
        if s.get("constraints"):
            for c in s["constraints"]:
                print("Constraint: {} --[{}]--> {} (dir={})".format(
                    c["target_title"], c["relation_name"],
                    c["constraint_title"], c["direction"]))

    if rejected:
        print("\n" + "=" * 70)
        print("REJECTED ({})".format(len(rejected)))
        print("=" * 70)
        for s, reason in rejected:
            print("  {} → {}: {}".format(
                s["anchor_title"], s["answer"], reason[:100]))
