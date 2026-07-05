"""
PGKC End-to-End Pipeline v3

Orchestrates the full flow:
  1. Build (or load) entity graph            (pgkc_graph.py)
  2. Synthesize P-K chains (v3)              (pgkc_synthesizer.py)
     - 5-6 hops, treewidth>=2, P-K alternation
     - Hub avoidance, semantic domain diversity
     - Information-concealed questions
  3. Rule-based filtering                    (pgkc_filter.py Level 1)
  4. LLM validation + rewriting              (pgkc_llm_rewriter.py)
  5. (Optional) Model-based anti-shortcut    (pgkc_filter.py Level 2-5)
  6. (Optional) Run VDR Agent episodes       (vdr_agent.py)
  7. Evaluate & export results

Usage:
    # Generate, filter only (no LLM/VLM required)
    python pgkc_pipeline.py --num-samples 100 --dry-run

    # With LLM validation + rewriting
    python pgkc_pipeline.py --num-samples 50 --llm-enhance

    # Full pipeline with model-based filtering
    python pgkc_pipeline.py --num-samples 50 --model-filter

    # Full run with VDR Agent evaluation
    python pgkc_pipeline.py --num-samples 20 --vlm-url http://localhost:8000/v1
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time

logger = logging.getLogger(__name__)

GRAPH_CACHE = "/tmp/pgkc_graph.pkl"
DEFAULT_OUTPUT_DIR = "/tmp/pgkc_output"


def build_or_load_graph(force_rebuild=False):
    """Build the PGKC graph or load from cache."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from searcheyes.pgkc_graph import PGKCGraph

    if not force_rebuild and os.path.exists(GRAPH_CACHE):
        logger.info("Loading cached graph from %s", GRAPH_CACHE)
        graph = pickle.load(open(GRAPH_CACHE, "rb"))
        logger.info("  %d nodes, %d edges",
                    len(graph.nodes),
                    sum(len(n.out_edges) for n in graph.nodes.values()))
        return graph

    logger.info("Building graph from scratch...")
    graph = PGKCGraph().build()
    pickle.dump(graph, open(GRAPH_CACHE, "wb"))
    logger.info("  Saved to %s", GRAPH_CACHE)
    return graph


def synthesize_questions(graph, num_samples, min_hops, max_hops, seed):
    """Generate multi-hop questions with v3 constraints:
    - 5-6 hops, treewidth>=2, P-K alternation
    - Hub avoidance, semantic domain diversity (>=3 domains)
    - Information-concealed questions with fuzzy constraints
    """
    from searcheyes.pgkc_synthesizer import PGKCSynthesizer

    synth = PGKCSynthesizer(graph, seed=seed)
    samples = synth.generate_batch(
        num_samples, min_hops=min_hops, max_hops=max_hops)
    return samples


def rule_filter(graph, samples):
    """Apply Level 1 rule-based filters (fast, no API)."""
    from searcheyes.pgkc_filter import PGKCFilter

    filt = PGKCFilter(graph, enable_model_filter=False)
    passed, rejected = filt.filter_batch(samples, skip_model=True)
    return passed, rejected


def llm_enhance(samples, graph, validate=True, rewrite=True, check_unique=True):
    """Apply LLM validation + rewriting (requires Python 3.10 + API)."""
    from searcheyes.pgkc_llm_rewriter import PGKCRewriter

    rewriter = PGKCRewriter()
    enhanced, rejected = rewriter.enhance_batch(
        samples, graph=graph,
        validate=validate, rewrite=rewrite, check_unique=check_unique)
    return enhanced, rejected


def model_filter(graph, samples, vlm_url, vlm_model):
    """Apply Level 2-5 model-based anti-shortcut filtering."""
    from searcheyes.pgkc_filter import PGKCFilter

    filt = PGKCFilter(
        graph, enable_model_filter=True,
        vlm_base_url=vlm_url, vlm_model=vlm_model)
    passed, rejected = filt.filter_batch(samples, skip_model=False)
    return passed, rejected


def evaluate_answer(predicted, ground_truth):
    """Fuzzy answer evaluation: exact-match + containment."""
    pred_lower = predicted.strip().lower()
    gt_lower = ground_truth.strip().lower()

    exact_match = pred_lower == gt_lower
    containment = gt_lower in pred_lower or pred_lower in gt_lower

    # Normalized (remove articles)
    pred_norm = pred_lower.replace("the ", "").replace("a ", "").replace("an ", "")
    gt_norm = gt_lower.replace("the ", "").replace("a ", "").replace("an ", "")
    normalized_match = pred_norm == gt_norm or gt_norm in pred_norm

    return {
        "exact_match": exact_match,
        "containment": containment,
        "normalized": normalized_match,
        "correct": exact_match or containment or normalized_match,
    }


def run_vdr_episodes(samples, vlm_url, model_name, max_steps):
    """Run VDR Agent on synthesized questions."""
    from searcheyes.vdr_agent import VDRAgent, VLMClient
    from searcheyes.vdr_tools import VDRToolKit
    from searcheyes.rag_engine import RagEngine

    logger.info("Initializing RAG engine...")
    rag = RagEngine()

    logger.info("Initializing VDR toolkit and agent...")
    toolkit = VDRToolKit(rag_engine=rag)
    vlm = VLMClient(base_url=vlm_url, model=model_name)
    agent = VDRAgent(toolkit=toolkit, vlm_client=vlm, max_steps=max_steps)

    results = []
    for idx, sample in enumerate(samples):
        logger.info("Running episode %d/%d: %s",
                    idx + 1, len(samples), sample["question"][:60])
        episode = agent.run_episode(
            question=sample["question"],
            image_path=sample["image_path"],
            task_id=sample["question_id"],
            ground_truth=sample["answer"],
        )

        eval_result = evaluate_answer(episode.final_answer, sample["answer"])
        episode.correct = eval_result["correct"]

        results.append({
            "question_id": sample["question_id"],
            "question": sample["question"],
            "answer_gt": sample["answer"],
            "answer_pred": episode.final_answer,
            "correct": episode.correct,
            "num_steps": episode.total_steps,
            "chain": sample["chain"],
            "num_hops": sample["num_hops"],
            "structure": sample.get("structure", "linear"),
        })

        status = "CORRECT" if episode.correct else "WRONG"
        logger.info("  [%s] GT=%s, Pred=%s",
                    status, sample["answer"], episode.final_answer)

    return results


def print_sample_summary(samples, limit=5):
    """Pretty-print sample summaries."""
    for s in samples[:limit]:
        print("-" * 60)
        print("Q: {}".format(s["question"]))
        print("A: {}".format(s["answer"]))
        print("Structure: {} | Hops: {} | Types: {}".format(
            s.get("structure", "linear"), s["num_hops"],
            s.get("hop_types", [])))
        chain_str = " -> ".join(
            "{}--[{}]-->{}".format(
                h["from_title"][:20], h["relation_name"][:15], h["to_title"][:20])
            for h in s["chain"]
        )
        print("Chain: {}".format(chain_str))
        if s.get("constraints"):
            for c in s["constraints"]:
                print("  Constraint: {} [{}] {} (dir={})".format(
                    c["target_title"][:20], c["relation_name"][:15],
                    c["constraint_title"][:20], c["direction"]))
        print("Image: {}".format(s.get("image_path", "N/A")))
        print()


def main():
    parser = argparse.ArgumentParser(description="PGKC Pipeline v2")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of questions to generate")
    parser.add_argument("--min-hops", type=int, default=5)
    parser.add_argument("--max-hops", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthesize + rule-filter only (no LLM/VLM)")
    parser.add_argument("--llm-enhance", action="store_true",
                        help="Run LLM validation + rewriting")
    parser.add_argument("--model-filter", action="store_true",
                        help="Run Level 2-5 model-based filtering")
    parser.add_argument("--vlm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vlm-model", type=str, default="/dev/shm/Qwen3.6-27B")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-graph", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Graph ────────────────────────────────────────────
    logger.info("═══ Step 1: Entity Graph ═══")
    graph = build_or_load_graph(force_rebuild=args.rebuild_graph)

    # ── Step 2: Synthesize ───────────────────────────────────────
    logger.info("═══ Step 2: Synthesizing %d questions (hops %d-%d) ═══",
                args.num_samples, args.min_hops, args.max_hops)
    raw_samples = synthesize_questions(
        graph, args.num_samples, args.min_hops, args.max_hops,
        args.seed)
    logger.info("  Generated %d raw samples", len(raw_samples))

    # ── Step 3: Rule-based filtering ─────────────────────────────
    logger.info("═══ Step 3: Rule-based filtering ═══")
    passed, rejected = rule_filter(graph, raw_samples)
    logger.info("  %d passed, %d rejected", len(passed), len(rejected))

    # Stats
    structure_counts = {}
    hop_dist = {}
    for s in passed:
        st = s.get("structure", "linear")
        structure_counts[st] = structure_counts.get(st, 0) + 1
        h = s["num_hops"]
        hop_dist[h] = hop_dist.get(h, 0) + 1
    logger.info("  Structure: %s", structure_counts)
    logger.info("  Hop distribution: %s", hop_dist)

    if args.dry_run:
        # Save and show
        synth_path = os.path.join(args.output_dir, "pgkc_v3_dry.json")
        with open(synth_path, "w") as fh:
            json.dump(passed, fh, indent=2, ensure_ascii=False)
        logger.info("Dry run complete. Saved %d samples to %s",
                    len(passed), synth_path)
        print("\n=== Sample Questions ===\n")
        print_sample_summary(passed, limit=8)
        return

    # ── Step 4: LLM Enhancement ──────────────────────────────────
    if args.llm_enhance:
        logger.info("═══ Step 4: LLM Validation + Rewriting ═══")
        enhanced, llm_rejected = llm_enhance(passed, graph)
        logger.info("  %d enhanced, %d rejected by LLM", len(enhanced), len(llm_rejected))
        passed = enhanced

    # ── Step 5: Model-based anti-shortcut ────────────────────────
    if args.model_filter:
        logger.info("═══ Step 5: Model-based anti-shortcut filtering ═══")
        passed, model_rejected = model_filter(
            graph, passed, args.vlm_url, args.vlm_model)
        logger.info("  %d passed model filter", len(passed))

    # Save final questions
    final_path = os.path.join(args.output_dir, "pgkc_v3_final.json")
    with open(final_path, "w") as fh:
        json.dump(passed, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d final questions to %s", len(passed), final_path)

    print("\n=== Final Questions ({}) ===\n".format(len(passed)))
    print_sample_summary(passed, limit=5)

    # ── Step 6: VDR Agent Evaluation (optional) ──────────────────
    if not args.dry_run and args.vlm_url:
        logger.info("═══ Step 6: VDR Agent Evaluation ═══")
        results = run_vdr_episodes(
            passed, args.vlm_url, args.vlm_model, args.max_steps)

        correct_count = sum(1 for r in results if r["correct"])
        total = len(results)
        accuracy = correct_count / max(total, 1)

        logger.info("═" * 50)
        logger.info("RESULTS: %d/%d correct (%.1f%%)",
                    correct_count, total, accuracy * 100)
        logger.info("═" * 50)

        # Breakdown by structure
        for st in ["linear", "branched"]:
            st_results = [r for r in results if r.get("structure") == st]
            if st_results:
                st_correct = sum(1 for r in st_results if r["correct"])
                logger.info("  %s: %d/%d (%.1f%%)",
                            st, st_correct, len(st_results),
                            100.0 * st_correct / len(st_results))

        results_path = os.path.join(args.output_dir, "pgkc_v2_results.json")
        with open(results_path, "w") as fh:
            json.dump({
                "accuracy": accuracy,
                "correct": correct_count,
                "total": total,
                "hop_distribution": hop_dist,
                "structure_counts": structure_counts,
                "results": results,
            }, fh, indent=2, ensure_ascii=False)
        logger.info("Saved results to %s", results_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()