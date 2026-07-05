#!/usr/bin/env python3
"""
eval_research.py — v2 Research World 评测
============================================
替代 eval_rag.py 的购物世界评测。

成功 = submit_report + fact_coverage >= threshold。
三层指标: L1 Recall (GT in results), L2 Oracle, L3 VLM e2e。

学长 philosophy: eval 定义了 measurement = 定义了 agent 学什么。
"找到正确商品并购买" → 购物行为
"找到相关证据、引用并提交报告" → 研究行为

用法:
    # Mock 评测（无 GPU，验证 pipeline 结构）
    python eval_research.py --mock --n-tasks 10

    # 真实模型评测（需要 GPU + embedding server）
    python eval_research.py --model-dir checkpoints/grpo_research/final \
        --chroma-db-path data/wit_kb_v2/chroma_db \
        --n-tasks 30 --output-json eval_research_run1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from searcheyes.research_contracts import (
    ResearchTask, ResearchDifficulty, ResearchState, ResearchPhase,
    FactSet, CitationObject, SearchResult,
)




# ═══════════════════════════════════════════════════════════
# MockRAG — generates search results from task wit_bindings
# ═══════════════════════════════════════════════════════════

class _MockRagFact:
    """Mimics searcheyes.multimodal_rag.RagFact interface."""
    def __init__(self, wit_id: str, title: str, caption: str, score: float):
        self.wit_id = wit_id
        self.title = title
        self.caption = caption
        self.score = score
        self.source_url = ""


class MockRAG:
    """Returns search results from task wit_bindings. No GPU needed.

    Call set_task() before each eval episode to configure bindings.
    """
    def __init__(self):
        self._bindings: list[dict] = []

    def set_task(self, task: ResearchTask):
        self._bindings = task.wit_bindings or []

    def get_rag_facts_combined(self, text: str = "", top_k: int = 6,
                               use_hybrid: bool = True, **kwargs) -> list:
        facts = []
        for b in self._bindings[:top_k]:
            facts.append(_MockRagFact(
                wit_id=b.get("wit_id", ""),
                title=b.get("caption", "")[:30],
                caption=b.get("caption", ""),
                score=0.85 - 0.05 * len(facts),  # descending scores
            ))
        return facts


# ═══════════════════════════════════════════════════════════
# Step & Result dataclasses
# ═══════════════════════════════════════════════════════════

@dataclass
class StepLog:
    step_idx: int = 0
    current_page: str = ""
    phase: str = ""
    action: str = ""
    params: dict = field(default_factory=dict)
    available_actions: list = field(default_factory=list)
    opened_result_id: int = -1
    opened_wit_id: str = ""
    cited_in_this_step: bool = False
    candidate_wit_ids: list = field(default_factory=list)
    gt_rank: int = -1
    reward: float = 0.0
    done: bool = False
    parse_error: bool = False
    is_invalid_action: bool = False


@dataclass
class EvalResult:
    task_id: str
    difficulty: str
    success: bool = False
    steps: int = 0
    final_action: str = ""
    gt_wit_id: str = ""
    episode_reward: float = 0.0
    error: str = ""
    # ── research-specific ──
    cited_wit_ids: list = field(default_factory=list)
    citation_count: int = 0
    coverage: float = 0.0
    report_submitted: bool = False
    # ── three-layer metrics ──
    gt_in_candidates: bool = False   # L1: GT in search results
    oracle_success: bool = False     # L2: ideal agent can cite GT
    vlm_success: bool = False        # L3: end-to-end
    # ── failure analysis ──
    goal: str = ""
    parse_error_count: int = 0
    invalid_action_count: int = 0
    opened_gt_anytime: bool = False
    opened_gt_step_indices: list = field(default_factory=list)
    cited_gt: bool = False
    search_candidates_wit_ids: list = field(default_factory=list)
    search_candidates_titles: list = field(default_factory=list)
    gt_rank_in_candidates: int = -1
    search_trigger_count: int = 0
    used_zoom: bool = False
    controller_strategy: str = ""
    controller_retry_triggered: bool = False
    step_logs: list = field(default_factory=list)
    failure_category: str = ""


# ═══════════════════════════════════════════════════════════
# Model loading & generation (GPU-dependent)
# ═══════════════════════════════════════════════════════════

def load_model(model_dir: str):
    """Load model and processor (same as eval_rag.py)."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    adapter_config = Path(model_dir) / "adapter_config.json"
    if adapter_config.exists():
        print(f"  Loading LoRA adapter: {model_dir}")
        config_data = json.loads(adapter_config.read_text())
        base_path = config_data.get("base_model_name_or_path", "")
        if not base_path:
            raise ValueError("adapter_config.json missing base_model_name_or_path")
        print(f"  Base model: {base_path}")
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_path, dtype=torch.bfloat16,
        ).to("cuda")
        model = PeftModel.from_pretrained(base_model, model_dir)
        processor = AutoProcessor.from_pretrained(base_path)
    else:
        print(f"  Loading full model: {model_dir}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16,
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(model_dir)

    model.eval()
    return model, processor


def generate_action(model, processor, image_path: str, prompt_text: str,
                    max_new_tokens: int = 128) -> str:
    """Generate one action JSON from model."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": []}]
    content = messages[0]["content"]
    if image_path and Path(image_path).exists():
        content.append({"type": "image", "image": f"file://{image_path}"})
    content.append({"type": "text", "text": prompt_text})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    result = processor.decode(generated_ids, skip_special_tokens=True)
    return result.strip()


# ═══════════════════════════════════════════════════════════
# Action parsing & prompt building
# ═══════════════════════════════════════════════════════════

def parse_action(text: str) -> dict | None:
    """Extract JSON action from model output."""
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*"action"[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def build_prompt(goal: str, state_description: str, available_actions: list[str]) -> str:
    """Build prompt identical to training format."""
    actions_text = ", ".join(available_actions)
    return (
        f"Goal: {goal}\n"
        f"State: {state_description}\n"
        f"Available actions: {actions_text}\n"
        "Choose exactly one next action.\n"
        'Return JSON only in the form {"action": "...", "params": {...}}.'
    ).strip()


# ═══════════════════════════════════════════════════════════
# Mock oracle agent (for --mock mode, no GPU needed)
# ═══════════════════════════════════════════════════════════

class MockOracleAgent:
    """Scripted oracle: search → open GT → cite → submit_report.

    Ideal trajectory for pipeline validation without GPU.
    """

    def __init__(self, task: ResearchTask):
        self.task = task
        self.gt_wit_id = task.ground_truth_wit_id
        self._phase = "search"  # search → open → cite → submit
        self._cited = False

    def decide(self, obs_available_actions: list[str],
               search_results: list = None) -> dict:

        if self._phase == "search" and "search" in obs_available_actions:
            self._phase = "open"
            return {"action": "search", "params": {"query_text": self.task.goal}}

        if self._phase == "open" and "open_result" in obs_available_actions and search_results:
            self._phase = "cite"
            for sr in search_results:
                if sr.wit_id == self.gt_wit_id:
                    return {"action": "open_result", "params": {"result_id": sr.result_id}}
            return {"action": "open_result", "params": {"result_id": search_results[0].result_id}}

        if self._phase == "cite" and "cite_source" in obs_available_actions:
            self._phase = "submit"
            self._cited = True
            return {"action": "cite_source", "params": {"evidence_text": "Evidence from document"}}

        if self._phase == "submit" and "submit_report" in obs_available_actions:
            return {"action": "submit_report", "params": {"report_text": "Research findings."}}

        # If submit not yet available (need to go back to results page first)
        if self._phase == "submit" and "back_to_results" in obs_available_actions:
            return {"action": "back_to_results", "params": {}}

        # Fallback
        return {"action": obs_available_actions[0], "params": {}}


# ═══════════════════════════════════════════════════════════
# Failure classification (research-specific)
# ═══════════════════════════════════════════════════════════

def classify_failure(r: EvalResult) -> str:
    if r.success:
        return "Success"
    if r.parse_error_count > 0:
        return "FormatOrActionError"
    if r.invalid_action_count > 0:
        return "FormatOrActionError"
    if not r.gt_in_candidates:
        return "GTNotInCandidates"
    if not r.opened_gt_anytime:
        return "NeverOpenedGT"
    if r.opened_gt_anytime and not r.cited_gt:
        return "OpenedGTButNeverCited"
    if r.cited_gt and not r.report_submitted:
        return "CitedButNotSubmitted"
    if r.report_submitted and r.coverage < 0.5:
        return "SubmittedWithLowCoverage"
    return "NeedsManualReview"


# ═══════════════════════════════════════════════════════════
# Core eval loop
# ═══════════════════════════════════════════════════════════

def eval_one_task(
    task: ResearchTask,
    model=None,
    processor=None,
    rag=None,
    images_dir: Path | None = None,
    max_steps: int = 12,
    inject_ground_truth: bool = True,
    search_controller: Any = None,
    mock: bool = False,
    enable_zoom: bool = False,
) -> EvalResult:
    """Evaluate one research task."""
    from searcheyes.rl_adapter import RLEnvironment

    result = EvalResult(
        task_id=task.task_id,
        difficulty=task.difficulty.value,
        gt_wit_id=task.ground_truth_wit_id or "",
        goal=task.goal,
    )

    oracle = MockOracleAgent(task) if mock else None

    try:
        env = RLEnvironment(
            task, rag=rag, images_dir=images_dir,
            inject_ground_truth=inject_ground_truth,
            search_controller=search_controller,
        )
        env.enable_zoom_search = enable_zoom
        obs = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            step_log = StepLog(step_idx=step)
            step_log.available_actions = list(obs.available_actions)

            # ── Decide action ──
            if mock:
                payload = oracle.decide(
                    obs.available_actions,
                    search_results=env.engine.search_results,
                )
            else:
                prompt_text = build_prompt(
                    goal=task.goal,
                    state_description=obs.state_description,
                    available_actions=obs.available_actions,
                )
                action_text = generate_action(model, processor, obs.screenshot_path, prompt_text)
                payload = parse_action(action_text)

            if payload is None:
                step_log.parse_error = True
                step_log.action = "parse_error"
                result.parse_error_count += 1
                result.final_action = "parse_error"
                result.step_logs.append(step_log)
                episode_reward -= 1.0
                break

            action = str(payload.get("action", "")).strip()
            params = payload.get("params", {}) or {}
            result.final_action = action

            step_log.action = action
            step_log.params = {k: v for k, v in params.items() if k != "query_image"}

            if action not in obs.available_actions:
                step_log.is_invalid_action = True
                result.invalid_action_count += 1

            step_result = env.step(action, params)
            episode_reward += step_result.reward
            obs = step_result.obs

            # ── Collect step-level data ──
            step_log.current_page = env.state.current_page
            step_log.phase = env.state.current_phase.value
            step_log.reward = step_result.reward
            step_log.done = step_result.done

            # Track open_result → detect GT opening
            if action == "open_result" and env.engine.current_document:
                doc = env.engine.current_document
                step_log.opened_result_id = doc.result_id
                step_log.opened_wit_id = doc.wit_id
                if doc.wit_id == result.gt_wit_id:
                    result.opened_gt_anytime = True
                    result.opened_gt_step_indices.append(step)

            # Track cite_source
            if action == "cite_source" and env.engine.citations:
                latest = env.engine.citations[-1]
                step_log.cited_in_this_step = True
                if latest.source_wit_id == result.gt_wit_id:
                    result.cited_gt = True

            # Track search → candidates
            if action == "search" and env.engine.search_results:
                result.search_trigger_count += 1
                wit_ids = [sr.wit_id for sr in env.engine.search_results]
                titles = [sr.title for sr in env.engine.search_results]
                step_log.candidate_wit_ids = wit_ids
                result.search_candidates_wit_ids = wit_ids
                result.search_candidates_titles = titles
                if result.gt_wit_id in wit_ids:
                    step_log.gt_rank = wit_ids.index(result.gt_wit_id)
                    result.gt_rank_in_candidates = step_log.gt_rank
                info = step_result.info
                if "search_controller" in info:
                    result.controller_strategy = info["search_controller"].get("accepted_strategy", "")
                    result.controller_retry_triggered = info["search_controller"].get("retry_triggered", False)

            if action == "zoom":
                result.used_zoom = True

            # Track submit_report
            if action == "submit_report":
                result.report_submitted = True

            result.step_logs.append(step_log)

            if step_result.done:
                break

        # ── Post-processing metrics ──
        result.steps = env.step_count
        result.episode_reward = episode_reward
        result.cited_wit_ids = [c.source_wit_id for c in env.engine.citations]
        result.citation_count = len(env.engine.citations)
        result.coverage = task.fact_set.coverage(env.engine.citations)

        # L1: GT in search results
        if result.gt_wit_id and env.engine.search_results:
            result.gt_in_candidates = any(
                sr.wit_id == result.gt_wit_id for sr in env.engine.search_results
            )
        result.oracle_success = result.gt_in_candidates

        # L3: end-to-end success = submitted + coverage >= 0.5
        result.success = result.report_submitted and result.coverage >= 0.5
        result.vlm_success = result.success

        result.failure_category = classify_failure(result)

    except Exception as e:
        import traceback
        result.error = str(e)
        result.failure_category = classify_failure(result)
        if "--verbose" in sys.argv:
            traceback.print_exc()

    return result


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

ENABLE_ZOOM_DEFAULT = False

def main():
    parser = argparse.ArgumentParser(description="v2 Research World Eval")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--n-tasks", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--chroma-db-path", type=str, default="data/wit_kb_v2/chroma_db")
    parser.add_argument("--embedding-server-url", type=str, default="http://localhost:8766")
    parser.add_argument("--wit-images-dir", type=str, default="data/wit_kb_v2/images")
    parser.add_argument("--task-jsonl", type=str, default="data/tasks/research_tasks_v2.jsonl")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--no-gt-injection", action="store_true",
                        help="Disable GT injection, test real retrieval (three-layer eval)")
    parser.add_argument("--use-search-controller", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true",
                        help="Mock mode: scripted oracle + mock RAG, no GPU needed")
    parser.add_argument("--enable-zoom", action="store_true", default=ENABLE_ZOOM_DEFAULT, help="Enable zoom_search action in FSM (default off)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # ── Init RAG (or mock) ──
    rag = None
    images_dir = None
    if not args.mock:
        from searcheyes.multimodal_rag import MultimodalRAG, RagConfig
        rag_config = RagConfig(
            chroma_db_path=args.chroma_db_path,
            embedding_server_url=args.embedding_server_url,
            collection_name="wit_knowledge_v2_qwen",
            top_k=20,
        )
        rag = MultimodalRAG(rag_config)
        images_dir = Path(args.wit_images_dir)
        print(f"RAG initialized: {args.chroma_db_path}")
    else:
        images_dir = Path(args.wit_images_dir) if Path(args.wit_images_dir).exists() else None
        rag = MockRAG()
        print("Mock mode: scripted oracle + MockRAG (no GPU)")

    # ── SearchController (optional) ──
    search_controller = None
    if args.use_search_controller and not args.mock:
        from searcheyes.search_controller import SearchController
        search_controller = SearchController()
        print("SearchController enabled")

    # ── Load tasks ──
    task_file = Path(args.task_jsonl)
    if not task_file.exists():
        print(f"Task file not found: {task_file}")
        sys.exit(1)

    all_tasks = []
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_tasks.append(ResearchTask.load_from_dict(json.loads(line)))
    print(f"Loaded {len(all_tasks)} tasks from {task_file}")

    # Balanced sampling
    rng = random.Random(args.seed)
    n = args.n_tasks
    easy = [t for t in all_tasks if t.difficulty == ResearchDifficulty.EASY]
    medium = [t for t in all_tasks if t.difficulty == ResearchDifficulty.MEDIUM]
    hard = [t for t in all_tasks if t.difficulty == ResearchDifficulty.HARD]
    n_each = max(1, n // 3)
    rng.shuffle(easy); rng.shuffle(medium); rng.shuffle(hard)
    eval_tasks = easy[:n_each] + medium[:n_each] + hard[:n_each]
    eval_tasks = eval_tasks[:n]
    print(f"Eval tasks: {len(eval_tasks)} "
          f"(easy={min(n_each, len(easy))}, "
          f"medium={min(n_each, len(medium))}, "
          f"hard={min(n_each, len(hard))})")

    # ── Load model (skip in mock mode) ──
    model, processor = None, None
    if not args.mock:
        if not args.model_dir:
            print("Error: --model-dir required when not using --mock")
            sys.exit(1)
        print(f"Loading model: {args.model_dir}")
        model, processor = load_model(args.model_dir)
        print("Model loaded")

    # ── Eval loop ──
    results: list[EvalResult] = []
    for i, task in enumerate(eval_tasks):
        tag = "mock" if args.mock else task.difficulty.value
        print(f"  [{i+1}/{len(eval_tasks)}] {task.task_id} ({tag}) ...", end=" ", flush=True)

        if args.mock and isinstance(rag, MockRAG):
            rag.set_task(task)

        r = eval_one_task(
            task, model, processor, rag, images_dir,
            max_steps=args.max_steps,
            inject_ground_truth=not args.no_gt_injection,
            search_controller=search_controller,
            mock=args.mock,
            enable_zoom=args.enable_zoom,
        )
        results.append(r)

        status = "OK" if r.success else "FAIL"
        l1 = "L1+" if r.gt_in_candidates else "L1-"
        cite_info = f"cites={r.citation_count} cov={r.coverage:.2f}"
        err_info = f" err={r.error[:50]}" if r.error else ""
        print(f"{status} {l1} steps={r.steps} act={r.final_action} {cite_info} rw={r.episode_reward:.2f}{err_info}")

    # ── Report ──
    total = len(results)
    if total == 0:
        print("No tasks evaluated.")
        return

    success_count = sum(1 for r in results if r.success)
    submitted_count = sum(1 for r in results if r.report_submitted)
    error_count = sum(1 for r in results if r.error)
    avg_steps = sum(r.steps for r in results) / total
    avg_reward = sum(r.episode_reward for r in results) / total
    avg_coverage = sum(r.coverage for r in results) / total
    avg_citations = sum(r.citation_count for r in results) / total

    gt_in_cand_count = sum(1 for r in results if r.gt_in_candidates)
    oracle_count = gt_in_cand_count

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Research Eval Report {'(MOCK)' if args.mock else ''}")
    print(sep)
    if not args.mock:
        print(f"  Model:         {args.model_dir}")
    print(f"  GT injection:  {'yes' if not args.no_gt_injection else 'no (real retrieval)'}")
    print(f"  Controller:    {'enabled' if search_controller else 'disabled'}")
    print(f"  Total tasks:   {total}")
    print(f"  -- Three-layer metrics --")
    print(f"  L1 Recall:     {gt_in_cand_count/total*100:.1f}% ({gt_in_cand_count}/{total})  [GT in search results]")
    print(f"  L2 Oracle:     {oracle_count/total*100:.1f}% ({oracle_count}/{total})  [GT citable]")
    print(f"  L3 VLM e2e:    {success_count/total*100:.1f}% ({success_count}/{total})  [submit + coverage >= 0.5]")
    print(f"  ────────────")
    print(f"  Submitted:     {submitted_count}/{total} ({submitted_count/total*100:.0f}%)")
    print(f"  Avg coverage:  {avg_coverage:.3f}")
    print(f"  Avg citations: {avg_citations:.1f}")
    print(f"  Avg steps:     {avg_steps:.1f}")
    print(f"  Avg reward:    {avg_reward:.3f}")
    if error_count:
        print(f"  Errors:        {error_count}")

    # By difficulty
    print(f"\n  By difficulty:")
    for diff_name in ["easy", "medium", "hard"]:
        group = [r for r in results if r.difficulty == diff_name]
        if not group:
            continue
        sr = sum(1 for r in group if r.success) / len(group) * 100
        cov = sum(r.coverage for r in group) / len(group)
        steps = sum(r.steps for r in group) / len(group)
        rw = sum(r.episode_reward for r in group) / len(group)
        print(f"    {diff_name:8s}: success={sr:.0f}%  coverage={cov:.2f}  steps={steps:.1f}  reward={rw:.2f}  (n={len(group)})")
    print(sep)

    # ── Failure buckets ──
    fail_count = total - success_count
    if fail_count > 0:
        print(f"\n  -- Failure buckets ({fail_count} failures) --")
        fail_dist = Counter(r.failure_category for r in results if not r.success)
        for cat, cnt in fail_dist.most_common():
            print(f"    {cat:30s}: {cnt} ({cnt/fail_count*100:.0f}%)")

    # ── SearchController metrics ──
    if search_controller:
        sc_metrics = search_controller.get_metrics()
        print(f"\n  -- SearchController --")
        print(f"  Total searches:  {sc_metrics['total_searches']}")
        print(f"  Retry rate:      {sc_metrics['retry_triggered_rate']*100:.1f}%")
        print(f"  Strategies:      {sc_metrics['strategy_distribution']}")

    # ── Save JSON ──
    if args.output_json:
        out = {
            "eval_type": "research_v2",
            "mock": args.mock,
            "model_dir": args.model_dir if not args.mock else "mock_oracle",
            "n_tasks": total,
            "success_rate": success_count / total,
            "avg_coverage": avg_coverage,
            "avg_citations": avg_citations,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "submitted_rate": submitted_count / total,
            "search_recall": gt_in_cand_count / total,
            "oracle_rate": oracle_count / total,
            "failure_distribution": dict(Counter(
                r.failure_category for r in results if not r.success
            )),
            "by_difficulty": {},
            "details": [],
        }
        for diff_name in ["easy", "medium", "hard"]:
            group = [r for r in results if r.difficulty == diff_name]
            if group:
                out["by_difficulty"][diff_name] = {
                    "count": len(group),
                    "success_rate": sum(1 for r in group if r.success) / len(group),
                    "avg_coverage": sum(r.coverage for r in group) / len(group),
                    "avg_steps": sum(r.steps for r in group) / len(group),
                    "avg_reward": sum(r.episode_reward for r in group) / len(group),
                }
        for r in results:
            out["details"].append({
                "task_id": r.task_id,
                "difficulty": r.difficulty,
                "goal": r.goal,
                "success": r.success,
                "failure_category": r.failure_category,
                "steps": r.steps,
                "final_action": r.final_action,
                "gt_wit_id": r.gt_wit_id,
                "cited_wit_ids": r.cited_wit_ids,
                "citation_count": r.citation_count,
                "coverage": r.coverage,
                "report_submitted": r.report_submitted,
                "episode_reward": r.episode_reward,
                "gt_in_candidates": r.gt_in_candidates,
                "gt_rank_in_candidates": r.gt_rank_in_candidates,
                "opened_gt_anytime": r.opened_gt_anytime,
                "opened_gt_step_indices": r.opened_gt_step_indices,
                "cited_gt": r.cited_gt,
                "search_candidates_wit_ids": r.search_candidates_wit_ids,
                "search_trigger_count": r.search_trigger_count,
                "used_zoom": r.used_zoom,
                "parse_error_count": r.parse_error_count,
                "invalid_action_count": r.invalid_action_count,
                "error": r.error,
                "step_logs": [
                    {
                        "step_idx": s.step_idx,
                        "current_page": s.current_page,
                        "phase": s.phase,
                        "action": s.action,
                        "params": s.params,
                        "available_actions": s.available_actions,
                        "is_invalid_action": s.is_invalid_action,
                        "opened_result_id": s.opened_result_id,
                        "opened_wit_id": s.opened_wit_id,
                        "cited_in_this_step": s.cited_in_this_step,
                        "candidate_wit_ids": s.candidate_wit_ids,
                        "gt_rank": s.gt_rank,
                        "reward": s.reward,
                        "done": s.done,
                        "parse_error": s.parse_error,
                    }
                    for s in r.step_logs
                ],
            })
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nResults saved: {args.output_json}")


if __name__ == "__main__":
    main()
