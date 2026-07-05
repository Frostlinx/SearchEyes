#!/usr/bin/env python3
"""
eval_rag.py — RAG+RL 模型评测（使用与训练相同的 prompt 格式）
================================================================
直接用 RLEnvironment + 模型推理，避免 AgentLoop 的 prompt 格式差异。

用法:
    # 模型评测
    python eval_rag.py --model-dir checkpoints/grpo_rag_run1/final \
        --chroma-db-path data/wit_subset_hf/chroma_db \
        --n-tasks 30 --output-json eval_rag_run1.json

    # 不同 checkpoint 对比
    python eval_rag.py --model-dir checkpoints/grpo_rag_run1/checkpoint-100 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class StepLog:
    step_idx: int = 0
    current_page: str = ""
    page_family: str = ""
    action: str = ""
    params: dict = field(default_factory=dict)
    available_actions: list = field(default_factory=list)
    selected_product_id: str = ""
    selected_wit_id: str = ""
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
    bought_wit_id: str = ""
    gt_wit_id: str = ""
    episode_reward: float = 0.0
    error: str = ""
    # 三层指标
    gt_in_candidates: bool = False   # L1: GT 是否进入 6 个候选（Recall@6）
    oracle_success: bool = False     # L2: 如果候选里有GT，理想选择器能否成功（= gt_in_candidates）
    vlm_success: bool = False        # L3: 端到端成功（= success）
    # ── failure analysis ──
    goal: str = ""
    final_selected_wit_id: str = ""
    parse_error_count: int = 0
    invalid_action_count: int = 0
    clicked_gt_anytime: bool = False
    clicked_gt_step_indices: list = field(default_factory=list)
    search_candidates_wit_ids: list = field(default_factory=list)
    search_candidates_titles: list = field(default_factory=list)
    gt_rank_in_candidates: int = -1
    search_trigger_count: int = 0
    used_zoom: bool = False
    controller_strategy: str = ""
    controller_retry_triggered: bool = False
    step_logs: list = field(default_factory=list)
    failure_category: str = ""


def load_model(model_dir: str):
    """加载模型和 processor，与训练时相同的方式。"""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    # 检查是否是 LoRA adapter
    adapter_config = Path(model_dir) / "adapter_config.json"
    if adapter_config.exists():
        print(f"  加载 LoRA adapter: {model_dir}")
        config_data = json.loads(adapter_config.read_text())
        base_path = config_data.get("base_model_name_or_path", "")
        if not base_path:
            raise ValueError("adapter_config.json 缺少 base_model_name_or_path")
        print(f"  Base model: {base_path}")
        # 加载 base model + LoRA adapter（不做 merge 以节省显存）
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_path, dtype=torch.bfloat16,
        ).to("cuda")
        model = PeftModel.from_pretrained(base_model, model_dir)
        processor = AutoProcessor.from_pretrained(base_path)
    else:
        print(f"  加载全参数模型: {model_dir}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16,
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(model_dir)

    model.eval()
    return model, processor


def generate_action(model, processor, image_path: str, prompt_text: str, max_new_tokens: int = 128) -> str:
    """用模型生成一个 action JSON。"""
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
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # 只取生成部分
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    result = processor.decode(generated_ids, skip_special_tokens=True)
    return result.strip()


def parse_action(text: str) -> dict | None:
    """从模型输出提取 JSON action。"""
    import re
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取 JSON 块
    match = re.search(r'\{[^{}]*"action"[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def build_prompt(goal: str, state_description: str, available_actions: list[str]) -> str:
    """构建与训练时相同格式的 prompt。"""
    actions_text = ", ".join(available_actions)
    return (
        f"Goal: {goal}\n"
        f"State: {state_description}\n"
        f"Available actions: {actions_text}\n"
        "Choose exactly one next action.\n"
        'Return JSON only in the form {"action": "...", "params": {...}}.'
    ).strip()


def build_feedback(result) -> str:
    """构建与训练时相同格式的环境反馈。"""
    available_actions = ", ".join(result.obs.available_actions)
    if result.info.get("success"):
        status = "Success! Task objective met."
    elif result.done:
        status = "Task failed."
    else:
        status = "Action executed."

    return (
        f"Observation: {result.obs.state_description}\n"
        f"Status: {status}\n"
        f"Available actions: {available_actions}"
    ).strip()


def classify_failure(r: EvalResult) -> str:
    """将失败 case 分桶（弱标签，部分类别需人工复核）。"""
    if r.success:
        return "Success"
    if r.parse_error_count > 0 or r.final_action.startswith("parse_error"):
        return "FormatOrActionError"
    if r.invalid_action_count > 0:
        return "FormatOrActionError"
    if not r.gt_in_candidates:
        return "NeverClickedGT_NotInCandidates"
    if not r.clicked_gt_anytime:
        return "NeverClickedGT"
    if r.clicked_gt_anytime and r.final_action in ("buy", "add_cart"):
        return "ClickedGTButFinalWrong"
    if not r.used_zoom:
        return "PossibleNeedsZoomOrCrop"
    return "NeedsManualReview_AfterZoom"


def eval_one_task(
    task,
    model,
    processor,
    rag,
    images_dir: Path | None,
    max_steps: int = 8,
    inject_ground_truth: bool = True,
    search_controller: Any = None,
    include_candidate_text: bool = False,
) -> EvalResult:
    """评测单个任务。"""
    from searcheyes.rl_adapter import RLEnvironment

    result = EvalResult(
        task_id=task.task_id,
        difficulty=task.difficulty.value,
        gt_wit_id=getattr(task, "ground_truth_wit_id", ""),
    )

    try:
        env = RLEnvironment(task, rag=rag, images_dir=images_dir,
                            inject_ground_truth=inject_ground_truth,
                            search_controller=search_controller,
                            include_candidate_text=include_candidate_text)
        obs = env.reset()

        result.goal = task.goal
        episode_reward = 0.0

        for step in range(max_steps):
            prompt_text = build_prompt(
                goal=task.goal,
                state_description=obs.state_description,
                available_actions=obs.available_actions,
            )
            image_path = obs.screenshot_path

            action_text = generate_action(model, processor, image_path, prompt_text)
            payload = parse_action(action_text)

            step_log = StepLog(step_idx=step)
            step_log.available_actions = list(obs.available_actions)

            if payload is None:
                step_log.parse_error = True
                step_log.action = f"parse_error: {action_text[:60]}"
                result.parse_error_count += 1
                result.final_action = step_log.action
                result.step_logs.append(step_log)
                episode_reward -= 1.0
                break

            action = str(payload.get("action", "")).strip()
            params = payload.get("params", {}) or {}
            result.final_action = action

            step_log.action = action
            step_log.params = {k: v for k, v in params.items()
                               if k not in ("query_image",)}

            # invalid action 检测（env.step 前）
            if action not in obs.available_actions:
                step_log.is_invalid_action = True
                result.invalid_action_count += 1

            step_result = env.step(action, params)
            episode_reward += step_result.reward
            obs = step_result.obs

            # ── 收集 step 级数据 ──
            step_log.current_page = getattr(env.state, "current_page", "")
            step_log.page_family = getattr(env.state, "page_family", "")
            step_log.reward = step_result.reward
            step_log.done = step_result.done
            step_log.selected_product_id = getattr(env.state, "selected_product_id", "") or ""

            if step_log.selected_product_id and env.engine.products:
                prod = env.engine.products.get(step_log.selected_product_id, {})
                step_log.selected_wit_id = prod.get("wit_id", "")

            # search 后收集候选
            if action == "search" and env.engine.products:
                result.search_trigger_count += 1
                wit_ids = [p.get("wit_id", "") for p in env.engine.products.values()]
                titles = [p.get("name", "") for p in env.engine.products.values()]
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

            # GT 点击检测：仅 click_product 命中 GT 才算
            if (action == "click_product"
                    and step_log.selected_wit_id == result.gt_wit_id
                    and result.gt_wit_id):
                result.clicked_gt_anytime = True
                result.clicked_gt_step_indices.append(step)

            result.step_logs.append(step_log)

            if step_result.done:
                break

        # ── 后处理指标 ──
        result.steps = env.step_count
        result.episode_reward = episode_reward

        if result.gt_wit_id and env.engine.products:
            result.gt_in_candidates = any(
                p.get("wit_id") == result.gt_wit_id
                for p in env.engine.products.values()
            )
        result.oracle_success = result.gt_in_candidates

        if result.final_action in ("buy", "add_cart"):
            pid = env.state.selected_product_id
            product = env.engine.products.get(pid, {})
            result.bought_wit_id = product.get("wit_id", "")
            result.final_selected_wit_id = result.bought_wit_id
            if result.gt_wit_id:
                result.success = result.bought_wit_id == result.gt_wit_id
            else:
                result.success = True
        result.vlm_success = result.success

        result.failure_category = classify_failure(result)

    except Exception as e:
        result.error = str(e)
        result.failure_category = classify_failure(result)

    return result


def main():
    parser = argparse.ArgumentParser(description="RAG+RL 模型评测")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--n-tasks", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--chroma-db-path", type=str, default="data/wit_subset_hf/chroma_db")
    parser.add_argument("--embedding-server-url", type=str, default="http://localhost:8000")
    parser.add_argument("--wit-images-dir", type=str, default="data/wit_subset_hf/images")
    parser.add_argument("--task-jsonl", type=str, default="data/tasks/rag_tasks.jsonl")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--no-gt-injection", action="store_true",
                        help="关闭 GT 注入，测真实检索能力（三层指标评测）")
    parser.add_argument("--use-search-controller", action="store_true",
                        help="启用 SearchController（judge + retry 闭环）")
    parser.add_argument("--seed", type=int, default=42, help="用于采样 eval 任务的种子")
    parser.add_argument("--include-candidate-text", action="store_true",
                        help="在 state_description 中加入候选产品文本列表")
    args = parser.parse_args()

    # ── 初始化 RAG ──
    from searcheyes.multimodal_rag import MultimodalRAG, RagConfig
    rag_config = RagConfig(
        chroma_db_path=args.chroma_db_path,
        embedding_server_url=args.embedding_server_url,
    )
    rag = MultimodalRAG(rag_config)
    images_dir = Path(args.wit_images_dir)
    print(f"RAG 已初始化: {args.chroma_db_path}")

    # ── SearchController（可选） ──
    search_controller = None
    if args.use_search_controller:
        from searcheyes.search_controller import SearchController
        search_controller = SearchController()
        print("SearchController 已启用（judge + retry 闭环）")

    # ── 加载任务 ──
    from searcheyes.task_schema import VisualTask, DifficultyLevel
    task_file = Path(args.task_jsonl)
    if not task_file.exists():
        print(f"任务文件不存在: {task_file}")
        sys.exit(1)

    all_tasks = []
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_tasks.append(VisualTask.load_from_dict(json.loads(line)))
    print(f"加载 {len(all_tasks)} 个任务")

    # 按难度均衡采样
    import random
    rng = random.Random(args.seed)
    n = args.n_tasks
    easy = [t for t in all_tasks if t.difficulty == DifficultyLevel.EASY]
    medium = [t for t in all_tasks if t.difficulty == DifficultyLevel.MEDIUM]
    hard = [t for t in all_tasks if t.difficulty == DifficultyLevel.HARD]
    n_each = max(1, n // 3)
    rng.shuffle(easy); rng.shuffle(medium); rng.shuffle(hard)
    eval_tasks = easy[:n_each] + medium[:n_each] + hard[:n_each]
    eval_tasks = eval_tasks[:n]
    print(f"评测任务: {len(eval_tasks)} 个 "
          f"(easy={min(n_each, len(easy))}, "
          f"medium={min(n_each, len(medium))}, "
          f"hard={min(n_each, len(hard))})")

    # ── 加载模型 ──
    print(f"加载模型: {args.model_dir}")
    model, processor = load_model(args.model_dir)
    print("模型加载完成")

    # ── 逐任务评测 ──
    results: list[EvalResult] = []
    for i, task in enumerate(eval_tasks):
        print(f"  [{i+1}/{len(eval_tasks)}] {task.task_id} ({task.difficulty.value}) ...", end=" ", flush=True)
        r = eval_one_task(task, model, processor, rag, images_dir, max_steps=args.max_steps,
                          inject_ground_truth=not args.no_gt_injection,
                          search_controller=search_controller,
                          include_candidate_text=args.include_candidate_text)
        results.append(r)
        status = "OK" if r.success else "FAIL"
        l1 = "L1✓" if r.gt_in_candidates else "L1✗"
        wit_info = ""
        if r.bought_wit_id or r.gt_wit_id:
            match = "match" if r.bought_wit_id == r.gt_wit_id else "mismatch"
            wit_info = f" wit={match}"
        err_info = f" err={r.error[:50]}" if r.error else ""
        print(f"{status} {l1} steps={r.steps} act={r.final_action}{wit_info} rw={r.episode_reward:.2f}{err_info}")

    # ── 报告 ──
    total = len(results)
    success_count = sum(1 for r in results if r.success)
    bought_count = sum(1 for r in results if r.final_action in ("buy", "add_cart"))
    error_count = sum(1 for r in results if r.error)
    avg_steps = sum(r.steps for r in results) / total if total else 0
    avg_reward = sum(r.episode_reward for r in results) / total if total else 0

    gt_in_cand_count = sum(1 for r in results if r.gt_in_candidates)
    oracle_count = gt_in_cand_count  # L2 = L1 in this formulation

    print(f"\n{'='*60}")
    print(f"评测报告 (RAG+RL)")
    print(f"{'='*60}")
    print(f"  模型:        {args.model_dir}")
    print(f"  GT注入:      {'是' if not args.no_gt_injection else '否（无拐杖）'}")
    print(f"  Controller:  {'启用' if search_controller else '未启用'}")
    print(f"  总任务数:    {total}")
    print(f"  ── 三层指标 ──")
    print(f"  L1 Recall@6: {gt_in_cand_count/total*100:.1f}% ({gt_in_cand_count}/{total})  [GT进入候选]")
    print(f"  L2 Oracle:   {oracle_count/total*100:.1f}% ({oracle_count}/{total})  [候选中有GT=可解]")
    print(f"  L3 VLM成功:  {success_count/total*100:.1f}% ({success_count}/{total})  [端到端成功]")
    print(f"  ────────────")
    print(f"  执行 buy:    {bought_count}/{total} ({bought_count/total*100:.0f}%)")
    print(f"  平均步数:    {avg_steps:.1f}")
    print(f"  平均 reward: {avg_reward:.3f}")
    if error_count:
        print(f"  错误数:      {error_count}")

    # 按难度
    print(f"\n  按难度分布:")
    for diff_name in ["easy", "medium", "hard"]:
        group = [r for r in results if r.difficulty == diff_name]
        if not group:
            continue
        sr = sum(1 for r in group if r.success) / len(group) * 100
        steps = sum(r.steps for r in group) / len(group)
        rw = sum(r.episode_reward for r in group) / len(group)
        print(f"    {diff_name:8s}: 成功率={sr:.0f}%  步数={steps:.1f}  reward={rw:.2f}  (n={len(group)})")
    print(f"{'='*60}")

    # ── SearchController 指标 ──
    if search_controller:
        sc_metrics = search_controller.get_metrics()
        print(f"\n  ── SearchController 指标 ──")
        print(f"  总搜索次数:     {sc_metrics['total_searches']}")
        print(f"  retry 触发率:   {sc_metrics['retry_triggered_rate']*100:.1f}% "
              f"({sc_metrics['retry_triggered_count']}/{sc_metrics['total_searches']})")
        print(f"  策略分布:       {sc_metrics['strategy_distribution']}")
        print(f"  质量提升(avg):  {sc_metrics['quality_improvement_avg']:.4f}")

    # ── Failure 分桶 ──
    fail_count = total - success_count
    if fail_count > 0:
        print(f"\n  ── Failure 分桶 ({fail_count} failures) ──")
        fail_dist = Counter(r.failure_category for r in results if not r.success)
        for cat, cnt in fail_dist.most_common():
            print(f"    {cat:40s}: {cnt} ({cnt/fail_count*100:.0f}%)")

    # ── 保存 JSON ──
    if args.output_json:
        out = {
            "model_dir": args.model_dir,
            "n_tasks": total,
            "success_rate": success_count / total if total else 0,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "bought_rate": bought_count / total if total else 0,
            "search_recall_at_6": gt_in_cand_count / total if total else 0,
            "oracle_on_top6": oracle_count / total if total else 0,
            "failure_distribution": dict(Counter(
                r.failure_category for r in results if not r.success
            )),
            "by_difficulty": {},
            "details": [],
        }
        if search_controller:
            out["controller_metrics"] = search_controller.get_metrics()
        for diff_name in ["easy", "medium", "hard"]:
            group = [r for r in results if r.difficulty == diff_name]
            if group:
                out["by_difficulty"][diff_name] = {
                    "count": len(group),
                    "success_rate": sum(1 for r in group if r.success) / len(group),
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
                "bought_wit_id": r.bought_wit_id,
                "final_selected_wit_id": r.final_selected_wit_id,
                "episode_reward": r.episode_reward,
                "gt_in_candidates": r.gt_in_candidates,
                "gt_rank_in_candidates": r.gt_rank_in_candidates,
                "clicked_gt_anytime": r.clicked_gt_anytime,
                "clicked_gt_step_indices": r.clicked_gt_step_indices,
                "search_candidates_wit_ids": r.search_candidates_wit_ids,
                "search_candidates_titles": r.search_candidates_titles,
                "search_trigger_count": r.search_trigger_count,
                "used_zoom": r.used_zoom,
                "parse_error_count": r.parse_error_count,
                "invalid_action_count": r.invalid_action_count,
                "controller_strategy": r.controller_strategy,
                "controller_retry_triggered": r.controller_retry_triggered,
                "error": r.error,
                "step_logs": [
                    {
                        "step_idx": s.step_idx,
                        "current_page": s.current_page,
                        "page_family": s.page_family,
                        "action": s.action,
                        "params": s.params,
                        "available_actions": s.available_actions,
                        "is_invalid_action": s.is_invalid_action,
                        "selected_product_id": s.selected_product_id,
                        "selected_wit_id": s.selected_wit_id,
                        "candidate_wit_ids": s.candidate_wit_ids,
                        "gt_rank": s.gt_rank,
                        "reward": s.reward,
                        "done": s.done,
                        "parse_error": s.parse_error,
                    }
                    for s in r.step_logs
                ],
            })
        Path(args.output_json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已保存: {args.output_json}")


if __name__ == "__main__":
    main()
