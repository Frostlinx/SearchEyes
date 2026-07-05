#!/usr/bin/env python3
"""
run_e2_experiment.py — E2 规模化对比实验（纯计算模式，不渲染截图）
=================================================================
用 TransitionEngine + reward 逻辑直接计算，跳过 Playwright 渲染。
在 scripted GT 轨迹下跑 baseline (无 RAG) vs RAG (有 RAG) 对比。

用法:
    python scripts/run_e2_experiment.py \
        --standard-tasks data/tasks/visual_tasks.jsonl \
        --research-tasks data/tasks/visual_research_tasks.jsonl \
        --num-standard 500 --num-research 500 \
        --output-dir data/e2_results
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent))

from searcheyes.task_schema import VisualTask, DifficultyLevel, TrajectoryStep
from searcheyes.transition_engine import TransitionEngine, EnvState
from searcheyes.validator import Validator


def load_tasks(path: str, limit: int) -> list[VisualTask]:
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(VisualTask)}
    tasks = []
    for line in Path(path).read_text("utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw["difficulty"] = DifficultyLevel(raw["difficulty"])
        raw["ground_truth_trajectory"] = [TrajectoryStep(**s) for s in raw.get("ground_truth_trajectory", [])]
        raw = {k: v for k, v in raw.items() if k in valid_fields}
        tasks.append(VisualTask(**raw))
        if len(tasks) >= limit:
            break
    return tasks


# ── 轻量 reward 计算（不依赖 Playwright）──────────────────

def _state_matches(state: EnvState, expected_state_str: str) -> bool:
    return state.current_page == expected_state_str


def _params_match(expected: dict, actual: dict) -> bool:
    if not expected:
        return True
    return all(actual.get(k) == v for k, v in expected.items())


def _is_terminal_success(expected_step, action: str, diff) -> bool:
    if expected_step is None:
        return False
    if action in ("buy", "add_cart") and action == expected_step.action:
        return diff.has_data_change
    return False


def compute_step_reward(
    prev_state: EnvState, action: str, params: dict,
    diff, new_state: EnvState, validation_passed: bool,
    expected_step: TrajectoryStep | None,
) -> tuple[float, dict]:
    """单步 reward 计算（从 rl_adapter._compute_reward 提取）"""
    reward = -0.1
    alignment = {
        "expected_action": expected_step.action if expected_step else "",
        "matched_action": False, "matched_params": False, "matched_state": False,
    }

    if expected_step is None:
        reward -= 0.2
    else:
        state_match = _state_matches(prev_state, expected_step.state)
        action_match = action == expected_step.action
        param_match = action_match and _params_match(expected_step.action_params, params)
        alignment["matched_state"] = state_match
        alignment["matched_action"] = action_match
        alignment["matched_params"] = param_match

        reward += 0.15 if state_match else -0.25
        reward += 0.7 if action_match else -0.55
        if param_match:
            reward += 0.45
        elif action_match and expected_step.action_params:
            reward -= 0.2

        if expected_step.requires_zoom:
            reward += 0.1 if action == "zoom" else -0.08

        if action_match and diff.events:
            reward += 0.2
        elif action != "zoom" and not diff.events and action_match:
            reward -= 0.15

    if not validation_passed:
        reward -= 0.8
    if action == "buy":
        reward += 0.35 if diff.has_data_change else -0.35
    if action == "add_cart":
        reward += 0.2 if diff.has_data_change else -0.15
    if _is_terminal_success(expected_step, action, diff):
        reward += 0.8

    reward = max(-1.5, min(2.0, reward))
    return reward, alignment


def compute_rag_reward(action: str, rag_facts: list[dict] | None,
                       gt_fact: str, gt_wit_id: str) -> float:
    """RAG reward 计算（从 rl_adapter.compute_rag_reward 提取）"""
    if rag_facts is None:
        return 0.0
    if not rag_facts:
        return -0.1
    if gt_wit_id:
        for fact in rag_facts:
            if fact.get("wit_id", "") == gt_wit_id:
                return 0.6 if action == "zoom" else 0.5
    if not gt_fact:
        return -0.1
    gt_tokens = {t for t in gt_fact.lower().split() if len(t) > 2}
    if not gt_tokens:
        return -0.1
    best_overlap = 0.0
    for fact in rag_facts:
        text = fact.get("caption", "")
        tokens = {t for t in text.lower().split() if len(t) > 2}
        if tokens:
            best_overlap = max(best_overlap, len(gt_tokens & tokens) / len(gt_tokens))
    if best_overlap > 0.5:
        return 0.3
    elif best_overlap > 0.2:
        return 0.1
    return -0.3


# ── 轻量 replay ──────────────────────────────────────────

def make_rag_facts(task: VisualTask) -> list[dict] | None:
    """为 RAG 条件生成模拟 facts（不需要 embedding server）"""
    if not task.requires_rag:
        return [
            {"wit_id": "wit_0042", "caption": "A sample cross-domain image"},
            {"wit_id": "wit_0100", "caption": "Another unrelated image"},
        ]
    else:
        return [
            {"wit_id": task.ground_truth_wit_id, "caption": task.ground_truth_caption},
            {"wit_id": "wit_0999", "caption": "A distractor image"},
            {"wit_id": "wit_0500", "caption": "Another distractor"},
        ]


def replay_task(task: VisualTask, with_rag: bool) -> dict:
    """用 GT 轨迹轻量回放，只计算 reward 不渲染。"""
    engine = TransitionEngine()
    validator = Validator(engine)
    state = EnvState()

    rewards = []
    rag_rewards = []
    gt_steps = [s for s in task.ground_truth_trajectory if s.action != "observe"]
    step_count = 0

    for i, gt_step in enumerate(gt_steps):
        action = gt_step.action
        params = gt_step.action_params or {}

        prev_state = state
        import copy
        new_state = copy.deepcopy(state)
        new_state, diff = engine.step(state, action, params)
        vr = validator.validate(new_state, diff)

        # Base reward
        reward, alignment = compute_step_reward(
            prev_state, action, params, diff, new_state, vr.passed, gt_step,
        )

        # RAG reward
        rag_facts = make_rag_facts(task) if with_rag else None
        gt_fact = task.ground_truth_caption if task.requires_rag else ""
        gt_wit_id = task.ground_truth_wit_id if task.requires_rag else ""
        rag_rwd = compute_rag_reward(action, rag_facts, gt_fact, gt_wit_id)

        total = max(-1.5, min(2.0, reward + rag_rwd))
        rewards.append(total)
        rag_rewards.append(rag_rwd)

        state = new_state
        step_count += 1

        done = action in ("buy", "add_cart") or not vr.passed
        if done:
            break

    last_action = gt_steps[-1].action if gt_steps else ""
    success = last_action in ("buy", "add_cart")

    return {
        "task_id": task.task_id,
        "difficulty": task.difficulty.value,
        "requires_rag": task.requires_rag,
        "steps": step_count,
        "reward_sum": sum(rewards),
        "reward_mean": mean(rewards) if rewards else 0,
        "rag_reward_sum": sum(rag_rewards),
        "rag_reward_mean": mean(rag_rewards) if rag_rewards else 0,
        "success": success,
        "rewards_per_step": rewards,
        "rag_rewards_per_step": rag_rewards,
    }


# ── 实验主循环 ──────────────────────────────────────────

def run_condition(tasks: list[VisualTask], with_rag: bool, label: str) -> list[dict]:
    results = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        try:
            result = replay_task(task, with_rag)
            results.append(result)
        except Exception as e:
            results.append({"task_id": task.task_id, "error": str(e), "success": False})
        if (i + 1) % 200 == 0:
            print(f"  [{label}] {i+1}/{len(tasks)} ({time.time()-t0:.1f}s)")
    print(f"  [{label}] {len(tasks)}/{len(tasks)} ({time.time()-t0:.1f}s)")
    return results


def compute_stats(results: list[dict], label: str) -> dict:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"label": label, "total": len(results), "valid": 0}

    rewards = [r["reward_sum"] for r in valid]
    rag_rewards = [r["rag_reward_sum"] for r in valid]
    successes = sum(1 for r in valid if r["success"])

    standard = [r for r in valid if not r.get("requires_rag")]
    research = [r for r in valid if r.get("requires_rag")]

    stats = {
        "label": label,
        "total": len(results),
        "valid": len(valid),
        "failed": len(results) - len(valid),
        "success_rate": successes / len(valid),
        "reward_mean": mean(rewards),
        "reward_std": stdev(rewards) if len(rewards) > 1 else 0,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "rag_reward_mean": mean(rag_rewards),
        "rag_reward_std": stdev(rag_rewards) if len(rag_rewards) > 1 else 0,
        "avg_steps": mean([r["steps"] for r in valid]),
    }

    for prefix, subset in [("standard", standard), ("research", research)]:
        if subset:
            stats[f"{prefix}_count"] = len(subset)
            stats[f"{prefix}_reward_mean"] = mean([r["reward_sum"] for r in subset])
            stats[f"{prefix}_reward_std"] = stdev([r["reward_sum"] for r in subset]) if len(subset) > 1 else 0
            stats[f"{prefix}_success_rate"] = sum(1 for r in subset if r["success"]) / len(subset)
            stats[f"{prefix}_rag_reward_mean"] = mean([r["rag_reward_sum"] for r in subset])

    for diff in ("easy", "medium", "hard"):
        subset = [r for r in valid if r.get("difficulty") == diff]
        if subset:
            stats[f"{diff}_count"] = len(subset)
            stats[f"{diff}_reward_mean"] = mean([r["reward_sum"] for r in subset])
            stats[f"{diff}_success_rate"] = sum(1 for r in subset if r["success"]) / len(subset)

    return stats


def print_comparison(bs: dict, rs: dict):
    print("\n" + "=" * 76)
    print("  E2 EXPERIMENT RESULTS: Baseline vs RAG (1000 tasks, scripted GT)")
    print("=" * 76)

    def row(name, key, fmt):
        bv = bs.get(key, 0)
        rv = rs.get(key, 0)
        d = rv - bv
        s = "+" if d > 0 else ""
        print(f"  {name:<35s} {bv:>12{fmt}} {rv:>12{fmt}} {s}{d:>10{fmt}}")

    print(f"  {'Metric':<35s} {'Baseline':>12s} {'With RAG':>12s} {'Delta':>10s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

    row("Total tasks", "total", "d")
    row("Success rate", "success_rate", ".2%")
    row("Reward mean", "reward_mean", ".3f")
    row("Reward std", "reward_std", ".3f")
    row("RAG reward mean", "rag_reward_mean", ".3f")
    row("Avg steps/task", "avg_steps", ".1f")

    print(f"\n  {'--- By task type ---':<35s}")
    for p, l in [("standard", "Standard"), ("research", "Visual Research")]:
        n = rs.get(f"{p}_count", 0)
        print(f"\n  {l + f' (n={n})':<35s}")
        row(f"  {l} reward", f"{p}_reward_mean", ".3f")
        row(f"  {l} success", f"{p}_success_rate", ".2%")
        row(f"  {l} RAG reward", f"{p}_rag_reward_mean", ".3f")

    print(f"\n  {'--- By difficulty ---':<35s}")
    for d in ("easy", "medium", "hard"):
        n = rs.get(f"{d}_count", 0)
        row(f"  {d.capitalize()} (n={n}) reward", f"{d}_reward_mean", ".3f")
        row(f"  {d.capitalize()} success", f"{d}_success_rate", ".2%")

    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="E2 scale experiment (no rendering)")
    parser.add_argument("--standard-tasks", default="data/tasks/visual_tasks.jsonl")
    parser.add_argument("--research-tasks", default="data/tasks/visual_research_tasks.jsonl")
    parser.add_argument("--num-standard", type=int, default=500)
    parser.add_argument("--num-research", type=int, default=500)
    parser.add_argument("--output-dir", default="data/e2_results")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[E2] Loading tasks...")
    standard = load_tasks(args.standard_tasks, args.num_standard)
    research = load_tasks(args.research_tasks, args.num_research)
    all_tasks = standard + research
    print(f"  Standard: {len(standard)}, Research: {len(research)}, Total: {len(all_tasks)}")

    print("\n[E2] Condition 1: BASELINE (no RAG)")
    t0 = time.time()
    baseline_results = run_condition(all_tasks, False, "baseline")
    bt = time.time() - t0

    print("\n[E2] Condition 2: WITH RAG")
    t0 = time.time()
    rag_results = run_condition(all_tasks, True, "with_rag")
    rt = time.time() - t0

    bs = compute_stats(baseline_results, "baseline")
    rs = compute_stats(rag_results, "with_rag")
    print_comparison(bs, rs)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "experiment": "E2_scale_comparison",
        "timestamp": stamp,
        "config": {"standard": len(standard), "research": len(research),
                   "total": len(all_tasks), "baseline_time": bt, "rag_time": rt},
        "baseline_stats": bs, "rag_stats": rs,
        "baseline_results": baseline_results, "rag_results": rag_results,
    }
    rp = out / f"e2_report_{stamp}.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    sp = out / f"e2_summary_{stamp}.json"
    sp.write_text(json.dumps({"baseline": bs, "with_rag": rs}, ensure_ascii=False, indent=2), "utf-8")

    print(f"\n[E2] Report: {rp}")
    print(f"[E2] Summary: {sp}")


if __name__ == "__main__":
    main()
