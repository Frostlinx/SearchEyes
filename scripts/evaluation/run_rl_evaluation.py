"""
第五阶段：RL 闭环评估与训练模拟
================================
1. 对第四阶段生成的 20 条轨迹进行 Reward 评估
2. 模拟 GRPO 训练过程
3. 生成训练曲线和奖励分布图表
4. 输出详尽的验证报告数据

运行: python3 run_rl_evaluation.py
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sandbox.rl_env import evaluate_trajectory, simulate_grpo_training, RolloutResult

TRAJ_DIR = Path(__file__).parent / "output" / "trajectories"
OUTPUT_DIR = Path(__file__).parent / "output" / "rl_evaluation"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    print("=" * 70)
    print("🎮 第五阶段：RL 闭环评估与训练模拟")
    print("=" * 70)

    # ================================================================
    # Step 1: 加载所有轨迹数据
    # ================================================================
    print("\n[Step 1] 加载轨迹数据...")
    traj_dirs = sorted(TRAJ_DIR.glob("trajectory_*"))
    all_metas = []
    for td in traj_dirs:
        meta_path = td / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                all_metas.append(json.load(f))
    print(f"  已加载 {len(all_metas)} 条轨迹")

    # ================================================================
    # Step 2: Reward 评估
    # ================================================================
    print("\n[Step 2] 逐条计算 Reward...")
    print(f"{'─' * 70}")
    print(f"{'ID':>4s} {'Task':<16s} {'Steps':>5s} {'Zoom':>4s} "
          f"{'VScore':>7s} {'LScore':>7s} {'Total':>7s} {'Dist(px)':>8s} {'Result'}")
    print(f"{'─' * 70}")

    rollout_results: list[RolloutResult] = []
    for meta in all_metas:
        result = evaluate_trajectory(meta)
        rollout_results.append(result)

        # 计算平均点击距离
        click_steps = [s for s in result.steps if s["click_distance_px"] >= 0]
        avg_dist = np.mean([s["click_distance_px"] for s in click_steps]) if click_steps else -1

        status = "✅🎯" if result.reached_correct_state and result.task_completed else \
                 "✅❌" if result.task_completed else "⏳"

        print(f"{result.trajectory_id:4d} {result.task_id:<16s} {result.total_steps:5d} "
              f"{result.total_zooms:4d} {result.avg_visual_score:7.4f} "
              f"{result.logic_score:7.4f} {result.total_reward:7.4f} "
              f"{avg_dist:8.1f} {status}")

    # ================================================================
    # Step 3: 汇总统计
    # ================================================================
    print(f"\n{'=' * 70}")
    print("[Step 3] Reward 分布统计")
    print(f"{'=' * 70}")

    all_total_rewards = [r.total_reward for r in rollout_results]
    all_visual = [r.avg_visual_score for r in rollout_results]
    all_logic = [r.logic_score for r in rollout_results]

    print(f"  Total Reward:  mean={np.mean(all_total_rewards):.4f}  "
          f"std={np.std(all_total_rewards):.4f}  "
          f"min={np.min(all_total_rewards):.4f}  max={np.max(all_total_rewards):.4f}")
    print(f"  Visual Score:  mean={np.mean(all_visual):.4f}  "
          f"std={np.std(all_visual):.4f}")
    print(f"  Logic Score:   mean={np.mean(all_logic):.4f}  "
          f"std={np.std(all_logic):.4f}")

    # 按任务分组统计
    print(f"\n  按任务分组:")
    task_groups = {}
    for r in rollout_results:
        task_groups.setdefault(r.task_id, []).append(r)

    for tid, results in task_groups.items():
        rewards = [r.total_reward for r in results]
        correct = sum(1 for r in results if r.reached_correct_state)
        print(f"    {tid:<16s}: n={len(results)}  "
              f"reward={np.mean(rewards):.4f}±{np.std(rewards):.4f}  "
              f"correct={correct}/{len(results)}")

    # ================================================================
    # Step 4: 模拟 GRPO 训练
    # ================================================================
    print(f"\n{'=' * 70}")
    print("[Step 4] 模拟 GRPO 训练 (10 epochs)...")
    print(f"{'=' * 70}")

    training_log = simulate_grpo_training(rollout_results, epochs=10)

    for entry in training_log:
        print(f"  Epoch {entry['epoch']:2d}: "
              f"mean_r={entry['mean_reward']:.4f}  "
              f"std={entry['std_reward']:.4f}  "
              f"positive_ratio={entry['positive_ratio']:.0%}  "
              f"sim_avg={entry['simulated_avg_reward']:.4f}")

    # ================================================================
    # Step 5: 生成可视化图表
    # ================================================================
    print(f"\n[Step 5] 生成可视化图表...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 5: RL Closed-Loop Evaluation & GRPO Training Simulation',
                 fontsize=14, fontweight='bold')

    # 5a: Reward 分布直方图
    ax = axes[0, 0]
    ax.hist(all_total_rewards, bins=10, color='#3498db', alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(all_total_rewards), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(all_total_rewards):.3f}')
    ax.set_xlabel('Total Reward', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(a) Reward Distribution Across Trajectories', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5b: Visual Score vs Logic Score 散点图
    ax = axes[0, 1]
    colors_map = {'buy_cheapest': '#e74c3c', 'buy_nike': '#3498db', 'buy_domestic': '#27ae60'}
    for r in rollout_results:
        c = colors_map.get(r.task_id, '#888')
        marker = '★' if r.reached_correct_state else 'o'
        ax.scatter(r.avg_visual_score, r.logic_score, c=c, s=80, alpha=0.8, edgecolors='white')
    # legend
    for tid, c in colors_map.items():
        ax.scatter([], [], c=c, s=60, label=tid)
    ax.set_xlabel('Average Visual Score', fontsize=11)
    ax.set_ylabel('Logic Score', fontsize=11)
    ax.set_title('(b) Visual vs Logic Score by Task', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5c: GRPO 训练曲线
    ax = axes[1, 0]
    epochs = [e["epoch"] for e in training_log]
    mean_rewards = [e["mean_reward"] for e in training_log]
    sim_rewards = [e["simulated_avg_reward"] for e in training_log]
    ax.plot(epochs, mean_rewards, 'o-', color='#e74c3c', linewidth=2, label='Batch Mean Reward')
    ax.plot(epochs, sim_rewards, 's--', color='#27ae60', linewidth=2, label='Simulated Policy Reward')
    ax.fill_between(epochs,
                    [m - s for m, s in zip(mean_rewards, [e["std_reward"] for e in training_log])],
                    [m + s for m, s in zip(mean_rewards, [e["std_reward"] for e in training_log])],
                    alpha=0.15, color='#e74c3c')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title('(c) GRPO Training Curve (Simulated)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5d: 逐步 Visual Score 热力图
    ax = axes[1, 1]
    max_steps = max(len(r.steps) for r in rollout_results)
    heatmap_data = np.full((len(rollout_results), max_steps), np.nan)
    for i, r in enumerate(rollout_results):
        for j, s in enumerate(r.steps):
            if s["visual_score"] > 0:
                heatmap_data[i, j] = s["visual_score"]

    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xlabel('Step Index', fontsize=11)
    ax.set_ylabel('Trajectory ID', fontsize=11)
    ax.set_title('(d) Per-Step Visual Score Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Visual Score')

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "rl_evaluation_charts.png"
    plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ 图表保存: {chart_path}")

    # ================================================================
    # Step 6: 保存完整评估报告 JSON
    # ================================================================
    report = {
        "summary": {
            "total_trajectories": len(rollout_results),
            "total_reward_mean": round(float(np.mean(all_total_rewards)), 4),
            "total_reward_std": round(float(np.std(all_total_rewards)), 4),
            "visual_score_mean": round(float(np.mean(all_visual)), 4),
            "logic_score_mean": round(float(np.mean(all_logic)), 4),
            "task_completion_rate": f"{sum(1 for r in rollout_results if r.task_completed)}/{len(rollout_results)}",
            "answer_correct_rate": f"{sum(1 for r in rollout_results if r.reached_correct_state)}/{len(rollout_results)}",
        },
        "reward_config": {
            "visual_weight_alpha": 0.3,
            "logic_weight_beta": 0.7,
            "visual_threshold_px": 50.0,
            "visual_formula": "exp(-dist^2 / (2 * threshold^2))",
            "logic_formula": "base * (0.5 + 0.5 * efficiency)",
        },
        "per_trajectory": [
            {
                "id": r.trajectory_id,
                "task": r.task_id,
                "visual": r.avg_visual_score,
                "logic": r.logic_score,
                "total": r.total_reward,
                "correct": r.reached_correct_state,
            }
            for r in rollout_results
        ],
        "training_log": training_log,
    }

    report_path = OUTPUT_DIR / "rl_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  ✅ 报告 JSON 保存: {report_path}")

    # ================================================================
    # Final Summary
    # ================================================================
    print(f"\n{'=' * 70}")
    print("🏁 第五阶段验证完成")
    print(f"{'=' * 70}")
    print(f"  Reward 公式: R = 0.3 × Visual + 0.7 × Logic")
    print(f"  Visual Score: 高斯衰减(点击距离, σ=50px)")
    print(f"  Logic Score:  阶梯(正确完成=1.0 / 到达=0.5 / 选错=0.2 / 失败=0.0)")
    print(f"  平均 Total Reward: {np.mean(all_total_rewards):.4f}")
    print(f"  GRPO 模拟 10 epochs: reward {training_log[0]['simulated_avg_reward']:.4f} → {training_log[-1]['simulated_avg_reward']:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
