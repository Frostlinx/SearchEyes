"""
bench_export.py — P4 VDR 风格评测协议导出
===========================================
将任务集导出为 Vision-DR / VDR benchmark 输入格式。
预留外部模型接入点。
"""

import json
from pathlib import Path
from searcheyes.task_schema import VisualTask
from dataclasses import asdict
from searcheyes.rl_adapter import RLEnvironment


BENCH_DIR = Path(__file__).parent.parent / "data" / "benchmark"


def load_tasks_from_jsonl(path: str | Path) -> list[VisualTask]:
    task_file = Path(path)
    tasks: list[VisualTask] = []
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(VisualTask.load_from_dict(json.loads(line)))
    return tasks


def export_vdr_format(tasks: list[VisualTask], output_name: str = "vdr_bench"):
    """导出为 VDR 风格的评测数据集"""
    out_dir = BENCH_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_items = []
    for task in tasks:
        item = {
            "id": task.task_id,
            "query": task.goal,
            "difficulty": task.difficulty.value,
            "visual_gap_count": task.visual_gap_count,
            "zoom_budget": task.zoom_budget,
            "dag_depth": task.dag_depth,
            "expected_answer": task.final_answer,
            "page_family_sequence": task.page_family_sequence,
            "difficulty_tags": task.difficulty_tags,
            "trajectory_length": len(task.ground_truth_trajectory),
        }
        bench_items.append(item)

    # 主文件
    meta = {
        "name": "Visual Agentic World Model Benchmark",
        "version": "0.1",
        "total_tasks": len(tasks),
        "difficulty_distribution": {
            "easy": sum(1 for t in tasks if t.difficulty.value == "easy"),
            "medium": sum(1 for t in tasks if t.difficulty.value == "medium"),
            "hard": sum(1 for t in tasks if t.difficulty.value == "hard"),
        },
        "metrics": [
            "visual_hit_rate",
            "click_coord_error",
            "success_rate",
            "average_steps",
            "zoom_usage_rate",
            "long_horizon_consistency",
        ]
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with open(out_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for item in bench_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 完整版 (含 trajectory)
    with open(out_dir / "tasks_full.jsonl", "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(asdict(task), ensure_ascii=False, default=str) + "\n")

    print(f"✅ VDR 评测集导出完成: {out_dir}")
    print(f"  tasks.jsonl:      {len(bench_items)} 条 (轻量版)")
    print(f"  tasks_full.jsonl: {len(tasks)} 条 (含轨迹)")
    print(f"  meta.json:        评测协议元数据")

    return out_dir


def export_rl_episode_jsonl(env: RLEnvironment, output_name: str = "", verl_format: bool = False) -> Path:
    """把单个 RL rollout 导出为训练样本 JSONL。"""
    episode_name = output_name or f"{env.task.task_id}_rollout"
    out_dir = BENCH_DIR / "verl_rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    return env.export_episode_jsonl(out_dir / f"{episode_name}.jsonl", verl_format=verl_format)


if __name__ == "__main__":
    # 从已生成的任务集加载
    task_file = Path(__file__).parent.parent / "data" / "tasks" / "visual_tasks.jsonl"
    if task_file.exists():
        tasks = load_tasks_from_jsonl(task_file)
        export_vdr_format(tasks)
    else:
        print("❌ 未找到任务集，请先运行 task_generator.py")
