"""
run_batch_rollout.py — 批量 episode 收集入口
===========================================
支持两种数据模式：
1. scripted: 生成 GT demonstration，可用于 SFT 热启动
2. local:    生成本地模型 rollout，可用于 GRPO / RL 数据收集
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from searcheyes.bench_export import load_tasks_from_jsonl
from searcheyes.rl_adapter import RLEnvironment
from searcheyes.task_schema import VisualTask
from searcheyes.vlm_agent import LocalQwenVisionBackend
from searcheyes.agent_loop import run_agent_loop
from searcheyes.multimodal_rag import MultimodalRAG, RagConfig


def main():
    args = parse_args()
    tasks = load_tasks_from_jsonl(args.task_jsonl)
    selected = tasks[args.start_index : args.start_index + args.num_tasks]
    if not selected:
        raise SystemExit("没有可执行的任务，请检查 start-index / num-tasks")

    batch_dir = make_batch_dir(args.output_root, args.backend)
    samples_dir = batch_dir / ("sft_demos" if args.backend == "scripted" else "grpo_rollouts")
    samples_dir.mkdir(parents=True, exist_ok=True)

    shared_backend = build_shared_backend(args)

    # 初始化 RAG（如果配置了）
    rag = None
    if args.rag_db and not args.no_rag:
        rag = MultimodalRAG(RagConfig(
            chroma_db_path=args.rag_db,
            embedding_server_url=args.embedding_url,
        ))

    # WIT 图片注入路径
    wit_meta = args.wit_meta or None
    wit_images = args.wit_images or None
    if not wit_meta and args.rag_db and not args.no_rag:
        inferred = Path(args.rag_db).parent / "meta.jsonl"
        if inferred.exists():
            wit_meta = str(inferred)
            wit_images = str(inferred.parent / "images")

    results: list[dict[str, Any]] = []

    for offset, task in enumerate(selected, start=args.start_index):
        print(f"[{offset}] {task.task_id} | {task.goal}")
        try:
            if args.backend == "scripted":
                rollout = replay_ground_truth(task)
                steps = rollout["steps"]
                final_validation = "GT_REPLAY"
                episode_dir = str(rollout["env"].episode_dir) if rollout["env"].episode_dir else ""
            else:
                backend = shared_backend or build_backend_for_task(args, task)
                summary = run_agent_loop(
                    backend=backend, task=task, rag=rag,
                    wit_meta_jsonl=wit_meta, wit_images_dir=wit_images,
                )
                rollout = replay_episode(task, summary)
                steps = len(summary["steps"])
                final_validation = summary["steps"][-1]["validation"] if summary["steps"] else "NO_STEPS"
                episode_dir = summary["episode_dir"]

            export_path = rollout["env"].export_episode_jsonl(
                samples_dir / f"{task.task_id}.jsonl",
                verl_format=args.verl_format,
            )
            result = {
                "task_id": task.task_id,
                "task_index": offset,
                "episode_dir": episode_dir,
                "sample_path": str(export_path),
                "steps": steps,
                "reward_sum": rollout["reward_sum"],
                "success": rollout["success"],
                "final_validation": final_validation,
            }
            results.append(result)
            print(
                f"  steps={result['steps']} reward_sum={result['reward_sum']:.2f} "
                f"success={result['success']} sample={export_path.name}"
            )
        except Exception as exc:
            result = {
                "task_id": task.task_id,
                "task_index": offset,
                "error": str(exc),
                "success": False,
            }
            results.append(result)
            print(f"  ERROR: {exc}")

    summary = build_batch_summary(args.backend, results, batch_dir)
    summary_path = batch_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"backend:      {args.backend}")
    print(f"batch_dir:     {batch_dir}")
    print(f"summary:       {summary_path}")
    print(f"completed:     {summary['completed_tasks']}/{summary['total_tasks']}")
    print(f"success_rate:  {summary['success_rate']:.2%}")
    print(f"avg_reward:    {summary['average_reward']:.2f}")
    print(f"avg_steps:     {summary['average_steps']:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch rollout collection for searcheyes")
    parser.add_argument("--backend", choices=["scripted", "local"], default="scripted")
    parser.add_argument(
        "--task-jsonl",
        default="./data/tasks/visual_tasks.jsonl",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument(
        "--output-root",
        default="./data/benchmark/batch_rollouts",
    )
    parser.add_argument("--verl-format", action="store_true", help="导出标准 messages 格式 prompt")
    parser.add_argument(
        "--model-path",
        default="./Qwen3-VL-4B-Instruct",
        help="Local backend only",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--server-url", default="")
    # RAG 参数
    parser.add_argument("--rag-db", default="", help="ChromaDB 路径（启用 RAG）")
    parser.add_argument("--embedding-url", default="http://localhost:8766")
    parser.add_argument("--no-rag", action="store_true")
    # WIT 图片注入参数
    parser.add_argument("--wit-meta", default="", help="WIT meta.jsonl 路径")
    parser.add_argument("--wit-images", default="", help="WIT 图片目录路径")
    return parser.parse_args()


def make_batch_dir(output_root: str | Path, backend: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(output_root) / f"{backend}_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def build_shared_backend(args: argparse.Namespace):
    if args.backend == "local":
        return LocalQwenVisionBackend(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            server_url=args.server_url,
        )
    return None


def build_backend_for_task(args: argparse.Namespace, task: VisualTask):
    if args.backend == "scripted":
        return None
    raise ValueError(f"不支持的 backend: {args.backend}")


def replay_episode(task: VisualTask, episode_summary: dict[str, Any]) -> dict[str, Any]:
    env = RLEnvironment(task)
    env.reset()

    reward_sum = 0.0
    success = False
    for step in episode_summary.get("steps", []):
        action = step["action"]["action"]
        params = step["action"].get("params", {}) or {}
        result = env.step(action, params)
        reward_sum += result.reward
        success = success or bool(result.info.get("success"))
        if result.done:
            break

    return {"env": env, "reward_sum": reward_sum, "success": success}


def replay_ground_truth(task: VisualTask) -> dict[str, Any]:
    env = RLEnvironment(task)
    env.reset()

    reward_sum = 0.0
    success = False
    executed_steps = 0
    for step in task.ground_truth_trajectory:
        if step.action == "observe":
            continue
        result = env.step(step.action, step.action_params)
        reward_sum += result.reward
        success = success or bool(result.info.get("success"))
        executed_steps += 1
        if result.done:
            break

    return {"env": env, "reward_sum": reward_sum, "success": success, "steps": executed_steps}


def build_batch_summary(backend: str, results: list[dict[str, Any]], batch_dir: Path) -> dict[str, Any]:
    completed = [item for item in results if "error" not in item]
    succeeded = [item for item in completed if item.get("success")]
    reward_values = [item["reward_sum"] for item in completed]
    step_values = [item["steps"] for item in completed]

    return {
        "backend": backend,
        "created_at": datetime.now().isoformat(),
        "batch_dir": str(batch_dir),
        "total_tasks": len(results),
        "completed_tasks": len(completed),
        "failed_tasks": len(results) - len(completed),
        "successful_tasks": len(succeeded),
        "success_rate": (len(succeeded) / len(completed)) if completed else 0.0,
        "average_reward": mean(reward_values) if reward_values else 0.0,
        "average_steps": mean(step_values) if step_values else 0.0,
        "results": results,
    }


if __name__ == "__main__":
    main()
