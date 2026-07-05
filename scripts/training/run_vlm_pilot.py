"""
run_vlm_pilot.py — Phase 8.5 Pilot 入口
======================================
支持两种模式：
1. scripted: 用 ground-truth 轨迹验证闭环
2. api:      用兼容 OpenAI Chat Completions 的视觉模型跑真实 pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from searcheyes.agent_loop import run_agent_loop
from searcheyes.task_schema import DifficultyLevel, TrajectoryStep, VisualTask
from searcheyes.vlm_agent import (
    LocalQwenVisionBackend,
    OpenAICompatibleVisionBackend,
    ScriptedPilotBackend,
)


def main():
    args = parse_args()
    task = load_task(args.task_jsonl, args.task_index)

    if args.backend == "scripted":
        backend = ScriptedPilotBackend(
            [
                {"action": step.action, "action_params": step.action_params}
                for step in task.ground_truth_trajectory
            ]
        )
    elif args.backend == "local":
        backend = LocalQwenVisionBackend(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            server_url=args.server_url,
        )
    else:
        if not args.model:
            raise SystemExit("API 模式下必须提供 --model 或配置环境变量")
        backend = OpenAICompatibleVisionBackend(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            timeout_seconds=args.timeout,
        )

    # 初始化 RAG（如果配置了）
    rag = None
    if args.rag_db and not args.no_rag:
        from searcheyes.multimodal_rag import MultimodalRAG, RagConfig

        rag = MultimodalRAG(RagConfig(
            chroma_db_path=args.rag_db,
            embedding_server_url=args.embedding_url,
        ))

    # WIT 图片注入（视觉调研任务需要）
    wit_meta = args.wit_meta or None
    wit_images = args.wit_images or None
    # 如果配置了 rag-db 但没有指定 wit-meta，从 rag-db 父目录推断
    if not wit_meta and args.rag_db and not args.no_rag:
        inferred = Path(args.rag_db).parent / "meta.jsonl"
        if inferred.exists():
            wit_meta = str(inferred)
            wit_images = str(inferred.parent / "images")

    summary = run_agent_loop(
        backend=backend, task=task, rag=rag,
        wit_meta_jsonl=wit_meta, wit_images_dir=wit_images,
    )

    print("=" * 70)
    print(f"backend:    {args.backend}")
    if args.backend == "local":
        print(f"model_path: {args.model_path}")
    print(f"task_id:    {summary['task_id']}")
    print(f"episode:    {summary['episode_dir']}")
    print(f"steps:      {len(summary['steps'])}")
    for step in summary["steps"]:
        print(
            f"step {step['step_idx']:02d} | "
            f"{step['action']['action']:<13s} | "
            f"{step['validation']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 8.5 VLM pilot loop")
    parser.add_argument("--backend", choices=["scripted", "api", "local"], default="scripted")
    parser.add_argument(
        "--task-jsonl",
        default="./data/tasks/visual_tasks.jsonl",
        help="Path to the task JSONL file",
    )
    parser.add_argument("--task-index", type=int, default=0, help="0-based task index")
    parser.add_argument("--model", default="", help="API mode only")
    parser.add_argument("--base-url", default="", help="API mode only")
    parser.add_argument("--api-key", default="", help="API mode only")
    parser.add_argument("--timeout", type=int, default=60, help="API timeout in seconds")
    parser.add_argument(
        "--model-path",
        default="./Qwen3-VL-4B-Instruct",
        help="Local backend only",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto", help="Local backend only")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Local backend only",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation length")
    parser.add_argument(
        "--server-url",
        default="",
        help="Local backend: 指向 local_model_server.py 的地址（如 http://localhost:8765）",
    )
    # RAG 参数
    parser.add_argument(
        "--rag-db",
        default="",
        help="ChromaDB 目录路径（如 data/wit_subset/chroma_db），为空则不启用 RAG",
    )
    parser.add_argument(
        "--embedding-url",
        default="http://localhost:8766",
        help="embedding_server.py 的地址",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="即使配置了 --rag-db 也禁用 RAG",
    )
    # WIT 图片注入参数
    parser.add_argument(
        "--wit-meta",
        default="",
        help="WIT meta.jsonl 路径，用于注入真实图片到产品",
    )
    parser.add_argument(
        "--wit-images",
        default="",
        help="WIT 图片目录路径",
    )
    return parser.parse_args()


def load_task(path: str | Path, index: int) -> VisualTask:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if index < 0 or index >= len(lines):
        raise IndexError(f"task_index 越界: {index}, total={len(lines)}")

    raw = json.loads(lines[index])
    raw["difficulty"] = DifficultyLevel(raw["difficulty"])
    raw["ground_truth_trajectory"] = [TrajectoryStep(**step) for step in raw["ground_truth_trajectory"]]
    # 移除非 VisualTask 字段（如 wit_bindings）
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(VisualTask)}
    raw = {k: v for k, v in raw.items() if k in valid_fields}
    return VisualTask(**raw)


if __name__ == "__main__":
    main()
