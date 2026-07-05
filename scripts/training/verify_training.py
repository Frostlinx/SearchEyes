#!/usr/bin/env python3
"""
verify_training.py — RAG-aware 训练后评测脚本
==============================================
1. 加载微调后的模型（LoRA adapter 或全参数）
2. 用 RAG 搜索引擎驱动的沙盒跑评测 episode
3. 对比 scripted baseline vs 微调模型
4. 输出成功率（wit_id 匹配）、平均步数、zoom 使用率等指标

用法:
    # RAG+RL 模型评测
    python verify_training.py --model-dir checkpoints/grpo_rag_run1/final \
        --chroma-db-path data/wit_subset_hf/chroma_db \
        --embedding-server-url http://localhost:8000 \
        --n-tasks 30 --output-json eval_rag_run1.json

    # Scripted baseline 评测
    python verify_training.py --scripted-baseline \
        --chroma-db-path data/wit_subset_hf/chroma_db \
        --embedding-server-url http://localhost:8000 \
        --n-tasks 30 --output-json eval_scripted.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from searcheyes.task_schema import VisualTask, DifficultyLevel
from searcheyes.vlm_agent import (
    ScriptedPilotBackend,
    LocalQwenVisionBackend,
    ActionDecision,
    DecisionContext,
)
from searcheyes.agent_loop import AgentLoop


# ─────────────────────────────────────────────
# 评测指标
# ─────────────────────────────────────────────

@dataclass
class EvalResult:
    task_id: str
    difficulty: str
    success: bool = False
    steps: int = 0
    zoom_count: int = 0
    final_action: str = ""
    bought_wit_id: str = ""       # 实际购买的产品 wit_id
    gt_wit_id: str = ""           # ground truth wit_id
    error: str = ""


@dataclass
class EvalSummary:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def avg_steps(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.steps for r in self.results) / len(self.results)

    @property
    def zoom_usage_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.zoom_count > 0) / len(self.results)

    def by_difficulty(self) -> dict[str, dict]:
        groups: dict[str, list[EvalResult]] = {}
        for r in self.results:
            groups.setdefault(r.difficulty, []).append(r)
        out = {}
        for diff, items in groups.items():
            out[diff] = {
                "count": len(items),
                "success_rate": sum(1 for i in items if i.success) / len(items),
                "avg_steps": sum(i.steps for i in items) / len(items),
                "zoom_usage": sum(1 for i in items if i.zoom_count > 0) / len(items),
            }
        return out

    def print_report(self):
        n = len(self.results)
        bought_count = sum(1 for r in self.results if r.final_action in {"buy", "add_cart"})
        error_count = sum(1 for r in self.results if r.error)
        print("\n" + "=" * 60)
        print("评测报告 (RAG-as-Search-Engine)")
        print("=" * 60)
        print(f"  总任务数:    {n}")
        print(f"  成功率:      {self.success_rate * 100:.1f}% ({sum(1 for r in self.results if r.success)}/{n})")
        print(f"  执行 buy:    {bought_count}/{n} ({bought_count/n*100:.0f}%)" if n else "")
        print(f"  平均步数:    {self.avg_steps:.1f}")
        print(f"  zoom 使用率: {self.zoom_usage_rate * 100:.1f}%")
        if error_count:
            print(f"  错误数:      {error_count}")
        print()
        print("  按难度分布:")
        for diff, stats in sorted(self.by_difficulty().items()):
            print(f"    {diff:8s}: 成功率={stats['success_rate']*100:.0f}%  "
                  f"步数={stats['avg_steps']:.1f}  "
                  f"zoom={stats['zoom_usage']*100:.0f}%  "
                  f"(n={stats['count']})")
        print("=" * 60)


# ─────────────────────────────────────────────
# 单任务评测
# ─────────────────────────────────────────────

async def eval_one_task(
    task: VisualTask,
    backend,
    max_steps: int = 12,
    rag: Any = None,
    images_dir: Path | None = None,
) -> EvalResult:
    result = EvalResult(
        task_id=task.task_id,
        difficulty=task.difficulty.value,
        gt_wit_id=getattr(task, "ground_truth_wit_id", ""),
    )
    try:
        loop = AgentLoop(
            backend=backend,
            task=task,
            output_root=PROJECT_ROOT / "output" / "eval_rollouts",
            max_steps=max_steps,
            rag=rag,
            images_dir=images_dir,
        )
        summary = await loop.run()
        steps = summary.get("steps", [])
        result.steps = len(steps)
        result.zoom_count = sum(
            1 for s in steps
            if s.get("action", {}).get("action") == "zoom"
        )
        if steps:
            last = steps[-1]
            last_action = last.get("action", {}).get("action", "")
            result.final_action = last_action
            # 成功判定：buy/add_cart 且 wit_id 匹配 ground truth
            if last_action in {"buy", "add_cart"}:
                # 从 engine 获取已选产品的 wit_id
                final_state = summary.get("final_state", {})
                selected_pid = final_state.get("selected_product_id")
                product = loop.engine.products.get(selected_pid, {})
                result.bought_wit_id = product.get("wit_id", "")
                if result.gt_wit_id:
                    result.success = result.bought_wit_id == result.gt_wit_id
                else:
                    result.success = True  # 无 GT 时退化为"执行了 buy"
    except Exception as e:
        result.error = str(e)
    return result


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

async def run_eval(
    model_dir: str | None,
    n_tasks: int,
    use_scripted: bool,
    server_url: str,
    chroma_db_path: str = "",
    embedding_server_url: str = "",
    wit_images_dir: str = "",
    task_jsonl: str = "",
) -> EvalSummary:
    # ── 初始化 RAG ──
    rag = None
    images_dir = None
    if chroma_db_path:
        from searcheyes.multimodal_rag import MultimodalRAG, RagConfig
        rag_config = RagConfig(
            chroma_db_path=chroma_db_path,
            embedding_server_url=embedding_server_url or "http://localhost:8000",
        )
        rag = MultimodalRAG(rag_config)
        images_dir = Path(wit_images_dir) if wit_images_dir else Path(chroma_db_path).parent / "images"
        print(f"RAG 已初始化: {chroma_db_path}")
    else:
        print("WARNING: 未指定 --chroma-db-path，评测将不使用 RAG")

    # ── 加载任务 ──
    task_file = Path(task_jsonl) if task_jsonl else PROJECT_ROOT / "data" / "tasks" / "rag_tasks.jsonl"
    if not task_file.exists():
        # fallback to old file
        task_file = PROJECT_ROOT / "data" / "tasks" / "visual_tasks.jsonl"
    if task_file.exists():
        tasks: list[VisualTask] = []
        with open(task_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(VisualTask.load_from_dict(json.loads(line)))
        print(f"从 {task_file.name} 加载 {len(tasks)} 个任务")
    else:
        print(f"任务文件不存在: {task_file}")
        sys.exit(1)

    # 取前 n_tasks 个，确保难度均衡
    easy   = [t for t in tasks if t.difficulty == DifficultyLevel.EASY][:max(1, n_tasks // 3)]
    medium = [t for t in tasks if t.difficulty == DifficultyLevel.MEDIUM][:max(1, n_tasks // 3)]
    hard   = [t for t in tasks if t.difficulty == DifficultyLevel.HARD][:max(1, n_tasks // 3)]
    eval_tasks = (easy + medium + hard)[:n_tasks]
    print(f"评测任务: {len(eval_tasks)} 个 (easy={len(easy)}, medium={len(medium)}, hard={len(hard)})")

    # ── 构建 backend ──
    if use_scripted:
        print("使用 ScriptedPilot baseline（ground truth 轨迹）")
        class PerTaskScriptedBackend:
            def __init__(self):
                self._inner = None
            def set_task(self, task: VisualTask):
                script = [s.__dict__ for s in task.ground_truth_trajectory]
                self._inner = ScriptedPilotBackend(script)
            def decide(self, context: DecisionContext) -> ActionDecision:
                return self._inner.decide(context)
        backend = PerTaskScriptedBackend()
    else:
        if not model_dir:
            print("请指定 --model-dir 或使用 --scripted-baseline")
            sys.exit(1)
        print(f"加载微调模型: {model_dir}")
        backend = LocalQwenVisionBackend(
            model_path=model_dir,
            device="auto",
            dtype="auto",
            server_url=server_url,
        )

    # ── 逐任务评测 ──
    summary = EvalSummary()
    for i, task in enumerate(eval_tasks):
        print(f"  [{i+1}/{len(eval_tasks)}] {task.task_id} ({task.difficulty.value}) ...", end=" ", flush=True)
        if use_scripted and hasattr(backend, "set_task"):
            backend.set_task(task)
        result = await eval_one_task(task, backend, rag=rag, images_dir=images_dir)
        summary.results.append(result)
        status = "OK" if result.success else "FAIL"
        extra = ""
        if result.bought_wit_id and result.gt_wit_id:
            match = "match" if result.bought_wit_id == result.gt_wit_id else "mismatch"
            extra = f" wit={match}"
        print(f"{status} steps={result.steps} zoom={result.zoom_count}{extra}"
              + (f" err={result.error[:50]}" if result.error else ""))

    return summary


def main():
    parser = argparse.ArgumentParser(description="RAG-aware 训练后评测")
    parser.add_argument("--model-dir",         type=str, default=None,
                        help="微调后模型目录（LoRA adapter 或全参数）")
    parser.add_argument("--n-tasks",           type=int, default=30,
                        help="评测任务数量（建议 3 的倍数，保证难度均衡）")
    parser.add_argument("--scripted-baseline", action="store_true",
                        help="用 ground truth 轨迹作为 baseline（不需要模型）")
    parser.add_argument("--server-url",        type=str, default="",
                        help="本地模型服务器 URL（可选，如 http://localhost:8765）")
    parser.add_argument("--output-json",       type=str, default="",
                        help="把评测结果保存为 JSON 文件")
    # RAG 相关参数
    parser.add_argument("--chroma-db-path",    type=str, default="",
                        help="ChromaDB 路径（如 data/wit_subset_hf/chroma_db）")
    parser.add_argument("--embedding-server-url", type=str, default="http://localhost:8000",
                        help="Embedding server URL")
    parser.add_argument("--wit-images-dir",    type=str, default="",
                        help="WIT 图片目录")
    parser.add_argument("--task-jsonl",        type=str, default="",
                        help="任务文件路径（默认 data/tasks/rag_tasks.jsonl）")
    args = parser.parse_args()

    if not args.model_dir and not args.scripted_baseline:
        print("请指定 --model-dir <路径> 或 --scripted-baseline")
        parser.print_help()
        sys.exit(1)

    summary = asyncio.run(run_eval(
        model_dir=args.model_dir,
        n_tasks=args.n_tasks,
        use_scripted=args.scripted_baseline,
        server_url=args.server_url,
        chroma_db_path=args.chroma_db_path,
        embedding_server_url=args.embedding_server_url,
        wit_images_dir=args.wit_images_dir,
        task_jsonl=args.task_jsonl,
    ))

    summary.print_report()

    if args.output_json:
        out = {
            "success_rate": summary.success_rate,
            "avg_steps": summary.avg_steps,
            "zoom_usage_rate": summary.zoom_usage_rate,
            "by_difficulty": summary.by_difficulty(),
            "details": [
                {
                    "task_id": r.task_id,
                    "difficulty": r.difficulty,
                    "success": r.success,
                    "steps": r.steps,
                    "zoom_count": r.zoom_count,
                    "final_action": r.final_action,
                    "bought_wit_id": r.bought_wit_id,
                    "gt_wit_id": r.gt_wit_id,
                    "error": r.error,
                }
                for r in summary.results
            ],
        }
        Path(args.output_json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已保存: {args.output_json}")


if __name__ == "__main__":
    main()
