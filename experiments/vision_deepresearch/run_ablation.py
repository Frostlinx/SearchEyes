"""
run_ablation.py — 主实验：Option A vs Option B vs Option A+B

用法（在项目根目录下）:
  # 快速对照：用 lazy caption（GT caption，验证Text Bridge上限）
  python experiments/vision_deepresearch/run_ablation.py --mode lazy --n 50

  # 真实 Option B（用VLM生成caption）
  python experiments/vision_deepresearch/run_ablation.py --mode vlm --n 50

  # 只跑 A（用于确认baseline）
  python experiments/vision_deepresearch/run_ablation.py --mode a_only --n 50

输出：results/ablation_{mode}_{timestamp}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# sys.path 设置
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

from config import TASKS_JSONL, IMAGES_DIR, EVAL_MAX_TASKS, TOP_K_DEFAULT, RESULTS_DIR
from metrics import MetricResult, compare
from query_transform import transform_image


def load_tasks(n: int) -> list[dict]:
    tasks = []
    with open(TASKS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
            if len(tasks) >= n:
                break
    return tasks


def find_image(task: dict) -> str | None:
    """从任务的 wit_bindings 或 ground_truth_wit_id 找到GT图片路径"""
    gt_wit_id = task.get("ground_truth_wit_id", "")
    # research_tasks_v2 格式：wit_bindings 里有 image_filename
    for binding in task.get("wit_bindings", []):
        if binding.get("wit_id") == gt_wit_id:
            fname = binding.get("image_filename", "")
            if fname:
                p = IMAGES_DIR / fname
                if p.exists():
                    return str(p)
    # 尝试 witb_ 前缀格式
    for ext in [".jpg", ".jpeg", ".png"]:
        p = IMAGES_DIR / f"{gt_wit_id}{ext}"
        if p.exists():
            return str(p)
    # 尝试文件名匹配
    for f in IMAGES_DIR.glob(f"*{gt_wit_id}*"):
        return str(f)
    return None


def run_option_a(tasks: list[dict], top_k: int, query_transform: str = "exact") -> tuple[MetricResult, list[dict]]:
    from retrieval_adapters import OptionA
    retriever = OptionA()
    label = "A" if query_transform == "exact" else f"A_{query_transform}"
    metric = MetricResult(mode=label)
    details = []

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"t{i}")
        gt_wit_id = task.get("ground_truth_wit_id", "")
        image_path = find_image(task)

        if not image_path or not gt_wit_id:
            print(f"[{label}] skip {task_id}: missing image or gt_wit_id")
            continue

        # Apply query transform (exact = no-op)
        query_img = transform_image(image_path, query_transform)

        t0 = time.monotonic()
        resp = retriever.retrieve(query_img, top_k=top_k)
        elapsed = (time.monotonic() - t0) * 1000
        rank = resp.compute_gt_rank(gt_wit_id, miss_rank=top_k + 1)
        metric.ranks.append(rank)

        details.append({
            "task_id": task_id, "gt_wit_id": gt_wit_id,
            "rank": rank, "hit_at_20": rank <= 20,
            "query_used": resp.query_used,
            "elapsed_ms": round(elapsed, 1),
        })
        print(f"[{label}] {i+1:03d}/{len(tasks)} {task_id} gt={gt_wit_id} rank={rank} ({elapsed:.0f}ms)")

    metric.compute()
    return metric, details


def run_option_b(tasks: list[dict], top_k: int, use_vlm: bool, query_transform: str = "exact") -> tuple[MetricResult, list[dict]]:
    from retrieval_adapters import OptionB
    base = "B_vlm" if use_vlm else "B_lazy"
    mode_label = base if query_transform == "exact" else f"{base}_{query_transform}"
    retriever = OptionB(use_vlm=use_vlm)
    metric = MetricResult(mode=mode_label)
    details = []

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"t{i}")
        gt_wit_id = task.get("ground_truth_wit_id", "")
        image_path = find_image(task)

        if not image_path or not gt_wit_id:
            print(f"[{mode_label}] skip {task_id}: missing image or gt_wit_id")
            continue

        query_img = transform_image(image_path, query_transform)

        t0 = time.monotonic()
        resp = retriever.retrieve(query_img, wit_id=gt_wit_id, top_k=top_k)
        elapsed = (time.monotonic() - t0) * 1000
        rank = resp.compute_gt_rank(gt_wit_id, miss_rank=top_k + 1)
        metric.ranks.append(rank)

        details.append({
            "task_id": task_id, "gt_wit_id": gt_wit_id,
            "rank": rank, "hit_at_20": rank <= 20,
            "query_used": resp.query_used,
            "elapsed_ms": round(elapsed, 1),
        })
        print(f"[{mode_label}] {i+1:03d}/{len(tasks)} {task_id} gt={gt_wit_id} rank={rank} query={resp.query_used[:60]!r}")

    metric.compute()
    return metric, details


def run_option_ab(tasks: list[dict], top_k: int, use_vlm: bool, query_transform: str = "exact") -> tuple[MetricResult, list[dict]]:
    from retrieval_adapters import OptionAB
    label = "AB" if query_transform == "exact" else f"AB_{query_transform}"
    retriever = OptionAB(use_vlm=use_vlm)
    metric = MetricResult(mode=label)
    details = []

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"t{i}")
        gt_wit_id = task.get("ground_truth_wit_id", "")
        image_path = find_image(task)

        if not image_path or not gt_wit_id:
            continue

        query_img = transform_image(image_path, query_transform)

        t0 = time.monotonic()
        resp = retriever.retrieve(query_img, wit_id=gt_wit_id, top_k=top_k)
        elapsed = (time.monotonic() - t0) * 1000
        rank = resp.compute_gt_rank(gt_wit_id, miss_rank=top_k + 1)
        metric.ranks.append(rank)

        details.append({
            "task_id": task_id, "gt_wit_id": gt_wit_id,
            "rank": rank, "hit_at_20": rank <= 20,
            "elapsed_ms": round(elapsed, 1),
        })
        print(f"[{label}] {i+1:03d}/{len(tasks)} {task_id} rank={rank}")

    metric.compute()
    return metric, details


def main():
    parser = argparse.ArgumentParser(description="Option A vs B ablation")
    parser.add_argument("--mode", choices=["a_only", "lazy", "vlm", "ab_lazy", "ab_vlm"],
                        default="lazy",
                        help="lazy=B with GT caption (upper bound), vlm=B with VLM, a_only=A only")
    parser.add_argument("--n", type=int, default=EVAL_MAX_TASKS, help="number of tasks")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--query-transform",
                        choices=["exact", "resize", "center_crop", "jpeg_q40"],
                        default="exact",
                        help="exact=byte-identical (self-match), others break exact pixel identity")
    args = parser.parse_args()

    if args.top_k < 20:
        parser.error("--top-k must be >= 20 to measure hit@20 correctly")

    print(f"[ablation] mode={args.mode} n={args.n} top_k={args.top_k} query_transform={args.query_transform}")
    tasks = load_tasks(args.n)
    print(f"[ablation] loaded {len(tasks)} tasks")

    results_payload: dict = {"mode": args.mode, "n": args.n, "top_k": args.top_k,
                              "query_transform": args.query_transform, "runs": {}}
    metrics_summary: list[MetricResult] = []

    # 始终跑 A 作为 baseline
    print(f"\n=== Option A (query_transform={args.query_transform}) ===")
    m_a, d_a = run_option_a(tasks, args.top_k, query_transform=args.query_transform)
    print(m_a.report())
    results_payload["runs"]["A"] = {"metric": vars(m_a), "details": d_a}
    metrics_summary.append(m_a)

    if args.mode in ("lazy", "vlm"):
        use_vlm = (args.mode == "vlm")
        print(f"\n=== Option B ({'VLM' if use_vlm else 'Lazy/GT caption'}, query_transform={args.query_transform}) ===")
        m_b, d_b = run_option_b(tasks, args.top_k, use_vlm=use_vlm, query_transform=args.query_transform)
        print(m_b.report())
        results_payload["runs"]["B"] = {"metric": vars(m_b), "details": d_b}
        metrics_summary.append(m_b)
        print(f"\n=== Comparison ===")
        print(compare(m_a, m_b))

    elif args.mode in ("ab_lazy", "ab_vlm"):
        use_vlm = (args.mode == "ab_vlm")
        print(f"\n=== Option A+B (RRF, query_transform={args.query_transform}) ===")
        m_ab, d_ab = run_option_ab(tasks, args.top_k, use_vlm=use_vlm, query_transform=args.query_transform)
        print(m_ab.report())
        results_payload["runs"]["AB"] = {"metric": vars(m_ab), "details": d_ab}
        metrics_summary.append(m_ab)
        print(f"\n=== Comparison A vs AB ===")
        print(compare(m_a, m_ab))

    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xform_tag = "" if args.query_transform == "exact" else f"_{args.query_transform}"
    out_path = RESULTS_DIR / f"ablation_{args.mode}{xform_tag}_{ts}.json"
    # ranks list 不序列化进 metric dict（太大）
    for run in results_payload["runs"].values():
        run["metric"].pop("ranks", None)
    out_path.write_text(json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ablation] results saved → {out_path}")

    # 最终摘要
    print("\n=== Final Summary ===")
    for m in metrics_summary:
        print(m.report())


if __name__ == "__main__":
    main()
