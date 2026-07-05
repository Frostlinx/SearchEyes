#!/usr/bin/env python3
"""
eval_rag_retrieval.py — RAG 检索质量独立评测
=============================================
不经过 agent/FSM/RL，直接测 embedding + ChromaDB 的检索能力。

两类评测：
  A) 诊断型 (diagnostic): 用 GT 图片/caption 做 query → 测 embedding 空间质量
  B) Case-realistic:       用真实任务 goal 做 query → 测真实 search 场景效果

指标：Recall@k, MRR, 自检索命中率
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from searcheyes.multimodal_rag import MultimodalRAG, RagConfig
from searcheyes.query_rewriter import rewrite as rewrite_query

def make_hybrid_rag(config: RagConfig) -> MultimodalRAG:
    """创建开启 BM25 hybrid 的 RAG 实例。"""
    from dataclasses import replace
    hybrid_config = replace(config, use_hybrid=True)
    return MultimodalRAG(hybrid_config)


# ── 数据结构 ────────────────────────────────────────────

@dataclass
class RetrievalResult:
    wit_id: str
    query_mode: str          # image_only / text_only / image_text / goal_text
    top_k_ids: list[str]     # 检索到的 wit_id 列表
    gt_rank: int             # GT 在结果中的排名（1-based），0 = 未命中
    distances: list[float]   # 对应距离


@dataclass
class EvalMetrics:
    mode: str
    total: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_6: float
    recall_at_10: float
    mrr: float               # Mean Reciprocal Rank
    self_retrieval_rate: float  # 自检索命中率 (rank == 1)
    avg_gt_rank: float        # GT 平均排名（仅命中的）
    miss_count: int           # 完全未命中数


# ── 核心评测逻辑 ────────────────────────────────────────

def evaluate_single(
    rag: MultimodalRAG,
    wit_id: str,
    image_path: str = "",
    text: str = "",
    query_mode: str = "image_only",
    top_k: int = 10,
) -> RetrievalResult:
    """对单个 WIT 条目执行一次检索，返回 GT 排名。"""
    if query_mode == "image_only":
        facts = rag.get_rag_facts_combined(image_path=image_path, top_k=top_k)
    elif query_mode == "text_only":
        facts = rag.get_rag_facts_combined(text=text, top_k=top_k)
    elif query_mode in ("image_text", "goal_text_image"):
        facts = rag.get_rag_facts_combined(image_path=image_path, text=text, top_k=top_k)
    elif query_mode == "goal_text":
        facts = rag.get_rag_facts_combined(text=text, top_k=top_k)
    else:
        facts = rag.get_rag_facts_combined(text=text, top_k=top_k)

    top_k_ids = [f.wit_id for f in facts]
    distances = [1.0 - f.score * 2 for f in facts]  # approx reverse of score

    gt_rank = 0
    for i, fid in enumerate(top_k_ids):
        if fid == wit_id:
            gt_rank = i + 1
            break

    return RetrievalResult(
        wit_id=wit_id,
        query_mode=query_mode,
        top_k_ids=top_k_ids,
        gt_rank=gt_rank,
        distances=distances,
    )


def compute_metrics(results: list[RetrievalResult], mode: str) -> EvalMetrics:
    """从一组 RetrievalResult 计算汇总指标。"""
    total = len(results)
    if total == 0:
        return EvalMetrics(mode=mode, total=0, recall_at_1=0, recall_at_3=0,
                           recall_at_5=0, recall_at_6=0, recall_at_10=0,
                           mrr=0, self_retrieval_rate=0, avg_gt_rank=0, miss_count=0)

    hits_at = {1: 0, 3: 0, 5: 0, 6: 0, 10: 0}
    rr_sum = 0.0
    rank_1_count = 0
    hit_ranks = []
    miss_count = 0

    for r in results:
        if r.gt_rank == 0:
            miss_count += 1
            continue

        for k in hits_at:
            if r.gt_rank <= k:
                hits_at[k] += 1

        rr_sum += 1.0 / r.gt_rank
        hit_ranks.append(r.gt_rank)

        if r.gt_rank == 1:
            rank_1_count += 1

    return EvalMetrics(
        mode=mode,
        total=total,
        recall_at_1=hits_at[1] / total,
        recall_at_3=hits_at[3] / total,
        recall_at_5=hits_at[5] / total,
        recall_at_6=hits_at[6] / total,
        recall_at_10=hits_at[10] / total,
        mrr=rr_sum / total,
        self_retrieval_rate=rank_1_count / total,
        avg_gt_rank=sum(hit_ranks) / len(hit_ranks) if hit_ranks else 0,
        miss_count=miss_count,
    )


# ── A) 诊断型评测 ──────────────────────────────────────

def run_diagnostic_eval(
    rag: MultimodalRAG,
    meta_entries: list[dict],
    images_dir: Path,
    top_k: int = 10,
    max_entries: int = 0,
) -> dict[str, EvalMetrics]:
    """用 GT 图片/caption 做 query，测 embedding 本身能力。"""
    entries = meta_entries[:max_entries] if max_entries > 0 else meta_entries

    results_by_mode: dict[str, list[RetrievalResult]] = {
        "image_only": [],
        "text_only": [],
        "image_text": [],
    }

    for i, entry in enumerate(entries):
        wit_id = entry["wit_id"]
        caption = entry.get("caption", "")
        img_path = str(images_dir / entry.get("image_filename", f"{wit_id}.jpg"))
        img_exists = Path(img_path).exists()

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  诊断型评测: {i+1}/{len(entries)} ...", flush=True)

        # Image-only 自检索
        if img_exists:
            r = evaluate_single(rag, wit_id, image_path=img_path,
                                query_mode="image_only", top_k=top_k)
            results_by_mode["image_only"].append(r)

        # Text-only (caption)
        if caption:
            r = evaluate_single(rag, wit_id, text=caption,
                                query_mode="text_only", top_k=top_k)
            results_by_mode["text_only"].append(r)

        # Image + Text 组合
        if img_exists and caption:
            r = evaluate_single(rag, wit_id, image_path=img_path, text=caption,
                                query_mode="image_text", top_k=top_k)
            results_by_mode["image_text"].append(r)

    metrics = {}
    for mode, results in results_by_mode.items():
        m = compute_metrics(results, f"diagnostic_{mode}")
        metrics[f"diagnostic_{mode}"] = m
    return metrics


# ── B) Case-Realistic 评测 ─────────────────────────────

def run_case_realistic_eval(
    rag: MultimodalRAG,
    tasks: list[dict],
    images_dir: Path,
    top_k: int = 10,
    max_tasks: int = 0,
    hybrid_rag: MultimodalRAG | None = None,
) -> dict[str, EvalMetrics]:
    """用真实任务 goal 做 query，测实际 search 场景效果。

    A/B 对比（按照实验设计顺序）：
      A) raw goal（baseline）
      B) rewritten text-only（Query Rewrite v1）
      D) GT caption（理想文本上界）
      E) GT image（理想图像上界）
      F) rewritten + BM25 hybrid（仅当 hybrid_rag 传入时）
    """
    entries = tasks[:max_tasks] if max_tasks > 0 else tasks

    results_by_mode: dict[str, list[RetrievalResult]] = {
        "A_raw_goal":       [],   # baseline
        "B_rewritten_text": [],   # Query Rewrite v1
        "D_gt_caption":     [],   # 理想文本上界
        "E_gt_image":       [],   # 理想图像上界
    }
    if hybrid_rag is not None:
        results_by_mode["F_bm25_hybrid"] = []  # BM25 + vector RRF

    for i, task in enumerate(entries):
        gt_wit_id = task.get("ground_truth_wit_id", "")
        if not gt_wit_id:
            continue

        goal = task.get("goal", "")
        gt_caption = task.get("ground_truth_caption", "")
        gt_img_filename = ""
        # 从 wit_bindings 找 GT 图片路径
        for binding in task.get("wit_bindings", []):
            if binding.get("wit_id") == gt_wit_id:
                gt_img_filename = binding.get("image_filename", "")
                break
        gt_img_path = str(images_dir / gt_img_filename) if gt_img_filename else ""
        gt_img_exists = gt_img_filename and Path(gt_img_path).exists()

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  A/B 评测: {i+1}/{len(entries)} ...", flush=True)

        rewritten = rewrite_query(goal) if goal else ""

        # A: raw goal
        if goal:
            r = evaluate_single(rag, gt_wit_id, text=goal,
                                query_mode="goal_text", top_k=top_k)
            results_by_mode["A_raw_goal"].append(r)

        # B: rewritten text-only
        if rewritten:
            r = evaluate_single(rag, gt_wit_id, text=rewritten,
                                query_mode="goal_text", top_k=top_k)
            results_by_mode["B_rewritten_text"].append(r)

        # C: rewritten text + 搜索页截图（用 GT 图片代替搜索页截图做近似测试）
        # 注意：真实搜索页截图几乎没有视觉信息（只有搜索框）
        # 用 GT image 会高估；这里故意不做，标注为 N/A
        # 实际 A/B 中搜索页截图是噪声，等 pipeline 接入后再跑

        # D: GT caption（理想文本上界）
        if gt_caption:
            r = evaluate_single(rag, gt_wit_id, text=gt_caption,
                                query_mode="text_only", top_k=top_k)
            results_by_mode["D_gt_caption"].append(r)

        # E: GT image（理想图像上界）
        if gt_img_exists:
            r = evaluate_single(rag, gt_wit_id, image_path=gt_img_path,
                                query_mode="image_only", top_k=top_k)
            results_by_mode["E_gt_image"].append(r)

        # F: rewritten text + BM25 hybrid（RRF）
        if hybrid_rag is not None and rewritten:
            r = evaluate_single(hybrid_rag, gt_wit_id, text=rewritten,
                                query_mode="goal_text", top_k=top_k)
            results_by_mode["F_bm25_hybrid"].append(r)

    metrics = {}
    for mode, results in results_by_mode.items():
        if results:
            m = compute_metrics(results, f"ab_{mode}")
            metrics[f"ab_{mode}"] = m
    return metrics


# ── 输出 ────────────────────────────────────────────────

def print_metrics_table(all_metrics: dict[str, EvalMetrics]):
    """打印汇总表格。"""
    print("\n" + "=" * 90)
    print(f"{'Mode':<28} {'N':>5} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@6':>7} {'R@10':>7} {'MRR':>7} {'Miss':>5}")
    print("-" * 90)
    for name, m in all_metrics.items():
        print(f"{name:<28} {m.total:>5} {m.recall_at_1:>7.1%} {m.recall_at_3:>7.1%} "
              f"{m.recall_at_5:>7.1%} {m.recall_at_6:>7.1%} {m.recall_at_10:>7.1%} "
              f"{m.mrr:>7.3f} {m.miss_count:>5}")
    print("=" * 90)


def save_report(all_metrics: dict[str, EvalMetrics], output_path: str):
    """保存 JSON 报告。"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {k: asdict(v) for k, v in all_metrics.items()},
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {output_path}")


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG 检索质量独立评测")
    parser.add_argument("--chroma-db-path", required=True, help="ChromaDB 路径")
    parser.add_argument("--embedding-server-url", default="http://localhost:8000")
    parser.add_argument("--meta-jsonl", required=True, help="WIT meta.jsonl 路径")
    parser.add_argument("--images-dir", required=True, help="WIT 图片目录")
    parser.add_argument("--tasks-jsonl", default="", help="rag_tasks.jsonl 路径（case-realistic eval）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-entries", type=int, default=0, help="诊断型评测最大条目数（0=全部）")
    parser.add_argument("--max-tasks", type=int, default=0, help="Case-realistic 最大任务数（0=全部）")
    parser.add_argument("--output", default="reports/rag_retrieval_eval.json")
    parser.add_argument("--skip-diagnostic", action="store_true", help="跳过诊断型评测")
    parser.add_argument("--skip-realistic", action="store_true", help="跳过 case-realistic 评测")
    parser.add_argument("--with-bm25", action="store_true", help="加入 BM25 hybrid 对比组（F组）")
    args = parser.parse_args()

    # 初始化 RAG
    base_config = RagConfig(
        chroma_db_path=args.chroma_db_path,
        embedding_server_url=args.embedding_server_url,
    )
    rag = MultimodalRAG(base_config)

    # 健康检查
    print("检查 embedding server ...", flush=True)
    test_vec = rag._get_text_embedding("test")
    if test_vec is None:
        print("ERROR: embedding server 不可达", file=sys.stderr)
        sys.exit(1)
    print(f"  OK, embedding dim = {len(test_vec)}")

    # 加载 meta
    images_dir = Path(args.images_dir)
    with open(args.meta_jsonl) as f:
        meta_entries = [json.loads(line) for line in f if line.strip()]
    print(f"加载 {len(meta_entries)} 条 WIT 条目")

    all_metrics: dict[str, EvalMetrics] = {}

    # A) 诊断型评测
    if not args.skip_diagnostic:
        print("\n>>> A) 诊断型 Retrieval Eval")
        t0 = time.time()
        diag = run_diagnostic_eval(rag, meta_entries, images_dir,
                                   top_k=args.top_k, max_entries=args.max_entries)
        all_metrics.update(diag)
        print(f"  耗时: {time.time()-t0:.1f}s")

    # B) Case-realistic 评测
    if not args.skip_realistic and args.tasks_jsonl:
        print("\n>>> B) Case-Realistic Eval")
        with open(args.tasks_jsonl) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        print(f"加载 {len(tasks)} 条任务")
        t0 = time.time()
        hybrid_rag = make_hybrid_rag(base_config) if args.with_bm25 else None
        if hybrid_rag:
            print("  BM25 hybrid 已启用（F组）", flush=True)
        real = run_case_realistic_eval(rag, tasks, images_dir,
                                       top_k=args.top_k, max_tasks=args.max_tasks,
                                       hybrid_rag=hybrid_rag)
        all_metrics.update(real)
        print(f"  耗时: {time.time()-t0:.1f}s")

    # 输出
    print_metrics_table(all_metrics)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_report(all_metrics, args.output)


if __name__ == "__main__":
    main()
