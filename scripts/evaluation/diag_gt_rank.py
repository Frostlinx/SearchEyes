#!/usr/bin/env python3
"""
diag_gt_rank.py — diagnose ground-truth retrieval rank for text-to-WIT lookup.

Workflow:
1. Read the first N tasks from a JSONL task file.
2. Use `query_rewritten`, or fall back to `query_raw`, then `goal`.
3. Call the embedding server `/embed` endpoint with {"text": query}.
4. Query ChromaDB with the returned vector.
5. Find the 1-indexed rank of `ground_truth_wit_id`; if absent, use 101.
6. Print rank histogram, recall-style percentages, median/mean rank, and a
   heuristic conclusion on whether the main issue is:
   - A: cross-modal mismatch
   - C: top-k truncation
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS_PATH = PROJECT_ROOT / "data" / "tasks" / "research_tasks_v2.jsonl"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "wit_kb_v2" / "chroma_db"
DEFAULT_COLLECTION = "wit_knowledge_v2_qwen"
HISTOGRAM_BINS: list[tuple[str, int, int | None]] = [
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    (">100", 101, None),
]


def http_get_json(url: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_tasks(path: Path, max_tasks: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"task file not found: {path}")

    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid jsonl at line {line_no}: {exc}") from exc
            tasks.append(task)
            if len(tasks) >= max_tasks:
                break
    return tasks


def pick_query(task: dict[str, Any], fields: list[str]) -> tuple[str | None, str | None]:
    for field in fields:
        value = task.get(field)
        if isinstance(value, str) and value.strip():
            return field, value.strip()
    return None, None


def rank_of(target_id: str, ids: list[str]) -> int:
    for index, wit_id in enumerate(ids, start=1):
        if wit_id == target_id:
            return index
    return 101


def histogram(ranks: list[int]) -> list[tuple[str, int]]:
    counts: list[tuple[str, int]] = []
    for label, lower, upper in HISTOGRAM_BINS:
        if upper is None:
            value = sum(1 for rank in ranks if rank >= lower)
        else:
            value = sum(1 for rank in ranks if lower <= rank <= upper)
        counts.append((label, value))
    return counts


def pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (count / total) * 100.0


def infer_primary_cause(ranks: list[int]) -> str:
    total = len(ranks)
    if total == 0:
        return "结论: 无有效样本，无法判断主因。"

    not_found = sum(1 for rank in ranks if rank > 100)
    top6 = sum(1 for rank in ranks if rank <= 6)
    top20 = sum(1 for rank in ranks if rank <= 20)
    top50 = sum(1 for rank in ranks if rank <= 50)
    median_rank = statistics.median(ranks)
    found = total - not_found

    if not_found >= max(1, int(total * 0.4)) or found <= total * 0.5 or median_rank > 50:
        return (
            "结论: 主因更像 A(cross-modal mismatch)。"
            f" top-100 内命中仅 {found}/{total}，且 >100 有 {not_found}/{total}。"
        )

    if top50 - top6 >= max(2, int(total * 0.2)) or (top20 > top6 and top50 >= total * 0.7):
        return (
            "结论: 主因更像 C(top-k 截断)。"
            f" 大量样本能进 top-20/top-50（top-20={top20}/{total}, top-50={top50}/{total}），"
            f" 但 top-6 命中偏低（top-6={top6}/{total}）。"
        )

    return (
        "结论: 更偏向 A(cross-modal mismatch)。"
        f" 虽然部分样本进入 top-50，但排序前列命中没有形成明显的 top-k 截断特征。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ground-truth rank in Chroma retrieval")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-url", default="http://localhost:8766")
    parser.add_argument("--max-tasks", type=int, default=30)
    parser.add_argument("--n-results", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--query-fields",
        nargs="+",
        default=["query_rewritten", "query_raw", "goal"],
        help="Query field priority list",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    print(f"[diag] tasks={args.tasks}")
    print(f"[diag] db_path={args.db_path}")
    print(f"[diag] collection={args.collection}")
    print(f"[diag] embedding_url={args.embedding_url}")
    print(f"[diag] max_tasks={args.max_tasks} n_results={args.n_results}")
    print(f"[diag] query_fields={args.query_fields}")

    health_url = f"{args.embedding_url.rstrip('/')}/health"
    embed_url = f"{args.embedding_url.rstrip('/')}/embed"

    issues: list[str] = []
    print(f"[preflight] task_file_exists={args.tasks.exists()}")
    print(f"[preflight] chroma_db_exists={args.db_path.exists()}")

    health: dict[str, Any] | None = None
    try:
        health = http_get_json(health_url, timeout=args.timeout)
        print(f"[health] status={health.get('status')} dim={health.get('dim')}")
        if health.get("status") != "ok":
            issues.append(f"embedding server not ready: {health}")
    except urllib.error.URLError as exc:
        print(f"[health] error={exc}")
        issues.append(f"embedding server health check failed: {exc}")

    if not args.tasks.exists():
        issues.append(f"task file not found: {args.tasks}")

    if not args.db_path.exists():
        issues.append(f"chroma db path not found: {args.db_path}")

    try:
        import chromadb
    except ImportError as exc:
        issues.append(f"chromadb import failed: {exc}")
        chromadb = None  # type: ignore[assignment]

    client = None
    collection = None
    if args.db_path.exists() and "chromadb" in locals() and chromadb is not None:
        client = chromadb.PersistentClient(path=str(args.db_path))
        available = [getattr(col, "name", str(col)) for col in client.list_collections()]
        print(f"[chroma] collections={available}")
        if args.collection not in available:
            issues.append(
                f"collection not found: {args.collection}; available={available}"
            )
        else:
            collection = client.get_collection(args.collection)
            print(f"[chroma] count={collection.count()}")

    if issues:
        raise RuntimeError("; ".join(issues))

    tasks = load_tasks(args.tasks, args.max_tasks)
    if not tasks:
        raise RuntimeError("task file is empty after loading")

    processed = 0
    ranks: list[int] = []
    for index, task in enumerate(tasks, start=1):
        task_id = str(task.get("task_id", f"task_{index:04d}"))
        gt_id = str(task.get("ground_truth_wit_id", "")).strip()
        if not gt_id:
            raise RuntimeError(f"task {task_id} missing ground_truth_wit_id")

        query_field, query = pick_query(task, args.query_fields)
        if not query:
            raise RuntimeError(
                f"task {task_id} missing usable query in fields {args.query_fields}"
            )

        try:
            embed = http_post_json(embed_url, {"text": query}, timeout=args.timeout)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"/embed failed for task {task_id}: {exc}") from exc

        vector = embed.get("vector")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError(f"/embed returned invalid vector for task {task_id}")

        results = collection.query(
            query_embeddings=[vector],
            n_results=args.n_results,
        )
        ids = results.get("ids", [[]])[0]
        rank = rank_of(gt_id, ids)
        ranks.append(rank)
        processed += 1

        print(
            f"[task {index:02d}] task_id={task_id} query_field={query_field} "
            f"gt_id={gt_id} rank={rank} query={query!r}"
        )

    print("")
    print(f"[summary] processed={processed}")
    print("[summary] rank_histogram:")
    for label, count in histogram(ranks):
        print(f"  {label:>6}: {count}")

    top6 = sum(1 for rank in ranks if rank <= 6)
    top20 = sum(1 for rank in ranks if rank <= 20)
    top50 = sum(1 for rank in ranks if rank <= 50)
    not_found = sum(1 for rank in ranks if rank > 100)

    print(f"[summary] % in top-6     = {pct(top6, processed):.2f}%")
    print(f"[summary] % in top-20    = {pct(top20, processed):.2f}%")
    print(f"[summary] % in top-50    = {pct(top50, processed):.2f}%")
    print(f"[summary] % not found    = {pct(not_found, processed):.2f}%")
    print(f"[summary] median rank    = {statistics.median(ranks):.2f}")
    print(f"[summary] mean rank      = {statistics.mean(ranks):.2f}")
    print(infer_primary_cause(ranks))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
