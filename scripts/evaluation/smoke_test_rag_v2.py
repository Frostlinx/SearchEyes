#!/usr/bin/env python3
"""
smoke_test_rag_v2.py — embedding server + Chroma RAG 冒烟验证
============================================================
验证顺序:
1. GET  /health
2. POST /embed 发送单条 text query
3. 使用返回向量查询 ChromaDB，确认有结果
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_CANDIDATES = [
    PROJECT_ROOT / "data" / "wit_kb_v2" / "chroma_db",
    PROJECT_ROOT / "data" / "wit_subset_hf" / "chroma_db",
    PROJECT_ROOT / "data" / "wit_subset" / "chroma_db",
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


def resolve_db_path(explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    for candidate in DEFAULT_DB_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(p) for p in DEFAULT_DB_CANDIDATES)
    raise FileNotFoundError(f"未找到可用 ChromaDB 路径，已检查: {searched}")


def run(args: argparse.Namespace) -> int:
    db_path = resolve_db_path(args.db_path)
    health_url = f"{args.embedding_url.rstrip('/')}/health"
    embed_url = f"{args.embedding_url.rstrip('/')}/embed"

    print(f"[smoke] embedding_url={args.embedding_url}")
    print(f"[smoke] db_path={db_path}")
    print(f"[smoke] collection={args.collection}")
    print(f"[smoke] query={args.query}")

    try:
        health = http_get_json(health_url, timeout=args.timeout)
    except urllib.error.URLError as exc:
        print(f"[FAIL] health 检查失败: {exc}")
        return 1

    status = health.get("status")
    dim = int(health.get("dim", 0) or 0)
    print(f"[health] status={status} dim={dim}")
    if status != "ok":
        print("[FAIL] embedding server 尚未 ready")
        return 1

    try:
        embed = http_post_json(
            embed_url,
            {"text": args.query},
            timeout=args.timeout,
        )
    except urllib.error.URLError as exc:
        print(f"[FAIL] text embed 失败: {exc}")
        return 1

    vector = embed.get("vector")
    if not isinstance(vector, list) or not vector:
        print("[FAIL] /embed 未返回有效向量")
        return 1
    print(f"[embed] dim={len(vector)}")
    if dim and len(vector) != dim:
        print(f"[FAIL] 向量维度不匹配: health={dim}, embed={len(vector)}")
        return 1

    try:
        import chromadb
    except ImportError as exc:
        print(f"[FAIL] chromadb 未安装: {exc}")
        return 1

    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(args.collection)
        count = collection.count()
        print(f"[chroma] count={count}")
        if count <= 0:
            print("[FAIL] collection 为空")
            return 1

        results = collection.query(
            query_embeddings=[vector],
            n_results=args.top_k,
        )
    except Exception as exc:
        print(f"[FAIL] chroma query 失败: {exc}")
        return 1

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not ids:
        print("[FAIL] chroma query 无返回结果")
        return 1

    print("[query] top hits:")
    for index, wit_id in enumerate(ids, start=1):
        distance = distances[index - 1] if index - 1 < len(distances) else None
        metadata = metadatas[index - 1] if index - 1 < len(metadatas) else {}
        title = metadata.get("page_title", "")
        caption = metadata.get("caption", "")
        dist_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "N/A"
        print(
            f"  #{index} wit_id={wit_id} distance={dist_text} "
            f"title={title[:60]!r} caption={caption[:80]!r}"
        )

    print("[PASS] embedding server + text embed + chroma query 全部通过")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for text-RAG v2 pipeline")
    parser.add_argument("--embedding-url", default="http://localhost:8766")
    parser.add_argument("--db-path", default="", help="ChromaDB 路径；为空则自动探测")
    parser.add_argument("--collection", default="wit_knowledge_v2")
    parser.add_argument("--query", default="red bus in london city street")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
