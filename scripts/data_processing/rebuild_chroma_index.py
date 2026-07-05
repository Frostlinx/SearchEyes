#!/usr/bin/env python3
"""
rebuild_chroma_index.py — 重建 ChromaDB 向量索引（支持 1000+ WIT 条目）
======================================================================
从 meta.jsonl 读取条目，逐条调 embedding server 获取向量，存入 ChromaDB。

用法:
    python scripts/rebuild_chroma_index.py \
        --meta data/wit_subset_hf/meta.jsonl \
        --images data/wit_subset_hf/images \
        --chroma-db data/wit_subset_hf/chroma_db \
        --embedding-url http://localhost:8766
"""

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path


def rebuild_index(meta_path: str, images_dir: str, chroma_db_path: str,
                  embedding_url: str, collection_name: str = "wit_knowledge",
                  batch_size: int = 10):
    import chromadb

    meta = Path(meta_path)
    imgs = Path(images_dir)
    db_path = Path(chroma_db_path)

    entries = [json.loads(line) for line in meta.read_text("utf-8").splitlines() if line.strip()]
    print(f"[Index] 读取 {len(entries)} 条 WIT 元数据")

    # 清空旧索引
    if db_path.exists():
        shutil.rmtree(db_path)
        print(f"[Index] 已清空旧索引: {db_path}")

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    success = 0
    failed = 0
    t0 = time.time()

    for i, entry in enumerate(entries):
        img_path = imgs / entry["image_filename"]
        if not img_path.exists():
            failed += 1
            continue

        # 获取 embedding
        vector = get_embedding(str(img_path.resolve()), embedding_url)
        if vector is None:
            failed += 1
            continue

        collection.add(
            ids=[entry["wit_id"]],
            embeddings=[vector],
            metadatas=[{
                "page_title": entry.get("page_title", ""),
                "caption": entry.get("caption", ""),
                "source_url": entry.get("source_url", ""),
                "image_filename": entry.get("image_filename", ""),
            }],
        )
        success += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = success / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(entries)}] success={success} failed={failed} ({rate:.1f} img/s)")

    elapsed = time.time() - t0
    print(f"\n[Index] 完成!")
    print(f"  成功: {success}/{len(entries)}")
    print(f"  失败: {failed}")
    print(f"  耗时: {elapsed:.1f}s ({success/elapsed:.1f} img/s)")
    print(f"  索引: {db_path}")

    # 验证：self-retrieval
    print(f"\n[Index] Self-retrieval 验证...")
    test_entry = entries[0]
    test_img = imgs / test_entry["image_filename"]
    test_vec = get_embedding(str(test_img.resolve()), embedding_url)
    if test_vec:
        results = collection.query(query_embeddings=[test_vec], n_results=3)
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        print(f"  查询: {test_entry['wit_id']} ({test_entry['caption'][:40]})")
        for j, (rid, dist) in enumerate(zip(ids, distances)):
            print(f"  top-{j+1}: {rid} distance={dist:.4f}")
        if ids and ids[0] == test_entry["wit_id"]:
            print(f"  Self-retrieval: PASSED (distance={distances[0]:.6f})")
        else:
            print(f"  Self-retrieval: FAILED (top-1={ids[0] if ids else 'N/A'})")


def get_embedding(image_path: str, embedding_url: str) -> list[float] | None:
    url = f"{embedding_url}/embed"
    payload = json.dumps({"image_path": image_path}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("vector")
    except Exception as e:
        print(f"  [embed-error] {image_path}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--chroma-db", required=True)
    parser.add_argument("--embedding-url", default="http://localhost:8766")
    parser.add_argument("--collection", default="wit_knowledge")
    args = parser.parse_args()

    rebuild_index(args.meta, args.images, args.chroma_db, args.embedding_url, args.collection)
