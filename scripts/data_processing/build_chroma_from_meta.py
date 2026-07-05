#!/usr/bin/env python3
"""
scripts/build_chroma_from_meta.py
直接从 data/wit_kb_v2/meta.jsonl 的预计算 embedding 字段建 ChromaDB。
不需要 GPU，不需要启动 embedding server。

用法:
    python scripts/build_chroma_from_meta.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT       = Path("/root/autodl-tmp/QWEN/QWEN-project")
META       = ROOT / "data/wit_kb_v2/meta.jsonl"
CHROMA_DB  = ROOT / "data/wit_kb_v2/chroma_db"
COLLECTION = "wit_knowledge_v2"
BATCH_SIZE = 200

print("=" * 55)
print("build_chroma_from_meta.py")
print(f"  meta:       {META}")
print(f"  chroma_db:  {CHROMA_DB}")
print(f"  collection: {COLLECTION}")
print("=" * 55)

try:
    import chromadb
except ImportError:
    sys.exit("chromadb not installed: pip install chromadb")

# ── 读 meta.jsonl ─────────────────────────────────────────
entries = [json.loads(l) for l in META.read_text().splitlines() if l.strip()]
print(f"\n读取 {len(entries)} 条 entries")

no_emb = sum(1 for e in entries if not e.get("embedding"))
if no_emb:
    print(f"[WARNING] {no_emb} 条缺少 embedding，将跳过")

entries = [e for e in entries if e.get("embedding")]
print(f"有效条目: {len(entries)}")
print(f"embedding dim: {len(entries[0]['embedding'])}")

# ── 建 ChromaDB ───────────────────────────────────────────
CHROMA_DB.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(CHROMA_DB))

try:
    client.delete_collection(COLLECTION)
    print(f"\n删除旧 collection '{COLLECTION}'")
except Exception:
    pass

col = client.create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},
)
print(f"创建 collection '{COLLECTION}'")

# ── 批量写入 ──────────────────────────────────────────────
t0   = time.time()
ok   = 0
fail = 0

for batch_start in range(0, len(entries), BATCH_SIZE):
    batch = entries[batch_start : batch_start + BATCH_SIZE]
    try:
        col.add(
            ids        = [e["wit_id"] for e in batch],
            embeddings = [e["embedding"] for e in batch],
            documents  = [
                f"{e.get('page_title','')} | {e.get('caption','')}"
                for e in batch
            ],
            metadatas  = [{
                "page_title":    e.get("page_title", ""),
                "section_title": e.get("section_title", ""),
                "caption":       e.get("caption", ""),
                "image_url":     e.get("image_url", ""),
                "image_filename":e.get("image_filename", ""),
                "context":       e.get("context", "")[:200],
            } for e in batch],
        )
        ok += len(batch)
    except Exception as exc:
        print(f"  [batch error] {exc}")
        fail += len(batch)

    done = batch_start + len(batch)
    elapsed = time.time() - t0
    print(f"  [{done}/{len(entries)}] ok={ok} fail={fail} "
          f"elapsed={elapsed:.1f}s", flush=True)

print(f"\n完成: ok={ok} fail={fail} total={col.count()}")
print(f"耗时: {time.time()-t0:.1f}s")

# ── 快速验证 ─────────────────────────────────────────────
print("\n── 验证：用第一条 embedding 查询 top-3 ──")
q_vec = entries[0]["embedding"]
res   = col.query(query_embeddings=[q_vec], n_results=3)
ids   = res["ids"][0]
dists = res["distances"][0]
metas = res["metadatas"][0]
for rank, (wid, dist, meta) in enumerate(zip(ids, dists, metas), 1):
    print(f"  #{rank} {wid}  dist={dist:.4f}  title={meta['page_title'][:50]!r}")

print(f"\n[PASS] ChromaDB 已就绪: {CHROMA_DB}")
print(f"collection={COLLECTION}  count={col.count()}")
