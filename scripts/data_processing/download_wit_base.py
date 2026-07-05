#!/usr/bin/env python3
"""
download_wit_base.py — 方案B：从 wikimedia/wit_base 重建多模态 KB
=================================================================
步骤：
  1. 从 hf-mirror 下载 wit_base shard0 parquet（~452MB）到 /tmp/
  2. 逐 row-group 读取，过滤英文+有效 caption+有效图片bytes
  3. 图片 bytes 直接存盘，不经过 PIL（防 OOM）
  4. 收集 2000 条后停止
  5. 写新的 data/wit_kb_v2/meta.jsonl（备份旧的）

wit_id 格式: witb_{md5(image_bytes)[:8]}  (前缀 witb_ 区别于旧的 wit2_)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

# ── 配置 ──────────────────────────────────────────────────
ROOT        = Path("/root/autodl-tmp/QWEN/QWEN-project")
KB_DIR      = ROOT / "data/wit_kb_v2"
IMAGES_DIR  = KB_DIR / "images"
META_JSONL  = KB_DIR / "meta.jsonl"
META_BACKUP = KB_DIR / "meta_text_only_backup.jsonl"
TMP_PARQUET = Path("/tmp/wit_base_shard0.parquet")

TARGET      = 2000
MIN_CAPTION = 20        # caption 最短字符数
MIN_IMG     = 2048      # 图片 bytes 最小字节（过滤损坏/空图）
SEED        = 42

PARQUET_URL = (
    "https://hf-mirror.com/api/datasets/wikimedia/wit_base"
    "/parquet/default/train/0.parquet"
)

# ── Step 0: 备份旧 meta.jsonl ────────────────────────────
print("=" * 55)
print("Step 0: 备份旧 meta.jsonl")
print("=" * 55)
if META_JSONL.exists() and not META_BACKUP.exists():
    META_BACKUP.write_bytes(META_JSONL.read_bytes())
    print(f"备份 → {META_BACKUP}")
else:
    print("  (无需备份 或 备份已存在)")

# ── Step 1: 下载 parquet ─────────────────────────────────
print("\n" + "=" * 55)
print("Step 1: 下载 wit_base shard0 parquet")
print("=" * 55)

if TMP_PARQUET.exists() and TMP_PARQUET.stat().st_size > 400_000_000:
    print(f"已存在: {TMP_PARQUET} ({TMP_PARQUET.stat().st_size // 1024 // 1024} MB)，跳过下载")
else:
    print(f"目标: {PARQUET_URL}")
    print(f"下载到: {TMP_PARQUET}")
    t0 = time.time()

    # 用 requests 流式下载（自动跟随 hf-mirror 重定向到 CDN）
    import requests as _req
    with _req.get(PARQUET_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total_size = int(resp.headers.get("Content-Length", 0))
        print(f"  文件大小: {total_size // 1024 // 1024} MB")
        downloaded = 0
        last_print = 0
        with open(TMP_PARQUET, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_print >= 50 * 1024 * 1024:     # 每 50MB 打印
                    elapsed = time.time() - t0
                    speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
                    pct = downloaded * 100 // total_size if total_size else 0
                    print(f"  {pct}% ({downloaded // 1024 // 1024} MB / "
                          f"{total_size // 1024 // 1024} MB)  {speed:.1f} MB/s", flush=True)
                    last_print = downloaded
    elapsed = time.time() - t0
    print(f"  下载完成: {TMP_PARQUET.stat().st_size // 1024 // 1024} MB，耗时 {elapsed:.0f}s")

# ── Step 2: 扫描 parquet，提取图片 ──────────────────────
print("\n" + "=" * 55)
print(f"Step 2: 扫描 parquet，提取 {TARGET} 条 image-text pair")
print("=" * 55)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

pf          = pq.ParquetFile(TMP_PARQUET)
total_rg    = pf.metadata.num_row_groups
entries     = []
seen_ids    = set()
scanned     = 0
skipped_lang   = 0
skipped_cap    = 0
skipped_img    = 0
t0 = time.time()

print(f"  row groups: {total_rg}")

for rg_idx in range(total_rg):
    if len(entries) >= TARGET:
        break

    # wit_base schema:
    #   image: struct{bytes, path}
    #   image_url: string
    #   embedding: fixed_size_list[2048]   ← 预计算向量！
    #   wit_features: struct{
    #       language: list[str],
    #       page_title: list[str],
    #       section_title: list[str],
    #       caption_reference_description: list[str],
    #       caption_alt_text_description: list[str],
    #       context_section_description: list[str],
    #       context_page_description: list[str],
    #       ...
    #   }
    batch = pf.read_row_group(rg_idx, columns=[
        "image",
        "image_url",
        "wit_features",
        "embedding",
    ]).to_pydict()

    n = len(batch["image"])
    scanned += n

    image_col    = batch["image"]
    wf_col       = batch["wit_features"]   # list of dicts
    embed_col    = batch["embedding"]       # list of 2048-dim lists

    for i in range(n):
        if len(entries) >= TARGET:
            break

        # wit_features 是嵌套 struct，每个字段是 list（多语言）
        wf = wf_col[i] or {}
        languages = wf.get("language") or []

        # 找英文 index
        try:
            en_idx = languages.index("en")
        except ValueError:
            skipped_lang += 1
            continue

        def _get(key, idx):
            lst = wf.get(key) or []
            val = lst[idx] if idx < len(lst) else None
            return (val or "").strip()

        # caption
        caption = (
            _get("caption_reference_description", en_idx)
            or _get("caption_alt_text_description", en_idx)
            or ""
        )
        if len(caption) < MIN_CAPTION:
            skipped_cap += 1
            continue

        # page_title
        title = _get("page_title", en_idx)
        if not title:
            skipped_cap += 1
            continue

        # 图片 bytes
        img_entry = image_col[i]
        if img_entry is None:
            skipped_img += 1
            continue
        img_bytes = img_entry.get("bytes") if isinstance(img_entry, dict) else None
        if not img_bytes or len(img_bytes) < MIN_IMG:
            skipped_img += 1
            continue

        # wit_id（md5 of image bytes）
        wit_id = "witb_" + hashlib.md5(img_bytes).hexdigest()[:8]
        if wit_id in seen_ids:
            continue
        seen_ids.add(wit_id)

        # 存图片
        img_filename = wit_id + ".jpg"
        img_path = IMAGES_DIR / img_filename
        if not img_path.exists():
            img_path.write_bytes(img_bytes)

        # 辅助字段（从 wit_features 读）
        section   = _get("section_title", en_idx)
        ctx       = (
            _get("context_section_description", en_idx)
            or _get("context_page_description", en_idx)
        )[:300]
        image_url = (batch["image_url"][i] or "").strip()

        # 预计算 embedding（直接存入 meta，建 ChromaDB 时不需要 GPU）
        embedding = embed_col[i]
        if embedding is not None:
            embedding = list(embedding)   # numpy → list

        entries.append({
            "wit_id":         wit_id,
            "page_title":     title,
            "section_title":  section,
            "caption":        caption,
            "context":        ctx,
            "image_filename": img_filename,
            "image_url":      image_url,
            "embedding":      embedding,   # 2048-dim，可直接入 ChromaDB
        })

    # 进度
    done = len(entries)
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    print(f"  rg {rg_idx+1}/{total_rg} | collected={done} "
          f"skip(lang={skipped_lang} cap={skipped_cap} img={skipped_img}) "
          f"rate={rate:.1f}/s", flush=True)

print(f"\n  扫描完成: {len(entries)} 条，扫描行数={scanned}")

if len(entries) < TARGET:
    print(f"  [WARNING] 只收集到 {len(entries)} 条（目标 {TARGET}），考虑扫描更多 shard")

# ── Step 3: 写新 meta.jsonl ──────────────────────────────
print("\n" + "=" * 55)
print("Step 3: 写新 meta.jsonl")
print("=" * 55)

with open(META_JSONL, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"  写入 {len(entries)} 条 → {META_JSONL}")

# ── Step 4: 验证 ─────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 4: 验证")
print("=" * 55)

img_files   = list(IMAGES_DIR.glob("witb_*.jpg"))
no_img      = sum(1 for e in entries if not (IMAGES_DIR / e["image_filename"]).exists())
no_caption  = sum(1 for e in entries if not e["caption"])
dup_ids     = len(entries) - len(set(e["wit_id"] for e in entries))

print(f"  entries:       {len(entries)}")
print(f"  image files:   {len(img_files)}")
print(f"  missing image: {no_img}")
print(f"  empty caption: {no_caption}")
print(f"  dup wit_ids:   {dup_ids}")
print(f"\n  Sample:")
print(json.dumps(entries[0], indent=2, ensure_ascii=False))

status = "PASS" if no_img == 0 and no_caption == 0 and len(entries) >= 1000 else "WARN"
print(f"\n  [{status}] KB 构建{'完成' if status == 'PASS' else '部分完成'}")
print("\n下一步（需要 GPU）:")
print("  bash scripts/start_env.sh")
print("  python scripts/build_chroma_index_v2.py")
print("  python scripts/smoke_test_rag_v2.py")
