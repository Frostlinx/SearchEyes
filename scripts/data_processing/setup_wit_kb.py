#!/usr/bin/env python3
"""
setup_wit_kb.py — 构建 WIT 多模态知识库
=========================================
1. 清理磁盘垃圾文件（v1 旧数据）
2. 从已下载的 google/wit shard 筛选 5000 条英文条目
3. 从 Wikimedia Commons URL 下载对应图片
4. 生成 data/wit_kb_v2/meta.jsonl（含 page_title/section/caption/image）

不碰任何代码文件。
"""
import json
import os
import re
import shutil
import time
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path("/root/autodl-tmp/QWEN/QWEN-project")

# ══════════════════════════════════════════════════════
# Step 1: 磁盘清理（只删 v1 垃圾，不动代码和模型）
# ══════════════════════════════════════════════════════

def cleanup_disk():
    print("\n" + "="*55)
    print("Step 1: 清理磁盘垃圾")
    print("="*55)
    freed = 0

    targets = [
        # v1 rl rollouts (购物世界, env 已换代)
        ROOT / "output/rl_rollouts",
        ROOT / "output/agent_loops",
        ROOT / "output/trajectories",
        ROOT / "output/eval_rollouts",
        ROOT / "output/generator_verify",
        ROOT / "output/diverse_training_data",
        ROOT / "output/synthetic_pages_20260313_160049",
        ROOT / "output/synthetic_pages_20260313_160424",
        # 空 checkpoints (0 或 <100KB)
        ROOT / "checkpoints/grpo_multistep_run2",
        ROOT / "checkpoints/grpo_multistep_run3",
        ROOT / "checkpoints/grpo_multistep_run4",
        ROOT / "checkpoints/sft_skill_driven_test",
        ROOT / "checkpoints/grpo_5090_run1",
        # /tmp 垃圾
        Path("/tmp/wit_sample.parquet.partial"),
        Path("/tmp/patch_al.py"),
        Path("/tmp/patch_tr.py"),
    ]

    for t in targets:
        if t.exists():
            size = sum(f.stat().st_size for f in t.rglob("*") if f.is_file()) if t.is_dir() else t.stat().st_size
            freed += size
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink()
            print(f"  删除: {t.relative_to(ROOT) if str(ROOT) in str(t) else t}  ({size/1e6:.1f} MB)")

    print(f"\n  释放: {freed/1e6:.0f} MB")
    return freed


# ══════════════════════════════════════════════════════
# Step 2: 从 google/wit shard 筛选英文条目
# ══════════════════════════════════════════════════════

def load_wit_shard(shard_path: Path, n_sample: int = 5000, seed: int = 42) -> list[dict]:
    import pyarrow.parquet as pq
    import random

    print("\n" + "="*55)
    print(f"Step 2: 筛选英文 WIT 条目 (目标 {n_sample} 条)")
    print("="*55)

    pf = pq.ParquetFile(str(shard_path))
    total_rows = pf.metadata.num_rows
    print(f"  Shard 总行数: {total_rows:,}")

    rng = random.Random(seed)
    kept = []

    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx)
        df = table.to_pandas()

        # 筛选条件：英文、有 caption、有 image_url、page_title 非空
        mask = (
            (df["language"] == "en") &
            (df["caption_reference_description"].notna()) &
            (df["caption_reference_description"].str.len() > 20) &
            (df["image_url"].notna()) &
            (df["page_title"].notna()) &
            (df["page_title"].str.len() > 2)
        )
        en_rows = df[mask]

        for _, row in en_rows.iterrows():
            kept.append({
                "wit_id": f"wit2_{hashlib.md5(str(row['image_url']).encode()).hexdigest()[:8]}",
                "page_title": str(row.get("page_title", "")).strip(),
                "section_title": str(row.get("section_title", "") or "").strip(),
                "caption": str(row.get("caption_reference_description", "")).strip(),
                "context": str(row.get("context_section_description", "") or "")[:300].strip(),
                "image_url": str(row.get("image_url", "")).strip(),
                "source_url": f"https://en.wikipedia.org/wiki/{str(row.get('page_title','')).replace(' ','_')}",
            })

        print(f"  Row group {rg_idx}: {len(en_rows)} 英文条目 (累计 {len(kept)})")
        if len(kept) >= n_sample * 3:  # 收集 3x 以便多样性采样
            break

    # 多样性采样：按 page_title 去重后再采
    seen_titles = set()
    diverse = []
    for item in kept:
        if item["page_title"] not in seen_titles:
            seen_titles.add(item["page_title"])
            diverse.append(item)

    print(f"  去重后: {len(diverse)} 条唯一文章")

    rng.shuffle(diverse)
    sampled = diverse[:n_sample]
    print(f"  最终采样: {len(sampled)} 条")
    return sampled


# ══════════════════════════════════════════════════════
# Step 3: 下载图片
# ══════════════════════════════════════════════════════

def download_images(entries: list[dict], images_dir: Path,
                    max_workers: int = 8, timeout: int = 15) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n" + "="*55)
    print(f"Step 3: 下载图片 (目标 {len(entries)} 张)")
    print("="*55)

    images_dir.mkdir(parents=True, exist_ok=True)
    results = []
    fail_count = 0

    def download_one(entry: dict) -> dict:
        wit_id = entry["wit_id"]
        url = entry["image_url"]
        filename = f"{wit_id}.jpg"
        dst = images_dir / filename

        if dst.exists() and dst.stat().st_size > 1024:
            entry["image_filename"] = filename
            return entry

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (research bot)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if len(data) < 1024:
                    return None
                with open(dst, "wb") as f:
                    f.write(data)
            entry["image_filename"] = filename
            return entry
        except Exception:
            return None

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_one, e): e for e in entries}
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result:
                results.append(result)
            else:
                fail_count += 1
            if done % 200 == 0 or done == len(entries):
                print(f"  [{done}/{len(entries)}] OK={len(results)} FAIL={fail_count}")

    print(f"\n  完成: {len(results)} 张下载成功, {fail_count} 张失败")
    return results


# ══════════════════════════════════════════════════════
# Step 4: 保存 meta.jsonl
# ══════════════════════════════════════════════════════

def save_kb(entries: list[dict], kb_dir: Path):
    print("\n" + "="*55)
    print("Step 4: 保存知识库")
    print("="*55)
    kb_dir.mkdir(parents=True, exist_ok=True)
    meta_path = kb_dir / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for e in entries:
            # 最终格式与 env 兼容
            record = {
                "wit_id": e["wit_id"],
                "page_title": e["page_title"],
                "section_title": e.get("section_title", ""),
                "caption": e["caption"],
                "context": e.get("context", ""),
                "image_filename": e.get("image_filename", ""),
                "source_url": e["source_url"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  保存: {meta_path}")
    print(f"  条目数: {len(entries)}")
    print(f"  字段: wit_id / page_title / section_title / caption / context / image_filename / source_url")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    print("Visual DreamGym — WIT 知识库构建脚本")
    print("学长要求: 用一部分多模态维基百科数据搭 env")
    print()

    # Step 1
    cleanup_disk()

    # Step 2
    shard_path = Path("/tmp/wit_shard0.parquet")
    if not shard_path.exists():
        print(f"ERROR: {shard_path} 不存在，请先下载")
        print("命令: curl -L -o /tmp/wit_shard0.parquet 'https://hf-mirror.com/api/datasets/google/wit/parquet/default/train/0.parquet'")
        return
    entries = load_wit_shard(shard_path, n_sample=5000)

    # Step 3
    kb_dir = ROOT / "data/wit_kb_v2"
    images_dir = kb_dir / "images"
    entries_with_img = download_images(entries, images_dir)

    # Step 4
    save_kb(entries_with_img, kb_dir)

    # 统计
    print("\n" + "="*55)
    print("完成！")
    print(f"  知识库路径: {kb_dir}")
    print(f"  条目数: {len(entries_with_img)}")
    print(f"  下一步: 启动 embedding server 后运行 build_chroma_index.py 建索引")
    print("="*55)


if __name__ == "__main__":
    main()
