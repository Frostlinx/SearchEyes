#!/usr/bin/env python3
"""
extract_wit_from_parquet.py — 从 WIT parquet 分片提取 1000 条高质量英文条目
==========================================================================
parquet 文件内嵌 PIL Image，直接提取图片到磁盘 + 生成 meta.jsonl。

用法:
    python scripts/extract_wit_from_parquet.py \
        --parquet data/wit_shard_000.parquet \
        --output-dir data/wit_subset_hf \
        --count 1000
"""

import argparse
import json
import sys
from pathlib import Path


def extract_wit(parquet_path: str, output_dir: str, count: int = 1000):
    import pandas as pd

    out = Path(output_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.jsonl"

    print(f"[WIT-Extract] 读取 parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[WIT-Extract] 总行数: {len(df)}")

    # 过滤英文条目，提取有效 caption
    results = []
    skipped = 0

    def _to_list(val):
        """将 numpy array / scalar / list 统一转为 Python list"""
        if val is None:
            return []
        try:
            return list(val)
        except TypeError:
            return [val]

    for idx, row in df.iterrows():
        if len(results) >= count:
            break

        wf = row.get("wit_features")
        if not isinstance(wf, dict):
            skipped += 1
            continue

        languages = _to_list(wf.get("language"))

        # 找英文索引
        en_idx = None
        for i, lang in enumerate(languages):
            if str(lang) == "en":
                en_idx = i
                break
        if en_idx is None:
            skipped += 1
            continue

        # 提取 caption
        captions = _to_list(wf.get("caption_reference_description"))
        caption = captions[en_idx] if en_idx < len(captions) else None
        if not caption or len(str(caption)) < 15:
            skipped += 1
            continue
        caption = str(caption)

        # 提取 page_title
        titles = _to_list(wf.get("page_title"))
        page_title = str(titles[en_idx]) if en_idx < len(titles) else ""

        # 提取图片（parquet 内嵌 PIL Image）
        image = row.get("image")
        if image is None:
            skipped += 1
            continue

        wit_id = f"wit_{len(results):04d}"
        image_filename = f"{wit_id}.jpg"
        image_path = images_dir / image_filename

        try:
            # image 可能是 dict {'bytes': ..., 'path': ...} 或 PIL Image
            if isinstance(image, dict):
                img_bytes = image.get("bytes")
                if img_bytes:
                    image_path.write_bytes(img_bytes)
                else:
                    skipped += 1
                    continue
            else:
                # PIL Image
                image.save(str(image_path), "JPEG", quality=85)
        except Exception as e:
            print(f"  [skip] 图片保存失败 idx={idx}: {e}")
            skipped += 1
            continue

        # 提取 source_url
        page_urls = _to_list(wf.get("page_url"))
        source_url = str(page_urls[en_idx]) if en_idx < len(page_urls) else ""

        results.append({
            "wit_id": wit_id,
            "page_title": page_title or "",
            "caption": caption,
            "image_filename": image_filename,
            "source_url": source_url or "",
        })

        if len(results) % 100 == 0:
            print(f"  [WIT-Extract] {len(results)}/{count} 条已提取 (跳过 {skipped})")

    # 写 meta.jsonl
    with open(meta_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[WIT-Extract] 完成!")
    print(f"  提取: {len(results)} 条")
    print(f"  跳过: {skipped} 条")
    print(f"  图片: {images_dir}")
    print(f"  元数据: {meta_path}")

    # 验证
    actual_images = len(list(images_dir.glob("wit_*.jpg")))
    print(f"  图片文件数: {actual_images}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="parquet 分片路径")
    parser.add_argument("--output-dir", default="data/wit_subset_hf", help="输出目录")
    parser.add_argument("--count", type=int, default=1000, help="提取条目数")
    args = parser.parse_args()

    extract_wit(args.parquet, args.output_dir, args.count)
