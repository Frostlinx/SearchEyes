"""
wit_downloader_hf.py — 从 HuggingFace 下载 WIT 数据集（含图片）
================================================================
使用 wikimedia/wit_base，自带 300px 宽图片，无需爬 Wikimedia URL。
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path


def download_wit_subset(
    output_dir: str | Path,
    count: int = 1000,
    min_caption_len: int = 20,
    language: str = "en",
    seed: int = 42,
) -> Path:
    """
    从 wikimedia/wit_base 下载 count 条高质量英文图文对。

    筛选条件:
    - language == en
    - caption_reference_description 非空且 >= min_caption_len 字符
    - 图片可用（非 None）
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.jsonl"

    print(f"[WIT-HF] 加载 wikimedia/wit_base (streaming)...", flush=True)
    ds = load_dataset("wikimedia/wit_base", split="train", streaming=True)

    collected = 0
    attempted = 0
    max_attempts = count * 20  # 最多尝试 20x 条（很多行没有英文条目）

    with open(meta_path, "w", encoding="utf-8") as f:
        for row in ds:
            if collected >= count:
                break
            attempted += 1
            if attempted > max_attempts:
                print(f"[WIT-HF] 已尝试 {max_attempts} 条仍不足 {count}，停止", flush=True)
                break

            # wit_base 的元数据嵌套在 wit_features 中（列表形式，每个元素对应一种语言）
            wf = row.get("wit_features") or {}
            languages = wf.get("language", [])
            if isinstance(languages, str):
                languages = [languages]

            # 找到目标语言的索引
            en_idx = None
            for idx_lang, lang_val in enumerate(languages):
                if lang_val == language:
                    en_idx = idx_lang
                    break
            if en_idx is None:
                continue

            # Caption 过滤（从对应语言索引取值）
            captions = wf.get("caption_reference_description", [])
            if isinstance(captions, str):
                captions = [captions]
            caption = (captions[en_idx] if en_idx < len(captions) else None) or ""
            if not caption:
                # fallback to caption_attribution_description
                caption = (row.get("caption_attribution_description") or "").strip()
            caption = caption.strip()
            if len(caption) < min_caption_len:
                continue

            # 图片可用性
            image = row.get("image")
            if image is None:
                continue

            # 生成稳定文件名
            page_titles = wf.get("page_title", [])
            if isinstance(page_titles, str):
                page_titles = [page_titles]
            page_title = (page_titles[en_idx] if en_idx < len(page_titles) else "unknown") or "unknown"
            page_title = page_title.strip()
            section_titles = wf.get("section_title", [])
            if isinstance(section_titles, str):
                section_titles = [section_titles]
            section_title = (section_titles[en_idx] if en_idx < len(section_titles) else "") or ""
            section_title = section_title.strip()
            wit_id = f"wit_{collected:04d}"
            img_hash = hashlib.md5(f"{page_title}_{caption[:50]}".encode()).hexdigest()[:12]
            img_filename = f"{wit_id}_{img_hash}.jpg"
            img_path = images_dir / img_filename

            # 保存图片
            try:
                if hasattr(image, "save"):
                    image.save(str(img_path), "JPEG", quality=85)
                else:
                    continue
            except Exception as exc:
                print(f"  [skip] 图片保存失败: {exc}", flush=True)
                continue

            # 写元数据
            meta = {
                "wit_id": wit_id,
                "page_title": page_title,
                "section_title": section_title,
                "caption": caption,
                "image_filename": img_filename,
                "language": language,
                "is_main_image": False,
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            collected += 1

            if collected % 10 == 0:
                print(f"  [{collected}/{count}] 已收集", flush=True)

    print(f"[WIT-HF] 完成: {collected}/{count} 条, 保存到 {output_dir}", flush=True)
    return meta_path


def main():
    parser = argparse.ArgumentParser(description="Download WIT subset from HuggingFace")
    parser.add_argument(
        "--output-dir",
        default="data/wit_subset",
        help="输出目录",
    )
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--min-caption-len", type=int, default=20)
    parser.add_argument("--language", default="en")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_wit_subset(
        output_dir=args.output_dir,
        count=args.count,
        min_caption_len=args.min_caption_len,
        language=args.language,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
