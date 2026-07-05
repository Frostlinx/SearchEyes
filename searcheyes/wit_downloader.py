"""
wit_downloader.py — WIT 数据集下载与处理
==========================================
从 Google 的 WIT (Wikipedia Image-Text) 数据集中下载并筛选
高质量图文对，用于构建多模态 RAG 知识库。

数据来源: https://github.com/google-research-datasets/wit
"""

from __future__ import annotations
import csv
import gzip
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse
import hashlib

import requests

logger = logging.getLogger(__name__)

# WIT TSV 数据的 GCS 下载地址（英文子集，共 10 个 shard）
WIT_BASE_URL = "https://storage.googleapis.com/gresearch/wit"
WIT_SHARD_TEMPLATE = "wit_v1.train.all-{shard_id:05d}-of-00010.tsv.gz"

# 优先选择的视觉丰富类别关键词
VISUAL_CATEGORIES = {
    "electronics", "circuit", "computer", "chip", "processor",
    "animal", "bird", "insect", "fish", "mammal",
    "architecture", "building", "bridge", "tower", "church",
    "food", "cuisine", "dish", "fruit", "vegetable",
    "vehicle", "car", "aircraft", "ship", "train",
    "art", "painting", "sculpture", "museum",
    "geography", "mountain", "river", "lake", "island",
    "plant", "flower", "tree", "forest",
    "sport", "stadium", "athlete",
    "instrument", "guitar", "piano",
}

# WIT TSV 的列名
WIT_COLUMNS = [
    "language", "page_url", "image_url", "page_title",
    "section_title", "hierarchical_section_title",
    "caption_reference_description", "caption_attribution_description",
    "caption_alt_text_description", "mime_type",
    "original_height", "original_width",
    "is_main_image", "attribution_passes_lang_id",
    "page_changed_recently", "glob_url",
    "context_page_description", "context_section_description",
]


@dataclass
class WITEntry:
    """一条 WIT 图文对"""
    wit_id: str
    page_title: str
    section_title: str
    caption: str
    fact_text: str
    image_url: str
    image_local_path: str = ""
    original_width: int = 0
    original_height: int = 0
    page_url: str = ""
    context_section: str = ""
    category_keywords: list[str] = None

    def __post_init__(self):
        if self.category_keywords is None:
            self.category_keywords = []


def _make_wit_id(image_url: str, page_title: str) -> str:
    """根据 URL + 标题生成稳定的 WIT ID"""
    raw = f"{image_url}|{page_title}"
    return f"wit_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


def _build_fact_text(entry: dict) -> str:
    """拼接 caption + context 生成 fact_text"""
    parts = []
    title = entry.get("page_title", "").strip()
    if title:
        parts.append(title)
    caption = (entry.get("caption_reference_description") or
               entry.get("caption_alt_text_description") or
               entry.get("caption_attribution_description") or "").strip()
    if caption:
        parts.append(caption)
    ctx = (entry.get("context_section_description") or "").strip()
    if ctx and len(ctx) < 500:
        parts.append(ctx)
    return " — ".join(parts) if parts else ""


def _extract_categories(entry: dict) -> list[str]:
    """从标题/section/context 中提取匹配的视觉类别"""
    text = " ".join([
        entry.get("page_title", ""),
        entry.get("section_title", ""),
        entry.get("context_section_description", ""),
    ]).lower()
    return [kw for kw in VISUAL_CATEGORIES if kw in text]


def _is_high_quality(entry: dict) -> bool:
    """过滤高质量条目"""
    # 仅英文
    if entry.get("language", "") != "en":
        return False
    # 必须有图片 URL
    if not entry.get("image_url"):
        return False
    # 必须有 caption
    has_caption = any([
        entry.get("caption_reference_description"),
        entry.get("caption_alt_text_description"),
        entry.get("caption_attribution_description"),
    ])
    if not has_caption:
        return False
    # 图片尺寸 >= 200x200
    try:
        w = int(entry.get("original_width", 0))
        h = int(entry.get("original_height", 0))
        if w < 200 or h < 200:
            return False
    except (ValueError, TypeError):
        return False
    # 必须是主图
    if entry.get("is_main_image", "").lower() not in ("true", "1", "yes"):
        return False
    return True


def parse_wit_tsv(
    tsv_path: str | Path,
    max_entries: int = 2000,
    prioritize_visual: bool = True,
) -> list[WITEntry]:
    """
    解析 WIT TSV 文件，返回筛选后的 WITEntry 列表。

    Args:
        tsv_path: TSV(.gz) 文件路径
        max_entries: 最大返回条目数
        prioritize_visual: 是否优先选择视觉丰富类别
    """
    tsv_path = Path(tsv_path)
    opener = gzip.open if tsv_path.suffix == ".gz" else open

    visual_entries = []
    other_entries = []

    with opener(tsv_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=WIT_COLUMNS)
        # 跳过可能的表头行
        first_row = next(reader, None)
        if first_row and first_row.get("language") == "language":
            pass  # 跳过表头
        elif first_row:
            # 第一行就是数据
            if _is_high_quality(first_row):
                cats = _extract_categories(first_row)
                entry = _row_to_entry(first_row, cats)
                if cats:
                    visual_entries.append(entry)
                else:
                    other_entries.append(entry)

        for row in reader:
            if len(visual_entries) + len(other_entries) >= max_entries * 2:
                break  # 读够了
            if not _is_high_quality(row):
                continue
            cats = _extract_categories(row)
            entry = _row_to_entry(row, cats)
            if cats:
                visual_entries.append(entry)
            else:
                other_entries.append(entry)

    # 优先选视觉丰富类别
    if prioritize_visual:
        result = visual_entries[:max_entries]
        remaining = max_entries - len(result)
        if remaining > 0:
            result.extend(other_entries[:remaining])
    else:
        combined = visual_entries + other_entries
        result = combined[:max_entries]

    logger.info(
        f"WIT 解析完成: {len(visual_entries)} 视觉丰富 + "
        f"{len(other_entries)} 其他 → 选出 {len(result)} 条"
    )
    return result


def _row_to_entry(row: dict, categories: list[str]) -> WITEntry:
    """将 TSV 行转为 WITEntry"""
    fact_text = _build_fact_text(row)
    wit_id = _make_wit_id(row.get("image_url", ""), row.get("page_title", ""))
    return WITEntry(
        wit_id=wit_id,
        page_title=row.get("page_title", ""),
        section_title=row.get("section_title", ""),
        caption=(row.get("caption_reference_description") or
                 row.get("caption_alt_text_description") or
                 row.get("caption_attribution_description") or ""),
        fact_text=fact_text,
        image_url=row.get("image_url", ""),
        original_width=int(row.get("original_width", 0) or 0),
        original_height=int(row.get("original_height", 0) or 0),
        page_url=row.get("page_url", ""),
        context_section=row.get("context_section_description", ""),
        category_keywords=categories,
    )


def download_images(
    entries: list[WITEntry],
    output_dir: str | Path,
    timeout: int = 15,
    max_retries: int = 2,
) -> list[WITEntry]:
    """
    下载 WIT 图片到本地。

    Args:
        entries: WITEntry 列表
        output_dir: 图片保存目录
        timeout: 单张下载超时
        max_retries: 重试次数

    Returns:
        成功下载的 WITEntry 列表（image_local_path 已填充）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = []
    failed = 0
    session = requests.Session()
    session.headers.update({
        "User-Agent": "WIT-Research-Downloader/1.0 (academic research)"
    })

    for i, entry in enumerate(entries):
        # 生成文件名
        ext = _guess_extension(entry.image_url)
        local_name = f"{entry.wit_id}{ext}"
        local_path = output_dir / local_name

        # 已下载则跳过
        if local_path.exists() and local_path.stat().st_size > 1000:
            entry.image_local_path = str(local_path)
            success.append(entry)
            continue

        # 下载
        ok = False
        for attempt in range(max_retries + 1):
            try:
                resp = session.get(entry.image_url, timeout=timeout, stream=True)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                # 验证文件大小
                if local_path.stat().st_size < 1000:
                    local_path.unlink(missing_ok=True)
                    continue
                entry.image_local_path = str(local_path)
                success.append(entry)
                ok = True
                break
            except Exception as e:
                if attempt == max_retries:
                    logger.warning(f"[{i+1}/{len(entries)}] 下载失败: {entry.wit_id} - {e}")
                    failed += 1
                else:
                    time.sleep(0.5)

        if (i + 1) % 50 == 0:
            logger.info(f"下载进度: {i+1}/{len(entries)}, 成功={len(success)}, 失败={failed}")

    logger.info(f"图片下载完成: {len(success)} 成功, {failed} 失败")
    return success


def _guess_extension(url: str) -> str:
    """从 URL 猜测图片扩展名"""
    path = urlparse(url).path.lower()
    if path.endswith(".png"):
        return ".png"
    elif path.endswith(".gif"):
        return ".gif"
    elif path.endswith(".webp"):
        return ".webp"
    return ".jpg"


def download_wit_shard(
    shard_id: int = 0,
    output_path: str | Path = "data/wit_raw/",
) -> Path:
    """
    下载 WIT 的一个 TSV shard 文件。

    Args:
        shard_id: shard 编号 (0-9)
        output_path: 保存目录
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = WIT_SHARD_TEMPLATE.format(shard_id=shard_id)
    url = f"{WIT_BASE_URL}/{filename}"
    local_file = output_path / filename

    if local_file.exists():
        logger.info(f"已存在: {local_file}")
        return local_file

    logger.info(f"下载 WIT shard {shard_id}: {url}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(local_file, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (50 * 1024 * 1024) == 0:
                pct = downloaded / total * 100
                logger.info(f"  {pct:.1f}% ({downloaded // (1024*1024)} MB)")

    logger.info(f"下载完成: {local_file} ({local_file.stat().st_size // (1024*1024)} MB)")
    return local_file


def save_metadata(entries: list[WITEntry], path: str | Path):
    """保存元数据为 JSONL"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            line = json.dumps(asdict(entry), ensure_ascii=False)
            f.write(line + "\n")
    logger.info(f"保存 {len(entries)} 条元数据到 {path}")


def load_metadata(path: str | Path) -> list[WITEntry]:
    """从 JSONL 加载元数据"""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            entries.append(WITEntry(**data))
    return entries


def build_wit_subset(
    shard_id: int = 0,
    target_count: int = 100,
    data_dir: str | Path = "data",
) -> list[WITEntry]:
    """
    端到端构建 WIT 子集：下载 shard → 解析筛选 → 下载图片 → 保存元数据。

    Args:
        shard_id: WIT shard 编号
        target_count: 目标条目数
        data_dir: 数据根目录

    Returns:
        成功处理的 WITEntry 列表
    """
    data_dir = Path(data_dir)

    # Step 1: 下载 TSV
    logger.info(f"=== Step 1: 下载 WIT shard {shard_id} ===")
    tsv_path = download_wit_shard(shard_id, data_dir / "wit_raw")

    # Step 2: 解析筛选（多下载一些以应对图片 404）
    logger.info(f"=== Step 2: 解析筛选（目标 {target_count} 条）===")
    candidates = parse_wit_tsv(tsv_path, max_entries=target_count * 2)

    # Step 3: 下载图片
    logger.info(f"=== Step 3: 下载 {len(candidates)} 张候选图片 ===")
    success = download_images(candidates, data_dir / "wit_subset" / "images")

    # 裁剪到目标数量
    result = success[:target_count]

    # Step 4: 保存元数据
    logger.info(f"=== Step 4: 保存 {len(result)} 条元数据 ===")
    save_metadata(result, data_dir / "wit_subset" / "metadata.jsonl")

    # 统计
    stats = {
        "total_parsed": len(candidates),
        "images_downloaded": len(success),
        "final_count": len(result),
        "visual_categories": sum(1 for e in result if e.category_keywords),
        "avg_fact_length": sum(len(e.fact_text) for e in result) / max(len(result), 1),
    }
    stats_path = data_dir / "wit_subset" / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"统计信息: {stats}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="WIT 数据集下载与处理")
    parser.add_argument("--shard", type=int, default=0, help="WIT shard 编号 (0-9)")
    parser.add_argument("--count", type=int, default=100, help="目标条目数")
    parser.add_argument("--data-dir", type=str, default="data", help="数据目录")
    args = parser.parse_args()

    entries = build_wit_subset(
        shard_id=args.shard,
        target_count=args.count,
        data_dir=args.data_dir,
    )
    print(f"\n完成！共处理 {len(entries)} 条 WIT 图文对")
