"""
wit_indexer.py — WIT 数据集下载 + ChromaDB 向量索引
===================================================
Phase A1/A3：下载 WIT (Wikipedia Image-Text) 子集，
用 embedding server 提取向量，存入 ChromaDB。

用法:
    # 1. 下载 100 条 WIT 子集
    python searcheyes/wit_indexer.py --phase download --count 100

    # 2. 建立向量索引（需要先启动 embedding_server.py）
    python searcheyes/wit_indexer.py --phase index --embedding-url http://localhost:8766

    # 3. 验证检索质量
    python searcheyes/wit_indexer.py --phase verify --query-image data/wit_subset/images/wit_0000.jpg
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# WIT 数据集 Google Cloud Storage 地址（第一个 shard）
_WIT_SHARD_URL = (
    "https://storage.googleapis.com/gresearch/wit/"
    "wit_v1.train.all-00000-of-00010.tsv.gz"
)

# WIT TSV 列名
_WIT_COLUMNS = [
    "language", "page_url", "image_url", "page_title", "section_title",
    "hierarchical_section_title", "caption_reference_description",
    "caption_attribution_description", "caption_alt_text_description",
    "mime_type", "original_height", "original_width", "is_main_image",
    "attribution_passes_lang_id", "page_changed_recently",
    "context_page_description", "context_section_description",
]


# ══════════════════════════════════════════════════════════
# Phase A1：下载 WIT 子集
# ══════════════════════════════════════════════════════════

def download_wit_subset(
    output_dir: Path,
    count: int = 100,
    source: str = "",
    max_scan: int = 2000,
    image_timeout: int = 10,
) -> Path:
    """下载 WIT 子集到 output_dir。

    Args:
        output_dir: 输出目录（会创建 images/ 和 meta.jsonl）
        count: 目标下载数量
        source: WIT TSV 路径或 URL（空则用默认 Google URL）
        max_scan: 最多扫描多少行 TSV
        image_timeout: 下载单张图片的超时秒数

    Returns:
        output_dir 路径
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.jsonl"

    # 如果已经有足够数据，跳过
    if meta_path.exists():
        existing = sum(1 for _ in meta_path.open())
        if existing >= count:
            print(f"[wit] 已有 {existing} 条数据，跳过下载")
            return output_dir

    source = source or _WIT_SHARD_URL
    rows = _stream_wit_rows(source, max_scan)

    collected = 0
    meta_lines: list[str] = []

    for row in rows:
        if collected >= count:
            break

        # 筛选条件：英文、有描述、图片够大
        if not _is_good_row(row):
            continue

        image_url = row.get("image_url", "")
        if not image_url:
            continue

        # 下载图片
        wit_id = f"wit_{collected:04d}"
        ext = _guess_extension(image_url, row.get("mime_type", ""))
        filename = f"{wit_id}{ext}"
        save_path = images_dir / filename

        if not save_path.exists():
            ok = _download_image(image_url, save_path, timeout=image_timeout)
            if not ok:
                continue
            time.sleep(0.5)  # 限流：每张图间隔 0.5s

        # 记录元数据
        caption = (
            row.get("caption_reference_description", "")
            or row.get("caption_alt_text_description", "")
            or row.get("caption_attribution_description", "")
        )
        meta = {
            "wit_id": wit_id,
            "page_title": row.get("page_title", ""),
            "section_title": row.get("section_title", ""),
            "caption": caption.strip(),
            "image_filename": filename,
            "source_url": row.get("page_url", ""),
            "image_url": image_url,
            "width": int(row.get("original_width", 0) or 0),
            "height": int(row.get("original_height", 0) or 0),
        }
        meta_lines.append(json.dumps(meta, ensure_ascii=False))
        collected += 1
        print(f"  [{collected}/{count}] {wit_id}: {meta['page_title'][:50]}")

    # 写 meta.jsonl
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    print(f"[wit] 下载完成: {collected} 条 → {meta_path}")
    return output_dir


def _stream_wit_rows(source: str, max_rows: int):
    """从 WIT TSV（本地文件或 URL）流式读取行。"""
    if source.startswith("http://") or source.startswith("https://"):
        print(f"[wit] 从 URL 流式读取: {source[:80]}...")
        req = urllib.request.Request(source, headers={"User-Agent": "WIT-Indexer/1.0"})
        response = urllib.request.urlopen(req, timeout=60)
        if source.endswith(".gz"):
            stream = gzip.open(response, mode="rt", encoding="utf-8", errors="replace")
        else:
            stream = io.TextIOWrapper(response, encoding="utf-8", errors="replace")
    else:
        local = Path(source)
        if not local.exists():
            raise FileNotFoundError(f"WIT TSV 文件不存在: {source}")
        print(f"[wit] 从本地文件读取: {source}")
        if source.endswith(".gz"):
            stream = gzip.open(local, mode="rt", encoding="utf-8", errors="replace")
        else:
            stream = open(local, encoding="utf-8", errors="replace")

    reader = csv.DictReader(stream, delimiter="\t", fieldnames=_WIT_COLUMNS)
    yielded = 0
    for row in reader:
        if yielded >= max_rows:
            break
        yielded += 1
        yield row

    stream.close()


def _is_good_row(row: dict) -> bool:
    """筛选高质量 WIT 条目。"""
    lang = row.get("language", "")
    if lang != "en":
        return False

    caption = row.get("caption_reference_description", "").strip()
    if not caption or len(caption) < 10:
        return False

    try:
        width = int(row.get("original_width", 0) or 0)
    except (ValueError, TypeError):
        width = 0
    if width < 200:
        return False

    return True


def _guess_extension(url: str, mime_type: str) -> str:
    """从 URL 或 MIME 类型猜测文件扩展名。"""
    mime_map = {
        "image/jpeg": ".jpg", "image/jpg": ".jpg",
        "image/png": ".png", "image/gif": ".gif",
        "image/webp": ".webp", "image/svg+xml": ".svg",
    }
    if mime_type in mime_map:
        return mime_map[mime_type]
    url_lower = url.lower().split("?")[0]
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        if url_lower.endswith(ext):
            return ext
    return ".jpg"


def _download_image(url: str, save_path: Path, timeout: int = 15, max_retries: int = 2) -> bool:
    """下载单张图片，带限流重试。失败返回 False。"""
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if len(data) < 1000:
                    return False
                save_path.write_bytes(data)
                return True
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                wait = 2 * (attempt + 1)
                time.sleep(wait)
                continue
            print(f"    [skip] 下载失败: {str(exc)[:60]}")
            return False
        except Exception as exc:
            print(f"    [skip] 下载失败: {str(exc)[:60]}")
            return False
    return False


# ══════════════════════════════════════════════════════════
# Phase A3：构建 ChromaDB 向量索引
# ══════════════════════════════════════════════════════════

def index_wit_to_chroma(
    meta_jsonl: Path,
    images_dir: Path,
    chroma_db_path: Path,
    embedding_server_url: str = "http://localhost:8766",
    collection_name: str = "wit_knowledge",
) -> int:
    """把 WIT 图片通过 embedding server 向量化后存入 ChromaDB。

    Returns:
        成功索引的条目数
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_db_path))

    # 如果 collection 已存在，先删除重建
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # collection 不存在或其他错误，忽略
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    items = _load_meta(meta_jsonl)
    indexed = 0

    for item in items:
        image_path = images_dir / item["image_filename"]
        if not image_path.exists():
            print(f"  [skip] 图片不存在: {image_path.name}")
            continue

        vector = _call_embedding_server(str(image_path), embedding_server_url)
        if vector is None:
            print(f"  [skip] embedding 失败: {item['wit_id']}")
            continue

        collection.add(
            ids=[item["wit_id"]],
            embeddings=[vector],
            metadatas=[{
                "page_title": item.get("page_title", ""),
                "caption": item.get("caption", ""),
                "source_url": item.get("source_url", ""),
                "image_filename": item.get("image_filename", ""),
            }],
        )
        indexed += 1
        if indexed % 10 == 0:
            print(f"  [index] {indexed}/{len(items)}")

    print(f"[wit] 索引完成: {indexed} 条 → {chroma_db_path}")
    return indexed


def verify_retrieval(
    chroma_db_path: Path,
    query_image: Path,
    embedding_server_url: str = "http://localhost:8766",
    collection_name: str = "wit_knowledge",
    top_k: int = 5,
) -> list[dict]:
    """验证检索质量：给一张图查 top-k。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = client.get_collection(collection_name)

    vector = _call_embedding_server(str(query_image), embedding_server_url)
    if vector is None:
        print("[verify] embedding 失败")
        return []

    results = collection.query(query_embeddings=[vector], n_results=top_k)
    hits: list[dict] = []
    for i in range(len(results["ids"][0])):
        hit = {
            "rank": i + 1,
            "wit_id": results["ids"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
            **results["metadatas"][0][i],
        }
        hits.append(hit)
        print(
            f"  #{hit['rank']} [{hit['wit_id']}] "
            f"dist={hit['distance']:.4f} "
            f"title={hit.get('page_title', '')[:40]}"
        )

    return hits


# ══════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════

def _load_meta(meta_jsonl: Path) -> list[dict]:
    """读取 meta.jsonl。"""
    items = []
    for line in meta_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _call_embedding_server(image_path: str, server_url: str) -> list[float] | None:
    """调用 embedding server 获取图片向量。"""
    import http.client
    import urllib.parse

    payload = json.dumps({"image_path": image_path}).encode("utf-8")
    parsed = urllib.parse.urlsplit(f"{server_url}/embed")
    conn_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_cls(parsed.hostname, parsed.port, timeout=30)

    try:
        path = parsed.path or "/"
        conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status >= 400:
            print(f"  [embed] HTTP {resp.status}: {raw.decode('utf-8', errors='ignore')[:80]}")
            return None
        data = json.loads(raw)
        return data.get("vector")
    except Exception as exc:
        print(f"  [embed] 连接失败: {exc}")
        return None
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="WIT 数据集下载与 ChromaDB 索引")
    parser.add_argument(
        "--phase",
        choices=["download", "index", "verify"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "wit_subset"),
    )
    parser.add_argument("--count", type=int, default=100, help="下载数量")
    parser.add_argument("--source", default="", help="WIT TSV 路径或 URL")
    parser.add_argument("--max-scan", type=int, default=2000, help="最多扫描 TSV 行数")
    parser.add_argument(
        "--embedding-url",
        default="http://localhost:8766",
        help="embedding_server.py 地址",
    )
    parser.add_argument("--query-image", default="", help="verify 阶段的查询图片路径")
    parser.add_argument("--collection", default="wit_knowledge")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.phase == "download":
        download_wit_subset(
            output_dir=output_dir,
            count=args.count,
            source=args.source,
            max_scan=args.max_scan,
        )

    elif args.phase == "index":
        meta_jsonl = output_dir / "meta.jsonl"
        images_dir = output_dir / "images"
        chroma_db_path = output_dir / "chroma_db"
        if not meta_jsonl.exists():
            raise SystemExit(f"找不到 {meta_jsonl}，请先运行 --phase download")
        index_wit_to_chroma(
            meta_jsonl=meta_jsonl,
            images_dir=images_dir,
            chroma_db_path=chroma_db_path,
            embedding_server_url=args.embedding_url,
            collection_name=args.collection,
        )

    elif args.phase == "verify":
        chroma_db_path = output_dir / "chroma_db"
        query = args.query_image
        if not query:
            # 默认用第一张图
            images_dir = output_dir / "images"
            first = sorted(images_dir.glob("wit_*.*"))
            if not first:
                raise SystemExit("没有图片可用于验证")
            query = str(first[0])
        verify_retrieval(
            chroma_db_path=chroma_db_path,
            query_image=Path(query),
            embedding_server_url=args.embedding_url,
            collection_name=args.collection,
        )


if __name__ == "__main__":
    main()
