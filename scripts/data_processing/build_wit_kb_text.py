#!/usr/bin/env python3
"""
build_wit_kb_text.py — Build text-only WIT KB from /tmp/wit_shard0.parquet
No image download needed. image_url is stored for future use.
Output: data/wit_kb_v2/meta.jsonl
"""
import json
import hashlib
import random
from pathlib import Path

import pyarrow.parquet as pq

PARQUET = Path("/tmp/wit_shard0.parquet")
OUT_DIR = Path("/root/autodl-tmp/QWEN/QWEN-project/data/wit_kb_v2")
OUT_META = OUT_DIR / "meta.jsonl"
TARGET = 5000
SEED = 42

random.seed(SEED)

print("=" * 55)
print("Building text-only WIT KB")
print(f"Source: {PARQUET}")
print(f"Target: {TARGET} entries")
print("=" * 55)

# ── scan all row groups ───────────────────────────────────
pf = pq.ParquetFile(PARQUET)
entries = []
seen_titles = set()

for rg_idx in range(pf.metadata.num_row_groups):
    batch = pf.read_row_group(rg_idx, columns=[
        "language", "page_title", "section_title",
        "caption_reference_description", "caption_alt_text_description",
        "context_section_description", "image_url",
    ]).to_pydict()

    n = len(batch["language"])
    for i in range(n):
        lang = batch["language"][i] or ""
        if lang != "en":
            continue

        title = (batch["page_title"][i] or "").strip()
        if not title or title in seen_titles:
            continue

        caption = (
            batch["caption_reference_description"][i]
            or batch["caption_alt_text_description"][i]
            or ""
        ).strip()
        if len(caption) < 20:
            continue

        image_url = (batch["image_url"][i] or "").strip()
        if not image_url:
            continue

        ctx = (batch["context_section_description"][i] or "").strip()[:300]
        section = (batch["section_title"][i] or "").strip()

        wit_id = "wit2_" + hashlib.md5(image_url.encode()).hexdigest()[:8]
        seen_titles.add(title)
        entries.append({
            "wit_id": wit_id,
            "page_title": title,
            "section_title": section,
            "caption": caption,
            "context": ctx,
            "image_filename": None,    # placeholder — images not yet downloaded
            "image_url": image_url,    # stored for future download
        })

print(f"\nTotal English unique articles: {len(entries)}")

# ── sample ────────────────────────────────────────────────
if len(entries) > TARGET:
    entries = random.sample(entries, TARGET)
print(f"Sampled: {len(entries)}")

# ── write ─────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "images").mkdir(exist_ok=True)   # keep dir for future use

with open(OUT_META, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"\nWrote {len(entries)} entries → {OUT_META}")
print("\nSample entry:")
print(json.dumps(entries[0], indent=2, ensure_ascii=False))
print("\nDone. Text-only KB ready.")
