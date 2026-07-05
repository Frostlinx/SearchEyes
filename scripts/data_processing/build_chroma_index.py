#!/usr/bin/env python3
"""Build ChromaDB index for WIT images using the embedding server."""
import sys
from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/QWEN/QWEN-project")
sys.path.insert(0, str(PROJECT_ROOT))

from searcheyes.wit_indexer import index_wit_to_chroma, verify_retrieval

DATA_DIR = PROJECT_ROOT / "data" / "wit_subset_hf"
META_JSONL = DATA_DIR / "meta.jsonl"
IMAGES_DIR = DATA_DIR / "images"
CHROMA_DB = DATA_DIR / "chroma_db"
EMBED_URL = "http://localhost:8766"

print("=" * 60)
print("Phase A3: Building ChromaDB vector index")
print(f"  meta.jsonl: {META_JSONL}")
print(f"  images_dir: {IMAGES_DIR}")
print(f"  chroma_db:  {CHROMA_DB}")
print(f"  embed_url:  {EMBED_URL}")
print("=" * 60)

# Index
count = index_wit_to_chroma(
    meta_jsonl=META_JSONL,
    images_dir=IMAGES_DIR,
    chroma_db_path=CHROMA_DB,
    embedding_server_url=EMBED_URL,
    collection_name="wit_knowledge",
)
print(f"\nIndexed {count} images into ChromaDB")

# Verify
print("\n" + "=" * 60)
print("Verifying retrieval with first image...")
first_img = sorted(IMAGES_DIR.glob("wit_*.*"))[0]
print(f"  Query: {first_img.name}")
hits = verify_retrieval(
    chroma_db_path=CHROMA_DB,
    query_image=first_img,
    embedding_server_url=EMBED_URL,
    collection_name="wit_knowledge",
    top_k=5,
)
print(f"\nTop-1 match: {hits[0]['wit_id'] if hits else 'NONE'}")
if hits and hits[0]['wit_id'] == 'wit_0000':
    print("SELF-RETRIEVAL CHECK: PASSED")
else:
    print("SELF-RETRIEVAL CHECK: UNEXPECTED (first image should match itself)")

print("\nALL DONE")
