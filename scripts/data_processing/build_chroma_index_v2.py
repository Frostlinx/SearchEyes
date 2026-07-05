#!/usr/bin/env python3
"""
scripts/build_chroma_index_v2.py
Build ChromaDB index for wit_kb_v2 (text-only entries).
Uses caption + title + context as the embedding text.
Run AFTER starting embedding_server.py on port 8766.

Usage:
    python scripts/build_chroma_index_v2.py [--embed-url http://localhost:8766]
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/QWEN/QWEN-project")
sys.path.insert(0, str(PROJECT_ROOT))

META_JSONL   = PROJECT_ROOT / "data/wit_kb_v2/meta.jsonl"
CHROMA_DB    = PROJECT_ROOT / "data/wit_kb_v2/chroma_db"
COLLECTION   = "wit_knowledge_v2"
BATCH_SIZE   = 64
DEFAULT_EMBED_URL = "http://localhost:8766"


def embed_text(text: str, server_url: str) -> list[float] | None:
    parsed = urllib.parse.urlparse(server_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    payload = json.dumps({"text": text}).encode("utf-8")
    try:
        conn.request("POST", "/embed", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        return json.loads(resp.read())["vector"]
    except Exception as e:
        print(f"  [embed error] {e}")
        return None
    finally:
        conn.close()


def build_text_for_entry(e: dict) -> str:
    """Concatenate title + section + caption + context for embedding."""
    parts = []
    if e.get("page_title"):
        parts.append(e["page_title"])
    if e.get("section_title"):
        parts.append(e["section_title"])
    if e.get("caption"):
        parts.append(e["caption"])
    if e.get("context"):
        parts.append(e["context"][:200])
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
    args = parser.parse_args()

    # Health check
    print(f"Checking embedding server at {args.embed_url}...")
    try:
        parsed = urllib.parse.urlparse(args.embed_url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        health = json.loads(resp.read())
        print(f"  OK — dim={health.get('dim')}")
        conn.close()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB))
    # Delete existing collection if present
    try:
        client.delete_collection(COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}'")
    except Exception:
        pass
    col = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    entries = [json.loads(l) for l in META_JSONL.read_text().splitlines() if l.strip()]
    print(f"\nIndexing {len(entries)} entries into ChromaDB '{COLLECTION}'...")
    print(f"  chroma_db: {CHROMA_DB}")
    print(f"  embed_url: {args.embed_url}\n")

    ok = 0
    fail = 0
    t0 = time.time()

    for i, entry in enumerate(entries):
        text = build_text_for_entry(entry)
        vec = embed_text(text, args.embed_url)
        if vec is None:
            fail += 1
            continue

        col.add(
            ids=[entry["wit_id"]],
            embeddings=[vec],
            documents=[text],
            metadatas=[{
                "page_title":   entry.get("page_title", ""),
                "section_title":entry.get("section_title", ""),
                "caption":      entry.get("caption", ""),
                "image_url":    entry.get("image_url", ""),
                "context":      entry.get("context", "")[:200],
            }],
        )
        ok += 1

        if (i + 1) % 200 == 0 or (i + 1) == len(entries):
            elapsed = time.time() - t0
            rate = ok / elapsed if elapsed > 0 else 0
            eta = (len(entries) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(entries)}] ok={ok} fail={fail} "
                  f"rate={rate:.1f}/s eta={eta:.0f}s")

    print(f"\nDone. Indexed {ok}/{len(entries)} entries. "
          f"Failed: {fail}. Elapsed: {time.time()-t0:.0f}s")
    print(f"ChromaDB: {CHROMA_DB}")
    print(f"Collection: {COLLECTION}  count={col.count()}")


if __name__ == "__main__":
    main()
