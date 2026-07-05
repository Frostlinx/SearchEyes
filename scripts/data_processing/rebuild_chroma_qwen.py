#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path('/root/autodl-tmp/QWEN/QWEN-project')
META = ROOT / 'data' / 'wit_kb_v2' / 'meta.jsonl'
IMAGES_DIR = ROOT / 'data' / 'wit_kb_v2' / 'images'
CHROMA_DB = ROOT / 'data' / 'wit_kb_v2' / 'chroma_db'
COLLECTION = 'wit_knowledge_v2_qwen'
EMBEDDING_URL = 'http://localhost:8766'
EXPECTED_COUNT = 2000
EMBED_BATCH_SIZE = 8
CHROMA_BATCH_SIZE = 100
PROGRESS_EVERY = 100
TIMEOUT = 180


def http_get_json(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def http_post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def load_entries(meta_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in meta_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def resolve_image_path(entry: dict[str, Any], images_dir: Path, meta_path: Path) -> Path | None:
    candidates: list[Path] = []

    image_path = str(entry.get('image_path', '')).strip()
    if image_path:
        raw = Path(image_path)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append((meta_path.parent / raw).resolve())
            candidates.append((ROOT / raw).resolve())
            candidates.append((images_dir / raw.name).resolve())

    image_filename = str(entry.get('image_filename', '')).strip()
    if image_filename:
        candidates.append((images_dir / image_filename).resolve())

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def build_text(entry: dict[str, Any]) -> str:
    parts: list[str] = []
    page_title = str(entry.get('page_title', '')).strip()
    section_title = str(entry.get('section_title', '')).strip()
    caption = str(entry.get('caption', '')).strip()
    context = str(entry.get('context', '')).strip()

    if page_title:
        parts.append(f'page_title: {page_title}')
    if section_title:
        parts.append(f'section_title: {section_title}')
    if caption:
        parts.append(f'caption: {caption}')
    if context:
        parts.append(f'context: {context}')

    return '\n'.join(parts).strip()


def build_document(entry: dict[str, Any], text_fallback: str) -> str:
    if text_fallback:
        return text_fallback
    return str(entry.get('wit_id', '')).strip()


def build_metadata(entry: dict[str, Any], embed_mode: str) -> dict[str, Any]:
    return {
        'page_title': str(entry.get('page_title', '')),
        'section_title': str(entry.get('section_title', '')),
        'caption': str(entry.get('caption', '')),
        'context': str(entry.get('context', ''))[:500],
        'image_url': str(entry.get('image_url', '')),
        'source_url': str(entry.get('source_url', '') or entry.get('image_url', '')),
        'image_filename': str(entry.get('image_filename', '')),
        'embed_mode': embed_mode,
    }


def prepare_record(entry: dict[str, Any], meta_path: Path, images_dir: Path) -> dict[str, Any] | None:
    wit_id = str(entry.get('wit_id', '')).strip()
    if not wit_id:
        return None

    image_path = resolve_image_path(entry, images_dir=images_dir, meta_path=meta_path)
    text_fallback = build_text(entry)

    if image_path is not None:
        payload = {'image_path': str(image_path)}
        embed_mode = 'image'
    elif text_fallback:
        payload = {'text': text_fallback}
        embed_mode = 'text'
    else:
        return None

    return {
        'id': wit_id,
        'payload': payload,
        'embed_mode': embed_mode,
        'document': build_document(entry, text_fallback),
        'metadata': build_metadata(entry, embed_mode=embed_mode),
    }


def fetch_batch_embeddings(records: list[dict[str, Any]], embedding_url: str, timeout: int) -> list[list[float] | None]:
    if not records:
        return []

    base_url = embedding_url.rstrip('/')
    batch_url = base_url + '/embed_batch'
    single_url = base_url + '/embed'

    try:
        response = http_post_json(
            batch_url,
            {'items': [record['payload'] for record in records]},
            timeout=timeout,
        )
        vectors = response.get('vectors')
        if not isinstance(vectors, list) or len(vectors) != len(records):
            count_text = len(vectors) if isinstance(vectors, list) else 'invalid'
            raise RuntimeError(f'embed_batch returned {count_text} vectors for {len(records)} records')
        return [vector if isinstance(vector, list) and vector else None for vector in vectors]
    except Exception as exc:
        print(f'[warn] embed_batch failed for {len(records)} records: {exc}')

    vectors: list[list[float] | None] = []
    for record in records:
        record_id = record['id']
        try:
            response = http_post_json(single_url, record['payload'], timeout=timeout)
            vector = response.get('vector')
            vectors.append(vector if isinstance(vector, list) and vector else None)
        except Exception as exc:
            print(f'[warn] embed failed id={record_id}: {exc}')
            vectors.append(None)
    return vectors


def flush_add_buffer(collection: Any, add_buffer: list[dict[str, Any]], counters: dict[str, int], success_ids: list[str]) -> None:
    if not add_buffer:
        return

    def add_many(records: list[dict[str, Any]]) -> None:
        collection.add(
            ids=[record['id'] for record in records],
            embeddings=[record['embedding'] for record in records],
            metadatas=[record['metadata'] for record in records],
            documents=[record['document'] for record in records],
        )

    try:
        add_many(add_buffer)
        for record in add_buffer:
            mode = record['embed_mode']
            counters[f'{mode}_embed'] += 1
            success_ids.append(record['id'])
        add_buffer.clear()
        return
    except Exception as exc:
        print(f'[warn] chroma batch add failed for {len(add_buffer)} records: {exc}')

    failed = 0
    for record in add_buffer:
        record_id = record['id']
        mode = record['embed_mode']
        try:
            add_many([record])
            counters[f'{mode}_embed'] += 1
            success_ids.append(record_id)
        except Exception as exc:
            failed += 1
            counters['fail'] += 1
            print(f'[warn] chroma add failed id={record_id}: {exc}')
    add_buffer.clear()
    if failed:
        print(f'[warn] chroma add permanently failed for {failed} records')


def run() -> int:
    try:
        import chromadb
    except ImportError as exc:
        print(f'[error] chromadb import failed: {exc}')
        return 1

    if not META.exists():
        print(f'[error] meta file not found: {META}')
        return 1

    base_url = EMBEDDING_URL.rstrip('/')
    health_url = base_url + '/health'
    try:
        health = http_get_json(health_url, timeout=TIMEOUT)
        status = health.get('status')
        dim = health.get('dim')
        print(f'[health] status={status} dim={dim} url={EMBEDDING_URL}')
    except urllib.error.URLError as exc:
        print(f'[error] embedding server health check failed: {exc}')
        return 1

    entries = load_entries(META)
    print(f'[config] meta={META}')
    print(f'[config] images_dir={IMAGES_DIR}')
    print(f'[config] chroma_db={CHROMA_DB}')
    print(f'[config] collection={COLLECTION}')
    print(f'[config] entries={len(entries)} expected={EXPECTED_COUNT}')

    CHROMA_DB.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB))

    try:
        client.delete_collection(COLLECTION)
        print(f'[chroma] deleted existing collection={COLLECTION}')
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={'hnsw:space': 'cosine'},
    )

    counters = {'image_embed': 0, 'text_embed': 0, 'fail': 0}
    pending_records: list[dict[str, Any]] = []
    add_buffer: list[dict[str, Any]] = []
    success_ids: list[str] = []

    started_at = time.time()
    total = len(entries)

    for index, entry in enumerate(entries, start=1):
        prepared = prepare_record(entry, meta_path=META, images_dir=IMAGES_DIR)
        if prepared is None:
            counters['fail'] += 1
        else:
            pending_records.append(prepared)

        flush_due_to_batch = len(pending_records) >= EMBED_BATCH_SIZE
        flush_due_to_progress = (index % PROGRESS_EVERY) == 0
        flush_due_to_end = index == total

        if flush_due_to_batch or flush_due_to_progress or flush_due_to_end:
            vectors = fetch_batch_embeddings(
                pending_records,
                embedding_url=EMBEDDING_URL,
                timeout=TIMEOUT,
            )
            for record, vector in zip(pending_records, vectors):
                if vector is None:
                    counters['fail'] += 1
                    continue
                add_buffer.append({
                    'id': record['id'],
                    'embedding': vector,
                    'embed_mode': record['embed_mode'],
                    'metadata': record['metadata'],
                    'document': record['document'],
                })
            pending_records.clear()

        if len(add_buffer) >= CHROMA_BATCH_SIZE or flush_due_to_progress or flush_due_to_end:
            flush_add_buffer(
                collection=collection,
                add_buffer=add_buffer,
                counters=counters,
                success_ids=success_ids,
            )

        if flush_due_to_progress or flush_due_to_end:
            elapsed = time.time() - started_at
            image_embed = counters['image_embed']
            text_embed = counters['text_embed']
            fail = counters['fail']
            print(
                f'[progress] {index}/{total} image_embed={image_embed} '
                f'text_embed={text_embed} fail={fail} elapsed={elapsed:.1f}s'
            )

    count = collection.count()
    elapsed = time.time() - started_at
    image_embed = counters['image_embed']
    text_embed = counters['text_embed']
    fail = counters['fail']
    print(f'[done] count={count} image_embed={image_embed} text_embed={text_embed} fail={fail} elapsed={elapsed:.1f}s')

    if count != EXPECTED_COUNT:
        print(f'[error] collection count mismatch: expected={EXPECTED_COUNT} actual={count}')
        return 1

    if not success_ids:
        print('[error] no successful records were inserted')
        return 1

    sample_id = random.choice(success_ids)
    sample = collection.get(ids=[sample_id], include=['embeddings', 'metadatas'])
    embeddings = sample.get('embeddings')
    metadatas = sample.get('metadatas')
    if embeddings is None:
        embeddings = []
    if metadatas is None:
        metadatas = []
    if len(embeddings) == 0 or len(embeddings[0]) == 0:
        print(f'[error] failed to fetch sample embedding for id={sample_id}')
        return 1

    query = collection.query(query_embeddings=[embeddings[0]], n_results=3)
    print(f'[verify] random_sample_id={sample_id}')
    for rank, (wit_id, distance, metadata) in enumerate(
        zip(
            query.get('ids', [[]])[0],
            query.get('distances', [[]])[0],
            query.get('metadatas', [[]])[0],
        ),
        start=1,
    ):
        title = str((metadata or {}).get('page_title', ''))[:60]
        caption = str((metadata or {}).get('caption', ''))[:80]
        print(f'[verify] top-{rank} wit_id={wit_id} distance={distance:.4f} title={title!r} caption={caption!r}')

    sample_meta = metadatas[0] if metadatas else {}
    sample_title = str(sample_meta.get('page_title', ''))[:60]
    print(f'[pass] collection={COLLECTION} count={count} sample_title={sample_title!r}')
    return 0


if __name__ == '__main__':
    sys.exit(run())
