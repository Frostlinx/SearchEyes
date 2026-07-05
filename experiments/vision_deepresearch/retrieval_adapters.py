"""
retrieval_adapters.py — A / B / A+B 三种检索方式统一封装

OptionA: image → Qwen3-VL-Embed → ChromaDB
OptionB: image → CaptionBridge → text → text_embed + BM25
OptionAB: 两路结果 RRF 融合
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 把项目根加入 sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_URL, TOP_K_DEFAULT
)


@dataclass
class RetrievalResult:
    wit_id: str
    title: str
    caption: str
    score: float
    rank: int
    source: str  # "image_embed" | "text_embed" | "bm25" | "fused"


@dataclass
class RetrievalResponse:
    results: list[RetrievalResult]
    query_used: str        # 实际使用的检索query
    mode: str              # "A" | "B" | "AB"
    hit_gt: bool = False
    gt_rank: int = -1      # 1-based，-1表示未命中

    def compute_gt_rank(self, gt_wit_id: str, miss_rank: int = 101) -> int:
        for r in self.results:
            if r.wit_id == gt_wit_id:
                self.gt_rank = r.rank
                self.hit_gt = True
                return r.rank
        self.gt_rank = miss_rank  # 明确的未命中值，不依赖 results 长度
        self.hit_gt = False
        return self.gt_rank


def _http_post(url: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_collection(COLLECTION_NAME)


def _chroma_query(vector: list[float], top_k: int) -> list[RetrievalResult]:
    col = _get_chroma_collection()
    res = col.query(query_embeddings=[vector], n_results=top_k)
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    results = []
    for rank, (wit_id, dist, meta) in enumerate(zip(ids, dists, metas), start=1):
        score = max(0.0, 1.0 - dist / 2.0)  # monotonic, matches multimodal_rag.py
        results.append(RetrievalResult(
            wit_id=wit_id,
            title=meta.get("page_title", ""),
            caption=meta.get("caption", ""),
            score=score,
            rank=rank,
            source="image_embed",
        ))
    return results


def _rrf_fuse(list_a: list[RetrievalResult], list_b: list[RetrievalResult],
              top_k: int, k: int = 60) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion"""
    scores: dict[str, float] = {}
    meta_map: dict[str, RetrievalResult] = {}
    for r in list_a:
        scores[r.wit_id] = scores.get(r.wit_id, 0.0) + 1.0 / (k + r.rank)
        meta_map[r.wit_id] = r
    for r in list_b:
        scores[r.wit_id] = scores.get(r.wit_id, 0.0) + 1.0 / (k + r.rank)
        if r.wit_id not in meta_map:
            meta_map[r.wit_id] = r
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        RetrievalResult(
            wit_id=wit_id,
            title=meta_map[wit_id].title,
            caption=meta_map[wit_id].caption,
            score=score,
            rank=rank + 1,
            source="fused",
        )
        for rank, (wit_id, score) in enumerate(ranked)
    ]


# ── Option A ─────────────────────────────────────────────────────────

class OptionA:
    """直接图搜图：image → Qwen3-VL-Embed → ChromaDB"""

    def retrieve(self, image_path: str, top_k: int = TOP_K_DEFAULT) -> RetrievalResponse:
        embed_url = f"{EMBEDDING_URL}/embed"
        resp = _http_post(embed_url, {"image_path": image_path})
        vector = resp.get("vector")
        if not vector:
            raise RuntimeError(f"[OptionA] embedding server returned no vector for {image_path}")
        results = _chroma_query(vector, top_k)
        return RetrievalResponse(results=results, query_used=image_path, mode="A")


# ── Option B ─────────────────────────────────────────────────────────

class OptionB:
    """Text Bridge：image → VLM caption → text embedding + BM25"""

    def __init__(self, use_vlm: bool = True):
        from caption_bridge import CaptionBridge
        self.bridge = CaptionBridge(use_vlm=use_vlm)
        # 延迟加载 RAG（需要 BM25 支持）
        self._rag = None

    def _get_rag(self):
        if self._rag is None:
            from searcheyes.multimodal_rag import MultimodalRAG, RagConfig
            cfg = RagConfig(
                chroma_db_path=str(CHROMA_DB_PATH),
                embedding_server_url=EMBEDDING_URL,
                collection_name=COLLECTION_NAME,
                top_k=TOP_K_DEFAULT,
                use_hybrid=True,
            )
            self._rag = MultimodalRAG(cfg)
        return self._rag

    def retrieve(self, image_path: str, wit_id: str = "",
                 top_k: int = TOP_K_DEFAULT) -> RetrievalResponse:
        br = self.bridge.bridge(image_path, wit_id)
        query = br.search_query
        if not query:
            return RetrievalResponse(results=[], query_used="", mode="B")
        rag = self._get_rag()
        facts = rag.get_rag_facts_combined(
            text=query, top_k=top_k, use_hybrid=True
        )
        results = [
            RetrievalResult(
                wit_id=f.wit_id,
                title=f.title,
                caption=f.caption,
                score=f.score,
                rank=i + 1,
                source="text_embed+bm25",
            )
            for i, f in enumerate(facts)
        ]
        return RetrievalResponse(results=results, query_used=query, mode="B")


# ── Option A+B ───────────────────────────────────────────────────────

class OptionAB:
    """A+B RRF融合"""

    def __init__(self, use_vlm: bool = True):
        self.option_a = OptionA()
        self.option_b = OptionB(use_vlm=use_vlm)

    def retrieve(self, image_path: str, wit_id: str = "",
                 top_k: int = TOP_K_DEFAULT) -> RetrievalResponse:
        resp_a = self.option_a.retrieve(image_path, top_k=top_k * 2)
        resp_b = self.option_b.retrieve(image_path, wit_id, top_k=top_k * 2)
        fused = _rrf_fuse(resp_a.results, resp_b.results, top_k=top_k)
        return RetrievalResponse(
            results=fused,
            query_used=f"A:{image_path} | B:{resp_b.query_used}",
            mode="AB",
        )
