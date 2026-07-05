"""
multimodal_rag.py — 多模态 RAG 检索模块
========================================
Phase B1：封装 embedding + ChromaDB 检索为统一接口。
供 agent_loop.py 在决策前调用。

核心接口:
    rag = MultimodalRAG(RagConfig(chroma_db_path="data/wit_subset/chroma_db"))
    facts = rag.get_rag_facts("path/to/screenshot.png")
    prompt_text = MultimodalRAG.format_for_prompt(facts)
"""

from __future__ import annotations

import http.client
import json
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RagFact:
    """从多模态知识库检索到的单条知识。"""

    wit_id: str
    title: str
    caption: str
    score: float  # cosine similarity, 0-1（ChromaDB 返回 distance，需转换）
    source_url: str = ""

    def as_prompt_line(self, max_chars: int = 80) -> str:
        """生成适合注入 VLM prompt 的单行文本。"""
        text = f"{self.title}: {self.caption}"
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text


@dataclass
class RagConfig:
    """RAG 管线配置。"""

    chroma_db_path: str
    embedding_server_url: str = "http://localhost:8766"
    collection_name: str = "wit_knowledge"
    top_k: int = 20
    score_threshold: float = 1.0  # cosine distance 阈值（越小越相似，1.0=宽松，0.5=严格）
    use_hybrid: bool = False       # True = BM25 + 向量混合（RRF）；False = 纯向量


class MultimodalRAG:
    """多模态 RAG 检索器。

    工作流程:
        1. 将图片发送到 embedding server 获取向量
        2. 在 ChromaDB 中执行近邻搜索
        3. 过滤并返回 RagFact 列表
    """

    def __init__(self, config: RagConfig):
        self.config = config
        self._collection = None
        self._bm25_index = None       # 懒加载
        self._bm25_corpus_ids = None  # 与 BM25 corpus 对应的 wit_id 列表

    def get_rag_facts(self, image_path: str) -> list[RagFact]:
        """主入口：图片路径 → 相关知识列表。

        如果 embedding server 不可达或 ChromaDB 查询失败，
        返回空列表（不中断 agent loop）。
        """
        vector = self._get_embedding(image_path)
        if vector is None:
            return []

        return self._query_chroma(vector)

    def _get_embedding(self, image_path: str) -> list[float] | None:
        """调用 embedding server 获取图片向量。"""
        url = f"{self.config.embedding_server_url}/embed"
        payload = {"image_path": image_path}

        try:
            return self._post_json(url, payload).get("vector")
        except Exception as exc:
            print(f"[RAG] embedding 请求失败: {exc}")
            return None

    def _query_chroma(self, vector: list[float], top_k: int | None = None) -> list[RagFact]:
        """在 ChromaDB 中搜索最近邻。"""
        try:
            collection = self._ensure_collection()
            results = collection.query(
                query_embeddings=[vector],
                n_results=top_k or self.config.top_k,
            )
        except Exception as exc:
            print(f"[RAG] ChromaDB 查询失败: {exc}")
            return []

        facts: list[RagFact] = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, wit_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            # ChromaDB cosine distance: 0 = 完全相同, 2 = 完全相反
            # 转换为 similarity score: 1 - distance/2
            score = max(0.0, 1.0 - distance / 2.0)

            if distance > self.config.score_threshold:
                continue

            meta = metadatas[i] if i < len(metadatas) else {}
            facts.append(RagFact(
                wit_id=wit_id,
                title=meta.get("page_title", ""),
                caption=meta.get("caption", ""),
                score=score,
                source_url=meta.get("source_url", ""),
            ))

        return facts

    def get_rag_facts_combined(
        self,
        image_path: str = "",
        text: str = "",
        top_k: int | None = None,
        use_hybrid: bool | None = None,
    ) -> list[RagFact]:
        """组合查询：支持图+文、纯图、纯文本三种模式。

        当同时提供 image_path 和 text 时，对两个 embedding 取平均。
        top_k 可覆盖 config 默认值（用于 search 动作需要更多结果）。
        use_hybrid 可覆盖 config.use_hybrid（None = 使用 config 默认值）。
        """
        _use_hybrid = self.config.use_hybrid if use_hybrid is None else use_hybrid
        vectors: list[list[float]] = []

        if image_path:
            v = self._get_embedding(image_path)
            if v:
                vectors.append(v)

        if text:
            v = self._get_text_embedding(text)
            if v:
                vectors.append(v)

        if not vectors:
            return []

        # 平均多个向量
        if len(vectors) == 1:
            combined = vectors[0]
        else:
            dim = len(vectors[0])
            combined = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]

        vector_results = self._query_chroma(combined, top_k=top_k)

        # BM25 混合（仅当 use_hybrid=True 且有文本 query）
        if _use_hybrid and text:
            bm25_results = self._bm25_search(text, top_k=top_k)
            return self._fuse_results(vector_results, bm25_results, top_k=top_k or self.config.top_k)

        return vector_results

    def get_fact_by_id(self, wit_id: str) -> RagFact | None:
        """按 wit_id 直接从 ChromaDB 获取单条记录。

        用于保证 ground truth 出现在搜索结果中。
        """
        try:
            collection = self._ensure_collection()
            result = collection.get(ids=[wit_id], include=["metadatas"])
            ids = result.get("ids", [])
            metadatas = result.get("metadatas", [])
            if ids and metadatas:
                meta = metadatas[0]
                return RagFact(
                    wit_id=ids[0],
                    title=meta.get("page_title", ""),
                    caption=meta.get("caption", ""),
                    score=0.5,  # 注入的 GT，给中等分数
                    source_url=meta.get("source_url", ""),
                )
        except Exception as exc:
            print(f"[RAG] get_fact_by_id 失败: {exc}")
        return None

    def _get_text_embedding(self, text: str) -> list[float] | None:
        """调用 embedding server 获取文本向量。"""
        url = f"{self.config.embedding_server_url}/embed"
        payload = {"text": text}
        try:
            return self._post_json(url, payload).get("vector")
        except Exception as exc:
            print(f"[RAG] text embedding 请求失败: {exc}")
            return None

    def _ensure_collection(self):
        """懒加载 ChromaDB collection。"""
        if self._collection is not None:
            return self._collection

        import chromadb

        client = chromadb.PersistentClient(path=self.config.chroma_db_path)
        self._collection = client.get_collection(self.config.collection_name)
        return self._collection

    @staticmethod
    def format_for_prompt(facts: list[RagFact], max_facts: int = 3) -> str:
        """将 facts 格式化为适合注入 VLM prompt 的文本块。"""
        if not facts:
            return ""

        lines = ["Knowledge (from visual search):"]
        for fact in facts[:max_facts]:
            lines.append(f"  - {fact.as_prompt_line()}")
        return "\n".join(lines)

    def _ensure_bm25(self):
        """懒加载 BM25 索引，从 ChromaDB 读取所有文档。"""
        if self._bm25_index is not None:
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("BM25 需要 rank_bm25：pip install rank-bm25")

        collection = self._ensure_collection()
        # 获取所有条目的 metadata
        all_data = collection.get(include=["metadatas"])
        ids = all_data.get("ids", [])
        metadatas = all_data.get("metadatas", [])

        corpus = []
        self._bm25_corpus_ids = []
        for wit_id, meta in zip(ids, metadatas):
            title = meta.get("page_title", "")
            caption = meta.get("caption", "")
            doc = f"{title} {caption}".lower().split()
            corpus.append(doc)
            self._bm25_corpus_ids.append(wit_id)

        self._bm25_index = BM25Okapi(corpus)

    def _bm25_search(self, text: str, top_k: int | None = None) -> list[RagFact]:
        """BM25 关键词检索，返回 top-k 结果。"""
        self._ensure_bm25()
        k = top_k or self.config.top_k
        tokens = text.lower().split()
        scores = self._bm25_index.get_scores(tokens)

        # 取 top-k 索引（分数从高到低）
        import heapq
        top_indices = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])

        collection = self._ensure_collection()
        facts = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            wit_id = self._bm25_corpus_ids[idx]
            try:
                result = collection.get(ids=[wit_id], include=["metadatas"])
                meta = result.get("metadatas", [{}])[0]
            except Exception:
                continue
            # BM25 分数归一化到 0-1（近似）
            max_score = max(scores) if max(scores) > 0 else 1.0
            norm_score = scores[idx] / max_score
            facts.append(RagFact(
                wit_id=wit_id,
                title=meta.get("page_title", ""),
                caption=meta.get("caption", ""),
                score=norm_score,
                source_url=meta.get("source_url", ""),
            ))
        return facts

    def _fuse_results(
        self,
        vector_results: list[RagFact],
        bm25_results: list[RagFact],
        top_k: int = 6,
    ) -> list[RagFact]:
        """Reciprocal Rank Fusion (RRF)：融合向量检索和 BM25 结果。

        RRF 公式：score(d) = Σ 1/(k + rank(d))，k=60 是平滑常数。
        """
        RRF_K = 60
        scores: dict[str, float] = {}
        meta_map: dict[str, RagFact] = {}

        for rank, fact in enumerate(vector_results):
            scores[fact.wit_id] = scores.get(fact.wit_id, 0) + 1.0 / (RRF_K + rank + 1)
            meta_map[fact.wit_id] = fact

        for rank, fact in enumerate(bm25_results):
            scores[fact.wit_id] = scores.get(fact.wit_id, 0) + 1.0 / (RRF_K + rank + 1)
            if fact.wit_id not in meta_map:
                meta_map[fact.wit_id] = fact

        # 按 RRF 分数排序，取 top_k
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [
            RagFact(
                wit_id=wid,
                title=meta_map[wid].title,
                caption=meta_map[wid].caption,
                score=scores[wid],
                source_url=meta_map[wid].source_url,
            )
            for wid in sorted_ids
        ]

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """stdlib HTTP POST，复用 vlm_agent.py 的模式。"""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        parsed = urllib.parse.urlsplit(url)
        conn_cls = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        conn = conn_cls(parsed.hostname, parsed.port, timeout=30)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        try:
            conn.request(
                "POST", path, body=data,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                raise RuntimeError(
                    f"HTTP {resp.status}: {raw.decode('utf-8', errors='ignore')[:120]}"
                )
            return json.loads(raw.decode("utf-8"))
        finally:
            conn.close()
