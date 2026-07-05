"""
rag_engine.py — 多模态 RAG 检索引擎 (Wiki6M Backend)
======================================================
基于 OVEN-Wiki6M 构建的混合检索引擎，为 RL 训练环境提供
高效的知识库检索能力。

架构设计:
  Layer 1: 预计算 embedding 向量检索 (cosine similarity)
  Layer 2: BM25 关键词检索
  Layer 3: RRF 混合融合
  Layer 4: DCI 细粒度操作 (filter, grep, read_full, compare)

数据源: Wiki6M_ver_1_0.jsonl (OVEN-Wiki)
  每条: {wikidata_id, wikipedia_title, wikipedia_content,
         wikipedia_image_url, wikipedia_summary}

接口兼容:
  - transition_engine.py: get_rag_facts_combined(), get_fact_by_id()
  - rag_facts_to_search_results(): RagFact 需有 wit_id, title, caption, score
  - _apply_crop_and_search(): _get_embedding(), _query_vectors()
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. RagFact — 检索结果的统一数据类
# ═══════════════════════════════════════════════════════════

@dataclass
class RagFact:
    """单条检索结果。

    字段命名兼容 transition_engine.rag_facts_to_search_results() 的期望:
      - wit_id: 实体唯一标识 (wikidata_id)
      - title: 显示标题 (wikipedia_title)
      - caption: 摘要片段 (wikipedia_summary 截取)
      - score: 检索相关度分数
    """
    wit_id: str
    title: str
    caption: str
    score: float = 0.0
    content: str = ""
    image_url: str = ""
    summary: str = ""

    @property
    def fact_text(self) -> str:
        """兼容 rag_reward.py 的 fact_text 字段"""
        return self.summary or self.caption


# ═══════════════════════════════════════════════════════════
# 2. Wiki6M 实体的内存存储
# ═══════════════════════════════════════════════════════════

@dataclass
class _EntityRecord:
    """内存中的实体记录（紧凑表示，节省内存）"""
    idx: int                    # 在数组中的索引位置
    wikidata_id: str
    title: str
    summary: str                # 短摘要
    image_url: str              # 图片 URL（可为空）
    content_offset: int = 0     # content 在 mmap 文件中的偏移（预留）
    content_length: int = 0     # content 长度


# ═══════════════════════════════════════════════════════════
# 3. BM25 索引（轻量级纯 Python 实现）
# ═══════════════════════════════════════════════════════════

class BM25Index:
    """轻量级 BM25 索引，纯 Python 实现。

    为 255 万文档构建倒排索引，支持快速关键词检索。
    使用 title + summary 作为索引文本（不索引全文，节省内存）。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_length = 0.0
        self.doc_lengths: list[int] = []
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self._built = False

    def build(self, documents: list[str]):
        """构建倒排索引。

        Args:
            documents: 每个元素是一个文档的文本（title + summary 拼接）
        """
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.inverted_index = {}
        total_length = 0

        for doc_idx, doc_text in enumerate(documents):
            tokens = _tokenize_for_bm25(doc_text)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((doc_idx, count))

            if (doc_idx + 1) % 500000 == 0:
                logger.info(f"  BM25 索引构建: {doc_idx + 1}/{self.doc_count}")

        self.avg_doc_length = total_length / max(self.doc_count, 1)
        self._built = True
        logger.info(
            f"BM25 索引构建完成: {self.doc_count} 文档, "
            f"{len(self.inverted_index)} 唯一词项, "
            f"平均文档长度 {self.avg_doc_length:.1f}"
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """BM25 检索。

        Returns:
            list of (doc_idx, bm25_score), 按分数降序
        """
        if not self._built:
            return []

        query_tokens = _tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = {}

        for token in query_tokens:
            postings = self.inverted_index.get(token)
            if not postings:
                continue

            document_frequency = len(postings)
            idf = math.log(
                (self.doc_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
                + 1.0
            )

            for doc_idx, term_frequency in postings:
                doc_length = self.doc_lengths[doc_idx]
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avg_doc_length
                )
                score = idf * numerator / denominator
                scores[doc_idx] = scores.get(doc_idx, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: str | Path):
        """保存 BM25 索引到磁盘"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_doc_length": self.avg_doc_length,
            "doc_lengths": self.doc_lengths,
            "inverted_index": {
                k: v for k, v in self.inverted_index.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(f"BM25 索引已保存到 {path} ({path.stat().st_size / 1e6:.1f} MB)")

    def load(self, path: str | Path):
        """从磁盘加载 BM25 索引"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.k1 = data["k1"]
        self.b = data["b"]
        self.doc_count = data["doc_count"]
        self.avg_doc_length = data["avg_doc_length"]
        self.doc_lengths = data["doc_lengths"]
        self.inverted_index = {
            k: [tuple(pair) for pair in v]
            for k, v in data["inverted_index"].items()
        }
        self._built = True
        logger.info(
            f"BM25 索引已加载: {self.doc_count} 文档, "
            f"{len(self.inverted_index)} 唯一词项"
        )


def _tokenize_for_bm25(text: str) -> list[str]:
    """BM25 分词：小写 + 按非字母数字分割 + 过滤短 token + 去停用词"""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _BM25_STOPWORDS]


_BM25_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "had", "her", "was", "one", "our", "out", "has",
    "his", "how", "its", "may", "new", "now", "old", "see",
    "way", "who", "did", "get", "let", "say", "she", "too",
    "use", "that", "with", "have", "this", "will", "your",
    "from", "they", "been", "said", "each", "which", "their",
    "also", "into", "than", "them", "then", "what", "when",
    "were", "some", "would", "make", "like", "could", "other",
    "after", "about", "there", "these", "most", "more", "very",
})


# ═══════════════════════════════════════════════════════════
# 4. RagEngine — 主引擎
# ═══════════════════════════════════════════════════════════

class RagEngine:
    """多模态 RAG 检索引擎。

    支持三层检索:
      1. 预计算 embedding 向量检索 (cosine similarity on numpy)
      2. BM25 关键词检索
      3. RRF 混合融合

    以及 DCI 细粒度操作:
      - filter_results: 按条件过滤
      - grep_in_entity: 在实体内容中 grep
      - read_full_entity: 读取完整实体文本
      - compare_entities: 对比两个实体

    Args:
        wiki6m_path: Wiki6M jsonl 文件路径
        embeddings_dir: 预计算 embedding 目录（含 text_embeddings.npy 等）
        bm25_index_path: BM25 索引文件路径（如已构建）
        max_entities: 最大加载实体数（用于调试，None=全量）
        only_with_image: 是否只加载有图的实体
        embedding_model: 可选的 embedding 模型实例，需实现
            ``encode_text(text: str) -> np.ndarray`` 和
            ``encode_image(image_path: str) -> np.ndarray`` 方法。
            传入后可在无预计算 embedding 时进行实时向量检索。
    """

    def __init__(
        self,
        wiki6m_path: str | Path = "/dev/shm/oven_wiki/Wiki6M_ver_1_0.jsonl",
        embeddings_dir: str | Path | None = None,
        bm25_index_path: str | Path | None = None,
        max_entities: int | None = None,
        only_with_image: bool = False,
        embedding_model: object | None = None,
    ):
        self.wiki6m_path = Path(wiki6m_path)
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.bm25_index_path = Path(bm25_index_path) if bm25_index_path else None
        self.only_with_image = only_with_image
        self.max_entities = max_entities
        self._embedding_model = embedding_model

        # 核心数据结构
        self._entities: list[_EntityRecord] = []
        self._id_to_idx: dict[str, int] = {}
        self._contents: list[str] = []  # 全文内容（按需加载可改为 lazy）

        # 检索索引
        self._text_embeddings: Optional[np.ndarray] = None  # (N, D) float16
        self._image_embeddings: Optional[np.ndarray] = None  # (M, D) float16
        self._image_idx_to_entity_idx: Optional[np.ndarray] = None  # 有图实体的索引映射
        self._gpu_embeddings = None  # torch.Tensor on GPU (optional, for fast search)
        self._bm25: Optional[BM25Index] = None

        # 状态
        self._loaded = False
        self._embedding_dim = 0

    # ── 初始化与加载 ──────────────────────────────────────

    def load(self):
        """加载所有数据和索引。"""
        logger.info(f"开始加载 RagEngine: {self.wiki6m_path}")
        self._load_wiki6m()
        self._load_embeddings()
        self._load_or_build_bm25()
        self._loaded = True
        logger.info(
            f"RagEngine 加载完成: {len(self._entities)} 实体, "
            f"text_emb={'✅' if self._text_embeddings is not None else '❌'}, "
            f"image_emb={'✅' if self._image_embeddings is not None else '❌'}, "
            f"bm25={'✅' if self._bm25 is not None else '❌'}"
        )

    def _load_wiki6m(self):
        """加载 Wiki6M jsonl 到内存。"""
        logger.info(f"加载 Wiki6M: {self.wiki6m_path}")
        entities = []
        contents = []
        id_to_idx = {}
        skipped = 0

        with open(self.wiki6m_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if self.max_entities and len(entities) >= self.max_entities:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                image_url = record.get("wikipedia_image_url") or ""
                if self.only_with_image and not image_url:
                    continue

                wikidata_id = record.get("wikidata_id", "")
                if not wikidata_id:
                    skipped += 1
                    continue

                idx = len(entities)
                title = record.get("wikipedia_title", "")
                summary = record.get("wikipedia_summary", "")
                content = record.get("wikipedia_content", "")

                entities.append(_EntityRecord(
                    idx=idx,
                    wikidata_id=wikidata_id,
                    title=title,
                    summary=summary[:500],  # 截断摘要节省内存
                    image_url=image_url,
                ))
                contents.append(content)
                id_to_idx[wikidata_id] = idx

                if (idx + 1) % 500000 == 0:
                    logger.info(f"  已加载 {idx + 1} 实体...")

        self._entities = entities
        self._contents = contents
        self._id_to_idx = id_to_idx
        logger.info(
            f"Wiki6M 加载完成: {len(entities)} 实体 "
            f"(跳过 {skipped}), "
            f"有图 {sum(1 for e in entities if e.image_url)} 条"
        )

    def _load_embeddings(self):
        """加载预计算的 embedding 向量。"""
        if not self.embeddings_dir:
            logger.info("未指定 embeddings_dir，跳过向量检索")
            return

        text_emb_path = self.embeddings_dir / "text_embeddings.npy"
        if text_emb_path.exists():
            self._text_embeddings = np.load(str(text_emb_path), mmap_mode="r")
            self._embedding_dim = self._text_embeddings.shape[1]
            logger.info(
                f"Text embeddings 已加载: shape={self._text_embeddings.shape}, "
                f"dtype={self._text_embeddings.dtype}"
            )
        else:
            logger.info(f"Text embeddings 文件不存在: {text_emb_path}")

        image_emb_path = self.embeddings_dir / "image_embeddings.npy"
        if image_emb_path.exists():
            self._image_embeddings = np.load(str(image_emb_path), mmap_mode="r")
            logger.info(
                f"Image embeddings 已加载: shape={self._image_embeddings.shape}"
            )
            # 加载 image idx → entity idx 映射
            mapping_path = self.embeddings_dir / "image_idx_mapping.npy"
            if mapping_path.exists():
                self._image_idx_to_entity_idx = np.load(str(mapping_path))
        else:
            logger.info(f"Image embeddings 文件不存在: {image_emb_path}")

    def _load_or_build_bm25(self):
        """加载或构建 BM25 索引。"""
        if self.bm25_index_path and self.bm25_index_path.exists():
            self._bm25 = BM25Index()
            self._bm25.load(self.bm25_index_path)
            return

        logger.info("构建 BM25 索引（首次加载需要 1-2 分钟）...")
        documents = []
        for entity in self._entities:
            doc_text = f"{entity.title} {entity.summary}"
            documents.append(doc_text)

        self._bm25 = BM25Index()
        self._bm25.build(documents)

        if self.bm25_index_path:
            self._bm25.save(self.bm25_index_path)

    # ── 核心检索接口（transition_engine 调用）─────────────

    def get_fact_by_id(self, entity_id: str) -> Optional[RagFact]:
        """按 wikidata_id 精确获取单条 RagFact。

        兼容 transition_engine 的 get_fact_by_id(wit_id) 接口。
        注意: transition_engine 传入的可能是 wikidata_id 或旧的 wit_id 格式。
        """
        idx = self._id_to_idx.get(entity_id)
        if idx is None:
            return None
        return self._entity_to_ragfact(idx, score=1.0)

    def get_rag_facts_combined(
        self,
        text: str = "",
        image_path: str = "",
        top_k: int = 20,
        use_hybrid: bool = True,
    ) -> list[RagFact]:
        """混合检索，返回 top_k 个 RagFact。

        兼容 transition_engine 的调用签名:
            rag.get_rag_facts_combined(text=..., top_k=20, use_hybrid=True)

        策略:
          1. 如有 text embedding → 向量检索
          2. BM25 关键词检索
          3. RRF 融合两路结果
          4. 如仅有一路可用，直接返回该路结果
        """
        if not self._loaded:
            logger.warning("RagEngine 未加载，调用 load() 后重试")
            return []

        if not text and not image_path:
            return []

        vector_results: list[tuple[int, float]] = []
        bm25_results: list[tuple[int, float]] = []

        # Layer 1: 向量检索
        if text and self._text_embeddings is not None:
            vector_results = self._vector_search_text(text, top_k=top_k * 2)

        if image_path and self._image_embeddings is not None:
            image_vector_results = self._vector_search_image(
                image_path, top_k=top_k * 2
            )
            if image_vector_results and not vector_results:
                vector_results = image_vector_results
            elif image_vector_results:
                vector_results = self._merge_scored_lists(
                    vector_results, image_vector_results
                )

        # Layer 2: BM25 检索
        if text and self._bm25 is not None:
            bm25_results = self._bm25.search(text, top_k=top_k * 2)

        # Layer 3: 融合
        if use_hybrid and vector_results and bm25_results:
            merged_indices = self._rrf_fusion(
                vector_results, bm25_results, top_k=top_k
            )
        elif vector_results:
            merged_indices = [(idx, score) for idx, score in vector_results[:top_k]]
        elif bm25_results:
            merged_indices = [(idx, score) for idx, score in bm25_results[:top_k]]
        else:
            return []

        return [self._entity_to_ragfact(idx, score) for idx, score in merged_indices]

    # ── 向量检索 ─────────────────────────────────────────

    def _vector_search_text(
        self, query_text: str, top_k: int = 40
    ) -> list[tuple[int, float]]:
        """文本向量检索: query_text → embedding → cosine similarity。

        需要预计算好的 text_embeddings.npy 和 query embedding 模型。
        如果没有实时 embedding 模型，使用 BM25 fallback。
        """
        query_vec = self._get_text_embedding(query_text)
        if query_vec is None:
            return []
        return self._cosine_search(self._text_embeddings, query_vec, top_k)

    def _vector_search_image(
        self, image_path: str, top_k: int = 40
    ) -> list[tuple[int, float]]:
        """图片向量检索: image → embedding → cosine similarity against image index。"""
        query_vec = self._get_embedding(image_path)
        if query_vec is None or self._image_embeddings is None:
            return []

        raw_results = self._cosine_search(self._image_embeddings, query_vec, top_k)

        if self._image_idx_to_entity_idx is not None:
            return [
                (int(self._image_idx_to_entity_idx[img_idx]), score)
                for img_idx, score in raw_results
                if img_idx < len(self._image_idx_to_entity_idx)
            ]
        return raw_results

    def _cosine_search(
        self, embeddings: np.ndarray, query_vec: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """在 embedding 矩阵上做 cosine similarity 检索。

        策略（按优先级）：
          1. 如果 _gpu_embeddings 已缓存在 GPU → torch matmul（~0.3s）
          2. 否则 numpy CPU 分块计算（~60s for 2.5M × 4096）

        注意: 预计算的 embeddings 已经归一化（norm≈1），无需再次归一化。
        """
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # 尝试 GPU 加速
        if self._gpu_embeddings is not None:
            return self._cosine_search_gpu(query_norm, top_k)

        # CPU fallback（numpy）
        query_f16 = query_norm.astype(np.float16)
        num_entities = embeddings.shape[0]

        # 直接点积（embeddings 已归一化）
        chunk_size = 500000
        all_scores = np.empty(num_entities, dtype=np.float32)
        for start in range(0, num_entities, chunk_size):
            end = min(start + chunk_size, num_entities)
            chunk = embeddings[start:end].astype(np.float32)
            all_scores[start:end] = chunk @ query_f16.astype(np.float32)

        top_indices = np.argpartition(all_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]

        return [
            (int(idx), float(all_scores[idx]))
            for idx in top_indices
        ]

    def _cosine_search_gpu(
        self, query_norm: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """GPU 加速的 cosine search（torch matmul on cached tensor）。"""
        import torch

        query_tensor = torch.from_numpy(query_norm.astype(np.float32)).to(
            device=self._gpu_embeddings.device, dtype=self._gpu_embeddings.dtype
        )
        # (N, D) @ (D,) → (N,)
        scores = (self._gpu_embeddings @ query_tensor).float()
        top_values, top_indices = torch.topk(scores, k=min(top_k, len(scores)))

        return [
            (int(idx), float(val))
            for idx, val in zip(top_indices.cpu().numpy(), top_values.cpu().numpy())
        ]

    def load_embeddings_to_gpu(self, device: str = "cuda:0"):
        """将 text embeddings 加载到 GPU 显存以加速检索。

        255万 × 4096 × float16 ≈ 21 GB 显存。
        调用后 _cosine_search 自动走 GPU 路径。
        """
        import torch

        if self._text_embeddings is None:
            logger.warning("text_embeddings 未加载，无法迁移到 GPU")
            return

        logger.info(f"正在将 text_embeddings 加载到 {device}...")
        t0 = __import__("time").time()
        # mmap 的 numpy array → torch tensor → GPU
        tensor = torch.from_numpy(
            np.array(self._text_embeddings, dtype=np.float16)
        ).to(device=device, dtype=torch.float16)
        self._gpu_embeddings = tensor
        elapsed = __import__("time").time() - t0
        logger.info(
            f"GPU embeddings 就绪: {tensor.shape}, {tensor.device}, "
            f"{tensor.element_size() * tensor.nelement() / 1e9:.1f} GB, "
            f"加载耗时 {elapsed:.1f}s"
        )

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取文本的 embedding 向量。

        使用通过构造函数传入的 embedding_model 进行实时编码。
        如果未传入模型则返回 None（调用方会自动降级到纯 BM25）。
        子类也可覆写此方法接入自定义的 embedding 模型。
        """
        if self._embedding_model is None:
            return None
        try:
            vec = self._embedding_model.encode_text(text)
            if vec is not None:
                if hasattr(vec, 'cpu'):
                    vec = vec.float().cpu().numpy()
                return np.asarray(vec, dtype=np.float32).ravel()
        except Exception as exc:
            logger.warning("encode_text 失败: %s", exc)
        return None

    def _get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """获取图片的 embedding 向量。

        兼容 transition_engine._apply_crop_and_search() 的调用。
        使用通过构造函数传入的 embedding_model 进行实时编码。
        如果未传入模型则返回 None。
        子类也可覆写此方法接入自定义的 VLM embedding 模型。
        """
        if self._embedding_model is None:
            return None
        try:
            vec = self._embedding_model.encode_image(image_path)
            if vec is not None:
                if hasattr(vec, 'cpu'):
                    vec = vec.float().cpu().numpy()
                return np.asarray(vec, dtype=np.float32).ravel()
        except Exception as exc:
            logger.warning("encode_image 失败 (%s): %s", image_path, exc)
        return None

    # ── RRF 融合 ─────────────────────────────────────────

    @staticmethod
    def _rrf_fusion(
        list_a: list[tuple[int, float]],
        list_b: list[tuple[int, float]],
        top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion (RRF) 融合两路检索结果。

        RRF score = Σ 1 / (k + rank_i)
        """
        scores: dict[int, float] = {}

        for rank, (doc_idx, _) in enumerate(list_a):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, (doc_idx, _) in enumerate(list_b):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _merge_scored_lists(
        list_a: list[tuple[int, float]],
        list_b: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """合并两路带分数的结果列表，取最高分。"""
        scores: dict[int, float] = {}
        for doc_idx, score in list_a:
            scores[doc_idx] = max(scores.get(doc_idx, 0.0), score)
        for doc_idx, score in list_b:
            scores[doc_idx] = max(scores.get(doc_idx, 0.0), score)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ── DCI 细粒度操作接口 ────────────────────────────────

    def filter_results(
        self,
        results: list[RagFact],
        has_image: Optional[bool] = None,
        title_contains: str = "",
        min_content_length: int = 0,
    ) -> list[RagFact]:
        """在已有检索结果上做精确过滤（DCI Layer）。

        相比纯 top-k 语义检索，支持:
          - 过滤无图实体
          - 标题关键词匹配
          - 最小内容长度
        """
        filtered = []
        for fact in results:
            idx = self._id_to_idx.get(fact.wit_id)
            if idx is None:
                continue
            entity = self._entities[idx]

            if has_image is not None:
                if has_image and not entity.image_url:
                    continue
                if not has_image and entity.image_url:
                    continue

            if title_contains:
                if title_contains.lower() not in entity.title.lower():
                    continue

            if min_content_length > 0:
                content = self._contents[idx] if idx < len(self._contents) else ""
                if len(content) < min_content_length:
                    continue

            filtered.append(fact)

        return filtered

    def grep_in_entity(
        self, entity_id: str, pattern: str, case_sensitive: bool = False
    ) -> list[str]:
        """在实体全文中做 grep（DCI Layer）。

        相当于 DCI 论文的 "直接操作语料库" 能力——
        Agent 先 search 拿到候选实体，再用 grep 做精确验证。

        Returns:
            匹配到的上下文片段列表（每个片段前后各 100 字符）
        """
        idx = self._id_to_idx.get(entity_id)
        if idx is None:
            return []

        content = self._contents[idx] if idx < len(self._contents) else ""
        if not content:
            return []

        flags = 0 if case_sensitive else re.IGNORECASE
        matches = []
        for match in re.finditer(re.escape(pattern), content, flags):
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 100)
            snippet = content[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            matches.append(snippet)
            if len(matches) >= 5:
                break

        return matches

    def read_full_entity(self, entity_id: str) -> Optional[dict]:
        """读取完整实体信息（DCI Layer）。

        相当于 VDR 的 visit 工具——从候选 URL 读取完整网页内容。

        Returns:
            dict with keys: wikidata_id, title, summary, content, image_url
        """
        idx = self._id_to_idx.get(entity_id)
        if idx is None:
            return None
        entity = self._entities[idx]
        content = self._contents[idx] if idx < len(self._contents) else ""
        return {
            "wikidata_id": entity.wikidata_id,
            "title": entity.title,
            "summary": entity.summary,
            "content": content,
            "image_url": entity.image_url,
        }

    def compare_entities(
        self, entity_id_a: str, entity_id_b: str, aspect: str = ""
    ) -> dict:
        """对比两个实体（DCI Layer）。

        Agent 可以在多个候选中做精确对比，而不是只靠 top-k 排序。

        Returns:
            dict with comparison results
        """
        entity_a = self.read_full_entity(entity_id_a)
        entity_b = self.read_full_entity(entity_id_b)

        if not entity_a or not entity_b:
            return {"error": "entity not found", "found_a": entity_a is not None, "found_b": entity_b is not None}

        result = {
            "entity_a": {
                "id": entity_id_a,
                "title": entity_a["title"],
                "has_image": bool(entity_a["image_url"]),
                "content_length": len(entity_a["content"]),
                "summary": entity_a["summary"][:200],
            },
            "entity_b": {
                "id": entity_id_b,
                "title": entity_b["title"],
                "has_image": bool(entity_b["image_url"]),
                "content_length": len(entity_b["content"]),
                "summary": entity_b["summary"][:200],
            },
        }

        if aspect:
            grep_a = self.grep_in_entity(entity_id_a, aspect)
            grep_b = self.grep_in_entity(entity_id_b, aspect)
            result["aspect_matches"] = {
                "aspect": aspect,
                "matches_a": grep_a,
                "matches_b": grep_b,
            }

        # 计算文本重叠度
        tokens_a = set(_tokenize_for_bm25(entity_a["content"]))
        tokens_b = set(_tokenize_for_bm25(entity_b["content"]))
        if tokens_a and tokens_b:
            overlap = len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))
            result["token_overlap"] = round(overlap, 3)

        return result

    # ── 兼容 transition_engine._apply_crop_and_search() ──

    def _query_chroma(self, query_vec: list | np.ndarray, top_k: int = 20) -> list[RagFact]:
        """兼容 transition_engine._apply_crop_and_search() 的调用。

        旧接口: rag._query_chroma(vec, top_k=20)
        新实现: 在预计算的 image_embeddings 上做 cosine search。
        """
        if isinstance(query_vec, list):
            query_vec = np.array(query_vec, dtype=np.float32)

        if self._image_embeddings is not None:
            raw_results = self._cosine_search(
                self._image_embeddings, query_vec, top_k
            )
            if self._image_idx_to_entity_idx is not None:
                results = [
                    (int(self._image_idx_to_entity_idx[img_idx]), score)
                    for img_idx, score in raw_results
                    if img_idx < len(self._image_idx_to_entity_idx)
                ]
            else:
                results = raw_results
        elif self._text_embeddings is not None:
            results = self._cosine_search(
                self._text_embeddings, query_vec, top_k
            )
        else:
            return []

        return [self._entity_to_ragfact(idx, score) for idx, score in results]

    # ── 内部工具 ─────────────────────────────────────────

    def _entity_to_ragfact(self, idx: int, score: float) -> RagFact:
        """将内部实体索引转为 RagFact。"""
        entity = self._entities[idx]
        return RagFact(
            wit_id=entity.wikidata_id,
            title=entity.title,
            caption=entity.summary[:200],
            score=score,
            content=self._contents[idx][:500] if idx < len(self._contents) else "",
            image_url=entity.image_url,
            summary=entity.summary,
        )

    # ── 统计 / 调试接口 ──────────────────────────────────

    def stats(self) -> dict:
        """返回引擎统计信息"""
        return {
            "total_entities": len(self._entities),
            "entities_with_image": sum(1 for e in self._entities if e.image_url),
            "text_embeddings_loaded": self._text_embeddings is not None,
            "text_embeddings_shape": (
                self._text_embeddings.shape if self._text_embeddings is not None else None
            ),
            "image_embeddings_loaded": self._image_embeddings is not None,
            "image_embeddings_shape": (
                self._image_embeddings.shape if self._image_embeddings is not None else None
            ),
            "bm25_built": self._bm25 is not None and self._bm25._built,
            "bm25_vocab_size": (
                len(self._bm25.inverted_index) if self._bm25 else 0
            ),
            "embedding_dim": self._embedding_dim,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        count = len(self._entities)
        return f"RagEngine({status}, {count} entities)"
