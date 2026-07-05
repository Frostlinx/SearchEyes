"""
search_controller.py — 检索质量控制器（v2 修复版）
====================================================
在 RAG 检索与状态机之间插入 judge → retry 闭环。

规则驱动（不依赖训练），按成本递增尝试多种检索策略，
首次达到 grade A 即短路返回。

v2 修复：
- judge 改为 rank-based（不看 raw score 绝对值，兼容 RRF / cosine / BM25）
- 禁用 image_query（当前 phase 不用截图检索）
- 策略顺序：text_hybrid → relaxed_topk → bm25_only(cleaned_text)
- _select_best 和 quality_improvement 基于 result_count（量纲无关）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from searcheyes.multimodal_rag import MultimodalRAG, RagFact


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchQuality:
    """单次检索的质量评估。"""

    top1_score: float       # 最高分（仅记录，不用于跨策略比较）
    mean_score: float       # 平均分（仅记录）
    score_spread: float     # max - min
    result_count: int       # 返回条数（量纲无关，可跨策略比较）
    grade: str = "F"        # "A" | "B" | "F"


@dataclass
class SearchAttempt:
    """单次检索策略的执行记录。"""

    strategy: str           # "text_hybrid" | "relaxed_topk" | "bm25_only"
    query_text: str         # 实际使用的 query
    query_image: str        # 实际使用的图像路径（空 = 未使用）
    top_k: int
    use_hybrid: bool
    facts: list[RagFact]
    quality: SearchQuality
    elapsed_ms: float


@dataclass
class SearchResult:
    """SearchController 的最终输出。"""

    facts: list[RagFact]
    attempts: list[SearchAttempt]
    accepted_strategy: str
    retry_triggered: bool
    quality_before_retry: SearchQuality | None
    quality_after_retry: SearchQuality | None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class SearchController:
    """规则驱动的检索质量控制器，带 judge → retry 闭环。

    v2 设计原则：非常保守。text_hybrid 是默认主力，只在明显失败时 fallback。
    """

    MIN_RESULTS = 3
    DEFAULT_TOP_K = 6

    def __init__(self) -> None:
        # 累积指标
        self._total_searches = 0
        self._retry_triggered_count = 0
        self._strategy_wins: dict[str, int] = {}
        self._quality_before_sum = 0.0   # result_count 累积
        self._quality_after_sum = 0.0    # result_count 累积
        self._retry_quality_pairs = 0
        self._grade_counts: dict[str, int] = {"A": 0, "B": 0, "F": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_search(
        self,
        query_image: str,
        query_text: str,
        rag: MultimodalRAG,
    ) -> SearchResult:
        """主入口：按策略优先级尝试，grade A 短路返回。

        策略顺序（v2）：
          1. text_hybrid  — rewrite + vector + BM25（主力）
          2. relaxed_topk — 扩大候选池
          3. bm25_only    — 纯 BM25（cleaned_text）
          image_query 当前 phase 禁用。
        """
        from searcheyes.query_rewriter import rewrite as _rewrite_query

        self._total_searches += 1
        attempts: list[SearchAttempt] = []
        cleaned_text = _rewrite_query(query_text) if query_text else ""

        # --- Strategy 1: text_hybrid（主力，成本最低） ---
        attempt = self._try_strategy(
            "text_hybrid", rag,
            query_text=cleaned_text,
            query_image="",
            top_k=self.DEFAULT_TOP_K,
            use_hybrid=True,
        )
        attempts.append(attempt)

        if attempt.quality.grade == "A":
            self._grade_counts["A"] = self._grade_counts.get("A", 0) + 1
            return self._finalize(attempts, retry_triggered=False)

        quality_before = attempt.quality

        # --- Strategy 2: relaxed_topk（扩大候选池） ---
        attempt = self._try_strategy(
            "relaxed_topk", rag,
            query_text=cleaned_text,
            query_image="",
            top_k=self.DEFAULT_TOP_K * 2,
            use_hybrid=True,
        )
        attempts.append(attempt)

        if attempt.quality.grade == "A":
            self._grade_counts["A"] = self._grade_counts.get("A", 0) + 1
            return self._finalize(
                attempts, retry_triggered=True,
                quality_before=quality_before,
            )

        # --- Strategy 3: bm25_only（用 cleaned_text，不是原始 query） ---
        attempt = self._try_strategy(
            "bm25_only", rag,
            query_text=cleaned_text,
            query_image="",
            top_k=self.DEFAULT_TOP_K,
            use_hybrid=False,
            force_bm25=True,
        )
        attempts.append(attempt)

        # image_query 当前 phase 禁用
        # 等有 zoom crop refinement search 再单独接回来

        # 没有 grade A，选最优
        best = self._select_best(attempts)
        self._grade_counts[best.quality.grade] = (
            self._grade_counts.get(best.quality.grade, 0) + 1
        )
        return self._finalize(
            attempts,
            retry_triggered=True,
            quality_before=quality_before,
        )

    def get_metrics(self) -> dict[str, Any]:
        """返回累积的 controller 指标。"""
        total = self._total_searches or 1
        return {
            "total_searches": self._total_searches,
            "retry_triggered_count": self._retry_triggered_count,
            "retry_triggered_rate": self._retry_triggered_count / total,
            "strategy_distribution": dict(self._strategy_wins),
            "quality_improvement_avg": (
                (self._quality_after_sum - self._quality_before_sum)
                / self._retry_quality_pairs
                if self._retry_quality_pairs > 0 else 0.0
            ),
            "grade_distribution": dict(self._grade_counts),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _judge(facts: list[RagFact]) -> SearchQuality:
        """量纲无关的质量评级。

        保守原则：≥3 条结果 → 默认 grade A（信任主策略）。
        只在明显失败时判 F。
        """
        if not facts:
            return SearchQuality(
                top1_score=0.0, mean_score=0.0, score_spread=0.0,
                result_count=0, grade="F",
            )

        scores = [f.score for f in facts]
        top1 = max(scores)
        mean = sum(scores) / len(scores)
        spread = top1 - min(scores)
        count = len(facts)

        # --- 防御性检查：明显失败 → grade F ---
        if count < SearchController.MIN_RESULTS:
            grade = "F"
        elif any(not f.title and not f.caption for f in facts[:3]):
            # 前 3 条有空内容 → 检索质量差
            grade = "F"
        # --- ≥3 条有内容的结果 → 默认 accept ---
        else:
            grade = "A"

        return SearchQuality(
            top1_score=top1, mean_score=mean, score_spread=spread,
            result_count=count, grade=grade,
        )

    def _try_strategy(
        self,
        strategy: str,
        rag: MultimodalRAG,
        *,
        query_text: str,
        query_image: str,
        top_k: int,
        use_hybrid: bool,
        force_bm25: bool = False,
    ) -> SearchAttempt:
        """执行单次检索策略并评估质量。"""
        t0 = time.monotonic()

        try:
            if force_bm25:
                facts = rag._bm25_search(query_text, top_k=top_k)
            else:
                facts = rag.get_rag_facts_combined(
                    image_path=query_image or "",
                    text=query_text or "",
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                )
        except Exception as exc:
            print(f"[SearchController] strategy={strategy} failed: {exc}")
            facts = []

        # relaxed_topk：取回更多再截断
        if strategy == "relaxed_topk" and len(facts) > self.DEFAULT_TOP_K:
            facts = facts[: self.DEFAULT_TOP_K]

        elapsed = (time.monotonic() - t0) * 1000
        quality = self._judge(facts)

        return SearchAttempt(
            strategy=strategy,
            query_text=query_text,
            query_image=query_image,
            top_k=top_k,
            use_hybrid=use_hybrid,
            facts=facts,
            quality=quality,
            elapsed_ms=elapsed,
        )

    def _select_best(self, attempts: list[SearchAttempt]) -> SearchAttempt:
        """从多次尝试中选最优：优先 grade，同 grade 下选 result_count 多的。

        不比较 raw score（量纲不统一）。
        """
        grade_order = {"A": 0, "B": 1, "F": 2}
        return min(
            attempts,
            key=lambda a: (
                grade_order.get(a.quality.grade, 9),
                -a.quality.result_count,
                a.elapsed_ms,
            ),
        )

    def _finalize(
        self,
        attempts: list[SearchAttempt],
        retry_triggered: bool,
        quality_before: SearchQuality | None = None,
    ) -> SearchResult:
        """选出最优结果，更新累积指标。"""
        best = self._select_best(attempts)

        self._strategy_wins[best.strategy] = (
            self._strategy_wins.get(best.strategy, 0) + 1
        )
        if retry_triggered:
            self._retry_triggered_count += 1
            if quality_before is not None:
                # 用 result_count 做质量比较（量纲无关）
                self._quality_before_sum += quality_before.result_count
                self._quality_after_sum += best.quality.result_count
                self._retry_quality_pairs += 1

        return SearchResult(
            facts=best.facts,
            attempts=attempts,
            accepted_strategy=best.strategy,
            retry_triggered=retry_triggered,
            quality_before_retry=quality_before,
            quality_after_retry=best.quality if retry_triggered else None,
        )
