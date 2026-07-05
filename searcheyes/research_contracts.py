"""
research_contracts.py — Phase 0: v2 环境契约定义
==================================================
定义 search/report world 的所有数据契约：
- ResearchTask: 研究任务 schema（替代 VisualTask 的购物语义）
- SearchResult: 检索结果（替代 products）
- CitationObject: 引用证据（可验证来源链接）
- ResearchState: 环境状态（替代 EnvState）
- FactSet: ground truth 事实集合

设计原则（来自 Codex review）：
1. 不在 hashed state 中存大块原始文本
2. 用 stable ID 追踪（query_id, result_id, citation_id）
3. UI shell state（page/panel）与 research progress state 分离
4. Citation 是结构化对象，不是裸字符串
5. 支持多条有效轨迹（多种合法证据收集顺序）
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════
# 1. Research Phase（FSM 状态）
# ═══════════════════════════════════════════════════════════

class ResearchPhase(str, Enum):
    """研究阶段 — Agent 在 search/report world 中的 FSM 状态"""
    FORMULATE = "formulate"    # 初始状态，准备提交第一个搜索
    SEARCH    = "search"       # 搜索结果已返回，可浏览/选择/重搜
    BROWSE    = "browse"       # 正在阅读某篇文档
    SYNTHESIZE = "synthesize"  # 正在编写/提交报告（可选显式阶段）


# ═══════════════════════════════════════════════════════════
# 2. SearchResult（替代 products dict）
# ═══════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """单条搜索结果 — 由 RAG 检索返回，渲染为搜索结果列表中的一项"""
    result_id: int               # 结果列表中的位置 (1-based)
    wit_id: str                  # WIT 条目 ID（用于 GT 匹配）
    title: str                   # 文档标题
    snippet: str                 # 摘要片段（caption 截取）
    relevance_score: float       # RRF/cosine 融合分数
    virtual_url: str = ""        # 伪 URL（如 wiki://wit_0747）
    image_path: str = ""         # 图片路径（用于渲染缩略图）

    @property
    def display_snippet(self) -> str:
        """截断显示用摘要"""
        return self.snippet[:120] + "..." if len(self.snippet) > 120 else self.snippet


@dataclass
class DocumentView:
    """打开的文档详情 — 从 SearchResult 展开"""
    result_id: int               # 对应的 SearchResult.result_id
    wit_id: str
    title: str
    body_text: str               # 完整文本（caption + 扩展上下文）
    image_path: str = ""
    source_url: str = ""


# ═══════════════════════════════════════════════════════════
# 3. CitationObject（结构化引用，可验证来源）
# ═══════════════════════════════════════════════════════════

@dataclass
class CitationObject:
    """一条引用证据 — 从文档中提取，带来源追踪"""
    citation_id: int             # 自增 ID
    source_result_id: int        # 来自哪个 SearchResult
    source_wit_id: str           # 来源 WIT ID
    evidence_text: str           # 提取的证据文本
    source_title: str = ""       # 来源文档标题

    def matches_fact(self, fact_wit_id: str) -> bool:
        """检查此引用是否覆盖了某个 GT fact"""
        return self.source_wit_id == fact_wit_id


# ═══════════════════════════════════════════════════════════
# 4. FactSet — Ground Truth 事实集合
# ═══════════════════════════════════════════════════════════

@dataclass
class FactSet:
    """任务的 ground truth 事实集合。
    
    成功标准：Agent 收集的 citations 覆盖了多少 GT facts。
    支持多种有效轨迹 — 只看最终覆盖率，不强制顺序。
    """
    facts: list[dict] = field(default_factory=list)
    # 每个 fact: {"wit_id": str, "caption": str, "is_primary": bool}
    # is_primary=True 的是必须覆盖的核心事实

    @property
    def primary_wit_ids(self) -> set[str]:
        return {f["wit_id"] for f in self.facts if f.get("is_primary", True)}

    @property
    def all_wit_ids(self) -> set[str]:
        return {f["wit_id"] for f in self.facts}

    def coverage(self, citations: list[CitationObject]) -> float:
        """计算 citations 对 primary facts 的覆盖率"""
        if not self.primary_wit_ids:
            return 1.0
        cited_wit_ids = {c.source_wit_id for c in citations}
        covered = self.primary_wit_ids & cited_wit_ids
        return len(covered) / len(self.primary_wit_ids)

    def is_relevant_citation(self, citation: CitationObject) -> bool:
        """判断一条 citation 是否引用了 GT 事实集中的文档"""
        return citation.source_wit_id in self.all_wit_ids


# ═══════════════════════════════════════════════════════════
# 5. ResearchState（替代 EnvState）
# ═══════════════════════════════════════════════════════════

@dataclass
class ResearchState:
    """环境完整可序列化状态 — search/report world。
    
    设计原则：
    - 大块文本（document body, report draft）不参与 hash
    - 用 stable ID 追踪 research progress
    - UI shell state 与 research progress 分开字段但同一类
    """
    # ── Research progress state ──
    current_phase: ResearchPhase = ResearchPhase.FORMULATE
    query_history: list[str] = field(default_factory=list)      # 已执行的 queries
    current_result_ids: list[int] = field(default_factory=list)  # 当前搜索结果的 result_ids
    opened_result_id: Optional[int] = None                       # 当前打开的文档
    citation_count: int = 0                                       # 已收集引用数
    cited_wit_ids: list[str] = field(default_factory=list)       # 已引用的 wit_ids（用于 hash）
    report_submitted: bool = False
    step_count: int = 0
    zoom_count: int = 0                                           # 累计 zoom_search 次数
    last_zoom_rank_improvement: int = 0                           # 上次zoom带来的GT rank提升（正=提升）

    # ── UI shell state（复用壳子）──
    current_page: str = "formulate"     # 页面 ID（用于渲染器选择模板）
    active_modal: Optional[str] = None
    active_panel: Optional[str] = None  # "evidence" | None

    # ── v1 backward compat properties (for template_renderer.py legacy templates) ──
    @property
    def selected_product_id(self):
        return self.opened_result_id

    @property
    def active_dropdown(self):
        return None

    @property
    def active_filter(self) -> dict:
        return {}

    @property
    def page_family(self):
        """v1 compat: map current_phase to PageFamily for template_renderer."""
        from searcheyes.screen_ir import PageFamily
        mapping = {
            ResearchPhase.FORMULATE: PageFamily.SEARCH,
            ResearchPhase.SEARCH: PageFamily.RESULTS,
            ResearchPhase.BROWSE: PageFamily.DOCUMENT,
            ResearchPhase.SYNTHESIZE: PageFamily.REPORT,
        }
        return mapping.get(self.current_phase, PageFamily.SEARCH)

    def hash(self) -> str:
        """只 hash 影响决策的状态字段，排除大块文本"""
        hashable = {
            "phase": self.current_phase.value,
            "query_count": len(self.query_history),
            "last_query": self.query_history[-1] if self.query_history else "",
            "result_count": len(self.current_result_ids),
            "opened": self.opened_result_id,
            "citation_count": self.citation_count,
            "cited_wits": sorted(self.cited_wit_ids),
            "submitted": self.report_submitted,
            "step": self.step_count,
            "zoom_count": self.zoom_count,
            "page": self.current_page,
            "modal": self.active_modal,
            "panel": self.active_panel,
        }
        s = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════
# 6. ResearchTask（替代 VisualTask 的研究任务 schema）
# ═══════════════════════════════════════════════════════════

class ResearchDifficulty(str, Enum):
    EASY   = "easy"    # 1 search, 1-2 citations needed
    MEDIUM = "medium"  # 1-2 searches, 2-3 citations, may need refine
    HARD   = "hard"    # 2-3 searches, 3+ citations, requires refine_query


@dataclass
class ResearchTask:
    """研究任务 — 可有多种有效轨迹完成"""
    # 基础信息
    task_id: str
    goal: str                                   # 自然语言研究问题
    difficulty: ResearchDifficulty = ResearchDifficulty.MEDIUM

    # Ground Truth
    fact_set: FactSet = field(default_factory=FactSet)  # GT 事实集合
    ground_truth_wit_id: str = ""              # 主要目标 WIT ID（向后兼容）
    ground_truth_caption: str = ""

    # 任务参数
    max_steps: int = 12
    min_citations_for_submit: int = 1          # 防 "submit early" hack

    # WIT bindings（检索池）
    wit_bindings: list[dict] = field(default_factory=list)

    # 元数据
    difficulty_tags: list[str] = field(default_factory=list)
    requires_rag: bool = True

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load_from_dict(cls, data: dict) -> ResearchTask:
        import dataclasses as _dc
        valid_fields = {f.name for f in _dc.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        filtered["difficulty"] = ResearchDifficulty(filtered.get("difficulty", "medium"))
        if "fact_set" in filtered and isinstance(filtered["fact_set"], dict):
            filtered["fact_set"] = FactSet(**filtered["fact_set"])
        return cls(**filtered)

    @classmethod
    def from_visual_task(cls, vt_data: dict) -> ResearchTask:
        """从 v1 VisualTask JSON 转换为 ResearchTask。
        
        映射逻辑：
        - goal: 移除"购买"等电商语义，改为"找到并引用相关信息"
        - ground_truth_wit_id → fact_set (single primary fact)
        - wit_bindings 保留（检索池）
        """
        import re
        goal = vt_data.get("goal", "")
        # 清洗电商语义 — template-level (6 templates cover all 200 tasks)
        _templates = [
            (r"放大查看每个商品的图片进行视觉调研，找到与(「[^」]+」)最相关的商品并购买",
             r"查看各文档图片进行视觉调研，找到关于\1最相关的文档并引用相关证据"),
            (r"搜索商品并放大查看图片，确认哪个商品展示了(「[^」]+」)后购买",
             r"搜索并查看文档图片，确认哪个文档展示了\1后引用相关证据"),
            (r"搜索(「[^」]+」)相关商品，比较价格后购买最便宜的",
             r"搜索\1相关文档，分析内容后引用最相关的证据"),
            (r"查看各商品图片，找到展示了(「[^」]+」)的商品并购买",
             r"查看各文档图片，找到展示了\1的文档并引用相关证据"),
            (r"搜索商品，找到与(「[^」]+」)相关的产品并购买",
             r"搜索文档，找到关于\1的相关文档并引用证据"),
            (r"根据图片内容判断哪个商品展示了(「[^」]+」)，将其加入购物车",
             r"根据图片内容判断哪个文档展示了\1，引用相关证据"),
        ]
        for pattern, replacement in _templates:
            new_goal = re.sub(pattern, replacement, goal)
            if new_goal != goal:
                goal = new_goal
                break
        else:
            # Fallback word-level for unknown templates
            for old, new in [("比较价格后购买最便宜的", "分析内容后引用最相关的证据"),
                             ("将其加入购物车", "引用相关证据"),
                             ("并购买", "并引用相关证据"), ("后购买", "后引用相关证据"),
                             ("购买", "引用"), ("产品", "文档"), ("商品", "文档"),
                             ("找到与", "找到关于")]:
                goal = goal.replace(old, new)

        gt_wit_id = vt_data.get("ground_truth_wit_id", "")
        gt_caption = vt_data.get("ground_truth_caption", "")

        fact_set = FactSet(facts=[{
            "wit_id": gt_wit_id,
            "caption": gt_caption,
            "is_primary": True,
        }])

        return cls(
            task_id=vt_data.get("task_id", ""),
            goal=goal,
            difficulty=ResearchDifficulty(vt_data.get("difficulty", "medium")),
            fact_set=fact_set,
            ground_truth_wit_id=gt_wit_id,
            ground_truth_caption=gt_caption,
            max_steps=vt_data.get("dag_depth", 5) * 2,
            min_citations_for_submit=1,
            wit_bindings=vt_data.get("wit_bindings", []),
            difficulty_tags=vt_data.get("difficulty_tags", []),
            requires_rag=vt_data.get("requires_rag", True),
        )

    def validate(self) -> tuple[bool, list[str]]:
        errors = []
        if not self.goal:
            errors.append("goal is empty")
        if not self.fact_set.primary_wit_ids:
            errors.append("no primary facts in fact_set")
        if self.max_steps < 3:
            errors.append("max_steps too small (minimum 3)")
        if self.min_citations_for_submit < 1:
            errors.append("min_citations_for_submit must be >= 1")
        return len(errors) == 0, errors

    def summary(self) -> str:
        return (
            f"ResearchTask[{self.task_id}] {self.difficulty.value} "
            f"facts={len(self.fact_set.primary_wit_ids)} "
            f"max_steps={self.max_steps} | {self.goal[:60]}"
        )
