"""
task_schema.py — P2 视觉断层任务 Schema
==========================================
定义正式的任务数据结构。每条任务必须包含视觉断层
(Visual Discontinuity) 的显式量化，而非仅靠标签隐含。
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Optional


class DifficultyLevel(str, Enum):
    EASY   = "easy"    # 1-2 步, 0-1 次 zoom
    MEDIUM = "medium"  # 3-5 步, 1-2 次 zoom
    HARD   = "hard"    # 5-10 步, 3+ 次 zoom, 多页跳转


class DifficultyTag(str, Enum):
    ZOOM_REQUIRED     = "zoom_required"      # 需要放大才能看清目标
    VISUAL_AMBIGUITY  = "visual_ambiguity"   # 多个相似元素需区分
    MULTI_PAGE        = "multi_page"         # 跨页面跳转
    COMPARISON        = "comparison"         # 需要比较多个值
    SMALL_TARGET      = "small_target"       # 目标元素面积 < 0.5% 视口
    HIDDEN_INFO       = "hidden_info"        # 信息需 hover/expand 才可见
    NUMERICAL_REASONING = "numerical_reasoning"  # 需要数值推理 (如找最低价)
    RAG_KNOWLEDGE     = "rag_knowledge"      # 需要 RAG 检索外部知识才能完成


@dataclass
class TrajectoryStep:
    """ground truth 轨迹中的单步"""
    step_idx: int
    state: str                   # 当前 FSM 状态
    action: str                  # 执行的动作 (search, open_result, cite_source, etc.)
    action_params: dict = field(default_factory=dict)
    target_element_id: Optional[int] = None
    target_bbox: Optional[dict] = None  # {"x", "y", "width", "height"}
    requires_zoom: bool = False  # 本步是否构成视觉断层（需 zoom)
    cot_reasoning: str = ""      # Chain-of-Thought 推理


@dataclass
class VisualTask:
    """
    视觉断层任务。
    
    视觉断层 (Visual Discontinuity) = 任务中存在的、仅靠当前视口
    无法获取所需信息的"信息鸿沟"。Agent 必须通过视觉锚点定位、
    局部放大(zoom)、跨页面跳转才能跨越这些断层。
    """
    # 基础信息
    task_id: str
    goal: str                                    # 自然语言任务描述
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM

    # 页面与状态
    page_family_sequence: list[str] = field(default_factory=list)  # ["search", "results", "detail"]
    initial_state: str = "search"

    # 视觉断层核心字段
    visual_anchors: list[str] = field(default_factory=list)  # 视觉锚点 ID
    visual_gap_count: int = 0    # 需跨越的视觉断层次数
    zoom_budget: int = 0         # 至少需要几次 zoom 才可解
    dag_depth: int = 0           # 最短路径步数

    # Ground Truth
    ground_truth_trajectory: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str = ""

    # 标签
    difficulty_tags: list[str] = field(default_factory=list)

    # RAG ground truth（Phase C3 视觉调研任务）
    requires_rag: bool = False
    ground_truth_wit_id: str = ""       # 目标商品绑定的 WIT 条目 ID
    ground_truth_caption: str = ""      # 目标商品图片的 GT caption

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load_from_dict(cls, data: dict) -> VisualTask:
        import dataclasses as _dc
        valid_fields = {f.name for f in _dc.fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_fields}
        data["difficulty"] = DifficultyLevel(data["difficulty"])
        data["ground_truth_trajectory"] = [
            TrajectoryStep(**s) for s in data.get("ground_truth_trajectory", [])
        ]
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> VisualTask:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.load_from_dict(data)

    def validate(self) -> tuple[bool, list[str]]:
        """校验任务合法性"""
        errors = []
        if not self.goal:
            errors.append("goal 为空")
        if self.dag_depth < 1:
            errors.append("dag_depth 必须 >= 1")
        if self.visual_gap_count < 0:
            errors.append("visual_gap_count 不能为负")
        if self.zoom_budget > self.visual_gap_count:
            errors.append("zoom_budget 不应超过 visual_gap_count")
        if len(self.ground_truth_trajectory) < self.dag_depth:
            errors.append(f"轨迹长度({len(self.ground_truth_trajectory)}) < dag_depth({self.dag_depth})")
        zoom_steps = sum(1 for s in self.ground_truth_trajectory if s.requires_zoom)
        if zoom_steps < self.zoom_budget:
            errors.append(f"轨迹中 zoom 步数({zoom_steps}) < zoom_budget({self.zoom_budget})")
        return len(errors) == 0, errors

    def summary(self) -> str:
        return (f"Task[{self.task_id}] {self.difficulty.value} "
                f"gaps={self.visual_gap_count} zoom={self.zoom_budget} "
                f"depth={self.dag_depth} | {self.goal[:50]}")
