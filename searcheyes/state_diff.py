"""
state_diff.py — StateDiff 一等公民协议
=========================================
整个四层引擎的合法性建立在"只处理受控事件"的前提上。
StateDiff 是 transition_engine 和 delta_world_model 之间的唯一通信协议。
禁止任何不在此协议中定义的事件类型流入视觉层。

v2: search/report world 事件类型。
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class DiffEventType(str, Enum):
    """将所有合法的状态变化事件类型固定下来，禁止自由幻想。

    v2 事件类型分为三组:
    1. 通用 UI shell 事件（v1 保留）
    2. 研究 workflow 事件（v2 新增）
    3. 已废弃的 shopping 事件（移除）
    """
    # ── 通用 UI shell 事件（保留） ──
    MODAL_OPENED        = "modal_opened"
    MODAL_CLOSED        = "modal_closed"
    PAGE_NAVIGATED      = "page_navigated"
    TOAST_SHOWN         = "toast_shown"

    # ── 研究 workflow 事件（v2 新增） ──
    RESULTS_RETURNED    = "results_returned"     # search 返回结果
    DOCUMENT_OPENED     = "document_opened"      # 打开某篇文档
    EVIDENCE_COLLECTED  = "evidence_collected"   # cite_source 收集了证据
    QUERY_REFINED       = "query_refined"        # refine_query 改写了搜索
    REPORT_SUBMITTED    = "report_submitted"     # submit_report 提交了报告
    ZOOM_SEARCH_COMPLETED = "zoom_search_completed"  # zoom_search 完成了局部检索


@dataclass
class DiffEvent:
    """单个状态变化事件"""
    event_type: DiffEventType
    target_element_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    # payload 示例:
    #   results_returned: {"query": "...", "result_count": 6}
    #   document_opened: {"result_id": 3, "wit_id": "wit_0747", "title": "..."}
    #   evidence_collected: {"citation_id": 1, "source_wit_id": "wit_0747"}
    #   query_refined: {"old_query": "...", "new_query": "..."}
    #   report_submitted: {"citation_count": 3, "coverage": 0.85}
    #   page_navigated: {"from_page": "results", "to_page": "document_3"}
    #   toast_shown: {"message": "Evidence collected", "type": "success"}


@dataclass
class StateDiff:
    """
    一次 transition 产生的完整状态差分。
    transition_engine 的唯一输出、delta_world_model 的唯一输入。
    """
    events: list[DiffEvent] = field(default_factory=list)
    old_state_hash: str = ""
    new_state_hash: str = ""

    @property
    def has_visual_change(self) -> bool:
        """是否包含需要视觉更新的事件"""
        visual_events = {
            DiffEventType.MODAL_OPENED, DiffEventType.MODAL_CLOSED,
            DiffEventType.PAGE_NAVIGATED, DiffEventType.TOAST_SHOWN,
            DiffEventType.RESULTS_RETURNED, DiffEventType.DOCUMENT_OPENED,
        }
        return any(e.event_type in visual_events for e in self.events)

    @property
    def has_data_change(self) -> bool:
        """是否包含研究进度变更"""
        data_events = {
            DiffEventType.EVIDENCE_COLLECTED,
            DiffEventType.QUERY_REFINED,
            DiffEventType.REPORT_SUBMITTED,
        }
        return any(e.event_type in data_events for e in self.events)

    def summary(self) -> str:
        types = [e.event_type.value for e in self.events]
        return "StateDiff[" + str(len(self.events)) + " events: " + ", ".join(types) + "]"
