"""
screen_ir.py — ScreenIR 中间表示
==================================
所有前端采集、模板渲染、任务合成、实验评估都围绕这个数据结构。
它是连接"真实网页皮囊"与"受控沙盒逻辑"的统一桥梁。

v2: search/report world 页面族和语义类型。
"""

from __future__ import annotations
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


class PageFamily(str, Enum):
    """页面族分类 — v2 search/report world"""
    SEARCH    = "search"       # 搜索输入页 (formulate)
    RESULTS   = "results"      # 搜索结果列表页
    DOCUMENT  = "document"     # 文档详情页 (替代 DETAIL)
    REPORT    = "report"       # 报告编写/提交页
    MODAL     = "modal"        # 模态弹窗

    # v1 兼容 (不再主动使用，但保留避免已有代码 import 报错)
    DETAIL    = "detail"
    FORM      = "form"
    RANKING   = "ranking"


class ElementSemanticType(str, Enum):
    """可交互元素的统一语义归类"""
    # ── 通用 UI ──
    NAV_LINK        = "nav_link"
    SEARCH_INPUT    = "search_input"
    SEARCH_BUTTON   = "search_button"
    BACK_BUTTON     = "back_button"
    GENERIC_LINK    = "generic_link"
    GENERIC_BUTTON  = "generic_button"
    REAL_IMAGE      = "real_image"
    UNKNOWN         = "unknown"

    # ── v2 研究 UI ──
    RESULT_ITEM     = "result_item"       # 搜索结果条目 (可点击打开文档)
    CITE_BUTTON     = "cite_button"       # 引用按钮 (在文档页)
    SUBMIT_BUTTON   = "submit_button"     # 提交报告按钮
    EVIDENCE_PANEL  = "evidence_panel"    # 已收集证据面板
    REPORT_PANEL    = "report_panel"      # 报告编写面板
    REFINE_BUTTON   = "refine_button"     # 重新搜索 / 改写 query 按钮
    DOC_BODY        = "doc_body"          # 文档正文区域
    DOC_IMAGE       = "doc_image"         # 文档中的图片

    # ── 通用 UI 控件 ──
    FILTER_CONTROL  = "filter_control"
    PAGINATION      = "pagination"
    MODAL_TRIGGER   = "modal_trigger"
    FORM_FIELD      = "form_field"
    FORM_SUBMIT     = "form_submit"
    TAB_SWITCH      = "tab_switch"
    DROPDOWN        = "dropdown"


@dataclass
class BBox:
    """元素的精确边界框"""
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Interactable:
    """一个可交互 DOM 元素的完整描述"""
    element_id: int
    tag: str
    text: str
    bbox: BBox
    role: str = ""
    href: str = ""
    input_type: str = ""
    css_classes: list[str] = field(default_factory=list)
    is_visible: bool = True
    parent_section: str = ""
    semantic_type: str = "unknown"
    image_path: str = ""
    image_caption: str = ""
    wit_id: str = ""


@dataclass
class StyleDigest:
    """页面视觉风格摘要"""
    primary_color: str = ""
    bg_color: str = ""
    font_family: str = ""
    layout_mode: str = ""
    has_sidebar: bool = False
    has_navbar: bool = False
    has_footer: bool = False


@dataclass
class StyleBundle(StyleDigest):
    """用于模板注入的完整样式变量集合。"""
    style_id: str = ""
    source_url: str = ""
    secondary_color: str = ""
    accent_color: str = ""
    surface_color: str = ""
    text_color: str = ""
    muted_text_color: str = ""
    border_color: str = ""
    nav_bg_color: str = ""
    nav_text_color: str = ""
    hero_gradient_from: str = ""
    hero_gradient_to: str = ""
    card_radius: float = 12.0
    button_radius: float = 8.0
    shadow_strength: float = 0.06
    spacing_scale: float = 1.0
    density: str = "normal"


@dataclass
class ScreenIR:
    """
    Screen Intermediate Representation
    ===================================
    一个网页的完整中间表示。
    """
    page_id: str
    source_url: str
    page_family: PageFamily
    title: str = ""
    screenshot_path: str = ""
    viewport_width: int = 1280
    viewport_height: int = 720
    interactables: list[Interactable] = field(default_factory=list)
    style: StyleBundle = field(default_factory=StyleBundle)
    ui_tokens: list[str] = field(default_factory=list)
    dom_element_count: int = 0
    text_content_hash: str = ""
    crawl_timestamp: str = ""

    def generate_ui_tokens(self) -> list[str]:
        tokens = []
        tokens.append("PAGE:" + self.page_family.value)
        sections = set(e.parent_section for e in self.interactables if e.is_visible)
        for s in sorted(sections):
            count = len([e for e in self.interactables if e.parent_section == s and e.is_visible])
            tokens.append(s.upper() + ":" + str(count) + "items")
        if self.style.has_navbar:
            tokens.append("LAYOUT:navbar")
        if self.style.has_sidebar:
            tokens.append("LAYOUT:sidebar")
        if self.style.has_footer:
            tokens.append("LAYOUT:footer")
        if self.style.density:
            tokens.append("STYLE:density:" + self.style.density)
        if self.style.font_family:
            tokens.append("STYLE:font:" + self.style.font_family)
        from collections import Counter
        sem_dist = Counter(e.semantic_type for e in self.interactables if e.is_visible)
        for sem, cnt in sem_dist.most_common(5):
            if sem != "unknown":
                tokens.append("SEM:" + sem + ":" + str(cnt))
        self.ui_tokens = tokens
        return tokens

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load(cls, path: str | Path) -> ScreenIR:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["page_family"] = PageFamily(data["page_family"])
        data["style"] = StyleBundle(**data.get("style", {}))
        data["interactables"] = [
            Interactable(
                **{**item, "bbox": BBox(**item["bbox"])}
            )
            for item in data.get("interactables", [])
        ]
        return cls(**data)

    @property
    def clickable_count(self) -> int:
        return len([e for e in self.interactables if e.is_visible])

    def get_element_by_id(self, element_id: int) -> Optional[Interactable]:
        for e in self.interactables:
            if e.element_id == element_id:
                return e
        return None

    def summary(self) -> str:
        return (
            "ScreenIR[" + self.page_id + "] family=" + self.page_family.value +
            " elements=" + str(self.clickable_count) +
            " viewport=" + str(self.viewport_width) + "x" + str(self.viewport_height)
        )
