"""
component_extractor.py - UI Component Extractor
=================================================
从真实网页中提取可复用的 UI 组件（navbar, card, form, footer 等），
保存为 HTML/CSS 片段，供合成页面混搭使用。

使用 Playwright DOM 分析而非 VLM 推理，更快更可靠。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright


# ── 组件类型定义 ──────────────────────────────────────────

COMPONENT_TYPES = [
    "navbar",       # 导航栏
    "hero",         # 首屏大图/标语区
    "search_bar",   # 搜索框
    "card",         # 卡片（商品/文章/项目）
    "card_grid",    # 卡片网格容器
    "list_item",    # 列表项
    "list_group",   # 列表容器
    "form_group",   # 表单组
    "button_group", # 按钮组
    "table",        # 表格
    "footer",       # 页脚
    "sidebar",      # 侧边栏
    "modal",        # 弹窗
    "breadcrumb",   # 面包屑
    "pagination",   # 分页
    "tab_bar",      # 标签栏
    "dropdown",     # 下拉菜单
]


@dataclass
class ExtractedComponent:
    """提取的 UI 组件"""
    component_id: str
    component_type: str          # navbar / card / form_group / ...
    source_url: str
    source_page_id: str
    html_snippet: str            # 组件的 HTML 片段
    css_snippet: str             # 组件的关键 CSS
    computed_styles: dict        # 计算后的样式属性
    bbox: dict                   # {x, y, w, h}
    text_content: str            # 文本内容摘要
    child_count: int = 0         # 子元素数量
    quality_score: float = 0.0   # 质量评分


@dataclass
class PageComponents:
    """一个页面提取出的所有组件"""
    page_id: str
    source_url: str
    title: str
    viewport: dict
    components: list[ExtractedComponent] = field(default_factory=list)
    global_css: str = ""         # 页面级 CSS 变量/reset
    extraction_time: str = ""


# ── 组件检测 JS ──────────────────────────────────────────

_DETECT_COMPONENTS_JS = """
() => {
    const components = [];
    let cid = 0;

    function getRect(el) {
        const r = el.getBoundingClientRect();
        return {x: r.x, y: r.y, w: r.width, h: r.height};
    }

    function isVisible(el) {
        const s = window.getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden' || s.opacity === '0') return false;
        const r = el.getBoundingClientRect();
        return r.width > 10 && r.height > 10;
    }

    function getComputedProps(el) {
        const s = window.getComputedStyle(el);
        return {
            backgroundColor: s.backgroundColor,
            color: s.color,
            fontSize: s.fontSize,
            fontFamily: s.fontFamily,
            borderRadius: s.borderRadius,
            boxShadow: s.boxShadow,
            padding: s.padding,
            margin: s.margin,
            display: s.display,
            flexDirection: s.flexDirection,
            gap: s.gap,
            border: s.border,
            textAlign: s.textAlign,
        };
    }

    // Per-type limits to avoid one type dominating
    const typeCounts = {};
    const MAX_PER_TYPE = 3;
    const seenRects = new Set();

    function addComponent(el, type) {
        if (!isVisible(el)) return;
        const rect = getRect(el);
        if (rect.w < 50 || rect.h < 20) return;

        // Per-type limit
        typeCounts[type] = (typeCounts[type] || 0);
        if (typeCounts[type] >= MAX_PER_TYPE) return;

        // Dedup by position (avoid overlapping elements)
        const key = `${Math.round(rect.x/20)}_${Math.round(rect.y/20)}_${Math.round(rect.w/20)}`;
        if (seenRects.has(key)) return;
        seenRects.add(key);

        typeCounts[type]++;
        components.push({
            id: cid++,
            type: type,
            tag: el.tagName.toLowerCase(),
            outerHTML: el.outerHTML.substring(0, 8000),
            computedStyles: getComputedProps(el),
            bbox: rect,
            text: (el.innerText || '').substring(0, 200).trim(),
            childCount: el.children.length,
            classes: Array.from(el.classList).slice(0, 10),
        });
    }

    // 1. Navbar / Header
    const navbars = document.querySelectorAll('nav, header, [role="navigation"], [role="banner"]');
    navbars.forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.y < 200 && rect.width > 500) {
            addComponent(el, 'navbar');
        }
    });

    // 2. Search bar
    const searchInputs = document.querySelectorAll(
        'input[type="search"], input[name*="search"], input[placeholder*="Search"], input[placeholder*="搜索"], form[role="search"]'
    );
    searchInputs.forEach(el => {
        const form = el.closest('form') || el.parentElement;
        if (form) addComponent(form, 'search_bar');
    });

    // 3. Cards (common patterns)
    const cardSelectors = [
        '[class*="card"]', '[class*="Card"]',
        '[class*="item"]', '[class*="Item"]',
        'article', '.repo-list li', '.athing',
        '[class*="product"]', '[class*="Product"]',
    ];
    const cardEls = document.querySelectorAll(cardSelectors.join(','));
    const seenCards = new Set();
    cardEls.forEach(el => {
        const rect = el.getBoundingClientRect();
        // 合理的卡片尺寸
        if (rect.width > 150 && rect.height > 80 && rect.width < 1200 && rect.height < 800) {
            const key = `${Math.round(rect.x)}_${Math.round(rect.y)}`;
            if (!seenCards.has(key)) {
                seenCards.add(key);
                addComponent(el, 'card');
            }
        }
    });

    // 4. Card grid / list container
    const containers = document.querySelectorAll(
        '[class*="grid"]', '[class*="list"]', '[class*="container"]',
        'ul', 'ol', 'tbody'
    );
    containers.forEach(el => {
        const rect = el.getBoundingClientRect();
        const children = el.children.length;
        if (children >= 3 && rect.width > 300 && rect.height > 200) {
            const s = window.getComputedStyle(el);
            if (s.display === 'grid' || s.display === 'flex') {
                addComponent(el, 'card_grid');
            } else if (el.tagName === 'TBODY') {
                addComponent(el.closest('table'), 'table');
            } else if (el.tagName === 'UL' || el.tagName === 'OL') {
                addComponent(el, 'list_group');
            }
        }
    });

    // 5. Forms
    const forms = document.querySelectorAll('form');
    forms.forEach(el => {
        const inputs = el.querySelectorAll('input, textarea, select');
        if (inputs.length >= 1) {
            addComponent(el, 'form_group');
        }
    });

    // 6. Footer
    const footers = document.querySelectorAll('footer, [role="contentinfo"]');
    footers.forEach(el => addComponent(el, 'footer'));

    // 7. Sidebar
    const sidebars = document.querySelectorAll('aside, [role="complementary"], [class*="sidebar"], [class*="Sidebar"]');
    sidebars.forEach(el => addComponent(el, 'sidebar'));

    // 8. Buttons
    const btnGroups = document.querySelectorAll('[class*="btn-group"], [class*="button-group"], [class*="actions"]');
    btnGroups.forEach(el => {
        const btns = el.querySelectorAll('button, a[class*="btn"]');
        if (btns.length >= 2) addComponent(el, 'button_group');
    });

    // 9. Tabs
    const tabs = document.querySelectorAll('[role="tablist"], [class*="tab"], [class*="Tab"]');
    tabs.forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.width > 200 && el.children.length >= 2) {
            addComponent(el, 'tab_bar');
        }
    });

    // 10. Pagination
    const pagers = document.querySelectorAll('[class*="pagination"], [class*="pager"], nav[aria-label*="page"]');
    pagers.forEach(el => addComponent(el, 'pagination'));

    // 11. Breadcrumb
    const breadcrumbs = document.querySelectorAll('[class*="breadcrumb"], [aria-label*="breadcrumb"], nav[class*="crumb"]');
    breadcrumbs.forEach(el => addComponent(el, 'breadcrumb'));

    // 12. Hero section
    const heroes = document.querySelectorAll('[class*="hero"], [class*="Hero"], [class*="banner"], [class*="jumbotron"]');
    heroes.forEach(el => addComponent(el, 'hero'));

    return components;
}
"""

_EXTRACT_GLOBAL_CSS_JS = """
() => {
    const root = document.documentElement;
    const s = window.getComputedStyle(root);
    const body = document.body;
    const bs = window.getComputedStyle(body);

    return {
        rootFontSize: s.fontSize,
        rootFontFamily: s.fontFamily,
        rootColor: s.color,
        rootBg: s.backgroundColor,
        bodyBg: bs.backgroundColor,
        bodyColor: bs.color,
        bodyFontFamily: bs.fontFamily,
        bodyFontSize: bs.fontSize,
        bodyLineHeight: bs.lineHeight,
    };
}
"""


# ── 主提取器 ──────────────────────────────────────────────

class ComponentExtractor:
    """从真实网页提取 UI 组件"""

    def __init__(self, output_dir: str | Path = "data/component_library"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def extract_from_url(
        self,
        url: str,
        page_id: str,
        viewport_w: int = 1280,
        viewport_h: int = 720,
        save_screenshot: bool = True,
    ) -> PageComponents:
        """从 URL 提取所有 UI 组件"""

        page_dir = self.output_dir / page_id
        page_dir.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": viewport_w, "height": viewport_h},
                device_scale_factor=1,
                locale="zh-CN",
            )
            page = await context.new_page()

            try:
                await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
            except Exception as e:
                await browser.close()
                raise RuntimeError(f"Page load failed: {url} -> {e}")

            # 截图
            if save_screenshot:
                await page.screenshot(path=str(page_dir / "screenshot.png"))

            title = await page.title() or ""

            # 提取组件
            raw_components = await page.evaluate(_DETECT_COMPONENTS_JS)
            global_css_info = await page.evaluate(_EXTRACT_GLOBAL_CSS_JS)

            await browser.close()

        # 转换为 ExtractedComponent
        components = []
        for raw in raw_components:
            cid = f"{page_id}__{raw['type']}_{raw['id']}"
            comp = ExtractedComponent(
                component_id=cid,
                component_type=raw["type"],
                source_url=url,
                source_page_id=page_id,
                html_snippet=raw["outerHTML"],
                css_snippet="",  # 后续可从 stylesheet 提取
                computed_styles=raw["computedStyles"],
                bbox=raw["bbox"],
                text_content=raw["text"],
                child_count=raw["childCount"],
                quality_score=self._score_component(raw),
            )
            components.append(comp)

        from datetime import datetime
        result = PageComponents(
            page_id=page_id,
            source_url=url,
            title=title,
            viewport={"w": viewport_w, "h": viewport_h},
            components=components,
            global_css=json.dumps(global_css_info, ensure_ascii=False),
            extraction_time=datetime.now().isoformat(),
        )

        # 保存
        self._save_page_components(result, page_dir)

        return result

    def _score_component(self, raw: dict) -> float:
        """基于启发式规则评分组件质量"""
        score = 0.0
        bbox = raw["bbox"]

        # 尺寸合理性
        w, h = bbox["w"], bbox["h"]
        if 100 < w < 1300 and 30 < h < 800:
            score += 0.3
        elif w > 50 and h > 20:
            score += 0.1

        # 有文本内容
        if len(raw.get("text", "")) > 5:
            score += 0.2

        # 有子元素（结构丰富）
        children = raw.get("childCount", 0)
        if 2 <= children <= 20:
            score += 0.2
        elif children > 0:
            score += 0.1

        # HTML 片段不太短也不太长
        html_len = len(raw.get("outerHTML", ""))
        if 200 < html_len < 5000:
            score += 0.2
        elif html_len > 50:
            score += 0.1

        # 有 CSS 类名（说明有样式）
        if len(raw.get("classes", [])) > 0:
            score += 0.1

        return min(1.0, score)

    def _save_page_components(self, page_comp: PageComponents, page_dir: Path):
        """保存提取结果"""
        # 保存索引
        index = {
            "page_id": page_comp.page_id,
            "source_url": page_comp.source_url,
            "title": page_comp.title,
            "viewport": page_comp.viewport,
            "global_css": page_comp.global_css,
            "extraction_time": page_comp.extraction_time,
            "component_count": len(page_comp.components),
            "components": [],
        }

        for comp in page_comp.components:
            # 每个组件单独保存 HTML
            comp_file = page_dir / f"{comp.component_id}.html"
            comp_file.write_text(comp.html_snippet, encoding="utf-8")

            index["components"].append({
                "component_id": comp.component_id,
                "component_type": comp.component_type,
                "bbox": comp.bbox,
                "text_preview": comp.text_content[:80],
                "child_count": comp.child_count,
                "quality_score": comp.quality_score,
                "html_file": comp_file.name,
                "computed_styles": comp.computed_styles,
            })

        index_file = page_dir / "components.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)


# ── 批量提取 ──────────────────────────────────────────────

# 目标网站列表（覆盖多种页面类型和视觉风格）
DEFAULT_TARGETS = [
    # 搜索类
    ("https://www.bing.com/", "bing_search"),
    # 列表/结果类
    ("https://github.com/trending", "github_trending"),
    ("https://news.ycombinator.com/", "hn_results"),
    # 详情类
    ("https://en.wikipedia.org/wiki/Artificial_intelligence", "wiki_ai"),
    # 表单类
    ("https://github.com/login", "github_login"),
    # 排行类
    ("https://www.tiobe.com/tiobe-index/", "tiobe_ranking"),
    # 通用
    ("https://example.com/", "example_basic"),
    ("https://httpbin.org/forms/post", "httpbin_form"),
]


async def batch_extract(
    targets: list[tuple[str, str]] | None = None,
    output_dir: str = "data/component_library",
) -> list[PageComponents]:
    """批量提取多个页面的组件"""
    targets = targets or DEFAULT_TARGETS
    extractor = ComponentExtractor(output_dir)

    results = []
    for url, pid in targets:
        print(f"\n[{pid}] {url}")
        try:
            page_comp = await extractor.extract_from_url(url, pid)
            comp_types = {}
            for c in page_comp.components:
                comp_types[c.component_type] = comp_types.get(c.component_type, 0) + 1
            type_str = ", ".join(f"{t}:{n}" for t, n in sorted(comp_types.items()))
            print(f"  -> {len(page_comp.components)} components: {type_str}")
            results.append(page_comp)
        except Exception as e:
            print(f"  -> FAILED: {e}")

    # 汇总统计
    total = sum(len(r.components) for r in results)
    print(f"\nDone: {len(results)}/{len(targets)} pages, {total} components total")

    return results


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    asyncio.run(batch_extract())
