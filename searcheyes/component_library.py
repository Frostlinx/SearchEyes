"""
component_library.py - UI Component Library Manager
=====================================================
管理提取的 UI 组件，支持按类型检索、质量过滤、混搭组装合成页面。
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ComponentEntry:
    """组件库条目"""
    component_id: str
    component_type: str
    source_url: str
    source_page_id: str
    quality_score: float
    bbox: dict
    text_preview: str
    child_count: int
    computed_styles: dict
    html_file: str  # 相对路径


class ComponentLibrary:
    """组件库：索引、检索、采样"""

    def __init__(self, library_dir: str | Path = "data/component_library"):
        self.library_dir = Path(library_dir)
        self.entries: list[ComponentEntry] = []
        self.by_type: dict[str, list[ComponentEntry]] = defaultdict(list)
        self._load_all()

    def _load_all(self):
        """扫描所有页面目录，加载组件索引"""
        self.entries.clear()
        self.by_type.clear()

        if not self.library_dir.exists():
            return

        for page_dir in sorted(self.library_dir.iterdir()):
            if not page_dir.is_dir():
                continue
            index_file = page_dir / "components.json"
            if not index_file.exists():
                continue

            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            page_id = data.get("page_id", page_dir.name)
            source_url = data.get("source_url", "")

            for comp in data.get("components", []):
                entry = ComponentEntry(
                    component_id=comp["component_id"],
                    component_type=comp["component_type"],
                    source_url=source_url,
                    source_page_id=page_id,
                    quality_score=comp.get("quality_score", 0),
                    bbox=comp.get("bbox", {}),
                    text_preview=comp.get("text_preview", ""),
                    child_count=comp.get("child_count", 0),
                    computed_styles=comp.get("computed_styles", {}),
                    html_file=comp.get("html_file", ""),
                )
                self.entries.append(entry)
                self.by_type[entry.component_type].append(entry)

    def get_types(self) -> dict[str, int]:
        """返回各类型组件数量"""
        return {t: len(entries) for t, entries in sorted(self.by_type.items())}

    def query(
        self,
        component_type: str,
        min_quality: float = 0.0,
        exclude_pages: set[str] | None = None,
    ) -> list[ComponentEntry]:
        """按类型查询组件"""
        exclude_pages = exclude_pages or set()
        results = []
        for entry in self.by_type.get(component_type, []):
            if entry.quality_score >= min_quality:
                if entry.source_page_id not in exclude_pages:
                    results.append(entry)
        return sorted(results, key=lambda e: -e.quality_score)

    def sample(
        self,
        component_type: str,
        n: int = 1,
        min_quality: float = 0.3,
        exclude_pages: set[str] | None = None,
    ) -> list[ComponentEntry]:
        """随机采样指定类型的组件"""
        candidates = self.query(component_type, min_quality, exclude_pages)
        if not candidates:
            return []
        return random.sample(candidates, min(n, len(candidates)))

    def load_html(self, entry: ComponentEntry) -> str:
        """加载组件的 HTML 片段"""
        html_path = self.library_dir / entry.source_page_id / entry.html_file
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return ""

    def statistics(self) -> dict:
        """统计信息"""
        total = len(self.entries)
        pages = len(set(e.source_page_id for e in self.entries))
        avg_quality = (
            sum(e.quality_score for e in self.entries) / total if total else 0
        )
        high_quality = sum(1 for e in self.entries if e.quality_score >= 0.7)

        return {
            "total_components": total,
            "source_pages": pages,
            "types": self.get_types(),
            "average_quality": round(avg_quality, 3),
            "high_quality_count": high_quality,
        }


# ── 合成页面组装器 ──────────────────────────────────────────

@dataclass
class SyntheticPageSpec:
    """合成页面规格"""
    page_family: str          # search / results / detail / form / ranking
    navbar: Optional[str] = None       # component_id
    main_content: list[str] = field(default_factory=list)  # component_ids
    footer: Optional[str] = None       # component_id
    sidebar: Optional[str] = None      # component_id
    style_overrides: dict = field(default_factory=dict)


class SyntheticPageAssembler:
    """
    从组件库中采样组件，组装成合成页面。
    
    策略：
    1. 按页面族选择必需组件（navbar + main + footer）
    2. 从不同来源网站采样，确保视觉多样性
    3. 统一 CSS 变量，使混搭组件风格协调
    """

    # 各页面族需要的组件类型
    PAGE_RECIPES = {
        "search": {
            "required": ["navbar", "search_bar"],
            "optional": ["hero", "footer"],
        },
        "results": {
            "required": ["navbar"],
            "main_choices": ["card_grid", "list_group", "table"],
            "optional": ["search_bar", "sidebar", "pagination", "footer"],
        },
        "detail": {
            "required": ["navbar"],
            "main_choices": ["card"],
            "optional": ["breadcrumb", "sidebar", "footer"],
        },
        "form": {
            "required": ["navbar", "form_group"],
            "optional": ["footer"],
        },
        "ranking": {
            "required": ["navbar"],
            "main_choices": ["table", "list_group"],
            "optional": ["tab_bar", "footer"],
        },
    }

    def __init__(self, library: ComponentLibrary):
        self.library = library

    def generate_spec(
        self,
        page_family: str,
        style_overrides: dict | None = None,
    ) -> SyntheticPageSpec:
        """生成一个合成页面规格"""
        recipe = self.PAGE_RECIPES.get(page_family, self.PAGE_RECIPES["results"])
        used_pages: set[str] = set()  # 跟踪已用来源，鼓励跨站混搭

        spec = SyntheticPageSpec(
            page_family=page_family,
            style_overrides=style_overrides or {},
        )

        # Navbar
        if "navbar" in recipe.get("required", []):
            navbars = self.library.sample("navbar", 1, min_quality=0.3, exclude_pages=used_pages)
            if navbars:
                spec.navbar = navbars[0].component_id
                used_pages.add(navbars[0].source_page_id)

        # Main content
        main_types = recipe.get("main_choices", [])
        required = [t for t in recipe.get("required", []) if t != "navbar"]

        for comp_type in required + main_types[:1]:
            comps = self.library.sample(comp_type, 1, min_quality=0.2, exclude_pages=used_pages)
            if comps:
                spec.main_content.append(comps[0].component_id)
                used_pages.add(comps[0].source_page_id)

        # Optional
        for comp_type in recipe.get("optional", []):
            if random.random() < 0.5:  # 50% 概率包含可选组件
                comps = self.library.sample(comp_type, 1, min_quality=0.3)
                if comps:
                    if comp_type == "footer":
                        spec.footer = comps[0].component_id
                    elif comp_type == "sidebar":
                        spec.sidebar = comps[0].component_id
                    else:
                        spec.main_content.append(comps[0].component_id)

        return spec

    def assemble_html(
        self,
        spec: SyntheticPageSpec,
        color_scheme: dict | None = None,
    ) -> str:
        """将规格组装成完整 HTML"""
        color_scheme = color_scheme or {
            "primary": "#4f46e5",
            "secondary": "#06b6d4",
            "bg": "#ffffff",
            "text": "#1e293b",
            "muted": "#64748b",
            "border": "#e2e8f0",
            "nav_bg": "#1e293b",
            "nav_text": "#f8fafc",
        }

        # CSS Reset + 变量
        css_vars = "\n".join(f"  --{k}: {v};" for k, v in color_scheme.items())

        parts = []
        parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Synthetic Page - {spec.page_family}</title>
<style>
:root {{
{css_vars}
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}}
.synth-navbar {{ background: var(--nav_bg); color: var(--nav_text); padding: 12px 24px; }}
.synth-main {{ max-width: 1200px; margin: 24px auto; padding: 0 24px; }}
.synth-sidebar {{ float: right; width: 280px; padding: 16px; }}
.synth-footer {{ background: var(--nav_bg); color: var(--nav_text); padding: 24px; text-align: center; margin-top: 48px; }}
.synth-section {{ margin-bottom: 24px; }}
</style>
</head>
<body>""")

        # Navbar
        if spec.navbar:
            html = self._load_component_html(spec.navbar)
            parts.append(f'<div class="synth-navbar synth-section">{html}</div>')

        # Layout
        parts.append('<div class="synth-main">')

        if spec.sidebar:
            html = self._load_component_html(spec.sidebar)
            parts.append(f'<aside class="synth-sidebar synth-section">{html}</aside>')

        # Main content
        for comp_id in spec.main_content:
            html = self._load_component_html(comp_id)
            parts.append(f'<div class="synth-section">{html}</div>')

        parts.append('</div>')

        # Footer
        if spec.footer:
            html = self._load_component_html(spec.footer)
            parts.append(f'<div class="synth-footer">{html}</div>')

        parts.append('</body></html>')

        return "\n".join(parts)

    def _load_component_html(self, component_id: str) -> str:
        """通过 ID 加载组件 HTML"""
        for entry in self.library.entries:
            if entry.component_id == component_id:
                return self.library.load_html(entry)
        return f"<!-- component {component_id} not found -->"

    def batch_generate(
        self,
        num_pages: int = 10,
        page_families: list[str] | None = None,
    ) -> list[tuple[SyntheticPageSpec, str]]:
        """批量生成合成页面"""
        families = page_families or ["search", "results", "detail", "form", "ranking"]
        results = []

        for i in range(num_pages):
            family = families[i % len(families)]
            spec = self.generate_spec(family)
            html = self.assemble_html(spec)
            results.append((spec, html))

        return results
