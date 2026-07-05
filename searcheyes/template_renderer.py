"""
template_renderer.py — 第②层：ScreenIR→HTML 模板渲染
=======================================================
接收 EnvState + 产品数据 → 根据 PageFamily 选择模板 → 生成 HTML
→ 用 Playwright 渲染为截图，产出基底图供 DeltaWorldModel 叠加差分。

此层的核心职责：把抽象逻辑状态 "翻译" 为视觉皮囊。
模板设计吸收了 ScreenIR 采集到的真实网页样式先验（色系/布局/组件），
但逻辑完全由 TransitionEngine 控制——不保留原始业务逻辑。
"""

from __future__ import annotations
import asyncio
import hashlib
import json
from pathlib import Path
from playwright.async_api import async_playwright

from searcheyes.screen_ir import (
    ScreenIR, PageFamily, Interactable, BBox, StyleBundle, ElementSemanticType
)
from searcheyes.transition_engine import EnvState


# ═══════════════════════════════════════════════════════════
# HTML 模板：6 种 PageFamily 各一套
# ═══════════════════════════════════════════════════════════

def _default_style_bundle() -> StyleBundle:
    return StyleBundle(
        style_id="default_shopsim",
        primary_color="#2563eb",
        secondary_color="#0f766e",
        accent_color="#dc2626",
        bg_color="#f5f6fa",
        surface_color="#ffffff",
        text_color="#1f2937",
        muted_text_color="#6b7280",
        border_color="#dfe6e9",
        font_family="Segoe UI",
        nav_bg_color="#1f2937",
        nav_text_color="#ffffff",
        hero_gradient_from="#2563eb",
        hero_gradient_to="#0f766e",
        card_radius=14.0,
        button_radius=10.0,
        shadow_strength=0.08,
        spacing_scale=1.0,
        density="normal",
        has_navbar=True,
        has_footer=True,
    )


def _normalize_style(style: StyleBundle | None) -> StyleBundle:
    base = _default_style_bundle()
    if style is None:
        return base

    bundle = StyleBundle(**{**base.__dict__, **style.__dict__})
    bundle.primary_color = bundle.primary_color or base.primary_color
    bundle.secondary_color = bundle.secondary_color or bundle.primary_color or base.secondary_color
    bundle.accent_color = bundle.accent_color or "#dc2626"
    bundle.bg_color = bundle.bg_color or base.bg_color
    bundle.surface_color = bundle.surface_color or "#ffffff"
    bundle.text_color = bundle.text_color or base.text_color
    bundle.muted_text_color = bundle.muted_text_color or base.muted_text_color
    bundle.border_color = bundle.border_color or base.border_color
    bundle.font_family = bundle.font_family or base.font_family
    bundle.nav_bg_color = bundle.nav_bg_color or bundle.primary_color
    bundle.nav_text_color = bundle.nav_text_color or "#ffffff"
    bundle.hero_gradient_from = bundle.hero_gradient_from or bundle.primary_color
    bundle.hero_gradient_to = bundle.hero_gradient_to or bundle.secondary_color
    bundle.card_radius = max(6.0, bundle.card_radius or base.card_radius)
    bundle.button_radius = max(6.0, bundle.button_radius or base.button_radius)
    bundle.shadow_strength = min(0.24, max(0.02, bundle.shadow_strength or base.shadow_strength))
    bundle.spacing_scale = min(1.4, max(0.85, bundle.spacing_scale or base.spacing_scale))
    bundle.density = bundle.density or base.density
    return bundle


def _scale(style: StyleBundle, px: int) -> int:
    return max(4, int(round(px * style.spacing_scale)))


def _shadow(style: StyleBundle) -> str:
    blur = max(10, int(18 * style.spacing_scale))
    spread = max(2, int(8 * style.spacing_scale))
    return f"0 6px {blur}px rgba(15, 23, 42, {style.shadow_strength:.2f})"


def _common_head(style: StyleBundle) -> str:
    return f"""
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  :root {{
    --bg-color: {style.bg_color};
    --surface-color: {style.surface_color};
    --text-color: {style.text_color};
    --muted-text: {style.muted_text_color};
    --primary-color: {style.primary_color};
    --secondary-color: {style.secondary_color};
    --accent-color: {style.accent_color};
    --border-color: {style.border_color};
    --nav-bg: {style.nav_bg_color};
    --nav-text: {style.nav_text_color};
    --hero-from: {style.hero_gradient_from};
    --hero-to: {style.hero_gradient_to};
    --card-radius: {style.card_radius:.1f}px;
    --button-radius: {style.button_radius:.1f}px;
    --shadow: {_shadow(style)};
  }}
  body {{ font-family:'{style.font_family}','PingFang SC',sans-serif; background:var(--bg-color); color:var(--text-color); }}
  .navbar {{ background:var(--nav-bg); color:var(--nav-text); min-height:{_scale(style, 56)}px; display:flex; align-items:center; padding:0 {_scale(style, 24)}px; justify-content:space-between; }}
  .navbar .logo {{ font-size:{_scale(style, 20)}px; font-weight:700; letter-spacing:0.02em; }}
  .navbar .cart-badge {{ background:var(--accent-color); color:#fff; border-radius:999px; min-width:{_scale(style, 22)}px; height:{_scale(style, 22)}px; display:inline-flex; align-items:center; justify-content:center; font-size:{_scale(style, 12)}px; margin-left:{_scale(style, 6)}px; padding:0 {_scale(style, 6)}px; }}
  .btn {{ padding:{_scale(style, 10)}px {_scale(style, 24)}px; border:1px solid transparent; border-radius:var(--button-radius); cursor:pointer; font-size:{_scale(style, 14)}px; font-weight:600; transition:all .2s; }}
  .btn-primary {{ background:var(--primary-color); color:#fff; }}
  .btn-primary:hover {{ filter:brightness(0.94); }}
  .btn-danger {{ background:var(--accent-color); color:#fff; }}
  .btn-success {{ background:var(--secondary-color); color:#fff; }}
  .card {{ background:var(--surface-color); border-radius:var(--card-radius); box-shadow:var(--shadow); overflow:hidden; border:1px solid rgba(15, 23, 42, 0.03); }}
  .price {{ color:var(--accent-color); font-weight:700; font-size:{_scale(style, 20)}px; }}
  .small-price {{ font-size:{_scale(style, 12)}px; color:var(--accent-color); }}
  .tag {{ display:inline-block; padding:{_scale(style, 2)}px {_scale(style, 8)}px; border-radius:{max(4, int(style.button_radius * 0.5))}px; font-size:{_scale(style, 11)}px; margin-right:{_scale(style, 4)}px; }}
  input[type="text"],input[type="search"] {{ padding:{_scale(style, 10)}px {_scale(style, 16)}px; border:1px solid var(--border-color); border-radius:var(--button-radius); font-size:{_scale(style, 14)}px; width:100%; background:#fff; }}
  .footer {{ background:var(--nav-bg); color:var(--muted-text); padding:{_scale(style, 20)}px {_scale(style, 24)}px; text-align:center; font-size:{_scale(style, 12)}px; margin-top:{_scale(style, 40)}px; }}
</style>
"""


def _render_search_html(state: EnvState, products: dict, style: StyleBundle) -> str:
    hero_top = _scale(style, 120)
    hero_bottom = _scale(style, 60)
    hero_gap = _scale(style, 24)
    search_gap = _scale(style, 12)
    max_width = int(600 * style.spacing_scale)
    cite_count = getattr(state, "citation_count", 0)
    return f"""<!DOCTYPE html><html><head>{_common_head(style)}<title>SearchEyes - Research</title>
    <style>
      .search-hero {{ text-align:center; padding:{hero_top}px 24px {hero_bottom}px; }}
      .search-hero h1 {{ font-size:{_scale(style, 36)}px; margin-bottom:{hero_gap}px; background:linear-gradient(135deg,var(--hero-from),var(--hero-to)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
      .search-hero p {{ color:var(--muted-text); margin-bottom:{_scale(style, 12)}px; line-height:1.6; }}
      .search-box {{ max-width:{max_width}px; margin:0 auto; display:flex; gap:{search_gap}px; }}
      .search-box input {{ flex:1; padding:{_scale(style, 14)}px {_scale(style, 20)}px; font-size:{_scale(style, 16)}px; border-radius:{max(int(style.button_radius * 2.2), 18)}px; }}
      .search-box button {{ padding:{_scale(style, 14)}px {_scale(style, 32)}px; border-radius:{max(int(style.button_radius * 2.2), 18)}px; }}
    </style></head><body>
    <nav class="navbar"><span class="logo">🔍 Research Agent</span><span>Citations: {cite_count}</span></nav>
    <div class="search-hero">
      <h1>Visual Deep Research</h1>
      <p>Search the knowledge base to find evidence and answer research questions.</p>
      <p style="font-size:{_scale(style, 13)}px;color:var(--muted-text);">Enter a query to begin your investigation.</p>
      <div class="search-box">
        <input type="search" id="search-input" placeholder="Search Wikipedia knowledge base..." />
        <button class="btn btn-primary" id="search-btn">Search</button>
      </div>
    </div>
    <div class="footer">SearchEyes © 2026 — Multimodal Research Agent Environment</div>
    </body></html>"""


def _render_results_html(state, products, style):
    """v2: Search results as document list with titles, snippets, relevance."""
    s = style
    head = _common_head(s)
    cite_count = getattr(state, "citation_count", 0)
    qh = getattr(state, "query_history", [])
    query = qh[-1][:60] if qh else ""

    cards = ""
    for pid in sorted(products.keys()):
        p = products[pid]
        name = p.get("name", "Document")
        caption = p.get("image_caption", "")
        snippet = (caption[:150] + "...") if len(caption) > 150 else caption
        wit_id = p.get("wit_id", "")
        img = p.get("image_path", "")

        thumb = ""
        if img:
            thumb = (
                '<img id="thumb-' + str(pid) + '" src="file://' + img
                + '" style="width:80px;height:80px;object-fit:cover;border-radius:6px;flex-shrink:0;" />'
            )

        cards += (
            '<div id="result-' + str(pid) + '" style="display:flex;gap:16px;background:'
            + s.surface_color + ';padding:16px;border-radius:' + str(int(s.card_radius))
            + 'px;box-shadow:' + _shadow(s) + ';cursor:pointer;">'
            + thumb
            + '<div style="flex:1;">'
            '<h3 style="font-size:16px;margin-bottom:4px;color:' + s.primary_color + ';">'
            '[' + str(pid) + '] ' + name + '</h3>'
            '<p style="font-size:13px;color:' + s.muted_text_color + ';margin-bottom:4px;">'
            + snippet + '</p>'
            '<span style="font-size:11px;color:' + s.secondary_color + ';">wiki://' + wit_id + '</span>'
            '</div></div>'
        )

    submit_btn = ""
    if cite_count >= 1:
        submit_btn = (
            '<button id="submit-report" style="background:' + s.accent_color
            + ';color:white;border:none;padding:10px 20px;border-radius:'
            + str(int(s.button_radius)) + 'px;cursor:pointer;">Submit Report ('
            + str(cite_count) + ' citations)</button>'
        )

    html = head + (
        '</style></head>'
        '<body style="background:' + s.bg_color + ';font-family:' + s.font_family + ';color:' + s.text_color + ';">'
        '<nav style="background:' + s.nav_bg_color + ';color:' + s.nav_text_color
        + ';padding:12px 24px;display:flex;align-items:center;justify-content:space-between;">'
        '<div style="font-size:18px;font-weight:bold;">Research Agent</div>'
        '<div>Citations: ' + str(cite_count) + '</div>'
        '</nav>'
        '<div style="max-width:800px;margin:20px auto;padding:0 20px;">'
        '<div style="display:flex;gap:12px;margin-bottom:20px;">'
        '<input id="search-input" type="text" value="' + query
        + '" style="flex:1;padding:10px 16px;border:1px solid ' + s.border_color
        + ';border-radius:' + str(int(s.button_radius)) + 'px;font-size:14px;" />'
        '<button id="refine-btn" style="background:' + s.primary_color
        + ';color:white;border:none;padding:10px 16px;border-radius:'
        + str(int(s.button_radius)) + 'px;cursor:pointer;">Refine</button>'
        '</div>'
        '<div style="display:flex;flex-direction:column;gap:12px;">'
        + cards + '</div>'
        '<div style="margin-top:20px;text-align:center;">' + submit_btn + '</div>'
        '</div></body></html>'
    )
    return html


def _render_detail_html(state: EnvState, products: dict, style: StyleBundle) -> str:
    """v1 compat stub — redirects to document template in v2."""
    return _render_document_html(state, products, style)


def _render_form_html(state: EnvState, products: dict, style: StyleBundle) -> str:
    """v1 compat stub — redirects to report template in v2."""
    return _render_report_html(state, products, style)


def _render_ranking_html(state: EnvState, products: dict, style: StyleBundle) -> str:
    """v1 compat stub — redirects to results template in v2."""
    return _render_results_html(state, products, style)


def _render_modal_html(state: EnvState, products: dict, style: StyleBundle) -> str:
    """Modal overlay on top of current page (e.g. submit confirmation)."""
    base = _render_results_html(state, products, style)
    modal_name = state.active_modal or "confirm"
    s = style

    overlay = (
        '<div id="modal-overlay" style="position:fixed;top:0;left:0;width:100%;height:100%;'
        'background:rgba(0,0,0,0.4);z-index:100;display:flex;align-items:center;justify-content:center;">'
        '<div style="background:' + s.surface_color + ';width:' + str(int(420 * s.spacing_scale)) + 'px;'
        'padding:32px;border-radius:' + str(int(s.card_radius)) + 'px;box-shadow:' + _shadow(s) + ';">'
        '<h3 style="margin-bottom:16px;">' + modal_name + '</h3>'
        '<p style="color:' + s.muted_text_color + ';margin-bottom:24px;">Confirm this action?</p>'
        '<div style="display:flex;gap:12px;justify-content:flex-end;">'
        '<button id="modal-cancel" style="background:' + s.border_color + ';border:none;padding:8px 16px;'
        'border-radius:' + str(int(s.button_radius)) + 'px;cursor:pointer;">Cancel</button>'
        '<button id="modal-confirm" style="background:' + s.primary_color + ';color:white;border:none;'
        'padding:8px 16px;border-radius:' + str(int(s.button_radius)) + 'px;cursor:pointer;">Confirm</button>'
        '</div></div></div>'
    )
    return base.replace("</body>", overlay + "</body>")


# ═══════════════════════════════════════════════════════════
# TemplateRenderer 主类
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# v2 Research World Templates
# ═══════════════════════════════════════════════════════════

def _render_document_html(state, products, style):
    """v2: Document detail page with cite button."""
    s = style
    head = _common_head(s)
    doc_id = getattr(state, "opened_result_id", None)
    doc = products.get(doc_id, {}) if doc_id else {}
    title = doc.get("name", "Document")
    body = doc.get("image_caption", "No content available.")
    img = doc.get("image_path", "")
    wit_id = doc.get("wit_id", "")
    cite_count = getattr(state, "citation_count", 0)

    img_html = ""
    if img:
        img_html = (
            '<div style="margin:20px 0;text-align:center;">'
            '<img id="doc-image" src="file://' + img + '" '
            'style="max-width:100%;max-height:300px;border-radius:8px;" />'
            '</div>'
        )

    html = head + (
        '</style></head>'
        '<body style="background:' + s.bg_color + ';font-family:' + s.font_family + ';color:' + s.text_color + ';">'
        '<nav style="background:' + s.nav_bg_color + ';color:' + s.nav_text_color + ';padding:12px 24px;display:flex;align-items:center;justify-content:space-between;">'
        '<div style="font-size:18px;font-weight:bold;">Research Agent</div>'
        '<div>Citations: ' + str(cite_count) + '</div>'
        '</nav>'
        '<div style="max-width:800px;margin:20px auto;padding:20px;">'
        '<button id="back-btn" style="background:' + s.secondary_color + ';color:white;border:none;padding:8px 16px;border-radius:' + str(int(s.button_radius)) + 'px;cursor:pointer;margin-bottom:16px;">Back to Results</button>'
        '<div style="background:' + s.surface_color + ';padding:24px;border-radius:' + str(int(s.card_radius)) + 'px;box-shadow:' + _shadow(s) + ';">'
        '<h1 style="font-size:24px;margin-bottom:12px;">' + title + '</h1>'
        '<div style="color:' + s.muted_text_color + ';font-size:13px;margin-bottom:16px;">Source: wiki://' + wit_id + '</div>'
        + img_html +
        '<div id="doc-body" style="line-height:1.7;font-size:15px;">' + body + '</div>'
        '<div style="margin-top:20px;padding-top:16px;border-top:1px solid ' + s.border_color + ';">'
        '<button id="cite-btn" style="background:' + s.primary_color + ';color:white;border:none;padding:10px 20px;border-radius:' + str(int(s.button_radius)) + 'px;cursor:pointer;font-size:14px;">Cite this Source</button>'
        '</div></div></div></body></html>'
    )
    return html


def _render_report_html(state, products, style):
    """v2: Report submitted confirmation page."""
    s = style
    head = _common_head(s)
    cite_count = getattr(state, "citation_count", 0)

    html = head + (
        '</style></head>'
        '<body style="background:' + s.bg_color + ';font-family:' + s.font_family + ';color:' + s.text_color
        + ';display:flex;align-items:center;justify-content:center;min-height:100vh;">'
        '<div style="text-align:center;background:' + s.surface_color + ';padding:48px;border-radius:'
        + str(int(s.card_radius)) + 'px;box-shadow:' + _shadow(s) + ';">'
        '<div style="font-size:48px;margin-bottom:16px;">&#x2705;</div>'
        '<h1 style="font-size:24px;margin-bottom:8px;">Report Submitted</h1>'
        '<p style="color:' + s.muted_text_color + ';font-size:16px;">Citations collected: ' + str(cite_count) + '</p>'
        '</div></body></html>'
    )
    return html


_TEMPLATE_MAP = {
    PageFamily.SEARCH:  _render_search_html,
    PageFamily.RESULTS: _render_results_html,
    PageFamily.DETAIL:  _render_detail_html,
    PageFamily.DOCUMENT: _render_document_html,
    PageFamily.REPORT:  _render_report_html,
    PageFamily.FORM:    _render_form_html,
    PageFamily.RANKING: _render_ranking_html,
    PageFamily.MODAL:   _render_modal_html,
}


class TemplateRenderer:
    """
    第②层：ScreenIR→HTML 模板渲染器。
    
    根据 EnvState.page_family 选择对应模板，
    填充产品数据和状态信息，生成完整 HTML，
    再通过 Playwright 渲染为截图（基底图）。

    基底图交给 DeltaWorldModel 叠加受控视觉差分。
    """

    def __init__(self, viewport_w: int = 1280, viewport_h: int = 720, style_library_dir: str | Path | None = None):
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.style_library_dir = Path(style_library_dir or Path(__file__).parent.parent / "data" / "screen_ir")
        self.style_library = self._load_style_library()

    def render_html(self, state: EnvState, products: dict, style_bundle: StyleBundle | None = None) -> str:
        """根据当前状态生成 HTML 字符串"""
        style_bundle = _normalize_style(style_bundle)
        # 如果有 modal，先用 MODAL 模板
        if state.active_modal:
            renderer = _TEMPLATE_MAP[PageFamily.MODAL]
        else:
            renderer = _TEMPLATE_MAP.get(state.page_family, _render_search_html)
        return renderer(state, products, style_bundle)

    def pick_style_bundle(self, style_key: str = "", page_family: PageFamily | None = None) -> StyleBundle:
        candidates = self.style_library
        if not candidates:
            return _default_style_bundle()
        if page_family is not None:
            family_candidates = [item for item in candidates if item.layout_mode == page_family.value]
            if family_candidates:
                candidates = family_candidates
        if not style_key:
            return candidates[0]
        digest = hashlib.md5(style_key.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(candidates)
        return candidates[idx]

    def _load_style_library(self) -> list[StyleBundle]:
        bundles: list[StyleBundle] = []
        if not self.style_library_dir.exists():
            return [_default_style_bundle()]

        for json_path in sorted(self.style_library_dir.glob("*/screen_ir.json")):
            if json_path.parent.name.startswith("synth_"):
                continue
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                style = StyleBundle(**payload.get("style", {}))
                style.style_id = style.style_id or payload.get("page_id", json_path.parent.name)
                style.source_url = style.source_url or payload.get("source_url", "")
                style.layout_mode = style.layout_mode or payload.get("page_family", "")
                bundles.append(_normalize_style(style))
            except Exception:
                continue

        if not bundles:
            bundles.append(_default_style_bundle())
        return bundles

    async def render_screenshot(self, state: EnvState, products: dict,
                                 output_path: str | Path | None = None,
                                 style_bundle: StyleBundle | None = None) -> bytes:
        """
        完整渲染管线：State → HTML → Playwright → Screenshot (PNG bytes)
        """
        html = self.render_html(state, products, style_bundle=style_bundle)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": self.viewport_w, "height": self.viewport_h},
                device_scale_factor=1
            )
            page = await context.new_page()
            await page.set_content(html, wait_until="domcontentloaded")
            await page.wait_for_timeout(500)

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(output_path))

            screenshot_bytes = await page.screenshot()
            await browser.close()

        return screenshot_bytes

    async def render_zoom_screenshot(
        self,
        state: EnvState,
        products: dict,
        bbox: BBox | dict[str, float],
        output_path: str | Path | None = None,
        padding: int = 24,
        device_scale_factor: float = 2.0,
        style_bundle: StyleBundle | None = None,
    ) -> bytes:
        """
        基于源码重渲染局部区域，而不是对已有位图做放大。
        """
        html = self.render_html(state, products, style_bundle=style_bundle)
        clip = self._build_clip_box(bbox, padding)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": self.viewport_w, "height": self.viewport_h},
                device_scale_factor=device_scale_factor,
            )
            page = await context.new_page()
            await page.set_content(html, wait_until="domcontentloaded")
            await page.wait_for_timeout(500)

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(output_path), clip=clip)

            screenshot_bytes = await page.screenshot(clip=clip)
            await browser.close()

        return screenshot_bytes

    async def render_to_screen_ir(
        self,
        state: EnvState,
        products: dict,
        page_id: str,
        output_dir: Path,
        style_bundle: StyleBundle | None = None,
    ) -> ScreenIR:
        """
        完整管线：State → HTML → Screenshot → ScreenIR。
        产出标准 ScreenIR，与 Harvester 采集的格式对齐。
        """
        from datetime import datetime, timezone

        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = output_dir / "screenshot.png"
        style_bundle = _normalize_style(style_bundle or self.pick_style_bundle(page_id, state.page_family))

        # 渲染截图
        html = self.render_html(state, products, style_bundle=style_bundle)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": self.viewport_w, "height": self.viewport_h},
                device_scale_factor=1
            )
            page = await context.new_page()
            await page.set_content(html, wait_until="domcontentloaded")
            await page.wait_for_timeout(500)
            await page.screenshot(path=str(screenshot_path))

            # 提取可交互元素
            raw_elements = await page.evaluate("""
                () => {
                    const selectors = 'a, button, input, select, [id]';
                    const els = document.querySelectorAll(selectors);
                    const results = [];
                    let id = 0;
                    for (const el of els) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width < 5 || rect.height < 5) continue;
                        results.push({
                            element_id: id++,
                            tag: el.tagName.toLowerCase(),
                            text: (el.innerText || el.value || el.placeholder || '').substring(0, 80).trim(),
                            x: rect.x, y: rect.y, w: rect.width, h: rect.height,
                            dom_id: el.id || '',
                        });
                    }
                    return results;
                }
            """)

            await browser.close()

        # 构建 ScreenIR
        interactables = [
            Interactable(
                element_id=e["element_id"],
                tag=e["tag"],
                text=e["text"],
                bbox=BBox(x=e["x"], y=e["y"], width=e["w"], height=e["h"]),
                is_visible=True,
                semantic_type=_infer_semantic_type(e["dom_id"], e["tag"]),
            )
            for e in raw_elements
        ]

        ir = ScreenIR(
            page_id=page_id,
            source_url=f"template://{style_bundle.style_id or 'shopsim'}",
            page_family=state.page_family if not state.active_modal else PageFamily.MODAL,
            title="ResearchAgent - " + str(state.current_page),
            screenshot_path=str(screenshot_path),
            viewport_width=self.viewport_w,
            viewport_height=self.viewport_h,
            interactables=interactables,
            style=style_bundle,
            crawl_timestamp=datetime.now(timezone.utc).isoformat()
        )
        ir.generate_ui_tokens()
        ir.save(output_dir / "screen_ir.json")
        return ir

    def _build_clip_box(self, bbox: BBox | dict[str, float], padding: int) -> dict[str, float]:
        if isinstance(bbox, BBox):
            x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        else:
            x = float(bbox["x"])
            y = float(bbox["y"])
            w = float(bbox["width"])
            h = float(bbox["height"])

        left = max(0.0, x - padding)
        top = max(0.0, y - padding)
        right = min(float(self.viewport_w), x + w + padding)
        bottom = min(float(self.viewport_h), y + h + padding)

        return {
            "x": left,
            "y": top,
            "width": max(1.0, right - left),
            "height": max(1.0, bottom - top),
        }


def _infer_semantic_type(dom_id: str, tag: str) -> str:
    """根据 DOM id 和 tag 推断语义类型。v2 研究 UI 元素优先判断。"""
    # ── v2 Research UI (高优先级) ──
    if "cite" in dom_id:
        return ElementSemanticType.CITE_BUTTON.value
    if "submit-report" in dom_id:
        return ElementSemanticType.SUBMIT_BUTTON.value
    if dom_id.startswith("result-"):
        return ElementSemanticType.RESULT_ITEM.value
    if "refine" in dom_id:
        return ElementSemanticType.REFINE_BUTTON.value
    if "doc-body" in dom_id:
        return ElementSemanticType.DOC_BODY.value
    if "doc-image" in dom_id:
        return ElementSemanticType.DOC_IMAGE.value
    if "evidence" in dom_id:
        return ElementSemanticType.EVIDENCE_PANEL.value

    # ── 通用 UI ──
    if "search" in dom_id and tag == "input":
        return ElementSemanticType.SEARCH_INPUT.value
    if "search" in dom_id and tag == "button":
        return ElementSemanticType.SEARCH_BUTTON.value
    if "back" in dom_id:
        return ElementSemanticType.BACK_BUTTON.value
    if dom_id.startswith("real-image-") or dom_id.startswith("thumb-"):
        return ElementSemanticType.REAL_IMAGE.value

    # ── 通用 UI 控件 (低优先级) ──
    if "sort" in dom_id or "dropdown" in dom_id:
        return ElementSemanticType.DROPDOWN.value
    if "modal" in dom_id:
        return ElementSemanticType.MODAL_TRIGGER.value
    if "submit" in dom_id:
        return ElementSemanticType.FORM_SUBMIT.value
    if dom_id.startswith("price-") or dom_id == "detail-price":
        return ElementSemanticType.PRICE_TAG.value

    # ── 通用 fallback ──
    if tag == "input":
        return ElementSemanticType.FORM_FIELD.value
    if tag == "a":
        return ElementSemanticType.GENERIC_LINK.value
    if tag == "button":
        return ElementSemanticType.GENERIC_BUTTON.value
    return ElementSemanticType.UNKNOWN.value
