"""
harvester.py — Frontend Harvest 管线
=======================================
从真实 URL 或本地 HTML 采集前端资产，生成标准 ScreenIR。
核心原则：只提取"视觉皮囊"和"页面语法"，不搬运后端代码。
"""

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from playwright.async_api import async_playwright

from searcheyes.screen_ir import (
    ScreenIR, PageFamily, Interactable, BBox, StyleBundle
)

DATA_DIR = Path(__file__).parent.parent / "data" / "screen_ir"


def _classify_page(title: str, url: str, dom_hints: dict) -> PageFamily:
    """基于启发式规则推断页面族。URL 启发式优先于 DOM 检测。"""
    url_lower = url.lower()
    title_lower = title.lower()

    # --- 优先级 1: URL 启发式 (最可靠) ---
    if any(k in url_lower for k in ["login", "register", "signup", "form", "checkout"]):
        return PageFamily.FORM
    if any(k in url_lower for k in ["search", "query", "q="]):
        return PageFamily.SEARCH
    if any(k in url_lower for k in ["result", "list", "browse", "trending"]):
        return PageFamily.RESULTS
    if any(k in url_lower for k in ["detail", "product", "item", "article", "wiki/"]):
        return PageFamily.DETAIL
    if any(k in url_lower for k in ["rank", "top", "chart", "leaderboard", "tiobe", "index"]):
        return PageFamily.RANKING

    # --- 优先级 2: DOM 结构启发式 ---
    if dom_hints.get("has_search_input"):
        return PageFamily.SEARCH
    if dom_hints.get("list_item_count", 0) > 20:
        return PageFamily.RESULTS
    # modal 仅作为 fallback，因为大多数现代网站都有 cookie/overlay
    if dom_hints.get("has_modal") and dom_hints.get("list_item_count", 0) < 5:
        return PageFamily.MODAL

    return PageFamily.DETAIL  # 默认


async def harvest_page(url: str, page_id: str, viewport_w: int = 1280, viewport_h: int = 720) -> ScreenIR:
    """
    采集单个页面的完整 ScreenIR。

    Returns:
        ScreenIR 实例，同时截图已保存到 data/screen_ir/{page_id}/
    """
    out_dir = DATA_DIR / page_id
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = out_dir / "screenshot.png"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": viewport_w, "height": viewport_h},
            device_scale_factor=1,
            locale="zh-CN"
        )
        page = await context.new_page()

        try:
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)
        except Exception as e:
            print(f"  ⚠️ 页面加载异常: {e}")
            await browser.close()
            raise

        # --- 截图 ---
        await page.screenshot(path=str(screenshot_path))

        # --- 提取标题 ---
        title = await page.title() or ""

        # --- 提取所有可交互元素 ---
        raw_elements = await page.evaluate("""
            () => {
                const selectors = 'a, button, input, select, textarea, [role="button"], [onclick]';
                const els = document.querySelectorAll(selectors);
                const results = [];
                let id = 0;
                for (const el of els) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width < 5 || rect.height < 5) continue;
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;

                    // 判断父级区域
                    let section = 'main';
                    let parent = el;
                    while (parent) {
                        const tag = parent.tagName?.toLowerCase();
                        if (['header', 'nav'].includes(tag)) { section = 'header'; break; }
                        if (tag === 'footer') { section = 'footer'; break; }
                        if (tag === 'aside') { section = 'sidebar'; break; }
                        if (parent.getAttribute?.('role') === 'dialog') { section = 'modal'; break; }
                        parent = parent.parentElement;
                    }

                    results.push({
                        element_id: id++,
                        tag: el.tagName.toLowerCase(),
                        text: (el.innerText || el.value || el.placeholder || el.alt || '').substring(0, 100).trim(),
                        x: rect.x, y: rect.y, w: rect.width, h: rect.height,
                        role: el.getAttribute('role') || '',
                        href: el.href || '',
                        input_type: el.type || '',
                        classes: Array.from(el.classList).slice(0, 5),
                        parent_section: section
                    });
                }
                return results;
            }
        """)

        # --- DOM 结构分析 ---
        dom_hints = await page.evaluate("""
            () => {
                return {
                    has_search_input: !!document.querySelector('input[type="search"], input[name*="search"], input[placeholder*="搜索"], input[placeholder*="Search"]'),
                    has_modal: !!document.querySelector('[role="dialog"], .modal, .popup, .overlay'),
                    list_item_count: document.querySelectorAll('li, .card, .item, article, .result').length,
                    total_elements: document.querySelectorAll('*').length
                };
            }
        """)

        # --- 提取视觉风格 ---
        style_data = await page.evaluate("""
            () => {
                const isMeaningful = (value) => {
                    if (!value) return false;
                    const lower = String(value).toLowerCase();
                    return !lower.includes('rgba(0, 0, 0, 0)') &&
                        lower !== 'transparent' &&
                        lower !== 'initial' &&
                        lower !== 'inherit';
                };

                const firstStyleValue = (selectors, getter) => {
                    for (const selector of selectors) {
                        const node = document.querySelector(selector);
                        if (!node) continue;
                        const cs = window.getComputedStyle(node);
                        const value = getter(cs);
                        if (isMeaningful(value)) {
                            return value;
                        }
                    }
                    return '';
                };

                const parseRadius = (value, fallback) => {
                    const n = parseFloat(value);
                    return Number.isFinite(n) ? n : fallback;
                };

                const body = document.body;
                const cs = window.getComputedStyle(body);
                const nav = document.querySelector('nav, header');
                const sidebar = document.querySelector('aside, .sidebar');
                const footer = document.querySelector('footer');
                const navStyle = nav ? window.getComputedStyle(nav) : null;
                const primaryColor = firstStyleValue(
                    ['button', '.btn', '[role="button"]', 'a[href]', 'input[type="submit"]'],
                    (style) => style.backgroundColor || style.color
                );
                const secondaryColor = firstStyleValue(
                    ['.tag', '.badge', '.chip', '.pill', 'select'],
                    (style) => style.backgroundColor || style.borderColor
                );
                const accentColor = firstStyleValue(
                    ['.price', '.danger', '.warning', '.accent', 'strong', 'mark'],
                    (style) => style.color || style.backgroundColor
                );
                const surfaceColor = firstStyleValue(
                    ['main', 'section', 'article', '.card', '.container'],
                    (style) => style.backgroundColor
                );
                const borderColor = firstStyleValue(
                    ['input', 'button', '.card', 'select'],
                    (style) => style.borderColor
                );
                const buttonRadius = parseRadius(
                    firstStyleValue(['button', '.btn', 'input[type="submit"]'], (style) => style.borderRadius),
                    8
                );
                const cardRadius = parseRadius(
                    firstStyleValue(['.card', 'article', 'section', 'main'], (style) => style.borderRadius),
                    12
                );
                const shadowStrength = firstStyleValue(
                    ['.card', 'article', 'section', 'main'],
                    (style) => style.boxShadow
                ) ? 0.12 : 0.04;
                const basePadding = parseFloat(
                    firstStyleValue(['main', '.container', '.card', 'section'], (style) => style.padding)
                );
                const spacingScale = Number.isFinite(basePadding)
                    ? Math.min(1.35, Math.max(0.85, basePadding / 16))
                    : 1.0;

                return {
                    style_id: location.hostname.replaceAll('.', '_') || document.title || 'harvested',
                    source_url: location.href,
                    primary_color: primaryColor,
                    secondary_color: secondaryColor,
                    accent_color: accentColor,
                    bg_color: cs.backgroundColor,
                    surface_color: surfaceColor || '#ffffff',
                    text_color: cs.color,
                    muted_text_color: firstStyleValue(['small', 'p', '.muted', '.secondary'], (style) => style.color),
                    border_color: borderColor,
                    font_family: cs.fontFamily.split(',')[0].trim().replace(/['"]/g, ''),
                    nav_bg_color: navStyle?.backgroundColor || '',
                    nav_text_color: navStyle?.color || '',
                    hero_gradient_from: primaryColor,
                    hero_gradient_to: secondaryColor || primaryColor,
                    card_radius: cardRadius,
                    button_radius: buttonRadius,
                    shadow_strength: shadowStrength,
                    spacing_scale: spacingScale,
                    density: spacingScale > 1.1 ? 'airy' : (spacingScale < 0.95 ? 'compact' : 'normal'),
                    has_navbar: !!nav,
                    has_sidebar: !!sidebar,
                    has_footer: !!footer,
                };
            }
        """)

        # --- 提取文本 hash (去重) ---
        page_text = await page.inner_text("body")
        text_hash = hashlib.md5(page_text.encode()).hexdigest()[:12]

        await browser.close()

    # --- 构建 ScreenIR ---
    interactables = [
        Interactable(
            element_id=e["element_id"],
            tag=e["tag"],
            text=e["text"],
            bbox=BBox(x=e["x"], y=e["y"], width=e["w"], height=e["h"]),
            role=e["role"],
            href=e["href"],
            input_type=e["input_type"],
            css_classes=e["classes"],
            is_visible=True,
            parent_section=e["parent_section"]
        )
        for e in raw_elements
    ]

    page_family = _classify_page(title, url, dom_hints)

    ir = ScreenIR(
        page_id=page_id,
        source_url=url,
        page_family=page_family,
        title=title,
        screenshot_path=str(screenshot_path),
        viewport_width=viewport_w,
        viewport_height=viewport_h,
        interactables=interactables,
        style=StyleBundle(
            style_id=style_data.get("style_id", page_id),
            source_url=style_data.get("source_url", url),
            primary_color=style_data.get("primary_color", ""),
            secondary_color=style_data.get("secondary_color", ""),
            accent_color=style_data.get("accent_color", ""),
            bg_color=style_data.get("bg_color", ""),
            surface_color=style_data.get("surface_color", ""),
            text_color=style_data.get("text_color", ""),
            muted_text_color=style_data.get("muted_text_color", ""),
            border_color=style_data.get("border_color", ""),
            font_family=style_data.get("font_family", ""),
            nav_bg_color=style_data.get("nav_bg_color", ""),
            nav_text_color=style_data.get("nav_text_color", ""),
            hero_gradient_from=style_data.get("hero_gradient_from", ""),
            hero_gradient_to=style_data.get("hero_gradient_to", ""),
            card_radius=style_data.get("card_radius", 12.0),
            button_radius=style_data.get("button_radius", 8.0),
            shadow_strength=style_data.get("shadow_strength", 0.06),
            spacing_scale=style_data.get("spacing_scale", 1.0),
            density=style_data.get("density", "normal"),
            has_navbar=style_data.get("has_navbar", False),
            has_sidebar=style_data.get("has_sidebar", False),
            has_footer=style_data.get("has_footer", False),
        ),
        dom_element_count=dom_hints.get("total_elements", 0),
        text_content_hash=text_hash,
        crawl_timestamp=datetime.now(timezone.utc).isoformat()
    )

    # 持久化
    ir.save(out_dir / "screen_ir.json")

    print(f"  ✅ {ir.summary()}")
    return ir


# ================================================================
# 批量采集入口
# ================================================================

# 首批目标页面：覆盖 6 种 PageFamily
DEFAULT_TARGETS = [
    # SEARCH 类
    ("https://www.bing.com/", "bing_search"),
    ("https://search.yahoo.com/", "yahoo_search"),

    # RESULTS 类
    ("https://github.com/trending", "github_trending"),
    ("https://news.ycombinator.com/", "hn_results"),

    # DETAIL 类
    ("https://en.wikipedia.org/wiki/Artificial_intelligence", "wiki_ai"),
    ("https://en.wikipedia.org/wiki/Reinforcement_learning", "wiki_rl"),

    # FORM 类
    ("https://github.com/login", "github_login"),

    # RANKING 类
    ("https://www.tiobe.com/tiobe-index/", "tiobe_ranking"),

    # 通用/混合
    ("https://example.com/", "example_basic"),
    ("https://httpbin.org/forms/post", "httpbin_form"),
]


async def batch_harvest(targets: list[tuple[str, str]] | None = None):
    """批量采集多个页面"""
    targets = targets or DEFAULT_TARGETS
    print(f"{'='*60}")
    print(f"🌐 Frontend Harvest 批量采集 ({len(targets)} 目标)")
    print(f"{'='*60}")

    results = []
    for url, pid in targets:
        print(f"\n📥 [{pid}] {url}")
        try:
            ir = await harvest_page(url, pid)
            results.append(ir)
        except Exception as e:
            print(f"  ❌ 失败: {e}")

    print(f"\n{'='*60}")
    print(f"✅ 采集完成: {len(results)}/{len(targets)} 成功")

    # 打印页面族分布
    from collections import Counter
    dist = Counter(ir.page_family.value for ir in results)
    for fam, cnt in sorted(dist.items()):
        print(f"  {fam}: {cnt}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    asyncio.run(batch_harvest())
