"""
run_component_pipeline.py - Component Extraction & Synthesis Pipeline
======================================================================
Step 1: Extract UI components from real websites
Step 2: Build component library index
Step 3: Generate synthetic pages by mixing components
Step 4: Render screenshots for visual inspection
"""

import asyncio
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from searcheyes.component_extractor import ComponentExtractor, batch_extract
from searcheyes.component_library import ComponentLibrary, SyntheticPageAssembler


# ── 可安全访问的目标网站 ──────────────────────────────────

SAFE_TARGETS = [
    ("https://example.com/", "example_basic"),
    ("https://httpbin.org/forms/post", "httpbin_form"),
    ("https://news.ycombinator.com/", "hn_results"),
    ("https://en.wikipedia.org/wiki/Artificial_intelligence", "wiki_ai"),
]


async def step1_extract(targets=None):
    """Step 1: Extract components from real websites"""
    targets = targets or SAFE_TARGETS
    print("=" * 60)
    print("STEP 1: Extract UI Components")
    print("=" * 60)

    results = await batch_extract(targets, output_dir="data/component_library")
    return results


def step2_build_library():
    """Step 2: Build and inspect component library"""
    print("\n" + "=" * 60)
    print("STEP 2: Build Component Library")
    print("=" * 60)

    library = ComponentLibrary("data/component_library")
    stats = library.statistics()

    print(f"\nLibrary Stats:")
    print(f"  Total components: {stats['total_components']}")
    print(f"  Source pages: {stats['source_pages']}")
    print(f"  Average quality: {stats['average_quality']:.3f}")
    print(f"  High quality (>=0.7): {stats['high_quality_count']}")
    print(f"\nBy type:")
    for t, n in stats["types"].items():
        print(f"  {t}: {n}")

    return library


async def step3_generate_synthetic(library, num_pages=5):
    """Step 3: Generate synthetic pages"""
    print("\n" + "=" * 60)
    print("STEP 3: Generate Synthetic Pages")
    print("=" * 60)

    assembler = SyntheticPageAssembler(library)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/synthetic_pages_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    color_schemes = [
        {"primary": "#4f46e5", "secondary": "#06b6d4", "bg": "#ffffff",
         "text": "#1e293b", "muted": "#64748b", "border": "#e2e8f0",
         "nav_bg": "#1e293b", "nav_text": "#f8fafc"},
        {"primary": "#dc2626", "secondary": "#f59e0b", "bg": "#fefce8",
         "text": "#1c1917", "muted": "#78716c", "border": "#d6d3d1",
         "nav_bg": "#7c2d12", "nav_text": "#fff7ed"},
        {"primary": "#059669", "secondary": "#8b5cf6", "bg": "#f0fdf4",
         "text": "#14532d", "muted": "#6b7280", "border": "#d1d5db",
         "nav_bg": "#064e3b", "nav_text": "#ecfdf5"},
        {"primary": "#7c3aed", "secondary": "#ec4899", "bg": "#faf5ff",
         "text": "#1e1b4b", "muted": "#6b7280", "border": "#e5e7eb",
         "nav_bg": "#312e81", "nav_text": "#eef2ff"},
        {"primary": "#0284c7", "secondary": "#f97316", "bg": "#f0f9ff",
         "text": "#0c4a6e", "muted": "#64748b", "border": "#bae6fd",
         "nav_bg": "#0c4a6e", "nav_text": "#e0f2fe"},
    ]

    families = ["search", "results", "detail", "form", "ranking"]
    pages = []

    for i in range(num_pages):
        family = families[i % len(families)]
        scheme = color_schemes[i % len(color_schemes)]

        spec = assembler.generate_spec(family)
        html = assembler.assemble_html(spec, color_scheme=scheme)

        # Save HTML
        html_path = output_dir / f"synth_{i:03d}_{family}.html"
        html_path.write_text(html, encoding="utf-8")

        comp_count = len(spec.main_content) + (1 if spec.navbar else 0) + (1 if spec.footer else 0)
        print(f"  [{i}] {family} -> {comp_count} components -> {html_path.name}")
        pages.append((spec, html, html_path))

    return pages


async def step4_render_screenshots(pages):
    """Step 4: Render screenshots for visual inspection"""
    print("\n" + "=" * 60)
    print("STEP 4: Render Screenshots")
    print("=" * 60)

    from playwright.async_api import async_playwright

    # Derive output_dir from the first page's html_path
    output_dir = pages[0][2].parent if pages else Path("output/synthetic_pages")
    screenshots = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            device_scale_factor=1,
        )

        for spec, html, html_path in pages:
            try:
                page = await context.new_page()
                await page.set_content(html, wait_until="domcontentloaded")
                await page.wait_for_timeout(500)

                png_path = html_path.with_suffix(".png")
                await page.screenshot(path=str(png_path))
                await page.close()

                screenshots.append(png_path)
                print(f"  -> {png_path.name}")
            except Exception as e:
                print(f"  -> FAILED: {e}")

        await browser.close()

    # Generate HTML gallery
    gallery = _build_gallery_html(pages, screenshots)
    gallery_path = output_dir / "gallery.html"
    gallery_path.write_text(gallery, encoding="utf-8")
    print(f"\nGallery: {gallery_path}")

    return screenshots


def _build_gallery_html(pages, screenshots):
    """Build an HTML gallery of synthetic pages"""
    cards = ""
    for i, ((spec, _, html_path), png_path) in enumerate(zip(pages, screenshots)):
        comp_count = len(spec.main_content) + (1 if spec.navbar else 0) + (1 if spec.footer else 0)
        cards += f"""
        <div style="break-inside:avoid;margin-bottom:24px;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
            <img src="{png_path.name}" style="width:100%;display:block;" />
            <div style="padding:16px;">
                <div style="font-weight:600;font-size:16px;">#{i} {spec.page_family}</div>
                <div style="color:#64748b;font-size:13px;margin-top:4px;">{comp_count} components | <a href="{html_path.name}">HTML</a></div>
            </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Synthetic Pages Gallery</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #f1f5f9; padding: 32px; }}
h1 {{ text-align: center; margin-bottom: 32px; color: #1e293b; }}
.grid {{ columns: 2; column-gap: 24px; max-width: 1200px; margin: 0 auto; }}
</style></head><body>
<h1>Synthetic Pages Gallery ({len(pages)} pages)</h1>
<div class="grid">{cards}</div>
</body></html>"""


async def main():
    print("Component Extraction & Synthesis Pipeline")
    print("=" * 60)

    # Step 1
    results = await step1_extract()

    # Step 2
    library = step2_build_library()

    if library.statistics()["total_components"] == 0:
        print("\nNo components extracted. Check network connectivity.")
        return

    # Step 3
    pages = await step3_generate_synthetic(library, num_pages=5)

    # Step 4
    screenshots = await step4_render_screenshots(pages)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Components: {library.statistics()['total_components']}")
    print(f"  Synthetic pages: {len(pages)}")
    print(f"  Screenshots: {len(screenshots)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
