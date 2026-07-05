"""
generate_diverse_training_data.py
==================================
集成样式仓库 + TemplateRenderer，批量生成多样化的高质量训练数据。

每个样式 x 每种页面族 x 每种状态 = 一组 (screenshot + ScreenIR)，
直接可用于 agent loop / SFT / GRPO 训练管线。
"""

import asyncio
import json
import sys
import random
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from searcheyes.template_renderer import TemplateRenderer
from searcheyes.transition_engine import EnvState, _FALLBACK_PRODUCTS
from searcheyes.screen_ir import PageFamily, StyleBundle
from searcheyes.style_repository import StyleRepository


PRODUCTS_DICT = {p["id"]: p for p in _FALLBACK_PRODUCTS}


# ── 状态模板：覆盖所有页面族 ──────────────────────────────

def build_state_matrix() -> list[tuple[str, EnvState]]:
    """构建覆盖所有页面族的状态矩阵"""
    states = []

    # Search
    states.append(("search", EnvState(
        current_page="search",
        page_family=PageFamily.SEARCH,
    )))

    # Results
    states.append(("results", EnvState(
        current_page="results",
        page_family=PageFamily.RESULTS,
    )))

    # Detail (不同商品)
    for pid in [1, 2, 3, 4]:
        states.append((f"detail_{pid}", EnvState(
            current_page=f"detail_{pid}",
            page_family=PageFamily.DETAIL,
            selected_product_id=pid,
        )))

    # Form
    states.append(("form", EnvState(
        current_page="form",
        page_family=PageFamily.FORM,
        selected_product_id=1,
    )))

    # Ranking
    states.append(("ranking", EnvState(
        current_page="ranking",
        page_family=PageFamily.RANKING,
    )))

    # Modal (在 detail 页上)
    states.append(("modal_confirm", EnvState(
        current_page="detail_1",
        page_family=PageFamily.DETAIL,
        selected_product_id=1,
        active_modal="confirm",
    )))

    return states


async def generate_training_batch(
    num_styles: int = 10,
    output_root: str = "output/diverse_training_data",
    min_quality: float = 0.5,
):
    """
    批量生成训练数据。

    Args:
        num_styles: 使用多少个样式
        output_root: 输出根目录
        min_quality: 最低样式质量分
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"batch_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载样式仓库
    repo = StyleRepository("data/style_repository")
    stats = repo.get_statistics()
    print(f"Style Repository: {stats['total_styles']} styles, avg quality {stats['average_quality']:.2f}")

    # 获取高质量样式
    style_ids = repo.get_high_quality_styles(min_score=min_quality, limit=num_styles)
    if not style_ids:
        print("No styles found in repository, using default style")
        style_ids = ["__default__"]

    print(f"Selected {len(style_ids)} styles")

    # 构建状态矩阵
    state_matrix = build_state_matrix()
    print(f"State matrix: {len(state_matrix)} states")
    print(f"Total combinations: {len(style_ids)} x {len(state_matrix)} = {len(style_ids) * len(state_matrix)}")

    renderer = TemplateRenderer()
    results = []
    total = len(style_ids) * len(state_matrix)
    done = 0

    for style_id in style_ids:
        # 加载样式
        if style_id == "__default__":
            style = None
            style_name = "default"
        else:
            style = repo.load_style(style_id)
            style_name = style_id
            if not style:
                print(f"  Skip {style_id}: load failed")
                continue

        for state_name, state in state_matrix:
            page_id = f"{style_name}__{state_name}"
            ir_dir = output_dir / page_id

            try:
                ir = await renderer.render_to_screen_ir(
                    state=state,
                    products=PRODUCTS_DICT,
                    page_id=page_id,
                    output_dir=ir_dir,
                    style_bundle=style,
                )

                results.append({
                    "page_id": page_id,
                    "style_id": style_name,
                    "state_name": state_name,
                    "page_family": str(state.page_family.value),
                    "interactable_count": len(ir.interactables),
                    "screenshot": str(ir_dir / "screenshot.png"),
                    "screen_ir": str(ir_dir / "screen_ir.json"),
                })
                done += 1

                if done % 10 == 0 or done == total:
                    print(f"  [{done}/{total}] {page_id} -> {len(ir.interactables)} elements")

            except Exception as e:
                print(f"  FAILED {page_id}: {e}")
                done += 1

    # 保存批次摘要
    summary = {
        "batch_id": f"batch_{ts}",
        "timestamp": ts,
        "num_styles": len(style_ids),
        "num_states": len(state_matrix),
        "total_generated": len(results),
        "total_attempted": total,
        "output_dir": str(output_dir),
        "page_family_distribution": {},
        "results": results,
    }

    # 统计页面族分布
    for r in results:
        fam = r["page_family"]
        summary["page_family_distribution"][fam] = summary["page_family_distribution"].get(fam, 0) + 1

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 生成 Gallery HTML
    gallery = _build_gallery(results, output_dir)
    gallery_path = output_dir / "gallery.html"
    gallery_path.write_text(gallery, encoding="utf-8")

    print(f"\nDone: {len(results)}/{total} pages generated")
    print(f"Output: {output_dir}")
    print(f"Gallery: {gallery_path}")
    print(f"Summary: {summary_path}")

    return summary


def _build_gallery(results: list[dict], output_dir: Path) -> str:
    """生成 Gallery HTML"""
    cards = ""
    for i, r in enumerate(results):
        # 相对路径
        png_rel = Path(r["screenshot"]).relative_to(output_dir)
        cards += f"""
        <div class="card">
            <img src="{png_rel}" loading="lazy" />
            <div class="info">
                <div class="title">{r['page_id']}</div>
                <div class="meta">{r['page_family']} | {r['interactable_count']} elements</div>
            </div>
        </div>"""

    # 按页面族统计
    fam_counts = {}
    for r in results:
        fam = r["page_family"]
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    stats_html = " | ".join(f"{k}: {v}" for k, v in sorted(fam_counts.items()))

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Training Data Gallery</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 32px; }}
h1 {{ text-align: center; margin-bottom: 8px; }}
.stats {{ text-align: center; color: #94a3b8; margin-bottom: 32px; font-size: 14px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
.card {{ background: #1e293b; border-radius: 12px; overflow: hidden; }}
.card img {{ width: 100%; display: block; height: 200px; object-fit: cover; object-position: top; }}
.info {{ padding: 12px 16px; }}
.title {{ font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.meta {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
</style></head><body>
<h1>Training Data Gallery ({len(results)} pages)</h1>
<div class="stats">{stats_html}</div>
<div class="grid">{cards}</div>
</body></html>"""


async def main():
    print("=" * 60)
    print("Diverse Training Data Generator")
    print("StyleRepository + TemplateRenderer")
    print("=" * 60)

    summary = await generate_training_batch(
        num_styles=10,
        min_quality=0.5,
    )

    print(f"\nPage family distribution:")
    for fam, cnt in sorted(summary["page_family_distribution"].items()):
        print(f"  {fam}: {cnt}")


if __name__ == "__main__":
    asyncio.run(main())
