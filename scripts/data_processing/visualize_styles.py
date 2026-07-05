"""
visualize_styles.py - Style Visualization Tool
===============================================
Generate HTML comparison report for manual quality inspection.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from searcheyes.style_repository import StyleRepository
from searcheyes.template_renderer import TemplateRenderer
from searcheyes.transition_engine import EnvState, PRODUCTS

# 构建 products dict（TemplateRenderer 需要 {id: product} 格式）
PRODUCTS_DICT = {p["id"]: p for p in PRODUCTS}


async def generate_style_comparison(
    repository: StyleRepository,
    style_ids: list[str],
    output_dir: Path,
    num_samples: int = 5
):
    """
    生成样式对比报告
    
    Args:
        repository: 样式仓库
        style_ids: 要对比的样式ID列表
        output_dir: 输出目录
        num_samples: 每个样式生成几个页面示例
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    renderer = TemplateRenderer()
    
    # 准备测试状态（不同页面类型）
    test_states = [
        # 搜索页
        EnvState(current_page="search", page_family="search"),
        # 结果页（4个产品）
        EnvState(
            current_page="results",
            page_family="results",
            selected_product_id=None,
            cart=[]
        ),
        # 详情页
        EnvState(
            current_page="detail_0",
            page_family="detail",
            selected_product_id=0,
            cart=[]
        ),
    ]
    
    # 为每个样式生成截图
    style_samples = {}
    
    for style_id in style_ids:
        print(f"\n📸 生成样式 [{style_id}] 的截图...")
        style = repository.load_style(style_id)
        if not style:
            print(f"  ⚠️ 样式 {style_id} 不存在")
            continue
        
        meta = repository.index.get(style_id)
        samples = []
        
        for idx, state in enumerate(test_states[:num_samples]):
            try:
                screenshot_path = output_dir / f"{style_id}_page{idx}.png"
                # 使用正确的 render_screenshot API
                await renderer.render_screenshot(
                    state, PRODUCTS_DICT,
                    output_path=screenshot_path,
                    style_bundle=style
                )
                
                samples.append({
                    "page_type": state.page_family,
                    "screenshot": screenshot_path.name,
                })
                print(f"  ✅ {state.page_family} 页面")
            except Exception as e:
                print(f"  ❌ {state.page_family} 页面失败: {e}")
        
        style_samples[style_id] = {
            "style": style,
            "meta": meta,
            "samples": samples,
        }
    
    # 生成HTML报告
    html_content = _generate_comparison_html(style_samples, repository)
    report_path = output_dir / "style_comparison.html"
    report_path.write_text(html_content, encoding='utf-8')
    
    print(f"\n✅ 对比报告已生成: {report_path}")
    return report_path


def _generate_comparison_html(style_samples: dict, repository: StyleRepository) -> str:
    """生成HTML对比报告"""
    
    # 统计信息
    stats = repository.get_statistics()
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>样式库可视化对比报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', 'PingFang SC', sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }}
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            font-size: 16px;
            opacity: 0.9;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .stat-card .label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 8px;
        }}
        .stat-card .value {{
            font-size: 28px;
            font-weight: 700;
            color: #667eea;
        }}
        .style-section {{
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }}
        .style-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 2px solid #ecf0f1;
        }}
        .style-title {{
            font-size: 24px;
            font-weight: 700;
            color: #2c3e50;
        }}
        .style-meta {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .meta-badge {{
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }}
        .badge-quality {{
            background: #d4edda;
            color: #155724;
        }}
        .badge-type {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        .badge-score {{
            background: #fff3cd;
            color: #856404;
        }}
        .color-palette {{
            display: flex;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .color-swatch {{
            text-align: center;
        }}
        .color-box {{
            width: 80px;
            height: 80px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            margin-bottom: 8px;
        }}
        .color-label {{
            font-size: 11px;
            color: #7f8c8d;
            font-weight: 600;
        }}
        .color-value {{
            font-size: 10px;
            color: #95a5a6;
            font-family: 'Courier New', monospace;
        }}
        .screenshots {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .screenshot-card {{
            border: 2px solid #ecf0f1;
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .screenshot-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }}
        .screenshot-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .screenshot-label {{
            padding: 12px;
            background: #f8f9fa;
            font-size: 13px;
            font-weight: 600;
            color: #495057;
            text-align: center;
        }}
        .quality-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .quality-high {{ background: #28a745; }}
        .quality-medium {{ background: #ffc107; }}
        .quality-low {{ background: #dc3545; }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 样式库可视化对比报告</h1>
        <div class="subtitle">Visual DreamGym Style Repository Comparison</div>
        <div class="subtitle">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="label">总样式数</div>
            <div class="value">{stats['total_styles']}</div>
        </div>
        <div class="stat-card">
            <div class="label">真实样式</div>
            <div class="value">{stats['real_styles']}</div>
        </div>
        <div class="stat-card">
            <div class="label">混合样式</div>
            <div class="value">{stats['mixed_styles']}</div>
        </div>
        <div class="stat-card">
            <div class="label">平均质量分</div>
            <div class="value">{stats['average_quality']:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">高质量样式</div>
            <div class="value">{stats['high_quality_count']}</div>
        </div>
    </div>
"""
    
    # 为每个样式生成详细卡片
    for style_id, data in style_samples.items():
        style = data['style']
        meta = data['meta']
        samples = data['samples']
        
        # 质量指示器
        if meta.quality_score >= 0.7:
            quality_class = "quality-high"
            quality_text = "优秀"
        elif meta.quality_score >= 0.5:
            quality_class = "quality-medium"
            quality_text = "良好"
        else:
            quality_class = "quality-low"
            quality_text = "需改进"
        
        html += f"""
    <div class="style-section">
        <div class="style-header">
            <div>
                <div class="style-title">{style_id}</div>
                <div style="margin-top: 8px; font-size: 13px; color: #7f8c8d;">
                    来源: {meta.source_urls[0] if meta.source_urls else 'N/A'}
                </div>
            </div>
            <div class="style-meta">
                <span class="meta-badge badge-quality">
                    <span class="quality-indicator {quality_class}"></span>
                    {quality_text}
                </span>
                <span class="meta-badge badge-type">{meta.source_type}</span>
                <span class="meta-badge badge-score">质量分: {meta.quality_score:.2f}</span>
                <span class="meta-badge badge-score">可读性: {meta.readability_score:.2f}</span>
            </div>
        </div>
        
        <div class="color-palette">
            <div class="color-swatch">
                <div class="color-box" style="background: {style.primary_color};"></div>
                <div class="color-label">主色</div>
                <div class="color-value">{style.primary_color}</div>
            </div>
            <div class="color-swatch">
                <div class="color-box" style="background: {style.secondary_color};"></div>
                <div class="color-label">次色</div>
                <div class="color-value">{style.secondary_color}</div>
            </div>
            <div class="color-swatch">
                <div class="color-box" style="background: {style.accent_color};"></div>
                <div class="color-label">强调色</div>
                <div class="color-value">{style.accent_color}</div>
            </div>
            <div class="color-swatch">
                <div class="color-box" style="background: {style.bg_color};"></div>
                <div class="color-label">背景色</div>
                <div class="color-value">{style.bg_color}</div>
            </div>
            <div class="color-swatch">
                <div class="color-box" style="background: {style.text_color}; border: 1px solid #ddd;"></div>
                <div class="color-label">文字色</div>
                <div class="color-value">{style.text_color}</div>
            </div>
        </div>
        
        <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; font-size: 13px;">
            <strong>样式参数:</strong> 
            字体: {style.font_family} | 
            圆角: {style.card_radius:.1f}px | 
            阴影: {style.shadow_strength:.2f} | 
            间距: {style.spacing_scale:.2f}x | 
            密度: {style.density}
        </div>
        
        <div class="screenshots">
"""
        
        for sample in samples:
            html += f"""
            <div class="screenshot-card">
                <img src="{sample['screenshot']}" alt="{sample['page_type']}">
                <div class="screenshot-label">{sample['page_type'].upper()} 页面</div>
            </div>
"""
        
        html += """
        </div>
    </div>
"""
    
    html += """
    <div class="footer">
        <p>Visual DreamGym Project - Style Repository Visualization</p>
        <p>Generated by style_repository.py</p>
    </div>
</body>
</html>
"""
    
    return html


async def main():
    """主函数：生成样式对比报告"""
    from pathlib import Path
    
    # 初始化仓库
    repo_dir = Path("data/style_repository")
    repository = StyleRepository(repo_dir)
    
    print("=" * 60)
    print("🎨 样式库可视化工具")
    print("=" * 60)
    
    # 显示统计信息
    stats = repository.get_statistics()
    print(f"\n📊 当前仓库状态:")
    print(f"  总样式数: {stats['total_styles']}")
    print(f"  真实样式: {stats['real_styles']}")
    print(f"  混合样式: {stats['mixed_styles']}")
    print(f"  平均质量: {stats['average_quality']:.2f}")
    print(f"  高质量样式: {stats['high_quality_count']}")
    
    if stats['total_styles'] == 0:
        print("\n⚠️ 仓库为空，请先运行 build_style_repository.py 采集样式")
        return
    
    # 选择要可视化的样式
    print(f"\n🎯 选择样式进行可视化...")
    
    # 获取高质量样式
    high_quality = repository.get_high_quality_styles(min_score=0.6, limit=10)
    
    if not high_quality:
        print("  ⚠️ 没有找到高质量样式，使用所有样式")
        style_ids = list(repository.index.keys())[:10]
    else:
        style_ids = high_quality
    
    print(f"  已选择 {len(style_ids)} 个样式")
    
    # 生成对比报告
    output_dir = Path("output/style_comparison")
    await generate_style_comparison(
        repository=repository,
        style_ids=style_ids,
        output_dir=output_dir,
        num_samples=3  # 每个样式生成3个页面示例
    )
    
    print(f"\n✅ 完成！请打开 {output_dir}/style_comparison.html 查看报告")


if __name__ == "__main__":
    asyncio.run(main())
