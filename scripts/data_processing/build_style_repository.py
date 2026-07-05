"""
build_style_repository.py - Build Style Repository
===================================================
Build style repository from existing screen_ir data and generate mixed styles.
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from searcheyes.screen_ir import ScreenIR
from searcheyes.style_repository import StyleRepository


async def build_repository_from_screen_ir(
    screen_ir_dir: Path,
    repository: StyleRepository,
    min_quality: float = 0.5
):
    """从screen_ir目录构建样式仓库"""
    
    screen_ir_dir = Path(screen_ir_dir)
    if not screen_ir_dir.exists():
        print(f"❌ 目录不存在: {screen_ir_dir}")
        return
    
    print(f"\n📥 扫描 screen_ir 目录: {screen_ir_dir}")
    
    # 查找所有screen_ir.json文件
    ir_files = list(screen_ir_dir.glob("*/screen_ir.json"))
    print(f"  找到 {len(ir_files)} 个 ScreenIR 文件")
    
    added_count = 0
    skipped_count = 0
    
    for ir_file in ir_files:
        try:
            # 加载ScreenIR
            screen_ir = ScreenIR.load(ir_file)
            
            # 跳过合成样式（synth_开头）
            if screen_ir.page_id.startswith("synth_"):
                skipped_count += 1
                continue
            
            # 检查样式完整性
            if not screen_ir.style.primary_color:
                print(f"  ⚠️ 跳过 {screen_ir.page_id}: 样式不完整")
                skipped_count += 1
                continue
            
            # 添加到仓库
            style_id = repository.add_real_style(
                style=screen_ir.style,
                source_url=screen_ir.source_url,
                page_family=screen_ir.page_family
            )
            
            meta = repository.index[style_id]
            
            # 只保留高质量样式
            if meta.quality_score >= min_quality:
                print(f"  ✅ {screen_ir.page_id} → {style_id} (质量: {meta.quality_score:.2f})")
                added_count += 1
            else:
                print(f"  ⚠️ {screen_ir.page_id} 质量不足 ({meta.quality_score:.2f}), 已跳过")
                # 从仓库中移除
                del repository.index[style_id]
                style_file = repository.real_styles_dir / f"{style_id}.json"
                if style_file.exists():
                    style_file.unlink()
                skipped_count += 1
                
        except Exception as e:
            print(f"  ❌ 处理 {ir_file.parent.name} 失败: {e}")
            skipped_count += 1
    
    repository.save_index()
    
    print(f"\n📊 导入完成:")
    print(f"  成功添加: {added_count}")
    print(f"  跳过: {skipped_count}")
    
    return added_count


async def main():
    """主函数"""
    print("=" * 60)
    print("🏗️  构建样式仓库")
    print("=" * 60)
    
    # 初始化仓库
    repo_dir = Path("data/style_repository")
    repository = StyleRepository(repo_dir)
    
    # 从screen_ir导入真实样式
    screen_ir_dir = Path("data/screen_ir")
    added = await build_repository_from_screen_ir(
        screen_ir_dir=screen_ir_dir,
        repository=repository,
        min_quality=0.5  # 最低质量阈值
    )
    
    if added == 0:
        print("\n⚠️ 没有成功导入任何样式")
        return
    
    # 显示统计
    stats = repository.get_statistics()
    print(f"\n📊 当前仓库状态:")
    print(f"  真实样式: {stats['real_styles']}")
    print(f"  平均质量: {stats['average_quality']:.2f}")
    print(f"  高质量样式 (≥0.7): {stats['high_quality_count']}")
    
    # 生成混合样式
    if stats['real_styles'] >= 2:
        print(f"\n🎨 开始生成混合样式...")
        
        # 根据真实样式数量决定生成数量
        num_mixed = min(100, stats['real_styles'] * 5)
        
        mixed_ids = repository.generate_mixed_styles(
            num_styles=num_mixed,
            min_quality=0.6,
            strategies=["balanced", "color_from_1", "layout_from_1"]
        )
        
        print(f"\n✅ 成功生成 {len(mixed_ids)} 个混合样式")
        
        # 更新统计
        stats = repository.get_statistics()
        print(f"\n📊 最终仓库状态:")
        print(f"  总样式数: {stats['total_styles']}")
        print(f"  真实样式: {stats['real_styles']}")
        print(f"  混合样式: {stats['mixed_styles']}")
        print(f"  平均质量: {stats['average_quality']:.2f}")
        print(f"  高质量样式: {stats['high_quality_count']}")
    else:
        print(f"\n⚠️ 真实样式不足2个，无法生成混合样式")
    
    print(f"\n✅ 样式仓库构建完成！")
    print(f"   位置: {repo_dir}")
    print(f"\n💡 下一步:")
    print(f"   1. 运行 python visualize_styles.py 查看样式对比")
    print(f"   2. 在训练中使用样式仓库提供多样化数据")


if __name__ == "__main__":
    asyncio.run(main())
