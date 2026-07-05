"""
style_repository.py — 样式仓库管理系统
==========================================
管理真实样式采集、混搭生成、质量评分和检索。
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from searcheyes.screen_ir import StyleBundle, PageFamily


@dataclass
class StyleMetadata:
    """样式元数据"""
    style_id: str
    source_type: str  # "real" | "mixed"
    source_urls: list[str]  # 来源URL
    page_family: str
    quality_score: float = 0.0
    visual_appeal: float = 0.0
    readability_score: float = 0.0
    created_at: str = ""
    parent_styles: list[str] = None  # 混搭时的父样式ID
    
    def __post_init__(self):
        if self.parent_styles is None:
            self.parent_styles = []


class StyleQualityScorer:
    """样式质量评分器"""
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """转换hex颜色到RGB"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def calculate_luminance(rgb: tuple[int, int, int]) -> float:
        """计算相对亮度（WCAG标准）"""
        r, g, b = [x / 255.0 for x in rgb]
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    @staticmethod
    def contrast_ratio(color1: str, color2: str) -> float:
        """计算对比度（WCAG标准）"""
        try:
            rgb1 = StyleQualityScorer.hex_to_rgb(color1)
            rgb2 = StyleQualityScorer.hex_to_rgb(color2)
            l1 = StyleQualityScorer.calculate_luminance(rgb1)
            l2 = StyleQualityScorer.calculate_luminance(rgb2)
            lighter = max(l1, l2)
            darker = min(l1, l2)
            return (lighter + 0.05) / (darker + 0.05)
        except:
            return 1.0
    
    def score_readability(self, style: StyleBundle) -> float:
        """评分可读性（0-1）"""
        score = 0.0
        
        # 文字与背景对比度（WCAG AA标准：4.5:1）
        text_bg_contrast = self.contrast_ratio(style.text_color, style.bg_color)
        if text_bg_contrast >= 7.0:  # AAA级
            score += 0.4
        elif text_bg_contrast >= 4.5:  # AA级
            score += 0.3
        elif text_bg_contrast >= 3.0:
            score += 0.1
        
        # 导航栏对比度
        nav_contrast = self.contrast_ratio(style.nav_text_color, style.nav_bg_color)
        if nav_contrast >= 4.5:
            score += 0.2
        elif nav_contrast >= 3.0:
            score += 0.1
        
        # 字体合理性
        if style.font_family and len(style.font_family) > 0:
            score += 0.2
        
        # 圆角合理性
        if 6 <= style.card_radius <= 24:
            score += 0.1
        
        # 阴影强度合理性
        if 0.02 <= style.shadow_strength <= 0.15:
            score += 0.1
        
        return min(1.0, score)
    
    def score_visual_appeal(self, style: StyleBundle) -> float:
        """评分视觉美观度（0-1）"""
        score = 0.0
        
        # 颜色和谐度（主色和次色不应过于相似或冲突）
        primary_rgb = self.hex_to_rgb(style.primary_color)
        secondary_rgb = self.hex_to_rgb(style.secondary_color)
        color_distance = sum((a - b) ** 2 for a, b in zip(primary_rgb, secondary_rgb)) ** 0.5
        
        if 50 < color_distance < 200:  # 适度差异
            score += 0.3
        elif color_distance >= 200:  # 差异较大也可接受
            score += 0.2
        
        # 布局完整性
        if style.has_navbar:
            score += 0.15
        if style.has_footer:
            score += 0.15
        
        # 间距合理性
        if 0.9 <= style.spacing_scale <= 1.2:
            score += 0.2
        
        # 密度合理性
        if style.density in ["normal", "comfortable"]:
            score += 0.2
        
        return min(1.0, score)
    
    def score_overall(self, style: StyleBundle) -> float:
        """综合评分"""
        readability = self.score_readability(style)
        visual_appeal = self.score_visual_appeal(style)
        
        # 加权平均（可读性更重要）
        overall = readability * 0.6 + visual_appeal * 0.4
        return overall


class StyleMixer:
    """样式混搭器"""
    
    def __init__(self, scorer: StyleQualityScorer):
        self.scorer = scorer
    
    @staticmethod
    def interpolate_color(color1: str, color2: str, ratio: float) -> str:
        """插值两个颜色"""
        try:
            rgb1 = StyleQualityScorer.hex_to_rgb(color1)
            rgb2 = StyleQualityScorer.hex_to_rgb(color2)
            
            r = int(rgb1[0] * ratio + rgb2[0] * (1 - ratio))
            g = int(rgb1[1] * ratio + rgb2[1] * (1 - ratio))
            b = int(rgb1[2] * ratio + rgb2[2] * (1 - ratio))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return color1
    
    def blend_styles(
        self,
        style1: StyleBundle,
        style2: StyleBundle,
        ratio: float = 0.5,
        strategy: str = "balanced"
    ) -> StyleBundle:
        """
        混合两个样式
        
        Args:
            style1: 第一个样式
            style2: 第二个样式
            ratio: 混合比例（0-1，越接近1越像style1）
            strategy: 混合策略
                - "balanced": 平衡混合
                - "color_from_1": 颜色来自style1，布局来自style2
                - "layout_from_1": 布局来自style1，颜色来自style2
        """
        if strategy == "color_from_1":
            # 颜色来自style1，布局参数来自style2
            return StyleBundle(
                style_id=f"mix_{style1.style_id}_{style2.style_id}_c1",
                source_url=f"mixed:{style1.source_url}+{style2.source_url}",
                primary_color=style1.primary_color,
                secondary_color=style1.secondary_color,
                accent_color=style1.accent_color,
                bg_color=style1.bg_color,
                surface_color=style1.surface_color,
                text_color=style1.text_color,
                muted_text_color=style1.muted_text_color,
                border_color=style1.border_color,
                nav_bg_color=style1.nav_bg_color,
                nav_text_color=style1.nav_text_color,
                hero_gradient_from=style1.hero_gradient_from,
                hero_gradient_to=style1.hero_gradient_to,
                # 布局来自style2
                font_family=style2.font_family,
                card_radius=style2.card_radius,
                button_radius=style2.button_radius,
                shadow_strength=style2.shadow_strength,
                spacing_scale=style2.spacing_scale,
                density=style2.density,
                has_navbar=style2.has_navbar,
                has_sidebar=style2.has_sidebar,
                has_footer=style2.has_footer,
            )
        
        elif strategy == "layout_from_1":
            # 布局来自style1，颜色来自style2
            return StyleBundle(
                style_id=f"mix_{style1.style_id}_{style2.style_id}_l1",
                source_url=f"mixed:{style1.source_url}+{style2.source_url}",
                # 颜色来自style2
                primary_color=style2.primary_color,
                secondary_color=style2.secondary_color,
                accent_color=style2.accent_color,
                bg_color=style2.bg_color,
                surface_color=style2.surface_color,
                text_color=style2.text_color,
                muted_text_color=style2.muted_text_color,
                border_color=style2.border_color,
                nav_bg_color=style2.nav_bg_color,
                nav_text_color=style2.nav_text_color,
                hero_gradient_from=style2.hero_gradient_from,
                hero_gradient_to=style2.hero_gradient_to,
                # 布局来自style1
                font_family=style1.font_family,
                card_radius=style1.card_radius,
                button_radius=style1.button_radius,
                shadow_strength=style1.shadow_strength,
                spacing_scale=style1.spacing_scale,
                density=style1.density,
                has_navbar=style1.has_navbar,
                has_sidebar=style1.has_sidebar,
                has_footer=style1.has_footer,
            )
        
        else:  # balanced
            # 平衡混合
            return StyleBundle(
                style_id=f"mix_{style1.style_id}_{style2.style_id}_bal",
                source_url=f"mixed:{style1.source_url}+{style2.source_url}",
                primary_color=self.interpolate_color(style1.primary_color, style2.primary_color, ratio),
                secondary_color=self.interpolate_color(style1.secondary_color, style2.secondary_color, ratio),
                accent_color=self.interpolate_color(style1.accent_color, style2.accent_color, ratio),
                bg_color=self.interpolate_color(style1.bg_color, style2.bg_color, ratio),
                surface_color=self.interpolate_color(style1.surface_color, style2.surface_color, ratio),
                text_color=self.interpolate_color(style1.text_color, style2.text_color, ratio),
                muted_text_color=self.interpolate_color(style1.muted_text_color, style2.muted_text_color, ratio),
                border_color=self.interpolate_color(style1.border_color, style2.border_color, ratio),
                nav_bg_color=self.interpolate_color(style1.nav_bg_color, style2.nav_bg_color, ratio),
                nav_text_color=self.interpolate_color(style1.nav_text_color, style2.nav_text_color, ratio),
                hero_gradient_from=self.interpolate_color(style1.hero_gradient_from, style2.hero_gradient_from, ratio),
                hero_gradient_to=self.interpolate_color(style1.hero_gradient_to, style2.hero_gradient_to, ratio),
                font_family=style1.font_family if ratio > 0.5 else style2.font_family,
                card_radius=style1.card_radius * ratio + style2.card_radius * (1 - ratio),
                button_radius=style1.button_radius * ratio + style2.button_radius * (1 - ratio),
                shadow_strength=style1.shadow_strength * ratio + style2.shadow_strength * (1 - ratio),
                spacing_scale=style1.spacing_scale * ratio + style2.spacing_scale * (1 - ratio),
                density=style1.density if ratio > 0.5 else style2.density,
                has_navbar=style1.has_navbar or style2.has_navbar,
                has_sidebar=style1.has_sidebar or style2.has_sidebar,
                has_footer=style1.has_footer or style2.has_footer,
            )
    
    def add_variation(self, style: StyleBundle, variation_level: float = 0.1) -> StyleBundle:
        """添加微小变化，创造变体"""
        import copy
        varied = copy.deepcopy(style)
        varied.style_id = f"{style.style_id}_var"
        
        # 微调数值参数
        varied.card_radius += random.uniform(-2, 2) * variation_level
        varied.button_radius += random.uniform(-2, 2) * variation_level
        varied.shadow_strength *= random.uniform(0.8, 1.2)
        varied.spacing_scale *= random.uniform(0.95, 1.05)
        
        # 确保合法范围
        varied.card_radius = max(6.0, min(24.0, varied.card_radius))
        varied.button_radius = max(4.0, min(16.0, varied.button_radius))
        varied.shadow_strength = max(0.02, min(0.2, varied.shadow_strength))
        varied.spacing_scale = max(0.85, min(1.3, varied.spacing_scale))
        
        return varied


class StyleRepository:
    """样式仓库"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.real_styles_dir = self.data_dir / "real_styles"
        self.mixed_styles_dir = self.data_dir / "mixed_styles"
        self.index_file = self.data_dir / "style_index.json"
        
        self.real_styles_dir.mkdir(parents=True, exist_ok=True)
        self.mixed_styles_dir.mkdir(parents=True, exist_ok=True)
        
        self.scorer = StyleQualityScorer()
        self.mixer = StyleMixer(self.scorer)
        
        self.index: dict[str, StyleMetadata] = {}
        self.load_index()
    
    def load_index(self):
        """加载样式索引"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.index = {
                    k: StyleMetadata(**v) for k, v in data.items()
                }
    
    def save_index(self):
        """保存样式索引"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(
                {k: asdict(v) for k, v in self.index.items()},
                f,
                indent=2,
                ensure_ascii=False
            )
    
    def add_real_style(
        self,
        style: StyleBundle,
        source_url: str,
        page_family: PageFamily
    ) -> str:
        """添加真实样式到仓库"""
        style_id = style.style_id or f"real_{len(self.index)}"
        style.style_id = style_id
        
        # 评分
        quality = self.scorer.score_overall(style)
        readability = self.scorer.score_readability(style)
        visual_appeal = self.scorer.score_visual_appeal(style)
        
        # 保存样式文件
        style_file = self.real_styles_dir / f"{style_id}.json"
        with open(style_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(style), f, indent=2, ensure_ascii=False)
        
        # 更新索引
        from datetime import datetime
        self.index[style_id] = StyleMetadata(
            style_id=style_id,
            source_type="real",
            source_urls=[source_url],
            page_family=page_family.value,
            quality_score=quality,
            visual_appeal=visual_appeal,
            readability_score=readability,
            created_at=datetime.now().isoformat()
        )
        
        self.save_index()
        return style_id
    
    def load_style(self, style_id: str) -> Optional[StyleBundle]:
        """加载样式"""
        # 先尝试真实样式
        style_file = self.real_styles_dir / f"{style_id}.json"
        if not style_file.exists():
            # 再尝试混合样式
            style_file = self.mixed_styles_dir / f"{style_id}.json"
        
        if not style_file.exists():
            return None
        
        with open(style_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return StyleBundle(**data)
    
    def get_high_quality_styles(self, min_score: float = 0.7, limit: int = None) -> list[str]:
        """获取高质量样式ID列表"""
        qualified = [
            sid for sid, meta in self.index.items()
            if meta.quality_score >= min_score
        ]
        qualified.sort(key=lambda sid: self.index[sid].quality_score, reverse=True)
        return qualified[:limit] if limit else qualified
    
    def generate_mixed_styles(
        self,
        num_styles: int = 100,
        min_quality: float = 0.6,
        strategies: list[str] = None
    ) -> list[str]:
        """批量生成混合样式"""
        if strategies is None:
            strategies = ["balanced", "color_from_1", "layout_from_1"]
        
        # 获取高质量基础样式
        base_styles = self.get_high_quality_styles(min_score=0.7)
        if len(base_styles) < 2:
            print("⚠️ 基础样式不足，需要至少2个高质量样式")
            return []
        
        generated_ids = []
        attempts = 0
        max_attempts = num_styles * 3
        
        while len(generated_ids) < num_styles and attempts < max_attempts:
            attempts += 1
            
            # 随机选择两个不同的样式
            style1_id, style2_id = random.sample(base_styles, 2)
            style1 = self.load_style(style1_id)
            style2 = self.load_style(style2_id)
            
            if not style1 or not style2:
                continue
            
            # 随机选择混合策略
            strategy = random.choice(strategies)
            ratio = random.uniform(0.3, 0.7)
            
            # 混合
            mixed = self.mixer.blend_styles(style1, style2, ratio, strategy)
            
            # 质量检查
            quality = self.scorer.score_overall(mixed)
            if quality < min_quality:
                continue
            
            # 保存
            mixed_id = f"mixed_{len(generated_ids):04d}_{style1_id}_{style2_id}"
            mixed.style_id = mixed_id
            
            style_file = self.mixed_styles_dir / f"{mixed_id}.json"
            with open(style_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(mixed), f, indent=2, ensure_ascii=False)
            
            # 更新索引
            from datetime import datetime
            self.index[mixed_id] = StyleMetadata(
                style_id=mixed_id,
                source_type="mixed",
                source_urls=[style1.source_url, style2.source_url],
                page_family="mixed",
                quality_score=quality,
                visual_appeal=self.scorer.score_visual_appeal(mixed),
                readability_score=self.scorer.score_readability(mixed),
                created_at=datetime.now().isoformat(),
                parent_styles=[style1_id, style2_id]
            )
            
            generated_ids.append(mixed_id)
            
            if len(generated_ids) % 10 == 0:
                print(f"  已生成 {len(generated_ids)}/{num_styles} 个混合样式")
        
        self.save_index()
        print(f"✅ 共生成 {len(generated_ids)} 个混合样式")
        return generated_ids
    
    def get_statistics(self) -> dict:
        """获取仓库统计信息"""
        real_count = sum(1 for m in self.index.values() if m.source_type == "real")
        mixed_count = sum(1 for m in self.index.values() if m.source_type == "mixed")
        
        qualities = [m.quality_score for m in self.index.values()]
        avg_quality = sum(qualities) / len(qualities) if qualities else 0
        
        return {
            "total_styles": len(self.index),
            "real_styles": real_count,
            "mixed_styles": mixed_count,
            "average_quality": avg_quality,
            "high_quality_count": sum(1 for q in qualities if q >= 0.7),
        }
