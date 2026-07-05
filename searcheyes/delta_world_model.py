"""
delta_world_model.py — 第③层：局部视觉差分模型
==================================================
严格以 StateDiff 为输入，仅处理协议内定义的受控视觉事件。
禁止自由幻想。每种 DiffEventType 对应一种确定性的视觉变化处理函数。
"""

from __future__ import annotations
from PIL import Image, ImageDraw
from searcheyes.state_diff import StateDiff, DiffEvent, DiffEventType


class DeltaWorldModel:
    """受控的视觉差分渲染器——只画 StateDiff 里定义的东西"""

    def apply_diff(self, base_img: Image.Image, diff: StateDiff) -> Image.Image:
        """将 StateDiff 中的所有视觉事件叠加到基础截图上"""
        img = base_img.copy().convert("RGBA")

        for event in diff.events:
            handler = self._get_handler(event.event_type)
            if handler:
                img = handler(img, event)

        return img.convert("RGB")

    def _get_handler(self, event_type: DiffEventType):
        mapping = {
            DiffEventType.MODAL_OPENED:      self._draw_modal,
            DiffEventType.MODAL_CLOSED:       None,
            DiffEventType.DROPDOWN_EXPANDED:  self._draw_dropdown,
            DiffEventType.DROPDOWN_COLLAPSED: None,
            DiffEventType.HIGHLIGHT_CHANGED:  self._draw_highlight,
            DiffEventType.TOAST_SHOWN:        self._draw_toast,
            DiffEventType.PAGE_NAVIGATED:     None,  # 由 template_renderer 处理
            DiffEventType.EVIDENCE_COLLECTED: self._draw_toast,  # v2: 证据收集提示
            DiffEventType.RESULTS_RETURNED:   None,  # 由 template_renderer 处理
            DiffEventType.DOCUMENT_OPENED:    None,  # 由 template_renderer 处理
            DiffEventType.QUERY_REFINED:      None,  # 由 template_renderer 处理
        }
        return mapping.get(event_type)

    def _draw_modal(self, img: Image.Image, event: DiffEvent) -> Image.Image:
        """绘制模态弹窗"""
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size

        # 半透明遮罩
        draw.rectangle([0, 0, w, h], fill=(0, 0, 0, 100))

        # 弹窗主体
        mw, mh = 400, 180
        mx, my = (w - mw) // 2, (h - mh) // 2
        draw.rounded_rectangle([mx, my, mx+mw, my+mh], radius=12,
                               fill=(255,255,255,250), outline=(100,100,100,255), width=2)

        title = event.payload.get("modal_name", "Confirm")
        draw.text((mx+20, my+20), title, fill=(44,62,80,255))
        draw.text((mx+20, my+60), event.payload.get("modal_content", "确认此操作？"),
                  fill=(100,100,100,255))

        # 按钮
        draw.rounded_rectangle([mx+mw-120, my+mh-50, mx+mw-20, my+mh-15],
                               radius=6, fill=(52,152,219,255))
        draw.text((mx+mw-100, my+mh-42), "Confirm", fill=(255,255,255,255))

        return Image.alpha_composite(img, overlay)

    def _draw_dropdown(self, img: Image.Image, event: DiffEvent) -> Image.Image:
        """绘制下拉展开"""
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 固定位置（实际应从元素坐标推导）
        dx, dy = 200, 60
        dw, dh = 180, 100
        draw.rectangle([dx, dy, dx+dw, dy+dh], fill=(255,255,255,245),
                       outline=(200,200,200,255), width=1)
        items = ["Price: Low→High", "Price: High→Low", "Newest"]
        for i, item in enumerate(items):
            ly = dy + 8 + i * 30
            draw.text((dx+10, ly), item, fill=(60,60,60,255))

        return Image.alpha_composite(img, overlay)

    def _draw_highlight(self, img: Image.Image, event: DiffEvent) -> Image.Image:
        """绘制元素高亮"""
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        bbox = event.payload.get("bbox", {"x": 100, "y": 100, "w": 200, "h": 50})
        x, y, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        draw.rectangle([x-3, y-3, x+bw+3, y+bh+3], outline=(52,152,219,200), width=3)

        return Image.alpha_composite(img, overlay)

    def _draw_toast(self, img: Image.Image, event: DiffEvent) -> Image.Image:
        """绘制 Toast 提示"""
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w = img.size[0]

        msg = event.payload.get("message", "操作成功")
        t_type = event.payload.get("type", "success")
        color = (46,204,113,230) if t_type == "success" else (52,152,219,230)

        tw, th = min(len(msg) * 14 + 40, 500), 44
        tx = (w - tw) // 2
        draw.rounded_rectangle([tx, 15, tx+tw, 15+th], radius=22, fill=color)
        draw.text((tx+20, 27), msg, fill=(255,255,255,255))

        return Image.alpha_composite(img, overlay)

