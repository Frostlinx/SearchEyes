"""
crop_engine.py — 多尺度裁剪引擎（zoom_search 预备）

从图片生成多个候选 crop，供 Option A/B 检索。
目前实现：固定网格裁剪（top/bottom/left/right/center）
未来：MLLM bounding box 生成（Vision-DeepResearch 方式）
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CropSpec:
    label: str          # "full" | "top_half" | "bottom_half" | "center" | "entity_box"
    left: float         # 0.0-1.0 相对坐标
    top: float
    right: float
    bottom: float


# 固定多尺度裁剪方案（不需要 MLLM）
FIXED_CROPS: list[CropSpec] = [
    CropSpec("full",          0.0, 0.0, 1.0, 1.0),
    CropSpec("top_half",      0.0, 0.0, 1.0, 0.5),
    CropSpec("bottom_half",   0.0, 0.5, 1.0, 1.0),
    CropSpec("center",        0.25, 0.25, 0.75, 0.75),
    CropSpec("left_half",     0.0, 0.0, 0.5, 1.0),
    CropSpec("right_half",    0.5, 0.0, 1.0, 1.0),
]


class CropEngine:
    """
    给定图片，生成多个候选 crop。
    每个 crop 可以单独送入 Option A 或 Option B 检索。
    """

    def crop_image(self, image_path: str, spec: CropSpec,
                   output_path: Optional[str] = None) -> str:
        """
        裁剪图片，返回裁剪后图片的路径。
        output_path 为 None 时写入临时文件。
        """
        from PIL import Image
        import tempfile

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        box = (
            int(spec.left * w),
            int(spec.top * h),
            int(spec.right * w),
            int(spec.bottom * h),
        )
        cropped = img.crop(box)

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            output_path = tmp.name

        cropped.save(output_path, "JPEG", quality=90)
        return output_path

    def get_crops(self, image_path: str, crop_dir: Optional[Path] = None,
                  specs: Optional[list[CropSpec]] = None) -> list[tuple[CropSpec, str]]:
        """
        生成所有 crop，返回 [(CropSpec, crop_image_path), ...]
        """
        specs = specs or FIXED_CROPS
        if crop_dir:
            crop_dir = Path(crop_dir)
            crop_dir.mkdir(parents=True, exist_ok=True)

        results = []
        stem = Path(image_path).stem
        for spec in specs:
            if crop_dir:
                out = str(crop_dir / f"{stem}_{spec.label}.jpg")
            else:
                out = None
            try:
                out_path = self.crop_image(image_path, spec, out)
                results.append((spec, out_path))
            except Exception as exc:
                print(f"[CropEngine] crop {spec.label} failed: {exc}")
        return results
