"""
query_transform.py — 让 query 图与 KB 索引图不再 byte-identical

变换：
  - exact:       不变（原 self-match baseline）
  - resize:      下采样到 224x224 后保存（破坏精确像素，保留语义）
  - center_crop: 中心 50% 面积裁剪（模拟 zoom_search 输入）
  - jpeg_q40:    重新以 JPEG quality=40 压缩（破坏频域）
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Literal

TransformName = Literal["exact", "resize", "center_crop", "jpeg_q40"]

_CACHE_DIR = Path(tempfile.gettempdir()) / "vdr_query_xform"
_CACHE_DIR.mkdir(exist_ok=True)


def transform_image(image_path: str, transform: TransformName) -> str:
    """
    返回变换后的图片路径。
    exact 直接返回原路径；其他变换会写入缓存目录。
    """
    if transform == "exact":
        return image_path

    src = Path(image_path)
    cache_key = hashlib.md5(f"{src.resolve()}::{transform}".encode()).hexdigest()[:12]
    out_path = _CACHE_DIR / f"{src.stem}_{transform}_{cache_key}.jpg"
    if out_path.exists():
        return str(out_path)

    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if transform == "resize":
        img = img.resize((224, 224), Image.LANCZOS)
        img.save(out_path, "JPEG", quality=90)

    elif transform == "center_crop":
        # 中心 50% 面积 = 边长 √0.5 ≈ 0.707 → 用 0.5 边长比例（25% 面积，更激进）
        # 改用 50% 边长（25% 面积），模拟"放大的局部"
        side_ratio = 0.5
        cw, ch = int(w * side_ratio), int(h * side_ratio)
        left = (w - cw) // 2
        top = (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))
        img.save(out_path, "JPEG", quality=90)

    elif transform == "jpeg_q40":
        img.save(out_path, "JPEG", quality=40)

    else:
        raise ValueError(f"unknown transform: {transform}")

    return str(out_path)
