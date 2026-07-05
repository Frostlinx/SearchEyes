"""
caption_bridge.py — Option B 核心：image → text description → retrieval query

用 Qwen3-VL-4B-Instruct 给图片生成文字描述，
再用文字描述走 text embedding + BM25 混合检索。

两种模式:
  - lazy: 用元数据里的 caption（如果已有）
  - vlm:  真正调 VLM 生成描述（需要GPU）
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import VLM_MODEL_PATH, VLM_DEVICE, VLM_MAX_NEW_TOKENS, META_JSONL

_PROMPT_FILE = Path(__file__).parent / "prompts" / "caption_bridge.txt"


@dataclass
class BridgeResult:
    image_path: str
    subject: str        # 主体实体名
    description: str    # 一句话描述
    mode: str           # "lazy" | "vlm" | "fallback"
    raw_output: str = ""

    @property
    def search_query(self) -> str:
        """组合成检索用的文本 query"""
        if self.subject and self.description:
            return f"{self.subject} {self.description}"
        return self.subject or self.description or ""


class CaptionBridge:
    """
    图片 → 文字描述桥接器。

    优先级：
    1. lazy 模式：从 meta.jsonl 直接拿 caption（快，但是 GT 信息，仅用于 oracle 测试）
    2. vlm 模式：真正调 Qwen3-VL-4B 推理（真实场景）
    3. fallback：返回空字符串（不崩溃）
    """

    def __init__(self, use_vlm: bool = True):
        self.use_vlm = use_vlm
        self._vlm_model = None
        self._vlm_processor = None
        self._meta_cache: dict[str, dict] = {}
        self._prompt = _PROMPT_FILE.read_text(encoding="utf-8").strip()
        self._load_meta_cache()

    # ── 公共接口 ─────────────────────────────────────────────────────

    def bridge(self, image_path: str, wit_id: str = "") -> BridgeResult:
        """主入口：给定图片路径，返回 BridgeResult。"""
        img_path = Path(image_path)
        if not img_path.exists():
            return BridgeResult(image_path=image_path, subject="", description="", mode="fallback")

        if self.use_vlm:
            return self._vlm_bridge(image_path)
        else:
            # lazy 模式：从 meta 拿 caption（仅用于对照实验）
            return self._lazy_bridge(image_path, wit_id)

    def bridge_batch(self, items: list[dict]) -> list[BridgeResult]:
        """
        批量处理。items: [{"image_path": str, "wit_id": str}, ...]
        VLM 模式下批量推理效率更高。
        """
        if self.use_vlm:
            return self._vlm_bridge_batch(items)
        return [self.bridge(item["image_path"], item.get("wit_id", "")) for item in items]

    # ── 内部方法 ─────────────────────────────────────────────────────

    def _load_meta_cache(self):
        """加载 meta.jsonl 到内存（用于 lazy 模式）"""
        if not META_JSONL.exists():
            return
        with open(META_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    wit_id = entry.get("wit_id", "")
                    if wit_id:
                        self._meta_cache[wit_id] = entry
                except json.JSONDecodeError:
                    pass

    def _lazy_bridge(self, image_path: str, wit_id: str) -> BridgeResult:
        """从 meta.jsonl 直接拿 caption — 用于对照实验（模拟理想 text bridge）"""
        meta = self._meta_cache.get(wit_id, {})
        caption = meta.get("caption", "")
        page_title = meta.get("page_title", "")
        return BridgeResult(
            image_path=image_path,
            subject=page_title,
            description=caption,
            mode="lazy",
        )

    def _load_vlm(self):
        """延迟加载 VLM（只在需要时）"""
        if self._vlm_model is not None:
            return
        print(f"[CaptionBridge] Loading VLM from {VLM_MODEL_PATH} ...")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            _ModelClass = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            pass
        # Qwen3-VL uses Qwen3VLForConditionalGeneration (transformers >= 4.52)
        try:
            from transformers import Qwen3VLForConditionalGeneration
            _ModelClass = Qwen3VLForConditionalGeneration
        except ImportError:
            pass
        import torch

        self._vlm_processor = AutoProcessor.from_pretrained(
            str(VLM_MODEL_PATH), trust_remote_code=True
        )
        self._vlm_model = _ModelClass.from_pretrained(
            str(VLM_MODEL_PATH),
            torch_dtype=torch.bfloat16,
            device_map=VLM_DEVICE,
            trust_remote_code=True,
        ).eval()
        print(f"[CaptionBridge] VLM loaded ({_ModelClass.__name__}).")

    def _vlm_bridge(self, image_path: str) -> BridgeResult:
        """单张图片 VLM 推理"""
        results = self._vlm_bridge_batch([{"image_path": image_path, "wit_id": ""}])
        return results[0]

    def _vlm_bridge_batch(self, items: list[dict]) -> list[BridgeResult]:
        """批量 VLM 推理"""
        self._load_vlm()
        import torch
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise RuntimeError(
                "qwen-vl-utils not installed. Run: pip install qwen-vl-utils"
            ) from e

        results = []
        for item in items:
            image_path = item["image_path"]
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
                            {"type": "text", "text": self._prompt},
                        ],
                    }
                ]
                text = self._vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self._vlm_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(VLM_DEVICE)

                with torch.no_grad():
                    generated_ids = self._vlm_model.generate(
                        **inputs,
                        max_new_tokens=VLM_MAX_NEW_TOKENS,
                        do_sample=False,
                    )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                raw = self._vlm_processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                subject, description = self._parse_output(raw)
                results.append(BridgeResult(
                    image_path=image_path,
                    subject=subject,
                    description=description,
                    mode="vlm",
                    raw_output=raw,
                ))
            except Exception as exc:
                print(f"[CaptionBridge] VLM failed for {image_path}: {exc}")
                results.append(BridgeResult(
                    image_path=image_path, subject="", description="", mode="fallback"
                ))
        return results

    @staticmethod
    def _parse_output(raw: str) -> tuple[str, str]:
        """解析 VLM 输出的 SUBJECT/DESCRIPTION 格式"""
        subject = ""
        description = ""
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("SUBJECT:"):
                subject = re.sub(r"^SUBJECT:\s*", "", line, flags=re.IGNORECASE).strip()
            elif line.upper().startswith("DESCRIPTION:"):
                description = re.sub(r"^DESCRIPTION:\s*", "", line, flags=re.IGNORECASE).strip()
        if not subject and not description:
            # fallback: 把整个输出当 description
            description = raw[:200]
        return subject, description
