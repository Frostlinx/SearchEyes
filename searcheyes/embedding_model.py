"""
embedding_model.py — 多模态嵌入模型封装
========================================
封装 Qwen-VL-Embedding / CLIP / MockEmbedder，提供统一的
图片/文本向量化接口，供 wit_indexer 和 multimodal_rag 使用。
"""

from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """嵌入模型抽象基类"""

    @abstractmethod
    def encode_image(self, image_path: str | Path) -> np.ndarray:
        """对图片生成归一化向量"""
        ...

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """对文本生成归一化向量"""
        ...

    def encode_image_text(self, image_path: str | Path, text: str) -> np.ndarray:
        """对图文对生成融合向量（默认: 加权平均后归一化）"""
        img_vec = self.encode_image(image_path)
        txt_vec = self.encode_text(text)
        fused = 0.6 * img_vec + 0.4 * txt_vec
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        return fused

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        ...


class MockEmbedder(BaseEmbedder):
    """
    Mock 嵌入模型 — 生成确定性随机向量，用于无 GPU 环境下的开发测试。
    同一输入始终返回相同向量（基于输入的 hash）。
    """

    def __init__(self, dim: int = 768):
        self._dim = dim
        logger.info(f"MockEmbedder 初始化 (dim={dim})")

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        seed = hash(str(image_path)) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def encode_text(self, text: str) -> np.ndarray:
        seed = hash(text) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    @property
    def embedding_dim(self) -> int:
        return self._dim


class QwenVLEmbedder(BaseEmbedder):
    """
    Qwen-VL-Embedding 封装。
    延迟加载模型，首次调用 encode 时才加载到 GPU。
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "auto",
        dtype: str = "auto",
    ):
        self._model_name = model_name_or_path
        self._device = device
        self._dtype = dtype
        self._model = None
        self._processor = None
        self._dim: int | None = None
        logger.info(f"QwenVLEmbedder 配置: model={model_name_or_path}, device={device}")

    def _ensure_loaded(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        logger.info(f"加载 Qwen-VL 模型: {self._model_name}")
        t0 = time.time()

        dtype_map = {
            "auto": torch.float16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype, torch.float16)

        self._processor = AutoProcessor.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self._model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        if self._device == "auto":
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
        else:
            self._model = self._model.to(self._device)

        self._model.eval()

        # 测试一下获取维度
        dummy = self._processor("test", return_tensors="pt", padding=True)
        dummy = {k: v.to(self._model.device) for k, v in dummy.items() if hasattr(v, 'to')}
        with torch.no_grad():
            out = self._model(**dummy)
            if hasattr(out, 'last_hidden_state'):
                self._dim = out.last_hidden_state.shape[-1]
            elif hasattr(out, 'pooler_output'):
                self._dim = out.pooler_output.shape[-1]
            else:
                self._dim = 768

        logger.info(f"模型加载完成 ({time.time() - t0:.1f}s), dim={self._dim}")

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        self._ensure_loaded()
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items() if hasattr(v, 'to')}

        with torch.no_grad():
            out = self._model(**inputs)
            if hasattr(out, 'last_hidden_state'):
                vec = out.last_hidden_state.mean(dim=1).squeeze()
            elif hasattr(out, 'pooler_output'):
                vec = out.pooler_output.squeeze()
            else:
                vec = out[0].mean(dim=1).squeeze()

        vec = vec.float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_text(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        import torch

        inputs = self._processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items() if hasattr(v, 'to')}

        with torch.no_grad():
            out = self._model(**inputs)
            if hasattr(out, 'last_hidden_state'):
                vec = out.last_hidden_state.mean(dim=1).squeeze()
            elif hasattr(out, 'pooler_output'):
                vec = out.pooler_output.squeeze()
            else:
                vec = out[0].mean(dim=1).squeeze()

        vec = vec.float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        return self._dim


class CLIPEmbedder(BaseEmbedder):
    """
    CLIP 嵌入模型（fallback 方案）。
    使用 openai/clip-vit-large-patch14。
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "auto"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None
        self._tokenizer = None
        logger.info(f"CLIPEmbedder 配置: model={model_name}")

    def _ensure_loaded(self):
        if self._model is not None:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"加载 CLIP 模型: {self._model_name}")
        self._processor = CLIPProcessor.from_pretrained(self._model_name)
        self._model = CLIPModel.from_pretrained(self._model_name)

        if self._device == "auto":
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
        else:
            self._model = self._model.to(self._device)

        self._model.eval()
        logger.info("CLIP 加载完成")

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        self._ensure_loaded()
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items() if hasattr(v, 'to')}

        with torch.no_grad():
            vec = self._model.get_image_features(**inputs).squeeze()

        vec = vec.float().cpu().numpy()
        return vec / np.linalg.norm(vec)

    def encode_text(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        import torch

        inputs = self._processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items() if hasattr(v, 'to')}

        with torch.no_grad():
            vec = self._model.get_text_features(**inputs).squeeze()

        vec = vec.float().cpu().numpy()
        return vec / np.linalg.norm(vec)

    @property
    def embedding_dim(self) -> int:
        return 768  # CLIP ViT-L/14


def create_embedder(
    backend: str = "mock",
    model_name: str = "",
    device: str = "auto",
    **kwargs,
) -> BaseEmbedder:
    """
    工厂函数：根据 backend 创建对应的嵌入模型。

    Args:
        backend: "mock" | "qwen" | "clip"
        model_name: 模型名称/路径（可选）
        device: 设备
    """
    if backend == "mock":
        return MockEmbedder(dim=kwargs.get("dim", 768))
    elif backend == "qwen":
        return QwenVLEmbedder(
            model_name_or_path=model_name or "Qwen/Qwen2-VL-2B-Instruct",
            device=device,
        )
    elif backend == "clip":
        return CLIPEmbedder(
            model_name=model_name or "openai/clip-vit-large-patch14",
            device=device,
        )
    else:
        raise ValueError(f"未知 backend: {backend}，可选: mock, qwen, clip")
