#!/usr/bin/env python3
"""Quick test: can we import and load the embedding model?"""
import sys
sys.path.insert(0, "/root/autodl-tmp/QWEN/Qwen3-VL-Embedding-2B/scripts")

from qwen3_vl_embedding import Qwen3VLEmbedder
print("Import OK")

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

model = Qwen3VLEmbedder(
    model_name_or_path="/root/autodl-tmp/QWEN/Qwen3-VL-Embedding-2B",
    torch_dtype=torch.float16,
)
cfg = model.model.config
dim = getattr(cfg, "hidden_size", None) or getattr(cfg.text_config, "hidden_size", 2048)
print(f"Model loaded, dim={dim}, device={model.model.device}")

# Test with text
emb = model.process([{"text": "a photo of a cat"}])
print(f"Text embedding shape: {emb.shape}")
print(f"First 5 values: {emb[0][:5].tolist()}")
print("ALL TESTS PASSED")
