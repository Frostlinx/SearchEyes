#!/usr/bin/env python3
"""
embedding_server.py — Qwen3-VL-Embedding-2B 常驻嵌入服务器
============================================================
启动后持续持有 Embedding 权重，通过 HTTP 接口提供向量化服务。
架构完全对齐 local_model_server.py。

用法:
    # 5090 服务器
    python searcheyes/embedding_server.py \
        --model-path /root/autodl-tmp/QWEN/Qwen3-VL-Embedding-2B \
        --port 8766

    # 测试
    curl http://localhost:8766/health
    curl -X POST http://localhost:8766/embed \
         -H 'Content-Type: application/json' \
         -d '{"image_path": "/path/to/image.jpg"}'

GET  /health       -> {"status": "ok", "dim": 2048}
POST /embed        -> {"image_path": "..."} 和/或 {"text": "..."}
                      返回 {"vector": [...], "dim": 2048}
POST /embed_batch  -> {"items": [{"image_path":...}, {"text":...}, ...]}
                      返回 {"vectors": [[...], ...], "count": N, "dim": 2048}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 模块级单例
_embedder = None
_embed_dim: int = 2048


def _load_embedder(model_path: str, dtype: str = "auto"):
    """加载 Qwen3-VL-Embedding-2B 模型。"""
    global _embedder, _embed_dim

    import torch

    # 将模型自带的 scripts/ 加入 sys.path
    scripts_dir = Path(model_path) / "scripts"
    if scripts_dir.exists() and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from qwen3_vl_embedding import Qwen3VLEmbedder

    kwargs = {}
    if dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16

    print(f"[embed-server] 正在加载模型: {model_path}", flush=True)
    t0 = time.time()
    _embedder = Qwen3VLEmbedder(model_name_or_path=model_path, **kwargs)
    # Qwen3-VL config: hidden_size 在 text_config 子对象中
    cfg = _embedder.model.config
    _embed_dim = getattr(cfg, "hidden_size", None) or getattr(cfg.text_config, "hidden_size", 2048)
    print(
        f"[embed-server] 模型已就绪 dim={_embed_dim} "
        f"device={_embedder.model.device} ({time.time() - t0:.1f}s)",
        flush=True,
    )


def _embed_single(
    image_path: str = "",
    text: str = "",
    instruction: str = "",
) -> list[float]:
    """对单个输入计算归一化嵌入向量。"""
    item: dict = {}
    if image_path:
        p = Path(image_path)
        item["image"] = str(p.resolve()) if not p.is_absolute() else image_path
    if text:
        item["text"] = text
    if instruction:
        item["instruction"] = instruction
    if not item:
        raise ValueError("必须提供 image_path 或 text")

    embeddings = _embedder.process([item])  # (1, dim)
    return embeddings[0].cpu().tolist()


def _embed_batch(
    items: list[dict],
    instruction: str = "",
) -> list[list[float]]:
    """批量嵌入。"""
    processed = []
    for it in items:
        entry: dict = {}
        if it.get("image_path"):
            p = Path(it["image_path"])
            entry["image"] = str(p.resolve()) if not p.is_absolute() else it["image_path"]
        if it.get("text"):
            entry["text"] = it["text"]
        if instruction:
            entry["instruction"] = instruction
        if entry:
            processed.append(entry)
    if not processed:
        return []
    embeddings = _embedder.process(processed)
    return [emb.cpu().tolist() for emb in embeddings]


class _Handler(BaseHTTPRequestHandler):
    """HTTP 处理器：/health, /embed, /embed_batch"""

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "status": "ok" if _embedder is not None else "loading",
                "dim": _embed_dim,
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        if self.path == "/embed":
            try:
                data = json.loads(raw)
                vec = _embed_single(
                    image_path=data.get("image_path", ""),
                    text=data.get("text", ""),
                    instruction=data.get("instruction", ""),
                )
                self._send(200, {"vector": vec, "dim": len(vec)})
            except Exception as exc:
                print(f"[embed-server] ERROR /embed: {exc}", flush=True)
                self._send(500, {"error": str(exc)})

        elif self.path == "/embed_batch":
            try:
                data = json.loads(raw)
                vecs = _embed_batch(
                    data.get("items", []),
                    data.get("instruction", ""),
                )
                self._send(200, {
                    "vectors": vecs,
                    "count": len(vecs),
                    "dim": _embed_dim,
                })
            except Exception as exc:
                print(f"[embed-server] ERROR /embed_batch: {exc}", flush=True)
                self._send(500, {"error": str(exc)})

        else:
            self._send(404, {"error": "not found"})

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-Embedding-2B 嵌入服务")
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/QWEN/Qwen3-VL-Embedding-2B",
        help="模型权重路径",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    _load_embedder(args.model_path, args.dtype)

    print(f"[embed-server] 监听 http://0.0.0.0:{args.port}", flush=True)
    print("[embed-server] Ctrl-C 退出", flush=True)

    with ThreadingHTTPServer(("0.0.0.0", args.port), _Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[embed-server] 收到中断，退出。", flush=True)


if __name__ == "__main__":
    main()
