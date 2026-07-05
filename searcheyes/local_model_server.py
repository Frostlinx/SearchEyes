#!/usr/bin/env python3
"""
local_model_server.py — Qwen3-VL 模型常驻服务器
================================================
启动后持续持有 4B 权重，通过 HTTP 接口提供推理服务，
避免每次调用 run_vlm_pilot.py 都重新加载模型。

用法:
    python searcheyes/local_model_server.py \
        --model-path ./Qwen3-VL-4B-Instruct \
        --device mps --dtype float16 --port 8765

    # 然后在 run_vlm_pilot.py 里加 --server-url http://localhost:8765
    python run_vlm_pilot.py --backend local --server-url http://localhost:8765 --task-index 0

GET  /health  -> {"status": "ok"}
POST /decide  -> 接收 serialize_context() 格式的 JSON，返回 ActionDecision JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# 保证从项目根 import searcheyes
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from searcheyes.vlm_agent import ActionOption, DecisionContext, LocalQwenVisionBackend

# 模块级单例：进程生命周期内只加载一次
_backend: LocalQwenVisionBackend | None = None


class _Handler(BaseHTTPRequestHandler):
    """极简 HTTP 处理器：只处理 /health 和 /decide。"""

    def log_message(self, fmt, *args):  # 静默 access log
        pass

    # ------------------------------------------------------------------
    def do_GET(self):
        if self.path == "/health":
            loaded = _backend is not None and _backend._model is not None
            self._send(200, {"status": "ok" if loaded else "loading"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/decide":
            self._send(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        try:
            data = json.loads(raw)
            context = _deserialize_context(data)
            decision = _backend.decide(context)
            self._send(200, {
                "option_id": decision.option_id,
                "rationale": decision.rationale,
                "confidence": decision.confidence,
            })
        except Exception as exc:
            print(f"[server] ERROR: {exc}", flush=True)
            self._send(500, {"error": str(exc)})

    # ------------------------------------------------------------------
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _deserialize_context(data: dict) -> DecisionContext:
    """把 serialize_context() 格式的字典还原成 DecisionContext。"""
    options = [
        ActionOption(
            option_id=opt["option_id"],
            action=opt["action"],
            params=opt.get("params", {}),
            description=opt.get("description", ""),
            bbox=opt.get("bbox"),
        )
        for opt in data.get("options", [])
    ]
    return DecisionContext(
        task_goal=data.get("task_goal", ""),
        screenshot_path=data["screenshot_path"],
        focused_screenshot_path=data.get("focused_screenshot_path", ""),
        state_summary=data.get("state_summary", ""),
        ui_tokens=data.get("ui_tokens", []),
        options=options,
        rag_facts=data.get("rag_facts", []),
    )


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 模型常驻推理服务")
    parser.add_argument(
        "--model-path",
        default="./Qwen3-VL-4B-Instruct",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    global _backend
    _backend = LocalQwenVisionBackend(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"[server] 正在加载模型: {args.model_path}", flush=True)
    _backend._ensure_loaded()
    print(f"[server] 模型已就绪 ({args.device}/{args.dtype})", flush=True)
    print(f"[server] 监听 http://127.0.0.1:{args.port}", flush=True)
    print("[server] Ctrl-C 退出", flush=True)

    with HTTPServer(("127.0.0.1", args.port), _Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[server] 收到中断，退出。", flush=True)


if __name__ == "__main__":
    main()
