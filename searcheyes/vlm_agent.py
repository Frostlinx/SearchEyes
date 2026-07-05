"""
vlm_agent.py — Phase 8.5 结构化 VLM 决策层
===========================================
把“截图 -> 动作”收敛为严格 JSON 决策，避免自由文本污染环境。
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

_MAX_OPTIONS_IN_PROMPT = 8  # 传给模型的最大候选数，超出截断


@dataclass
class ActionOption:
    """环境当前允许的离散动作候选。"""

    option_id: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    bbox: dict[str, float] | None = None


@dataclass
class DecisionContext:
    """VLM 决策所需的最小上下文。"""

    screenshot_path: str
    task_goal: str = ""
    focused_screenshot_path: str = ""
    state_summary: str = ""
    ui_tokens: list[str] = field(default_factory=list)
    options: list[ActionOption] = field(default_factory=list)
    rag_facts: list[str] = field(default_factory=list)  # 多模态 RAG 检索到的知识


@dataclass
class ActionDecision:
    """VLM 返回的结构化决策。"""

    option_id: str
    rationale: str = ""
    confidence: float = 0.0
    raw_response: str = ""


class DecisionBackend(Protocol):
    def decide(self, context: DecisionContext) -> ActionDecision:
        ...


class JSONActionParser:
    """从模型输出中提取合法 JSON。"""

    @staticmethod
    def parse(raw_text: str, valid_option_ids: set[str]) -> ActionDecision:
        payload = JSONActionParser._extract_json(raw_text)
        option_id = str(payload.get("option_id", "")).strip()
        if option_id not in valid_option_ids:
            raise ValueError(f"非法 option_id: {option_id}")

        confidence = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        return ActionDecision(
            option_id=option_id,
            rationale=str(payload.get("rationale", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
            raw_response=raw_text,
        )

    @staticmethod
    def _extract_json(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if not text:
            raise ValueError("空响应")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("未找到 JSON 对象")
        return json.loads(text[start : end + 1])


class ScriptedPilotBackend:
    """
    用 ground-truth 轨迹驱动的后端。
    主要用于闭环 smoke test，不依赖外部 API。
    """

    def __init__(self, script: list[dict[str, Any]]):
        self.script = script
        self.cursor = 0

    def decide(self, context: DecisionContext) -> ActionDecision:
        while self.cursor < len(self.script):
            target = self.script[self.cursor]
            self.cursor += 1
            if target.get("action") == "observe":
                continue

            for option in context.options:
                if option.action != target.get("action"):
                    continue
                if option.params != target.get("action_params", {}):
                    continue
                return ActionDecision(
                    option_id=option.option_id,
                    rationale=f"matched scripted step {self.cursor - 1}",
                    confidence=1.0,
                )

            return ActionDecision(
                option_id=context.options[0].option_id,
                rationale=f"fallback for scripted step {self.cursor - 1}",
                confidence=0.2,
            )

        if self.cursor >= len(self.script):
            option_id = context.options[0].option_id
            return ActionDecision(option_id=option_id, rationale="script exhausted", confidence=1.0)

        option_id = context.options[0].option_id
        return ActionDecision(option_id=option_id, rationale="script exhausted", confidence=1.0)


class OpenAICompatibleVisionBackend:
    """
    面向兼容 OpenAI Chat Completions 的视觉模型接口。
    可用于 qwen-vl-max 之类的外部服务，只要求 base_url/model/api_key。
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 60,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
        self.timeout_seconds = timeout_seconds

    def decide(self, context: DecisionContext) -> ActionDecision:
        if not self.api_key:
            raise RuntimeError("缺少 API key")
        if not self.base_url:
            raise RuntimeError("缺少 base_url")

        valid_option_ids = {option.option_id for option in context.options}
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_content(context)},
            ],
        }

        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        raw_text = self._extract_message_text(payload["choices"][0]["message"]["content"])
        return JSONActionParser.parse(raw_text, valid_option_ids)

    def _system_prompt(self) -> str:
        return (
            "You are a visual web agent. Choose exactly one allowed action option. "
            "Return JSON only with keys option_id, rationale, confidence."
        )

    def _user_content(self, context: DecisionContext) -> list[dict[str, Any]]:
        text_lines = [
            f"Goal: {context.task_goal or 'N/A'}",
        ]
        # Knowledge 区域始终存在（RAG 是核心搜索引擎）
        if context.rag_facts:
            text_lines.append("Knowledge (from visual search):")
            for fact in context.rag_facts[:3]:
                text_lines.append(f"  - {fact}")
        else:
            text_lines.append("Knowledge: No relevant knowledge retrieved yet.")
        text_lines.append(f"State: {context.state_summary}")
        text_lines.append(f"UI tokens: {', '.join(context.ui_tokens) if context.ui_tokens else 'N/A'}")
        text_lines.append("Allowed options:")
        for option in context.options:
            row = f"- {option.option_id}: {option.description}"
            if option.bbox:
                row += f" @ {option.bbox}"
            text_lines.append(row)
        text_lines.append(
            'Return JSON only, for example: {"option_id":"opt_1","rationale":"...","confidence":0.82}'
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "\n".join(text_lines)},
            {"type": "image_url", "image_url": {"url": self._as_data_url(context.screenshot_path)}},
        ]
        if context.focused_screenshot_path:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._as_data_url(context.focused_screenshot_path)},
                }
            )
        return content

    def _as_data_url(self, path: str) -> str:
        blob = Path(path).read_bytes()
        return "data:image/png;base64," + base64.b64encode(blob).decode("ascii")

    def _extract_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item and isinstance(item["text"], str):
                        parts.append(item["text"])
                    elif item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
            return "\n".join(parts).strip()
        raise ValueError(f"无法解析 message content: {type(content)!r}")


class LocalQwenVisionBackend:
    """
    直接加载本地 Qwen3-VL 权重目录，走 transformers 推理。

    如果传入 server_url（例如 "http://localhost:8765"），则优先将推理请求
    转发给 local_model_server.py 常驻进程，避免重复加载 4B 权重。
    仅当服务器不可达时才回退到进程内推理。
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 64,
        server_url: str = "",
    ):
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.server_url = server_url.rstrip("/")
        self._model = None
        self._processor = None
        self._torch = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def decide(self, context: DecisionContext) -> ActionDecision:
        # 优先走常驻服务器
        if self.server_url:
            result = self._try_server(context)
            if result is not None:
                return result
            # 服务器不可达时降级为进程内推理
            print("[LocalQwenVisionBackend] 服务器不可达，回退进程内推理", flush=True)

        self._ensure_loaded()
        valid_option_ids = {option.option_id for option in context.options}

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._system_prompt()}]},
            {"role": "user", "content": self._user_content(context)},
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_to_device(inputs)

        with self._torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return JSONActionParser.parse(output_text, valid_option_ids)

    def _try_server(self, context: DecisionContext) -> ActionDecision | None:
        """向常驻服务器发送推理请求；失败返回 None。"""
        try:
            payload = self._post_json(
                url=f"{self.server_url}/decide",
                payload=serialize_context(context),
                connect_timeout=3.0,
                read_timeout=60.0,
            )
            valid_option_ids = {opt.option_id for opt in context.options}
            option_id = str(payload.get("option_id", "")).strip()
            if option_id not in valid_option_ids:
                raise ValueError(f"服务器返回非法 option_id: {option_id}")
            confidence = float(payload.get("confidence", 0.0))
            return ActionDecision(
                option_id=option_id,
                rationale=str(payload.get("rationale", "")).strip(),
                confidence=max(0.0, min(1.0, confidence)),
                raw_response=json.dumps(payload),
            )
        except Exception as exc:
            print(f"[LocalQwenVisionBackend] 服务器请求失败: {exc}", flush=True)
            return None

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        connect_timeout: float,
        read_timeout: float,
    ) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        parsed = urllib.parse.urlsplit(url)
        connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = connection_cls(parsed.hostname, parsed.port, timeout=connect_timeout)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        try:
            connection.request("POST", path, body=data, headers={"Content-Type": "application/json"})
            if connection.sock is not None:
                connection.sock.settimeout(read_timeout)
            response = connection.getresponse()
            raw = response.read()
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {raw.decode('utf-8', errors='ignore')}")
            return json.loads(raw.decode("utf-8"))
        finally:
            connection.close()

    def _ensure_loaded(self):
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._torch = torch
        resolved_device = self._resolve_device(torch)

        load_kwargs: dict[str, Any] = {}
        torch_dtype = self._resolve_torch_dtype(torch)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        if resolved_device == "mps":
            load_kwargs["device_map"] = None
        elif resolved_device == "cpu":
            load_kwargs["device_map"] = None
        else:
            load_kwargs["device_map"] = resolved_device

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_path, **load_kwargs)
        self._processor = AutoProcessor.from_pretrained(self.model_path)

        if resolved_device == "mps":
            self._model.to("mps")
        elif resolved_device == "cpu":
            self._model.to("cpu")
        self.device = resolved_device

    def _resolve_torch_dtype(self, torch) -> Any:
        if self.dtype == "auto":
            return None
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"不支持的 dtype: {self.dtype}")

    def _move_to_device(self, inputs):
        if getattr(self._model, "device", None) is not None:
            return inputs.to(self._model.device)
        if self.device == "mps":
            return inputs.to("mps")
        if self.device == "cpu":
            return inputs.to("cpu")
        return inputs

    def _resolve_device(self, torch) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _system_prompt(self) -> str:
        return (
            "Visual web agent. Pick exactly one option. "
            'Output JSON only: {"option_id":"...","rationale":"...","confidence":0.9}'
        )

    def _user_content(self, context: DecisionContext) -> list[dict[str, Any]]:
        # 截断候选数量，防止 prompt 过长拖慢生成
        options = context.options[:_MAX_OPTIONS_IN_PROMPT]

        lines = [f"Goal: {context.task_goal or 'N/A'}"]
        if context.rag_facts:
            lines.append("Knowledge (from visual search):")
            for fact in context.rag_facts[:3]:
                lines.append(f"  - {fact}")
        else:
            lines.append("Knowledge: No relevant knowledge retrieved yet.")
        lines.append(f"State: {context.state_summary}")
        lines.append("Options:")
        for opt in options:
            lines.append(f"- {opt.option_id}: {opt.description}")
        lines.append('Reply JSON: {"option_id":"...","rationale":"...","confidence":0.9}')

        content: list[dict[str, Any]] = [
            {"type": "image", "image": context.screenshot_path},
            {"type": "text", "text": "\n".join(lines)},
        ]
        if context.focused_screenshot_path:
            content.insert(1, {"type": "image", "image": context.focused_screenshot_path})
        return content


def serialize_context(context: DecisionContext) -> dict[str, Any]:
    """便于落盘调试。"""

    return {
        "task_goal": context.task_goal,
        "screenshot_path": context.screenshot_path,
        "focused_screenshot_path": context.focused_screenshot_path,
        "state_summary": context.state_summary,
        "ui_tokens": list(context.ui_tokens),
        "options": [asdict(option) for option in context.options],
        "rag_facts": list(context.rag_facts),
    }


# ── 训练工具函数（供 train_grpo.py 使用）────────────────────

def resolve_device_choice(torch_module: Any, choice: str) -> str:
    if choice != "auto":
        return choice
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype_choice(torch_module: Any, choice: str, device: str) -> Any:
    if choice == "bfloat16":
        return torch_module.bfloat16
    if choice == "float16":
        return torch_module.float16
    if choice == "float32":
        return torch_module.float32
    # auto
    if device == "cuda":
        return torch_module.bfloat16
    return torch_module.float32


def resolve_quantization_choice(torch_module: Any, choice: str, device: str) -> str:
    if choice == "auto":
        return "4bit" if device == "cuda" else "none"
    return choice


def resolve_attn_implementation_choice(choice: str, device: str) -> str:
    if choice != "auto":
        return choice
    if device == "cuda":
        return "sdpa"
    return "eager"


def build_quantization_config(
    torch_module: Any,
    transformers_module: Any,
    quantization: str,
    compute_dtype: Any,
) -> Any:
    if quantization == "4bit":
        return transformers_module.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if quantization == "8bit":
        return transformers_module.BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization: {quantization}")
