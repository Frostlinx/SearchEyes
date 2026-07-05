"""
vdr_tools.py — Vision-DeepResearch 工具集
==========================================
实现 VDR 论文定义的 4 个 text-observation 工具，底层接 RagEngine。

工具列表:
  1. search(query)                — 文本检索 Wiki6M 知识库
  2. crop_and_search(image, region) — 裁剪图片区域后重新检索
  3. visit(entity_id)             — 访问实体完整内容
  4. python_interpreter(code)     — 执行 Python 代码片段

设计原则 (来自 VDR 论文 arXiv:2601.22060):
  - 所有工具返回纯文本 observation（不是 GUI 截图）
  - observation 长度有上限，防止 context window 爆炸
  - 每个工具调用都是确定性的（相同输入 → 相同输出）
"""

from __future__ import annotations

import io
import logging
import re
import sys
import tempfile
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 1. ToolResult — 工具调用的统一返回
# ═══════════════════════════════════════════════════════════

@dataclass
class ToolResult:
    """单次工具调用的返回结果。"""
    tool_name: str
    observation: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def truncated_observation(self, max_chars: int = 4000) -> str:
        """截断 observation 防止 context window 爆炸。"""
        if len(self.observation) <= max_chars:
            return self.observation
        half = max_chars // 2
        return (
            self.observation[:half]
            + f"\n\n... [truncated {len(self.observation) - max_chars} chars] ...\n\n"
            + self.observation[-half:]
        )


# ═══════════════════════════════════════════════════════════
# 2. ToolKit — 4 个工具的统一入口
# ═══════════════════════════════════════════════════════════

# 裁剪区域规格 (label -> (left, top, right, bottom) 相对坐标)
CROP_SPECS: dict[str, tuple[float, float, float, float]] = {
    "full":        (0.0, 0.0, 1.0, 1.0),
    "top_half":    (0.0, 0.0, 1.0, 0.5),
    "bottom_half": (0.0, 0.5, 1.0, 1.0),
    "center":      (0.25, 0.25, 0.75, 0.75),
    "left_half":   (0.0, 0.0, 0.5, 1.0),
    "right_half":  (0.5, 0.0, 1.0, 1.0),
}

# Python interpreter 安全限制
_INTERPRETER_MAX_OUTPUT = 2000
_INTERPRETER_TIMEOUT_SECONDS = 10
_INTERPRETER_FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "shutil", "socket", "http",
    "urllib", "requests", "sys", "importlib",
})


class VDRToolKit:
    """Vision-DeepResearch 工具集，底层接 RagEngine。

    Args:
        rag_engine: 已加载的 RagEngine 实例
        images_dir: 图片目录（用于 crop_and_search）
        search_top_k: search 返回的最大结果数
        max_observation_chars: observation 最大字符数
    """

    TOOL_DESCRIPTIONS: dict[str, str] = {
        "search": (
            "search(query: str) -> str\n"
            "Search the Wikipedia knowledge base with a text query.\n"
            "Returns a numbered list of matching entities with title and summary."
        ),
        "crop_and_search": (
            "crop_and_search(image_path: str, region: str) -> str\n"
            "Crop a region from the query image and search for visually similar entities.\n"
            "region must be one of: full, top_half, bottom_half, center, left_half, right_half."
        ),
        "visit": (
            "visit(entity_id: str) -> str\n"
            "Visit a specific Wikipedia entity by its ID (e.g., 'Q12345').\n"
            "Returns the full article content including title, summary, and body text."
        ),
        "python_interpreter": (
            "python_interpreter(code: str) -> str\n"
            "Execute a Python code snippet for computation or string processing.\n"
            "Only basic math and string operations are allowed. No file/network access."
        ),
    }

    def __init__(
        self,
        rag_engine: Any,
        images_dir: str | Path = "/dev/shm/oven_wiki/images",
        search_top_k: int = 10,
        max_observation_chars: int = 4000,
    ):
        self.rag = rag_engine
        self.images_dir = Path(images_dir)
        self.search_top_k = search_top_k
        self.max_observation_chars = max_observation_chars
        self.current_image_path: str = ""  # 当前 episode 的查询图片路径

    # ── 统一调度 ─────────────────────────────────────────

    def call(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """统一工具调度入口。"""
        dispatch = {
            "search": self._tool_search,
            "crop_and_search": self._tool_crop_and_search,
            "visit": self._tool_visit,
            "python_interpreter": self._tool_python_interpreter,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return ToolResult(
                tool_name=tool_name,
                observation=f"Unknown tool '{tool_name}'. Available: {', '.join(dispatch)}",
                success=False,
            )
        try:
            result = handler(**kwargs)
            # 截断过长的 observation
            if len(result.observation) > self.max_observation_chars:
                result.observation = result.truncated_observation(self.max_observation_chars)
            return result
        except Exception as exc:
            logger.exception("Tool %s failed", tool_name)
            return ToolResult(
                tool_name=tool_name,
                observation=f"Error executing {tool_name}: {exc}",
                success=False,
            )

    def get_tool_descriptions(self) -> str:
        """返回所有工具的描述，用于 system prompt。"""
        lines = []
        for idx, (name, desc) in enumerate(self.TOOL_DESCRIPTIONS.items(), 1):
            lines.append(f"Tool {idx}: {desc}")
        return "\n\n".join(lines)

    # ── Tool 1: search ───────────────────────────────────

    def _tool_search(self, query: str = "", **kwargs: Any) -> ToolResult:
        """文本检索 Wiki6M 知识库。"""
        if not query or not query.strip():
            return ToolResult(
                tool_name="search",
                observation="Error: query cannot be empty.",
                success=False,
            )

        query = query.strip()
        facts = self.rag.get_rag_facts_combined(
            text=query,
            top_k=self.search_top_k,
            use_hybrid=True,
        )

        if not facts:
            return ToolResult(
                tool_name="search",
                observation=f"No results found for query: '{query}'",
                success=True,
                metadata={"query": query, "num_results": 0},
            )

        # 格式化为 text observation
        lines = [f"Search results for '{query}' ({len(facts)} results):"]
        lines.append("")
        for idx, fact in enumerate(facts, 1):
            title = fact.title or "(untitled)"
            summary = (fact.caption or fact.summary or "")[:200]
            has_image = "🖼" if fact.image_url else ""
            lines.append(f"[{idx}] {title} {has_image}")
            lines.append(f"    ID: {fact.wit_id}")
            if summary:
                lines.append(f"    {summary}")
            lines.append("")

        return ToolResult(
            tool_name="search",
            observation="\n".join(lines),
            success=True,
            metadata={
                "query": query,
                "num_results": len(facts),
                "entity_ids": [f.wit_id for f in facts],
            },
        )

    # ── Tool 2: crop_and_search ──────────────────────────

    def _tool_crop_and_search(
        self,
        image_path: str = "",
        region: str = "center",
        **kwargs: Any,
    ) -> ToolResult:
        """裁剪图片区域后用 image embedding 重新检索。"""
        if not image_path:
            return ToolResult(
                tool_name="crop_and_search",
                observation="Error: image_path is required.",
                success=False,
            )

        image_path_obj = Path(image_path)
        # Fallback: Agent 可能传 "image.png" / "user_provided_image" 等非绝对路径
        if not image_path_obj.exists() and self.current_image_path:
            image_path_obj = Path(self.current_image_path)

        if not image_path_obj.exists():
            return ToolResult(
                tool_name="crop_and_search",
                observation=f"Error: image not found at '{image_path}'.",
                success=False,
            )

        region = region.strip().lower()
        if region not in CROP_SPECS:
            return ToolResult(
                tool_name="crop_and_search",
                observation=(
                    f"Error: unknown region '{region}'. "
                    f"Must be one of: {', '.join(CROP_SPECS)}"
                ),
                success=False,
            )

        # 裁剪图片
        spec = CROP_SPECS[region]
        try:
            from PIL import Image
            img = Image.open(str(image_path_obj)).convert("RGB")
            width, height = img.size
            box = (
                int(spec[0] * width),
                int(spec[1] * height),
                int(spec[2] * width),
                int(spec[3] * height),
            )
            cropped = img.crop(box)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cropped.save(tmp.name, "JPEG", quality=90)
                crop_path = tmp.name
        except Exception as exc:
            return ToolResult(
                tool_name="crop_and_search",
                observation=f"Error cropping image: {exc}",
                success=False,
            )

        # 获取 image embedding 并检索
        embedding = self.rag._get_embedding(crop_path)
        if embedding is None:
            # fallback: 没有 image embedding model，尝试用 caption bridge
            return ToolResult(
                tool_name="crop_and_search",
                observation=(
                    "Image embedding model not available. "
                    "Use search(query) with a text description instead."
                ),
                success=False,
                metadata={"region": region, "fallback": "no_embedding_model"},
            )

        facts = self.rag._query_chroma(embedding, top_k=self.search_top_k)

        if not facts:
            return ToolResult(
                tool_name="crop_and_search",
                observation=f"No visual matches found for {region} region of the image.",
                success=True,
                metadata={"region": region, "num_results": 0},
            )

        # 格式化
        lines = [f"Visual search results for '{region}' crop ({len(facts)} results):"]
        lines.append("")
        for idx, fact in enumerate(facts, 1):
            title = fact.title or "(untitled)"
            summary = (fact.caption or fact.summary or "")[:200]
            lines.append(f"[{idx}] {title}")
            lines.append(f"    ID: {fact.wit_id}")
            if summary:
                lines.append(f"    {summary}")
            lines.append("")

        return ToolResult(
            tool_name="crop_and_search",
            observation="\n".join(lines),
            success=True,
            metadata={
                "region": region,
                "num_results": len(facts),
                "entity_ids": [f.wit_id for f in facts],
            },
        )

    # ── Tool 3: visit ────────────────────────────────────

    def _tool_visit(self, entity_id: str = "", **kwargs: Any) -> ToolResult:
        """访问实体的完整内容。"""
        if not entity_id or not entity_id.strip():
            return ToolResult(
                tool_name="visit",
                observation="Error: entity_id is required (e.g., 'Q12345').",
                success=False,
            )

        entity_id = entity_id.strip()
        full_entity = self.rag.read_full_entity(entity_id)

        if full_entity is None:
            return ToolResult(
                tool_name="visit",
                observation=f"Entity '{entity_id}' not found in the knowledge base.",
                success=False,
            )

        title = full_entity.get("title", "(untitled)")
        summary = full_entity.get("summary", "")
        content = full_entity.get("content", "")
        image_url = full_entity.get("image_url", "")

        lines = [f"=== {title} ==="]
        lines.append(f"Entity ID: {entity_id}")
        if image_url:
            lines.append(f"Image: {image_url}")
        lines.append("")

        if summary:
            lines.append("## Summary")
            lines.append(summary)
            lines.append("")

        if content:
            lines.append("## Full Content")
            lines.append(content)

        observation = "\n".join(lines)
        return ToolResult(
            tool_name="visit",
            observation=observation,
            success=True,
            metadata={"entity_id": entity_id, "title": title},
        )

    # ── Tool 4: python_interpreter ───────────────────────

    def _tool_python_interpreter(self, code: str = "", **kwargs: Any) -> ToolResult:
        """在沙箱中执行 Python 代码。仅允许基本计算和字符串处理。"""
        if not code or not code.strip():
            return ToolResult(
                tool_name="python_interpreter",
                observation="Error: code cannot be empty.",
                success=False,
            )

        # 安全检查：禁止危险模块
        violation = self._check_code_safety(code)
        if violation:
            return ToolResult(
                tool_name="python_interpreter",
                observation=f"Security error: {violation}",
                success=False,
            )

        # 在受限环境中执行
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        local_vars: dict[str, Any] = {}

        try:
            compiled = compile(code, "<agent_code>", "exec")
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled, {"__builtins__": _SAFE_BUILTINS}, local_vars)
        except Exception:
            error_text = traceback.format_exc()
            return ToolResult(
                tool_name="python_interpreter",
                observation=f"Execution error:\n{error_text}",
                success=False,
            )

        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()

        output_parts = []
        if stdout_text:
            output_parts.append(stdout_text.rstrip())
        if stderr_text:
            output_parts.append(f"[stderr]\n{stderr_text.rstrip()}")

        # 如果没有 print 输出，检查是否有 result 变量
        if not output_parts and "result" in local_vars:
            output_parts.append(str(local_vars["result"]))

        observation = "\n".join(output_parts) if output_parts else "(no output)"

        # 截断过长的输出
        if len(observation) > _INTERPRETER_MAX_OUTPUT:
            observation = (
                observation[:_INTERPRETER_MAX_OUTPUT]
                + f"\n... [output truncated at {_INTERPRETER_MAX_OUTPUT} chars]"
            )

        return ToolResult(
            tool_name="python_interpreter",
            observation=observation,
            success=True,
        )

    @staticmethod
    def _check_code_safety(code: str) -> str:
        """检查代码是否包含危险操作。返回违规描述或空字符串。"""
        for module in _INTERPRETER_FORBIDDEN_MODULES:
            # 匹配 import xxx 或 from xxx import
            pattern = rf'\b(?:import\s+{module}|from\s+{module}\b)'
            if re.search(pattern, code):
                return f"Importing '{module}' is not allowed."

        # 禁止 open / exec / eval 等
        dangerous_calls = ["open(", "exec(", "eval(", "__import__"]
        for call in dangerous_calls:
            if call in code:
                return f"Using '{call.rstrip('(')}' is not allowed."

        return ""

    # ── 工具解析 ─────────────────────────────────────────

    @staticmethod
    def parse_action(action_text: str) -> tuple[str, dict[str, str]]:
        """从 ReAct Action 文本中解析工具名和参数。

        支持的格式:
          search(query="Eiffel Tower history")
          visit(entity_id="Q243")
          crop_and_search(image_path="/path/to/img.jpg", region="center")
          python_interpreter(code="print(2+2)")

        Returns:
            (tool_name, kwargs_dict)
        """
        action_text = action_text.strip()

        # 格式: tool_name(key="value", ...)
        match = re.match(r'(\w+)\s*\((.*)\)\s*$', action_text, re.DOTALL)
        if not match:
            return action_text, {}

        tool_name = match.group(1)
        args_str = match.group(2).strip()

        if not args_str:
            return tool_name, {}

        # 解析 key=value 对
        kwargs: dict[str, str] = {}
        # 用正则匹配 key="value" 或 key='value' 或 key="""value"""
        arg_pattern = re.compile(
            r'(\w+)\s*=\s*'
            r'(?:'
            r'"""(.*?)"""|'    # triple-double-quoted
            r"'''(.*?)'''|"    # triple-single-quoted
            r'"((?:[^"\\]|\\.)*)"|'    # double-quoted
            r"'((?:[^'\\]|\\.)*)'|"    # single-quoted
            r'([^,\s]+)'              # unquoted
            r')',
            re.DOTALL,
        )
        for arg_match in arg_pattern.finditer(args_str):
            key = arg_match.group(1)
            # 取第一个非 None 的捕获组作为 value
            value = next(
                (g for g in arg_match.groups()[1:] if g is not None),
                "",
            )
            # 还原转义字符
            value = value.replace('\\"', '"').replace("\\'", "'")
            kwargs[key] = value

        return tool_name, kwargs


# ═══════════════════════════════════════════════════════════
# 3. 安全内建函数白名单（Python Interpreter 沙箱）
# ═══════════════════════════════════════════════════════════

_SAFE_BUILTINS: dict[str, Any] = {
    # 类型
    "True": True, "False": False, "None": None,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "frozenset": frozenset, "bytes": bytes, "bytearray": bytearray,
    "complex": complex,
    # 数学
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "divmod": divmod,
    # 序列
    "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed,
    "all": all, "any": any,
    # 字符串
    "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin,
    "format": format, "repr": repr, "type": type,
    # IO
    "print": print,
    # 异常
    "ValueError": ValueError, "TypeError": TypeError,
    "IndexError": IndexError, "KeyError": KeyError,
    "Exception": Exception,
    # 其他
    "isinstance": isinstance, "issubclass": issubclass,
    "hasattr": hasattr, "getattr": getattr,
    "hash": hash, "id": id, "callable": callable,
    "iter": iter, "next": next,
    "input": lambda *_: "",  # 禁止真正的 input
}
