"""
vdr_agent.py — Vision-DeepResearch ReAct Agent
================================================
Text-based ReAct Agent，参考 VDR 论文 (arXiv:2601.22060)。

工作流:
  1. 接收 ResearchTask（含 query image + question）
  2. ReAct 循环: Thought → Action(tool_call) → Observation → ...
  3. 最终输出 Final Answer

所有 observation 为纯文本，不依赖 GUI 截图。
通过 OpenAI-compatible API 调用 VLM（如 vLLM 服务的 Qwen3）。
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from searcheyes.vdr_tools import VDRToolKit, ToolResult

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 1. 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class AgentStep:
    """单步 ReAct 记录。"""
    step_idx: int
    thought: str = ""
    action_raw: str = ""
    tool_name: str = ""
    tool_kwargs: dict[str, str] = field(default_factory=dict)
    observation: str = ""
    success: bool = True
    elapsed_ms: float = 0.0


@dataclass
class EpisodeResult:
    """一次完整 episode 的结果。"""
    task_id: str
    question: str
    image_path: str
    final_answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    total_steps: int = 0
    total_elapsed_ms: float = 0.0
    ground_truth: str = ""
    correct: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


# ═══════════════════════════════════════════════════════════
# 2. Prompt 模板
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """\
You are a research agent that answers visual questions by searching a Wikipedia knowledge base.

You have access to the following tools:

{tool_descriptions}

## Instructions

1. You will be given an image and a question about the image.
2. Use the tools to search for relevant information to answer the question.
3. Think step by step. For each step, output your reasoning as "Thought:", then call a tool as "Action:".
4. After each tool call, you will receive an "Observation:" with the result.
5. When you have enough information, output "Final Answer:" followed by your answer.

## Output Format

Always use this exact format:

Thought: <your reasoning about what to do next>
Action: <tool_call, e.g., search(query="Eiffel Tower")>

After receiving an observation:

Thought: <your reasoning about the observation>
Action: <next tool_call>

When ready to answer:

Thought: <your final reasoning>
Final Answer: <your answer>

## Important Rules

- Be concise in your thoughts.
- Use search() first to find relevant entities, then visit() to read details.
- Use crop_and_search() when the image contains specific objects to identify.
- Use python_interpreter() only for calculations or string processing.
- You must output a Final Answer within {max_steps} steps.
- The Final Answer should be a specific entity name, not a description.
"""

USER_PROMPT_TEMPLATE = """\
Question: {question}

Please analyze the image and use the available tools to find the answer.\
"""


# ═══════════════════════════════════════════════════════════
# 3. LLM 调用层
# ═══════════════════════════════════════════════════════════

class VLMClient:
    """OpenAI-compatible VLM API 客户端。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "/dev/shm/Qwen3.6-27B",
        api_key: str = "EMPTY",
        temperature: float = 0.6,
        max_tokens: int = 1024,
        timeout_seconds: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def chat(
        self,
        messages: list[dict[str, Any]],
        stop: list[str] | None = None,
    ) -> str:
        """调用 chat completions API，返回 assistant 回复文本。"""
        import urllib.request

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Disable thinking mode for Qwen3 — ReAct prompt already
            # structures reasoning via explicit Thought/Action format
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if stop:
            body["stop"] = stop

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            text_parts = [
                item["text"] if isinstance(item, dict) else str(item)
                for item in content
                if isinstance(item, str) or (isinstance(item, dict) and item.get("type") == "text")
            ]
            return "\n".join(text_parts)
        return str(content)

    @staticmethod
    def encode_image_to_data_url(image_path: str) -> str:
        """将图片编码为 base64 data URL。"""
        path = Path(image_path)
        if not path.exists():
            return ""
        suffix = path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp",
        }.get(suffix, "image/jpeg")
        blob = path.read_bytes()
        return f"data:{mime};base64," + base64.b64encode(blob).decode("ascii")


# ═══════════════════════════════════════════════════════════
# 4. ReAct Agent 主循环
# ═══════════════════════════════════════════════════════════

class VDRAgent:
    """Vision-DeepResearch ReAct Agent。

    Args:
        toolkit: VDRToolKit 实例（已接入 RagEngine）
        vlm_client: VLM API 客户端
        max_steps: 最大交互步数
        max_observation_chars: observation 截断长度
    """

    def __init__(
        self,
        toolkit: VDRToolKit,
        vlm_client: VLMClient | None = None,
        max_steps: int = 50,
        max_observation_chars: int = 1024,
        enable_sliding_window: bool = False,
        max_context_tokens: int = 28000,
    ):
        self.toolkit = toolkit
        self.vlm = vlm_client or VLMClient()
        self.max_steps = max_steps
        self.max_observation_chars = max_observation_chars
        self.enable_sliding_window = enable_sliding_window
        self.max_context_tokens = max_context_tokens

    def run_episode(
        self,
        question: str,
        image_path: str = "",
        task_id: str = "",
        ground_truth: str = "",
    ) -> EpisodeResult:
        """执行一个完整的 ReAct episode。

        Args:
            question: 要回答的问题
            image_path: 查询图片路径（可选）
            task_id: 任务 ID（用于日志）
            ground_truth: 正确答案（用于评估，不传给 Agent）
        """
        episode = EpisodeResult(
            task_id=task_id,
            question=question,
            image_path=image_path,
            ground_truth=ground_truth,
        )
        episode_start = time.time()

        # 设置 toolkit 的当前图片路径（供 crop_and_search fallback）
        self.toolkit.current_image_path = image_path

        # 构建初始 messages
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=self.toolkit.get_tool_descriptions(),
            max_steps=self.max_steps,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        # User message: 图片 + 问题
        user_content = self._build_user_message(question, image_path)
        messages.append({"role": "user", "content": user_content})

        # ReAct 循环
        accumulated_text = ""
        for step_idx in range(self.max_steps):
            step = AgentStep(step_idx=step_idx)
            step_start = time.time()

            # 调用 VLM 生成 Thought + Action (或 Final Answer)
            try:
                response = self.vlm.chat(
                    messages=messages,
                    stop=["Observation:", "observation:"],
                )
            except Exception as exc:
                logger.error("VLM call failed at step %d: %s", step_idx, exc)
                step.thought = f"[VLM call failed: {exc}]"
                step.success = False
                step.elapsed_ms = (time.time() - step_start) * 1000
                episode.steps.append(step)
                break

            response = response.strip()
            accumulated_text = response

            # 解析 Thought 和 Action / Final Answer
            thought, action_or_answer = self._parse_response(response)
            step.thought = thought

            # 检查是否是 Final Answer
            final_answer = self._extract_final_answer(response)
            if final_answer:
                step.elapsed_ms = (time.time() - step_start) * 1000
                episode.steps.append(step)
                episode.final_answer = final_answer
                break

            # 解析并执行 Action
            if not action_or_answer:
                # 没有 Action 也没有 Final Answer，追加提示让模型继续
                step.observation = "[No action found. Please output an Action or Final Answer.]"
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": step.observation})
                step.success = False
                step.elapsed_ms = (time.time() - step_start) * 1000
                episode.steps.append(step)
                continue

            step.action_raw = action_or_answer
            tool_name, tool_kwargs = VDRToolKit.parse_action(action_or_answer)
            step.tool_name = tool_name
            step.tool_kwargs = tool_kwargs

            # 如果 Agent 调用 crop_and_search 但没提供 image_path，自动填入
            if tool_name == "crop_and_search" and "image_path" not in tool_kwargs and image_path:
                tool_kwargs["image_path"] = image_path

            # 执行工具
            tool_result = self.toolkit.call(tool_name, **tool_kwargs)
            observation = tool_result.truncated_observation(self.max_observation_chars)
            step.observation = observation
            step.success = tool_result.success

            # 将 assistant 回复和 observation 追加到 messages
            messages.append({"role": "assistant", "content": response})

            # 在接近步数上限时注入催促信息
            remaining_steps = self.max_steps - step_idx - 1
            urgency_note = ""
            if remaining_steps == 3:
                urgency_note = "\n\n[NOTICE: You have only 3 steps remaining. Start concluding your research and prepare your Final Answer.]"
            elif remaining_steps == 1:
                urgency_note = "\n\n[WARNING: This is your LAST step. You MUST output 'Final Answer: <answer>' now.]"

            messages.append({
                "role": "user",
                "content": f"Observation: {observation}{urgency_note}",
            })

            step.elapsed_ms = (time.time() - step_start) * 1000
            episode.steps.append(step)

            logger.info(
                "[Step %d] %s(%s) → %s (%.0f ms)",
                step_idx, tool_name,
                ", ".join(f"{k}={v[:30]}..." if len(v) > 30 else f"{k}={v}" for k, v in tool_kwargs.items()),
                "ok" if tool_result.success else "FAIL",
                step.elapsed_ms,
            )

        # 如果循环结束还没有 Final Answer，尝试强制生成
        if not episode.final_answer:
            episode.final_answer = self._force_final_answer(messages)

        episode.total_steps = len(episode.steps)
        episode.total_elapsed_ms = (time.time() - episode_start) * 1000

        # 评估正确性（简单字符串匹配）
        if ground_truth:
            episode.correct = self._check_answer(episode.final_answer, ground_truth)

        logger.info(
            "Episode %s: %d steps, answer='%s', correct=%s (%.1f s)",
            task_id, episode.total_steps, episode.final_answer[:50],
            episode.correct, episode.total_elapsed_ms / 1000,
        )

        return episode

    # ── 消息构建 ─────────────────────────────────────────

    def _build_user_message(
        self, question: str, image_path: str
    ) -> list[dict[str, Any]] | str:
        """构建 user message（文本 + 可选图片）。"""
        text = USER_PROMPT_TEMPLATE.format(question=question)

        if not image_path or not Path(image_path).exists():
            return text

        data_url = VLMClient.encode_image_to_data_url(image_path)
        if not data_url:
            return text

        return [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": text},
        ]

    # ── 响应解析 ─────────────────────────────────────────

    @staticmethod
    def _parse_response(response: str) -> tuple[str, str]:
        """从 LLM 输出中提取 Thought 和 Action。

        Returns:
            (thought, action_text)  action_text 为空表示没有 Action
        """
        thought = ""
        action = ""

        # 提取 Thought
        thought_match = re.search(
            r'Thought:\s*(.*?)(?=Action:|Final Answer:|$)',
            response, re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # 提取 Action
        action_match = re.search(
            r'Action:\s*(.*?)(?=Observation:|Thought:|Final Answer:|$)',
            response, re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            action = action_match.group(1).strip()

        return thought, action

    @staticmethod
    def _extract_final_answer(response: str) -> str:
        """提取 Final Answer。"""
        match = re.search(
            r'Final Answer:\s*(.*?)$',
            response, re.DOTALL | re.IGNORECASE,
        )
        if match:
            answer = match.group(1).strip()
            # 去掉可能的尾部 Thought/Action（模型有时会在 Final Answer 后继续）
            for stop_token in ["Thought:", "Action:"]:
                idx = answer.find(stop_token)
                if idx > 0:
                    answer = answer[:idx].strip()
            return answer
        return ""

    def _force_final_answer(self, messages: list[dict[str, Any]]) -> str:
        """如果 Agent 用完步数还没给出 Final Answer，强制生成。"""
        messages.append({
            "role": "user",
            "content": (
                "You have used all available steps. You MUST provide your Final Answer NOW. "
                "Based on everything you've found, what is the most likely answer? "
                "Output exactly: Final Answer: <your answer>"
            ),
        })
        try:
            response = self.vlm.chat(messages=messages)
            answer = self._extract_final_answer(response)
            if answer:
                return self._clean_answer(answer)
            # 如果还是没有格式化的答案，尝试清洗后返回
            return self._clean_answer(response.strip())
        except Exception as exc:
            logger.error("Force final answer failed: %s", exc)
            return ""

    @staticmethod
    def _clean_answer(answer: str) -> str:
        """清洗 answer 字符串，去掉 Thought: 前缀和多余内容。"""
        # 去掉开头的 "Thought: ..." 前缀
        thought_match = re.match(
            r'^Thought:.*?(?:Final Answer:|$)', answer, re.DOTALL | re.IGNORECASE
        )
        if thought_match and "Final Answer:" in answer:
            fa_idx = answer.lower().index("final answer:")
            answer = answer[fa_idx + len("final answer:"):].strip()
        elif answer.lower().startswith("thought:"):
            # 没有 Final Answer，尝试提取最后一句有意义的内容
            lines = [l.strip() for l in answer.split('\n') if l.strip()]
            # 找最后一个不以 "Thought:" 开头的有意义行
            for line in reversed(lines):
                if not line.lower().startswith("thought:") and not line.lower().startswith("action:"):
                    answer = line
                    break
        # 去掉尾部的 Thought:/Action:
        for stop_token in ["Thought:", "Action:", "\n"]:
            idx = answer.find(stop_token)
            if idx > 0:
                answer = answer[:idx].strip()
        return answer.strip()

    # ── 答案评估 ─────────────────────────────────────────

    @staticmethod
    def _check_answer(predicted: str, ground_truth: str) -> bool:
        """简单的答案匹配检查。

        支持:
          - 精确匹配（忽略大小写）
          - ground_truth 包含在 predicted 中
          - predicted 包含在 ground_truth 中
        """
        predicted_lower = predicted.lower().strip()
        ground_truth_lower = ground_truth.lower().strip()

        if not predicted_lower:
            return False

        # 精确匹配
        if predicted_lower == ground_truth_lower:
            return True

        # 子串匹配
        if ground_truth_lower in predicted_lower:
            return True
        if predicted_lower in ground_truth_lower:
            return True

        return False


# ═══════════════════════════════════════════════════════════
# 5. 便捷入口
# ═══════════════════════════════════════════════════════════

def create_vdr_agent(
    rag_engine: Any,
    images_dir: str | Path = "/dev/shm/oven_wiki/images",
    vlm_base_url: str = "http://localhost:8000/v1",
    vlm_model: str = "/dev/shm/Qwen3.6-27B",
    max_steps: int = 50,
    search_top_k: int = 10,
    enable_sliding_window: bool = False,
) -> VDRAgent:
    """创建完整的 VDR Agent 实例。

    Args:
        rag_engine: 已 load() 的 RagEngine 实例
        images_dir: 图片目录
        vlm_base_url: vLLM API 地址
        vlm_model: 模型名
        max_steps: 最大步数
        search_top_k: 每次 search 返回的结果数
        enable_sliding_window: 是否启用滑动窗口 context 管理

    Example::

        from searcheyes.rag_engine import RagEngine
        from searcheyes.vdr_agent import create_vdr_agent

        rag = RagEngine(wiki6m_path="/dev/shm/oven_wiki/Wiki6M_ver_1_0.jsonl")
        rag.load()

        agent = create_vdr_agent(rag)
        result = agent.run_episode(
            question="What is the name of this building?",
            image_path="/path/to/query_image.jpg",
            task_id="test_001",
            ground_truth="Eiffel Tower",
        )
        print(result.final_answer)
    """
    toolkit = VDRToolKit(
        rag_engine=rag_engine,
        images_dir=images_dir,
        search_top_k=search_top_k,
    )
    vlm_client = VLMClient(
        base_url=vlm_base_url,
        model=vlm_model,
    )
    return VDRAgent(
        toolkit=toolkit,
        vlm_client=vlm_client,
        max_steps=max_steps,
        enable_sliding_window=enable_sliding_window,
    )
