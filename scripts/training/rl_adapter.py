"""
rl_adapter.py — P4 verl-compatible RL 环境封装
=================================================
将 searcheyes 的 task / reward / obs 封装为标准 RL 接口。
预埋 GRPO / verl 训练级别的接口，本阶段不执行真实训练。
"""

from __future__ import annotations
import asyncio
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from searcheyes.screen_ir import ScreenIR, StyleBundle
from searcheyes.template_renderer import TemplateRenderer
from searcheyes.transition_engine import TransitionEngine, EnvState
from searcheyes.state_diff import StateDiff
from searcheyes.validator import Validator
from searcheyes.task_schema import TrajectoryStep, VisualTask


@dataclass
class Observation:
    """Agent 观察到的信息"""
    screenshot_path: str = ""
    state_description: str = ""
    available_actions: list[str] = field(default_factory=list)
    step_idx: int = 0


@dataclass
class StepResult:
    """一步交互的结果"""
    obs: Observation
    reward: float = 0.0
    rag_reward: float = 0.0   # RAG 命中率奖励（独立记录）
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class TransitionRecord:
    """单步训练样本记录。"""

    task_id: str
    step_idx: int
    prompt: dict[str, Any] = field(default_factory=dict)
    chosen_action: str = ""
    chosen_params: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    expected_action: str = ""
    expected_params: dict[str, Any] = field(default_factory=dict)
    validation: str = ""
    diff: str = ""
    state_hash: str = ""
    next_screenshot_path: str = ""


class RLEnvironment:
    """
    verl-compatible RL 环境。
    接口设计对齐 OpenAI Gym / verl 标准。

    search 动作通过 RAG 检索 WIT 知识库填充 products（RAG = 搜索引擎）。
    reward 以结果导向为主（buy 正确 wit_id 产品 = 高奖励）。
    """

    def __init__(self, task: VisualTask, rag: Any = None, images_dir: str | Path | None = None,
                 inject_ground_truth: bool = True):
        self.task = task
        self.rag = rag
        self.images_dir = Path(images_dir) if images_dir else None
        self._inject_ground_truth = inject_ground_truth
        gt_wit_id = getattr(task, "ground_truth_wit_id", "") or ""
        self.engine = TransitionEngine(
            rag=rag, images_dir=self.images_dir, ground_truth_wit_id=gt_wit_id,
            inject_ground_truth=inject_ground_truth,
        )
        self.validator = Validator(self.engine)
        self.renderer = TemplateRenderer()
        self.state: EnvState = EnvState()
        self.step_count = 0
        self.max_steps = task.dag_depth * 2  # 允许的最大步数
        self.last_screen_ir: ScreenIR | None = None
        self.style_bundle: StyleBundle | None = None
        self.output_root = Path(__file__).parent.parent / "output" / "rl_rollouts"
        self.episode_dir: Path | None = None
        self.history: list[TransitionRecord] = []

    def reset(self) -> Observation:
        """重置环境到初始状态"""
        self.state = EnvState()
        gt_wit_id = getattr(self.task, "ground_truth_wit_id", "") or ""
        self.engine = TransitionEngine(
            rag=self.rag, images_dir=self.images_dir, ground_truth_wit_id=gt_wit_id,
            inject_ground_truth=self._inject_ground_truth,
        )
        self.step_count = 0
        self.history = []
        self.episode_dir = self._make_episode_dir()
        self.style_bundle = self.renderer.pick_style_bundle(self.task.task_id, self.state.page_family)
        self.last_screen_ir = self._render_current_state(step_idx=0)
        return Observation(
            screenshot_path=self.last_screen_ir.screenshot_path if self.last_screen_ir else "",
            state_description=f"page={self.state.current_page}",
            available_actions=["search", "zoom"],
            step_idx=0
        )

    def step(self, action: str, params: dict = None) -> StepResult:
        """执行一步交互。

        search 动作自动注入 query_image 用于 RAG 检索。
        reward 以结果导向为主（wit_id 匹配）。
        """
        params = params or {}
        pre_action_obs = Observation(
            screenshot_path=self.last_screen_ir.screenshot_path if self.last_screen_ir else "",
            state_description=f"page={self.state.current_page}",
            available_actions=self._get_available_actions(),
            step_idx=self.step_count,
        )

        # search 动作：传截图和任务目标给 RAG 搜索引擎
        if action == "search" and self.last_screen_ir:
            params.setdefault("query_image", self.last_screen_ir.screenshot_path)
            params.setdefault("query_text", self.task.goal)

        prev_state = self.state
        self.step_count += 1
        new_state, diff = self.engine.step(self.state, action, params)
        vr = self.validator.validate(new_state, diff)

        # 结果导向 reward
        reward = self._compute_reward(action, diff, new_state, vr.passed)
        total_reward = max(-2.0, min(3.0, reward))

        # 检查终止
        done = (
            self.step_count >= self.max_steps or
            action == "buy" or
            not vr.passed
        )

        self.state = new_state
        self.last_screen_ir = self._render_current_state(step_idx=self.step_count)

        success = self._is_terminal_success(action)
        self.history.append(
            TransitionRecord(
                task_id=self.task.task_id,
                step_idx=self.step_count - 1,
                prompt=self._build_prompt_payload(pre_action_obs),
                chosen_action=action,
                chosen_params={k: v for k, v in params.items() if k not in ("query_image", "query_text")},
                reward=total_reward,
                done=done,
                expected_action="",
                expected_params={},
                validation=vr.summary(),
                diff=diff.summary(),
                state_hash=new_state.hash(),
                next_screenshot_path=self.last_screen_ir.screenshot_path if self.last_screen_ir else "",
            )
        )

        return StepResult(
            obs=Observation(
                screenshot_path=self.last_screen_ir.screenshot_path if self.last_screen_ir else "",
                state_description=f"page={new_state.current_page}",
                available_actions=self._get_available_actions(),
                step_idx=self.step_count
            ),
            reward=total_reward,
            rag_reward=0.0,
            done=done,
            info={
                "diff": diff.summary(),
                "validation": vr.summary(),
                "state_hash": new_state.hash(),
                "screen_ir_path": str(self._current_frame_dir(self.step_count) / "screen_ir.json"),
                "success": success,
            }
        )

    def _compute_reward(
        self,
        action: str,
        diff: StateDiff,
        new_state: EnvState,
        validation_passed: bool,
    ) -> float:
        """结果导向 reward：buy 正确 wit_id 产品 = 高奖励。

        过程塑形信号帮助 Agent 学会 search → browse → zoom → buy 的流程。
        """
        reward = -0.05  # 每步微惩罚，鼓励效率

        if not validation_passed:
            reward -= 0.8
            return reward

        gt_wit_id = getattr(self.task, "ground_truth_wit_id", "") or ""

        # ── 过程塑形 ──
        if action == "search":
            reward += 0.1  # 搜索是必要的第一步
            # 检查 GT 是否出现在 RAG 结果中
            if gt_wit_id and self.engine.products:
                gt_in_results = any(
                    p.get("wit_id") == gt_wit_id for p in self.engine.products.values()
                )
                reward += 0.3 if gt_in_results else -0.2

        elif action == "click_product":
            pid = new_state.selected_product_id
            product = self.engine.products.get(pid, {})
            if gt_wit_id and product.get("wit_id") == gt_wit_id:
                reward += 0.3  # 选对了
            else:
                reward -= 0.1  # 选错了

        elif action == "zoom":
            # zoom GT 产品图片给奖励
            pid = new_state.selected_product_id
            if pid:
                product = self.engine.products.get(pid, {})
                if gt_wit_id and product.get("wit_id") == gt_wit_id:
                    reward += 0.2

        elif action == "buy":
            if self._is_terminal_success(action):
                reward += 2.0  # 买对了！
            elif diff.has_data_change:
                reward -= 1.0  # 买错了
            else:
                reward -= 0.5  # buy 但没有数据变化

        elif action == "add_cart":
            pid = new_state.selected_product_id
            product = self.engine.products.get(pid, {})
            if gt_wit_id and product.get("wit_id") == gt_wit_id:
                reward += 0.5
            else:
                reward -= 0.3

        elif action == "back":
            reward -= 0.1  # 回退轻微惩罚

        return reward

    def _get_available_actions(self) -> list[str]:
        if self.state.page_family.value == "search":
            return ["search", "zoom"]
        elif self.state.page_family.value == "results":
            return ["click_product", "zoom", "toggle_dropdown", "apply_filter", "back"]
        elif self.state.page_family.value == "detail":
            return ["buy", "add_cart", "zoom", "open_modal", "back"]
        return ["back"]

    # ── verl 兼容接口 ──────────────────────────────────

    def get_obs_for_verl(self) -> dict[str, Any]:
        """导出为 verl 标准 observation dict"""
        return {
            "image": self.last_screen_ir.screenshot_path if self.last_screen_ir else "",
            "text": self.task.goal,
            "state": self.state.hash(),
            "step": self.step_count,
            "available_actions": self._get_available_actions(),
        }

    def get_reward_for_verl(self, action_output: str) -> float:
        """verl 回调: 接收模型输出，返回 reward"""
        try:
            payload = json.loads(action_output)
            action, params = self._decode_action_output(payload)
            if not action:
                return -1.0
            result = self.step(action, params)
            return result.reward
        except Exception:
            return -1.0

    def export_episode_jsonl(self, output_path: str | Path | None = None, verl_format: bool = False) -> Path:
        """导出当前 rollout 为 verl/GRPO 可消费的 step-level JSONL。"""
        if self.episode_dir is None:
            self.episode_dir = self._make_episode_dir()
        output_path = Path(output_path or (self.episode_dir / "train_samples.jsonl"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for record in self.history:
                item = {
                    "task_id": record.task_id,
                    "step_idx": record.step_idx,
                    "prompt": self._to_verl_messages(record.prompt) if verl_format else record.prompt,
                    "chosen": json.dumps(
                        {"action": record.chosen_action, "params": record.chosen_params},
                        ensure_ascii=False,
                    ),
                    "reward": record.reward,
                    "done": record.done,
                    "expected": {
                        "action": record.expected_action,
                        "params": record.expected_params,
                    },
                    "metadata": {
                        "validation": record.validation,
                        "diff": record.diff,
                        "state_hash": record.state_hash,
                        "next_screenshot_path": record.next_screenshot_path,
                    },
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return output_path

    def _make_episode_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = self.output_root / f"{self.task.task_id}_{stamp}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir

    def _current_frame_dir(self, step_idx: int) -> Path:
        if self.episode_dir is None:
            self.episode_dir = self._make_episode_dir()
        frame_dir = self.episode_dir / f"step_{step_idx:02d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        return frame_dir

    def _render_current_state(self, step_idx: int) -> ScreenIR:
        frame_dir = self._current_frame_dir(step_idx)
        return asyncio.run(
            self.renderer.render_to_screen_ir(
                self.state,
                self.engine.products,
                page_id=f"rl_step_{step_idx:02d}",
                output_dir=frame_dir,
                style_bundle=self.style_bundle,
            )
        )

    def _is_terminal_success(self, action: str) -> bool:
        """Agent 是否成功完成任务：buy 了 ground_truth_wit_id 对应的产品。"""
        if action != "buy":
            return False
        pid = self.state.selected_product_id
        if pid is None:
            return False
        product = self.engine.products.get(pid, {})
        gt_wit_id = getattr(self.task, "ground_truth_wit_id", "") or ""
        if not gt_wit_id:
            return True  # 没有 GT wit_id 时，只要 buy 就算成功
        return product.get("wit_id", "") == gt_wit_id

    def _build_prompt_payload(self, obs: Observation) -> dict[str, Any]:
        return {
            "goal": self.task.goal,
            "image": obs.screenshot_path,
            "state_description": obs.state_description,
            "available_actions": list(obs.available_actions),
            "prompt_text": (
                f"Goal: {self.task.goal}\n"
                f"State: {obs.state_description}\n"
                f"Available actions: {', '.join(obs.available_actions)}"
            ),
        }

    def _to_verl_messages(self, prompt: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": prompt["image"]}},
                    {"type": "text", "text": prompt["prompt_text"]},
                ],
            }
        ]

    def _decode_action_output(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        action = str(payload.get("action", "")).strip()
        params = payload.get("params", {}) or {}
        if action:
            return action, params

        option_id = str(payload.get("option_id", "")).strip()
        if not option_id:
            return "", {}

        if option_id in {"search", "back", "buy", "add_cart"}:
            return option_id, {}
        if option_id == "toggle_sort":
            return "toggle_dropdown", {"dropdown_name": "sort"}
        if option_id == "zoom_detail_price":
            return "zoom", {"target": "detail_price"}

        match = re.fullmatch(r"click_product_(\d+)", option_id)
        if match:
            return "click_product", {"product_id": int(match.group(1))}

        match = re.fullmatch(r"zoom_search_(\d+)", option_id)
        if match:
            return "zoom", {"target": "search_input"}

        match = re.fullmatch(r"zoom_price_(\d+)", option_id)
        if match:
            product_id = int(match.group(1))
            visible = self._visible_products()
            for idx, product in enumerate(visible):
                if product["id"] == product_id:
                    return "zoom", {"target": f"price_tag_{idx}"}
            return "zoom", {"product_id": product_id}

        return "", {}

    def _visible_products(self) -> list[dict[str, Any]]:
        products = list(self.engine.products.values())
        for key, value in self.state.active_filter.items():
            products = [p for p in products if str(p.get(key, "")).lower() == str(value).lower()]
        return products
