"""
rl_adapter.py — P4 verl-compatible RL 环境封装 (v2 Search/Report World)
========================================================================
将 searcheyes 的 task / reward / obs 封装为标准 RL 接口。

v2: search/report world — Agent 执行 search -> browse -> cite -> submit_report。
reward: placeholder only (学长指示暂停 reward 设计)。
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from searcheyes.screen_ir import ScreenIR, StyleBundle, PageFamily
from searcheyes.template_renderer import TemplateRenderer
from searcheyes.transition_engine import TransitionEngine
from searcheyes.research_contracts import (
    ResearchState, ResearchPhase, ResearchTask, FactSet, CitationObject,
)
from searcheyes.state_diff import StateDiff
from searcheyes.validator import Validator


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
    validation: str = ""
    diff: str = ""
    state_hash: str = ""
    next_screenshot_path: str = ""


class RLEnvironment:
    """
    verl-compatible RL 环境 (v2 Search/Report World).

    search 动作通过 RAG 检索 WIT 知识库（RAG = 搜索引擎）。
    reward: placeholder (per-step penalty + submit bonus * coverage)。
    """

    def __init__(self, task: ResearchTask, rag: Any = None,
                 images_dir: str | Path | None = None,
                 inject_ground_truth: bool = True,
                 search_controller: Any = None):
        # Feature flag: zoom_search action (default off to preserve existing eval baseline)
        self.enable_zoom_search = False
        self.task = task
        self.rag = rag
        self.images_dir = Path(images_dir) if images_dir else None
        self._inject_ground_truth = inject_ground_truth
        self._search_controller = search_controller
        gt_wit_id = task.ground_truth_wit_id or ""
        gt_seed = int(hashlib.md5(task.task_id.encode()).hexdigest()[:8], 16)
        self.engine = TransitionEngine(
            rag=rag, images_dir=self.images_dir, ground_truth_wit_id=gt_wit_id,
            inject_ground_truth=inject_ground_truth,
            search_controller=search_controller,
            gt_inject_seed=gt_seed,
        )
        self.validator = Validator(self.engine)
        self.renderer = TemplateRenderer()
        self.state: ResearchState = ResearchState()
        self.step_count = 0
        self.max_steps = task.max_steps
        self.last_screen_ir: ScreenIR | None = None
        self.style_bundle: StyleBundle | None = None
        self.output_root = Path(__file__).parent.parent / "output" / "rl_rollouts"
        self.episode_dir: Path | None = None
        self.history: list[TransitionRecord] = []

    def reset(self) -> Observation:
        self.state = ResearchState()
        gt_wit_id = self.task.ground_truth_wit_id or ""
        gt_seed = int(hashlib.md5(self.task.task_id.encode()).hexdigest()[:8], 16)
        self.engine = TransitionEngine(
            rag=self.rag, images_dir=self.images_dir, ground_truth_wit_id=gt_wit_id,
            inject_ground_truth=self._inject_ground_truth,
            search_controller=self._search_controller,
            gt_inject_seed=gt_seed,
        )
        self.validator = Validator(self.engine)
        self.step_count = 0
        self.history = []
        self.episode_dir = self._make_episode_dir()
        self.style_bundle = self.renderer.pick_style_bundle(
            self.task.task_id, PageFamily.SEARCH
        )
        self.last_screen_ir = self._render_current_state(step_idx=0)
        return Observation(
            screenshot_path=self.last_screen_ir.screenshot_path if self.last_screen_ir else "",
            state_description=self._build_state_description(self.state),
            available_actions=self._get_available_actions(),
            step_idx=0,
        )

    def step(self, action: str, params: dict = None) -> StepResult:
        params = params or {}
        pre_action_obs = Observation(
            screenshot_path=self.last_screen_ir.screenshot_path if self.last_screen_ir else "",
            state_description=self._build_state_description(self.state),
            available_actions=self._get_available_actions(),
            step_idx=self.step_count,
        )

        # search: inject query from task goal if not provided
        if action == "search" and self.last_screen_ir:
            params.setdefault("query_image", self.last_screen_ir.screenshot_path)
            params.setdefault("query_text", self.task.goal)

        self.step_count += 1
        new_state, diff = self.engine.step(self.state, action, params)
        vr = self.validator.validate(new_state, diff)

        reward = self._compute_reward(action, diff, new_state, vr.passed)
        total_reward = max(-2.0, min(3.0, reward))

        done = (
            self.step_count >= self.max_steps or
            action == "submit_report" or
            not vr.passed
        )

        self.state = new_state
        self.last_screen_ir = self._render_current_state(step_idx=self.step_count)

        success = self._is_terminal_success(action)
        self.history.append(TransitionRecord(
            task_id=self.task.task_id,
            step_idx=self.step_count - 1,
            prompt=self._build_prompt_payload(pre_action_obs),
            chosen_action=action,
            chosen_params={k: v for k, v in params.items() if k not in ("query_image",)},
            reward=total_reward,
            done=done,
            validation=vr.summary(),
            diff=diff.summary(),
            state_hash=new_state.hash(),
            next_screenshot_path=self.last_screen_ir.screenshot_path if self.last_screen_ir else "",
        ))

        return StepResult(
            obs=Observation(
                screenshot_path=self.last_screen_ir.screenshot_path if self.last_screen_ir else "",
                state_description=self._build_state_description(new_state),
                available_actions=self._get_available_actions(),
                step_idx=self.step_count,
            ),
            reward=total_reward,
            done=done,
            info=self._build_info(action, diff, vr, new_state, success),
        )

    def _build_state_description(self, state: ResearchState) -> str:
        """Rich state description — sufficient information for model decision-making."""
        lines = []
        lines.append("phase=" + state.current_phase.value)
        lines.append("page=" + state.current_page)
        lines.append("queries=" + str(len(state.query_history)))
        if state.query_history:
            lines.append("last_query=" + state.query_history[-1][:60])
        lines.append("results=" + str(len(state.current_result_ids)))
        lines.append("citations=" + str(state.citation_count))
        if state.opened_result_id is not None:
            lines.append("viewing_doc=" + str(state.opened_result_id))

        # Include search results summary if on results page
        if state.current_phase == ResearchPhase.SEARCH and self.engine.search_results:
            lines.append("Search results:")
            for sr in self.engine.search_results:
                lines.append("  [" + str(sr.result_id) + "] " + sr.title + " | " + sr.display_snippet)

        # Include document info if browsing
        if state.current_phase == ResearchPhase.BROWSE and self.engine.current_document:
            doc = self.engine.current_document
            lines.append("Document: " + doc.title)
            lines.append("Content: " + doc.body_text[:200])

        # Include collected evidence summary
        if self.engine.citations:
            lines.append("Collected evidence:")
            for c in self.engine.citations:
                lines.append("  [" + str(c.citation_id) + "] from " + c.source_title + ": " + c.evidence_text[:60])

        return "\n".join(lines)

    def _compute_reward(self, action: str, diff: StateDiff, new_state: ResearchState,
                        validation_passed: bool) -> float:
        """Fact-coverage based reward for search/report world.

        Reward signal:
        - per-step penalty: encourages efficiency
        - cite_source: higher reward if cited document matches GT facts
        - submit_report: coverage-proportional bonus
        - refine_query: small reward for exploration
        """
        reward = -0.02  # per-step efficiency penalty

        if not validation_passed:
            reward -= 0.5
            return reward

        if action == "search":
            # Base reward for searching; slightly higher if GT appears in results
            reward += 0.05
            gt_wit_id = self.task.ground_truth_wit_id or ""
            if gt_wit_id and any(sr.wit_id == gt_wit_id for sr in self.engine.search_results):
                reward += 0.1  # GT retrieved successfully

        elif action == "cite_source":
            # Reward based on whether cited document is relevant to GT
            latest_citation = self.engine.citations[-1] if self.engine.citations else None
            if latest_citation and self.task.fact_set.is_relevant_citation(latest_citation):
                reward += 0.3  # meaningful evidence collected
            else:
                reward += 0.05  # citation exists but not GT-relevant

        elif action == "refine_query":
            reward += 0.03  # small exploration bonus

        elif action == "open_result":
            # Reward for opening GT document
            opened_id = new_state.opened_result_id
            if opened_id is not None:
                opened_sr = self.engine._find_result(opened_id)
                gt_wit_id = self.task.ground_truth_wit_id or ""
                if opened_sr and opened_sr.wit_id == gt_wit_id:
                    reward += 0.2  # opened the right document

        elif action == "submit_report":
            coverage = self.task.fact_set.coverage(self.engine.citations)
            reward += 2.0 * coverage
            if new_state.citation_count < self.task.min_citations_for_submit:
                reward -= 1.0

        return reward

    def _get_available_actions(self) -> list[str]:
        phase = self.state.current_phase
        if phase == ResearchPhase.FORMULATE:
            return ["search"]
        elif phase == ResearchPhase.SEARCH:
            actions = ["open_result", "refine_query"]
            if self.enable_zoom_search and self.state.current_result_ids:
                actions.append("zoom_search")
            if self.state.citation_count >= self.task.min_citations_for_submit:
                actions.append("submit_report")
            return actions
        elif phase == ResearchPhase.BROWSE:
            actions = ["cite_source", "back_to_results"]
            if self.enable_zoom_search and self.state.current_result_ids:
                actions.append("zoom_search")
            if self.state.citation_count >= self.task.min_citations_for_submit:
                actions.append("submit_report")
            return actions
        return ["submit_report"]

    def _is_terminal_success(self, action: str) -> bool:
        if action != "submit_report":
            return False
        coverage = self.task.fact_set.coverage(self.engine.citations)
        return coverage >= 0.5  # at least half of primary facts covered

    def _build_info(self, action: str, diff: StateDiff, vr: Any,
                    new_state: ResearchState, success: bool) -> dict:
        info: dict[str, Any] = {
            "diff": diff.summary(),
            "validation": vr.summary(),
            "state_hash": new_state.hash(),
            "success": success,
            "coverage": self.task.fact_set.coverage(self.engine.citations),
        }
        if action == "search" and self.engine.last_search_result is not None:
            sr = self.engine.last_search_result
            info["search_controller"] = {
                "retry_triggered": sr.retry_triggered,
                "accepted_strategy": sr.accepted_strategy,
                "num_attempts": len(sr.attempts),
                "top1_score": sr.facts[0].score if sr.facts else 0.0,
            }
        return info

    def _make_episode_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = self.output_root / (self.task.task_id + "_" + stamp)
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir

    def _current_frame_dir(self, step_idx: int) -> Path:
        if self.episode_dir is None:
            self.episode_dir = self._make_episode_dir()
        frame_dir = self.episode_dir / ("step_" + str(step_idx).zfill(2))
        frame_dir.mkdir(parents=True, exist_ok=True)
        return frame_dir

    def _render_current_state(self, step_idx: int) -> ScreenIR:
        frame_dir = self._current_frame_dir(step_idx)
        return asyncio.run(
            self.renderer.render_to_screen_ir(
                self.state,
                self.engine.products,  # v1 compat property
                page_id="rl_step_" + str(step_idx).zfill(2),
                output_dir=frame_dir,
                style_bundle=self.style_bundle,
            )
        )

    def _build_prompt_payload(self, obs: Observation) -> dict[str, Any]:
        return {
            "goal": self.task.goal,
            "image": obs.screenshot_path,
            "state_description": obs.state_description,
            "available_actions": list(obs.available_actions),
            "prompt_text": (
                "Goal: " + self.task.goal + "\n" +
                "State: " + obs.state_description + "\n" +
                "Available actions: " + ", ".join(obs.available_actions)
            ),
        }

    # ── verl compat ──
    def get_obs_for_verl(self) -> dict[str, Any]:
        return {
            "image": self.last_screen_ir.screenshot_path if self.last_screen_ir else "",
            "text": self.task.goal,
            "state": self.state.hash(),
            "step": self.step_count,
            "available_actions": self._get_available_actions(),
        }

    def export_episode_jsonl(self, output_path: str | Path | None = None,
                             verl_format: bool = False) -> Path:
        if self.episode_dir is None:
            self.episode_dir = self._make_episode_dir()
        output_path = Path(output_path or (self.episode_dir / "train_samples.jsonl"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in self.history:
                item = {
                    "task_id": record.task_id,
                    "step_idx": record.step_idx,
                    "prompt": record.prompt,
                    "chosen": json.dumps(
                        {"action": record.chosen_action, "params": record.chosen_params},
                        ensure_ascii=False,
                    ),
                    "reward": record.reward,
                    "done": record.done,
                    "metadata": {
                        "validation": record.validation,
                        "diff": record.diff,
                        "state_hash": record.state_hash,
                    },
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return output_path
