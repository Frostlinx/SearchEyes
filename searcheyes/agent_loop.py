"""
agent_loop.py — v2 Research Agent Loop
========================================
把渲染、决策、状态转移、校验串成可跑的 episode。
v2: search/report world — search -> browse -> cite -> submit_report。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from searcheyes.screen_ir import ElementSemanticType, ScreenIR, StyleBundle, PageFamily
from searcheyes.research_contracts import (
    ResearchTask, ResearchState, ResearchPhase,
)
from searcheyes.template_renderer import TemplateRenderer
from searcheyes.transition_engine import TransitionEngine
from searcheyes.validator import Validator
from searcheyes.vlm_agent import (
    ActionDecision,
    ActionOption,
    DecisionBackend,
    DecisionContext,
    serialize_context,
)


@dataclass
class EpisodeStep:
    step_idx: int
    screenshot_path: str
    focused_screenshot_path: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    state_before: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    validation: str = ""
    diff: str = ""
    state_after: dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    def __init__(
        self,
        backend: DecisionBackend,
        task: ResearchTask | None = None,
        output_root: str | Path | None = None,
        max_steps: int | None = None,
        rag: Any | None = None,
        images_dir: str | Path | None = None,
        search_controller: Any | None = None,
    ):
        self.backend = backend
        self.task = task
        self.output_root = Path(output_root or Path(__file__).parent.parent / "output" / "agent_loops")
        self.max_steps = max_steps or (task.max_steps if task else 12)
        gt_wit_id = ""
        if task:
            gt_wit_id = task.ground_truth_wit_id or ""
        self.engine = TransitionEngine(
            rag=rag, images_dir=images_dir, ground_truth_wit_id=gt_wit_id,
            search_controller=search_controller,
        )
        self.validator = Validator(self.engine)
        self.renderer = TemplateRenderer()
        self.state = ResearchState()
        self.style_bundle: StyleBundle | None = None
        self.rag = rag

    async def run(self) -> dict[str, Any]:
        episode_dir = self._make_episode_dir()
        logs: list[EpisodeStep] = []
        focused_screenshot_path = ""
        self.style_bundle = self.renderer.pick_style_bundle(
            self.task.task_id if self.task else "adhoc",
            self.state.page_family,
        )

        for step_idx in range(self.max_steps):
            frame_dir = episode_dir / ("step_" + str(step_idx).zfill(2))
            frame_dir.mkdir(parents=True, exist_ok=True)
            screen_ir = await self.renderer.render_to_screen_ir(
                self.state,
                self.engine.products,
                page_id="step_" + str(step_idx).zfill(2),
                output_dir=frame_dir,
                style_bundle=self.style_bundle,
            )
            options = self._build_action_options(screen_ir)
            if not options:
                break

            # Build context for VLM decision
            rag_facts_text: list[str] = []
            for sr in self.engine.search_results:
                rag_facts_text.append(sr.title + ": " + sr.snippet[:60])

            context = DecisionContext(
                task_goal=self.task.goal if self.task else "",
                screenshot_path=screen_ir.screenshot_path,
                focused_screenshot_path=focused_screenshot_path,
                state_summary=self._summarize_state(screen_ir),
                ui_tokens=screen_ir.ui_tokens,
                options=options,
                rag_facts=rag_facts_text,
            )
            decision = self.backend.decide(context)
            option = self._resolve_option(options, decision)

            state_before = {
                "phase": self.state.current_phase.value,
                "page": self.state.current_page,
                "citations": self.state.citation_count,
                "queries": len(self.state.query_history),
            }
            focused_screenshot_path = ""

            # search: inject query from task goal
            if option.action == "search":
                option.params["query_image"] = screen_ir.screenshot_path
                if self.task:
                    option.params["query_text"] = self.task.goal

            if option.action == "zoom" and option.bbox:
                zoom_path = frame_dir / "focused.png"
                await self.renderer.render_zoom_screenshot(
                    self.state,
                    self.engine.products,
                    option.bbox,
                    output_path=zoom_path,
                    style_bundle=self.style_bundle,
                )
                focused_screenshot_path = str(zoom_path)

            new_state, diff = self.engine.step(self.state, option.action, option.params)
            validation = self.validator.validate(new_state, diff)
            self.state = new_state

            logs.append(
                EpisodeStep(
                    step_idx=step_idx,
                    screenshot_path=screen_ir.screenshot_path,
                    focused_screenshot_path=focused_screenshot_path,
                    context=serialize_context(context),
                    state_before=state_before,
                    decision=asdict(decision),
                    action=asdict(option),
                    validation=validation.summary(),
                    diff=diff.summary(),
                    state_after={
                        "phase": new_state.current_phase.value,
                        "page": new_state.current_page,
                        "citations": new_state.citation_count,
                    },
                )
            )

            if not validation.passed or self._is_terminal(option.action):
                break

        summary = {
            "task_id": self.task.task_id if self.task else "adhoc",
            "steps": [asdict(step) for step in logs],
            "final_state": {
                "phase": self.state.current_phase.value,
                "page": self.state.current_page,
                "citations": self.state.citation_count,
                "submitted": self.state.report_submitted,
            },
            "episode_dir": str(episode_dir),
        }
        (episode_dir / "trajectory.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def _make_episode_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = self.task.task_id if self.task else "adhoc"
        episode_dir = self.output_root / (task_id + "_" + stamp)
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir

    def _summarize_state(self, screen_ir: ScreenIR) -> str:
        return (
            "phase=" + self.state.current_phase.value
            + " page=" + self.state.current_page
            + " queries=" + str(len(self.state.query_history))
            + " citations=" + str(self.state.citation_count)
            + " elements=" + str(screen_ir.clickable_count)
        )

    def _resolve_option(self, options: list[ActionOption], decision: ActionDecision) -> ActionOption:
        for option in options:
            if option.option_id == decision.option_id:
                return option
        raise ValueError("Cannot find option_id=" + decision.option_id)

    def _build_action_options(self, screen_ir: ScreenIR) -> list[ActionOption]:
        phase = self.state.current_phase
        if phase == ResearchPhase.FORMULATE:
            return self._formulate_options(screen_ir)
        elif phase == ResearchPhase.SEARCH:
            return self._search_results_options(screen_ir)
        elif phase == ResearchPhase.BROWSE:
            return self._document_options(screen_ir)
        elif phase == ResearchPhase.SYNTHESIZE:
            return []  # terminal
        return [ActionOption(option_id="search", action="search", description="Submit search")]

    def _formulate_options(self, screen_ir: ScreenIR) -> list[ActionOption]:
        return [
            ActionOption(
                option_id="search",
                action="search",
                description="Submit search query to find relevant documents",
            ),
        ]

    def _search_results_options(self, screen_ir: ScreenIR) -> list[ActionOption]:
        options: list[ActionOption] = []

        # Open each search result
        for sr in self.engine.search_results:
            result_elements = [
                e for e in screen_ir.interactables
                if e.semantic_type == ElementSemanticType.RESULT_ITEM.value
                and str(sr.result_id) in (getattr(e, "text", "") or "")
            ]
            bbox = asdict(result_elements[0].bbox) if result_elements else None
            options.append(
                ActionOption(
                    option_id="open_result_" + str(sr.result_id),
                    action="open_result",
                    params={"result_id": sr.result_id},
                    description="Open document: " + sr.title[:40],
                    bbox=bbox,
                )
            )

        # Refine query
        options.append(
            ActionOption(
                option_id="refine_query",
                action="refine_query",
                params={"new_query": ""},
                description="Refine search with a new query",
            )
        )

        # Submit report (only if enough citations)
        min_cites = 1
        if self.task:
            min_cites = self.task.min_citations_for_submit
        if self.state.citation_count >= min_cites:
            options.append(
                ActionOption(
                    option_id="submit_report",
                    action="submit_report",
                    params={"report_text": ""},
                    description="Submit research report (" + str(self.state.citation_count) + " citations)",
                )
            )

        return options

    def _document_options(self, screen_ir: ScreenIR) -> list[ActionOption]:
        options = [
            ActionOption(
                option_id="back_to_results",
                action="back_to_results",
                description="Return to search results",
            ),
        ]

        # Cite source
        cite_elements = [
            e for e in screen_ir.interactables
            if e.semantic_type == ElementSemanticType.CITE_BUTTON.value
        ]
        bbox = asdict(cite_elements[0].bbox) if cite_elements else None
        options.append(
            ActionOption(
                option_id="cite_source",
                action="cite_source",
                params={"evidence_text": ""},
                description="Cite evidence from this document",
                bbox=bbox,
            )
        )

        # Zoom on document image
        doc_images = [
            e for e in screen_ir.interactables
            if e.semantic_type == ElementSemanticType.DOC_IMAGE.value
        ]
        if doc_images:
            options.append(
                ActionOption(
                    option_id="zoom_doc_image",
                    action="zoom",
                    params={"target": "doc_image"},
                    description="Zoom in on document image for visual research",
                    bbox=asdict(doc_images[0].bbox),
                )
            )

        # Submit report (if enough citations)
        min_cites = 1
        if self.task:
            min_cites = self.task.min_citations_for_submit
        if self.state.citation_count >= min_cites:
            options.append(
                ActionOption(
                    option_id="submit_report",
                    action="submit_report",
                    params={"report_text": ""},
                    description="Submit research report (" + str(self.state.citation_count) + " citations)",
                )
            )

        return options

    def _is_terminal(self, action: str) -> bool:
        return action == "submit_report"


def run_agent_loop(
    backend: DecisionBackend,
    task: ResearchTask | None = None,
    rag: Any = None,
    images_dir: str | Path | None = None,
) -> dict[str, Any]:
    return asyncio.run(AgentLoop(
        backend=backend, task=task, rag=rag, images_dir=images_dir,
    ).run())
