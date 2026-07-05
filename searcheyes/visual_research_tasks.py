"""
visual_research_tasks.py — 视觉研究任务生成器 (v2 Search/Report World)
=====================================================================
从 WIT 元数据生成需要视觉识别 + RAG 检索的研究任务。
Agent 必须 search → browse → cite → submit_report。
"""

from __future__ import annotations
import json
import random
from pathlib import Path

from searcheyes.task_schema import (
    DifficultyLevel,
    DifficultyTag,
    TrajectoryStep,
    VisualTask,
)
from searcheyes.wit_downloader import WITEntry, load_metadata

TASK_TEMPLATES = [
    "搜索知识库，找到与目标图片内容匹配的文档，引用关键证据后提交研究报告",
    "通过图片信息检索知识库，找到相关文档，收集证据并提交报告",
    "在知识库中搜索目标内容，打开相关文档确认信息后引用证据",
    "搜索并浏览知识库文档，通过视觉确认找到目标条目，引用证据后提交",
    "利用视觉信息搜索知识库，从多个结果中找到目标文档并引用关键信息",
]

CATEGORY_TEMPLATES = {
    "electronics": "搜索知识库找到电子产品相关文档，确认型号信息后引用证据",
    "animal": "搜索知识库找到动物相关文档，确认物种信息后引用证据",
    "architecture": "搜索知识库找到建筑相关文档，确认名称和位置后引用证据",
    "food": "搜索知识库找到食物相关文档，确认名称信息后引用证据",
    "vehicle": "搜索知识库找到交通工具相关文档，确认型号后引用证据",
    "art": "搜索知识库找到艺术品相关文档，确认作品信息后引用证据",
    "plant": "搜索知识库找到植物相关文档，确认种类后引用证据",
    "sport": "搜索知识库找到运动相关文档，确认项目信息后引用证据",
}


def _build_ground_truth_trajectory(difficulty: DifficultyLevel) -> list[TrajectoryStep]:
    """构建 v2 研究 ground truth 轨迹"""
    steps = [
        TrajectoryStep(
            step_idx=0, state="search", action="search",
            action_params={"query": "target content"},
            cot_reasoning="提交搜索检索知识库中的相关文档",
        ),
        TrajectoryStep(
            step_idx=1, state="results", action="open_result",
            action_params={"result_id": 1},
            cot_reasoning="打开第一条搜索结果查看文档详情",
        ),
    ]
    if difficulty == DifficultyLevel.HARD:
        steps.append(TrajectoryStep(
            step_idx=2, state="document_1", action="back_to_results",
            cot_reasoning="该文档不够相关，返回结果列表查看其他选项",
        ))
        steps.append(TrajectoryStep(
            step_idx=3, state="results", action="open_result",
            action_params={"result_id": 2},
            cot_reasoning="打开第二条结果，该文档看起来更相关",
        ))
        steps.append(TrajectoryStep(
            step_idx=4, state="document_2", action="cite_source",
            action_params={"evidence_text": "key evidence"},
            cot_reasoning="文档内容匹配研究目标，引用关键证据",
        ))
        steps.append(TrajectoryStep(
            step_idx=5, state="document_2", action="submit_report",
            action_params={"report_text": "Research findings."},
            cot_reasoning="证据充分，提交研究报告",
        ))
    else:
        steps.append(TrajectoryStep(
            step_idx=2, state="document_1", action="cite_source",
            action_params={"evidence_text": "key evidence"},
            cot_reasoning="文档内容匹配目标，引用证据",
        ))
        steps.append(TrajectoryStep(
            step_idx=3, state="document_1", action="submit_report",
            action_params={"report_text": "Research findings."},
            cot_reasoning="引用完成，提交报告",
        ))
    return steps


def generate_visual_research_tasks(
    wit_metadata_path: str | Path,
    count: int = 100,
    seed: int = 42,
) -> list[VisualTask]:
    """从 WIT 元数据生成视觉研究任务。"""
    rng = random.Random(seed)
    entries = load_metadata(wit_metadata_path)
    if not entries:
        return []

    rng.shuffle(entries)
    selected = entries[:count]

    tasks = []
    for i, entry in enumerate(selected):
        categories = entry.category_keywords or []
        goal = None
        for cat in categories:
            for key, template in CATEGORY_TEMPLATES.items():
                if key in cat:
                    goal = template
                    break
            if goal:
                break
        if not goal:
            goal = rng.choice(TASK_TEMPLATES)

        has_visual_cats = bool(categories)
        if len(entry.fact_text) > 200 and has_visual_cats:
            difficulty = DifficultyLevel.HARD
        elif has_visual_cats:
            difficulty = DifficultyLevel.MEDIUM
        else:
            difficulty = DifficultyLevel.EASY

        trajectory = _build_ground_truth_trajectory(difficulty)

        task = VisualTask(
            task_id=f"vr_{entry.wit_id}_{i:04d}",
            goal=goal,
            difficulty=difficulty,
            page_family_sequence=["search", "results", "document"],
            initial_state="search",
            visual_anchors=[f"wit_image_{entry.wit_id}"],
            visual_gap_count=0,
            zoom_budget=0,
            dag_depth=len(trajectory),
            ground_truth_trajectory=trajectory,
            final_answer=entry.page_title,
            rag_ground_truth=entry.fact_text,
            wit_image_id=entry.wit_id,
            difficulty_tags=[
                DifficultyTag.VISUAL_RESEARCH.value,
                DifficultyTag.RAG_REQUIRED.value,
            ],
        )
        valid, _ = task.validate()
        if valid:
            tasks.append(task)

    return tasks


def save_tasks(tasks: list[VisualTask], output_path: str | Path):
    """保存任务列表为 JSONL"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for task in tasks:
            from dataclasses import asdict
            f.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成视觉研究任务")
    parser.add_argument("--metadata", type=str, default="data/wit_kb_v2/metadata.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/tasks/visual_research_tasks.jsonl")
    args = parser.parse_args()

    tasks = generate_visual_research_tasks(args.metadata, args.count)
    save_tasks(tasks, args.output)
    print(f"生成 {len(tasks)} 个视觉研究任务 -> {args.output}")
