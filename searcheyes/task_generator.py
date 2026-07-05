"""
task_generator.py — RAG 驱动的研究任务合成器 (v2 Search/Report World)
=====================================================================
所有任务都基于 WIT 知识库条目生成。
search 动作通过 RAG 检索知识库，Agent 需要找到并引用正确的文档。
"""

import json
import random
from dataclasses import asdict
from pathlib import Path
from searcheyes.task_schema import (
    VisualTask, TrajectoryStep, DifficultyLevel, DifficultyTag
)

TASK_DIR = Path(__file__).parent.parent / "data" / "tasks"

# ── 研究任务模板库 ──────────────────────────────

RESEARCH_TASK_TEMPLATES = {
    "identify_and_cite": {
        "goal": "搜索与「{caption_hint}」相关的文档，找到对应条目并引用关键证据，提交研究报告",
        "tags": [DifficultyTag.RAG_KNOWLEDGE],
    },
    "visual_research": {
        "goal": "通过图片内容搜索知识库，找到展示了「{caption_hint}」的文档，引用证据后提交报告",
        "tags": [DifficultyTag.ZOOM_REQUIRED, DifficultyTag.RAG_KNOWLEDGE, DifficultyTag.VISUAL_AMBIGUITY],
    },
    "multi_source_verify": {
        "goal": "搜索「{caption_hint}」相关资料，浏览多个文档进行交叉验证，收集证据后提交报告",
        "tags": [DifficultyTag.RAG_KNOWLEDGE, DifficultyTag.MULTI_PAGE],
    },
    "refine_search": {
        "goal": "搜索「{caption_hint}」，如果初次结果不理想则改写关键词重新搜索，找到目标文档并引用",
        "tags": [DifficultyTag.RAG_KNOWLEDGE],
    },
    "deep_investigate": {
        "goal": "深入调研「{caption_hint}」：搜索知识库、放大查看文档图片、收集多条证据、交叉验证后提交完整报告",
        "tags": [DifficultyTag.ZOOM_REQUIRED, DifficultyTag.RAG_KNOWLEDGE, DifficultyTag.VISUAL_AMBIGUITY, DifficultyTag.MULTI_PAGE],
    },
    "zoom_and_cite": {
        "goal": "搜索知识库找到「{caption_hint}」相关文档，放大查看图片细节确认内容后引用证据",
        "tags": [DifficultyTag.ZOOM_REQUIRED, DifficultyTag.RAG_KNOWLEDGE],
    },
}

DIFFICULTY_TEMPLATE_KEYS = {
    DifficultyLevel.EASY: ["identify_and_cite", "refine_search"],
    DifficultyLevel.MEDIUM: ["visual_research", "zoom_and_cite", "multi_source_verify"],
    DifficultyLevel.HARD: ["deep_investigate", "multi_source_verify", "visual_research"],
}


def generate_task(
    task_idx: int,
    wit_entries: list[dict],
    rng: random.Random,
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
) -> VisualTask:
    """合成一条研究任务。

    所有任务都从 WIT 条目生成，ground_truth_wit_id 指向目标文档。
    search 动作通过 RAG 检索知识库。
    """
    n_results = min(6, len(wit_entries))
    selected_entries = rng.sample(wit_entries, n_results)

    target_idx = rng.randrange(n_results)
    target_entry = selected_entries[target_idx]
    target_result_id = target_idx + 1

    caption_hint = _extract_caption_hint(target_entry.get("caption", ""), rng)
    if not caption_hint:
        caption_hint = target_entry.get("page_title", "unknown document")

    template_keys = DIFFICULTY_TEMPLATE_KEYS[difficulty]
    template_key = rng.choice(template_keys)
    template = RESEARCH_TASK_TEMPLATES[template_key]
    goal = template["goal"].format(caption_hint=caption_hint)

    # 构造 ground truth trajectory
    trajectory = _build_research_trajectory(
        difficulty, target_result_id, n_results, goal, rng
    )

    vgap = sum(1 for s in trajectory if s.requires_zoom)

    task = VisualTask(
        task_id=f"research_{task_idx:04d}",
        goal=goal,
        difficulty=difficulty,
        page_family_sequence=["search", "results", "document"],
        initial_state="search",
        visual_anchors=["doc_image", "result_item", "cite_button"],
        visual_gap_count=vgap,
        zoom_budget=vgap,
        dag_depth=len(trajectory),
        ground_truth_trajectory=trajectory,
        final_answer=f"document_{target_result_id} (wit_id={target_entry.get('wit_id', '')})",
        difficulty_tags=[t.value for t in template["tags"]],
        requires_rag=True,
        ground_truth_wit_id=target_entry.get("wit_id", ""),
        ground_truth_caption=target_entry.get("caption", ""),
    )

    wit_bindings = [
        {"result_id": i + 1, "wit_id": e.get("wit_id", ""), "caption": e.get("caption", ""),
         "image_filename": e.get("image_filename", "")}
        for i, e in enumerate(selected_entries)
    ]
    task._wit_bindings = wit_bindings  # type: ignore[attr-defined]
    return task


def _build_research_trajectory(
    difficulty: DifficultyLevel,
    target_result_id: int,
    n_results: int,
    goal: str,
    rng: random.Random,
) -> list[TrajectoryStep]:
    """根据难度构建研究 GT 轨迹。"""
    trajectory = []
    step = 0

    # Step 0: 搜索
    trajectory.append(TrajectoryStep(
        step_idx=step, state="search", action="search",
        cot_reasoning=f"任务要求: {goal[:80]}。提交搜索触发 RAG 检索。"
    ))
    step += 1

    if difficulty == DifficultyLevel.EASY:
        # 直接打开目标文档 → 引用 → 提交
        trajectory.append(TrajectoryStep(
            step_idx=step, state="results", action="open_result",
            action_params={"result_id": target_result_id},
            cot_reasoning=f"在搜索结果中识别目标文档（第{target_result_id}条），打开查看。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state=f"document_{target_result_id}", action="cite_source",
            action_params={"evidence_text": "relevant evidence from document"},
            cot_reasoning="文档内容与研究目标匹配，引用关键证据。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state=f"document_{target_result_id}", action="submit_report",
            action_params={"report_text": "Research findings based on collected evidence."},
            cot_reasoning="已收集足够证据，提交研究报告。",
        ))
        step += 1

    elif difficulty == DifficultyLevel.MEDIUM:
        # 打开一个非目标文档 → 返回 → 打开目标 → zoom → 引用 → 提交
        other_ids = [i + 1 for i in range(n_results) if i + 1 != target_result_id]
        distractor_id = rng.choice(other_ids) if other_ids else target_result_id

        trajectory.append(TrajectoryStep(
            step_idx=step, state="results", action="open_result",
            action_params={"result_id": distractor_id},
            cot_reasoning=f"先查看第{distractor_id}条结果，评估相关性。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state=f"document_{distractor_id}", action="back_to_results",
            cot_reasoning="该文档与目标不够相关，返回结果列表继续查看。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state="results", action="open_result",
            action_params={"result_id": target_result_id},
            cot_reasoning=f"打开第{target_result_id}条结果，该文档看起来更相关。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state=f"document_{target_result_id}", action="cite_source",
            action_params={"evidence_text": "key evidence matching research goal"},
            cot_reasoning="文档内容与研究目标高度匹配，引用关键证据。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state=f"document_{target_result_id}", action="submit_report",
            action_params={"report_text": "Research report with verified evidence."},
            cot_reasoning="证据充分，提交报告。",
        ))
        step += 1

    else:  # HARD
        # 搜索 → 浏览 → 返回 → refine → 打开目标 → zoom → 引用 → 交叉验证 → 提交
        other_ids = [i + 1 for i in range(n_results) if i + 1 != target_result_id]
        distractor_id = rng.choice(other_ids) if other_ids else target_result_id

        trajectory.append(TrajectoryStep(
            step_idx=step, state="results", action="open_result",
            action_params={"result_id": distractor_id},
            cot_reasoning=f"先查看第{distractor_id}条结果。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state=f"document_{distractor_id}", action="back_to_results",
            cot_reasoning="初次搜索结果不够精确，返回改写 query。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state="results", action="refine_query",
            action_params={"new_query": "refined search terms"},
            cot_reasoning="改写搜索关键词以获得更精确的结果。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state="results", action="open_result",
            action_params={"result_id": target_result_id},
            cot_reasoning=f"改写后结果更好，打开第{target_result_id}条目标文档。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state=f"document_{target_result_id}", action="cite_source",
            action_params={"evidence_text": "primary evidence from target document"},
            cot_reasoning="找到目标文档，引用主要证据。",
            requires_zoom=True,
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state=f"document_{target_result_id}", action="back_to_results",
            cot_reasoning="返回搜索结果查找补充证据。",
        ))
        step += 1

        second_id = rng.choice([i for i in other_ids if i != distractor_id]) if len(other_ids) > 1 else distractor_id
        trajectory.append(TrajectoryStep(
            step_idx=step, state="results", action="open_result",
            action_params={"result_id": second_id},
            cot_reasoning=f"打开第{second_id}条文档寻找补充证据。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state=f"document_{second_id}", action="cite_source",
            action_params={"evidence_text": "supporting evidence from secondary source"},
            cot_reasoning="从第二份文档中引用补充证据进行交叉验证。",
        ))
        step += 1

        trajectory.append(TrajectoryStep(
            step_idx=step, state=f"document_{second_id}", action="submit_report",
            action_params={"report_text": "Comprehensive report with cross-referenced evidence."},
            cot_reasoning="已从多个来源收集证据并交叉验证，提交完整研究报告。",
        ))
        step += 1

    return trajectory


def _extract_caption_hint(caption: str, rng: random.Random) -> str:
    """从 WIT caption 中提取适合做任务 goal 的简短提示。"""
    hint = caption.strip()
    for prefix in ("A photo of ", "An image of ", "Image of ", "Picture of ",
                   "Photo of ", "Picture showing ", "A view of "):
        if hint.startswith(prefix):
            hint = hint[len(prefix):]
            break
    if len(hint) > 60:
        cut = hint[:60].rsplit(" ", 1)[0] if " " in hint[:60] else hint[:57]
        hint = cut.rstrip(",. ") + "..."
    return hint


def generate_task_set(
    wit_meta_jsonl: str | Path,
    count: int = 100,
    seed: int = 42,
    validate: bool = True,
) -> list[VisualTask]:
    """批量合成研究任务集。"""
    meta_path = Path(wit_meta_jsonl)
    if not meta_path.exists():
        raise FileNotFoundError(f"WIT meta.jsonl not found: {meta_path}")

    wit_entries = [
        json.loads(line) for line in meta_path.read_text("utf-8").splitlines() if line.strip()
    ]
    wit_entries = [e for e in wit_entries if e.get("caption")]
    if len(wit_entries) < 6:
        raise ValueError(f"有效 WIT 条目不足（需要至少6条，实际{len(wit_entries)}条）")

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    distribution = (
        [DifficultyLevel.EASY] * int(count * 0.3) +
        [DifficultyLevel.MEDIUM] * int(count * 0.4) +
        [DifficultyLevel.HARD] * (count - int(count * 0.3) - int(count * 0.4))
    )
    rng.shuffle(distribution)

    tasks = []
    valid_count = 0

    for i, diff in enumerate(distribution):
        task = generate_task(i, wit_entries, rng, diff)
        if validate:
            ok, _ = task.validate()
            if ok:
                valid_count += 1
        tasks.append(task)

    jsonl_path = TASK_DIR / "research_tasks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for t in tasks:
            d = asdict(t)
            if hasattr(t, "_wit_bindings"):
                d["wit_bindings"] = t._wit_bindings
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")

    print(f"Research Task Set: {count} tasks, {valid_count} valid → {jsonl_path}")
    return tasks


generate_visual_research_task_set = generate_task_set

if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wit-meta", default="", help="WIT meta.jsonl path")
    a = p.parse_args()
    wit_meta = a.wit_meta or str(Path(__file__).parent.parent / "data" / "wit_kb_v2" / "meta.jsonl")
    generate_task_set(wit_meta, count=a.count, seed=a.seed, validate=True)
