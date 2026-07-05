"""
convert_skill_tasks_to_visual_tasks.py
=======================================
将技能驱动任务转换为 VisualTask 格式，用于训练管线。
"""

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from searcheyes.task_schema import VisualTask, DifficultyLevel, TrajectoryStep
from searcheyes.skill_system import SkillType


def convert_proposed_task_to_visual_task(proposed_task: dict) -> VisualTask:
    """将 ProposedTask 转换为 VisualTask"""
    
    # 难度映射
    difficulty_map = {
        1: DifficultyLevel.EASY,
        2: DifficultyLevel.EASY,
        3: DifficultyLevel.MEDIUM,
        4: DifficultyLevel.HARD,
        5: DifficultyLevel.HARD,
    }
    
    # 构建页面序列
    page_sequence = ["search"]
    skills = proposed_task["required_skills"]
    
    if "click_product" in skills:
        page_sequence.append("results")
        page_sequence.append("detail")
    
    if "back" in skills and "memory" in skills:
        page_sequence.append("results")
        page_sequence.append("detail")
    
    # 计算视觉断层
    zoom_budget = 1 if proposed_task["requires_zoom"] else 0
    visual_gap_count = zoom_budget + (1 if proposed_task["requires_memory"] else 0)
    
    # 构建 ground truth 轨迹（简化版，实际应该从 verifier 的结果中获取）
    trajectory = []
    step_idx = 0
    
    for skill in skills:
        if skill == "search":
            trajectory.append(TrajectoryStep(
                step_idx=step_idx,
                state="search",
                action="search",
                action_params={},
                requires_zoom=False,
            ))
            step_idx += 1
        
        elif skill == "click_product":
            trajectory.append(TrajectoryStep(
                step_idx=step_idx,
                state="results",
                action="click_product",
                action_params={"product_id": 1},
                target_element_id=1,
                requires_zoom=False,
            ))
            step_idx += 1
        
        elif skill == "zoom":
            trajectory.append(TrajectoryStep(
                step_idx=step_idx,
                state="detail",
                action="zoom",
                action_params={"element_id": "price"},
                requires_zoom=True,
            ))
            step_idx += 1
        
        elif skill == "buy":
            trajectory.append(TrajectoryStep(
                step_idx=step_idx,
                state="detail",
                action="buy",
                action_params={},
                requires_zoom=False,
            ))
            step_idx += 1
    
    # 难度标签
    difficulty_tags = []
    if proposed_task["requires_zoom"]:
        difficulty_tags.append("zoom_required")
    if proposed_task["requires_memory"]:
        difficulty_tags.append("multi_page")
    if "compare" in skills:
        difficulty_tags.append("comparison")
    if "aggregate" in skills:
        difficulty_tags.append("numerical_reasoning")
    
    return VisualTask(
        task_id=proposed_task["task_id"],
        goal=proposed_task["goal"],
        difficulty=difficulty_map[proposed_task["difficulty"]],
        page_family_sequence=page_sequence,
        initial_state="search",
        visual_anchors=[],
        visual_gap_count=visual_gap_count,
        zoom_budget=zoom_budget,
        dag_depth=len(trajectory),
        ground_truth_trajectory=trajectory,
        final_answer="",
        difficulty_tags=difficulty_tags,
    )


def main():
    print("=" * 60)
    print("转换技能驱动任务为 VisualTask 格式")
    print("=" * 60)
    
    # 加载技能驱动任务
    input_file = Path("data/tasks/visual_tasks_skill_driven.jsonl")
    if not input_file.exists():
        print(f"错误: {input_file} 不存在")
        print("请先运行 generate_skill_driven_tasks.py")
        return
    
    proposed_tasks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            proposed_tasks.append(json.loads(line))
    
    print(f"\n加载了 {len(proposed_tasks)} 个技能驱动任务")
    
    # 转换为 VisualTask
    visual_tasks = []
    for proposed in proposed_tasks:
        visual_task = convert_proposed_task_to_visual_task(proposed)
        visual_tasks.append(visual_task)
    
    print(f"转换了 {len(visual_tasks)} 个 VisualTask")
    
    # 验证
    valid_count = 0
    for task in visual_tasks:
        is_valid, errors = task.validate()
        if is_valid:
            valid_count += 1
        else:
            print(f"  警告: {task.task_id} 验证失败: {errors}")
    
    print(f"验证通过: {valid_count}/{len(visual_tasks)}")
    
    # 保存为 JSONL
    output_file = Path("data/tasks/visual_tasks.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for task in visual_tasks:
            # 手动序列化，确保 TrajectoryStep 正确转换
            task_dict = {
                "task_id": task.task_id,
                "goal": task.goal,
                "difficulty": task.difficulty.value,
                "page_family_sequence": task.page_family_sequence,
                "initial_state": task.initial_state,
                "visual_anchors": task.visual_anchors,
                "visual_gap_count": task.visual_gap_count,
                "zoom_budget": task.zoom_budget,
                "dag_depth": task.dag_depth,
                "ground_truth_trajectory": [
                    {
                        "step_idx": s.step_idx,
                        "state": s.state,
                        "action": s.action,
                        "action_params": s.action_params,
                        "target_element_id": s.target_element_id,
                        "target_bbox": s.target_bbox,
                        "requires_zoom": s.requires_zoom,
                        "cot_reasoning": s.cot_reasoning,
                    }
                    for s in task.ground_truth_trajectory
                ],
                "final_answer": task.final_answer,
                "difficulty_tags": task.difficulty_tags,
            }
            f.write(json.dumps(task_dict, ensure_ascii=False) + "\n")
    
    print(f"\n保存到: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
