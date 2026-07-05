"""
task_proposer.py - Research Task Proposer (v2 Search/Report World)
==================================================================
自动生成技能驱动的研究任务，支持：
- 基于技能组合自动出题
- 约束条件采样（文档类型/领域/证据数量）
- 自然语言目标生成
- 难度分布控制
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from searcheyes.skill_system import SkillGraph, SkillType, SkillCombo


@dataclass
class TaskConstraints:
    """研究任务约束条件"""
    domain: Optional[str] = None           # 领域约束 (science/history/geography/art)
    min_citations: int = 1                 # 最少引用数
    max_steps: int = 12                    # 最大步数
    requires_cross_reference: bool = False # 是否需要交叉验证
    requires_multi_search: bool = False    # 是否需要多轮搜索
    objective: str = "cite_and_submit"     # cite_and_submit / deep_investigate


@dataclass
class ProposedTask:
    """生成的研究任务"""
    task_id: str
    goal: str
    constraints: TaskConstraints
    required_skills: list[SkillType]
    skill_combo_id: str
    difficulty: int  # 1-5
    min_steps: int
    max_steps: int
    requires_zoom: bool
    requires_memory: bool
    requires_multi_search: bool = False
    metadata: dict = field(default_factory=dict)


# 研究领域列表
RESEARCH_DOMAINS = ["science", "history", "geography", "art", "biology", "technology"]

# 目标生成模板
_GOAL_TEMPLATES = {
    1: [
        "搜索知识库找到相关文档，引用关键证据后提交报告",
        "在知识库中搜索目标内容，找到对应文档并引用证据",
    ],
    2: [
        "搜索知识库，浏览结果列表找到最相关的文档，引用证据后提交报告",
        "搜索知识库，打开文档确认内容匹配后引用证据并提交",
    ],
    3: [
        "搜索知识库，比较多个搜索结果，选择最相关的文档引用证据后提交报告",
        "搜索后浏览多个文档，找到最匹配的证据引用并提交",
    ],
    4: [
        "多轮搜索知识库，跨搜索记忆关键信息，收集多条证据后提交完整研究报告",
        "搜索、阅读文档并记忆信息，改写关键词重搜，收集充分证据后提交",
    ],
    5: [
        "深度研究：多轮搜索、放大验证文档图片、跨文档交叉验证、综合多条证据后提交完整报告",
    ],
}


class TaskProposer:
    """研究任务生成器"""

    def __init__(self, skill_graph: SkillGraph):
        self.skill_graph = skill_graph

    def propose_batch(
        self,
        num_tasks: int = 100,
        difficulty_dist: dict[int, float] = None,
        seed: Optional[int] = None,
    ) -> list[ProposedTask]:
        if seed is not None:
            random.seed(seed)

        if difficulty_dist is None:
            difficulty_dist = {1: 0.25, 2: 0.30, 3: 0.25, 4: 0.15, 5: 0.05}

        tasks = []
        for difficulty in [1, 2, 3, 4, 5]:
            count = int(num_tasks * difficulty_dist.get(difficulty, 0))
            combos = self.skill_graph.get_combos_by_difficulty(difficulty)
            if not combos:
                continue
            for _ in range(count):
                combo = random.choice(combos)
                task = self._generate_task_from_combo(combo)
                tasks.append(task)

        random.shuffle(tasks)
        return tasks

    def _generate_task_from_combo(self, combo: SkillCombo) -> ProposedTask:
        constraints = self._sample_constraints(combo)
        goal = self._compose_goal(combo, constraints)
        task_id = f"research_proposed_{uuid.uuid4().hex[:8]}"

        return ProposedTask(
            task_id=task_id,
            goal=goal,
            constraints=constraints,
            required_skills=combo.skills,
            skill_combo_id=combo.combo_id,
            difficulty=combo.difficulty,
            min_steps=combo.min_steps,
            max_steps=combo.max_steps,
            requires_zoom=combo.requires_zoom,
            requires_memory=combo.requires_memory,
            requires_multi_search=combo.requires_multi_search,
            metadata={"combo_description": combo.description},
        )

    def _sample_constraints(self, combo: SkillCombo) -> TaskConstraints:
        constraints = TaskConstraints()
        constraints.domain = random.choice(RESEARCH_DOMAINS)

        if SkillType.CROSS_REFERENCE in combo.skills:
            constraints.requires_cross_reference = True
            constraints.min_citations = max(2, constraints.min_citations)

        if combo.requires_multi_search:
            constraints.requires_multi_search = True

        if combo.difficulty >= 4:
            constraints.min_citations = max(3, constraints.min_citations)
            constraints.objective = "deep_investigate"

        return constraints

    def _compose_goal(self, combo: SkillCombo, constraints: TaskConstraints) -> str:
        templates = _GOAL_TEMPLATES.get(combo.difficulty, _GOAL_TEMPLATES[1])
        goal = random.choice(templates)

        if constraints.domain:
            goal = f"[{constraints.domain}] " + goal

        return goal

    def save_tasks(self, tasks: list[ProposedTask], output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for task in tasks:
                task_dict = {
                    "task_id": task.task_id,
                    "goal": task.goal,
                    "constraints": asdict(task.constraints),
                    "required_skills": [s.value for s in task.required_skills],
                    "skill_combo_id": task.skill_combo_id,
                    "difficulty": task.difficulty,
                    "min_steps": task.min_steps,
                    "max_steps": task.max_steps,
                    "requires_zoom": task.requires_zoom,
                    "requires_memory": task.requires_memory,
                    "requires_multi_search": task.requires_multi_search,
                    "metadata": task.metadata,
                }
                f.write(json.dumps(task_dict, ensure_ascii=False) + "\n")

    def get_statistics(self, tasks: list[ProposedTask]) -> dict:
        from collections import Counter
        return {
            "total_tasks": len(tasks),
            "difficulty_distribution": dict(Counter(t.difficulty for t in tasks)),
            "skill_distribution": dict(Counter(
                skill.value for t in tasks for skill in t.required_skills
            )),
            "requires_zoom": sum(1 for t in tasks if t.requires_zoom),
            "requires_memory": sum(1 for t in tasks if t.requires_memory),
            "requires_multi_search": sum(1 for t in tasks if t.requires_multi_search),
        }
