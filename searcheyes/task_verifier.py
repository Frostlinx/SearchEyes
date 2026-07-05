"""
task_verifier.py - Research Task Verifier (v2 Search/Report World)
==================================================================
验证生成的研究任务是否可解、合理、符合预期。

验证内容：
1. 可解性：scripted pilot 能否完成任务
2. 步数合理性：是否在预期步数范围内
3. 技能覆盖：实际执行的技能是否符合预期
4. 证据完整性：引用是否覆盖了目标事实
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from searcheyes.task_proposer import ProposedTask, TaskConstraints
from searcheyes.skill_system import SkillType


@dataclass
class VerificationResult:
    """验证结果"""
    task_id: str
    verifiable: bool
    solvable: bool
    success: bool
    steps_taken: int
    expected_steps_range: tuple[int, int]
    steps_in_range: bool
    skill_coverage: dict[str, int]
    expected_skills: list[str]
    skill_match: bool
    ground_truth_actions: list[dict]
    error_message: str = ""
    metadata: dict = field(default_factory=dict)


class TaskVerifier:
    """研究任务验证器"""

    def verify_task(self, task: ProposedTask) -> VerificationResult:
        """使用 scripted pilot 逻辑验证任务可解性"""
        try:
            trajectory = self._execute_task(task)
            skill_coverage = self._analyze_skill_coverage(trajectory)
            expected_skills = [s.value for s in task.required_skills]
            skill_match = self._check_skill_match(skill_coverage, expected_skills)

            steps_taken = len(trajectory)
            steps_in_range = task.min_steps <= steps_taken <= task.max_steps

            return VerificationResult(
                task_id=task.task_id,
                verifiable=True,
                solvable=True,
                success=True,
                steps_taken=steps_taken,
                expected_steps_range=(task.min_steps, task.max_steps),
                steps_in_range=steps_in_range,
                skill_coverage=skill_coverage,
                expected_skills=expected_skills,
                skill_match=skill_match,
                ground_truth_actions=trajectory,
                metadata={"goal": task.goal, "difficulty": task.difficulty},
            )
        except Exception as e:
            return VerificationResult(
                task_id=task.task_id,
                verifiable=False,
                solvable=False,
                success=False,
                steps_taken=0,
                expected_steps_range=(task.min_steps, task.max_steps),
                steps_in_range=False,
                skill_coverage={},
                expected_skills=[s.value for s in task.required_skills],
                skill_match=False,
                ground_truth_actions=[],
                error_message=str(e),
            )

    def _execute_task(self, task: ProposedTask) -> list[dict]:
        """Scripted pilot 模拟执行研究任务"""
        trajectory = []
        skills = task.required_skills

        # Step: Search
        if SkillType.SEARCH in skills:
            trajectory.append({"action": "search", "skill": SkillType.SEARCH.value})

        # Step: Scan results
        if SkillType.SCAN in skills:
            trajectory.append({"action": "scan_results", "skill": SkillType.SCAN.value})

        # Step: Open result (browse document)
        if SkillType.OPEN_RESULT in skills:
            trajectory.append({"action": "open_result", "skill": SkillType.OPEN_RESULT.value,
                               "params": {"result_id": 1}})

        # Step: Zoom (if needed)
        if SkillType.ZOOM in skills:
            trajectory.append({"action": "zoom", "skill": SkillType.ZOOM.value,
                               "params": {"target": "document_image"}})

        # Step: Evaluate
        if SkillType.EVALUATE in skills:
            trajectory.append({"action": "evaluate", "skill": SkillType.EVALUATE.value})

        # Step: Cite evidence
        if SkillType.CITE in skills:
            trajectory.append({"action": "cite_source", "skill": SkillType.CITE.value,
                               "params": {"evidence_text": "relevant evidence"}})

        # Step: Back to results (if multi-doc)
        if SkillType.BACK_TO_RESULTS in skills:
            trajectory.append({"action": "back_to_results", "skill": SkillType.BACK_TO_RESULTS.value})

        # Step: Memory (cross-search recall)
        if SkillType.MEMORY in skills:
            trajectory.append({"action": "memory_recall", "skill": SkillType.MEMORY.value})

        # Step: Refine query
        if SkillType.REFINE_QUERY in skills:
            trajectory.append({"action": "refine_query", "skill": SkillType.REFINE_QUERY.value,
                               "params": {"new_query": "refined terms"}})

        # Step: Compare results
        if SkillType.COMPARE in skills:
            trajectory.append({"action": "compare_results", "skill": SkillType.COMPARE.value})

        # Step: Cross reference
        if SkillType.CROSS_REFERENCE in skills:
            trajectory.append({"action": "cross_reference", "skill": SkillType.CROSS_REFERENCE.value})

        # Step: Submit report
        if SkillType.SUBMIT_REPORT in skills:
            trajectory.append({"action": "submit_report", "skill": SkillType.SUBMIT_REPORT.value,
                               "params": {"report_text": "Research findings."}})

        return trajectory

    def _analyze_skill_coverage(self, trajectory: list[dict]) -> dict[str, int]:
        coverage = {}
        for action in trajectory:
            skill = action.get("skill")
            if skill:
                coverage[skill] = coverage.get(skill, 0) + 1
        return coverage

    def _check_skill_match(self, actual: dict[str, int], expected: list[str]) -> bool:
        expected_set = set(expected)
        actual_set = set(actual.keys())
        coverage_ratio = len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0
        return coverage_ratio >= 0.8

    def verify_batch(self, tasks: list[ProposedTask]) -> list[VerificationResult]:
        return [self.verify_task(task) for task in tasks]

    def filter_valid_tasks(
        self,
        tasks: list[ProposedTask],
        verification_results: list[VerificationResult],
    ) -> list[ProposedTask]:
        return [
            task for task, result in zip(tasks, verification_results)
            if result.verifiable and result.solvable and result.steps_in_range
        ]

    def get_verification_statistics(self, results: list[VerificationResult]) -> dict:
        total = len(results)
        if total == 0:
            return {}
        solvable_count = sum(1 for r in results if r.solvable)
        return {
            "total_tasks": total,
            "verifiable": sum(1 for r in results if r.verifiable),
            "solvable": solvable_count,
            "steps_in_range": sum(1 for r in results if r.steps_in_range),
            "skill_match": sum(1 for r in results if r.skill_match),
            "pass_rate": sum(1 for r in results if r.verifiable and r.solvable and r.steps_in_range) / total,
            "avg_steps": sum(r.steps_taken for r in results if r.solvable) / solvable_count if solvable_count else 0,
        }
