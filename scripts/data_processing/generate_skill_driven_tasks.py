"""
generate_skill_driven_tasks.py
===============================
技能驱动的任务生成管线

流程：
1. 初始化技能图谱
2. 生成任务批次（基于技能组合）
3. 验证任务可解性
4. 过滤有效任务
5. 保存到 JSONL
6. 生成统计报告
"""

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from searcheyes.skill_system import SkillGraph
from searcheyes.task_proposer import TaskProposer
from searcheyes.task_verifier import TaskVerifier


def main():
    print("=" * 60)
    print("技能驱动的任务生成系统")
    print("Skill-Driven Task Generation Pipeline")
    print("=" * 60)
    
    # Step 1: 初始化技能图谱
    print("\n[Step 1] 初始化技能图谱...")
    skill_graph = SkillGraph()
    stats = skill_graph.get_statistics()
    print(f"  技能总数: {stats['total_skills']}")
    print(f"  原子技能: {stats['atomic_skills']}")
    print(f"  技能组合: {stats['total_combos']}")
    print(f"  难度分布: {stats['difficulty_distribution']}")
    
    # Step 2: 生成任务批次
    print("\n[Step 2] 生成任务批次...")
    proposer = TaskProposer(skill_graph)
    
    # 难度分布：金字塔型
    difficulty_dist = {
        1: 0.25,  # 25% 简单任务
        2: 0.30,  # 30% 中等任务
        3: 0.25,  # 25% 中等偏难
        4: 0.15,  # 15% 困难任务
        5: 0.05,  # 5% 极难任务
    }
    
    tasks = proposer.propose_batch(
        num_tasks=100,
        difficulty_dist=difficulty_dist,
        seed=42,  # 可复现
    )
    
    task_stats = proposer.get_statistics(tasks)
    print(f"  生成任务数: {task_stats['total_tasks']}")
    print(f"  难度分布: {task_stats['difficulty_distribution']}")
    print(f"  需要 zoom: {task_stats['requires_zoom']}")
    print(f"  需要 memory: {task_stats['requires_memory']}")
    print(f"  有干扰项: {task_stats['has_distractors']}")
    print(f"  平均技能数: {task_stats['avg_skills_per_task']:.2f}")
    
    # 显示前 5 个任务示例
    print("\n  任务示例:")
    for i, task in enumerate(tasks[:5]):
        print(f"    [{i}] {task.goal}")
        print(f"        难度={task.difficulty}, 技能={len(task.required_skills)}, 步数={task.min_steps}-{task.max_steps}")
    
    # Step 3: 验证任务
    print("\n[Step 3] 验证任务可解性...")
    verifier = TaskVerifier()
    results = verifier.verify_batch(tasks)
    
    verify_stats = verifier.get_verification_statistics(results)
    print(f"  可验证: {verify_stats['verifiable']}/{verify_stats['total_tasks']}")
    print(f"  可解: {verify_stats['solvable']}/{verify_stats['total_tasks']}")
    print(f"  步数合理: {verify_stats['steps_in_range']}/{verify_stats['total_tasks']}")
    print(f"  技能匹配: {verify_stats['skill_match']}/{verify_stats['total_tasks']}")
    print(f"  通过率: {verify_stats['pass_rate']:.1%}")
    print(f"  平均步数: {verify_stats['avg_steps']:.1f}")
    
    # Step 4: 过滤有效任务
    print("\n[Step 4] 过滤有效任务...")
    valid_tasks = verifier.filter_valid_tasks(tasks, results)
    print(f"  有效任务: {len(valid_tasks)}/{len(tasks)}")
    
    # Step 5: 保存任务
    print("\n[Step 5] 保存任务...")
    output_dir = Path("data/tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存有效任务
    task_file = output_dir / "visual_tasks_skill_driven.jsonl"
    proposer.save_tasks(valid_tasks, task_file)
    print(f"  任务文件: {task_file}")
    
    # 保存验证结果
    verify_file = output_dir / "task_verification_results.json"
    with open(verify_file, "w", encoding="utf-8") as f:
        verify_data = {
            "statistics": verify_stats,
            "results": [
                {
                    "task_id": r.task_id,
                    "verifiable": r.verifiable,
                    "solvable": r.solvable,
                    "steps_taken": r.steps_taken,
                    "expected_steps": r.expected_steps_range,
                    "skill_coverage": r.skill_coverage,
                    "error": r.error_message,
                }
                for r in results
            ]
        }
        json.dump(verify_data, f, indent=2, ensure_ascii=False)
    print(f"  验证结果: {verify_file}")
    
    # Step 6: 生成报告
    print("\n[Step 6] 生成统计报告...")
    report = _generate_report(skill_graph, tasks, valid_tasks, results, task_stats, verify_stats)
    report_file = output_dir / "task_generation_report.md"
    report_file.write_text(report, encoding="utf-8")
    print(f"  报告文件: {report_file}")
    
    print("\n" + "=" * 60)
    print("任务生成完成")
    print(f"  有效任务: {len(valid_tasks)}")
    print(f"  通过率: {verify_stats['pass_rate']:.1%}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)


def _generate_report(
    skill_graph: SkillGraph,
    all_tasks: list,
    valid_tasks: list,
    results: list,
    task_stats: dict,
    verify_stats: dict,
) -> str:
    """生成 Markdown 报告"""
    
    report = f"""# 技能驱动任务生成报告

## 概览

- **生成时间**: {Path(__file__).stat().st_mtime}
- **总任务数**: {len(all_tasks)}
- **有效任务数**: {len(valid_tasks)}
- **通过率**: {verify_stats['pass_rate']:.1%}

## 技能图谱

- **技能总数**: {skill_graph.get_statistics()['total_skills']}
- **技能组合**: {skill_graph.get_statistics()['total_combos']}
- **难度分布**: {skill_graph.get_statistics()['difficulty_distribution']}

## 任务统计

### 难度分布

| 难度 | 数量 | 占比 |
|------|------|------|
"""
    
    for diff, count in sorted(task_stats['difficulty_distribution'].items()):
        ratio = count / len(all_tasks) * 100
        report += f"| {diff} | {count} | {ratio:.1f}% |\n"
    
    report += f"""
### 技能覆盖

| 技能 | 使用次数 |
|------|----------|
"""
    
    for skill, count in sorted(task_stats['skill_distribution'].items(), key=lambda x: -x[1])[:10]:
        report += f"| {skill} | {count} |\n"
    
    report += f"""
### 特殊要求

- **需要 zoom**: {task_stats['requires_zoom']} ({task_stats['requires_zoom']/len(all_tasks)*100:.1f}%)
- **需要 memory**: {task_stats['requires_memory']} ({task_stats['requires_memory']/len(all_tasks)*100:.1f}%)
- **有干扰项**: {task_stats['has_distractors']} ({task_stats['has_distractors']/len(all_tasks)*100:.1f}%)
- **平均技能数**: {task_stats['avg_skills_per_task']:.2f}

## 验证结果

- **可验证**: {verify_stats['verifiable']}/{verify_stats['total_tasks']} ({verify_stats['verifiable']/verify_stats['total_tasks']*100:.1f}%)
- **可解**: {verify_stats['solvable']}/{verify_stats['total_tasks']} ({verify_stats['solvable']/verify_stats['total_tasks']*100:.1f}%)
- **步数合理**: {verify_stats['steps_in_range']}/{verify_stats['total_tasks']} ({verify_stats['steps_in_range']/verify_stats['total_tasks']*100:.1f}%)
- **技能匹配**: {verify_stats['skill_match']}/{verify_stats['total_tasks']} ({verify_stats['skill_match']/verify_stats['total_tasks']*100:.1f}%)
- **平均步数**: {verify_stats['avg_steps']:.1f}

## 任务示例

"""
    
    # 每个难度选 2 个示例
    for difficulty in [1, 2, 3, 4, 5]:
        diff_tasks = [t for t in valid_tasks if t.difficulty == difficulty]
        if diff_tasks:
            report += f"\n### 难度 {difficulty}\n\n"
            for task in diff_tasks[:2]:
                report += f"**{task.task_id}**: {task.goal}\n"
                report += f"- 技能: {', '.join(s.value for s in task.required_skills)}\n"
                report += f"- 步数: {task.min_steps}-{task.max_steps}\n\n"
    
    report += """
## 使用方法

```python
# 加载任务
import json
with open('data/tasks/visual_tasks_skill_driven.jsonl', 'r', encoding='utf-8') as f:
    tasks = [json.loads(line) for line in f]

# 按难度筛选
easy_tasks = [t for t in tasks if t['difficulty'] == 1]
hard_tasks = [t for t in tasks if t['difficulty'] >= 4]

# 按技能筛选
zoom_tasks = [t for t in tasks if t['requires_zoom']]
memory_tasks = [t for t in tasks if t['requires_memory']]
```
"""
    
    return report


if __name__ == "__main__":
    main()
