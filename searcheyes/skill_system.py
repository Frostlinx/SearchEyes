"""
skill_system.py - Skill-Driven Task Generation System (v2 Research World)
==========================================================================
定义研究型原子技能、技能依赖图、技能组合策略。

v2: 从 shopping world 迁移到 search/report world。
技能对应 Agent 在研究环境中的原子操作。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from searcheyes.screen_ir import PageFamily


class SkillType(str, Enum):
    """研究技能类型 (v2 search/report world)"""
    # 搜索类
    SEARCH = "search"                  # 输入关键词检索知识库
    REFINE_QUERY = "refine_query"      # 改写 query 重新搜索
    
    # 浏览类
    OPEN_RESULT = "open_result"        # 打开搜索结果中的文档
    BACK_TO_RESULTS = "back_to_results"  # 返回搜索结果列表
    
    # 视觉类
    ZOOM = "zoom"                      # 放大查看图片/文档细节
    SCAN = "scan"                      # 扫描结果页找目标文档
    
    # 证据收集类
    CITE = "cite"                      # 从文档中引用证据
    CROSS_REFERENCE = "cross_reference"  # 跨文档交叉验证
    
    # 决策类
    COMPARE = "compare"                # 比较多条搜索结果的相关性
    EVALUATE = "evaluate"              # 评估文档与任务目标的匹配度
    MEMORY = "memory"                  # 跨搜索记忆（记住之前搜到的信息）
    
    # 终结类
    SUBMIT_REPORT = "submit_report"    # 提交最终研究报告


@dataclass
class Skill:
    """技能定义"""
    skill_id: SkillType
    name: str
    description: str
    required_page_families: list[PageFamily]
    atomic: bool = True
    prerequisites: list[SkillType] = field(default_factory=list)
    difficulty_weight: float = 1.0


# ── 技能定义库 ──────────────────────────────────────────

SKILL_DEFINITIONS = {
    SkillType.SEARCH: Skill(
        skill_id=SkillType.SEARCH,
        name="搜索知识库",
        description="输入关键词检索 WIT 维基百科知识库",
        required_page_families=[PageFamily.SEARCH],
        difficulty_weight=1.0,
    ),

    SkillType.REFINE_QUERY: Skill(
        skill_id=SkillType.REFINE_QUERY,
        name="改写搜索词",
        description="根据已有结果改写 query 重新搜索",
        required_page_families=[PageFamily.RESULTS],
        prerequisites=[SkillType.SEARCH],
        difficulty_weight=2.0,
    ),

    SkillType.OPEN_RESULT: Skill(
        skill_id=SkillType.OPEN_RESULT,
        name="打开文档",
        description="打开搜索结果中的某条文档详情",
        required_page_families=[PageFamily.RESULTS],
        prerequisites=[SkillType.SEARCH],
        difficulty_weight=1.0,
    ),

    SkillType.BACK_TO_RESULTS: Skill(
        skill_id=SkillType.BACK_TO_RESULTS,
        name="返回结果",
        description="从文档页返回搜索结果列表",
        required_page_families=[PageFamily.DOCUMENT],
        difficulty_weight=1.0,
    ),

    SkillType.ZOOM: Skill(
        skill_id=SkillType.ZOOM,
        name="放大查看",
        description="放大查看文档中的图片或细节文字",
        required_page_families=[PageFamily.RESULTS, PageFamily.DOCUMENT],
        difficulty_weight=2.0,
    ),

    SkillType.SCAN: Skill(
        skill_id=SkillType.SCAN,
        name="扫描结果",
        description="扫描搜索结果列表找到目标文档",
        required_page_families=[PageFamily.RESULTS],
        difficulty_weight=1.5,
    ),

    SkillType.CITE: Skill(
        skill_id=SkillType.CITE,
        name="引用证据",
        description="从当前文档中提取并引用关键证据",
        required_page_families=[PageFamily.DOCUMENT],
        prerequisites=[SkillType.OPEN_RESULT],
        difficulty_weight=2.0,
    ),

    SkillType.CROSS_REFERENCE: Skill(
        skill_id=SkillType.CROSS_REFERENCE,
        name="交叉验证",
        description="对比多个文档中的信息进行交叉验证",
        required_page_families=[PageFamily.DOCUMENT],
        prerequisites=[SkillType.CITE, SkillType.MEMORY],
        difficulty_weight=3.0,
    ),

    SkillType.COMPARE: Skill(
        skill_id=SkillType.COMPARE,
        name="比较结果",
        description="比较多条搜索结果的相关性和可信度",
        required_page_families=[PageFamily.RESULTS],
        prerequisites=[SkillType.SCAN],
        difficulty_weight=2.5,
    ),

    SkillType.EVALUATE: Skill(
        skill_id=SkillType.EVALUATE,
        name="评估匹配度",
        description="评估当前文档与研究目标的匹配程度",
        required_page_families=[PageFamily.DOCUMENT],
        prerequisites=[SkillType.OPEN_RESULT],
        difficulty_weight=2.0,
    ),

    SkillType.MEMORY: Skill(
        skill_id=SkillType.MEMORY,
        name="跨搜索记忆",
        description="记住之前搜索中看到的关键信息",
        required_page_families=[PageFamily.RESULTS, PageFamily.DOCUMENT],
        prerequisites=[SkillType.BACK_TO_RESULTS],
        difficulty_weight=3.0,
    ),

    SkillType.SUBMIT_REPORT: Skill(
        skill_id=SkillType.SUBMIT_REPORT,
        name="提交报告",
        description="整理收集的证据并提交研究报告",
        required_page_families=[PageFamily.RESULTS, PageFamily.DOCUMENT],
        prerequisites=[SkillType.CITE],
        difficulty_weight=1.0,
    ),
}


# ── 技能组合模板 ──────────────────────────────────────────

@dataclass
class SkillCombo:
    """技能组合（研究任务模板）"""
    combo_id: str
    skills: list[SkillType]
    difficulty: int  # 1-5
    description: str
    min_steps: int
    max_steps: int
    requires_zoom: bool = False
    requires_memory: bool = False
    requires_multi_search: bool = False


# 难度 1: 单次搜索 + 直接引用
DIFFICULTY_1_COMBOS = [
    SkillCombo(
        combo_id="d1_search_only",
        skills=[SkillType.SEARCH],
        difficulty=1,
        description="仅搜索",
        min_steps=1,
        max_steps=1,
    ),
    SkillCombo(
        combo_id="d1_search_open",
        skills=[SkillType.SEARCH, SkillType.OPEN_RESULT],
        difficulty=1,
        description="搜索后打开第一个结果",
        min_steps=2,
        max_steps=2,
    ),
    SkillCombo(
        combo_id="d1_search_open_cite",
        skills=[SkillType.SEARCH, SkillType.OPEN_RESULT, SkillType.CITE],
        difficulty=1,
        description="搜索、打开文档、引用证据",
        min_steps=3,
        max_steps=3,
    ),
]

# 难度 2: 搜索 + 浏览多个结果
DIFFICULTY_2_COMBOS = [
    SkillCombo(
        combo_id="d2_search_scan_open_cite",
        skills=[SkillType.SEARCH, SkillType.SCAN, SkillType.OPEN_RESULT, SkillType.CITE],
        difficulty=2,
        description="搜索、扫描结果、打开目标文档、引用",
        min_steps=3,
        max_steps=4,
    ),
    SkillCombo(
        combo_id="d2_search_open_cite_submit",
        skills=[SkillType.SEARCH, SkillType.OPEN_RESULT, SkillType.CITE, SkillType.SUBMIT_REPORT],
        difficulty=2,
        description="搜索、引用、提交报告",
        min_steps=4,
        max_steps=4,
    ),
    SkillCombo(
        combo_id="d2_search_zoom_cite",
        skills=[SkillType.SEARCH, SkillType.OPEN_RESULT, SkillType.ZOOM, SkillType.CITE],
        difficulty=2,
        description="搜索、打开文档、放大查看后引用",
        min_steps=4,
        max_steps=4,
        requires_zoom=True,
    ),
]

# 难度 3: 多文档比较 + 证据收集
DIFFICULTY_3_COMBOS = [
    SkillCombo(
        combo_id="d3_search_compare_cite",
        skills=[SkillType.SEARCH, SkillType.SCAN, SkillType.COMPARE,
                SkillType.OPEN_RESULT, SkillType.CITE, SkillType.SUBMIT_REPORT],
        difficulty=3,
        description="搜索、比较结果、选择最佳文档、引用、提交",
        min_steps=5,
        max_steps=6,
    ),
    SkillCombo(
        combo_id="d3_search_open_back_open_cite",
        skills=[SkillType.SEARCH, SkillType.OPEN_RESULT, SkillType.BACK_TO_RESULTS,
                SkillType.OPEN_RESULT, SkillType.CITE, SkillType.SUBMIT_REPORT],
        difficulty=3,
        description="搜索、浏览多个文档、引用最佳、提交",
        min_steps=5,
        max_steps=6,
    ),
    SkillCombo(
        combo_id="d3_refine_and_cite",
        skills=[SkillType.SEARCH, SkillType.REFINE_QUERY, SkillType.OPEN_RESULT,
                SkillType.CITE, SkillType.SUBMIT_REPORT],
        difficulty=3,
        description="搜索、改写query重搜、引用、提交",
        min_steps=5,
        max_steps=6,
        requires_multi_search=True,
    ),
]

# 难度 4: 跨搜索记忆 + 多轮证据收集
DIFFICULTY_4_COMBOS = [
    SkillCombo(
        combo_id="d4_multi_search_memory",
        skills=[SkillType.SEARCH, SkillType.OPEN_RESULT, SkillType.CITE,
                SkillType.BACK_TO_RESULTS, SkillType.MEMORY,
                SkillType.REFINE_QUERY, SkillType.OPEN_RESULT, SkillType.CITE,
                SkillType.SUBMIT_REPORT],
        difficulty=4,
        description="多轮搜索、跨搜索记忆、收集多条证据、提交",
        min_steps=7,
        max_steps=9,
        requires_memory=True,
        requires_multi_search=True,
    ),
    SkillCombo(
        combo_id="d4_zoom_cross_reference",
        skills=[SkillType.SEARCH, SkillType.OPEN_RESULT, SkillType.ZOOM,
                SkillType.CITE, SkillType.BACK_TO_RESULTS,
                SkillType.OPEN_RESULT, SkillType.CITE,
                SkillType.CROSS_REFERENCE, SkillType.SUBMIT_REPORT],
        difficulty=4,
        description="搜索、放大查看、引用、交叉验证、提交",
        min_steps=7,
        max_steps=9,
        requires_zoom=True,
        requires_memory=True,
    ),
]

# 难度 5: 长链深度研究
DIFFICULTY_5_COMBOS = [
    SkillCombo(
        combo_id="d5_deep_research_chain",
        skills=[SkillType.SEARCH, SkillType.SCAN, SkillType.OPEN_RESULT,
                SkillType.ZOOM, SkillType.CITE, SkillType.BACK_TO_RESULTS,
                SkillType.MEMORY, SkillType.REFINE_QUERY,
                SkillType.OPEN_RESULT, SkillType.EVALUATE, SkillType.CITE,
                SkillType.CROSS_REFERENCE, SkillType.SUBMIT_REPORT],
        difficulty=5,
        description="深度研究链：多轮搜索、放大验证、跨文档比对、综合提交",
        min_steps=9,
        max_steps=13,
        requires_zoom=True,
        requires_memory=True,
        requires_multi_search=True,
    ),
]

ALL_COMBOS = (
    DIFFICULTY_1_COMBOS +
    DIFFICULTY_2_COMBOS +
    DIFFICULTY_3_COMBOS +
    DIFFICULTY_4_COMBOS +
    DIFFICULTY_5_COMBOS
)


# ── 技能图谱 ──────────────────────────────────────────────

class SkillGraph:
    """技能依赖图和组合管理"""

    def __init__(self):
        self.skills = SKILL_DEFINITIONS
        self.combos = ALL_COMBOS

    def get_skill(self, skill_id: SkillType) -> Skill:
        return self.skills[skill_id]

    def get_prerequisites(self, skill_id: SkillType) -> list[SkillType]:
        return self.skills[skill_id].prerequisites

    def get_combos_by_difficulty(self, difficulty: int) -> list[SkillCombo]:
        return [c for c in self.combos if c.difficulty == difficulty]

    def get_combo_by_id(self, combo_id: str) -> Optional[SkillCombo]:
        for combo in self.combos:
            if combo.combo_id == combo_id:
                return combo
        return None

    def calculate_combo_difficulty(self, skills: list[SkillType]) -> float:
        if not skills:
            return 0.0
        base_difficulty = len(skills) * 0.5
        skill_weights = sum(self.skills[s].difficulty_weight for s in skills)
        prereq_penalty = sum(
            len(self.skills[s].prerequisites) * 0.3
            for s in skills
        )
        return base_difficulty + skill_weights + prereq_penalty

    def validate_combo(self, skills: list[SkillType]) -> tuple[bool, str]:
        seen = set()
        for skill in skills:
            prereqs = self.get_prerequisites(skill)
            missing = [p for p in prereqs if p not in seen]
            if missing:
                return False, f"Skill {skill} requires {missing} but not seen yet"
            seen.add(skill)
        return True, "OK"

    def get_statistics(self) -> dict:
        return {
            "total_skills": len(self.skills),
            "atomic_skills": sum(1 for s in self.skills.values() if s.atomic),
            "total_combos": len(self.combos),
            "difficulty_distribution": {
                i: len(self.get_combos_by_difficulty(i))
                for i in range(1, 6)
            },
        }
