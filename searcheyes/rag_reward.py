"""
rag_reward.py — RAG 命中率 Reward 计算
=========================================
将 RAG 检索质量转化为 RL 中途奖励信号。
用于在训练中激励 Agent 有效利用多模态知识检索。
"""

from __future__ import annotations


def compute_rag_hit_reward(
    rag_facts: list[dict],
    ground_truth_fact: str,
    ground_truth_wit_id: str = "",
) -> float:
    """
    计算 RAG 检索命中奖励。

    Args:
        rag_facts: RAG 返回的 fact 列表，每个 dict 包含:
            - wit_id: str
            - fact_text: str
            - relevance_score: float
        ground_truth_fact: 正确的 fact 文本
        ground_truth_wit_id: 正确的 WIT 条目 ID

    Returns:
        reward 值，范围 [-0.3, 0.5]:
        - 精确 WIT ID 命中 top-k:  +0.5
        - 高文本相似度 (>50%):      +0.3
        - 中等相似度 (>20%):        +0.1
        - 无结果或低质量:            -0.1
        - RAG 调用但全部低于阈值:    -0.3
    """
    if not rag_facts:
        return -0.1  # 未调用 RAG 或无结果

    # 检查精确 ID 命中
    if ground_truth_wit_id:
        for fact in rag_facts:
            if fact.get("wit_id") == ground_truth_wit_id:
                return 0.5

    # 检查文本相似度（token overlap）
    if not ground_truth_fact:
        return -0.1

    gt_tokens = _tokenize(ground_truth_fact)
    if not gt_tokens:
        return -0.1

    best_overlap = 0.0
    for fact in rag_facts:
        fact_text = fact.get("fact_text", "")
        fact_tokens = _tokenize(fact_text)
        if not fact_tokens:
            continue
        overlap = len(gt_tokens & fact_tokens) / len(gt_tokens)
        best_overlap = max(best_overlap, overlap)

    if best_overlap > 0.5:
        return 0.3
    elif best_overlap > 0.2:
        return 0.1
    elif best_overlap > 0.05:
        return -0.1
    else:
        return -0.3  # 全部不相关


def _tokenize(text: str) -> set[str]:
    """简单分词：小写 + 按空格分割 + 过滤短 token"""
    return {t for t in text.lower().split() if len(t) > 2}


def compute_rag_step_reward(
    action: str,
    rag_facts: list[dict] | None,
    ground_truth_fact: str = "",
    ground_truth_wit_id: str = "",
) -> float:
    """
    结合动作类型计算 RAG 步骤奖励。

    - 如果当前步是 zoom 且 RAG 返回了有效结果: 额外奖励
    - 如果当前步不涉及 RAG: 返回 0（中性）
    """
    if rag_facts is None:
        return 0.0  # 本步未触发 RAG

    base_reward = compute_rag_hit_reward(
        rag_facts, ground_truth_fact, ground_truth_wit_id
    )

    # zoom 后 RAG 命中，额外加成（鼓励 zoom → RAG 链路）
    if action == "zoom" and base_reward > 0:
        return base_reward * 1.2

    return base_reward
