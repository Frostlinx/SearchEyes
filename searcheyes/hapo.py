"""
hapo.py — Hop-Anchored Policy Optimization (HaPO) with SAPO
==============================================================
Core RL algorithm for training multi-hop search agents.

Key innovations:
  1. Hop-Anchored Credit Assignment: Uses gold entity IDs from multi-hop
     reasoning chains as semantic anchors for step-level grouping.
     Unlike GiGPO's exact state match, we use entity ID presence in
     observations as a soft semantic anchor — works in stochastic
     search environments where exact state repetition is rare.

  2. SAPO Optimizer: Replaces hard PPO/GRPO clipping with smooth
     temperature-controlled sigmoid gate for stable long-horizon training.

  3. Fatal-Aware Masking: Masks gradient contribution from steps after
     consecutive tool errors to prevent learning from degenerate suffixes.

Algorithm:
  Â_final(i, t) = α · Â_episode(i) + (1-α) · Â_hop(i, t)

  where:
    - Â_episode: standard GRPO trajectory-level group-relative advantage
    - Â_hop: hop-anchored step-level advantage computed by grouping
      trajectories that hit the same gold entity, then comparing
      post-anchor outcomes within each group

References:
  - GiGPO: Group-in-Group Policy Optimization (NeurIPS 2025)
  - SAPO: Soft Adaptive Policy Optimization (Qwen, 2025)
  - HGPO: Hierarchy-of-Groups Policy Optimization (ICLR 2026)
"""

import re
import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. Data Structures
# ═══════════════════════════════════════════════════════════

class StepData(object):
    """Single step in a trajectory."""

    def __init__(self, step_index, action_token_ids=None, observation_text="",
                 retrieved_entity_ids=None, is_error=False, num_tokens=0):
        self.step_index = step_index
        self.action_token_ids = action_token_ids or []
        self.observation_text = observation_text
        self.retrieved_entity_ids = retrieved_entity_ids or set()
        self.is_error = is_error
        self.num_tokens = num_tokens


class TrajectoryData(object):
    """Complete trajectory for one rollout."""

    def __init__(self, trajectory_index, question_id="", gold_chain_entity_ids=None,
                 steps=None, outcome_reward=0.0, is_correct=False, total_tokens=0):
        self.trajectory_index = trajectory_index
        self.question_id = question_id
        self.gold_chain_entity_ids = gold_chain_entity_ids or []
        self.steps = steps or []
        self.outcome_reward = outcome_reward
        self.is_correct = is_correct
        self.total_tokens = total_tokens


class HopAnchorGroup(object):
    """A group of (trajectory, step) pairs sharing the same hop entity anchor."""

    def __init__(self, anchor_entity_id, anchor_entity_title="",
                 members=None, member_outcomes=None):
        self.anchor_entity_id = anchor_entity_id
        self.anchor_entity_title = anchor_entity_title
        self.members = members or []
        self.member_outcomes = member_outcomes or []


# ═══════════════════════════════════════════════════════════
# 2. SAPO Loss Module
# ═══════════════════════════════════════════════════════════

class SAPOLoss:
    """Soft Adaptive Policy Optimization loss function.

    Replaces hard clipping with smooth sigmoid gate:
        gate(r) = σ(τ · (r - 1)) · 4/τ
        loss = -gate(ratio) · advantage

    Asymmetric temperatures: τ_neg > τ_pos for stability.
    """

    def __init__(self, tau_pos: float = 1.0, tau_neg: float = 1.05):
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def compute_loss(
        self,
        log_ratios: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SAPO per-token loss.

        Args:
            log_ratios: (batch_size, seq_len) log(π_θ / π_old) per token
            advantages: (batch_size,) or (batch_size, seq_len) advantage values
            action_mask: (batch_size, seq_len) binary mask for action tokens

        Returns:
            Scalar loss value
        """
        ratios = torch.exp(log_ratios)

        if advantages.dim() == 1:
            advantages_expanded = advantages.unsqueeze(1).expand_as(ratios)
        else:
            advantages_expanded = advantages

        gate_pos = torch.sigmoid(self.tau_pos * (ratios - 1)) * (4.0 / self.tau_pos)
        gate_neg = torch.sigmoid(self.tau_neg * (ratios - 1)) * (4.0 / self.tau_neg)
        is_positive = advantages_expanded > 0
        soft_gate = torch.where(is_positive, gate_pos, gate_neg)

        per_token_loss = -soft_gate * advantages_expanded

        masked_loss = per_token_loss * action_mask
        token_count = action_mask.sum()
        if token_count > 0:
            return masked_loss.sum() / token_count
        return masked_loss.sum()


# ═══════════════════════════════════════════════════════════
# 3. Hop-Anchored Advantage Computation
# ═══════════════════════════════════════════════════════════

def extract_entity_ids_from_observation(observation: str) -> Set[str]:
    """Extract Wikidata entity IDs from observation text.

    Observations contain patterns like: '[1] Nokia, Finland (ID: Q192870)'
    """
    return set(re.findall(r"ID:\s*(Q\d+)", observation))


def find_hop_anchor_groups(
    trajectories: List[TrajectoryData],
    gold_chain_entities: List[str],
) -> List[HopAnchorGroup]:
    """Identify hop-anchored groups across trajectories.

    For each gold entity in the reasoning chain, find which trajectories
    first retrieved it and at which step. Trajectories sharing the same
    hop anchor are grouped together for step-level advantage comparison.

    Args:
        trajectories: List of trajectory rollouts for the same question
        gold_chain_entities: Ordered list of gold entity IDs in the hop chain

    Returns:
        List of HopAnchorGroup, one per gold entity that appears in >= 2 trajectories
    """
    anchor_groups = []

    for entity_id in gold_chain_entities:
        group = HopAnchorGroup(anchor_entity_id=entity_id)

        for traj in trajectories:
            first_hit_step = None
            for step in traj.steps:
                if entity_id in step.retrieved_entity_ids:
                    first_hit_step = step.step_index
                    break

            if first_hit_step is not None:
                group.members.append((traj.trajectory_index, first_hit_step))
                group.member_outcomes.append(traj.outcome_reward)

        if len(group.members) >= 2:
            anchor_groups.append(group)

    return anchor_groups


def compute_hop_anchored_advantages(
    trajectories,  # type: List[TrajectoryData]
    gold_chain_entities,  # type: List[str]
    epsilon=1e-6,  # type: float
):
    # type: (...) -> Dict[Tuple[int, int], float]
    """Compute hop-anchored step-level advantages.

    For each hop anchor group:
      1. Group trajectories that hit the same gold entity
      2. Compare their outcomes (group-relative normalization)
      3. Assign the normalized advantage to the anchor step AND all subsequent
         steps, using a "latest anchor wins" policy (closer anchor = more
         precise credit assignment).

    This mirrors GiGPO's core insight: actions taken FROM the same state
    (= anchor) should be compared group-relatively. Here the "state" is
    defined semantically by having retrieved the same gold entity.

    Args:
        trajectories: Rollout trajectories for one question
        gold_chain_entities: Gold entity IDs in hop chain order
        epsilon: Numerical stability constant

    Returns:
        Dict mapping (trajectory_index, step_index) -> hop_advantage
    """
    anchor_groups = find_hop_anchor_groups(trajectories, gold_chain_entities)

    # Pre-compute per-group statistics: only groups with variance > 0
    # provide meaningful credit assignment signal.
    group_info = []  # (group, std, per-member normalized advantages)
    for group in anchor_groups:
        outcomes = torch.tensor(group.member_outcomes, dtype=torch.float32)
        mean_outcome = outcomes.mean().item()
        std_outcome = outcomes.std().item() if len(outcomes) > 1 else 0.0
        has_variance = std_outcome > epsilon
        member_advs = []
        for outcome in group.member_outcomes:
            if has_variance:
                member_advs.append((outcome - mean_outcome) / (std_outcome + epsilon))
            else:
                member_advs.append(0.0)
        group_info.append((group, has_variance, member_advs))

    # For each (traj, step), track the advantage from the LATEST (nearest)
    # anchor that has discriminative power (variance > 0).
    # Key: (traj_idx, step_index) -> (anchor_step, advantage)
    hop_with_anchor_step = {}  # type: Dict[Tuple[int, int], Tuple[int, float]]

    for group, has_variance, member_advs in group_info:
        # Skip groups where all members have the same outcome —
        # they cannot provide step-level credit differentiation.
        if not has_variance:
            continue

        for idx, ((traj_idx, anchor_step), adv) in enumerate(
            zip(group.members, member_advs)
        ):
            traj = trajectories[traj_idx]
            for step in traj.steps:
                # Assign to anchor step itself AND all subsequent steps
                if step.step_index >= anchor_step:
                    key = (traj_idx, step.step_index)
                    existing = hop_with_anchor_step.get(key)
                    # "Latest informative anchor wins": overwrite only if
                    # this anchor is more recent (closer to the step)
                    if existing is None or anchor_step > existing[0]:
                        hop_with_anchor_step[key] = (anchor_step, adv)

    # Extract just the advantage values
    hop_advantages = {}  # type: Dict[Tuple[int, int], float]
    for key, (_, adv) in hop_with_anchor_step.items():
        hop_advantages[key] = adv

    return hop_advantages


# ═══════════════════════════════════════════════════════════
# 4. Episode-Level Advantage (Standard GRPO)
# ═══════════════════════════════════════════════════════════

def compute_episode_advantages(
    trajectories: List[TrajectoryData],
    epsilon: float = 1e-6,
) -> List[float]:
    """Standard GRPO episode-level group-relative advantage.

    Args:
        trajectories: Rollout trajectories for one question

    Returns:
        List of normalized advantages, one per trajectory
    """
    outcomes = torch.tensor(
        [t.outcome_reward for t in trajectories], dtype=torch.float32
    )
    mean_outcome = outcomes.mean().item()
    std_outcome = outcomes.std().item() if len(outcomes) > 1 else 1.0

    advantages = []
    for t in trajectories:
        adv = (t.outcome_reward - mean_outcome) / (std_outcome + epsilon)
        advantages.append(adv)

    return advantages


# ═══════════════════════════════════════════════════════════
# 5. Fatal-Aware Masking
# ═══════════════════════════════════════════════════════════

def detect_fatal_step(trajectory: TrajectoryData, max_consecutive_errors: int = 3) -> int:
    """Detect fatal step index where consecutive errors exceed threshold.

    After the fatal step, all subsequent tokens should be masked from
    gradient updates to prevent learning from degenerate suffixes.

    Args:
        trajectory: Single trajectory
        max_consecutive_errors: Number of consecutive errors before fatal

    Returns:
        Step index of fatal point, or -1 if no fatal detected
    """
    consecutive_errors = 0
    for step in trajectory.steps:
        if step.is_error:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                return step.step_index - max_consecutive_errors + 1
        else:
            consecutive_errors = 0
    return -1


def apply_fatal_masking(
    advantages: Dict[Tuple[int, int], float],
    trajectory: TrajectoryData,
    max_consecutive_errors: int = 3,
) -> Dict[Tuple[int, int], float]:
    """Zero out advantages for steps after fatal point.

    Also applies one-sided clamping: for fatal trajectories,
    only allow non-negative advantages (don't penalize the valid prefix).
    """
    fatal_step = detect_fatal_step(trajectory, max_consecutive_errors)
    if fatal_step < 0:
        return advantages

    traj_idx = trajectory.trajectory_index
    masked = {}
    for (ti, si), adv in advantages.items():
        if ti != traj_idx:
            masked[(ti, si)] = adv
        elif si >= fatal_step:
            masked[(ti, si)] = 0.0
        else:
            masked[(ti, si)] = max(adv, 0.0)

    return masked


# ═══════════════════════════════════════════════════════════
# 6. Combined HaPO Advantage
# ═══════════════════════════════════════════════════════════

def compute_hapo_advantages(
    trajectories: List[TrajectoryData],
    gold_chain_entities: List[str],
    alpha: float = 0.5,
    epsilon: float = 1e-6,
    max_consecutive_errors: int = 3,
) -> Dict[Tuple[int, int], float]:
    """Compute final HaPO advantages combining episode + hop-anchored levels.

    Â_final(i, t) = α · Â_episode(i) + (1-α) · Â_hop(i, t)

    Args:
        trajectories: All rollout trajectories for one question
        gold_chain_entities: Gold entity IDs in the multi-hop chain
        alpha: Mixing coefficient (0=pure hop, 1=pure episode)
        epsilon: Numerical stability
        max_consecutive_errors: Fatal masking threshold

    Returns:
        Dict mapping (trajectory_index, step_index) -> final_advantage
    """
    episode_advantages = compute_episode_advantages(trajectories, epsilon)
    hop_advantages = compute_hop_anchored_advantages(
        trajectories, gold_chain_entities, epsilon
    )

    final_advantages: Dict[Tuple[int, int], float] = {}

    for traj in trajectories:
        traj_idx = traj.trajectory_index
        episode_adv = episode_advantages[traj_idx]

        for step in traj.steps:
            key = (traj_idx, step.step_index)
            hop_adv = hop_advantages.get(key, 0.0)
            combined = alpha * episode_adv + (1.0 - alpha) * hop_adv
            final_advantages[key] = combined

        final_advantages = apply_fatal_masking(
            final_advantages, traj, max_consecutive_errors
        )

    return final_advantages


# ═══════════════════════════════════════════════════════════
# 7. Full HaPO Training Step
# ═══════════════════════════════════════════════════════════

class HaPOConfig(object):
    """Configuration for HaPO training."""

    def __init__(self, alpha=0.5, tau_pos=1.0, tau_neg=1.05, kl_coef=0.01,
                 epsilon=1e-6, max_consecutive_errors=3, observation_loss_mask=True):
        self.alpha = alpha
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.kl_coef = kl_coef
        self.epsilon = epsilon
        self.max_consecutive_errors = max_consecutive_errors
        self.observation_loss_mask = observation_loss_mask


class HaPOTrainer:
    """HaPO training logic.

    Combines:
      - Hop-anchored advantage computation
      - SAPO loss function
      - Fatal-aware masking
      - Observation token masking
      - KL penalty
    """

    def __init__(self, config: HaPOConfig):
        self.config = config
        self.sapo_loss = SAPOLoss(
            tau_pos=config.tau_pos, tau_neg=config.tau_neg
        )

    def compute_training_loss(
        self,
        trajectories: List[TrajectoryData],
        gold_chain_entities: List[str],
        log_probs_current: torch.Tensor,
        log_probs_old: torch.Tensor,
        log_probs_ref: torch.Tensor,
        action_mask: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute full HaPO training loss for a batch of trajectories.

        Args:
            trajectories: Rollout data for one question (G trajectories)
            gold_chain_entities: Gold entity IDs in hop chain
            log_probs_current: (G, max_seq_len) current policy log probs
            log_probs_old: (G, max_seq_len) behavior policy log probs
            log_probs_ref: (G, max_seq_len) reference policy log probs
            action_mask: (G, max_seq_len) 1 for action tokens, 0 for others
            observation_mask: (G, max_seq_len) 1 for observation tokens

        Returns:
            Dict with 'loss', 'policy_loss', 'kl_loss', 'metrics'
        """
        advantages_dict = compute_hapo_advantages(
            trajectories=trajectories,
            gold_chain_entities=gold_chain_entities,
            alpha=self.config.alpha,
            epsilon=self.config.epsilon,
            max_consecutive_errors=self.config.max_consecutive_errors,
        )

        batch_size = log_probs_current.shape[0]
        seq_len = log_probs_current.shape[1]
        advantages_tensor = torch.zeros(batch_size, seq_len, device=log_probs_current.device)

        for traj in trajectories:
            traj_idx = traj.trajectory_index
            for step in traj.steps:
                key = (traj_idx, step.step_index)
                adv_value = advantages_dict.get(key, 0.0)
                start_token = self._get_step_token_start(traj, step.step_index)
                end_token = self._get_step_token_end(traj, step.step_index)
                advantages_tensor[traj_idx, start_token:end_token] = adv_value

        effective_mask = action_mask.clone()
        if self.config.observation_loss_mask:
            effective_mask = effective_mask * (1 - observation_mask)

        log_ratios = log_probs_current - log_probs_old
        policy_loss = self.sapo_loss.compute_loss(
            log_ratios=log_ratios,
            advantages=advantages_tensor,
            action_mask=effective_mask,
        )

        kl_divergence = (log_probs_current - log_probs_ref) * effective_mask
        kl_loss = self.config.kl_coef * kl_divergence.sum() / effective_mask.sum().clamp(min=1)

        total_loss = policy_loss + kl_loss

        num_hop_anchors = len([
            g for g in find_hop_anchor_groups(trajectories, gold_chain_entities)
        ])
        metrics = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "total_loss": total_loss.detach(),
            "num_hop_anchors": torch.tensor(float(num_hop_anchors)),
            "mean_advantage": advantages_tensor[effective_mask.bool()].mean().detach()
            if effective_mask.sum() > 0 else torch.tensor(0.0),
            "num_correct": sum(1 for t in trajectories if t.is_correct),
            "num_trajectories": len(trajectories),
        }

        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "metrics": metrics,
        }

    def _get_step_token_start(self, traj: TrajectoryData, step_index: int) -> int:
        """Get token start position for a given step.

        In practice, this maps step_index to token positions in the
        flattened sequence. Implementation depends on tokenization strategy.
        """
        offset = 0
        for step in traj.steps:
            if step.step_index == step_index:
                return offset
            offset += step.num_tokens
        return offset

    def _get_step_token_end(self, traj: TrajectoryData, step_index: int) -> int:
        """Get token end position for a given step."""
        offset = 0
        for step in traj.steps:
            offset += step.num_tokens
            if step.step_index == step_index:
                return offset
        return offset


# ═══════════════════════════════════════════════════════════
# 8. Trajectory Parsing Utilities
# ═══════════════════════════════════════════════════════════

def parse_trajectory_from_rollout(
    rollout_data: dict,
    question_metadata: dict,
    trajectory_index: int = 0,
) -> TrajectoryData:
    """Parse a raw rollout into structured TrajectoryData.

    Args:
        rollout_data: Raw rollout dict with 'steps' field
        question_metadata: Question metadata with 'chain' field
        trajectory_index: Index in the rollout group

    Returns:
        Parsed TrajectoryData
    """
    gold_entities = []
    chain = question_metadata.get("chain", [])
    if chain:
        gold_entities.append(chain[0].get("from_qid", ""))
    for hop in chain:
        gold_entities.append(hop.get("to_qid", ""))

    steps = []
    for raw_step in rollout_data.get("steps", []):
        obs = raw_step.get("raw_observation", "")
        action = raw_step.get("action", "")
        retrieved_ids = extract_entity_ids_from_observation(obs)

        if "read_entity" in action:
            eid_match = re.search(r'entity_id":\s*"(Q\d+)"', action)
            if eid_match:
                retrieved_ids.add(eid_match.group(1))

        is_error = "error" in obs.lower() or "invalid" in obs.lower()

        step = StepData(
            step_index=raw_step.get("step_index", len(steps)),
            observation_text=obs,
            retrieved_entity_ids=retrieved_ids,
            is_error=is_error,
            num_tokens=raw_step.get("num_tokens", 100),
        )
        steps.append(step)

    is_correct = rollout_data.get("is_correct", False)
    outcome_reward = 1.0 if is_correct else 0.0

    return TrajectoryData(
        trajectory_index=trajectory_index,
        question_id=rollout_data.get("question_id", ""),
        gold_chain_entity_ids=gold_entities,
        steps=steps,
        outcome_reward=outcome_reward,
        is_correct=is_correct,
        total_tokens=sum(s.num_tokens for s in steps),
    )


# ═══════════════════════════════════════════════════════════
# 9. Repetition Detection & Early Stopping
# ═══════════════════════════════════════════════════════════

def detect_repetition(
    actions: List[str],
    ngram_size: int = 3,
    threshold: float = 0.5,
) -> bool:
    """Detect if agent is stuck in repetitive loop.

    Checks if recent actions contain repeated n-grams above threshold.

    Args:
        actions: List of recent action strings
        ngram_size: Size of n-grams to check
        threshold: Ratio of repeated n-grams to trigger early stop

    Returns:
        True if repetition detected
    """
    if len(actions) < ngram_size * 2:
        return False

    recent = actions[-ngram_size * 4:]
    ngrams = []
    for i in range(len(recent) - ngram_size + 1):
        ngram = tuple(recent[i:i + ngram_size])
        ngrams.append(ngram)

    if not ngrams:
        return False

    unique_ratio = len(set(ngrams)) / len(ngrams)
    return unique_ratio < (1.0 - threshold)


# ═══════════════════════════════════════════════════════════
# 10. Convenience: Compute Advantages for verl-agent Integration
# ═══════════════════════════════════════════════════════════

def compute_hapo_token_advantages(
    token_level_rewards,  # type: torch.Tensor
    response_mask,        # type: torch.Tensor
    index,                # type: np.ndarray
    traj_index,           # type: np.ndarray
    hop_anchor_data,      # type: dict
    alpha=0.5,            # type: float
    epsilon=1e-6,         # type: float
):
    # type: (...) -> Tuple[torch.Tensor, torch.Tensor]
    """verl-agent compatible interface for HaPO advantage computation.

    This function provides a drop-in replacement for
    `compute_grpo_outcome_advantage` in verl-agent's core_algos.py.

    Args:
        token_level_rewards: (bs, response_length) — outcome reward at last token
        response_mask: (bs, response_length) — valid response token mask
        index: (bs,) — question/prompt group index
        traj_index: (bs,) — trajectory index within group
        hop_anchor_data: dict mapping batch_idx -> {
            'gold_entities': List[str],
            'step_entity_hits': Dict[int, Set[str]],  # step_idx -> retrieved entity IDs
            'step_boundaries': List[Tuple[int, int]],  # (start_token, end_token) per step
        }
        alpha: Mixing weight for episode vs hop advantage
        epsilon: Numerical stability

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) (same as advantages for outcome-only)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    bs, seq_len = token_level_rewards.shape
    advantages = torch.zeros(bs, seq_len, device=token_level_rewards.device)

    # Group by question/prompt index (same as GRPO)
    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        for i in range(bs):
            idx = index[i]
            id2scores[idx].append(scores[i].item())
            id2indices[idx].append(i)

        for idx in id2scores:
            group_scores_list = id2scores[idx]
            group_scores_t = torch.tensor(group_scores_list, dtype=torch.float32)
            group_mean = group_scores_t.mean().item()
            group_std = group_scores_t.std().item() if len(group_scores_t) > 1 else 1.0

            batch_indices = id2indices[idx]

            # --- Episode-level advantages (standard GRPO) ---
            episode_advs = {}
            for bi in batch_indices:
                episode_advs[bi] = (scores[bi].item() - group_mean) / (group_std + epsilon)

            # --- Hop-anchored advantages ---
            # Reconstruct TrajectoryData-like structures for the group
            # to reuse the core hop advantage logic
            hop_advs = _compute_group_hop_advantages(
                batch_indices, group_scores_list, hop_anchor_data, epsilon
            )

            # --- Combine and assign to token positions ---
            for list_idx, bi in enumerate(batch_indices):
                if bi in hop_anchor_data:
                    step_bounds = hop_anchor_data[bi]["step_boundaries"]
                    for step_idx, (start, end) in enumerate(step_bounds):
                        hop_adv = hop_advs.get((bi, step_idx), 0.0)
                        combined = alpha * episode_advs[bi] + (1.0 - alpha) * hop_adv
                        advantages[bi, start:end] = combined
                else:
                    advantages[bi] = episode_advs[bi] * response_mask[bi]

    return advantages, advantages


def _compute_group_hop_advantages(
    batch_indices,   # type: List[int]
    group_scores,    # type: List[float]
    hop_anchor_data, # type: dict
    epsilon,         # type: float
):
    # type: (...) -> Dict[Tuple[int, int], float]
    """Compute hop-anchored step advantages for a group of trajectories.

    Mirrors the core `compute_hop_anchored_advantages` logic:
    - For each gold entity, find which trajectories hit it and at which step
    - Form anchor groups, skip groups with zero variance
    - "Latest informative anchor wins" for each (traj, step)

    Returns:
        Dict mapping (batch_idx, step_idx) -> hop_advantage
    """
    # Collect gold entities (should be same for all in group)
    gold_entities = None
    for bi in batch_indices:
        if bi in hop_anchor_data:
            gold_entities = hop_anchor_data[bi]["gold_entities"]
            break
    if gold_entities is None:
        return {}

    # For each gold entity, form anchor group: which trajs hit it, at which step
    # anchor_groups[entity_id] = [(list_idx, first_hit_step, score)]
    anchor_groups = {}
    for entity_id in gold_entities:
        members = []  # (list_idx, batch_idx, first_hit_step, score)
        for list_idx, bi in enumerate(batch_indices):
            if bi not in hop_anchor_data:
                continue
            step_hits = hop_anchor_data[bi].get("step_entity_hits", {})
            first_hit = None
            for s_idx in sorted(step_hits.keys()):
                if entity_id in step_hits[s_idx]:
                    first_hit = s_idx
                    break
            if first_hit is not None:
                members.append((list_idx, bi, first_hit, group_scores[list_idx]))
        if len(members) >= 2:
            anchor_groups[entity_id] = members

    # For each (batch_idx, step_idx), track: (anchor_step, advantage)
    # "Latest informative anchor wins"
    hop_with_anchor = {}  # type: Dict[Tuple[int, int], Tuple[int, float]]

    for entity_id, members in anchor_groups.items():
        member_scores = torch.tensor([m[3] for m in members], dtype=torch.float32)
        std_score = member_scores.std().item() if len(member_scores) > 1 else 0.0

        # Skip groups with no variance (cannot provide credit differentiation)
        if std_score < epsilon:
            continue

        mean_score = member_scores.mean().item()

        for (list_idx, bi, anchor_step, score) in members:
            normalized_adv = (score - mean_score) / (std_score + epsilon)

            if bi not in hop_anchor_data:
                continue
            step_bounds = hop_anchor_data[bi]["step_boundaries"]

            for step_idx in range(len(step_bounds)):
                # Assign to anchor step and all subsequent steps
                if step_idx >= anchor_step:
                    key = (bi, step_idx)
                    existing = hop_with_anchor.get(key)
                    # Latest informative anchor wins
                    if existing is None or anchor_step > existing[0]:
                        hop_with_anchor[key] = (anchor_step, normalized_adv)

    # Extract advantage values
    result = {}
    for key, (_, adv) in hop_with_anchor.items():
        result[key] = adv

    return result

