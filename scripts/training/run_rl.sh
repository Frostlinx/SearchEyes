#!/bin/bash
set -x

# ═══════════════════════════════════════════════════════════
# SearchEyes RL Training with HaPO + SAPO
# Model: Qwen3.5-9B (SFT checkpoint)
# Data:  16225 multi-hop KB search questions
# GPUs:  8×H20
# ═══════════════════════════════════════════════════════════

# ── NCCL fixes for container environment ──
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export PYTHONUNBUFFERED=1
export PYTHONPATH=".:${PYTHONPATH}"

# ── Paths ──
MODEL_PATH="/dev/shm/sft_output/final"
TRAIN_DATA="/dev/shm/searcheyes_rl/train.parquet"
VAL_DATA="/dev/shm/searcheyes_rl/val.parquet"
TOOL_CONFIG="./tool_config.yaml"
REWARD_FN_PATH="./reward_fn.py"
REWARD_FN_NAME="searcheyes_compute_score"
OUTPUT_DIR="/dev/shm/searcheyes_rl_output"
PYTHON="python"

$PYTHON -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=hapo \
    +algorithm.hapo.alpha=0.5 \
    algorithm.gamma=1.0 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.return_raw_chat=True \
    data.shuffle=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=$REWARD_FN_NAME \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
    actor_rollout_ref.rollout.multi_turn.format=react \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=searcheyes \
    trainer.experiment_name=hapo_qwen35_9b \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=99999 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 \
    trainer.val_before_train=False \
    "$@"
