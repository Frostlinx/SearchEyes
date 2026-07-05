from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
_LEGACY_PROJECT_MARKERS = [
    "/QWEN-project/",
    "\\QWEN-project\\",
]
_LEGACY_PREFIXES = [
    ".",
    ".",
]


class QwenGRPOTrainerMixin:
    def _generate_and_score_completions(self, inputs):
        if self._is_multistep_mode():
            prepared_inputs = self._prepare_multistep_inputs(inputs)
            self._current_rollout_inputs = prepared_inputs
            try:
                output = super()._generate_and_score_completions(prepared_inputs)
            finally:
                self._current_rollout_inputs = None
            return self._inject_qwen_mm_token_type_ids(output, prepared_inputs)

        output = super()._generate_and_score_completions(inputs)
        return self._inject_qwen_mm_token_type_ids(output, inputs)

    def _inject_qwen_mm_token_type_ids(self, output, inputs):

        if not inputs:
            return output
        if "image" not in inputs[0] and "images" not in inputs[0]:
            return output

        images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        if all(image_list is None for image_list in images):
            return output

        prompts = [example["prompt"] for example in inputs]
        prompts_text = [
            self.processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            for prompt in prompts
        ]
        prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
        if "mm_token_type_ids" in prompt_inputs:
            output["token_type_ids"] = prompt_inputs["mm_token_type_ids"].to(output["prompt_ids"].device)
        return output

    def _generate_single_turn(self, prompts):
        if self._is_multistep_mode():
            prepared_inputs = getattr(self, "_current_rollout_inputs", None)
            if prepared_inputs is None:
                raise RuntimeError("multistep rollout 缺少当前 batch 上下文")
            return self._generate_multistep_rollouts(prepared_inputs)
        return super()._generate_single_turn(prompts)

    def _get_per_token_logps_and_entropies(self, *args, mm_token_type_ids=None, **kwargs):
        import torch

        input_ids = args[1] if len(args) > 1 else kwargs.get("input_ids")
        token_type_ids = kwargs.get("token_type_ids")
        if mm_token_type_ids is None and token_type_ids is not None:
            mm_token_type_ids = token_type_ids
            kwargs["token_type_ids"] = None
        if mm_token_type_ids is not None and input_ids is not None:
            target_width = input_ids.shape[1]
            current_width = mm_token_type_ids.shape[1]
            if current_width < target_width:
                pad = mm_token_type_ids.new_zeros((mm_token_type_ids.shape[0], target_width - current_width))
                mm_token_type_ids = torch.cat([mm_token_type_ids, pad], dim=1)
        return super()._get_per_token_logps_and_entropies(
            *args,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

    def _is_multistep_mode(self) -> bool:
        return getattr(self, "_grpo_rollout_mode", "step") == "multistep"

    def _prepare_multistep_inputs(self, inputs):
        prepared_inputs: list[dict[str, Any]] = []
        for item in inputs:
            task_id = str(item.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("multistep 样本缺少 task_id")
            task = self._task_lookup.get(task_id)
            if task is None:
                raise KeyError(f"未找到 task_id={task_id} 对应的任务")

            from searcheyes.rl_adapter import RLEnvironment

            env = RLEnvironment(
                task,
                rag=getattr(self, "_rag", None),
                images_dir=getattr(self, "_wit_images_dir", None),
                include_candidate_text=getattr(self, "_include_candidate_text", False),
            )
            obs = env.reset()
            prompt_text = build_multistep_prompt_text(
                goal=task.goal,
                state_description=obs.state_description,
                available_actions=obs.available_actions,
            )
            image_path = resolve_image_path(obs.screenshot_path)
            content: list[dict[str, Any]] = []
            if image_path:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt_text})

            prepared = dict(item)
            prepared.update(
                {
                    "task_id": task_id,
                    "goal": task.goal,
                    "prompt": [{"role": "user", "content": content}],
                    "image": image_path,
                    "_env": env,
                }
            )
            prepared_inputs.append(prepared)
        return prepared_inputs

    def _generate_multistep_rollouts(self, prepared_inputs):
        import torch

        prompt_ids_list: list[list[int]] = []
        completion_ids_list: list[list[int]] = []
        env_masks: list[list[int]] = []
        extra_fields: dict[str, list[Any]] = {
            "env_mask": env_masks,
            "episode_reward": [],
            "episode_success": [],
            "episode_steps": [],
            "episode_trace": [],
        }

        for item in prepared_inputs:
            rollout = self._run_single_multistep_episode(item)
            prompt_ids_list.append(rollout["prompt_ids"])
            completion_ids_list.append(rollout["completion_ids"])
            env_masks.append(rollout["env_mask"])
            extra_fields["episode_reward"].append(rollout["episode_reward"])
            extra_fields["episode_success"].append(rollout["episode_success"])
            extra_fields["episode_steps"].append(rollout["episode_steps"])
            extra_fields["episode_trace"].append(rollout["episode_trace"])

        return prompt_ids_list, completion_ids_list, None, extra_fields

    def _run_single_multistep_episode(self, item: dict[str, Any]) -> dict[str, Any]:
        import logging
        _log = logging.getLogger("grpo.rollout")

        env = item["_env"]
        task_id = item.get("task_id", "?")
        messages = copy.deepcopy(item["prompt"])
        image_path = item.get("image", "")
        # [fix P0] 用列表存当前帧，每步结束后更新（参考 Code2World m3a_wm.py 的 per-step 截图传递）
        image_paths = [image_path] if image_path else []
        max_episode_steps = max(1, min(getattr(self, "_multistep_max_episode_steps", 8), env.max_steps))

        _log.info(
            "[rollout] START task=%s max_steps=%d init_image=%s",
            task_id, max_episode_steps, image_path or "none",
        )

        prompt_ids = self._encode_conversation_ids(messages, image_paths, add_generation_prompt=True)
        current_ids = list(prompt_ids)
        completion_ids: list[int] = []
        env_mask: list[int] = []
        episode_reward = 0.0
        episode_success = False
        episode_steps = 0
        trace: list[str] = []

        for step_idx in range(max_episode_steps):
            _log.debug(
                "[rollout] task=%s step=%d image=%s msg_turns=%d",
                task_id, step_idx, image_paths[0] if image_paths else "none", len(messages),
            )

            action_text = self._sample_action_text(messages, image_paths)
            _log.info("[rollout] task=%s step=%d action_text=%r", task_id, step_idx, action_text)
            trace.append(f"assistant[{step_idx}]: {action_text}")
            messages.append({"role": "assistant", "content": action_text})
            next_ids = self._encode_conversation_ids(messages, image_paths, add_generation_prompt=False)
            assistant_segment = next_ids[len(current_ids):]
            current_ids = next_ids
            completion_ids.extend(assistant_segment)
            env_mask.extend([1] * len(assistant_segment))

            payload = parse_action_payload(action_text)
            if payload is None:
                episode_reward += -1.0
                trace.append("env: invalid_action_json")
                _log.warning("[rollout] task=%s step=%d invalid JSON -> -1.0 break", task_id, step_idx)
                break

            action = str(payload.get("action", "")).strip()
            params = payload.get("params", {}) or {}
            if not action:
                episode_reward += -1.0
                trace.append("env: empty_action")
                _log.warning("[rollout] task=%s step=%d empty action -> -1.0 break", task_id, step_idx)
                break

            result = env.step(action, params)
            episode_reward += float(result.reward)
            episode_success = episode_success or bool(result.info.get("success"))
            episode_steps += 1

            # ── [fix P0] 每步结束后更新截图（Bug 修复核心）──
            # 参考 Code2World m3a_wm.py：world model 预测下一帧后立即传给 agent
            # 此处直接用环境渲染的真实新截图，比预测更可靠
            new_screenshot = result.obs.screenshot_path
            if new_screenshot and Path(new_screenshot).exists():
                # [fix P2] 替换（不累积）当前帧：feedback 消息不含 <image> token，
                # 所以 image_paths 始终保持 1 张（对应 prompt 里的唯一 <image> token）。
                # 在行动生成 (_sample_action_text) 时热换为最新截图，让模型"看到"新状态；
                # 但 completion_ids 里没有额外 image token，TRL forward pass 不会报特征不匹配。
                image_paths = [new_screenshot]
                _log.debug(
                    "[rollout] task=%s step=%d screenshot_hotswap=%s",
                    task_id, step_idx, new_screenshot,
                )
            else:
                # 新截图无效：保留上一帧（image_paths 长度仍为 1）
                _log.warning(
                    "[rollout] task=%s step=%d screenshot_missing=%r keep_prev",
                    task_id, step_idx, new_screenshot,
                )

            if result.done:
                trace.append(
                    f"env[{step_idx}]: reward={result.reward:.2f} "
                    f"state={result.obs.state_description} done=1"
                )
                _log.info(
                    "[rollout] task=%s step=%d DONE reward=%.3f success=%s",
                    task_id, step_idx, result.reward, result.info.get("success"),
                )
                break

            # ── 构建环境反馈消息（纯文本，不含 <image> token）──
            # [fix P2] feedback 消息仅含文本，不重复插入 <image> token。
            # 原因：completion_ids 若含额外 image token，TRL forward pass 会因
            # "image features != image tokens" 报错（features 来自 prompt 的唯一初始截图）。
            # 视觉更新通过 image_paths hot-swap 实现：下一步 _sample_action_text 调用时
            # image_paths 已指向最新截图，prompt 里的唯一 <image> slot 被替换为新帧。
            obs_feedback_text = self._format_multistep_feedback(result)
            obs_content: list[dict[str, Any]] = [{"type": "text", "text": obs_feedback_text}]

            trace.append(
                f"env[{step_idx}]: reward={result.reward:.2f} "
                f"state={result.obs.state_description} done=0"
            )
            _log.debug(
                "[rollout] task=%s step=%d env_reward=%.3f state=%s current_screenshot=%s",
                task_id, step_idx, result.reward,
                result.obs.state_description,
                image_paths[0] if image_paths else "none",
            )

            messages.append({"role": "user", "content": obs_content})
            next_ids = self._encode_conversation_ids(messages, image_paths, add_generation_prompt=True)
            user_segment = next_ids[len(current_ids):]
            current_ids = next_ids
            completion_ids.extend(user_segment)
            env_mask.extend([0] * len(user_segment))

        if not completion_ids:
            completion_ids = [self.eos_token_id]
            env_mask = [1]

        _log.info(
            "[rollout] END task=%s steps=%d episode_reward=%.3f success=%s "
            "completion_tokens=%d",
            task_id, episode_steps, episode_reward, episode_success, len(completion_ids),
        )

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "env_mask": env_mask,
            "episode_reward": episode_reward,
            "episode_success": episode_success,
            "episode_steps": episode_steps,
            "episode_trace": "\n".join(trace),
        }

    def _sample_action_text(self, messages, image_paths):
        import torch

        generate_inputs = self._build_processor_inputs(messages, image_paths, add_generation_prompt=True)
        generate_inputs = self._move_batch_to_device(generate_inputs)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        was_training = unwrapped_model.training
        unwrapped_model.eval()
        try:
            with torch.no_grad():
                output_ids = unwrapped_model.generate(
                    **generate_inputs,
                    generation_config=self.generation_config,
                    disable_compile=True,
                )
        finally:
            if was_training:
                unwrapped_model.train()
        prompt_length = generate_inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_length:].tolist()
        if self.eos_token_id in generated_ids:
            generated_ids = generated_ids[: generated_ids.index(self.eos_token_id) + 1]
        return self.processing_class.decode(generated_ids, skip_special_tokens=True).strip()

    def _encode_conversation_ids(self, messages, image_paths, add_generation_prompt):
        inputs = self._build_processor_inputs(messages, image_paths, add_generation_prompt=add_generation_prompt)
        return inputs["input_ids"][0].tolist()

    def _build_processor_inputs(self, messages, image_paths, add_generation_prompt):
        import logging as _logging
        _dlog = _logging.getLogger("grpo.debug")
        prompt_text = self.processing_class.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        # 仅在不匹配时记录诊断日志（正常情况下不输出）
        pad_count = prompt_text.count("<|image_pad|")
        if pad_count != len(image_paths):
            for i, m in enumerate(messages):
                content_summary = [p.get("type", "?") for p in m.get("content", [])] if isinstance(m.get("content"), list) else [str(m.get("content", ""))[:50]]
                _dlog.warning("[build_inputs] MISMATCH msg[%d] role=%s content_types=%s", i, m["role"], content_summary)
            _dlog.warning(
                "[build_inputs] MISMATCH image_pad_tokens=%d image_paths=%d snippet=%r",
                pad_count, len(image_paths), prompt_text[:300],
            )
        kwargs: dict[str, Any] = {
            "text": [prompt_text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_paths:
            kwargs["images"] = [image_paths]
        return self.processing_class(**kwargs)

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(self.accelerator.device) if hasattr(value, "to") else value
        return moved

    def _format_multistep_feedback(self, result) -> str:
        """构建环境反馈文本。

        修复说明（参考 AWM verify.py system_prompt + Code2World vimo_reward.py）：
        - 去掉数值 reward（避免模型学到 spurious correlation）
        - 改用语义分类状态，和 AWM 的 complete/incomplete/agent_error 分类对齐
        - 保留 validation 失败的明确提示，帮助模型识别非法动作
        - 数值 reward 仅写入调试 trace，不出现在 agent 的 context 里
        """
        available_actions = ", ".join(result.obs.available_actions)
        validation_info = str(result.info.get("validation", "")).strip()
        alignment = result.info.get("alignment", {})
        no_state_change = alignment.get("no_state_change", False)
        repeat_count = alignment.get("repeat_count", 0)

        # AWM 风格：语义分类，不暴露数值
        if result.info.get("success"):
            status = "Success! Task objective met."
        elif not result.info.get("validation", True) or "fail" in validation_info.lower():
            status = "Invalid action: this action is not allowed in the current state."
        elif no_state_change:
            status = "No effect: this action did not change the page state. Try a different action."
        elif repeat_count >= 2:
            status = "Repeated action with no new information gained. Try a different approach."
        elif result.reward > 0.2:
            status = "Good progress."
        elif result.reward < -0.3:
            status = "Wrong direction. Reconsider your next action."
        else:
            status = "Action executed. No significant change."

        return (
            f"Result: {status}\n"
            f"State: {result.obs.state_description}\n"
            f"Available actions: {available_actions}\n"
            "Choose exactly one next action.\n"
            'Return JSON only: {"action": "...", "params": {...}}'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO warm-start training for Qwen3-VL")
    parser.add_argument("--train-mode", choices=["step", "multistep"], default="step")
    parser.add_argument(
        "--input-glob",
        default="data/benchmark/batch_rollouts/scripted_*/sft_demos/*.jsonl",
        help="Step-level rollout JSONL glob. Defaults to scripted demos.",
    )
    parser.add_argument(
        "--task-jsonl",
        default=str(PROJECT_ROOT / "data" / "tasks" / "rag_tasks.jsonl"),
        help="Multistep mode task JSONL (RAG-driven tasks).",
    )
    parser.add_argument(
        "--wit-images-dir",
        default=str(PROJECT_ROOT / "data" / "wit_subset_hf" / "images"),
        help="Directory containing WIT images for RAG.",
    )
    parser.add_argument(
        "--chroma-db-path",
        default=str(PROJECT_ROOT / "data" / "wit_subset_hf" / "chroma_db"),
        help="ChromaDB path for RAG retrieval.",
    )
    parser.add_argument(
        "--embedding-server-url",
        default="http://localhost:8000",
        help="Embedding server URL for RAG.",
    )
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=551)
    parser.add_argument("--max-episode-steps", type=int, default=8)
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "checkpoints" / "sft_5090_run1" / "adapter"),
        help="Warm-start model path. Can be a LoRA adapter directory or a full model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "checkpoints" / "grpo_5090_run1"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--quantization", choices=["auto", "none", "4bit", "8bit"], default="none")
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=0,
        help="If 0, auto-round up to a multiple of num_generations.",
    )
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--include-candidate-text", action="store_true",
                        help="在 state_description 中加入候选产品文本列表（Fix C）")
    return parser.parse_args()


def _setup_logging(output_dir: str) -> None:
    """配置训练日志：同时输出到 stdout 和文件，方便从日志定位 rollout / reward 问题。"""
    import logging
    import os

    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_grpo.log"

    fmt = "%(asctime)s %(levelname)s %(name)s  %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path), encoding="utf-8"),
        ],
        force=True,
    )
    # 第三方库保持 WARNING，避免日志被淹没
    for noisy in ("transformers", "peft", "accelerate", "datasets", "trl", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("grpo.rollout").setLevel(logging.INFO)
    logging.getLogger("grpo.reward").setLevel(logging.INFO)
    logging.getLogger("rl_adapter.reward").setLevel(logging.INFO)
    logging.getLogger("grpo.debug").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "日志已配置。输出文件: %s  (grpo.rollout / grpo.reward / rl_adapter.reward 均为 INFO+)",
        log_path,
    )


def main():
    args = parse_args()
    _setup_logging(args.output_dir)

    import torch
    import transformers
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from trl import GRPOConfig, GRPOTrainer

    class QwenGRPOTrainer(QwenGRPOTrainerMixin, GRPOTrainer):
        pass

    from searcheyes.vlm_agent import (
        build_quantization_config,
        resolve_attn_implementation_choice,
        resolve_device_choice,
        resolve_quantization_choice,
        resolve_torch_dtype_choice,
    )
    from searcheyes.bench_export import load_tasks_from_jsonl

    # ── RAG 初始化（搜索引擎）──
    rag = None
    if args.train_mode == "multistep":
        try:
            from searcheyes.multimodal_rag import MultimodalRAG, RagConfig
            rag_config = RagConfig(
                chroma_db_path=args.chroma_db_path,
                embedding_server_url=args.embedding_server_url,
            )
            rag = MultimodalRAG(rag_config)
            print(f"RAG initialized: chroma_db={args.chroma_db_path}")
        except Exception as e:
            print(f"WARNING: RAG init failed ({e}), multistep search will have no products")

    if args.train_mode == "multistep":
        dataset_records, task_lookup = load_multistep_task_records(args.task_jsonl, args.max_samples)
    else:
        dataset_records = load_grpo_records(args.input_glob, args.max_files, args.max_samples)
        task_lookup = {}
    if not dataset_records:
        raise SystemExit("没有找到可用的 GRPO 训练样本")

    train_dataset = Dataset.from_list(dataset_records)

    resolved_device = resolve_device_choice(torch, args.device)
    resolved_dtype = resolve_torch_dtype_choice(torch, args.dtype, resolved_device)
    resolved_quantization = resolve_quantization_choice(torch, args.quantization, resolved_device)
    resolved_attn = resolve_attn_implementation_choice(args.attn_implementation, resolved_device)

    model, processor = load_model_and_processor(
        model_path=args.model_path,
        resolved_device=resolved_device,
        resolved_dtype=resolved_dtype,
        resolved_quantization=resolved_quantization,
        resolved_attn=resolved_attn,
        torch_module=torch,
        transformers_module=transformers,
        prepare_model_for_kbit_training=prepare_model_for_kbit_training,
        auto_processor_cls=AutoProcessor,
        qwen_model_cls=Qwen3VLForConditionalGeneration,
        peft_model_cls=PeftModel,
        build_quantization_config_fn=build_quantization_config,
    )
    model.config.use_cache = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_batch_size = resolve_generation_batch_size(
        requested=args.generation_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
    )

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=resolved_device == "cuda" and resolved_dtype == torch.bfloat16,
        fp16=resolved_device == "cuda" and resolved_dtype == torch.float16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=args.num_generations,
        generation_batch_size=generation_batch_size,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
        beta=args.beta,
        use_vllm=False,
        seed=args.seed,
    )

    trainer = QwenGRPOTrainer(
        model=model,
        reward_funcs=multi_step_reward if args.train_mode == "multistep" else step_level_reward,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=processor,
    )
    trainer._grpo_rollout_mode = args.train_mode
    trainer._task_lookup = task_lookup
    trainer._multistep_max_episode_steps = args.max_episode_steps
    trainer._rag = rag
    trainer._wit_images_dir = args.wit_images_dir
    trainer._include_candidate_text = args.include_candidate_text

    train_kwargs: dict[str, Any] = {}
    if args.resume_from_checkpoint:
        train_kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint
    trainer.train(**train_kwargs)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)

    summary = {
        "sample_count": len(dataset_records),
        "train_mode": args.train_mode,
        "model_path": args.model_path,
        "device": resolved_device,
        "dtype": str(resolved_dtype),
        "quantization": resolved_quantization,
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "max_episode_steps": args.max_episode_steps,
        "num_generations": args.num_generations,
        "generation_batch_size": generation_batch_size,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "rag_enabled": rag is not None,
        "chroma_db_path": args.chroma_db_path,
        "wit_images_dir": args.wit_images_dir,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def resolve_image_path(raw_path: str) -> str:
    if not raw_path:
        return raw_path

    candidate = Path(raw_path)
    if candidate.exists():
        return str(candidate)

    normalized = raw_path.replace("\\", "/")
    for prefix in _LEGACY_PREFIXES:
        normalized_prefix = prefix.replace("\\", "/")
        if normalized.startswith(normalized_prefix):
            suffix = normalized[len(normalized_prefix) :].lstrip("/")
            remapped = PROJECT_ROOT / suffix
            if remapped.exists():
                return str(remapped)

    for marker in _LEGACY_PROJECT_MARKERS:
        normalized_marker = marker.replace("\\", "/")
        if normalized_marker in normalized:
            suffix = normalized.split(normalized_marker, 1)[1].lstrip("/")
            remapped = PROJECT_ROOT / suffix
            if remapped.exists():
                return str(remapped)

    return raw_path


def load_grpo_records(input_glob: str, max_files: int, max_samples: int) -> list[dict[str, Any]]:
    pattern = str((PROJECT_ROOT / input_glob).resolve()) if not Path(input_glob).is_absolute() else input_glob
    files = [Path(path) for path in sorted(glob.glob(pattern))]
    if max_files > 0:
        files = files[:max_files]

    records: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                converted = convert_rollout_record(json.loads(line))
                if converted is None:
                    continue
                records.append(converted)
                if 0 < max_samples <= len(records):
                    return records
    return records


def load_multistep_task_records(task_jsonl: str, max_samples: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from searcheyes.bench_export import load_tasks_from_jsonl

    tasks = load_tasks_from_jsonl(task_jsonl)
    if max_samples > 0:
        tasks = tasks[:max_samples]
    task_lookup = {task.task_id: task for task in tasks}
    records = [{"task_id": task.task_id} for task in tasks]
    return records, task_lookup


def resolve_generation_batch_size(requested: int, per_device_train_batch_size: int, num_generations: int) -> int:
    if requested > 0:
        if requested % num_generations != 0:
            raise ValueError("generation_batch_size 必须能被 num_generations 整除")
        return requested

    base = max(per_device_train_batch_size, num_generations)
    remainder = base % num_generations
    if remainder == 0:
        return base
    return base + (num_generations - remainder)


def convert_rollout_record(record: dict[str, Any]) -> dict[str, Any] | None:
    prompt = record.get("prompt")
    expected = record.get("expected", {}) or {}

    prompt_text, image_path, state_description, available_actions = extract_prompt_fields(prompt)
    if not prompt_text:
        return None
    if image_path:
        image_path = resolve_image_path(image_path)
        if image_path.startswith("/") and not Path(image_path).exists():
            return None

    content: list[dict[str, Any]] = []
    if image_path:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text})

    item: dict[str, Any] = {
        "prompt": [{"role": "user", "content": content}],
        "task_id": record.get("task_id", ""),
        "step_idx": int(record.get("step_idx", 0)),
        "expected_action": str(expected.get("action", "")),
        "expected_params_json": json.dumps(expected.get("params", {}) or {}, ensure_ascii=False, sort_keys=True),
        "reference_reward": float(record.get("reward", 0.0)),
        "done": bool(record.get("done", False)),
        "state_description": state_description,
        "available_actions": available_actions,
    }
    if image_path:
        item["image"] = image_path
    return item


def extract_prompt_fields(prompt: Any) -> tuple[str, str, str, list[str]]:
    if isinstance(prompt, dict):
        prompt_text = prompt.get("prompt_text") or build_prompt_text(
            goal=prompt.get("goal", ""),
            state_description=prompt.get("state_description", ""),
            available_actions=prompt.get("available_actions", []) or [],
        )
        state_description = str(prompt.get("state_description", ""))
        available_actions = [str(x).strip() for x in prompt.get("available_actions", []) if str(x).strip()]
        image_path = str(prompt.get("image", ""))
        return prompt_text.strip(), image_path, state_description, available_actions

    text_parts: list[str] = []
    image_path = ""
    for message in prompt or []:
        for item in message.get("content", []):
            item_type = item.get("type")
            if item_type == "image_url":
                image_path = item.get("image_url", {}).get("url", "") or image_path
            elif item_type == "image":
                image_path = item.get("image", "") or image_path
            elif item_type == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    text_parts.append(text)

    prompt_text = "\n".join(text_parts).strip()
    state_description = parse_prefixed_line(prompt_text, "State:")
    available_actions = parse_available_actions(prompt_text)
    return prompt_text, image_path, state_description, available_actions


def build_prompt_text(goal: str, state_description: str, available_actions: list[str]) -> str:
    actions_text = ", ".join(available_actions)
    return (
        f"Goal: {goal}\n"
        f"State: {state_description}\n"
        f"Available actions: {actions_text}"
    ).strip()


def build_multistep_prompt_text(goal: str, state_description: str, available_actions: list[str]) -> str:
    actions_text = ", ".join(available_actions)
    return (
        f"Goal: {goal}\n"
        f"State: {state_description}\n"
        f"Available actions: {actions_text}\n"
        "Choose exactly one next action.\n"
        'Return JSON only in the form {"action": "...", "params": {...}}.'
    ).strip()


def parse_prefixed_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def parse_available_actions(text: str) -> list[str]:
    raw = parse_prefixed_line(text, "Available actions:")
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_model_and_processor(
    model_path: str,
    resolved_device: str,
    resolved_dtype: Any,
    resolved_quantization: str,
    resolved_attn: str,
    torch_module,
    transformers_module,
    prepare_model_for_kbit_training,
    auto_processor_cls,
    qwen_model_cls,
    peft_model_cls,
    build_quantization_config_fn,
):
    model_path_obj = Path(model_path)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "attn_implementation": resolved_attn,
    }
    if resolved_quantization != "none":
        if resolved_device != "cuda":
            raise RuntimeError("低比特量化训练仅支持 CUDA")
        load_kwargs["quantization_config"] = build_quantization_config_fn(
            torch_module,
            transformers_module,
            resolved_quantization,
            resolved_dtype,
        )
        load_kwargs["device_map"] = "auto"
    elif resolved_device == "cuda":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = None

    adapter_config_path = model_path_obj / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise RuntimeError(f"adapter_config.json 缺少 base_model_name_or_path: {adapter_config_path}")
        base_model = qwen_model_cls.from_pretrained(base_model_path, **load_kwargs)
        if resolved_quantization != "none":
            base_model = prepare_model_for_kbit_training(base_model)
        model = peft_model_cls.from_pretrained(base_model, model_path, is_trainable=True)
        if (model_path_obj / "processor_config.json").exists():
            processor = auto_processor_cls.from_pretrained(model_path)
        else:
            processor = auto_processor_cls.from_pretrained(base_model_path)
    else:
        model = qwen_model_cls.from_pretrained(model_path, **load_kwargs)
        if resolved_quantization != "none":
            model = prepare_model_for_kbit_training(model)
        processor = auto_processor_cls.from_pretrained(model_path)

    if resolved_device == "mps":
        model.to("mps")
    elif resolved_device == "cpu":
        model.to("cpu")
    return model, processor


def step_level_reward(
    prompts,
    completions,
    expected_action,
    expected_params_json,
    reference_reward,
    done,
    available_actions,
    **kwargs,
) -> list[float]:
    rewards: list[float] = []
    for completion, exp_action, exp_params_raw, ref_reward, is_done, actions in zip(
        completions,
        expected_action,
        expected_params_json,
        reference_reward,
        done,
        available_actions,
        strict=True,
    ):
        reward = compute_single_reward(
            completion=completion,
            expected_action=str(exp_action),
            expected_params=json.loads(exp_params_raw or "{}"),
            reference_reward=float(ref_reward),
            done=bool(is_done),
            available_actions=actions or [],
        )
        rewards.append(reward)
    return rewards


def multi_step_reward(
    prompts,
    completions,
    episode_reward,
    episode_success,
    episode_steps,
    episode_trace,
    **kwargs,
) -> list[float]:
    """多步 episode 奖励函数。

    修复说明（参考 DreamGym 论文公式 + AWM verify.py）：
    1. 成功奖励提高到 +1.5（拉大组内方差，增强 GRPO credit assignment 信号）
    2. reward_std < ε 时返回全零（DreamGym 核心：group reward entropy 为零时
       GRPO advantage=0，等价于跳过该批次，避免无意义梯度更新）
    3. 详细日志便于定位训练内 reward 分布问题
    """
    import logging
    import statistics
    _log = logging.getLogger("grpo.reward")

    rewards: list[float] = []
    traces = list(episode_trace) if episode_trace is not None else [""] * len(list(episode_reward))

    for total_reward, success, steps, trace in zip(
        episode_reward,
        episode_success,
        episode_steps,
        traces,
        strict=False,
    ):
        reward = float(total_reward)
        if success:
            reward += 1.5   # 提高成功奖励，拉大组内对比度
        reward -= 0.02 * max(0, int(steps) - 1)   # 轻微步数惩罚
        reward = max(-4.0, min(8.0, reward))
        rewards.append(reward)

        _log.debug(
            "[multi_step_reward] raw=%.3f success=%s steps=%d final=%.3f",
            float(total_reward), success, int(steps), reward,
        )

    # ── [fix P1] DreamGym reward entropy 过滤 ──
    # 组内 reward 全相同 → advantage=0 → 梯度为零 → 无效更新，直接返回全零跳过
    if len(rewards) > 1:
        try:
            reward_std = statistics.stdev(rewards)
        except statistics.StatisticsError:
            reward_std = 0.0
        _log.info(
            "[multi_step_reward] group_size=%d reward_mean=%.3f reward_std=%.3f "
            "rewards=%s",
            len(rewards),
            statistics.mean(rewards),
            reward_std,
            [f"{r:.2f}" for r in rewards],
        )
        if reward_std < 1e-6:
            _log.warning(
                "[multi_step_reward] reward_std=0 -> zero advantages (DreamGym skip). "
                "Hint: increase num_generations or temperature to improve group diversity."
            )
            return [0.0] * len(rewards)
    else:
        _log.info(
            "[multi_step_reward] single sample reward=%.3f (no std filter)",
            rewards[0] if rewards else 0.0,
        )

    return rewards


def compute_single_reward(
    completion: Any,
    expected_action: str,
    expected_params: dict[str, Any],
    reference_reward: float,
    done: bool,
    available_actions: list[str],
) -> float:
    text = extract_completion_text(completion)
    payload = parse_action_payload(text)
    if payload is None:
        return -1.0

    action = str(payload.get("action", "")).strip()
    params = payload.get("params", {}) or {}
    allowed = [str(action_name).strip() for action_name in available_actions if str(action_name).strip()]
    if allowed and action not in allowed:
        return -1.2

    if action == expected_action and params_match(params, expected_params):
        return clamp_reward(reference_reward)

    reward = -0.1
    action_match = action == expected_action
    reward += 0.7 if action_match else -0.55

    if action_match:
        if params_match(params, expected_params):
            reward += 0.45
        elif expected_params:
            reward -= 0.2
        else:
            reward += 0.2

    if expected_action == "zoom":
        reward += 0.1 if action == "zoom" else -0.08
    if done and expected_action == action == "buy":
        reward += 0.8

    return clamp_reward(reward)


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        if "content" in completion:
            return extract_completion_text(completion["content"])
        if "text" in completion:
            return str(completion["text"]).strip()
        return json.dumps(completion, ensure_ascii=False)
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            text = extract_completion_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(completion).strip()


def parse_action_payload(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    candidate = text.strip()
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def params_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return {k: str(v) for k, v in actual.items()} == {k: str(v) for k, v in expected.items()}


def clamp_reward(value: float) -> float:
    return max(-1.5, min(2.0, float(value)))


if __name__ == "__main__":
    main()
