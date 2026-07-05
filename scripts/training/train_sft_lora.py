from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
_LEGACY_PROJECT_MARKERS = [
    "/QWEN-project/",
    "\\QWEN-project\\",
]
_LEGACY_PREFIXES = [
    ".",
    ".",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small-batch Qwen3-VL LoRA SFT for local GPU smoke tests")
    parser.add_argument(
        "--input-glob",
        default="data/benchmark/batch_rollouts/scripted_*/sft_demos/*.jsonl",
        help="Glob for scripted rollout JSONL files, relative to repo root by default",
    )
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "checkpoints" / "sft_smoke"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--quantization", choices=["auto", "none", "4bit", "8bit"], default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.per_device_train_batch_size != 1:
        raise SystemExit("当前脚本仅支持 --per-device-train-batch-size 1，以避免视觉张量对齐复杂度")

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments

    from searcheyes.vlm_agent import (
        build_quantization_config,
        resolve_attn_implementation_choice,
        resolve_device_choice,
        resolve_quantization_choice,
        resolve_torch_dtype_choice,
    )

    records = load_records(args.input_glob, args.max_files, args.max_samples)
    if not records:
        raise SystemExit("没有找到可训练样本，请先运行 scripted batch rollout")

    resolved_device = resolve_device_choice(torch, args.device)
    resolved_dtype = resolve_torch_dtype_choice(torch, args.dtype, resolved_device)
    resolved_quantization = resolve_quantization_choice(torch, args.quantization, resolved_device)
    resolved_attn = resolve_attn_implementation_choice(args.attn_implementation, resolved_device)

    load_kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "attn_implementation": resolved_attn,
    }
    if resolved_quantization != "none":
        if resolved_device != "cuda":
            raise SystemExit("低比特量化训练仅支持 CUDA")
        load_kwargs["quantization_config"] = build_quantization_config(
            torch,
            transformers,
            resolved_quantization,
            resolved_dtype,
        )
        load_kwargs["device_map"] = "auto"
    elif resolved_device == "cuda":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = None

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_path)

    if resolved_device == "cpu":
        model.to("cpu")
    elif resolved_device == "mps":
        model.to("mps")

    if resolved_quantization != "none":
        model = prepare_model_for_kbit_training(model)

    gradient_checkpointing = not args.disable_gradient_checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[name.strip() for name in args.target_modules.split(",") if name.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = RolloutStepDataset(records)
    collator = SingleExampleVisionCollator(processor=processor)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=[],
        bf16=resolved_device == "cuda" and resolved_dtype == torch.bfloat16,
        fp16=resolved_device == "cuda" and resolved_dtype == torch.float16,
        gradient_checkpointing=gradient_checkpointing,
        optim="paged_adamw_8bit" if resolved_quantization != "none" else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(output_dir / "adapter")
    processor.save_pretrained(output_dir / "adapter")
    summary = {
        "sample_count": len(records),
        "device": resolved_device,
        "dtype": str(resolved_dtype),
        "quantization": resolved_quantization,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_records(input_glob: str, max_files: int, max_samples: int) -> list[dict[str, Any]]:
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
                records.append(json.loads(line))
                if 0 < max_samples <= len(records):
                    return records
    return records


def resolve_image_path(raw_path: str) -> str:
    if not raw_path:
        return raw_path

    candidate = Path(raw_path)
    if candidate.exists():
        return str(candidate)

    normalized = raw_path.replace("\\", "/")
    for prefix in _LEGACY_PREFIXES:
        if normalized.startswith(prefix.replace("\\", "/")):
            suffix = normalized[len(prefix.replace("\\", "/")) :].lstrip("/")
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


class RolloutStepDataset:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


@dataclass
class SingleExampleVisionCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("当前 collator 只支持 batch size = 1")

        record = features[0]
        
        # 处理两种格式：消息列表或字典
        prompt = record["prompt"]
        if isinstance(prompt, dict):
            # 新格式：包含 goal, image, state_description 等字段
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resolve_image_path(prompt["image"])},
                        {"type": "text", "text": prompt.get("prompt_text", prompt["goal"])},
                    ],
                }
            ]
        else:
            # 旧格式：消息列表
            prompt_messages = normalize_messages(prompt)
        
        answer_text = str(record["chosen"])

        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            }
        ]
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        labels = full_inputs["input_ids"].clone()
        prompt_length = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_length] = -100
        if "attention_mask" in full_inputs:
            labels = labels.masked_fill(full_inputs["attention_mask"] == 0, -100)

        batch = {key: value for key, value in full_inputs.items()}
        batch["labels"] = labels
        return batch


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content: list[dict[str, Any]] = []
        for item in message.get("content", []):
            item_type = item.get("type")
            if item_type == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                content.append({"type": "image", "image": resolve_image_path(image_url)})
            elif item_type == "image":
                content.append({"type": "image", "image": resolve_image_path(item.get("image", ""))})
            elif item_type == "text":
                content.append({"type": "text", "text": item.get("text", "")})
        normalized.append({"role": message.get("role", "user"), "content": content})
    return normalized


if __name__ == "__main__":
    main()
