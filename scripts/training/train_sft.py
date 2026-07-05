"""
SFT Training Script for SearchEyes Agent (Vision-Language)
===========================================================
Full fine-tuning Qwen3.5-9B VLM on multi-hop search trajectories.
Each sample has 1 entity image in the user query + multi-turn ReAct trace.

Pure DDP on 8×H20 GPUs. Each GPU holds full model replica.
Manual dist.init + barrier before model loading to keep NCCL alive.

Usage:
  cd .
  torchrun --nproc_per_node=8 --master_port=29560 train_sft.py
"""
import json
import logging
import math
import os
import re
import time
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    Adafactor,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [R%(process)d] %(message)s")
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

MODEL_PATH = "./models/Qwen3.5-9B"
DATA_PATH = "/tmp/sft_output/sft_sharegpt.json"
OUTPUT_DIR = "/dev/shm/sft_output"  # save to tmpfs, not home (disk quota)
LOG_DIR = "/tmp/sft_output"  # training log + loss curve persisted here
MAX_SEQ_LENGTH = 24576  # covers P99=20436, saves ~30% activation memory vs 32768


def parse_sharegpt_to_qwen_messages(conversations: list[dict]) -> tuple[list[dict], str | None]:
    """Convert ShareGPT conversations to Qwen3.5 VLM message format."""
    messages = []
    image_path = None

    for msg in conversations:
        role = msg["role"]
        content = msg["content"]
        img_matches = re.findall(r"<image>(.*?)</image>", content)
        text_content = re.sub(r"<image>.*?</image>\s*", "", content).strip()

        if img_matches and image_path is None:
            image_path = img_matches[0]
            content_parts = [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text_content},
            ]
            messages.append({"role": role, "content": content_parts})
        else:
            messages.append({"role": role, "content": text_content})

    return messages, image_path


class VLMSFTDataset(TorchDataset):
    """Dataset that loads images on-the-fly and tokenizes with Qwen3VLProcessor."""

    def __init__(self, samples: list[dict], processor, max_length: int):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        messages, image_path = parse_sharegpt_to_qwen_messages(sample["conversations"])

        image = None
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                image = None

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        if image is not None:
            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt",
                padding=False, truncation=True, max_length=self.max_length,
            )
        else:
            inputs = self.processor(
                text=[text], return_tensors="pt",
                padding=False, truncation=True, max_length=self.max_length,
            )

        result = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == 1:
                if key == "image_grid_thw":
                    result[key] = value  # keep [1, 3]
                else:
                    result[key] = value.squeeze(0)
            else:
                result[key] = value

        result["labels"] = result["input_ids"].clone()
        return result


def collate_vlm_samples(batch: list[dict]) -> dict[str, Any]:
    """Custom collator for VLM samples with variable-length pixel_values."""
    max_len = max(sample["input_ids"].shape[0] for sample in batch)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    all_pixel_values = []
    all_image_grid_thw = []
    has_mm = any("mm_token_type_ids" in s for s in batch)
    padded_mm = []

    for sample in batch:
        seq_len = sample["input_ids"].shape[0]
        pad_len = max_len - seq_len

        padded_input_ids.append(torch.cat([sample["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        padded_attention_mask.append(torch.cat([sample["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([sample["labels"], torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]))

        if has_mm and "mm_token_type_ids" in sample:
            padded_mm.append(torch.cat([sample["mm_token_type_ids"], torch.zeros(pad_len, dtype=torch.long)]))

        if "pixel_values" in sample and sample["pixel_values"] is not None:
            all_pixel_values.append(sample["pixel_values"])
        if "image_grid_thw" in sample and sample["image_grid_thw"] is not None:
            all_image_grid_thw.append(sample["image_grid_thw"])

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }
    if padded_mm:
        result["mm_token_type_ids"] = torch.stack(padded_mm)
    if all_pixel_values:
        result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
    if all_image_grid_thw:
        result["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

    return result


def average_gradients(model, world_size):
    """Manually all-reduce gradients using default PG (no new communicator)."""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # ── Step 1: Init NCCL PG FIRST, before any heavy work ──
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Warm up NCCL communicator with a trivial all_reduce
    warmup_tensor = torch.ones(1, device=device)
    dist.all_reduce(warmup_tensor)
    if rank == 0:
        logger.info(f"NCCL initialized, world_size={world_size}, warmup={warmup_tensor.item()}")
    dist.barrier()

    # ── Step 2: Load data ──
    with open(DATA_PATH) as f:
        all_data = json.load(f)
    eval_size = max(int(len(all_data) * 0.02), 10)
    train_data = all_data[:-eval_size]

    if rank == 0:
        logger.info(f"Loaded {len(all_data)} samples, training on {len(train_data)}")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, padding_side="right",
    )
    train_dataset = VLMSFTDataset(train_data, processor, MAX_SEQ_LENGTH)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_vlm_samples,
        num_workers=0,
        pin_memory=True,
    )

    if rank == 0:
        logger.info(f"Dataset: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")

    # ── Step 3: Load model → GPU (serialize to avoid OOM + TCPStore timeout) ──
    # Load one rank at a time to prevent all 8 ranks from competing for CPU/IO
    for loading_rank in range(world_size):
        if rank == loading_rank:
            logger.info(f"Rank {rank}: loading model to {device}...")
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa",
                device_map={"": device},  # load directly to GPU
            )
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            logger.info(f"Rank {rank}: model loaded, {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
        dist.barrier()

    if rank == 0:
        logger.info("All ranks loaded model. No DDP — using manual gradient all_reduce.")

    # ── Step 4: Optimizer + Scheduler ──
    num_epochs = 3.0
    grad_accum = 1
    learning_rate = 1e-5

    # Adafactor: ~1x param memory vs AdamW's ~3x (no fp32 master weights or momentum)
    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.01,
    )

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = int(steps_per_epoch * num_epochs)
    warmup_steps = int(total_steps * 0.05)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    if rank == 0:
        logger.info(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}, warmup: {warmup_steps}")

    # ── Step 5: Training loop (manual gradient sync) ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Loss curve data — append every logging step, flush periodically
    loss_curve_path = os.path.join(LOG_DIR, "loss_curve.jsonl")
    loss_curve_file = None
    if rank == 0:
        loss_curve_file = open(loss_curve_path, "w")

    global_step = 0
    log_loss = 0.0
    micro_step = 0
    start_time = time.time()

    if rank == 0:
        logger.info("Starting training...")

    for epoch in range(int(math.ceil(num_epochs))):
        sampler.set_epoch(epoch)
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            fraction = epoch + batch_idx / len(train_loader)
            if fraction >= num_epochs:
                break

            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch_device)
                loss = outputs.loss / grad_accum

            loss.backward()

            micro_step += 1
            log_loss += loss.item() * grad_accum

            if micro_step % grad_accum == 0:
                # Manual gradient all_reduce across GPUs (uses default PG only)
                average_gradients(model, world_size)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step % 10 == 0:
                    avg_loss = log_loss / (10 * grad_accum)
                    elapsed = time.time() - start_time
                    lr_now = scheduler.get_last_lr()[0]
                    samples_per_sec = global_step * world_size / elapsed
                    logger.info(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e} | "
                        f"Time: {elapsed:.0f}s | {samples_per_sec:.1f} samples/s"
                    )
                    # Write loss curve data point
                    record = json.dumps({
                        "step": global_step,
                        "loss": round(avg_loss, 6),
                        "lr": lr_now,
                        "epoch": round(fraction, 4),
                        "elapsed_s": round(elapsed, 1),
                        "samples_per_sec": round(samples_per_sec, 2),
                    })
                    loss_curve_file.write(record + "\n")
                    loss_curve_file.flush()
                    log_loss = 0.0

    # ── Step 6: Save ONLY final model to /dev/shm ──
    if rank == 0:
        if loss_curve_file:
            loss_curve_file.close()

        final_path = os.path.join(OUTPUT_DIR, "final")
        os.makedirs(final_path, exist_ok=True)
        logger.info(f"Saving final model to {final_path}...")
        model.save_pretrained(final_path)
        processor.save_pretrained(final_path)
        elapsed = time.time() - start_time
        logger.info(f"Training complete! Time: {elapsed:.0f}s, saved to {final_path}")
        logger.info(f"Loss curve: {loss_curve_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()