"""
Short-context warmup - train on first 100 tokens of each dataset item using a small learning rate as warmup before full pre-training.
This is a simple way to "prime" the model's weights and can lead to faster convergence and better performance in the early stages of training.
"""

import json
import os
import torch, math, random, numpy as np
import torch.distributed as dist
from dataclasses import dataclass
from itertools import islice
from model_llama import GPTLlama
from auto_config import AutoConfigLlama
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from transformers import set_seed
from utils import save_trained_model
from datasets import load_dataset # pip install datasets

import matplotlib.pyplot as plt


SAVE_DIR = "train_products"

MAX_LEN = 100


@dataclass
class TrainerConfig:
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 5e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_accum_steps: int = 1



def custom_collate_fn(batch, max_seq_length, pad_token_id, eos_token_id, device, ignore_index=-100):
    """
    Custom collate function for variable-length text samples.

    Args:
        batch: list of tokenized samples
        eos_token_id: int, used for padding termination
        device: torch.device

    Returns:
        inputs_tensor: [batch_size, seq_len]
        targets_tensor: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        dataset_token_count: int, number of tokenized dataset tokens before EOS/padding
    """

    dataset_token_count = sum(int(item.numel()) for item in batch)

    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []
    attn_lst = []

    for item in batch:

        new_item = item.tolist() + [eos_token_id]
        real_len = len(new_item)

        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - real_len)

        # build attention mask from real_len (NOT from token values)
        attn = [1] * real_len + [0] * (batch_max_length - real_len)

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        am = torch.tensor(attn[:-1], dtype=torch.long)

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if max_seq_length is not None:
            inputs = inputs[:max_seq_length]
            targets = targets[:max_seq_length]
            am = am[:max_seq_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)
        attn_lst.append(am)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    attention_mask = torch.stack(attn_lst).to(device)
    return inputs_tensor, targets_tensor, attention_mask, dataset_token_count


class WikipediaTextDataset(IterableDataset):

    def __init__(self, hf_dataset, tokenizer, max_seq_length=MAX_LEN, max_rows=None, text_key="text", process_rank=0, num_processes=1, master_process=True):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_rows = max_rows
        self.text_key = text_key
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process

        total_rows = len(self.hf_dataset)
        self.total_rows = min(total_rows, max_rows) if max_rows is not None else total_rows
        self.usable_rows = self._get_usable_rows(self.total_rows)
        self.local_total_rows = self._get_local_total_rows(self.usable_rows)

        if self.master_process:
            print(
                f"WikipediaTextDataset::loaded rows.sz={self.total_rows}, usable_rows.sz={self.usable_rows}, local_rows.sz={self.local_total_rows}, max_rows={self.max_rows}, max_seq_length={self.max_seq_length}"
            )

    def _get_usable_rows(self, total_rows):
        if self.num_processes <= 1:
            return total_rows
        return (total_rows // self.num_processes) * self.num_processes

    def _get_local_total_rows(self, total_rows):
        if self.num_processes <= 1:
            return total_rows
        return total_rows // self.num_processes

    def __len__(self):
        return self.local_total_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset = self.hf_dataset

        if self.max_rows is not None:
            dataset = dataset.select(range(self.total_rows))

        if self.num_processes > 1:
            dataset = dataset.select(range(self.usable_rows))
            dataset = dataset.shard(num_shards=self.num_processes, index=self.process_rank, contiguous=True)

        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id, contiguous=True)

        for row in dataset:
            text = row.get(self.text_key, "")
            if text is None:
                text = ""
            elif not isinstance(text, str):
                text = str(text)

            yield self.tokenizer(
                text,
                truncation=True,
                add_special_tokens=False,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors="pt",
            )["input_ids"].squeeze(0)


class Trainer:

    def __init__(self, model, dataset, config, tokenizer, ddp=False, ddp_local_rank=0, master_process=True):
        self.losses = []
        self.step_losses = []
        self.epoch_dataset_token_counts = []
        self.dataset_tokens_processed = 0
        self.ddp = ddp
        self.master_process = master_process

        self.model = model.to(config.device).float()
        if self.ddp:
            device_ids = None
            if str(config.device).startswith("cuda"):
                device_index = torch.device(config.device).index
                if device_index is None:
                    device_index = ddp_local_rank
                device_ids = [device_index]
            self.model = DDP(self.model, device_ids=device_ids)
        self.raw_model = self.model.module if self.ddp else self.model
        self.config = config
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(self.raw_model.parameters(), lr=config.learning_rate)
        self.loader = DataLoader(
            dataset,
            batch_size = config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: custom_collate_fn(
                batch,
                #max_seq_length = model.config.block_size,
                max_seq_length = MAX_LEN,
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
                device = config.device,
                ),
            )


    def train(self):

        torch.set_float32_matmul_precision("high")
        num_loader_steps = max(1, len(self.loader))

        # 1) Gradient accumulation should be an explicit hyperparameter
        grad_accum_steps = int(getattr(self.config, "grad_accum_steps", 1))
        grad_accum_steps = max(1, grad_accum_steps)
        # if epoch has fewer batches than accum steps — clamp
        grad_accum_steps = min(grad_accum_steps, num_loader_steps)

        self.losses = []          # token-weighted epoch losses (good for PPL)
        self.step_losses = []     # avg per-window accumulation raw loss (for plotting)
        self.epoch_dataset_token_counts = []
        self.dataset_tokens_processed = 0

        self.model.train()
        for epoch in range(self.config.epochs):
            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}", disable=not self.master_process)

            total_loss_sum = 0.0   # sum of (mean_loss * num_valid_tokens)
            total_loss_tokens = 0  # number of non-ignored tokens
            total_dataset_tokens = 0
            first_loss = None

            self.optimizer.zero_grad(set_to_none=True)

            accum_raw_sum = 0.0

            for step, batch in enumerate(pbar):
                # NOTE: your collate already .to(device), so these .to() are redundant but harmless
                #x = x.to(self.config.device, non_blocking=True)
                #y = y.to(self.config.device, non_blocking=True)

                input_ids, labels, attention_mask, dataset_token_count = batch
                window_start = step - (step % grad_accum_steps)
                window_size = min(grad_accum_steps, num_loader_steps - window_start)
                should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == num_loader_steps)

                total_dataset_tokens += int(dataset_token_count)

                if self.ddp:
                    self.model.require_backward_grad_sync = should_step

                # Forward pass
                raw_loss = self.model(input_ids, labels).loss

                # ---- logging helpers ----
                raw = float(raw_loss.detach().cpu().item())
                accum_raw_sum += raw

                # token-weighted stats for correct epoch avg loss / PPL
                with torch.no_grad():
                    loss_token_count = int((labels != -100).sum().item())
                total_loss_sum += raw * loss_token_count
                total_loss_tokens += loss_token_count

                # ---- backward (accumulation) ----
                loss = raw_loss / window_size
                loss.backward()

                # Progress bar smoothing
                if first_loss is None:
                    first_loss = raw
                    if self.master_process:
                        pbar.set_postfix(loss=f"{first_loss:.4f}", accum_steps=str(grad_accum_steps))


                # Optimizer step
                if should_step:
                    if getattr(self.config, "max_grad_norm", None) is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm))
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # calculate the average raw loss for current accumulation window
                    step_avg_loss = accum_raw_sum / window_size
                    if self.ddp:
                        step_avg_loss_tensor = torch.tensor(step_avg_loss, dtype=torch.float64, device=self.config.device)
                        dist.all_reduce(step_avg_loss_tensor, op=dist.ReduceOp.AVG)
                        step_avg_loss = float(step_avg_loss_tensor.item())
                    accum_raw_sum = 0.0
                    self.step_losses.append(step_avg_loss)

                    if self.master_process:
                        pbar.set_postfix(loss=f"{step_avg_loss:.4f}", accum_steps=str(grad_accum_steps))


            # ---- epoch metrics (token-weighted, correct for variable lengths) ----
            if self.ddp:
                epoch_stats = torch.tensor(
                    [total_loss_sum, float(total_loss_tokens), float(total_dataset_tokens)],
                    dtype=torch.float64,
                    device=self.config.device,
                )
                dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
                total_loss_sum = float(epoch_stats[0].item())
                total_loss_tokens = int(epoch_stats[1].item())
                total_dataset_tokens = int(epoch_stats[2].item())

            if total_loss_tokens == 0:
                epoch_avg_loss = float("nan")
                ppl = float("nan")
            else:
                epoch_avg_loss = total_loss_sum / total_loss_tokens
                # # Calculate Perplexity, avoid overflow for huge losses
                ppl = math.exp(epoch_avg_loss) if epoch_avg_loss < 50 else float("inf")

            self.losses.append(epoch_avg_loss)
            self.epoch_dataset_token_counts.append(total_dataset_tokens)
            self.dataset_tokens_processed += total_dataset_tokens
            if self.master_process:
                print(f"Epoch {epoch+1}: epoch_avg_loss={epoch_avg_loss:.4f}, PPL={ppl:.4f}, dataset_tokens={total_dataset_tokens:_}")

        if self.master_process and self.losses:
            print(
                "✅ Training completed,",
                f"steps: {len(self.step_losses)}, final_avg_loss: {self.losses[-1]:.4f}, dataset_tokens_processed={self.dataset_tokens_processed:_}"
            )

        return self.losses, self.step_losses


def plot_losses(losses1: list, label1: str, x_label: str):

    plt.plot(range(len(losses1)), losses1, label=label1, color="blue")

    plt.xlabel(x_label)
    plt.ylabel("Loss")
    plt.title(f"Training")
    plt.legend()

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def run_warmup_stage(
    model,
    tokenizer,
    train_config,
    max_rows=None,
):
    ddp = dist.is_available() and dist.is_initialized()
    ddp_rank = dist.get_rank() if ddp else 0
    ddp_world_size = dist.get_world_size() if ddp else 1
    master_process = ddp_rank == 0
    ddp_local_rank = torch.cuda.current_device() if ddp and torch.cuda.is_available() else 0

    fw = load_dataset("aitetic/wikipedia", name="20220301.en", split="train")

    dataset = WikipediaTextDataset(
        fw,
        tokenizer,
        max_seq_length=MAX_LEN,
        max_rows=max_rows,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        master_process=master_process,
    )
    if len(dataset) == 0:
        if ddp:
            raise ValueError("warmup dataset shard is empty; increase warmup rows or reduce distributed world size")
        return model, [], []

    trainer = Trainer(
        model,
        dataset,
        train_config,
        tokenizer,
        ddp=ddp,
        ddp_local_rank=ddp_local_rank,
        master_process=master_process,
    )
    epoch_losses, step_losses = trainer.train()

    return trainer.raw_model, epoch_losses, step_losses


if __name__ == "__main__":

    tokenizer_type = "gpt-noomo-32k"

    model: GPTLlama = None

    train_config = TrainerConfig(learning_rate=8e-5, batch_size=10, grad_accum_steps=1)

    model, tokenizer = AutoConfigLlama.from_config(size_type="mini", tokenizer_type=tokenizer_type)


    #smoke_rows = SMOKE_ROWS if TRAIN_MODE == "smoke-train" else None

    print(f"model.sz={model.get_num_params()}")
    smoke_rows = 1080 #None

    model, epoch_losses, step_losses = run_warmup_stage(
        model,
        tokenizer,
        train_config,
        max_rows=smoke_rows,
    )

    save_trained_model(SAVE_DIR, model, model_type="llama-warmup", train_config=train_config, tokenizer_type=tokenizer_type)

    plot_losses(step_losses, type(model).__name__, "Steps")
