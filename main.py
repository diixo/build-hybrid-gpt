#--------------------------------------------------------------------------
# vanilla training implementation, non-DDP (Distributed Data Parallel) run
#--------------------------------------------------------------------------

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import time
import numpy as np
import torch
from hellaswag import render_example, iterate_examples, get_most_likely_row
from auto_config import AutoConfigModel
from dataloader import DataLoaderLite
from utils import generate_text
from transformers import GPT2TokenizerFast

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


SEQUENCE_LENGTH = 1024

LEARNING_RATE = 1e-4
eval_steps = 250

total_batch_size = 65536 # 2**16, ~65k, in number of tokens, should be divisible by (B * SEQUENCE_LENGTH * ddp_world_size)
B = 8 # micro batch size

# for high-load GPU:
#total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
#B = 64 # micro batch size


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    tokenizer = GPT2TokenizerFast.from_pretrained(f"data/gpt-noomo-32k", local_files_only=True)

    model = AutoConfigModel.from_pretrained("aitetic/gpt-r-0.3b-warmup")
    if model is None:
        raise SystemExit("Checkpoint 'aitetic/gpt-r-0.3b-warmup' was not found on Hugging Face Hub or in the local cache.")
    # -----------------------------------------------------------------------------

    ddp_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    T = SEQUENCE_LENGTH
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by (B * T * ddp_world_size)"

    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", master_process=master_process)

    torch.set_float32_matmul_precision('high')

    model.to(device)

    raw_model = model # always contains the "raw" unwrapped model

    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=LEARNING_RATE,
        device_type=device_type,
        master_process=master_process)

    tokens_per_micro_batch = B * T * ddp_world_size
    train_micro_batches = sum(
        (len(np.load(shard_path, mmap_mode="r")) - 1) // tokens_per_micro_batch
        for shard_path in train_loader.shards
    )
    max_steps = train_micro_batches // grad_accum_steps # one full pass through the current train shards
    assert max_steps > 0, "train shards do not contain enough tokens for one optimizer step"

    print(f"total desired batch size: {total_batch_size}")
    print(f"=> max_steps: {max_steps}, calculated grad_accum_steps: {grad_accum_steps}")


    # create the log directory we will write checkpoints to and log to
    log_dir = "train_products"

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step % eval_steps == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        output = model(x, y)
                        logits, loss = output.logits, output.loss
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()

            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)

        # once in a while evaluate hellaswag
        if (step % eval_steps == 0 or last_step):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example, tokenizer=tokenizer)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits = model(tokens).logits
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes

            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % eval_steps == 0) or last_step):
            generate_text("Hello, I'm a language model,", model, tokenizer, device, device_type, ddp_rank)

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                output = model(x, y)
                logits, loss = output.logits, output.loss
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
            torch.cuda.empty_cache() # !!!

        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {LEARNING_RATE:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

