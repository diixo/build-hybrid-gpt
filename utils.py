
import os
import torch
from torch.nn import functional as F
from transformers import PreTrainedTokenizerBase


def generate_text(prompt: str, model, tokenizer: PreTrainedTokenizerBase, device, device_type, ddp_rank):
    model.eval()
    num_return_sequences = 1
    max_length = 64
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    xgen = encoded["input_ids"].to(device)
    xgen = xgen.repeat(num_return_sequences, 1)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    with torch.no_grad():
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(xgen).logits # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        print(f"rank {ddp_rank} sample {i}: {decoded}")


def plot_loss(losses: list, model_type: str):
    import matplotlib.pyplot as plt

    plt.plot(range(1, len(losses) + 1), losses, marker='o', label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Trained on: {model_type}")
    plt.legend()

    plt.grid(True, which="both", linestyle="--", alpha=0.5)  # <- грид
    plt.tight_layout()

    plt.show()


