
import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F
from transformers import GPT2TokenizerFast
from model_gpt2 import GPT, GPTConfig

torch.set_float32_matmul_precision('high') # use tf32

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

device = "cuda" if torch.cuda.is_available() else "cpu"


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")


def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)


def render_example(example):
    """
    Given the Hellaswag example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings: # sz=4
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


@torch.no_grad()
def evaluate_hellaswag(model):

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)          # (4, max_len)

        # get the logits
        logits = model(tokens).logits   # (4, max_len, vocab_size)

        # evaluate the autoregressive loss at all positions
        # last logit is prediction of next token, but next token after last does not exist
        shift_logits = (logits[..., :-1, :]).contiguous()   # (4, max_len-1, vocab_size) 
        shift_tokens = (tokens[..., 1:]).contiguous()       # (4, max_len-1)

        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))    # (4 * (max_len-1), vocab_size) 
        flat_shift_tokens = shift_tokens.view(-1)                           # (4 * (max_len-1))

        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')  # (4 * (max_len-1))
        #print(shift_losses.cpu().numpy())

        shift_losses = shift_losses.view(tokens.size(0), -1)    # (tokens.size(0)=4, max_len-1)

        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)       # (4,)
        avg_loss = sum_loss / shift_mask.sum(dim=1)     # (4,)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()                 # (4,) --> (1)
        pred_norm = avg_loss.argmin().item()            # (4,) --> (1)

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


if __name__ == "__main__":

    # model_name = "gpt2"
    # model_name = "EleutherAI/gpt-neo-125M"              # = 0.2936
    # model_name = "rhysjones/gpt2-124M-edu-fineweb-10B"  # = 0.3099 # Andrej Karpathy model

    file_path = "models/nano-gpt/model_19072.pt"

    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768
    )

    ckpt = torch.load(file_path, map_location=device, weights_only=False)

    config = ckpt['config']

    model = GPT(**config) if isinstance(config, dict) else GPT(config)

    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    evaluate_hellaswag(model)

    # test text generated

    tokenizer = GPT2TokenizerFast.from_pretrained(f"data/gpt2", local_files_only=True)

    enc = tokenizer("Transformer", truncation=True, max_length=1024, add_special_tokens=False, return_tensors="pt")

    input_ids = enc["input_ids"]

    attention_mask = enc["attention_mask"]

    gen_ids = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=20,
                do_sample=False,
                top_k=10,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )[0]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    print("Generated text:", output_text)
