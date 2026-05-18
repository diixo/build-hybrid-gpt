"""
https://huggingface.co/datasets/aitetic/wikipedia/tree/main/20220301.en
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python wikipedia.py
Will save shards to the local directory "wikipedia-en".
"""

import os
import multiprocessing as mp
import numpy as np
from transformers import GPT2TokenizerFast
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm


# ------------------------------------------
local_dir = "wikipedia"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


# download the dataset
fw = load_dataset("aitetic/wikipedia", name="20220301.en", split="train")

tokenizer = GPT2TokenizerFast.from_pretrained("data/gpt-noomo-32k", local_files_only=True)

eot = tokenizer.eos_token_id
if eot is None:
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
if eot is None or eot < 0:
    raise ValueError("EOS token not found")


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents

    ids = tokenizer.encode(doc["text"], add_special_tokens=False)
    tokens.extend(ids)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)



def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


if __name__ == "__main__":

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        row_amount = 0
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            row_amount += 1

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{local_dir}_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar.close()
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0: len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{local_dir}_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

        print(f"tokens.all={token_count}, rows={row_amount}, shards={shard_index+1}, nprocs={nprocs}")
