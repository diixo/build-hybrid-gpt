from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from tqdm import tqdm
import json
from datasets import Dataset, concatenate_datasets



input_path = "data/noomo-32k"
output_path = "data/gpt-noomo-32k"

# export to gpt2 format for compatibility

tokenizer = GPT2TokenizerFast.from_pretrained(output_path, local_files_only=True, add_prefix_space=True)


added = tokenizer.add_special_tokens({
    "eos_token": "<|endoftext|>",
    "pad_token": "<|pad|>",
    "bos_token": "<|endoftext|>",
    "additional_special_tokens": [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|memory|>",
        "<|dialog|>",
        "<|task|>",
        "<|instruction|>",
        "###",
        ]
})

print(f"added: {added}, vocab_size: {len(tokenizer)}")

#tokenizer.save_pretrained(output_path)

test_text = "<|user|> What is the capital of France? <|assistant|> Paris. <|endoftext|>"

test_text = "<|user|> GPT is a type of large language model. <|assistant|> The chatGPT and other GPTs are based on a deep learning architecture called the transformer. <|endoftext|>"

print(f"Tokens: {tokenizer.tokenize(test_text)}")

print(f"Tokens: {tokenizer.tokenize('###What is Wikipedia? Assistant: ### Wikipedia is a free online encyclopedia.')}")
