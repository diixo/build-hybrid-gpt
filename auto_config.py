
import json
from transformers import GPT2TokenizerFast
from model_llama import GPTLlama
import torch
import os

class AutoConfigLlama:

    BLOCK_SIZE = 4096   # standard context length for llama models

    SIZE_MAP = {
        "gpt2": {
                "block_size": BLOCK_SIZE,
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "flash_attn": True,
            },
        "mini": {
                "block_size": BLOCK_SIZE,
                "n_layer": 16,
                "n_head": 16,
                "n_embd": 1024,
                "flash_attn": True,
            }
    }


    @staticmethod
    def from_config(size_type: str, tokenizer_type="gpt2"):

        if size_type not in AutoConfigLlama.SIZE_MAP:
            raise ValueError(f"Unknown size_type: {size_type}")

        tokenizer = GPT2TokenizerFast.from_pretrained(f"data/{tokenizer_type}", local_files_only=True)

        # Extract sizes
        vocab_sz = len(tokenizer.get_vocab())   # size include special tokens
        print("Vocab size: tokenizer =", vocab_sz)


        # Check alls special tokens
        print(f"Special tokens =\n{json.dumps(tokenizer.special_tokens_map, indent=2)}")

        # Check eos_token_id and the token itself
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("EOS token string:", repr(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)))

        config_kwargs = dict(vocab_size=vocab_sz, rope_base=10000.0, use_rope=True, model_type=size_type)

        config_kwargs.update(AutoConfigLlama.SIZE_MAP[size_type])

        print(f"config_kwargs =\n{json.dumps(config_kwargs, indent=2)}")

        # get the model class
        model = GPTLlama(**config_kwargs)

        return model, tokenizer


    @staticmethod
    def from_pretrained_model(file_path: str):

        if not os.path.exists(file_path): return None

        ckpt = torch.load(file_path, map_location="cpu", weights_only=False)

        extra = ckpt.get("extra", {})
        print("extra:", extra)

        config = ckpt['config']

        # get the model class from mapping
        model_cls = GPTLlama

        # create the model instance use mapped class
        model = model_cls(**config) if isinstance(config, dict) else model_cls(config)
        model.load_state_dict(ckpt['model'])

        return model

