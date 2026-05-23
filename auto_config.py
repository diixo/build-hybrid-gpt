
import json
from transformers import GPT2TokenizerFast
from model_llama import GPTRForCausalLM, GPTConfig
from pathlib import Path
import torch
import os


class AutoConfigModel:

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
    def _resolve_model_class(architecture: str | None):
        if not architecture or architecture == GPTRForCausalLM.__name__:
            return GPTRForCausalLM

        raise ValueError(f"Unsupported architecture: {architecture}")


    @staticmethod
    def from_config(size_type: str, tokenizer_type="gpt2"):

        if size_type not in AutoConfigModel.SIZE_MAP:
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

        config_kwargs.update(AutoConfigModel.SIZE_MAP[size_type])

        print(f"config_kwargs =\n{json.dumps(config_kwargs, indent=2)}")

        # get the model class
        model = GPTRForCausalLM(**config_kwargs)

        return model, tokenizer


    @staticmethod
    def from_pretrained_model(file_path: str):

        if not os.path.exists(file_path): return None

        ckpt = torch.load(file_path, map_location="cpu", weights_only=False)

        extra = ckpt.get("extra", {})
        print("extra:", extra)

        config = ckpt['config']

        # get the model class from mapping
        model_cls = AutoConfigModel._resolve_model_class(ckpt.get("architecture", None))

        # create the model instance use mapped class
        model = model_cls(**config) if isinstance(config, dict) else model_cls(config)
        model.load_state_dict(ckpt['model'])
        model.eval()
        return model


    @staticmethod
    def from_pretrained(
        repo_id_or_path: str,
        checkpoint_file: str | None = None,
        config_file: str = "config.json",
        revision: str | None = None,
        map_location: str | torch.device = "cpu",
        local_files_only: bool = False,
        strict: bool = True,
        config: GPTConfig | dict | None = None,
    ):
        """
        Load a GPTRForCausalLM checkpoint from a local path or a Hugging Face Hub repository.

        Supported checkpoint formats:
        1) raw state_dict
        2) dict with keys like {"model": state_dict, "config": {...}}
        3) separate config.json next to the checkpoint file

        Returns None when the requested Hugging Face repo, revision, or cached entry does not exist.
        """
        source_path = Path(repo_id_or_path)

        if source_path.is_file():
            checkpoint_path = source_path
            local_dir = source_path.parent
        else:
            if source_path.is_dir():
                local_dir = source_path
            else:
                try:
                    from huggingface_hub import snapshot_download
                    from huggingface_hub.utils import (
                        EntryNotFoundError,
                        HFValidationError,
                        LocalEntryNotFoundError,
                        RepositoryNotFoundError,
                        RevisionNotFoundError,
                    )
                except ImportError as exc:
                    raise ImportError(
                        "huggingface_hub is required to load checkpoints from Hugging Face Hub"
                    ) from exc

                allow_patterns = [config_file]
                if checkpoint_file is not None:
                    allow_patterns.append(checkpoint_file)
                else:
                    allow_patterns.extend([
                        "model*.pt",
                        "model*.pth",
                        "model*.bin",
                        "model.safetensors",
                        "pytorch_model.bin",
                    ])

                try:
                    local_dir = Path(snapshot_download(
                        repo_id_or_path,
                        revision=revision,
                        local_files_only=local_files_only,
                        allow_patterns=allow_patterns,
                    ))
                except (
                    EntryNotFoundError,
                    HFValidationError,
                    LocalEntryNotFoundError,
                    RepositoryNotFoundError,
                    RevisionNotFoundError,
                ):
                    return None

            if checkpoint_file is not None:
                checkpoint_path = local_dir / checkpoint_file
            else:
                preferred_names = [
                    "model.pt",
                    "model.pth",
                    "pytorch_model.bin",
                    "model.safetensors",
                ]
                checkpoint_path = None
                for name in preferred_names:
                    candidate = local_dir / name
                    if candidate.is_file():
                        checkpoint_path = candidate
                        break

                if checkpoint_path is None:
                    matches = []
                    for pattern in ("model*.pt", "model*.pth", "model*.bin", "*.safetensors"):
                        matches.extend(sorted(local_dir.glob(pattern)))

                    if len(matches) == 1:
                        checkpoint_path = matches[0]
                    elif len(matches) > 1:
                        raise ValueError(
                            "Multiple checkpoint files found. Pass checkpoint_file explicitly: "
                            + ", ".join(path.name for path in matches)
                        )
                    else:
                        raise FileNotFoundError(f"No checkpoint file found in {local_dir}")

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        if checkpoint_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "safetensors is required to load .safetensors checkpoints"
                ) from exc
            checkpoint = load_file(str(checkpoint_path), device=str(map_location))
        else:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=map_location)


        if isinstance(config, GPTConfig):
            model_config = config
        else:
            checkpoint_config = None
            if isinstance(checkpoint, dict) and "config" in checkpoint:
                checkpoint_config = checkpoint["config"]
            else:
                config_path = local_dir / config_file
                if config_path.is_file():
                    with open(config_path, "r", encoding="utf-8") as file_obj:
                        checkpoint_config = json.load(file_obj)

            if config is not None:
                checkpoint_config = config

            if checkpoint_config is None:
                raise ValueError(
                    "Model config not found. Provide config=..., store it inside the checkpoint, "
                    f"or add {config_file} next to the weights."
                )

            if isinstance(checkpoint_config, GPTConfig):
                model_config = checkpoint_config
            elif isinstance(checkpoint_config, dict):
                model_config = GPTConfig(**checkpoint_config)
            else:
                model_config = GPTConfig(**vars(checkpoint_config))

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            for prefix in ("module.", "_orig_mod."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            cleaned_state_dict[clean_key] = value

        architecture = checkpoint.get("architecture") if isinstance(checkpoint, dict) else None

        model_cls = AutoConfigModel._resolve_model_class(architecture)
        model = model_cls(config=model_config)
        model.load_state_dict(cleaned_state_dict, strict=strict)
        model.eval()
        return model
