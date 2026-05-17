"""
GPT-LLaMA model
1) Rotary Position Embeddings (RoPE) implementation.
2) tie_word_embeddings=True in GPT.from_pretrained to share weights between token embeddings and LM head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension
    flash_attn: bool = True # whether to use flash attention (scaled_dot_product_attention)
    model_type: str = ""    # model type

    # RoPE params:
    rope_base: float = 10000.0  # standard base (θ). For learning on length=2048 may use 10000.0
    use_rope: bool = True       # whether to use RoPE or not

    mlp_bias: bool = False


class RMSNorm(nn.Module):
    """
    RMSNorm with trainable scale-parameter, that compatible to LLaMA behavior.
    Equation:
        out = x / (sqrt(mean(x**2, dim=-1, keepdim=True)) + eps) * weight
    Where: weight — learnable vector (dim,)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Trainable scale-vector, initialize by ones like in LLaMA.
        self.weight = nn.Parameter(torch.ones(dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # calculate RMS along the last axis
        # use float32 accumulation for stability, but return as original dtype
        orig_dtype = x.dtype
        acc_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype

        x_acc = x.to(acc_dtype)
        rms = torch.sqrt((x_acc * x_acc).mean(dim=-1, keepdim=True) + self.eps)  # (..., 1)
        # normalize (into acc dtype), then scale by the trainable weight
        y = x_acc / rms
        # weight: (dim,) -> (1, 1, dim) or broadcastable
        weight = self.weight.to(acc_dtype)
        # support for arbitrary leading axes: to place the weight so that the last dim coincides
        # y shape: (..., dim); weight shape: (dim,) => broadcasting ok
        y = y * weight
        return y.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """
    Implements precomputed Rotary Position Embedding (RoPE) cache for efficiency.
    """
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 2048):

        super().__init__()
        assert dim % 2 == 0, "Head dimension must be even for RoPE"
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # precompute frequencies
        half_dim = dim // 2
        channel_range = 2 * torch.arange(0, half_dim, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                # (T, half_dim)
        freqs_cos = torch.cos(freqs)[None, None, :, :]  # (1, 1, T, half_dim)
        freqs_sin = torch.sin(freqs)[None, None, :, :]  # (1, 1, T, half_dim)

        # store as buffers (moved automatically with model.to(device))
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


    def apply_rotary(self, x: torch.Tensor, seq_len: int):
        assert x.ndim == 4, f"Expected 4D tensor (B, nH, T, d), got: {x.shape}"
        cos = self.freqs_cos[:, :, :seq_len, :].to(x.device)
        sin = self.freqs_sin[:, :, :seq_len, :].to(x.device)

        d = x.shape[3] // 2             # = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
        y1 = x1 * cos + x2 * sin        # rotate pairs of dims
        y2 = x1 * (-sin) + x2 * cos
        out = torch.cat([y1, y2], dim=3)  # re-assemble, = torch.cat([y1, y2], dim=-1)
        out = out.to(x.dtype)           # ensure input/output dtypes match
        return out


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.flash_attn = config.flash_attn
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.use_rope = config.use_rope
        self.rope_base = config.rope_base

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            self.rope = RotaryEmbedding(dim=head_dim, base=config.rope_base, max_seq_len=config.block_size)

        # causal mask via register_buffer
        if not self.flash_attn:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))


    def forward(self, x, attention_mask=None):

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention_mask: (B, T) -> (B, 1, 1, T) for broadcast

        # attention_mask: (B, T), where 1=real token, 0=pad
        if attention_mask is not None:
            # fix flash_attn with PyTorch back-end of scaled_dot_product_attention compatibility with is_causal=True
            # if self.flash_attn and torch.all(attention_mask == 1):
            # but make force solution more universally for both modes
            if torch.all(attention_mask == 1):
                # force fire-down attention mask for Flash Attention
                attn_mask = None
            else:
                # use attention_mask for manual attention implementation
                assert attention_mask.size(0) == B
                assert attention_mask.size(1) == T
                attn_mask = (attention_mask == 0)[:, None, None, :].to(device=x.device, dtype=torch.bool)  # (B,1,1,T)
        else:
            attn_mask = None

        if self.use_rope:
            q = self.rope.apply_rotary(q, T)
            k = self.rope.apply_rotary(k, T)

        if self.n_head == 1 or not self.flash_attn:
            # manual attention implementation
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1)) # (B, nh, T, T)

            # causal mask: as bool view(1, 1, T, T)
            causal_mask = (self.bias[:, :, :T, :T] == 0)

            # combine both masks
            if attn_mask is not None:
                # invert: True = masked
                full_mask = causal_mask | attn_mask
            else:
                full_mask = causal_mask

            # apply mask
            attn = attn.masked_fill(full_mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            y = attn @ v  # (B, nh, T, hs)
        else:
            # For Flash Attention, we need to combine the causal mask
            # and the padding mask into a single mask.
            # SDPA with Flash Attention works efficiently with boolean masks.

            # 1. Create the causal mask (lower triangular)
            # We can either use self.bias or build it on the fly
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)

            # 2. Combine it with the padding mask
            # padding_keep_mask has True where tokens are real (not padding)
            if attention_mask is not None:
                padding_keep_mask = (attention_mask == 1)[:, None, None, :].to(device=x.device, dtype=torch.bool)
                full_mask = causal_mask & padding_keep_mask
            else:
                full_mask = causal_mask

            # Use SDPA with the combined mask
            # Set is_causal=False because causality is already included in full_mask
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask, is_causal=False)


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd

        # SWiGLU: first is activation layer, second is gate layer
        self.c_fc1 = nn.Linear(config.n_embd, hidden_dim, bias=config.mlp_bias)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=config.mlp_bias)

        self.silu = nn.SiLU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.mlp_bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x):
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)

        # Element-wise product
        hidden = self.silu(x1) * x2
        return self.c_proj(hidden)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd) 
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class GPTLlama(nn.Module):
    def __init__(self, config: GPTConfig=None, **kwargs):
        super().__init__()
        if config is None:
            config = GPTConfig(**kwargs)

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme (tie_word_embeddings=True always)
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)


    def forward(self, idx, targets=None, attention_mask=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        if self.config.use_rope:
            x = tok_emb
        else:
            # forward the token and posisition embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
            x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            if attention_mask is not None:
                # do not calculate paddings in loss
                targets = targets.clone()
                targets[attention_mask == 0] = -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return GPTOutput(logits=logits, loss=loss)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rope:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,    # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
        max_new_tokens: int = 5,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:

        self.eval()
        block_size = self.config.block_size

        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else 0

        if attention_mask is not None:
            B, T0 = input_ids.shape

            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.long)
            assert attention_mask.dim() == 2 and attention_mask.size(0) == B, f"bad attention_mask {attention_mask.shape}"
            assert attention_mask.size(1) == input_ids.size(1), f"mask/input mismatch: {attention_mask.size()} vs {input_ids.size()}"


        pad = torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
        finished = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.bool)

        for _ in range(max_new_tokens):
            if eos_token_id is not None and torch.all(finished):
                break

            idx_cond = input_ids if input_ids.size(1) <= block_size else input_ids[:, -block_size:]

            if attention_mask is None:
                logits = self(idx_cond).logits # (B, t, V)
            else:

                if input_ids.size(1) <= block_size:
                    am_cond = attention_mask
                else:
                    am_cond = attention_mask[:, -block_size:] if attention_mask is not None else None

                # strong correspondence in length to idx_cond
                assert am_cond.size(1) == idx_cond.size(1)
                logits = self(idx_cond, attention_mask=am_cond).logits  # (B, t, V)

            logits = logits[:, -1, :]       # (B, V)

            if do_sample:
                logits = logits / temperature
                if top_k is not None:
                    # clamp top_k to valid range [1, V]
                    V = logits.size(-1)
                    k = max(1, min(int(top_k), V))
                    v, _ = torch.topk(logits, k, dim=-1)
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)      # (B,1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)   # (B,1)

            # if sequence already finished -> keep padding
            if eos_token_id is not None:
                idx_next = torch.where(finished[:, None], pad, idx_next)
                finished = finished | (idx_next.squeeze(1) == eos_token_id)

            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if attention_mask is not None:
                next_mask = (~finished).to(dtype=attention_mask.dtype).unsqueeze(1)  # (B,1) 1=active, 0=finished
                attention_mask = torch.cat((attention_mask, next_mask), dim=1)

        return input_ids
