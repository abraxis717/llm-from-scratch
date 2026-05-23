"""GPT model for SQL generation — adapted from nanoGPT.

This model is designed to convert natural-language prompts into SQL queries
against the engram database. It uses character-level tokenization with a
small vocabulary (65 chars) and a reduced transformer configuration suitable
for training on a laptop.

Architecture:
  Input → Token Embedding + Positional Embedding →
  Transformer Blocks (n_layer × [LayerNorm, Self-Attn, MLP]) →
  LayerNorm → Linear → logits → next-token prediction

For the SQL router we use the final hidden state (after [BOS]) as the
classification vector and pass it through a small head to predict the
next token in the SQL sequence.

Usage:
    >>> from model import GPTConfig, GPT
    >>> config = GPTConfig(vocab_size=65, block_size=128, n_layer=4,
    ...                    n_head=4, n_embd=128)
    >>> model = GPT(config)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class GPTConfig:
    """Configuration for the SQL router GPT model."""

    # Tokenizer
    vocab_size: int = 97          # ASCII printable (0-94) + newline (96) + padding (95)
    block_size: int = 128         # context window (shorter for SQL)

    # Transformer
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128             # embedding dimension

    # Dropout
    dropout: float = 0.1

    # Orthogonalization regularization
    ortho_coeff: float = 0.01     # weight for ||W@W^T - I||_F


# ── Modules ──────────────────────────────────────────────────────────


class Head(nn.Module):
    """A single self-attention head."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        w = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        out = w @ self.value(x)  # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention with parallel heads."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        heads = [Head(config) for _ in range(config.n_head)]
        self.heads = nn.ModuleList(heads)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.out_proj(out))
        return out


class MLP(nn.Module):
    """Standard transformer feed-forward block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Use a plain list (not tuple) to avoid named-tuple issues
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: pre-norm → MultiHeadAttn → residual → MLP → residual."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ── GPT Model ────────────────────────────────────────────────────────


class GPT(nn.Module):
    """Full GPT model with orthogonalization regularization."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying for small vocab
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") and "mlp" in pn:
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        idx: (B, T) token indices
        targets: (B, T) target token indices (for computing loss)

        Returns:
            loss scalar (if targets given) or logits (B, T, vocab_size)
        """
        B, T = idx.shape

        # Validate block size
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

        # Positional embeddings
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        pos_emb = self.transformer.wpe(pos)  # (1, T, C)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Get the logits for all positions
        logits = self.lm_head(x)

        # If we have targets, compute loss
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

            # Add orthogonalization regularization on key projection matrices
            ortho_loss = self._compute_ortho_loss()
            loss = loss + self.config.ortho_coeff * ortho_loss
            return loss

        return logits

    def _compute_ortho_loss(self) -> torch.Tensor:
        """Compute ||W @ W^T - I||_F for key and value projection matrices.

        This regularization encourages the attention projection matrices
        to be orthonormal, preventing spectral collapse of the embedding space.
        """
        ortho_sum = torch.tensor(0.0, device=self.lm_head.weight.device)
        count = 0

        for block in self.transformer.h:
            for name, module in block.attn.named_modules():
                if isinstance(module, Head):
                    for proj_name in ("key", "value"):
                        proj = getattr(module, proj_name)
                        W = proj.weight  # (head_size, embed_dim)
                        if W.shape[0] >= W.shape[1]:
                            # Square or wide: compute W @ W.T
                            WWt = W @ W.t()
                            I = torch.eye(WWt.shape[0], device=WWt.device)
                            ortho_sum += (WWt - I).pow(2).sum()
                        else:
                            # Tall: compute W.T @ W
                            WtW = W.t() @ W
                            I = torch.eye(WtW.shape[0], device=WtW.device)
                            ortho_sum += (WtW - I).pow(2).sum()
                        count += 1

        return ortho_sum / max(count, 1)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        """Autoregressively generate tokens.

        idx: (B, T) initial token sequence
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature
        top_k: if set, only sample from top-k logits
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
