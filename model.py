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
    """Configuration for the SQL router GPT model.

    Architecture tuned for SQL generation:
      vocab_size=4096  – covers SQL keywords, operators, and expanded ASCII
      n_embd=256       – small but sufficient for SQL syntax patterns
      n_layer=4        – minimal depth, fast inference
      n_head=8         – multi-head for attention diversity
      block_size=256   – context window for prompt + SQL generation
      d_ff = 4*n_embd  – standard transformer feedforward (1024)
    """

    # Tokenizer
    vocab_size: int = 97          # printable ASCII (32-126) = 95 chars + newline(96) + pad(0)
    block_size: int = 256         # context window (prompt + SQL)

    # Transformer
    n_layer: int = 4              # minimal depth, fast inference
    n_head: int = 8
    n_embd: int = 256             # d_model
    d_ff: int = 1024              # feedforward dimension (4 * n_embd)

    # Dropout
    dropout: float = 0.15          # increased from 0.1 to prevent overfitting to 22 targets

    # Orthogonalization regularization
    ortho_coeff: float = 0.0     # disabled — ortho regularization on weight matrices 
                                  # destroys representational capacity (uniform singular values)


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
            nn.Linear(config.n_embd, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.n_embd),
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
                wte=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=97),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Do NOT tie weights – weight tying severely constrains the embedding space
        # and causes spectral collapse on this task. Keep wte and lm_head separate.
        # self.transformer.wte.weight = self.lm_head.weight

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
                ignore_index=97,
            )

            # Add orthogonalization regularization on key projection matrices
            ortho_loss = self._compute_ortho_loss()
            loss = loss + self.config.ortho_coeff * ortho_loss
            return loss

        return logits

    def compute_ortho_loss(self) -> torch.Tensor:
        """Public alias for _compute_ortho_loss, used during training.
        
        ortho_loss = sum(||W@W^T - I||_F) * ortho_coeff for each linear layer's weight matrix.
        This encourages weight matrices to be orthogonal, improving training stability.
        
        Returns:
            Scalar loss tensor
        """
        return self._compute_ortho_loss()

    def _compute_ortho_loss(self) -> torch.Tensor:
        """Compute ||W @ W^T - I||_F for ALL linear layer weight matrices.

        This regularization encourages all weight matrices (not just attention)
        to be orthonormal, preventing spectral collapse of the embedding space.
        We normalize by the target rank to get a meaningful per-dimension metric.
        """
        ortho_sum = torch.tensor(0.0, device=self.lm_head.weight.device)
        count = 0

        # Iterate ALL linear layers in the model
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            W = module.weight  # (out_features, in_features)
            if W.numel() == 0:
                continue

            out_f, in_f = W.shape

            if in_f <= out_f:
                # Tall or square: use W^T @ W ≈ I (columns are orthonormal)
                WtW = W.t() @ W  # (in_f, in_f)
                I = torch.eye(in_f, device=W.device)
                ortho_sum += (WtW - I).pow(2).sum() / max(in_f, 1)
            else:
                # Wide: use W @ W^T ≈ I (rows are orthonormal)
                WWt = W @ W.t()  # (out_f, out_f)
                I = torch.eye(out_f, device=W.device)
                ortho_sum += (WWt - I).pow(2).sum() / max(out_f, 1)

            count += 1

        # Also add embedding diversity: encourage rows of wte to be diverse
        # by penalizing correlation between any two distinct embedding vectors
        wte = self.transformer.wte.weight  # (vocab_size, n_embd)
        if wte.shape[0] > 1 and wte.shape[1] > 1:
            # Sample a manageable number of embeddings for efficiency
            max_emb = min(wte.shape[0], 128)
            sampled = wte[:max_emb]
            # Normalize to unit vectors
            norms = sampled.norm(dim=1, keepdim=True).clamp(min=1e-8)
            sampled_normed = sampled / norms
            # Gram matrix: G[i,j] = dot(normalized_i, normalized_j)
            G = sampled_normed @ sampled_normed.t()  # (max_emb, max_emb)
            I = torch.eye(max_emb, device=G.device)
            # Off-diagonal correlation should be 0
            mask = 1 - I
            ortho_sum += (G * mask).pow(2).sum() / max(max_emb * (max_emb - 1), 1)

        return ortho_sum / max(count + 1, 1)  # normalize by total number of regularized terms

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        """Autoregressively generate tokens.

        idx: (B, T) initial token sequence
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature (0.0 = greedy argmax)
        top_k: if set, only sample from top-k logits
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0.0:
                # Greedy: pick the highest logit directly
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
