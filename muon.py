"""Muon optimizer — MomentUm Orthogonalized by Newton-Schulz.

Muon updates weights by:
  1. Computing the gradient G.
  2. Orthogonalizing G via Newton-Schulz iteration (5 steps).
  3. Applying momentum and weight decay in the orthogonalized space.

The Newton-Schulz iteration approximates the matrix inverse square root:
    G_ortho = G (G^T G)^{-1/2}

This keeps gradient directions orthogonal and prevents spectral collapse
of the embedding space during training.

Usage:
    >>> from muon import Muon
    >>> optimizer = Muon(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.95)
    >>> for input, target in dataloader:
    ...     loss = model(input, target)
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _newton_schulz_5(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute M * (M^T M)^{-1/2}.

    Given M (batch × features), returns the orthogonalized version:
        M_ortho = M @ (M^T M)^{-1/2}

    The iteration:
        A = M^T M
        B = I - A/2
        for _ in range(steps):
            B = 1.5 * I - B @ B   (approximates A^{-1/2})
        result = M @ B

    This converges to the polar decomposition of M, extracting the
    orthogonal component.

    Args:
        M: Input tensor of shape (..., n, d)
        steps: Number of Newton-Schulz iterations (5 converges well)

    Returns:
        Orthogonalized tensor of same shape as M
    """
    # Normalize M for numerical stability
    # M_ortho = M @ (M^T M)^{-1/2}
    # We compute B = (M^T M)^{-1/2} iteratively

    # Reshape to 2D for matrix ops
    orig_shape = M.shape
    if M.dim() > 2:
        M = M.view(-1, M.shape[-1])

    batch_size, dim = M.shape

    # Compute M^T M
    MtM = M.t() @ M  # (dim, dim)

    # Newton-Schulz iteration: compute (MtM)^{-1/2}
    # Initialize B = I - MtM/2 (convergence requires ||I-A|| < 1)
    # Scale MtM first for stability
    sigma = MtM.norm()
    if sigma < 1e-8:
        return M.view(orig_shape)

    MtM_scaled = MtM / (sigma + 1e-8)
    I = torch.eye(dim, device=MtM.device, dtype=MtM.dtype)

    # B = I - MtM_scaled / 2
    B = I - MtM_scaled * 0.5

    # Newton-Schulz: B = 1.5*I - B @ B  (converges to A^{-1/2})
    for _ in range(steps):
        B = 1.5 * I - B @ B

    # Now B approximates (MtM)^{-1/2}
    # Orthogonalize: M_ortho = M @ B * sigma^{1/2}
    M_ortho = (M @ B) * (sigma.sqrt() + 1e-8)

    return M_ortho.view(orig_shape)


class Muon(torch.optim.Optimizer):
    """Muon optimizer: MomentUm Orthogonalized by Newton-Schulz.

    This optimizer applies Newton-Schulz orthogonalization to the gradient
    before applying momentum updates. The result is that gradient directions
    are kept orthogonal, preventing embedding collapse.

    Key parameters:
        lr: Learning rate
        beta1: Momentum coefficient (like Adam's beta1)
        beta2: RMS momentum coefficient (like Adam's beta2)
        weight_decay: L2 weight decay (applied after orthogonalization)
        ns_steps: Number of Newton-Schulz iterations
        ns_eps: Epsilon for numerical stability
        momentum_threshold: Minimum gradient norm to trigger NS iteration

    The orthogonalization is only applied to weight matrices with at
    least 2D shape (e.g., Linear.weight, Embedding.weight). Other
    parameters (biases, LayerNorm) use standard momentum updates.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        momentum_threshold: float = 1e-6,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta1 < 0.0 or beta1 > 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if beta2 < 0.0 or beta2 > 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            momentum_threshold=momentum_threshold,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            threshold = group["momentum_threshold"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1

                # Bias correction
                m = state["m"]
                v = state["v"]
                step = state["step"]

                # Momentum update: m = beta1 * m + (1 - beta1) * grad
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # RMS momentum: v = beta2 * v + (1 - beta2) * grad^2
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                m_corr = m / (1.0 - beta1 ** step)
                v_corr = v / (1.0 - beta2 ** step)

                # Compute adaptive scale (like Adam)
                scale = m_corr / (v_corr.sqrt() + eps)

                # Apply Newton-Schulz orthogonalization to 2D+ parameters
                if p.dim() >= 2 and grad.norm() > threshold:
                    scale = _newton_schulz_5(scale, steps=ns_steps)

                # Apply weight update: p = p - lr * scale + weight_decay * p
                if weight_decay > 0:
                    p.add_(p, alpha=weight_decay)

                p.add_(scale, alpha=-lr)

        return loss


# ── Tokenization utilities ────────────────────────────────────────────


def _tokenize_text(text: str) -> list[int]:
    """Convert text to token IDs (ASCII character-level)."""
    tokens = []
    for c in text:
        if c == "\n":
            tokens.append(96)
        elif 32 <= ord(c) <= 126:
            tokens.append(ord(c) - 32)
        else:
            tokens.append(0)
    return tokens


def _detokenize_text(tokens: list[int]) -> str:
    """Convert token IDs back to text."""
    chars = []
    for t in tokens:
        if t == 96:
            chars.append("\n")
        elif 0 <= t <= 94:
            chars.append(chr(t + 32))
        else:
            chars.append("?")
    return "".join(chars)
