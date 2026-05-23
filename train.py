"""Training script for the Muon-optimized SQL router.

Trains a small GPT model to convert natural-language prompts into SQL
queries against the engram database. Uses the Muon optimizer with
Newton-Schulz orthogonalization and orthogonalization regularization.

After each epoch, the model is gated by the evaluation framework
(ECE, McNemar, PR_bc). If PR_bc < 3.0 or ECE > 0.10, training stops
and the previous epoch's weights are restored.

Usage:
    cd /mnt/primesauce/Elpis
    python -m llm-from-scratch.train

Or directly:
    cd /mnt/primesauce/Elpis/llm-from-scratch
    python train.py --epochs 100 --batch-size 32
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add llm-from-scratch to path for local imports
_elpis_root = Path(__file__).parent.parent  # /mnt/primesauce/Elpis
_elf_path = Path(__file__).parent  # llm-from-scratch directory
sys.path.insert(0, str(_elpis_root))
sys.path.insert(0, str(_elf_path))
from model import GPT, GPTConfig
from muon import Muon
from elpis.evaluation import EvaluationGate, EvaluationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("elpis.train")


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Training configuration for the Muon router."""

    # Data
    train_file: str = "data/router_train.jsonl"
    val_file: str = "data/router_val.jsonl"
    vocab_size: int = 97          # ASCII printable (0-94) + newline (96) + padding (95)
    block_size: int = 128         # context window

    # Model
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    ortho_coeff: float = 0.01     # orthogonalization regularization

    # Optimizer (Muon)
    optimizer_lr: float = 1e-3
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.95
    optimizer_weight_decay: float = 0.01
    optimizer_ns_steps: int = 5

    # Training
    epochs: int = 100
    batch_size: int = 32
    max_tokens: int = 256         # max tokens to generate per prompt

    # Evaluation gates
    ece_threshold: float = 0.05
    mcnemar_threshold: float = 0.05
    pr_bc_threshold: float = 3.0

    # Output
    model_dir: str = "models"
    checkpoint_interval: int = 1  # save checkpoint every N epochs


# ── Dataset ──────────────────────────────────────────────────────────


class RouterDataset(Dataset):
    """Dataset for SQL router training.

    Each example contains:
      - prompt: natural-language query
      - target: SQL query to retrieve the engram

    Tokenized as: prompt_tokens + target_tokens
    The model learns to predict the next token in the target sequence.
    """

    def __init__(
        self,
        file_path: str,
        vocab_size: int = 65,
        block_size: int = 128,
    ):
        self.file_path = Path(file_path)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.examples = self._load_examples()

    def _load_examples(self) -> List[Dict[str, str]]:
        """Load JSONL examples from file."""
        examples = []
        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        logger.info("Loaded %d examples from %s", len(examples), self.file_path)
        return examples

    def _tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs using ASCII encoding.

        Tokens are character-level: ord(c) for printable ASCII,
        with newline mapped to 96.
        """
        tokens = []
        for c in text:
            if c == "\n":
                tokens.append(96)
            elif 32 <= ord(c) <= 126:
                tokens.append(ord(c) - 32)  # 0-94 for printable ASCII
            else:
                tokens.append(0)  # unknown
        return tokens

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_tensor, target_tensor) for a single example.

        Input: prompt tokens
        Target: prompt tokens + target tokens (shifted for next-token prediction)
        """
        example = self.examples[idx]
        prompt = example["prompt"]
        target = example["target"]

        prompt_tokens = self._tokenize(prompt)
        target_tokens = self._tokenize(target)

        # Full sequence: prompt + target
        full_seq = prompt_tokens + target_tokens + [96]  # newline at end

        # Truncate to block_size
        if len(full_seq) > self.block_size:
            full_seq = full_seq[:self.block_size]

        # Pad to block_size
        while len(full_seq) < self.block_size:
            full_seq.append(0)  # padding token

        # Convert to tensors
        seq = torch.tensor(full_seq, dtype=torch.long)

        # For next-token prediction: input is seq[:-1], target is seq[1:]
        input_tensor = seq[:-1]
        target_tensor = seq[1:]

        return input_tensor, target_tensor


# ── Training Loop ────────────────────────────────────────────────────


class Trainer:
    """Trains the GPT model with Muon optimizer and evaluation gates."""

    def __init__(self, config: TrainConfig, device: torch.device):
        self.config = config
        self.device = device

        # Model
        model_config = GPTConfig(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            ortho_coeff=config.ortho_coeff,
        )
        self.model = GPT(model_config).to(device)
        self.model_config = model_config

        # Optimizer (Muon)
        self.optimizer = Muon(
            self.model.parameters(),
            lr=config.optimizer_lr,
            beta1=config.optimizer_beta1,
            beta2=config.optimizer_beta2,
            weight_decay=config.optimizer_weight_decay,
            ns_steps=config.optimizer_ns_steps,
        )

        # Evaluation gate
        self.gate = EvaluationGate(
            ece_threshold=config.ece_threshold,
            mcnemar_threshold=config.mcnemar_threshold,
            pr_bc_threshold=config.pr_bc_threshold,
        )

        # Best model tracking
        self.best_model_state = None
        self.best_result = None
        self.best_epoch = 0

    def train(self) -> None:
        """Run the full training loop with evaluation gates."""
        # Create datasets
        train_dataset = RouterDataset(
            self.config.train_file,
            self.config.vocab_size,
            self.config.block_size,
        )
        val_dataset = RouterDataset(
            self.config.val_file,
            self.config.vocab_size,
            self.config.block_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info("Training on %d examples, validating on %d",
                     len(train_dataset), len(val_dataset))
        logger.info("Model: %d parameters", self._count_params())

        # Save initial checkpoint
        self._save_checkpoint(0)
        self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        for epoch in range(1, self.config.epochs + 1):
            # Train one epoch
            epoch_loss = self._train_epoch(train_loader)
            logger.info("Epoch %d/%d - Loss: %.4f",
                        epoch, self.config.epochs, epoch_loss)

            # Evaluate
            result = self.gate.evaluate(self.model, val_loader, self.device)
            logger.info("  Evaluation: %s", result.summary())

            # Check gate
            if result.passed:
                logger.info("  Gate PASSED — saving checkpoint")
                # Update best
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                self.best_result = result
                self.best_epoch = epoch
            else:
                logger.warning("  Gate FAILED — restoring previous checkpoint")
                # Restore best weights
                self.model.load_state_dict(self.best_model_state)

            # Check for spectral collapse (hard stop condition)
            if result.pr_bc < 3.0:
                logger.error("SPECTRAL COLLAPSE DETECTED (PR_bc=%.2f < 3.0). STOPS.",
                             result.pr_bc)
                break

            # Check ECE threshold
            if result.ece > 0.10:
                logger.warning("ECE too high (%.4f > 0.10). Continuing with checkpoint restore.")

            # Save checkpoint if interval reached
            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

        # Save final model
        final_model_path = Path(self.config.model_dir) / "elpis_router.pt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_config": self.model_config.__dict__,
            "model_state": self.model.state_dict(),
            "best_epoch": self.best_epoch,
            "best_ece": self.best_result.ece if self.best_result else 0.0,
            "best_accuracy": self.best_result.accuracy if self.best_result else 0.0,
            "best_pr_bc": self.best_result.pr_bc if self.best_result else 0.0,
            "training_config": self.config.__dict__,
        }, str(final_model_path))
        logger.info("Saved trained router to %s", final_model_path)

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            # Forward pass
            loss = self.model(input_ids, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.model_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
        }, str(path))

    def _count_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    """Entry point for training."""
    # Determine base directory
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)

    # Create default data directory and generate data if needed
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    if not (data_dir / "router_train.jsonl").exists():
        logger.info("No training data found. Generating...")
        sys.path.insert(0, str(base_dir / "elpis"))
        from generate_training_data import main as gen_data
        gen_data()

    # Default config
    config = TrainConfig(
        train_file=str(data_dir / "router_train.jsonl"),
        val_file=str(data_dir / "router_val.jsonl"),
    )

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Create and run trainer
    trainer = Trainer(config, device)
    trainer.train()


if __name__ == "__main__":
    main()
