"""Training script for native SQL router — resumes from latest checkpoint.

Trains for up to 500 epochs or until val_loss < 0.3.
Saves checkpoints to models/native_sql_checkpoints/.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Add llm-from-scratch to path for local imports
_elf_path = Path(__file__).parent
sys.path.insert(0, str(_elf_path))
from model import GPT, GPTConfig

# Load muon module from hyphenated directory
_spec = importlib.util.spec_from_file_location("muon", _elf_path / "muon.py")
muon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(muon)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("elpis.train_native_sql")


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Training configuration for the SQL router."""

    # Data
    train_file: str = "data/router_train.jsonl"
    val_file: str = "data/router_val.jsonl"
    vocab_size: int = 98
    block_size: int = 256

    # Model architecture
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    d_ff: int = 1024
    dropout: float = 0.15
    ortho_coeff: float = 0.0

    # Optimizer (Muon with Newton-Schulz + momentum=0.9)
    optimizer_lr: float = 2e-4
    optimizer_beta1: float = 0.9

    # Training
    epochs: int = 500
    batch_size: int = 32
    max_tokens: int = 256

    # Native SQL accuracy evaluation
    sql_eval_batch_size: int = 16
    sql_top_k: int = 5
    sql_temperature: float = 0.0  # greedy for evaluation

    # Output
    model_dir: str = "models/native_sql_checkpoints"
    checkpoint_interval: int = 10


# ── Dataset ──────────────────────────────────────────────────────────


def _collate_fn(batch):
    """Custom collate: pad variable-length sequences within each batch."""
    seqs = [item[0] for item in batch]
    tgts = [item[1] for item in batch]
    max_len = max(len(s) for s in seqs)
    min_len = min(len(s) for s in seqs)
    pad = torch.full((max_len - min_len,), 97, dtype=seqs[0].dtype)
    padded_seq, padded_tgt = [], []
    for s, t in zip(seqs, tgts):
        slen = len(s)
        tlen = len(t)
        padded_seq.append(torch.cat([s, pad[:max_len - slen]]))
        padded_tgt.append(torch.cat([t, pad[:max_len - tlen]]))
    return torch.stack(padded_seq), torch.stack(padded_tgt)


class RouterDataset(Dataset):
    """Dataset for SQL router training."""

    def __init__(self, file_path: str, vocab_size: int = 98, block_size: int = 256):
        self.file_path = Path(file_path)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.examples = self._load_examples()

    def _load_examples(self) -> List[dict]:
        examples = []
        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        logger.info("Loaded %d examples from %s", len(examples), self.file_path)
        return examples

    def _tokenize(self, text: str) -> List[int]:
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
        example = self.examples[idx]
        prompt = example["prompt"]
        target = example["target"]

        prompt_tokens = self._tokenize(prompt)
        target_tokens = self._tokenize(target)

        full_seq = prompt_tokens + target_tokens + [96]  # newline at end

        if len(full_seq) > self.block_size:
            full_seq = full_seq[:self.block_size]

        seq = torch.tensor(full_seq, dtype=torch.long)
        return seq[:-1], seq[1:]


# ── Training Loop ────────────────────────────────────────────────────


class Trainer:
    """Trains the GPT model with Muon optimizer."""

    def __init__(self, config: TrainConfig, device: torch.device,
                 resume_from_checkpoint: str = None):
        self.config = config
        self.device = device

        # Model
        model_config = GPTConfig(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            d_ff=config.d_ff,
            dropout=config.dropout,
            ortho_coeff=config.ortho_coeff,
        )
        self.model = GPT(model_config).to(device)
        self.model_config = model_config

        # Optimizer (Muon with Newton-Schulz + beta1=0.9 momentum)
        self.optimizer = muon.Muon(
            self.model.parameters(),
            lr=config.optimizer_lr,
            beta1=config.optimizer_beta1,
        )

        # Best model tracking
        self.best_model_state = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.start_epoch = 1

        # Consecutive high-loss counter for early stopping
        self.stuck_epochs = 0
        self.high_loss_threshold = 0.3
        self.stuck_patience = 30

        # Load resume checkpoint if provided
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model, optimizer, and training state from checkpoint."""
        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Restore epoch counter
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info("  Resuming from epoch %d", self.start_epoch)
        
        # Restore best model state if available
        if "best_model_state" in checkpoint:
            self.best_model_state = {
                k: v.clone() for k, v in checkpoint["best_model_state"].items()
            }
            self.best_loss = checkpoint.get("best_loss", float("inf"))
            self.best_epoch = checkpoint.get("best_epoch", 0)
            logger.info("  Restored best model from epoch %d (loss=%.4f)",
                       self.best_epoch, self.best_loss)
        else:
            # No best state in checkpoint — use current model as initial best
            self.best_model_state = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }
            self.best_loss = float("inf")
            self.best_epoch = checkpoint.get("epoch", 0)
            logger.info("  Using epoch %d model as initial best", self.best_epoch)
        
        # Restore stuck counter if this is a resume
        self.stuck_epochs = checkpoint.get("stuck_epochs", 0)
        logger.info("  Stuck counter restored: %d", self.stuck_epochs)

        logger.info("  Model state loaded successfully")

    def train(self) -> None:
        """Run the full training loop."""
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
            collate_fn=_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_fn,
        )

        logger.info("Training on %d examples, validating on %d",
                     len(train_dataset), len(val_dataset))
        logger.info("Model: %d parameters", self._count_params())

        # Warmup: train one batch to check timing
        logger.info("Warmup epoch...")
        t0 = time.time()
        self._train_epoch(train_loader)
        warmup_time = time.time() - t0
        steps_per_epoch = len(train_loader)
        est_total_time = warmup_time * self.config.epochs
        logger.info("Warmup took %.1fs (%d steps). Estimated total: %.0fm (%.0fh)",
                     warmup_time, steps_per_epoch, est_total_time/60, est_total_time/3600)

        # Save initial checkpoint
        self._save_checkpoint(0)
        self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Load HebbianBrain for SQL accuracy evaluation
        try:
            base_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(base_dir / "elpis"))
            from hebbian_brain import HebbianBrain
            self.brain = HebbianBrain()
            logger.info("HebbianBrain loaded for SQL evaluation")
        except Exception as e:
            logger.warning("Failed to load HebbianBrain: %s — SQL accuracy will use fallback", e)
            self.brain = None

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            t0 = time.time()

            # Train one epoch
            epoch_loss = self._train_epoch(train_loader)
            train_time = time.time() - t0

            # Validate (loss)
            val_loss = self._validate(val_loader)

            # Native SQL accuracy evaluation
            sql_acc = 0.0
            if epoch % 50 == 0:  # Evaluate every 50 epochs to save time
                t_eval = time.time()
                sql_acc = self._eval_sql_accuracy(val_dataset, brain=self.brain)
                eval_time = time.time() - t_eval
                logger.info("  [SQL Eval] Accuracy: %.2f%% (%.1fs)", sql_acc * 100, eval_time)

            logger.info("Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f, SQL Acc: %.1f%% (%.1fs)",
                        epoch, self.config.epochs, epoch_loss, val_loss, sql_acc * 100, train_time)

            # Early stopping: stop when val_loss stays above 0.3 for 30 consecutive epochs
            if val_loss > 0.3 and epoch > 10:
                self.stuck_epochs += 1
            else:
                self.stuck_epochs = 0
            if self.stuck_epochs >= 30:
                logger.info("Early stopping: val_loss > 0.30 for %d consecutive epochs (best at epoch %d)",
                           self.stuck_epochs, self.best_epoch)
                break

            # Track best model by validation loss
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                logger.info("  New best val_loss=%.4f (epoch %d) — saving", val_loss, epoch)

            # Save checkpoint
            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model from epoch %d (val_loss=%.4f)",
                        self.best_epoch, self.best_loss)

        # Save final model as elpis_router_native_sql.pt
        final_model_path = Path(self.config.model_dir) / "elpis_router_native_sql.pt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_config": self.model_config.__dict__,
            "model_state": self.model.state_dict(),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_loss,
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

            # Forward pass (model.py includes ortho_loss internally)
            loss = self.model(input_ids, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self, dataloader: DataLoader) -> float:
        """Validate for one epoch. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                loss = self.model(input_ids, targets)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _eval_sql_accuracy(self, dataset: RouterDataset, brain=None) -> float:
        """Evaluate native SQL generation accuracy."""
        if len(dataset) == 0:
            return 0.0

        self.model.eval()
        correct = 0
        total = 0

        for i in range(0, len(dataset), self.config.sql_eval_batch_size):
            batch_idx = list(range(i, min(i + self.config.sql_eval_batch_size, len(dataset))))
            batch_examples = [dataset.examples[idx] for idx in batch_idx]

            batch_prompts = []
            for ex in batch_examples:
                tokens = dataset._tokenize(ex["prompt"]) + [96]
                batch_prompts.append(torch.tensor(tokens, dtype=torch.long, device=self.device))
            max_prompt_len = max(len(p) for p in batch_prompts)
            padded_prompts = []
            for p in batch_prompts:
                slen = len(p)
                padded_prompts.append(torch.cat([p, torch.full((max_prompt_len - slen,), 97, dtype=p.dtype, device=self.device)]))
            batch_tensor = torch.stack(padded_prompts)

            with torch.no_grad():
                generated = self.model.generate(
                    batch_tensor,
                    max_new_tokens=self.config.block_size // 2,
                    temperature=self.config.sql_temperature,
                )

            for j, ex in enumerate(batch_examples):
                gen_tokens = generated[j].tolist()
                prompt_len = len(dataset._tokenize(ex["prompt"])) + 1
                gen_text = self._detokenize_tokens(gen_tokens[prompt_len:])

                gen_sql = self._extract_sql(gen_text)

                if not gen_sql:
                    continue

                target_sql = ex.get("target", "")

                if brain is not None and target_sql:
                    try:
                        gen_results = brain.execute_sql(gen_sql, top_k=5)
                        target_results = brain.execute_sql(target_sql, top_k=5)

                        if gen_results and target_results:
                            gen_names = {r.get("name", "") for r in gen_results if r.get("name")}
                            target_names = {r.get("name", "") for r in target_results if r.get("name")}

                            if gen_names & target_names:
                                correct += 1
                            else:
                                gen_tok = set(gen_sql.lower().split())
                                tgt_tok = set(target_sql.lower().split())
                                if len(gen_tok & tgt_tok) >= 10:
                                    correct += 1
                    except Exception as e:
                        logger.debug(f"  Brain eval error: {e}")
                        gen_sql_lower = gen_sql.lower()
                        target_lower = target_sql.lower()
                        gen_tok = set(gen_sql_lower.split())
                        tgt_tok = set(target_lower.split())
                        if len(gen_tok & tgt_tok) >= 10:
                            correct += 1
                else:
                    gen_sql_lower = gen_sql.lower()
                    target_lower = target_sql.lower()
                    gen_tok = set(gen_sql_lower.split())
                    tgt_tok = set(target_lower.split())
                    if len(gen_tok & tgt_tok) >= 10:
                        correct += 1

                total += 1

        return correct / total if total > 0 else 0.0

    def _detokenize_tokens(self, tokens: list[int]) -> str:
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

    def _extract_sql(self, text: str) -> str:
        """Extract SQL query from generated text."""
        if not text:
            return ""
        text_upper = text.upper()
        select_pos = text_upper.find("SELECT")
        if select_pos == -1:
            return ""
        sql = text[select_pos:].strip()
        lines = sql.split("\n")
        sql = lines[0]
        if len(lines) > 1:
            sql += "\n" + "\n".join(lines[1:])
        if not sql.endswith(";"):
            sql += ";"
        return sql

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint with best model state."""
        checkpoint_dir = Path(self.config.model_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        save_dict = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        if self.best_model_state is not None:
            save_dict["best_model_state"] = self.best_model_state
            save_dict["best_loss"] = self.best_loss
            save_dict["best_epoch"] = self.best_epoch
        save_dict["stuck_epochs"] = self.stuck_epochs
        torch.save(save_dict, str(path))
        logger.info("  Saved checkpoint epoch %d (best: epoch %d, loss=%.4f)",
                   epoch, self.best_epoch, self.best_loss)

    def _count_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    """Entry point for training."""
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)

    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    if not (data_dir / "router_train.jsonl").exists():
        logger.info("No training data found. Generating...")
        sys.path.insert(0, str(base_dir / "elpis"))
        from generate_training_data import main as gen_data
        gen_data()

    config = TrainConfig(
        train_file=str(data_dir / "router_train.jsonl"),
        val_file=str(data_dir / "router_val.jsonl"),
    )

    # Find the latest checkpoint to resume from
    checkpoint_dir = base_dir / "models"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))
    if checkpoints:
        resume_from = str(checkpoints[-1])
        logger.info("Resuming from checkpoint: %s", resume_from)
    else:
        resume_from = None
        logger.info("No checkpoint found. Training from scratch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    trainer = Trainer(config, device, resume_from_checkpoint=resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
