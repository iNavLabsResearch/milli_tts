"""Optimizer + LR schedule construction."""

from __future__ import annotations

import math
from typing import List

import torch

from milli_tts.core.config import TrainingConfig


def build_optimizer(model: torch.nn.Module, cfg: TrainingConfig):
    """AdamW with no weight decay on norms/biases/embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower() \
                or "emb" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                             eps=1e-8)


class CosineWarmupSchedule:
    """Linear warmup then cosine decay to ``min_lr``."""

    def __init__(self, optimizer, *, warmup_steps: int, max_steps: int,
                 base_lr: float, min_lr: float) -> None:
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.max_steps = max(self.warmup + 1, max_steps)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self._step = 0

    def _lr(self, step: int) -> float:
        if step < self.warmup:
            return self.base_lr * step / self.warmup
        progress = (step - self.warmup) / (self.max_steps - self.warmup)
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cos

    def step(self) -> float:
        self._step += 1
        lr = self._lr(self._step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def state_dict(self):
        return {"_step": self._step}

    def load_state_dict(self, sd):
        self._step = sd.get("_step", 0)
