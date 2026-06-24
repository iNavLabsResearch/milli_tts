"""Checkpoint save/load + rotation."""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional

import torch

from milli_tts.core.logger import get_logger

log = get_logger("training.checkpoint")


class CheckpointManager:
    def __init__(self, ckpt_dir: str, keep_last_n: int = 3) -> None:
        self.dir = ckpt_dir
        self.keep_last_n = keep_last_n
        os.makedirs(ckpt_dir, exist_ok=True)

    def _path(self, step: int) -> str:
        return os.path.join(self.dir, f"step_{step:08d}.pt")

    @property
    def latest_path(self) -> str:
        return os.path.join(self.dir, "latest.pt")

    @property
    def best_path(self) -> str:
        return os.path.join(self.dir, "best.pt")

    def save(self, *, step: int, model, optimizer, scheduler,
             extra: Optional[Dict] = None, is_best: bool = False) -> str:
        payload = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "extra": extra or {},
        }
        path = self._path(step)
        torch.save(payload, path)
        torch.save(payload, self.latest_path)
        if is_best:
            torch.save(payload, self.best_path)
            log.info("New best checkpoint @ step %d", step)
        self._rotate()
        log.info("Saved checkpoint -> %s", path)
        return path

    def _rotate(self) -> None:
        ckpts = sorted(glob.glob(os.path.join(self.dir, "step_*.pt")))
        excess = len(ckpts) - self.keep_last_n
        for old in ckpts[:max(0, excess)]:
            try:
                os.remove(old)
            except OSError:
                pass

    def resolve(self, resume_from: str) -> Optional[str]:
        if not resume_from or resume_from.lower() == "none":
            return None
        if resume_from == "latest":
            return self.latest_path if os.path.exists(self.latest_path) else None
        if resume_from == "best":
            return self.best_path if os.path.exists(self.best_path) else None
        return resume_from if os.path.exists(resume_from) else None

    def load(self, path: str, *, model, optimizer=None, scheduler=None,
             map_location="cpu") -> Dict:
        payload = torch.load(path, map_location=map_location)
        model.load_state_dict(payload["model"])
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        log.info("Resumed from %s (step %d)", path, payload.get("step", 0))
        return payload
