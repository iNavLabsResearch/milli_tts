"""Checkpoint save/load + rotation."""

from __future__ import annotations

import glob
import os

import shutil
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

    @staticmethod
    def _atomic_save(payload: Dict, path: str) -> None:
        """Serialize to ``<path>.tmp`` then atomically rename onto ``path``.

        A crash or out-of-disk failure mid-write leaves the previous ``path``
        (if any) intact instead of a half-written, unloadable file — so a full
        disk never corrupts the resume point.
        """
        tmp = f"{path}.tmp"
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)  # atomic on the same filesystem
        except Exception:
            try:  # don't leave a partial temp file eating disk after a failure
                os.remove(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_link(src: str, dst: str) -> None:
        """Point ``dst`` at ``src`` without writing a second multi-GB copy.

        Uses a hardlink (same inode, ~0 extra bytes) so ``latest.pt`` /
        ``best.pt`` don't each duplicate the checkpoint — this is what kept the
        dir at ~3x the per-checkpoint size and filled the disk. Falls back to a
        real copy if hardlinks aren't supported (e.g. different volumes). Builds
        a temp first then renames, so ``dst`` is never left half-updated.
        """
        tmp = f"{dst}.tmp"
        try:
            os.remove(tmp)
        except OSError:
            pass
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dst)

    def save(self, *, step: int, model, optimizer, scheduler,
             extra: Optional[Dict] = None, is_best: bool = False) -> str:
        payload = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "extra": extra or {},
        }
        # Free rotated-out checkpoints *before* writing the new one so peak disk
        # usage stays bounded by keep_last_n, not keep_last_n + 1.
        self._rotate(reserve=1)
        path = self._path(step)
        try:
            self._atomic_save(payload, path)
            self._atomic_link(path, self.latest_path)
            if is_best:
                self._atomic_link(path, self.best_path)
                log.info("New best checkpoint @ step %d", step)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to write checkpoint into {self.dir!r}: {exc}. This is "
                "almost always a full disk — each checkpoint here is multi-GB. "
                "Check `df -h`, free space (old runs, HF/wandb caches), lower "
                "training.keep_last_n_checkpoints, or point "
                "paths.checkpoint_dir at a larger volume. Your previous "
                "latest.pt is left intact, so resume with "
                "training.resume_from=latest.") from exc
        self._rotate()
        log.info("Saved checkpoint -> %s", path)
        return path

    def prune_after_push(self) -> None:
        """Delete local ``step_*.pt`` snapshots once the resume point is on HF.

        Called from the HF upload thread *after a confirmed successful push* of
        ``latest.pt``. ``latest.pt`` and ``best.pt`` stay (they're hardlinks, so
        the data survives even though the ``step_*`` names are removed); the
        backed-up history lives on the Hub. This caps local checkpoint disk at
        ~one checkpoint, so periodic snapshots can't accumulate and refill the
        disk between rotations.
        """
        keep = {os.path.realpath(self.latest_path),
                os.path.realpath(self.best_path)}
        removed = 0
        for old in glob.glob(os.path.join(self.dir, "step_*.pt")):
            if os.path.realpath(old) in keep:
                continue  # don't unlink the inode latest/best point at
            try:
                os.remove(old)
                removed += 1
            except OSError:
                pass
        if removed:
            log.info("Pruned %d local step checkpoint(s) after HF push.", removed)

    def _rotate(self, reserve: int = 0) -> None:
        """Delete oldest ``step_*.pt`` beyond ``keep_last_n``.

        ``reserve`` trims an extra ``reserve`` files to make room for that many
        about-to-be-written checkpoints, bounding peak disk usage. Also sweeps
        stale ``*.tmp`` left by a previously interrupted save.
        """
        for stale in glob.glob(os.path.join(self.dir, "*.pt.tmp")):
            try:
                os.remove(stale)
            except OSError:
                pass
        ckpts = sorted(glob.glob(os.path.join(self.dir, "step_*.pt")))
        excess = len(ckpts) - max(0, self.keep_last_n - reserve)
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
