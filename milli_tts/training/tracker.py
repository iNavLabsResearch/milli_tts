"""Experiment tracking via Weights & Biases (Adapter/Facade pattern).

Wraps W&B behind a tiny interface so the trainer never imports wandb directly.
If wandb is disabled or unavailable, every method becomes a no-op — training
keeps working, just without dashboards. Realtime loss/lr/accuracy curves and
periodic audio samples are logged so you watch graphs live in Colab.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from milli_tts.core.config import AppConfig
from milli_tts.core.logger import get_logger

log = get_logger("training.tracker")


class WandbTracker:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.wandb.enabled)
        self._wandb = None
        self._run = None

    def start(self) -> "WandbTracker":
        if not self.enabled:
            log.info("W&B disabled (wandb.enabled=false).")
            return self
        try:
            import wandb
        except Exception as exc:  # pragma: no cover
            log.warning("wandb not importable (%s); disabling tracking.", exc)
            self.enabled = False
            return self

        api_key = self.cfg.wandb.api_key
        if api_key:
            os.environ.setdefault("WANDB_API_KEY", api_key)
            try:
                wandb.login(key=api_key, relogin=False)
            except Exception as exc:  # pragma: no cover
                log.warning("wandb.login failed (%s); using existing creds.", exc)

        self._wandb = wandb
        self._run = wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            name=self.cfg.project.run_name,
            mode=self.cfg.wandb.mode,
            config=self.cfg.raw,
            resume="allow",
        )
        # Make `train/step` the x-axis for all train panels (cleaner than the
        # internal wandb step, and robust to irregular logging cadence).
        try:
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("val/*", step_metric="train/step")
        except Exception:
            pass
        log.info("W&B run started: %s", getattr(self._run, "url", "(offline)"))
        return self

    def watch(self, model) -> None:
        if self.enabled and self._wandb and self.cfg.wandb.watch_model:
            try:
                self._wandb.watch(model, log="gradients", log_freq=200)
            except Exception:
                pass

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.enabled and self._wandb:
            # Deliberately DON'T pass wandb's internal `step`: the x-axis is
            # driven by the logged "train/step" metric (see define_metric in
            # start()). Forcing the internal step would make wandb DROP a second
            # log at the same step — exactly what was silently swallowing the
            # val/* points logged right after train/* on eval steps.
            self._wandb.log(metrics)

    def log_audio(self, tag: str, wav, sample_rate: int,
                  step: Optional[int] = None, caption: str = "") -> None:
        if not (self.enabled and self._wandb and self.cfg.wandb.log_audio_samples):
            return
        try:
            import numpy as np

            if hasattr(wav, "detach"):
                wav = wav.detach().float().cpu().numpy()
            wav = np.asarray(wav).squeeze()
            audio = self._wandb.Audio(wav, sample_rate=sample_rate, caption=caption)
            # No explicit step (see log()): keep the x-axis on train/step and
            # avoid same-step drops.
            self._wandb.log({tag: audio, "train/step": step})
        except Exception as exc:  # pragma: no cover
            log.debug("log_audio failed: %s", exc)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
