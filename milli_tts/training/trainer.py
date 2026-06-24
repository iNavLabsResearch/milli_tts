"""Trainer — orchestrates the full fine-tuning loop (Facade pattern).

Responsibilities:
* build dataloader (streaming IndicVoices) + collator
* Mimi-encode waveforms to codes on-GPU each step (codec stays frozen)
* mixed-precision forward/backward with gradient accumulation + clipping
* cosine-warmup LR schedule
* realtime W&B logging (loss/lr/acc) + periodic decoded audio samples
* checkpoint save/rotate/resume

It deliberately knows nothing about the model internals beyond the
:class:`BaseTTSModel` contract, so any registered architecture trains here.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from milli_tts.core.config import AppConfig
from milli_tts.core.logger import get_logger
from milli_tts.core.static_memory_cache import StaticMemoryCache
from milli_tts.data.collator import DelayedStreamCollator, TTSBatch
from milli_tts.data.dataset import IndicVoicesDataset
from milli_tts.data.mimi_codec import MimiCodec
from milli_tts.data.text_tokenizer import TextTokenizer
from milli_tts.data.voice_bank import VoiceBank
from milli_tts.models.factory import build_model
from milli_tts.training.checkpoint import CheckpointManager
from milli_tts.training.optim import CosineWarmupSchedule, build_optimizer
from milli_tts.training.tracker import WandbTracker

log = get_logger("training.trainer")


class Trainer:
    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        self.cfg = cfg or StaticMemoryCache.config()
        self.device = StaticMemoryCache.device(self.cfg.training.device)
        self.tcfg = self.cfg.training

        # ---- shared singletons --------------------------------------- #
        self.tokenizer = TextTokenizer.from_config()
        self.voice_bank = VoiceBank.from_config()
        self.codec = MimiCodec.from_config(device=self.device)
        self.samples_per_frame = int(round(
            self.cfg.codec.sample_rate / self.cfg.codec.frame_rate))

        # ---- model --------------------------------------------------- #
        self.model = build_model(
            text_vocab_size=self.tokenizer.vocab_size,
            num_codebooks=self.cfg.codec.num_codebooks,
            codebook_size=self.cfg.codec.codebook_size,
        ).to(self.device)
        log.info("Model: %s | trainable params: %.1fM",
                 self.cfg.model.arch, self.model.num_parameters() / 1e6)

        # ---- optim / sched / ckpt / tracker -------------------------- #
        self.optimizer = build_optimizer(self.model, self.tcfg)
        self.scheduler = CosineWarmupSchedule(
            self.optimizer, warmup_steps=self.tcfg.warmup_steps,
            max_steps=self.tcfg.max_steps, base_lr=self.tcfg.lr,
            min_lr=self.tcfg.min_lr)
        self.ckpt = CheckpointManager(self.cfg.paths.checkpoint_dir,
                                      self.tcfg.keep_last_n_checkpoints)
        self.tracker = WandbTracker(self.cfg)

        self.amp_dtype, self.use_scaler = self._resolve_precision()
        try:  # torch>=2.4 prefers torch.amp.GradScaler("cuda", ...)
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)
        self.step = 0
        self.best_loss = float("inf")

        if self.tcfg.compile_model:
            try:
                self.model = torch.compile(self.model)
                log.info("torch.compile enabled.")
            except Exception as exc:  # pragma: no cover
                log.warning("torch.compile failed: %s", exc)

    # ------------------------------------------------------------------ #
    def _resolve_precision(self):
        if self.device.type != "cuda":
            return torch.float32, False
        want = self.tcfg.precision.lower()
        if want == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        if want == "bf16":
            log.warning("bf16 unsupported on this GPU; using fp16 (good for T4).")
            return torch.float16, True
        if want == "fp16":
            return torch.float16, True
        return torch.float32, False

    def _build_loader(self) -> DataLoader:
        dataset = IndicVoicesDataset(tokenizer=self.tokenizer,
                                     voice_bank=self.voice_bank,
                                     register_voices=True)
        collator = DelayedStreamCollator(text_pad_id=self.tokenizer.pad_id)
        return DataLoader(
            dataset, batch_size=self.tcfg.batch_size, collate_fn=collator,
            num_workers=self.tcfg.num_workers, pin_memory=(self.device.type == "cuda"),
            drop_last=True, persistent_workers=self.tcfg.num_workers > 0)

    # ------------------------------------------------------------------ #
    def _preflight_data(self) -> bool:
        """Pull ONE sample in the main process (visible logs) before the loop.

        This is the "load the stream first, then train" step: it surfaces the
        dataset build logs, confirms access + field mapping, and times the first
        usable sample — instead of the loop silently hanging in a worker.
        """
        log.info("Preflight: pulling one sample from the stream to verify "
                 "access + field mapping (this is the slow part)…")
        probe = IndicVoicesDataset(tokenizer=self.tokenizer,
                                   voice_bank=self.voice_bank,
                                   register_voices=False)
        t0 = time.time()
        for s in probe:
            log.info("Preflight OK in %.1fs — dur=%.2fs spk=%s lang=%s "
                     "text_tokens=%d. Data is flowing.", time.time() - t0,
                     s["duration"], s["speaker_id"], s["lang"],
                     len(s["text_ids"]))
            return True
        return False

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _encode_batch(self, batch: TTSBatch):
        """Waveforms -> (audio_codes [B,Q,T], audio_mask [B,T]) on GPU."""
        wav = batch.wav.unsqueeze(1)  # [B, 1, Tw]
        codes = self.codec.encode(wav).to(self.device)  # [B, Q, frames]
        frames = codes.shape[-1]
        valid = (batch.wav_lengths.float() / self.samples_per_frame).floor().long()
        valid = valid.clamp(max=frames)
        ar = torch.arange(frames, device=self.device).unsqueeze(0)
        audio_mask = ar < valid.unsqueeze(1)  # [B, frames]
        return codes, audio_mask

    # ------------------------------------------------------------------ #
    def train(self) -> None:
        self.tracker.start()
        self.tracker.watch(self.model)
        self._maybe_resume()

        loader = self._build_loader()
        log.info("Starting training on %s (precision=%s) for %d steps",
                 self.device, self.amp_dtype, self.tcfg.max_steps)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum = max(1, self.tcfg.grad_accum_steps)
        micro = 0
        t0 = time.time()
        running = 0.0
        micro_since_log = 0
        last_log_step = self.step

        # Heartbeat so a `train` section appears in W&B immediately (no fake loss
        # point — real `train/loss` lands on the first optimizer step below).
        self.tracker.log({"train/heartbeat": 1, "train/step": self.step},
                         step=self.step)

        if not self._preflight_data():
            raise RuntimeError(
                "Data preflight failed — no usable samples from the stream. "
                "Run `python tools/check_data.py` to debug (check dataset access "
                "+ field mapping). Aborting before the training loop.")

        log.info("Warming up the streaming dataset — fetching the first batch "
                 "(IndicVoices streaming can take 1-3 min on first call)…")
        data_iter = iter(loader)
        first_batch = True
        warmup_t = time.time()
        while self.step < self.tcfg.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            if first_batch:
                log.info("First batch received after %.1fs (size=%d). Training "
                         "loop is live.", time.time() - warmup_t, len(batch))
                first_batch = False
            batch = batch.to(self.device)

            codes, audio_mask = self._encode_batch(batch)
            if audio_mask.sum() == 0:
                continue

            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype,
                                enabled=self.device.type == "cuda"):
                out = self.model.compute_loss(
                    text_ids=batch.text_ids, text_mask=batch.text_mask,
                    audio_codes=codes, audio_mask=audio_mask,
                    speaker_index=batch.speaker_index)
                loss = out["loss"] / accum

            self.scaler.scale(loss).backward()
            running += out["loss"].item()
            micro += 1
            micro_since_log += 1

            if micro % accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.tcfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                lr = self.scheduler.step()
                self.step += 1

                # Log every step for the first 50 (so the loss curve appears
                # within seconds of the first batch), then throttle to log_every.
                should_log = (self.step <= 50
                              or self.step % self.tcfg.log_every == 0)
                if should_log:
                    avg = running / max(1, micro_since_log)
                    running = 0.0
                    micro_since_log = 0
                    dt = time.time() - t0
                    t0 = time.time()
                    steps_done = max(1, self.step - last_log_step)
                    last_log_step = self.step
                    sps = steps_done / max(dt, 1e-6)
                    metrics = {
                        "train/loss": avg,
                        "train/acc": out["acc"].item(),
                        "train/acc_cb0": out["acc_cb0"].item(),
                        "train/ppl": float(out["ppl"]),
                        "train/lr": lr,
                        "train/steps_per_sec": sps,
                        "train/voices_seen": len(self.voice_bank),
                        "train/step": self.step,
                    }
                    self.tracker.log(metrics, step=self.step)
                    log.info("step %d | loss %.4f | acc %.3f | lr %.2e | %.2f it/s",
                             self.step, avg, out["acc"].item(), lr, sps)

                if self.step % self.tcfg.save_every == 0:
                    self._save(out_loss=out["loss"].item())

                if (self.cfg.wandb.log_audio_samples
                        and self.step % self.cfg.wandb.audio_sample_every == 0):
                    self._log_audio_sample(batch)

        self._save(out_loss=self.best_loss, final=True)
        self.voice_bank.save()
        self.tracker.finish()
        log.info("Training complete (%d steps).", self.step)

    # ------------------------------------------------------------------ #
    def _save(self, *, out_loss: float, final: bool = False) -> None:
        is_best = out_loss < self.best_loss
        self.best_loss = min(self.best_loss, out_loss)
        self.voice_bank.save()
        self.ckpt.save(step=self.step, model=self.model,
                       optimizer=self.optimizer, scheduler=self.scheduler,
                       extra={"loss": out_loss, "voices": len(self.voice_bank),
                              "tokenizer_vocab": self.tokenizer.vocab_size},
                       is_best=is_best)

    def _maybe_resume(self) -> None:
        path = self.ckpt.resolve(self.tcfg.resume_from)
        if path:
            payload = self.ckpt.load(path, model=self.model,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
                                     map_location=self.device)
            self.step = payload.get("step", 0)
            self.best_loss = payload.get("extra", {}).get("loss", float("inf"))

    @torch.no_grad()
    def _log_audio_sample(self, batch: TTSBatch) -> None:
        try:
            self.model.eval()
            text_ids = batch.text_ids[:1]
            spk = batch.speaker_index[:1]
            max_frames = self.codec.frames_for_seconds(6.0)
            codes = self.model.generate(
                text_ids=text_ids, speaker_index=spk, max_frames=max_frames,
                temperature=self.cfg.inference.temperature,
                top_k=self.cfg.inference.top_k, top_p=self.cfg.inference.top_p)
            if codes.shape[-1] > 0:
                wav = self.codec.decode(codes)[0]
                self.tracker.log_audio(
                    "samples/generated", wav, self.cfg.codec.sample_rate,
                    step=self.step, caption=batch.speaker_ids[0])
        except Exception as exc:  # pragma: no cover
            log.debug("audio sample failed: %s", exc)
        finally:
            self.model.train()
