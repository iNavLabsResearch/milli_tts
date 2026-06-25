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

import math
import time
from typing import List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
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
from milli_tts.training.distributed import barrier, setup_distributed
from milli_tts.training.hf_sync import HFCheckpointSync
from milli_tts.training.optim import CosineWarmupSchedule, build_optimizer
from milli_tts.training.tracker import WandbTracker

log = get_logger("training.trainer")


class Trainer:
    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        self.cfg = cfg or StaticMemoryCache.config()
        self.tcfg = self.cfg.training

        # ---- distributed (DDP) --------------------------------------- #
        # Idempotent: initializes the process group + pins this rank's GPU when
        # launched with >1 process (Kaggle 2×T4 self-spawns one per GPU); a
        # harmless no-op on single-GPU/CPU. Each rank uses cuda:local_rank.
        self.dist = setup_distributed()
        self.device = (self.dist.device if self.dist.is_distributed
                       else StaticMemoryCache.device(self.tcfg.device))

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

        # The forward we actually call for training. Under DDP it's the wrapped
        # module (so gradients are all-reduced across GPUs); otherwise the raw
        # model. `self.model` always stays the unwrapped module — used for
        # generate(), checkpoint state_dict (clean keys), and validation.
        self.train_model = self._wrap_ddp(self.model)

        # ---- optim / sched / ckpt / tracker -------------------------- #
        self.optimizer = build_optimizer(self.model, self.tcfg)
        self.scheduler = CosineWarmupSchedule(
            self.optimizer, warmup_steps=self.tcfg.warmup_steps,
            max_steps=self.tcfg.max_steps, base_lr=self.tcfg.lr,
            min_lr=self.tcfg.min_lr)
        self.ckpt = CheckpointManager(self.cfg.paths.checkpoint_dir,
                                      self.tcfg.keep_last_n_checkpoints)
        # Only rank 0 logs to W&B / writes checkpoints (avoids duplicate runs
        # and racing writers across GPUs).
        self.tracker = WandbTracker(self.cfg)
        if not self.dist.is_main:
            self.tracker.enabled = False

        # Continuous HF checkpoint backup (rank 0 only). Survives a Kaggle
        # disconnect; each run gets its own timestamped+indexed repo folder.
        self.hf_sync = HFCheckpointSync(
            repo=self.cfg.huggingface.checkpoint_repo,
            token=self.cfg.huggingface.token,
            enabled=(self.dist.is_main
                     and self.cfg.huggingface.push_checkpoints_to_hub))

        self.amp_dtype, self.use_scaler = self._resolve_precision()
        try:  # torch>=2.4 prefers torch.amp.GradScaler("cuda", ...)
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)
        self.step = 0
        self.best_loss = float("inf")     # best train loss (legacy)
        self.best_val = float("inf")      # best validation loss (checkpoint key)
        self._val_batches: Optional[List[TTSBatch]] = None

        if self.tcfg.compile_model:
            try:
                self.model = torch.compile(self.model)
                log.info("torch.compile enabled.")
            except Exception as exc:  # pragma: no cover
                log.warning("torch.compile failed: %s", exc)

    # ------------------------------------------------------------------ #
    def _wrap_ddp(self, model):
        """Wrap the model in DDP when running multi-GPU; else return it as-is.

        ``static_graph=True`` is used because (a) the graph is identical every
        step and (b) it cleanly tolerates the one structurally-unused parameter
        (``depth_in_emb`` of the last codebook is never an input during
        teacher-forced training) AND composes with activation checkpointing —
        avoiding the ``find_unused_parameters`` pitfalls. Falls back to
        ``find_unused_parameters=True`` if static_graph is unavailable.
        """
        if not self.dist.is_distributed:
            return model
        common = dict(device_ids=[self.dist.local_rank],
                      output_device=self.dist.local_rank,
                      gradient_as_bucket_view=True)
        try:
            ddp = DDP(model, static_graph=True, **common)
            log.info("DDP wrap OK (static_graph=True) on rank %d.", self.dist.rank)
        except Exception as exc:  # pragma: no cover - version dependent
            log.warning("static_graph DDP failed (%s); retrying with "
                        "find_unused_parameters=True.", exc)
            ddp = DDP(model, find_unused_parameters=True, **common)
        return ddp

    # ------------------------------------------------------------------ #
    def _resolve_precision(self):
        if self.device.type != "cuda":
            return torch.float32, False
        want = self.tcfg.precision.lower()
        # bf16 is only *tensor-core accelerated* on Ampere+ (sm_80+). On Turing
        # (T4 = sm_75) bf16 GEMMs fall off the tensor cores and run far slower
        # than fp16 — so we transparently switch to fp16 (+ GradScaler) there,
        # which lights up the T4's fp16 tensor cores. This is the single biggest
        # throughput win on a T4.
        cap = torch.cuda.get_device_capability(self.device)
        bf16_fast = cap[0] >= 8 and torch.cuda.is_bf16_supported()
        if want == "bf16" and bf16_fast:
            return torch.bfloat16, False
        if want == "bf16":
            log.warning("bf16 is not tensor-core accelerated on sm_%d%d (e.g. "
                        "T4) — using fp16 + GradScaler for ~1.5-2x faster GEMMs.",
                        cap[0], cap[1])
            return torch.float16, True
        if want == "fp16":
            return torch.float16, True
        return torch.float32, False

    @staticmethod
    def _worker_mp_context():
        """Start method for DataLoader workers.

        Under DDP the trainer processes are *spawned*, so the DataLoader would
        otherwise spawn its workers too — which pickles the whole dataset and
        fails on the (unpicklable) streaming state / locks it references. Forcing
        ``fork`` (available on Linux/Kaggle) lets workers inherit the dataset
        instead of pickling it, which is also the default in non-DDP runs.
        """
        import multiprocessing as _mp

        return "fork" if "fork" in _mp.get_all_start_methods() else None

    def _build_loader(self) -> DataLoader:
        # role="train" skips the first `eval_samples` stream rows (the held-out
        # validation prefix) and shards the rest across DDP ranks × workers.
        dataset = IndicVoicesDataset(tokenizer=self.tokenizer,
                                     voice_bank=self.voice_bank,
                                     register_voices=True, role="train",
                                     val_size=self.tcfg.eval_samples)
        collator = DelayedStreamCollator(text_pad_id=self.tokenizer.pad_id)
        nw = self.tcfg.num_workers
        return DataLoader(
            dataset, batch_size=self.tcfg.batch_size, collate_fn=collator,
            num_workers=nw, pin_memory=(self.device.type == "cuda"),
            drop_last=True, persistent_workers=nw > 0,
            multiprocessing_context=self._worker_mp_context() if nw > 0 else None)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _get_val_batches(self) -> List[TTSBatch]:
        """Build (once, on the main process) a FIXED held-out validation set.

        The set is the first ``eval_samples`` stream rows — disjoint from
        training, which skips them. Collated batches are cached on CPU so every
        evaluation runs on exactly the same data, making the val-loss curve
        directly comparable across steps.
        """
        if self._val_batches is not None:
            return self._val_batches
        log.info("Building fixed validation set (~%d samples)…",
                 self.tcfg.eval_samples)
        val_ds = IndicVoicesDataset(tokenizer=self.tokenizer,
                                    voice_bank=self.voice_bank,
                                    register_voices=False, role="val",
                                    val_size=self.tcfg.eval_samples)
        collator = DelayedStreamCollator(text_pad_id=self.tokenizer.pad_id)
        loader = DataLoader(val_ds, batch_size=self.tcfg.batch_size,
                            collate_fn=collator, num_workers=0, drop_last=False)
        batches, n = [], 0
        for b in loader:
            batches.append(b)
            n += len(b)
            if n >= self.tcfg.eval_samples:
                break
        self._val_batches = batches
        if not batches:
            log.warning("Validation set is EMPTY — no val metrics will be "
                        "logged. Check that the first %d stream rows pass the "
                        "duration/lang filters.", self.tcfg.eval_samples)
        else:
            log.info("Validation set ready: %d batches (%d samples).",
                     len(batches), n)
        return batches

    @torch.no_grad()
    def _validate(self) -> Optional[dict]:
        """Mean validation loss over the fixed val set (main process only).

        Runs on the raw (unwrapped) model — no gradients, no DDP collectives.
        Each batch's loss is the mean over its ``valid_frames × Q`` audio tokens;
        we re-weight by ``valid_frames`` so the aggregate is a true token-level
        mean rather than a mean-of-means biased by short batches.
        """
        if not self.dist.is_main:
            return None
        self.model.eval()
        loss_sum = acc_sum = acc0_sum = 0.0
        frame_sum = 0
        for batch in self._get_val_batches():
            batch = batch.to(self.device)
            codes, audio_mask = self._encode_batch(batch)
            n_frames = int(audio_mask.sum().item())
            if n_frames == 0:
                continue
            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype,
                                enabled=self.device.type == "cuda"):
                out = self.model.compute_loss(
                    text_ids=batch.text_ids, text_mask=batch.text_mask,
                    audio_codes=codes, audio_mask=audio_mask,
                    speaker_index=batch.speaker_index)
            loss_sum += out["loss"].item() * n_frames
            acc_sum += out["acc"].item() * n_frames
            acc0_sum += out["acc_cb0"].item() * n_frames
            frame_sum += n_frames
        self.model.train()
        if frame_sum == 0:
            return None
        val_loss = loss_sum / frame_sum
        return {
            "val/loss": val_loss,
            "val/acc": acc_sum / frame_sum,
            "val/acc_cb0": acc0_sum / frame_sum,
            "val/ppl": math.exp(min(20.0, val_loss)),
        }

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

        self.train_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum = max(1, self.tcfg.grad_accum_steps)
        micro = 0
        t0 = time.time()
        running = running_acc = running_acc0 = 0.0
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
            # Register speakers in the MAIN process (DataLoader workers register
            # in their own forked copy, which never propagates — that's why
            # train/voices_seen was stuck at 0). Indices are deterministic, so
            # this agrees with the per-sample index already inside the batch.
            if self.dist.is_main:
                for sid, lang in zip(batch.speaker_ids, batch.langs):
                    self.voice_bank.add_or_get(sid, lang=lang or None)

            codes, audio_mask = self._encode_batch(batch)
            if audio_mask.sum() == 0:
                continue

            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype,
                                enabled=self.device.type == "cuda"):
                # Route through `train_model` (the DDP wrapper under multi-GPU)
                # so its forward sets up cross-rank gradient averaging.
                out = self.train_model(
                    text_ids=batch.text_ids, text_mask=batch.text_mask,
                    audio_codes=codes, audio_mask=audio_mask,
                    speaker_index=batch.speaker_index,
                    label_smoothing=self.tcfg.label_smoothing)
                loss = out["loss"] / accum

            self.scaler.scale(loss).backward()
            running += out["loss"].item()
            running_acc += out["acc"].item()
            running_acc0 += out["acc_cb0"].item()
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
                    n = max(1, micro_since_log)
                    # Average ALL scalars over the log window (not just loss) so
                    # the curves reflect a real running mean instead of a single
                    # noisy micro-batch — this is what makes acc/acc_cb0 readable.
                    avg = running / n
                    avg_acc = running_acc / n
                    avg_acc0 = running_acc0 / n
                    running = running_acc = running_acc0 = 0.0
                    micro_since_log = 0
                    dt = time.time() - t0
                    t0 = time.time()
                    steps_done = max(1, self.step - last_log_step)
                    last_log_step = self.step
                    sps = steps_done / max(dt, 1e-6)
                    # `avg` is this rank's local loss; under DDP gradients are
                    # synced but losses aren't, so we log rank 0's estimate.
                    if self.dist.is_main:
                        metrics = {
                            "train/loss": avg,
                            "train/acc": avg_acc,
                            "train/acc_cb0": avg_acc0,
                            "train/ppl": math.exp(min(20.0, avg)),
                            "train/lr": lr,
                            "train/steps_per_sec": sps * self.dist.world_size,
                            "train/voices_seen": len(self.voice_bank),
                            "train/step": self.step,
                        }
                        self.tracker.log(metrics, step=self.step)
                        log.info("step %d | loss %.4f | acc %.3f | acc_cb0 %.3f "
                                 "| lr %.2e | %.2f it/s/gpu", self.step, avg,
                                 avg_acc, avg_acc0,
                                 lr, sps)

                # ---- periodic validation (all ranks reach the barrier) ---- #
                saved_this_step = False
                if self.step % self.tcfg.eval_every == 0:
                    try:
                        val = self._validate()  # metrics on rank 0, else None
                    except Exception as exc:  # never let val crash training
                        log.warning("validation failed at step %d: %s",
                                    self.step, exc)
                        val = None
                    if val is not None:
                        val["train/step"] = self.step
                        self.tracker.log(val, step=self.step)
                        log.info("VAL step %d | val_loss %.4f | val_acc %.3f | "
                                 "val_acc_cb0 %.3f | val_ppl %.2f", self.step,
                                 val["val/loss"], val["val/acc"],
                                 val["val/acc_cb0"], val["val/ppl"])
                        self._save(val_loss=val["val/loss"])
                        saved_this_step = True
                    barrier()

                # Periodic checkpoint (skip if validation just saved this step).
                if self.step % self.tcfg.save_every == 0 and not saved_this_step:
                    self._save()

                if (self.cfg.wandb.log_audio_samples and self.dist.is_main
                        and self.step % self.cfg.wandb.audio_sample_every == 0):
                    self._log_audio_sample(batch)

        self._save(final=True)
        if self.dist.is_main:
            self.voice_bank.save()
        self.tracker.finish()
        barrier()
        log.info("Training complete (%d steps).", self.step)

    # ------------------------------------------------------------------ #
    def _save(self, *, val_loss: Optional[float] = None,
              final: bool = False) -> None:
        """Checkpoint (rank 0 only). ``best.pt`` tracks lowest validation loss.

        After writing locally, the rolling ``latest.pt`` is mirrored to the HF
        repo (in the background) so a disconnect never loses progress; the final
        save is uploaded synchronously as ``final.pt``.
        """
        if not self.dist.is_main:
            return
        is_best = val_loss is not None and val_loss < self.best_val
        if val_loss is not None:
            self.best_val = min(self.best_val, val_loss)
        self.voice_bank.save()
        self.ckpt.save(step=self.step, model=self.model,
                       optimizer=self.optimizer, scheduler=self.scheduler,
                       extra={"val_loss": (val_loss if val_loss is not None
                                           else self.best_val),
                              "voices": len(self.voice_bank),
                              "tokenizer_vocab": self.tokenizer.vocab_size},
                       is_best=is_best)
        if final:
            self.hf_sync.push_final(self.ckpt.latest_path)
        else:
            self.hf_sync.push_latest(self.ckpt.latest_path)

    def _maybe_resume(self) -> None:
        # Every rank loads the same checkpoint into its (identical) raw module,
        # so all ranks resume from the exact same weights — no broadcast needed.
        path = self.ckpt.resolve(self.tcfg.resume_from)
        if path:
            payload = self.ckpt.load(path, model=self.model,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
                                     map_location=self.device)
            self.step = payload.get("step", 0)
            self.best_val = payload.get("extra", {}).get("val_loss", float("inf"))

    @torch.no_grad()
    def _log_audio_sample(self, batch: TTSBatch) -> None:
        try:
            self.model.eval()  # generate() runs on the raw (unwrapped) module
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
            self.train_model.train()  # restore train mode on the wrapper+module
