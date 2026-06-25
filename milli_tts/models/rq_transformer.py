"""RQ-Transformer TTS (the milli_tts model).

Architecture (decoder-only, Moshi/Mimi lineage — no separate text encoder):

    text ids ─► text embedding ─┐
                                ├─►  Backbone (temporal) Transformer  ─► h_t
    prev audio frame codes ─────┘            (causal over the
    + speaker conditioning                    text-prefix + audio frames)
                                                      │
                                                      ▼
                                          Depth Transformer (tiny, causal
                                          over the Q codebooks of frame t)
                                                      │
                                                      ▼
                                       Q codebook logits  ─► Mimi codes ─► wav

Why this shape:
* **One backbone** consumes a *prefix of text tokens* followed by *audio
  frames*. Causal attention lets every audio frame attend to all text — so the
  text acts as the conditioning, no cross-attention / no T5 encoder, no
  dimension-mismatch worries (everything is internal at ``d_model``).
* The **depth transformer** models the within-frame dependency across the Q
  residual codebooks (the RQ-Transformer trick from Moshi). It removes the need
  for a MusicGen-style codebook delay pattern.
* **Speaker identity** enters as an additive conditioning vector at every audio
  position (see :mod:`conditioning`), which is exactly the ``voice_id`` hook.

Training is fully teacher-forced and parallel over both frames (backbone) and
codebooks (depth). Inference is autoregressive: prefill the text once, then for
each frame run one backbone step + Q tiny depth steps.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from milli_tts.core.config import ModelConfig, VoiceConfig
from milli_tts.models.base import BaseTTSModel
from milli_tts.models.conditioning import build_conditioner
from milli_tts.models.layers import KVCache, TransformerStack
from milli_tts.models.sampling import sample_logits


class RQTransformerTTS(BaseTTSModel):
    def __init__(self, *, model_cfg: ModelConfig, voice_cfg: VoiceConfig,
                 text_vocab_size: int, num_codebooks: int,
                 codebook_size: int) -> None:
        super().__init__()
        self.cfg = model_cfg
        self.d_model = model_cfg.d_model
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        # ---- audio token cardinality: real codes + PAD + BOS ---------- #
        self.audio_pad = codebook_size          # padding / "no audio"
        self.audio_bos = codebook_size + 1      # start-of-audio
        self.audio_card = codebook_size + 2

        # ---- text side ------------------------------------------------ #
        self.text_pad = text_vocab_size         # extra pad id appended
        self.text_card = text_vocab_size + 1
        self.text_emb = nn.Embedding(self.text_card, self.d_model)

        # ---- per-codebook input embeddings for the backbone ----------- #
        self.audio_in_emb = nn.ModuleList(
            [nn.Embedding(self.audio_card, self.d_model)
             for _ in range(num_codebooks)])

        # modality (text vs audio) + speaker conditioning
        self.type_emb = nn.Embedding(2, self.d_model)
        self.conditioner = build_conditioner(
            voice_cfg.strategy, d_model=self.d_model,
            max_speakers=voice_cfg.max_speakers,
            embedding_dim=voice_cfg.embedding_dim)

        # ---- backbone (temporal) transformer -------------------------- #
        self.backbone = TransformerStack(
            dim=self.d_model, depth=model_cfg.backbone_layers,
            heads=model_cfg.backbone_heads, ffn_mult=model_cfg.ffn_mult,
            rope_theta=model_cfg.rope_theta, dropout=model_cfg.dropout,
            grad_checkpoint=model_cfg.__dict__.get("gradient_checkpointing", False))

        # ---- depth transformer (over codebooks) ----------------------- #
        self.depth_dim = model_cfg.depth_dim
        self.h_to_depth = (nn.Identity() if self.depth_dim == self.d_model
                           else nn.Linear(self.d_model, self.depth_dim, bias=False))
        self.depth_in_emb = nn.ModuleList(
            [nn.Embedding(self.audio_card, self.depth_dim)
             for _ in range(num_codebooks)])
        self.depth_pos_emb = nn.Parameter(
            torch.zeros(num_codebooks, self.depth_dim))
        self.depth = TransformerStack(
            dim=self.depth_dim, depth=model_cfg.depth_layers,
            heads=model_cfg.depth_heads, ffn_mult=model_cfg.ffn_mult,
            rope_theta=model_cfg.rope_theta, dropout=model_cfg.dropout)
        self.depth_heads = nn.ModuleList(
            [nn.Linear(self.depth_dim, codebook_size, bias=False)
             for _ in range(num_codebooks)])

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------ #
    # Embedding helpers
    # ------------------------------------------------------------------ #
    def _embed_audio_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum the Q per-codebook embeddings of a frame. codes: [B, Q] -> [B, D]."""
        out = 0
        for q in range(self.num_codebooks):
            out = out + self.audio_in_emb[q](codes[:, q])
        return out

    def _backbone_inputs(self, text_ids: torch.Tensor,
                         audio_in: torch.Tensor,
                         speaker_index: torch.Tensor) -> torch.Tensor:
        """Build the full [B, Lt+Ta, D] backbone input sequence.

        ``audio_in`` are the teacher-forcing *input* codes per frame
        ``[B, Q, Ta]`` (already shifted so frame t sees frame t-1; the first
        frame is BOS).
        """
        b, lt = text_ids.shape
        spk = self.conditioner(speaker_index)  # [B, D]

        text_h = self.text_emb(text_ids) + self.type_emb.weight[0]  # [B, Lt, D]

        # Vectorized over time: sum the Q per-codebook embeddings for ALL frames
        # at once (Q embedding lookups total, not Q*Ta). audio_in: [B, Q, Ta].
        audio_h = self._embed_audio_seq(audio_in)  # [B, Ta, D]
        audio_h = audio_h + self.type_emb.weight[1] + spk.unsqueeze(1)

        return torch.cat([text_h, audio_h], dim=1)  # [B, Lt+Ta, D]

    def _embed_audio_seq(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum per-codebook embeddings over a whole frame sequence.

        codes: ``[B, Q, Ta]`` -> ``[B, Ta, D]``.
        """
        out = self.audio_in_emb[0](codes[:, 0, :])
        for q in range(1, self.num_codebooks):
            out = out + self.audio_in_emb[q](codes[:, q, :])
        return out

    # ------------------------------------------------------------------ #
    # Depth transformer: predict Q codebooks from backbone hidden states
    # ------------------------------------------------------------------ #
    def _depth_forward_train(self, h: torch.Tensor,
                             target_codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced depth pass.

        Args:
            h: backbone hidden states at audio positions ``[N, D]`` (N = B*Ta).
            target_codes: ground-truth codes ``[N, Q]``.
        Returns:
            logits ``[N, Q, codebook_size]``.
        """
        n = h.shape[0]
        h_d = self.h_to_depth(h)  # [N, depth_dim]

        # Build the depth input sequence of length Q:
        #   step 0 input = h_d (+ pos0)
        #   step q input = h_d + emb(code_{q-1}) (+ pos q)
        steps = [h_d + self.depth_pos_emb[0]]
        for q in range(1, self.num_codebooks):
            prev = self.depth_in_emb[q - 1](target_codes[:, q - 1])
            steps.append(h_d + prev + self.depth_pos_emb[q])
        x = torch.stack(steps, dim=1)  # [N, Q, depth_dim]

        x, _ = self.depth(x, causal=True)  # [N, Q, depth_dim]
        logits = torch.stack(
            [self.depth_heads[q](x[:, q]) for q in range(self.num_codebooks)],
            dim=1)  # [N, Q, cb]
        return logits

    @torch.no_grad()
    def _depth_generate(self, h: torch.Tensor, *, temperature: float,
                        top_k: int, top_p: float) -> torch.Tensor:
        """Autoregressively sample Q codebooks for one frame. h: [B, D] -> [B, Q]."""
        b = h.shape[0]
        h_d = self.h_to_depth(h)
        caches: Optional[List[KVCache]] = None
        out_codes = torch.zeros(b, self.num_codebooks, dtype=torch.long,
                                device=h.device)
        prev_emb = torch.zeros(b, self.depth_dim, device=h.device, dtype=h_d.dtype)
        for q in range(self.num_codebooks):
            inp = (h_d + prev_emb + self.depth_pos_emb[q]).unsqueeze(1)
            x, caches = self.depth(inp, causal=True, caches=caches, pos_offset=q)
            logits = self.depth_heads[q](x[:, -1])  # [B, cb]
            code = sample_logits(logits, temperature=temperature,
                                 top_k=top_k, top_p=top_p)  # [B]
            out_codes[:, q] = code
            prev_emb = self.depth_in_emb[q](code)
        return out_codes

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def compute_loss(self, *, text_ids: torch.Tensor, text_mask: torch.Tensor,
                     audio_codes: torch.Tensor, audio_mask: torch.Tensor,
                     speaker_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """audio_codes: [B, Q, Ta] (Mimi codes). audio_mask: [B, Ta] bool."""
        b, q, ta = audio_codes.shape
        device = audio_codes.device

        # teacher-forcing inputs: BOS frame, then frames 0..Ta-2
        bos = torch.full((b, q, 1), self.audio_bos, dtype=torch.long, device=device)
        audio_in = torch.cat([bos, audio_codes[:, :, :-1]], dim=2)  # [B, Q, Ta]
        # pad masked positions in the *input* with audio_pad
        pad_in = (~audio_mask).unsqueeze(1)  # [B, 1, Ta]
        audio_in = audio_in.masked_fill(pad_in, self.audio_pad)

        seq = self._backbone_inputs(text_ids, audio_in, speaker_index)
        h, _ = self.backbone(seq, causal=True)  # [B, Lt+Ta, D]

        lt = text_ids.shape[1]
        h_audio = h[:, lt:, :]  # [B, Ta, D] hidden used to predict frame t

        # depth predicts the *current* frame's codes from h_audio[t]
        h_flat = h_audio.reshape(b * ta, self.d_model)
        tgt_flat = audio_codes.permute(0, 2, 1).reshape(b * ta, q)  # [B*Ta, Q]
        logits = self._depth_forward_train(h_flat, tgt_flat)  # [B*Ta, Q, cb]

        # loss
        mask_flat = audio_mask.reshape(b * ta)  # [B*Ta]
        logits = logits[mask_flat]              # [M, Q, cb]
        targets = tgt_flat[mask_flat]           # [M, Q]
        loss = F.cross_entropy(
            logits.reshape(-1, self.codebook_size), targets.reshape(-1))

        with torch.no_grad():
            pred = logits.argmax(-1)
            acc = (pred == targets).float().mean()
            # per-codebook accuracy for the first codebook (semantic-ish)
            acc0 = (pred[:, 0] == targets[:, 0]).float().mean()

        return {"loss": loss, "acc": acc, "acc_cb0": acc0,
                "ppl": loss.exp().detach()}

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, *, text_ids: torch.Tensor, speaker_index: torch.Tensor,
                 max_frames: int, temperature: float = 0.7, top_k: int = 250,
                 top_p: float = 0.95, prefix_codes: Optional[torch.Tensor] = None,
                 eos_silence_frames: int = 8) -> torch.Tensor:
        """Autoregressively generate Mimi codes ``[B, Q, frames]``.

        Prefills the text (and optional reference ``prefix_codes`` for zero-shot
        cloning) once, then samples frame-by-frame with a KV cache.
        """
        self.eval()
        b = text_ids.shape[0]
        device = text_ids.device
        q = self.num_codebooks
        spk = self.conditioner(speaker_index)  # [B, D]

        # ---- prefill text prefix into backbone KV cache --------------- #
        text_h = self.text_emb(text_ids) + self.type_emb.weight[0]
        caches: List[KVCache] = [KVCache(k=torch.empty(0), v=torch.empty(0))
                                 for _ in self.backbone.blocks]
        h, caches = self.backbone(text_h, causal=True, caches=caches, pos_offset=0)
        pos = text_ids.shape[1]

        generated: List[torch.Tensor] = []

        # optional reference-clip prefix (zero-shot cloning): teacher-feed it
        if prefix_codes is not None:
            pf = prefix_codes.to(device)
            if pf.dim() == 2:
                pf = pf.unsqueeze(0).expand(b, -1, -1)
            cur = torch.full((b, q), self.audio_bos, dtype=torch.long, device=device)
            for t in range(pf.shape[-1]):
                emb = (self._embed_audio_frame(cur) + self.type_emb.weight[1]
                       + spk).unsqueeze(1)
                h, caches = self.backbone(emb, causal=True, caches=caches,
                                          pos_offset=pos)
                pos += 1
                cur = pf[:, :, t]
            last_in = cur
        else:
            last_in = torch.full((b, q), self.audio_bos, dtype=torch.long,
                                 device=device)

        # ---- autoregressive frame loop -------------------------------- #
        silence_run = 0
        for _ in range(max_frames):
            emb = (self._embed_audio_frame(last_in) + self.type_emb.weight[1]
                   + spk).unsqueeze(1)
            h, caches = self.backbone(emb, causal=True, caches=caches,
                                      pos_offset=pos)
            pos += 1
            frame = self._depth_generate(h[:, -1], temperature=temperature,
                                         top_k=top_k, top_p=top_p)  # [B, Q]
            generated.append(frame)
            last_in = frame

            # crude EOS: long run of near-zero codebook-0 => stop
            if (frame[:, 0] == 0).all():
                silence_run += 1
                if silence_run >= eos_silence_frames:
                    break
            else:
                silence_run = 0

        if not generated:
            return torch.zeros(b, q, 0, dtype=torch.long, device=device)
        return torch.stack(generated, dim=2)  # [B, Q, frames]
