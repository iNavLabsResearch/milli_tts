"""RQ-Transformer TTS — modular rewrite.

Same architecture as rq_transformer.py but split into four self-contained
modules so each piece can be tested, swapped, or scaled independently:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  TextEmbedder          text_ids → [B, Lt, D]                       │
    │  AudioEmbedder         codes   → [B, Ta, D]  (sum Q embeddings)    │
    │  DepthHead             h[B,D]  → logits [B,Q,cb]  (depth Tx)       │
    │  RQTransformerTTS      top-level: backbone + depth + loss/generate  │
    └─────────────────────────────────────────────────────────────────────┘

Why the split matters:
* TextEmbedder is swappable — drop in a frozen LLM encoder later without
  touching the backbone.
* AudioEmbedder is testable in isolation (shape / embedding-sum correctness).
* DepthHead encapsulates the entire within-frame codebook AR pass, including
  the inference KV-cache loop — no logic leaks into the top-level model.
* The top-level model becomes a thin orchestrator: embed → backbone → depth
  → loss, which is easy to read and audit.

acc_cb0 note (why it lags):
  Codebook-0 in Mimi is a semantically-distilled token (WavLM-style), so
  it captures WHAT is being said, not just fine acoustic texture.  Predicting
  it from text is structurally similar to language modelling — very high
  entropy, random = 0.05 %.  Typical training trajectory:
    steps  1k → acc_cb0 ≈ 2–5 %
    steps  5k → acc_cb0 ≈ 8–15 %
    steps 20k → acc_cb0 ≈ 20–35 %
  cb1-7 rise faster because they predict fine residual corrections.
  Low early acc_cb0 is NORMAL.  If it stays at <1 % past 5k steps, check:
    1. speaker_id_source="row" with huge speaker count → switch to "gender"
    2. Training actually ran (check for the macOS DataLoader crash below)
    3. Learning rate — if loss is still at ln(2048)=7.6, warmup may be too slow
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from milli_tts.core.config import ModelConfig, VoiceConfig
from milli_tts.models.base import BaseTTSModel
from milli_tts.models.conditioning import build_conditioner
from milli_tts.models.layers import KVCache, TransformerStack
from milli_tts.models.sampling import sample_logits


# ══════════════════════════════════════════════════════════════════════════════
# Module 1 — TextEmbedder
# ══════════════════════════════════════════════════════════════════════════════

class TextEmbedder(nn.Module):
    """text_ids [B, Lt] → embedded sequence [B, Lt, D].

    Adds a learned modality-type bias so the backbone can distinguish text
    positions from audio positions in the unified sequence.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        # +1 for the PAD token appended beyond the real vocab
        self.pad_id = vocab_size
        self.card = vocab_size + 1
        self.emb = nn.Embedding(self.card, d_model)
        self.type_bias = nn.Parameter(torch.zeros(d_model))   # type_emb[0]
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """text_ids: [B, Lt] long → [B, Lt, D]."""
        return self.emb(text_ids) + self.type_bias


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 — AudioEmbedder
# ══════════════════════════════════════════════════════════════════════════════

class AudioEmbedder(nn.Module):
    """Encode a sequence of Mimi frames into backbone-ready embeddings.

    Each frame has Q residual codes. The per-codebook embeddings are SUMMED
    (not concatenated) so the output is always [B, T, D] regardless of Q.
    This is the same trick used in Moshi/EnCodec — it keeps the backbone
    dimensionality fixed while letting each codebook vote independently.

    Special tokens
    --------------
    audio_pad  (= codebook_size)     — padding, never attends
    audio_bos  (= codebook_size + 1) — start-of-audio / BOS frame

    Args
    ----
    num_codebooks : Q
    codebook_size : vocabulary per codebook (2048 for Mimi)
    d_model       : embedding + output dim
    """

    def __init__(self, num_codebooks: int, codebook_size: int,
                 d_model: int) -> None:
        super().__init__()
        self.Q = num_codebooks
        self.codebook_size = codebook_size
        self.audio_pad = codebook_size
        self.audio_bos = codebook_size + 1
        self.audio_card = codebook_size + 2   # pad + bos

        # One embedding table per codebook — each has its own vocabulary so
        # the same code index means different things in different codebooks.
        self.embs = nn.ModuleList(
            [nn.Embedding(self.audio_card, d_model)
             for _ in range(num_codebooks)])
        self.type_bias = nn.Parameter(torch.zeros(d_model))  # type_emb[1]
        for e in self.embs:
            nn.init.normal_(e.weight, std=0.02)

    def embed_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """Single-frame embedding.  codes: [B, Q] → [B, D]."""
        out = self.embs[0](codes[:, 0])
        for q in range(1, self.Q):
            out = out + self.embs[q](codes[:, q])
        return out

    def embed_sequence(self, codes: torch.Tensor) -> torch.Tensor:
        """Whole-sequence embedding.  codes: [B, Q, T] → [B, T, D]."""
        # Vectorised: Q lookups over the full time axis, then sum.
        out = self.embs[0](codes[:, 0, :])          # [B, T, D]
        for q in range(1, self.Q):
            out = out + self.embs[q](codes[:, q, :])
        return out

    def teacher_force_input(self, audio_codes: torch.Tensor,
                            audio_mask: torch.Tensor,
                            speaker_vec: torch.Tensor) -> torch.Tensor:
        """Build the teacher-forced audio input sequence [B, Ta, D].

        Shifts codes right by one frame (prepend BOS) so frame t sees the
        TARGET codes of frame t-1.  Masked (padding) positions are filled
        with audio_pad so they don't leak real content.

        Args
        ----
        audio_codes  : [B, Q, Ta] target codes from Mimi.encode()
        audio_mask   : [B, Ta]    True = real frame
        speaker_vec  : [B, D]     additive speaker conditioning
        """
        b, q, ta = audio_codes.shape
        device = audio_codes.device
        bos = torch.full((b, q, 1), self.audio_bos, dtype=torch.long,
                         device=device)
        # shift right: [BOS | frame0 | frame1 | … | frame Ta-2]
        audio_in = torch.cat([bos, audio_codes[:, :, :-1]], dim=2)
        # mask padded positions in the input
        pad_mask = (~audio_mask).unsqueeze(1)       # [B, 1, Ta]
        audio_in = audio_in.masked_fill(pad_mask, self.audio_pad)

        h = self.embed_sequence(audio_in)           # [B, Ta, D]
        h = h + self.type_bias + speaker_vec.unsqueeze(1)
        return h


# ══════════════════════════════════════════════════════════════════════════════
# Module 3 — DepthHead
# ══════════════════════════════════════════════════════════════════════════════

class DepthHead(nn.Module):
    """Per-frame codebook prediction (the "depth" Transformer).

    Takes the backbone hidden state at a frame position and autoregressively
    predicts Q codebook tokens in order cb0 → cb1 → … → cb(Q-1).

    This is the RQ-Transformer's core trick: instead of a MusicGen-style
    codebook delay pattern (which complicates the backbone), the within-frame
    AR dependency lives in a tiny, separate Transformer over Q positions.

    Training  : teacher-forced, fully parallel over frames AND codebooks.
    Inference : Q sequential steps per frame (fast because Q=8 is tiny).

    Inputs
    ------
    h_backbone : [N, D]   backbone hidden state (N = B*Ta for train, B for infer)
    codes      : [N, Q]   ground-truth codes (train), or sampled codes (infer)

    Outputs
    -------
    logits : [N, Q, codebook_size]
    """

    def __init__(self, backbone_dim: int, depth_dim: int, depth_layers: int,
                 depth_heads: int, ffn_mult: int, rope_theta: float,
                 num_codebooks: int, codebook_size: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.Q = num_codebooks
        self.codebook_size = codebook_size
        self.depth_dim = depth_dim

        # Project backbone hidden dim → depth dim (Identity if equal).
        self.proj = (nn.Identity() if backbone_dim == depth_dim
                     else nn.Linear(backbone_dim, depth_dim, bias=False))

        # Per-codebook input embeddings (depth side, separate from backbone's).
        self.in_embs = nn.ModuleList(
            [nn.Embedding(codebook_size + 2, depth_dim)
             for _ in range(num_codebooks)])

        # Learnable positional bias for each codebook step (Q positions).
        self.pos_emb = nn.Parameter(torch.zeros(num_codebooks, depth_dim))

        # Tiny Transformer (same block type as backbone).
        self.transformer = TransformerStack(
            dim=depth_dim, depth=depth_layers, heads=depth_heads,
            ffn_mult=ffn_mult, rope_theta=rope_theta, dropout=dropout)

        # Per-codebook linear heads → logits over codebook_size.
        self.heads = nn.ModuleList(
            [nn.Linear(depth_dim, codebook_size, bias=False)
             for _ in range(num_codebooks)])

        for e in self.in_embs:
            nn.init.normal_(e.weight, std=0.02)

    # ------------------------------------------------------------------
    def forward_train(self, h: torch.Tensor,
                      target_codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced pass over all Q codebooks.

        Args
        ----
        h            : [N, backbone_D]  backbone hidden state (N = B*Ta)
        target_codes : [N, Q]           ground-truth codes

        Returns
        -------
        logits : [N, Q, codebook_size]
        """
        h_d = self.proj(h)                      # [N, depth_D]

        # Build the Q-step depth input sequence:
        #   step 0 : h_d + pos[0]                 (no prior code)
        #   step q : h_d + emb(code_{q-1}) + pos[q]
        steps: List[torch.Tensor] = [h_d + self.pos_emb[0]]
        for q in range(1, self.Q):
            prev = self.in_embs[q - 1](target_codes[:, q - 1])
            steps.append(h_d + prev + self.pos_emb[q])

        x = torch.stack(steps, dim=1)           # [N, Q, depth_D]
        x, _ = self.transformer(x, causal=True)

        logits = torch.stack(
            [self.heads[q](x[:, q]) for q in range(self.Q)], dim=1)
        return logits                           # [N, Q, cb]

    @torch.no_grad()
    def forward_generate(self, h: torch.Tensor, *,
                         temperature: float, top_k: int,
                         top_p: float) -> torch.Tensor:
        """Autoregressive sampling of Q codebooks for one frame.

        Args
        ----
        h : [B, backbone_D]  backbone hidden at the current frame

        Returns
        -------
        codes : [B, Q]  sampled codebook indices
        """
        b = h.shape[0]
        h_d = self.proj(h)                      # [B, depth_D]
        caches: Optional[List[KVCache]] = None
        codes = torch.zeros(b, self.Q, dtype=torch.long, device=h.device)
        prev_emb = torch.zeros(b, self.depth_dim, device=h.device,
                               dtype=h_d.dtype)
        for q in range(self.Q):
            inp = (h_d + prev_emb + self.pos_emb[q]).unsqueeze(1)  # [B,1,D]
            x, caches = self.transformer(inp, causal=True,
                                         caches=caches, pos_offset=q)
            logit = self.heads[q](x[:, -1])     # [B, cb]
            code = sample_logits(logit, temperature=temperature,
                                 top_k=top_k, top_p=top_p)
            codes[:, q] = code
            prev_emb = self.in_embs[q](code)
        return codes                            # [B, Q]


# ══════════════════════════════════════════════════════════════════════════════
# Top-level model
# ══════════════════════════════════════════════════════════════════════════════

class RQTransformerTTSv2(BaseTTSModel):
    """Modular RQ-Transformer TTS.

    Orchestrates TextEmbedder → AudioEmbedder → Backbone → DepthHead.
    All heavy logic lives in the sub-modules above; this class handles:
      * weight initialisation & residual scaling
      * building the joint [text | audio] sequence for the backbone
      * compute_loss (training) and generate (inference)
    """

    def __init__(self, *, model_cfg: ModelConfig, voice_cfg: VoiceConfig,
                 text_vocab_size: int, num_codebooks: int,
                 codebook_size: int) -> None:
        super().__init__()
        self.cfg = model_cfg
        D = model_cfg.d_model
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        # ── sub-modules ──────────────────────────────────────────────── #
        self.text_embedder = TextEmbedder(text_vocab_size, D)
        self.audio_embedder = AudioEmbedder(num_codebooks, codebook_size, D)
        self.conditioner = build_conditioner(
            voice_cfg.strategy, d_model=D,
            max_speakers=voice_cfg.max_speakers,
            embedding_dim=voice_cfg.embedding_dim)

        self.backbone = TransformerStack(
            dim=D, depth=model_cfg.backbone_layers,
            heads=model_cfg.backbone_heads, ffn_mult=model_cfg.ffn_mult,
            rope_theta=model_cfg.rope_theta, dropout=model_cfg.dropout,
            grad_checkpoint=model_cfg.__dict__.get("gradient_checkpointing",
                                                    False))

        self.depth_head = DepthHead(
            backbone_dim=D,
            depth_dim=model_cfg.depth_dim,
            depth_layers=model_cfg.depth_layers,
            depth_heads=model_cfg.depth_heads,
            ffn_mult=model_cfg.ffn_mult,
            rope_theta=model_cfg.rope_theta,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            dropout=model_cfg.dropout)

        # Per-codebook CE weights (None = uniform). Renormalized to mean 1 so the
        # loss scale stays comparable to a uniform run — see ModelConfig docs.
        self.cb_loss_w = self._build_codebook_weights(
            getattr(model_cfg, "codebook_loss_weights", None), num_codebooks)

        # ── init ─────────────────────────────────────────────────────── #
        self.apply(self._init_weights)
        self._scale_residual_init(self.backbone, model_cfg.backbone_layers)
        self._scale_residual_init(self.depth_head.transformer,
                                  model_cfg.depth_layers)

    # ── weight init ──────────────────────────────────────────────────── #

    def _build_codebook_weights(self, weights, num_codebooks: int):
        """Normalize per-codebook loss weights to mean 1 (None = uniform)."""
        if not weights:
            return None
        if len(weights) != num_codebooks:
            raise ValueError(
                f"codebook_loss_weights has {len(weights)} entries but the codec "
                f"has {num_codebooks} codebooks — they must match.")
        w = torch.tensor(weights, dtype=torch.float32)
        if (w <= 0).any():
            raise ValueError("codebook_loss_weights must all be > 0.")
        w = w * (num_codebooks / w.sum())
        self.register_buffer("_cb_loss_w", w, persistent=False)
        return self._cb_loss_w

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _scale_residual_init(stack: TransformerStack, n_layers: int) -> None:
        """GPT-2 / nanoGPT residual projection down-scaling."""
        scale = (2.0 * max(1, n_layers)) ** -0.5
        for block in stack.blocks:
            with torch.no_grad():
                block.attn.proj.weight.mul_(scale)
                block.mlp.w3.weight.mul_(scale)

    # ── helpers ──────────────────────────────────────────────────────── #

    def _build_backbone_seq(self, text_ids: torch.Tensor,
                             audio_codes: torch.Tensor,
                             audio_mask: torch.Tensor,
                             speaker_index: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the full backbone input + key-padding mask.

        Returns
        -------
        seq  : [B, Lt+Ta, D]
        kpm  : [B, Lt+Ta] bool — True = real token (False = pad, never attend)
        """
        text_mask = (text_ids != self.text_embedder.pad_id)  # [B, Lt]
        speaker_vec = self.conditioner(speaker_index)         # [B, D]

        text_h = self.text_embedder(text_ids)                 # [B, Lt, D]
        audio_h = self.audio_embedder.teacher_force_input(
            audio_codes, audio_mask, speaker_vec)             # [B, Ta, D]

        seq = torch.cat([text_h, audio_h], dim=1)             # [B, Lt+Ta, D]
        kpm = torch.cat([text_mask, audio_mask.bool()], dim=1)
        return seq, kpm

    # ── training ─────────────────────────────────────────────────────── #

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """DDP entry-point. Routes to compute_loss."""
        return self.compute_loss(**kwargs)

    def compute_loss(self, *, text_ids: torch.Tensor, text_mask: torch.Tensor,
                     audio_codes: torch.Tensor, audio_mask: torch.Tensor,
                     speaker_index: torch.Tensor,
                     label_smoothing: float = 0.0) -> Dict[str, torch.Tensor]:
        """Teacher-forced loss over all codebooks and all frames.

        audio_codes : [B, Q, Ta]  Mimi codes (ground truth / target)
        audio_mask  : [B, Ta]     True = real frame

        Returns dict with: loss, acc (mean over Q), acc_cb0 (cb0 only), ppl.
        """
        b, q, ta = audio_codes.shape

        # 1. Build backbone input sequence
        seq, kpm = self._build_backbone_seq(
            text_ids, audio_codes, audio_mask, speaker_index)

        # 2. Backbone forward
        h, _ = self.backbone(seq, causal=True, key_padding_mask=kpm)

        # 3. Slice audio hidden states (backbone positions Lt : Lt+Ta)
        lt = text_ids.shape[1]
        h_audio = h[:, lt:, :]                   # [B, Ta, D]

        # 4. Depth head: flatten batch×time, run depth, unflatten
        h_flat = h_audio.reshape(b * ta, -1)                  # [N, D]
        tgt_flat = audio_codes.permute(0, 2, 1).reshape(b * ta, q)  # [N, Q]
        logits = self.depth_head.forward_train(h_flat, tgt_flat)     # [N, Q, cb]

        # 5. Mask out padding frames before loss
        mask = audio_mask.reshape(b * ta)         # [N]
        logits_m = logits[mask]                   # [M, Q, cb]
        targets_m = tgt_flat[mask]                # [M, Q]

        if self.cb_loss_w is None:
            loss = F.cross_entropy(
                logits_m.reshape(-1, self.codebook_size),
                targets_m.reshape(-1),
                label_smoothing=label_smoothing)
        else:
            # Per-token CE → [M, Q], weighted per codebook (mean-1 weights keep
            # the loss scale comparable while steering capacity toward cb0).
            ce = F.cross_entropy(
                logits_m.reshape(-1, self.codebook_size),
                targets_m.reshape(-1),
                label_smoothing=label_smoothing,
                reduction="none").reshape(-1, q)
            loss = (ce * self.cb_loss_w.to(ce.dtype)).mean()

        with torch.no_grad():
            pred = logits_m.argmax(-1)            # [M, Q]
            acc = (pred == targets_m).float().mean()
            acc_cb0 = (pred[:, 0] == targets_m[:, 0]).float().mean()
            # True perplexity (unsmoothed CE)
            ppl = F.cross_entropy(
                logits_m.reshape(-1, self.codebook_size),
                targets_m.reshape(-1)).exp()

        return {"loss": loss, "acc": acc, "acc_cb0": acc_cb0, "ppl": ppl}

    # ── inference ────────────────────────────────────────────────────── #

    @torch.no_grad()
    def generate(self, *, text_ids: torch.Tensor, speaker_index: torch.Tensor,
                 max_frames: int, temperature: float = 0.7, top_k: int = 250,
                 top_p: float = 0.95,
                 prefix_codes: Optional[torch.Tensor] = None,
                 eos_silence_frames: int = 8) -> torch.Tensor:
        """Autoregressive generation of Mimi codes.

        Returns [B, Q, frames].  Prefills text once, then samples frame by
        frame with a KV cache.
        """
        self.eval()
        b, device = text_ids.shape[0], text_ids.device
        Q = self.num_codebooks
        speaker_vec = self.conditioner(speaker_index)   # [B, D]

        # ── text prefill ─────────────────────────────────────────────── #
        text_h = self.text_embedder(text_ids)           # [B, Lt, D]
        caches = [KVCache(k=torch.empty(0), v=torch.empty(0))
                  for _ in self.backbone.blocks]
        _, caches = self.backbone(text_h, causal=True, caches=caches,
                                  pos_offset=0)
        pos = text_ids.shape[1]

        # ── optional reference prefix (zero-shot cloning) ────────────── #
        if prefix_codes is not None:
            pf = prefix_codes.to(device)
            if pf.dim() == 2:
                pf = pf.unsqueeze(0).expand(b, -1, -1)
            cur = torch.full((b, Q), self.audio_embedder.audio_bos,
                             dtype=torch.long, device=device)
            for t in range(pf.shape[-1]):
                emb = (self.audio_embedder.embed_frame(cur)
                       + self.audio_embedder.type_bias
                       + speaker_vec).unsqueeze(1)     # [B, 1, D]
                _, caches = self.backbone(emb, causal=True, caches=caches,
                                          pos_offset=pos)
                pos += 1
                cur = pf[:, :, t]
            last_in = cur
        else:
            last_in = torch.full((b, Q), self.audio_embedder.audio_bos,
                                 dtype=torch.long, device=device)

        # ── frame-by-frame AR loop ───────────────────────────────────── #
        generated: List[torch.Tensor] = []
        silence_run = 0

        for _ in range(max_frames):
            emb = (self.audio_embedder.embed_frame(last_in)
                   + self.audio_embedder.type_bias
                   + speaker_vec).unsqueeze(1)          # [B, 1, D]
            h, caches = self.backbone(emb, causal=True, caches=caches,
                                      pos_offset=pos)
            pos += 1

            frame = self.depth_head.forward_generate(
                h[:, -1],
                temperature=temperature, top_k=top_k, top_p=top_p)  # [B, Q]
            generated.append(frame)
            last_in = frame

            # Crude EOS: all-zero cb0 for N consecutive frames
            if (frame[:, 0] == 0).all():
                silence_run += 1
                if silence_run >= eos_silence_frames:
                    break
            else:
                silence_run = 0

        if not generated:
            return torch.zeros(b, Q, 0, dtype=torch.long, device=device)
        return torch.stack(generated, dim=2)            # [B, Q, frames]
