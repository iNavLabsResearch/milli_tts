"""End-to-end smoke test (no network, no GPU, no real Mimi).

Exercises: config singleton -> tokenizer -> dummy Mimi codec -> tiny model ->
one training step (loss + backward) -> autoregressive generation -> decode.
Uses a tiny model config so it runs in a few seconds on CPU.

Run:  python -m tests.smoke_test
"""

from __future__ import annotations

import torch

from milli_tts.core.static_memory_cache import StaticMemoryCache


def _shrink_config(cfg) -> None:
    """Mutate the (frozen) config to a tiny, fast variant for CPU testing."""
    m = cfg.model
    for k, v in dict(d_model=64, backbone_layers=2, backbone_heads=4,
                     depth_layers=2, depth_heads=2, depth_dim=64,
                     ffn_mult=2).items():
        object.__setattr__(m, k, v)
    object.__setattr__(cfg.codec, "num_codebooks", 4)
    object.__setattr__(cfg.voice, "embedding_dim", 64)
    object.__setattr__(cfg.training, "gradient_checkpointing", False)


def _check_text_pad_invariance(model, codes, audio_mask, speaker_index,
                               vocab_size: int) -> None:
    """Right-padded text PAD positions must not leak into audio conditioning.

    Build two batches that are *identical* on the real (unpadded) rows — same
    real text, same audio, same masks, same Lt (so RoPE positions match) — but
    whose masked-out trailing text positions hold different token ids (PAD vs
    random real tokens). With the key-padding mask threaded into the backbone,
    audio frames attend only to real text, so both must yield the same loss.
    Before the fix, audio attended to the PAD content and the losses diverged.
    """
    model.eval()
    b, _, _ = codes.shape
    lt, real_len = 12, 7  # rows are padded from `real_len` up to `lt`

    text_mask = torch.zeros(b, lt, dtype=torch.bool)
    text_mask[:, :real_len] = True
    real = torch.randint(0, vocab_size, (b, real_len))

    # Batch A: trailing positions = PAD id. Batch B: trailing = random tokens.
    text_a = torch.full((b, lt), model.text_pad, dtype=torch.long)
    text_a[:, :real_len] = real
    text_b = torch.randint(0, vocab_size, (b, lt))
    text_b[:, :real_len] = real  # identical real prefix; only PAD region differs

    with torch.no_grad():
        la = model.compute_loss(text_ids=text_a, text_mask=text_mask,
                                audio_codes=codes, audio_mask=audio_mask,
                                speaker_index=speaker_index)["loss"]
        lb = model.compute_loss(text_ids=text_b, text_mask=text_mask,
                                audio_codes=codes, audio_mask=audio_mask,
                                speaker_index=speaker_index)["loss"]
    delta = (la - lb).abs().item()
    assert delta < 1e-5, (
        f"text PAD content leaked into audio conditioning: |Δloss|={delta:.2e} "
        "(expected ~0 — the key-padding mask is not excluding PAD positions)")
    model.train()
    print(f"[ok] text-pad invariance: |Δloss| = {delta:.2e} (PAD ignored)")


def main() -> int:
    StaticMemoryCache._reset()
    StaticMemoryCache.load("config.json")
    cfg = StaticMemoryCache.config()
    _shrink_config(cfg)

    from milli_tts.data.mimi_codec import MimiCodec
    from milli_tts.data.text_tokenizer import TextTokenizer
    from milli_tts.models.factory import build_model

    device = torch.device("cpu")
    tok = TextTokenizer.from_config()
    codec = MimiCodec.from_config(allow_dummy=True, device=device)
    assert codec.backend in ("dummy", "moshi")
    print(f"[ok] codec backend = {codec.backend}, tokenizer vocab = {tok.vocab_size}")

    model = build_model(text_vocab_size=tok.vocab_size,
                        num_codebooks=cfg.codec.num_codebooks,
                        codebook_size=cfg.codec.codebook_size).to(device)
    print(f"[ok] model params = {model.num_parameters()/1e6:.2f}M")

    # fake batch: 2 utterances of ~2s
    b, sr = 2, cfg.codec.sample_rate
    wav = torch.randn(b, 1, int(2.0 * sr)) * 0.1
    codes = codec.encode(wav)                       # [B, Q, frames]
    frames = codes.shape[-1]
    audio_mask = torch.ones(b, frames, dtype=torch.bool)
    text_ids = torch.randint(0, tok.vocab_size, (b, 12))
    text_mask = torch.ones(b, 12, dtype=torch.bool)
    speaker_index = torch.tensor([0, 1])

    out = model.compute_loss(text_ids=text_ids, text_mask=text_mask,
                             audio_codes=codes, audio_mask=audio_mask,
                             speaker_index=speaker_index)
    loss = out["loss"]
    loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in model.parameters()
               if p.grad is not None)
    assert torch.isfinite(loss) and grad > 0, "no gradient flowed"
    print(f"[ok] train step: loss = {loss.item():.4f}, acc = {out['acc'].item():.3f}")

    _check_text_pad_invariance(model, codes, audio_mask, speaker_index,
                               tok.vocab_size)

    # generation + decode
    gen = model.generate(text_ids=text_ids[:1], speaker_index=speaker_index[:1],
                         max_frames=20, temperature=0.8, top_k=50, top_p=0.95)
    assert gen.dim() == 3 and gen.shape[1] == cfg.codec.num_codebooks
    audio = codec.decode(gen)
    print(f"[ok] generate: codes {tuple(gen.shape)} -> wav {tuple(audio.shape)}")

    print("\nSMOKE TEST PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
