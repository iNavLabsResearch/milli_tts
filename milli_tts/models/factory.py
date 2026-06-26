"""Model factory (Factory + Registry patterns).

New architectures register themselves and are built by name from
``config.json``'s ``model.arch`` field, so swapping models never touches the
trainer or inference engine.
"""

from __future__ import annotations

from typing import Optional

from milli_tts.core.registry import Registry
from milli_tts.core.static_memory_cache import StaticMemoryCache
from milli_tts.models.base import BaseTTSModel
from milli_tts.models.rq_transformer import RQTransformerTTS
from milli_tts.models.rq_transformer_v2 import RQTransformerTTSv2

MODEL_REGISTRY: Registry[BaseTTSModel] = Registry("models")


@MODEL_REGISTRY.register("rq_transformer_tts")
def _build_rq(*, text_vocab_size: int, num_codebooks: int,
              codebook_size: int) -> BaseTTSModel:
    cfg = StaticMemoryCache.config()
    # propagate grad checkpointing flag into the model config view
    object.__setattr__(cfg.model, "gradient_checkpointing",
                       cfg.training.gradient_checkpointing)
    return RQTransformerTTS(
        model_cfg=cfg.model, voice_cfg=cfg.voice,
        text_vocab_size=text_vocab_size, num_codebooks=num_codebooks,
        codebook_size=codebook_size)


@MODEL_REGISTRY.register("rq_transformer_tts_v2")
def _build_rq_v2(*, text_vocab_size: int, num_codebooks: int,
                 codebook_size: int) -> BaseTTSModel:
    """Modular rewrite of the RQ-Transformer (drop-in: same compute_loss /
    generate / checkpoint format). Switch to it by setting
    ``model.arch="rq_transformer_tts_v2"`` in config.json."""
    cfg = StaticMemoryCache.config()
    object.__setattr__(cfg.model, "gradient_checkpointing",
                       cfg.training.gradient_checkpointing)
    return RQTransformerTTSv2(
        model_cfg=cfg.model, voice_cfg=cfg.voice,
        text_vocab_size=text_vocab_size, num_codebooks=num_codebooks,
        codebook_size=codebook_size)


def build_model(*, text_vocab_size: int, num_codebooks: int,
                codebook_size: int, arch: Optional[str] = None) -> BaseTTSModel:
    cfg = StaticMemoryCache.config()
    arch = arch or cfg.model.arch
    builder = MODEL_REGISTRY.get(arch)
    return builder(text_vocab_size=text_vocab_size,
                   num_codebooks=num_codebooks, codebook_size=codebook_size)
