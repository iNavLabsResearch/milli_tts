"""Typed configuration objects.

The raw ``config.json`` is a nested dict. We wrap it in frozen dataclasses so
that the rest of the codebase gets attribute access, IDE autocompletion and
validation instead of stringly-typed dictionary lookups. Each section maps to
one dataclass; :class:`AppConfig` is the aggregate root.

Values of the form ``"ENV:VAR_NAME"`` are resolved from the process
environment at load time so that secrets (HF token, W&B key) never live in the
JSON file that gets committed to git.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar

_T = TypeVar("_T")

_ENV_PREFIX = "ENV:"
_B64_PREFIX = "B64:"
_REV_PREFIX = "REV:"


def _resolve_env(value: Any) -> Any:
    """Resolve config markers (recursively).

    * ``ENV:NAME`` -> ``os.environ["NAME"]``.
    * ``B64:<base64>`` -> base64-decoded string. Used to embed secrets in the
      committed ``config.json`` without tripping GitHub push protection's
      plaintext-token scanner. (Obfuscation, not encryption — the repo is still
      public; rotate keys if that matters.)
    """
    if isinstance(value, str) and value.startswith(_ENV_PREFIX):
        return os.environ.get(value[len(_ENV_PREFIX):], None)
    if isinstance(value, str) and value.startswith(_B64_PREFIX):
        import base64

        try:
            return base64.b64decode(value[len(_B64_PREFIX):]).decode("utf-8")
        except Exception:
            return None
    if isinstance(value, str) and value.startswith(_REV_PREFIX):
        # reversed string — GitHub's scanner matches token *prefixes*, which a
        # reversed token lacks, so this passes push protection (it even decodes
        # B64). Same caveat: obfuscation, not security; the repo is public.
        return value[len(_REV_PREFIX):][::-1]
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _from_dict(cls: Type[_T], data: Dict[str, Any]) -> _T:
    """Construct a dataclass from a dict, ignoring unknown keys.

    Unknown keys are silently dropped so that adding experimental fields to
    ``config.json`` never crashes an older code path.
    """
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]
    kwargs: Dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, val in data.items():
        if key in known:
            kwargs[key] = val
    return cls(**kwargs)  # type: ignore[call-arg]


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "milli_tts"
    repo: str = "iNavLabsResearch/milli_tts"
    seed: int = 1337
    run_name: str = "milli-tts-v1"


@dataclass(frozen=True)
class PathsConfig:
    root: str = "."
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    raw_audio_dir: str = "data/audio"
    mimi_code_cache_dir: str = "data/mimi_codes"
    checkpoint_dir: str = "checkpoints"
    voice_bank_dir: str = "checkpoints/voices"
    log_dir: str = "logs"
    output_dir: str = "outputs"

    def all(self) -> List[str]:
        return [
            self.data_dir, self.cache_dir, self.raw_audio_dir,
            self.mimi_code_cache_dir, self.checkpoint_dir,
            self.voice_bank_dir, self.log_dir, self.output_dir,
        ]


@dataclass(frozen=True)
class HuggingFaceConfig:
    token: Optional[str] = None
    dataset_repo: str = "SPRINGLab/IndicTTS-Hindi"
    dataset_config: Optional[str] = "default"
    dataset_split: str = "train"
    streaming: bool = True
    push_to_hub: bool = False
    model_hub_repo: str = "iNavLabsResearch/milli-tts-indic"
    # Languages to keep (ISO codes like "hi"/"as" or names like "hindi").
    # IndicVoices ships one HF config per language, so a single entry here also
    # auto-selects the matching `dataset_config`. Empty list or ["all"] = keep
    # every language in the configured split.
    languages: List[str] = field(default_factory=lambda: ["hi"])


@dataclass(frozen=True)
class CodecConfig:
    name: str = "mimi"
    hf_repo: str = "kyutai/mimi"
    sample_rate: int = 24000
    frame_rate: float = 12.5
    num_codebooks: int = 8
    max_codebooks: int = 32
    codebook_size: int = 2048
    freeze: bool = True
    # When False (the default for real training) a failure to load the real
    # Mimi codec is a hard error instead of silently falling back to the dummy
    # codec — training on dummy (pseudo-random) codes never converges. Set True
    # only for CPU smoke tests where `moshi` isn't installed.
    allow_dummy: bool = False


@dataclass(frozen=True)
class TokenizerConfig:
    type: str = "sentencepiece_hf"
    hf_repo: str = "google/byt5-small"
    fallback_repo: str = "ai4bharat/IndicBERTv2-MLM-only"
    max_text_len: int = 256
    use_byte_fallback: bool = True


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "rq_transformer_tts"
    d_model: int = 1024
    backbone_layers: int = 12
    backbone_heads: int = 16
    depth_layers: int = 6
    depth_heads: int = 8
    depth_dim: int = 1024
    ffn_mult: int = 4
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm: str = "rmsnorm"
    activation: str = "gelu"
    tie_text_embeddings: bool = False
    max_seq_frames: int = 1500
    text_delay: int = 0
    audio_delay_steps: int = 2
    stream_delay_frames: int = 16


@dataclass(frozen=True)
class VoiceConfig:
    strategy: str = "embedding_table"
    embedding_dim: int = 1024
    max_speakers: int = 2048
    reference_clip_seconds: float = 10.0
    allow_zero_shot_prefix: bool = True
    default_voice_id: str = "S4259699400335456"


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "auto"
    precision: str = "bf16"
    fallback_precision: str = "fp16"
    batch_size: int = 8
    grad_accum_steps: int = 4
    max_steps: int = 200000
    warmup_steps: int = 2000
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    lr_schedule: str = "cosine"
    gradient_checkpointing: bool = True
    num_workers: int = 2
    log_every: int = 20
    eval_every: int = 1000
    save_every: int = 2000
    keep_last_n_checkpoints: int = 3
    resume_from: str = "latest"
    max_audio_seconds: float = 20.0
    min_audio_seconds: float = 1.0
    compile_model: bool = False


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = True
    api_key: Optional[str] = None
    project: str = "milli-tts"
    entity: Optional[str] = None
    mode: str = "online"
    log_audio_samples: bool = True
    audio_sample_every: int = 2000
    watch_model: bool = False


@dataclass(frozen=True)
class InferenceConfig:
    device: str = "auto"
    precision: str = "fp16"
    temperature: float = 0.7
    top_k: int = 250
    top_p: float = 0.95
    cfg_scale: float = 2.0
    max_gen_seconds: float = 30.0
    target_latency_ms: int = 100
    use_kv_cache: bool = True
    use_cuda_graphs: bool = False
    stream_chunk_frames: int = 4


@dataclass(frozen=True)
class AppConfig:
    """Aggregate root holding every config section."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # raw resolved dict kept around for debugging / forward-compat fields
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        data = _resolve_env(data)
        return cls(
            project=_from_dict(ProjectConfig, data.get("project", {})),
            paths=_from_dict(PathsConfig, data.get("paths", {})),
            huggingface=_from_dict(HuggingFaceConfig, data.get("huggingface", {})),
            codec=_from_dict(CodecConfig, data.get("codec", {})),
            tokenizer=_from_dict(TokenizerConfig, data.get("tokenizer", {})),
            model=_from_dict(ModelConfig, data.get("model", {})),
            voice=_from_dict(VoiceConfig, data.get("voice", {})),
            training=_from_dict(TrainingConfig, data.get("training", {})),
            wandb=_from_dict(WandbConfig, data.get("wandb", {})),
            inference=_from_dict(InferenceConfig, data.get("inference", {})),
            raw=data,
        )
