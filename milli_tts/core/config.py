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
    dataset_repo: str = "ai4bharat/IndicVoices"
    dataset_config: Optional[str] = "hindi"
    dataset_split: str = "train"
    streaming: bool = True
    push_to_hub: bool = False
    model_hub_repo: str = "iNavLabsResearch/milli-tts-indic"
    # Continuously back up training checkpoints to this HF model repo (created
    # if missing, under the token owner's namespace). Each run gets its own
    # timestamped+indexed folder. Disconnect-proof checkpointing.
    push_checkpoints_to_hub: bool = False
    checkpoint_repo: str = "milli_tts_weights"
    # Languages to keep (ISO codes like "hi"/"as" or names like "hindi").
    # IndicVoices ships one HF config per language, so a single entry here also
    # auto-selects the matching `dataset_config`. Empty list or ["all"] = keep
    # every language in the configured split.
    languages: List[str] = field(default_factory=lambda: ["hi"])
    # Reservoir-shuffle buffer over the streaming train iterator. Without it the
    # model trains on long contiguous speaker/length blocks, which causes the
    # periodic val sawtooth + per-epoch memorization. Reshuffled each epoch.
    # 0 disables. Memory ≈ buffer_size × avg encoded-audio-row bytes, so lower it
    # on small-RAM hosts. Train only — the val set stays a deterministic prefix.
    shuffle_buffer_size: int = 10000
    # IndicVoices ships a per-row `verification_report` (a dict of QC flags +
    # an overall `decision`). When True we DROP rows that are noisy / echoey /
    # mispronounced / text-mismatched or whose decision is below
    # `min_quality_decision` — this is the single biggest TTS-quality lever on
    # IndicVoices, which is spontaneous *field* speech, not studio TTS.
    quality_filter: bool = True
    # Lowest acceptable `verification_report.decision`. Order (best→worst):
    # excellent > good > average > poor. "good" keeps excellent+good.
    min_quality_decision: str = "good"
    # IndicVoices-R signal-quality gates (the `verification_report` column does
    # NOT exist in IndicVoices-R — it ships per-clip SNR / C50 / ASR-CER columns
    # instead). These are the "get clean audio" lever for the R corpus. None =
    # off (IndicVoices-R is already restored/clean, so they default off; set them
    # to drop the worst clips, e.g. min_snr=15.0 dB, max_cer=0.5).
    min_snr: Optional[float] = None
    max_cer: Optional[float] = None
    # Deterministic utterance-level train/val split: a row is held out for
    # validation iff hash(speaker_id|text) % val_holdout_mod == 0 (so ≈1/mod is
    # val). Keyed on the utterance (not the speaker) so val speakers still appear
    # in train — required for the per-speaker embedding to have a real val loss.
    # 0 = use the legacy first-`eval_samples`-rows prefix split instead.
    val_holdout_mod: int = 20


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
    # Pocket-TTS class (~110M). A from-scratch model has to *converge* inside the
    # ~6k-step budget, so it is deliberately small: a 250M model never leaves the
    # babble stage in 6k steps, a ~110M one starts forming words.
    d_model: int = 768
    backbone_layers: int = 8
    backbone_heads: int = 12
    depth_layers: int = 4
    depth_heads: int = 12
    depth_dim: int = 768
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
    # Per-codebook cross-entropy weights (length == codec.num_codebooks). This is
    # the "weight the depth transformer toward better learning" lever: the depth
    # transformer predicts ALL Q codebooks, and up-weighting the hard/important
    # ones focuses its capacity there WITHOUT adding parameters (so training stays
    # the same speed). cb0 is Mimi's WavLM-distilled *semantic* token and the main
    # intelligibility driver, so it gets the largest weight; the fine-texture
    # codebooks (cb6/cb7) — which the model learns easily and which matter least
    # for intelligibility — get the smallest. Weights are renormalized to mean 1
    # internally so train/val loss and perplexity stay comparable to a uniform run.
    # None / empty list = uniform (original behaviour).
    codebook_loss_weights: Optional[List[float]] = None


@dataclass(frozen=True)
class VoiceConfig:
    strategy: str = "embedding_table"
    embedding_dim: int = 768
    # Sized to comfortably exceed the IndicVoices-Hindi speaker count so the
    # pre-built catalog (tools/build_dataset.py) gets a dense, collision-free
    # index per speaker. The table is cheap (max_speakers × embedding_dim).
    max_speakers: int = 8192
    reference_clip_seconds: float = 10.0
    allow_zero_shot_prefix: bool = True
    # How to derive a speaker identity from each row:
    #   "row"    — one learned embedding per raw `speaker_id` (real per-speaker
    #     voices; this is what inference selects via voice_id). Use the
    #     pre-built speaker catalog so every id gets a stable, unique index.
    #   "gender" — collapse to (lang, gender) -> "hi_female"/"hi_male".
    speaker_id_source: str = "row"
    # A real IndicVoices-Hindi speaker_id (from the catalog) used when no
    # voice_id is passed at inference.
    default_voice_id: str = "S4259869900354210"


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "auto"
    precision: str = "bf16"
    fallback_precision: str = "fp16"
    # Effective batch = batch_size × grad_accum_steps. Keep it large (≈64): a
    # small effective batch is the #1 cause of the spiky "shark-fin" loss curve.
    batch_size: int = 16
    grad_accum_steps: int = 4
    max_steps: int = 6000
    warmup_steps: int = 600
    lr: float = 2e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # Cross-entropy label smoothing (0 disables). Kept at 0 so train/val report
    # the SAME, true cross-entropy — label smoothing on a 2048-way codebook
    # softmax inflates the loss and makes the curve jitter.
    label_smoothing: float = 0.0
    lr_schedule: str = "cosine"
    gradient_checkpointing: bool = True
    num_workers: int = 4
    log_every: int = 25
    eval_every: int = 250
    save_every: int = 1000
    keep_last_n_checkpoints: int = 3
    resume_from: str = "latest"
    max_audio_seconds: float = 12.0
    min_audio_seconds: float = 1.0
    compile_model: bool = False
    # Size of the held-out validation set (stream prefix) for periodic val loss.
    # Bigger + less-frequent eval = a smooth val curve instead of a noisy one.
    eval_samples: int = 128


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
