"""milli_tts — production Indic + English end-to-end TTS.

A delayed-streams RQ-Transformer text-to-speech model built on top of the
frozen Mimi neural audio codec, designed to fine-tune on the ai4bharat
IndicVoices STT corpus and run sub-100ms streaming inference on a single T4.
"""

__version__ = "0.1.0"

from milli_tts.core.static_memory_cache import StaticMemoryCache

__all__ = ["StaticMemoryCache", "__version__"]
