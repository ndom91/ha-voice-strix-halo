"""Wyoming protocol event handler for Qwen3-TTS."""

import io
import logging
import os
import time
import wave
from typing import Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoFeatureExtractor, FeatureExtractionMixin
from qwen_tts import Qwen3TTSModel
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

# Monkey-patch AutoFeatureExtractor to handle custom Qwen tokenizer
# This works around the qwen-tts package trying to use AutoFeatureExtractor
# on a custom tokenizer that transformers doesn't recognize
_original_from_pretrained = AutoFeatureExtractor.from_pretrained

def _patched_from_pretrained(pretrained_model_name_or_path, **kwargs):
    """Patched from_pretrained that handles Qwen3TTS tokenizer gracefully."""
    try:
        return _original_from_pretrained(pretrained_model_name_or_path, **kwargs)
    except (ValueError, OSError) as e:
        error_msg = str(e)
        if "Unrecognized feature extractor" in error_msg and "qwen3_tts_tokenizer" in error_msg:
            _LOGGER.warning(
                "AutoFeatureExtractor doesn't recognize qwen3_tts_tokenizer - "
                "this is expected, qwen-tts will handle it internally"
            )
            # Return None - qwen-tts should handle the tokenizer loading internally
            return None
        raise

AutoFeatureExtractor.from_pretrained = _patched_from_pretrained
_LOGGER.info("Applied AutoFeatureExtractor monkey-patch for Qwen3TTS compatibility")

# Global model cache (thread-safe singleton)
_model_cache: Optional[Qwen3TTSModel] = None
_model_lock = None


def get_model(
    model_name: str,
    device: str,
    dtype: str,
    flash_attention: bool,
    cache_dir: Optional[str] = None,
) -> Qwen3TTSModel:
    """Get or create the Qwen3TTS model (cached singleton)."""
    global _model_cache, _model_lock

    # Initialize lock on first call
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()

    with _model_lock:
        if _model_cache is None:
            _LOGGER.info("Loading Qwen3-TTS model: %s", model_name)
            _LOGGER.info("Device: %s, dtype: %s, flash_attention: %s", device, dtype, flash_attention)

            # Convert dtype string to torch dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

            # Prepare kwargs for from_pretrained
            model_kwargs = {
                "device_map": device,
                "dtype": torch_dtype,
            }

            # Use SDPA (Scaled Dot Product Attention) instead of flash_attention_2
            # SDPA is PyTorch's built-in optimized attention and doesn't require flash-attn package
            # Flash-attn requires NVIDIA CUDA or specific AMD MI-series GPUs and is not available for RDNA
            model_kwargs["attn_implementation"] = "sdpa"
            _LOGGER.info("Using SDPA (PyTorch scaled dot product attention)")

            # Add cache directory if specified
            if cache_dir:
                model_kwargs["cache_dir"] = cache_dir

            # Fix directory structure issue with speech_tokenizer configs
            # Root cause: config files are in model root but qwen-tts expects them in speech_tokenizer/
            # Reference: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/discussions/1
            if not os.path.isdir(model_name):
                _LOGGER.info("Downloading model and fixing speech_tokenizer directory structure for %s", model_name)
                try:
                    # Download the complete model
                    # Use HF_HOME environment variable instead of cache_dir parameter
                    # to ensure we're modifying the correct cache location
                    hf_cache = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or cache_dir
                    model_path = snapshot_download(
                        repo_id=model_name,
                        cache_dir=hf_cache,
                    )
                    _LOGGER.info("Model downloaded to: %s (cache: %s)", model_path, hf_cache)

                    # Create symlinks for config files in speech_tokenizer subdirectory
                    tokenizer_path = os.path.join(model_path, "speech_tokenizer")
                    if os.path.exists(tokenizer_path):
                        # Link config.json (which exists)
                        config_src = os.path.join(model_path, "config.json")
                        config_dst = os.path.join(tokenizer_path, "config.json")
                        if os.path.exists(config_src) and not os.path.exists(config_dst):
                            os.symlink(config_src, config_dst)
                            _LOGGER.info("Created symlink: config.json")

                        # Create preprocessor_config.json with feature_extractor_type
                        # The original config.json doesn't have this key, causing AutoFeatureExtractor to fail
                        preprocessor_dst = os.path.join(tokenizer_path, "preprocessor_config.json")

                        if not os.path.exists(preprocessor_dst):
                            import json
                            # Read the config.json to get base configuration
                            if os.path.exists(config_src):
                                with open(config_src, 'r') as f:
                                    config_data = json.load(f)

                                # Add the feature_extractor_type key that transformers expects
                                # Using "wav2vec2" as it's an audio feature extractor
                                config_data["feature_extractor_type"] = "Wav2Vec2FeatureExtractor"

                                # Write the patched config as preprocessor_config.json
                                with open(preprocessor_dst, 'w') as f:
                                    json.dump(config_data, f, indent=2)
                                _LOGGER.info("Created preprocessor_config.json with feature_extractor_type")

                        # Also link configuration.json if it exists
                        configuration_src = os.path.join(model_path, "configuration.json")
                        configuration_dst = os.path.join(tokenizer_path, "configuration.json")
                        if os.path.exists(configuration_src) and not os.path.exists(configuration_dst):
                            os.symlink(configuration_src, configuration_dst)
                            _LOGGER.info("Created symlink: configuration.json")
                    else:
                        _LOGGER.warning("speech_tokenizer directory not found at %s", tokenizer_path)

                    _LOGGER.info("Speech tokenizer directory structure fixed")
                except Exception as e:
                    _LOGGER.warning("Failed to fix directory structure: %s (will attempt normal load)", e)

            try:
                _model_cache = Qwen3TTSModel.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                _LOGGER.info("Model loaded successfully")

                # Note: torch.compile doesn't work on Qwen3TTSModel class directly
                # Would need to compile individual forward methods, but this adds complexity
                # and may not provide significant speedup on this architecture

            except Exception as e:
                _LOGGER.error("Failed to load model: %s", e)
                raise

        return _model_cache


class QwenEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol events."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        model_name: str,
        voice_instruct: str,
        language: str,
        device: str,
        dtype: str,
        flash_attention: bool,
        samples_per_chunk: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize handler."""
        super().__init__(reader, writer)

        self.wyoming_info = wyoming_info
        self.model_name = model_name
        self.voice_instruct = voice_instruct
        self.language = language
        self.device = device
        self.dtype = dtype
        self.flash_attention = flash_attention
        self.samples_per_chunk = samples_per_chunk
        self.cache_dir = cache_dir
        self.model: Optional[Qwen3TTSModel] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle a Wyoming protocol event."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            _LOGGER.debug("Sent info")
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            _LOGGER.info("Synthesizing: %s", synthesize.text)

            try:
                # Load model lazily on first synthesis
                if self.model is None:
                    self.model = get_model(
                        self.model_name,
                        self.device,
                        self.dtype,
                        self.flash_attention,
                        self.cache_dir,
                    )

                # Generate audio using VoiceDesign
                _LOGGER.debug("Generating audio with voice_instruct: %s", self.voice_instruct)
                start_time = time.time()
                result = self.model.generate_voice_design(
                    text=synthesize.text,
                    language=self.language,
                    instruct=self.voice_instruct,
                )
                generation_time = time.time() - start_time
                _LOGGER.info("Audio generation took %.2f seconds", generation_time)

                # Debug: Log what we received
                _LOGGER.debug("Result type: %s, length: %s",
                             type(result).__name__,
                             len(result) if isinstance(result, (tuple, list)) else "N/A")
                if isinstance(result, (tuple, list)) and len(result) > 0:
                    _LOGGER.debug("Result[0] type: %s, shape/value: %s",
                                 type(result[0]).__name__,
                                 result[0].shape if hasattr(result[0], 'shape') else result[0])
                    if len(result) > 1:
                        _LOGGER.debug("Result[1] type: %s, shape/value: %s",
                                     type(result[1]).__name__,
                                     result[1].shape if hasattr(result[1], 'shape') else result[1])

                # Handle tuple/list return - check both possible orderings
                if isinstance(result, (tuple, list)):
                    # Try to determine which element is audio vs sample_rate
                    # Sample rate is typically an integer (22050, 24000, etc)
                    # Audio is typically a tensor/array with shape
                    elem0_is_scalar = isinstance(result[0], (int, float)) or (
                        hasattr(result[0], 'numel') and result[0].numel() == 1
                    )

                    if elem0_is_scalar and len(result) > 1:
                        # First element is sample_rate, second is audio
                        sample_rate = int(result[0])
                        audio_data = result[1]
                        _LOGGER.info("Detected tuple format: (sample_rate=%d, audio)", sample_rate)
                    else:
                        # First element is audio, second (if exists) is sample_rate
                        audio_data = result[0]
                        sample_rate = int(result[1]) if len(result) > 1 else 24000
                        _LOGGER.info("Detected tuple format: (audio, sample_rate=%d)", sample_rate)
                else:
                    audio_data = result
                    sample_rate = 24000  # Default for Qwen3-TTS
                    _LOGGER.info("Single result, using default sample_rate=%d", sample_rate)

                # Convert list to numpy array
                if isinstance(audio_data, list):
                    audio_data = np.array(audio_data)
                    _LOGGER.debug("Converted list to numpy array: shape=%s, dtype=%s",
                                 audio_data.shape, audio_data.dtype)

                # Convert to numpy array if tensor
                if torch.is_tensor(audio_data):
                    audio_data = audio_data.cpu().numpy()
                    _LOGGER.debug("Converted tensor to numpy: shape=%s, dtype=%s",
                                 audio_data.shape, audio_data.dtype)

                # Ensure audio is in correct format (float32)
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Normalize to [-1, 1] range if needed
                max_val = np.abs(audio_data).max()
                if max_val > 1.0:
                    audio_data = audio_data / max_val

                # Convert to 16-bit PCM
                audio_pcm = (audio_data * 32767).astype(np.int16)

                _LOGGER.info(
                    "Generated audio: %d samples, %d Hz, %d bytes",
                    len(audio_pcm),
                    sample_rate,
                    len(audio_pcm) * 2,
                )

                # Send audio start event
                await self.write_event(
                    AudioStart(
                        rate=sample_rate,
                        width=2,  # 16-bit = 2 bytes
                        channels=1,  # mono
                    ).event()
                )

                # Stream audio in chunks
                chunk_size = self.samples_per_chunk
                for i in range(0, len(audio_pcm), chunk_size):
                    chunk = audio_pcm[i : i + chunk_size]
                    chunk_bytes = chunk.tobytes()

                    await self.write_event(
                        AudioChunk(
                            audio=chunk_bytes,
                            rate=sample_rate,
                            width=2,
                            channels=1,
                        ).event()
                    )

                # Send audio stop event
                await self.write_event(AudioStop().event())
                _LOGGER.debug("Audio synthesis complete")

            except Exception as e:
                _LOGGER.error("Synthesis failed: %s", e, exc_info=True)
                # Send empty audio to indicate failure
                await self.write_event(AudioStart(rate=22050, width=2, channels=1).event())
                await self.write_event(AudioStop().event())

            return True

        return True
