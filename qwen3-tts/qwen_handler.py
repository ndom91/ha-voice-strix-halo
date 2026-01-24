"""Wyoming protocol event handler for Qwen3-TTS."""

import io
import logging
import wave
from typing import Optional

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

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

            try:
                _model_cache = Qwen3TTSModel(
                    model_name=model_name,
                    device=device,
                    dtype=torch_dtype,
                    use_flash_attention=flash_attention,
                    cache_dir=cache_dir,
                )
                _LOGGER.info("Model loaded successfully")
            except Exception as e:
                _LOGGER.error("Failed to load model: %s", e)
                raise

        return _model_cache


class QwenEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol events."""

    def __init__(
        self,
        wyoming_info: Info,
        model_name: str,
        voice_instruct: str,
        language: str,
        device: str,
        dtype: str,
        flash_attention: bool,
        samples_per_chunk: int,
        cache_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize handler."""
        super().__init__(*args, **kwargs)

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
                audio_data = self.model.generate_voice_design(
                    text=synthesize.text,
                    language=self.language,
                    instruct=self.voice_instruct,
                )

                # Convert to numpy array if tensor
                if torch.is_tensor(audio_data):
                    audio_data = audio_data.cpu().numpy()

                # Ensure audio is in correct format (float32)
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Normalize to [-1, 1] range if needed
                max_val = np.abs(audio_data).max()
                if max_val > 1.0:
                    audio_data = audio_data / max_val

                # Convert to 16-bit PCM
                audio_pcm = (audio_data * 32767).astype(np.int16)

                # Get sample rate from model (default to 22050 Hz for Qwen3-TTS-12Hz)
                sample_rate = 22050

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
