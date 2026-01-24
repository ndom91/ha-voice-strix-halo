"""Wyoming protocol event handler for Pocket TTS."""

import logging
import time
from typing import Optional

import numpy as np
import torch
from pocket_tts import TTSModel
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

# Global model cache (thread-safe singleton)
# Pocket TTS model loading is relatively slow, so we cache it
_model_cache: Optional[TTSModel] = None
_voice_state_cache: Optional[torch.Tensor] = None
_voice_prompt_cache: Optional[str] = None
_model_lock = None


def get_model_and_voice(
    voice_prompt: str,
    cache_dir: Optional[str] = None,
) -> tuple[TTSModel, torch.Tensor]:
    """Get or create the Pocket TTS model and voice state (cached)."""
    global _model_cache, _voice_state_cache, _voice_prompt_cache, _model_lock

    # Initialize lock on first call
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()

    with _model_lock:
        # Load model if not cached
        if _model_cache is None:
            _LOGGER.info("Loading Pocket TTS model...")
            start_time = time.time()
            _model_cache = TTSModel.load_model()
            load_time = time.time() - start_time
            _LOGGER.info("Model loaded in %.2f seconds", load_time)
            _LOGGER.info("Sample rate: %d Hz", _model_cache.sample_rate)

        # Load voice state if not cached or if voice changed
        if _voice_state_cache is None or _voice_prompt_cache != voice_prompt:
            _LOGGER.info("Loading voice state for: %s", voice_prompt)
            start_time = time.time()
            _voice_state_cache = _model_cache.get_state_for_audio_prompt(voice_prompt)
            load_time = time.time() - start_time
            _voice_prompt_cache = voice_prompt
            _LOGGER.info("Voice state loaded in %.2f seconds", load_time)

        return _model_cache, _voice_state_cache


class PocketEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol events."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        voice_prompt: str,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize handler."""
        super().__init__(reader, writer)

        self.wyoming_info = wyoming_info
        self.voice_prompt = voice_prompt
        self.cache_dir = cache_dir
        self.model: Optional[TTSModel] = None
        self.voice_state: Optional[torch.Tensor] = None

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
                # Load model and voice state lazily on first synthesis
                if self.model is None or self.voice_state is None:
                    self.model, self.voice_state = get_model_and_voice(
                        self.voice_prompt,
                        self.cache_dir,
                    )

                # Generate audio
                _LOGGER.debug("Generating audio with Pocket TTS")
                start_time = time.time()

                # Pocket TTS returns a 1D torch tensor with PCM audio data
                audio_tensor = self.model.generate_audio(
                    self.voice_state,
                    synthesize.text,
                )

                generation_time = time.time() - start_time
                _LOGGER.info("Audio generation took %.2f seconds", generation_time)

                # Convert to numpy array
                if torch.is_tensor(audio_tensor):
                    audio_data = audio_tensor.cpu().numpy()
                else:
                    audio_data = np.array(audio_tensor)

                _LOGGER.debug("Audio data shape: %s, dtype: %s", audio_data.shape, audio_data.dtype)

                # Ensure audio is 1D
                if audio_data.ndim > 1:
                    # If 2D with shape (1, N) or (N, 1), flatten
                    audio_data = audio_data.squeeze()

                # Ensure float32 for processing
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Normalize to [-1, 1] range if needed
                max_val = np.abs(audio_data).max()
                if max_val > 1.0:
                    audio_data = audio_data / max_val
                    _LOGGER.debug("Normalized audio from max_val: %.3f", max_val)

                # Convert to 16-bit PCM
                audio_pcm = (audio_data * 32767).astype(np.int16)

                # Get sample rate from model
                sample_rate = self.model.sample_rate

                # Calculate real-time factor for performance monitoring
                audio_duration = len(audio_pcm) / sample_rate
                rtf = generation_time / audio_duration if audio_duration > 0 else 0
                _LOGGER.info(
                    "Generated audio: %d samples, %d Hz, %.2f seconds, RTF: %.2fx",
                    len(audio_pcm),
                    sample_rate,
                    audio_duration,
                    rtf,
                )

                # Send audio start event
                await self.write_event(
                    AudioStart(
                        rate=sample_rate,
                        width=2,  # 16-bit = 2 bytes
                        channels=1,  # mono
                    ).event()
                )

                # Stream audio in chunks for low latency
                # Use smaller chunks for better perceived latency
                # Pocket TTS is fast enough that we can stream in small chunks
                chunk_size = 1024  # ~21ms at 48kHz, ~42ms at 24kHz
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
                await self.write_event(AudioStart(rate=24000, width=2, channels=1).event())
                await self.write_event(AudioStop().event())

            return True

        return True
