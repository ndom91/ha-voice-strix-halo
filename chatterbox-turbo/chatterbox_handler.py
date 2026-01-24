"""Wyoming protocol event handler for Chatterbox Turbo TTS."""

import io
import logging
import os
import time
from typing import Optional

import numpy as np
import torch
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

# Global model cache (thread-safe singleton)
_model_cache: Optional[object] = None
_model_lock = None


def get_model(
    device: str,
    cache_dir: Optional[str] = None,
):
    """Get or create the Chatterbox Turbo model (cached singleton)."""
    global _model_cache, _model_lock

    # Initialize lock on first call
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()

    with _model_lock:
        if _model_cache is None:
            _LOGGER.info("Loading Chatterbox Turbo TTS model")
            _LOGGER.info("Device: %s", device)

            try:
                # Don't import torchvision - not needed for chatterbox and causes
                # "operator torchvision::nms does not exist" on ROCm builds
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                _LOGGER.info("Calling ChatterboxTurboTTS.from_pretrained...")
                _model_cache = ChatterboxTurboTTS.from_pretrained(device=device)
                _LOGGER.info("Model loaded successfully")
                _LOGGER.info("Sample rate: %d Hz", _model_cache.sr)

            except ImportError as e:
                _LOGGER.error("Failed to import chatterbox: %s", e)
                _LOGGER.error("Install with: pip install git+https://github.com/resemble-ai/chatterbox.git")
                raise
            except Exception as e:
                _LOGGER.error("Failed to load model: %s", e, exc_info=True)
                raise

        return _model_cache


class ChatterboxEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol events."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        device: str,
        samples_per_chunk: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize handler."""
        super().__init__(reader, writer)

        self.wyoming_info = wyoming_info
        self.device = device
        self.samples_per_chunk = samples_per_chunk
        self.cache_dir = cache_dir
        self.model: Optional[object] = None

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
                        self.device,
                        self.cache_dir,
                    )

                # Generate audio
                _LOGGER.debug("Generating audio with Chatterbox Turbo")
                start_time = time.time()

                # Generate without voice cloning (no audio_prompt_path)
                wav = self.model.generate(synthesize.text, audio_prompt_path=None)

                generation_time = time.time() - start_time
                _LOGGER.info("Audio generation took %.2f seconds", generation_time)

                # Convert to numpy if tensor
                if torch.is_tensor(wav):
                    audio_data = wav.cpu().numpy()
                else:
                    audio_data = np.array(wav)

                _LOGGER.debug("Audio data type: %s, shape: %s",
                             type(audio_data).__name__,
                             audio_data.shape if hasattr(audio_data, 'shape') else 'N/A')

                # Squeeze to remove batch dimension if present
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.squeeze()

                # Ensure audio is in correct format (float32)
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Normalize to [-1, 1] range if needed
                max_val = np.abs(audio_data).max()
                if max_val > 1.0:
                    audio_data = audio_data / max_val
                    _LOGGER.debug("Normalized audio from max value: %.4f", max_val)

                # Convert to 16-bit PCM
                audio_pcm = (audio_data * 32767).astype(np.int16)

                # Get sample rate from model
                sample_rate = self.model.sr

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
