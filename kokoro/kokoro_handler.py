"""Wyoming protocol event handler for Kokoro TTS."""

import io
import logging
import time

import httpx
import numpy as np
from scipy.io import wavfile
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class KokoroEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol events."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        api_url: str,
        voice: str,
        speed: float = 1.0,
        api_timeout: float = 30.0,
    ) -> None:
        """Initialize handler."""
        super().__init__(reader, writer)

        self.wyoming_info = wyoming_info
        self.api_url = api_url.rstrip("/")
        self.voice = voice
        self.speed = speed
        self.api_timeout = api_timeout

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
                # Prepare API request
                endpoint = f"{self.api_url}/audio/speech"
                payload = {
                    "model": "kokoro",
                    "voice": self.voice,
                    "input": synthesize.text,
                    "response_format": "wav",
                    "speed": self.speed,
                }

                _LOGGER.debug("Calling Kokoro API: %s", endpoint)
                start_time = time.time()

                # Make async HTTP request to Kokoro-FastAPI
                async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                    response = await client.post(endpoint, json=payload)
                    response.raise_for_status()
                    audio_bytes = response.content

                api_time = time.time() - start_time
                _LOGGER.info("API call took %.2f seconds", api_time)

                # Parse audio response
                sample_rate, audio_data = self._parse_audio(audio_bytes)

                # Ensure int16 mono format
                if audio_data.dtype != np.int16:
                    # Normalize to [-1, 1] then convert to int16
                    if audio_data.dtype in [np.float32, np.float64]:
                        max_val = np.abs(audio_data).max()
                        if max_val > 1.0:
                            audio_data = audio_data / max_val
                        audio_data = (audio_data * 32767).astype(np.int16)
                    else:
                        audio_data = audio_data.astype(np.int16)

                # Ensure mono
                if audio_data.ndim > 1:
                    if audio_data.shape[1] == 1:
                        audio_data = audio_data.squeeze()
                    else:
                        # Mix to mono if stereo
                        audio_data = audio_data.mean(axis=1).astype(np.int16)

                # Calculate audio metrics
                audio_duration = len(audio_data) / sample_rate
                rtf = api_time / audio_duration if audio_duration > 0 else 0
                _LOGGER.info(
                    "Generated audio: %d samples, %d Hz, %.2f seconds, RTF: %.2fx",
                    len(audio_data),
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

                # Stream audio in chunks
                chunk_size = 1024
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i : i + chunk_size]
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

            except httpx.HTTPError as e:
                _LOGGER.error("HTTP request failed: %s", e, exc_info=True)
                # Send empty audio to indicate failure
                await self.write_event(AudioStart(rate=24000, width=2, channels=1).event())
                await self.write_event(AudioStop().event())
            except Exception as e:
                _LOGGER.error("Synthesis failed: %s", e, exc_info=True)
                # Send empty audio to indicate failure
                await self.write_event(AudioStart(rate=24000, width=2, channels=1).event())
                await self.write_event(AudioStop().event())

            return True

        return True

    def _parse_audio(self, audio_bytes: bytes) -> tuple[int, np.ndarray]:
        """Parse audio bytes (WAV or raw PCM) and return sample rate + data.

        Args:
            audio_bytes: Audio data (WAV format expected)

        Returns:
            Tuple of (sample_rate, audio_data_as_numpy_array)
        """
        # Check if it's a WAV file
        if audio_bytes.startswith(b"RIFF"):
            try:
                # Parse WAV file using scipy
                with io.BytesIO(audio_bytes) as wav_io:
                    sample_rate, audio_data = wavfile.read(wav_io)
                _LOGGER.debug(
                    "Parsed WAV: rate=%d Hz, shape=%s, dtype=%s",
                    sample_rate,
                    audio_data.shape,
                    audio_data.dtype,
                )
                return sample_rate, audio_data
            except Exception as e:
                _LOGGER.warning("Failed to parse WAV, treating as raw PCM: %s", e)

        # Fall back to raw PCM at 24000 Hz
        _LOGGER.debug("Treating as raw PCM at 24000 Hz")
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        return 24000, audio_data
