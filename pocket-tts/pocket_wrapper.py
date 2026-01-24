#!/usr/bin/env python3
"""Wyoming protocol server wrapper for Pocket TTS."""

import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from pocket_handler import PocketEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wyoming Pocket TTS Server")
    parser.add_argument(
        "--voice",
        default="hf://kyutai/tts-voices/alba-mackenna/casual.wav",
        help="Voice audio prompt (HuggingFace URL or local .wav file path)",
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to cache models",
    )
    parser.add_argument(
        "--uri",
        required=True,
        help="URI to bind server (e.g., tcp://0.0.0.0:10202)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.info("Starting Wyoming Pocket TTS server")
    _LOGGER.info("Voice: %s", args.voice)

    # Determine voice name for Home Assistant
    # Extract voice name from HuggingFace URL or use filename
    voice_name = "pocket-tts"
    voice_description = "Pocket TTS Voice"
    if args.voice.startswith("hf://"):
        # Extract voice name from HF URL (e.g., "alba-mackenna" from "hf://kyutai/tts-voices/alba-mackenna/casual.wav")
        parts = args.voice.split("/")
        if len(parts) >= 3:
            voice_name = parts[-2]  # Get the voice name part
            voice_description = f"Pocket TTS - {voice_name}"
    else:
        # Use filename without extension
        import os
        voice_name = os.path.splitext(os.path.basename(args.voice))[0]
        voice_description = f"Pocket TTS - {voice_name}"

    _LOGGER.info("Voice name: %s", voice_name)
    _LOGGER.info("Voice description: %s", voice_description)

    # Construct Wyoming protocol info
    # Pocket TTS currently supports English only
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="pocket-tts",
                description="Kyutai Pocket TTS - Fast, low-latency TTS",
                attribution=Attribution(
                    name="Kyutai",
                    url="https://github.com/kyutai-labs/pocket-tts",
                ),
                installed=True,
                version="1.0.0",
                voices=[
                    TtsVoice(
                        name=voice_name,
                        description=voice_description,
                        attribution=Attribution(
                            name="Kyutai",
                            url="https://github.com/kyutai-labs/pocket-tts",
                        ),
                        installed=True,
                        version="1.0.0",
                        languages=["en"],  # English only
                    )
                ],
            )
        ],
    )

    # Create event handler factory
    handler_factory = partial(
        PocketEventHandler,
        wyoming_info=wyoming_info,
        voice_prompt=args.voice,
        cache_dir=args.cache_dir,
    )

    # Start server
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Server listening on %s", args.uri)

    try:
        await server.run(handler_factory)
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
