#!/usr/bin/env python3
"""Wyoming protocol server wrapper for Chatterbox Turbo TTS."""

import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from chatterbox_handler import ChatterboxEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wyoming Chatterbox Turbo TTS Server")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--samples-per-chunk",
        type=int,
        default=1024,
        help="Number of samples per audio chunk",
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to cache models",
    )
    parser.add_argument(
        "--uri",
        required=True,
        help="URI to bind server (e.g., tcp://0.0.0.0:10200)",
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

    _LOGGER.info("Starting Wyoming Chatterbox Turbo TTS server")
    _LOGGER.info("Device: %s", args.device)

    # Construct Wyoming protocol info
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="chatterbox-turbo",
                description="Chatterbox Turbo - High-quality real-time TTS",
                attribution=Attribution(
                    name="Resemble AI",
                    url="https://github.com/resemble-ai/chatterbox",
                ),
                installed=True,
                version="1.0.0",
                voices=[
                    TtsVoice(
                        name="default",
                        description="Chatterbox Turbo default voice (no cloning)",
                        attribution=Attribution(
                            name="Resemble AI",
                            url="https://github.com/resemble-ai/chatterbox",
                        ),
                        installed=True,
                        version="1.0.0",
                        languages=["en"],  # Primary language is English
                    )
                ],
            )
        ],
    )

    # Create event handler factory
    handler_factory = partial(
        ChatterboxEventHandler,
        wyoming_info=wyoming_info,
        device=args.device,
        samples_per_chunk=args.samples_per_chunk,
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
