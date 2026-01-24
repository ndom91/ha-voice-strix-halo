#!/usr/bin/env python3
"""Wyoming protocol server wrapper for Qwen3-TTS."""

import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from qwen_handler import QwenEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wyoming Qwen3-TTS Server")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="Model name or path",
    )
    parser.add_argument(
        "--instruct",
        default="Clear, natural voice with medium pitch",
        help="Voice design instruction",
    )
    parser.add_argument(
        "--language",
        default="Auto",
        help="TTS language (Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model data type (bfloat16, float16, float32)",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Enable flash attention if available",
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

    _LOGGER.info("Starting Wyoming Qwen3-TTS server")
    _LOGGER.info("Model: %s", args.model)
    _LOGGER.info("Device: %s", args.device)
    _LOGGER.info("Voice instruction: %s", args.instruct)
    _LOGGER.info("Language: %s", args.language)

    # Construct Wyoming protocol info
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="qwen3-tts",
                description="Qwen3-TTS-12Hz VoiceDesign",
                attribution=Attribution(
                    name="Qwen",
                    url="https://github.com/QwenLM/Qwen-TTS",
                ),
                installed=True,
                version="1.7.0",
                voices=[
                    TtsVoice(
                        name="voice_design",
                        description=f"Voice Design: {args.instruct}",
                        attribution=Attribution(
                            name="Qwen",
                            url="https://github.com/QwenLM/Qwen-TTS",
                        ),
                        installed=True,
                        version="1.7.0",
                        languages=[args.language.lower()],
                    )
                ],
            )
        ],
    )

    # Create event handler factory
    handler_factory = partial(
        QwenEventHandler,
        wyoming_info=wyoming_info,
        model_name=args.model,
        voice_instruct=args.instruct,
        language=args.language,
        device=args.device,
        dtype=args.dtype,
        flash_attention=args.flash_attention,
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
