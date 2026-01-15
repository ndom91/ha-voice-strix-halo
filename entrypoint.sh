#!/bin/bash
set -e

# Use environment variables with defaults
WHISPER_MODEL="${WHISPER_MODEL:-medium}"
WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-float16}"
WHISPER_BEAM_SIZE="${WHISPER_BEAM_SIZE:-5}"

# Run wyoming-faster-whisper with configured parameters
exec python3 -m wyoming_faster_whisper \
    --model "$WHISPER_MODEL" \
    --device "cuda" \
    --compute-type "$WHISPER_COMPUTE_TYPE" \
    --beam-size "$WHISPER_BEAM_SIZE" \
    --uri "tcp://0.0.0.0:10300" \
    --data-dir "/data"
