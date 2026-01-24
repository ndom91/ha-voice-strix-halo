#!/bin/bash
set -e

# Default values
CHATTERBOX_DEVICE="${CHATTERBOX_DEVICE:-cuda:0}"
CHATTERBOX_SAMPLES_PER_CHUNK="${CHATTERBOX_SAMPLES_PER_CHUNK:-1024}"
CHATTERBOX_CACHE_DIR="${CHATTERBOX_CACHE_DIR:-}"
CHATTERBOX_DEBUG="${CHATTERBOX_DEBUG:-false}"

echo "==================================="
echo "Wyoming Chatterbox Turbo TTS Server"
echo "==================================="
echo "Device: $CHATTERBOX_DEVICE"
echo "Samples Per Chunk: $CHATTERBOX_SAMPLES_PER_CHUNK"
echo "Cache Dir: $CHATTERBOX_CACHE_DIR"
echo "Debug: $CHATTERBOX_DEBUG"
echo "==================================="

# Build command arguments
CMD_ARGS=(
    "--uri" "tcp://0.0.0.0:10200"
    "--device" "$CHATTERBOX_DEVICE"
    "--samples-per-chunk" "$CHATTERBOX_SAMPLES_PER_CHUNK"
)

# Add cache directory if specified
if [ -n "$CHATTERBOX_CACHE_DIR" ]; then
    CMD_ARGS+=("--cache-dir" "$CHATTERBOX_CACHE_DIR")
fi

# Add debug flag if enabled
if [ "$CHATTERBOX_DEBUG" = "true" ]; then
    CMD_ARGS+=("--debug")
fi

# Launch the server
echo "Starting Wyoming server..."
exec python3 /app/chatterbox_wrapper.py "${CMD_ARGS[@]}"
