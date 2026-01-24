#!/bin/bash
set -e

# Default values
POCKET_VOICE="${POCKET_VOICE:-hf://kyutai/tts-voices/alba-mackenna/casual.wav}"
POCKET_CACHE_DIR="${POCKET_CACHE_DIR:-}"
POCKET_DEBUG="${POCKET_DEBUG:-false}"

echo "==================================="
echo "Wyoming Pocket TTS Server"
echo "==================================="
echo "Voice: $POCKET_VOICE"
echo "Cache Dir: $POCKET_CACHE_DIR"
echo "Debug: $POCKET_DEBUG"
echo "==================================="

# Build command arguments
CMD_ARGS=(
    "--uri" "tcp://0.0.0.0:10202"
    "--voice" "$POCKET_VOICE"
)

# Add cache directory if specified
if [ -n "$POCKET_CACHE_DIR" ]; then
    CMD_ARGS+=("--cache-dir" "$POCKET_CACHE_DIR")
fi

# Add debug flag if enabled
if [ "$POCKET_DEBUG" = "true" ]; then
    CMD_ARGS+=("--debug")
fi

# Launch the server
echo "Starting Wyoming server..."
exec python3 /app/pocket_wrapper.py "${CMD_ARGS[@]}"
