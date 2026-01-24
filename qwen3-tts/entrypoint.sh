#!/bin/bash
set -e

# Default values
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
QWEN_VOICE_INSTRUCT="${QWEN_VOICE_INSTRUCT:-Clear, natural voice with medium pitch}"
QWEN_LANGUAGE="${QWEN_LANGUAGE:-Auto}"
QWEN_DEVICE="${QWEN_DEVICE:-cuda:0}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_FLASH_ATTENTION="${QWEN_FLASH_ATTENTION:-false}"
QWEN_SAMPLES_PER_CHUNK="${QWEN_SAMPLES_PER_CHUNK:-1024}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-}"
QWEN_DEBUG="${QWEN_DEBUG:-false}"

echo "==================================="
echo "Wyoming Qwen3-TTS Server"
echo "==================================="
echo "Model: $QWEN_MODEL"
echo "Voice Instruct: $QWEN_VOICE_INSTRUCT"
echo "Language: $QWEN_LANGUAGE"
echo "Device: $QWEN_DEVICE"
echo "Data Type: $QWEN_DTYPE"
echo "Flash Attention: $QWEN_FLASH_ATTENTION"
echo "Samples Per Chunk: $QWEN_SAMPLES_PER_CHUNK"
echo "Cache Dir: $QWEN_CACHE_DIR"
echo "Debug: $QWEN_DEBUG"
echo "==================================="

# Build command arguments
CMD_ARGS=(
    "--uri" "tcp://0.0.0.0:10200"
    "--model" "$QWEN_MODEL"
    "--instruct" "$QWEN_VOICE_INSTRUCT"
    "--language" "$QWEN_LANGUAGE"
    "--device" "$QWEN_DEVICE"
    "--dtype" "$QWEN_DTYPE"
    "--samples-per-chunk" "$QWEN_SAMPLES_PER_CHUNK"
)

# Add flash attention flag if enabled
if [ "$QWEN_FLASH_ATTENTION" = "true" ]; then
    CMD_ARGS+=("--flash-attention")
fi

# Add cache directory if specified
if [ -n "$QWEN_CACHE_DIR" ]; then
    CMD_ARGS+=("--cache-dir" "$QWEN_CACHE_DIR")
fi

# Add debug flag if enabled
if [ "$QWEN_DEBUG" = "true" ]; then
    CMD_ARGS+=("--debug")
fi

# Launch the server
echo "Starting Wyoming server..."
exec python3 /app/qwen_wrapper.py "${CMD_ARGS[@]}"
