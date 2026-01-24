# Chatterbox Turbo TTS - Wyoming Protocol Server

High-quality real-time text-to-speech using Chatterbox Turbo from Resemble AI.

## Features

- **Sub-200ms latency** - Real-time voice synthesis
- **<150ms streaming latency** - Even faster for streaming
- **High quality** - Outperforms ElevenLabs in blind tests (63.75% preference)
- **Paralinguistic tags** - Use [laugh], [cough], [chuckle] in text
- **350M parameters** - Efficient model size
- **ROCm compatible** - Runs on AMD GPUs

## Quick Start

```bash
# Build
docker compose build wyoming-chatterbox-turbo

# Start
docker compose up -d wyoming-chatterbox-turbo

# View logs
docker compose logs -f wyoming-chatterbox-turbo
```

## Configuration

Environment variables in `docker-compose.yml`:

- `CHATTERBOX_DEVICE` - Device to use (default: `cuda:0`)
- `CHATTERBOX_SAMPLES_PER_CHUNK` - Audio chunk size (default: `1024`)
- `CHATTERBOX_CACHE_DIR` - Model cache directory (default: `/data/models`)
- `CHATTERBOX_DEBUG` - Enable debug logging (default: `true`)

## Wyoming Protocol

The server listens on **port 10201** (mapped from container port 10200).

Connect from Home Assistant:
- Host: `<your-docker-host>`
- Port: `10201`

## Paralinguistic Tags

Add natural expressions in your text:
- `[laugh]` - Natural laughter
- `[chuckle]` - Brief laugh
- `[cough]` - Cough sound
- Example: `"Hi there [chuckle], how can I help you today?"`

## Model Details

- **Model**: ResembleAI/chatterbox-turbo
- **Parameters**: 350M
- **Sample Rate**: Model-dependent (typically 24kHz)
- **Generation**: Single-step distilled decoder
- **License**: MIT

## Performance

Expected on Radeon 8060S:
- **Generation time**: <1 second for typical sentences
- **Latency**: Sub-200ms for first audio
- **Quality**: Production-grade, beats commercial APIs

## Troubleshooting

### Model download issues
```bash
# Check cache directory
docker exec wyoming-chatterbox-turbo-rocm ls -la /data/hf_cache

# Clear cache and restart
docker compose down wyoming-chatterbox-turbo
rm -rf chatterbox-data/
docker compose up wyoming-chatterbox-turbo
```

### GPU not detected
```bash
# Check ROCm
docker exec wyoming-chatterbox-turbo-rocm rocm-smi

# Verify devices
docker exec wyoming-chatterbox-turbo-rocm ls -la /dev/dri /dev/kfd
```

### Audio quality issues
- Check generation logs for normalization warnings
- Verify sample rate matches Home Assistant expectations
- Try adjusting `CHATTERBOX_SAMPLES_PER_CHUNK`

## Comparison with Other Models

| Model | Latency | Quality | Size | Real-time? |
|-------|---------|---------|------|------------|
| **Chatterbox Turbo** | <150ms | ⭐⭐⭐⭐⭐ | 350M | ✅ Yes |
| Qwen3-TTS | ~50s | ⭐⭐⭐⭐⭐ | 1.7B | ❌ No |
| Piper | ~1-2s | ⭐⭐⭐ | 5-32M | ✅ Yes |
| Kokoro-82M | <300ms | ⭐⭐⭐⭐ | 82M | ✅ Yes |

## References

- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [Chatterbox-Turbo on Hugging Face](https://huggingface.co/ResembleAI/chatterbox-turbo)
- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
