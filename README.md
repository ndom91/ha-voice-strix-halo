# Wyoming Voice Services (ROCm 7.1.1)

Docker setup for Wyoming protocol speech services with [ctranslate2-rocm](https://github.com/paralin/ctranslate2-rocm/blob/rocm/ROCM.md) designed for the AMD Ryzen AI Max 395+ (Radeon 8060S)

## Features

- **Whisper** for high-quality speech recognition (STT)
- **Multiple TTS engines** - Qwen3, Chatterbox Turbo, Pocket, and Kokoro
- **ROCm 7.1.1** GPU acceleration for AMD GPUs (where applicable)
- **Wyoming Protocol** for easy Home Assistant integration
- **CTranslate2-rocm** (paralin fork) for native AMD GPU support with HIP

## Services

### Speech-to-Text (STT)
- **wyoming-whisper** - Speech-to-Text on port `10300`

### Text-to-Speech (TTS)
- **wyoming-qwen-tts** - Qwen3 TTS on port `10200` (GPU-accelerated, voice instructions)
- **wyoming-chatterbox-turbo** - Chatterbox Turbo on port `10201` (GPU-accelerated, sub-200ms latency)
- **wyoming-pocket-tts** - Pocket TTS on port `10202` (CPU-only, ultra-low latency)
- **wyoming-kokoro-tts** - Kokoro TTS on port `10203` (API proxy, multi-language, no GPU required)

## Prerequisites

- AMD GPU (i.e. Radeon 8060S from Ryzen AI Max 395+)
- ROCm drivers installed on host (version 7.1.1)
- Docker and Docker Compose
- ~5GB VRAM for Whisper medium model
- ~20GB disk space for Docker images (larger due to multi-arch build)

## Installation

### 1. Find Your GPU Architecture

```bash
rocminfo | grep "gfx"
```

You'll see output like `gfx1100`, `gfx1030`, `gfx906`, etc.

### 2. Configure GPU Architecture

Create the `.env` file based on the `.env.example` and set your GPU architecture override:

The default is `gfx1151` corresponding to `11.5.1` (for RDNA 3.5 / Radeon 8060S).

```bash
# For RDNA 3 (RX 7000 series): gfx1100, gfx1101, gfx1102
HSA_OVERRIDE_GFX_VERSION=11.0.0
# For RDNA 2 (RX 6000 series): gfx1030, gfx1031, gfx1032
HSA_OVERRIDE_GFX_VERSION=10.3.0
# For RDNA (RX 5000 series): gfx1010, gfx1012
HSA_OVERRIDE_GFX_VERSION=10.1.0
# For Vega: gfx900, gfx906
HSA_OVERRIDE_GFX_VERSION=9.0.0
```

### 3. Build and Run

```bash
docker compose up -d
```

## Configuration

All configuration is via `.env` file. Copy `.env.example` to `.env` and adjust.

### Whisper (STT) Configuration

Available environment variables:
- `WHISPER_MODEL` - Model size: tiny, base, small, medium (default), large
- `WHISPER_COMPUTE_TYPE` - float16 (default), int8
- `WHISPER_BEAM_SIZE` - 1-10, default 5 (higher = better quality, slower)
- `WHISPER_DEBUG` - true/false

Model sizes and VRAM requirements:
- **tiny**: ~1GB VRAM, fastest, good for simple commands
- **base**: ~1.5GB VRAM, balanced
- **small**: ~2GB VRAM, better accuracy
- **medium**: ~5GB VRAM, high accuracy (default)
- **large**: ~10GB VRAM, best accuracy, slower

### TTS Configuration

#### Qwen3-TTS (Port 10200)
- `QWEN_MODEL` - Model choice (default: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- `QWEN_VOICE_INSTRUCT` - Text description of desired voice
- `QWEN_LANGUAGE` - Language selection (Auto, Chinese, English, Japanese, etc.)
- `QWEN_DEVICE` - cuda:0 (GPU) or cpu
- `QWEN_DTYPE` - bfloat16 (default), float16, float32, int8, int4
- `QWEN_FLASH_ATTENTION` - true/false
- `QWEN_DEBUG` - true/false

#### Chatterbox Turbo (Port 10201)
- `CHATTERBOX_DEVICE` - cuda:0 (GPU) or cpu
- `CHATTERBOX_SAMPLES_PER_CHUNK` - Audio streaming chunk size (default: 1024)
- `CHATTERBOX_DEBUG` - true/false

Requires HF_TOKEN for gated model access.

#### Pocket TTS (Port 10202)
- `POCKET_VOICE` - Built-in voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma
- `POCKET_DEBUG` - true/false

CPU-only, ultra-low latency (~200ms to first audio chunk).

#### Kokoro TTS (Port 10203)
- `KOKORO_API_URL` - Kokoro-FastAPI endpoint (default: http://10.0.3.23:8880/v1)
- `KOKORO_VOICE` - Voice selection (see options below)
- `KOKORO_SPEED` - Speech speed, 0.5-2.0 (default: 1.0)
- `KOKORO_TIMEOUT` - API request timeout in seconds (default: 30)
- `KOKORO_DEBUG` - true/false

**Voice Options:**
- Female American: `af_bella`, `af_sarah`, `af_sky`
- Male American: `am_adam`, `am_michael`
- Female British: `bf_emma`, `bf_isabella`
- Male British: `bm_george`, `bm_lewis`

**Voice Mixing:**
- Simple: `af_bella+af_sky` (equal mix)
- Weighted: `af_bella(2)+af_sky(1)` (2:1 ratio)

**Features:**
- Multi-language support (en, ja, zh, ko, fr, es)
- No local GPU required - lightweight proxy service
- Voice changes require container restart

## Home Assistant Integration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "Wyoming Protocol"
3. Add each service separately:
   - **Whisper**: Host = your-docker-host, Port = 10300
   - **Qwen3-TTS**: Host = your-docker-host, Port = 10200
   - **Chatterbox Turbo**: Host = your-docker-host, Port = 10201
   - **Pocket TTS**: Host = your-docker-host, Port = 10202
   - **Kokoro TTS**: Host = your-docker-host, Port = 10203
4. Configure your voice assistant pipeline in **Settings** → **Voice Assistants**

## Resources

### Wyoming & STT
- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)

### TTS Engines
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [Chatterbox Turbo](https://huggingface.co/MycroftAI/Chatterbox-Turbo)
- [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

### ROCm
- [paralin/ctranslate2-rocm](https://github.com/paralin/ctranslate2-rocm) - ROCm fork used
- [paralin/whisperX-rocm](https://github.com/paralin/whisperX-rocm) - Reference for ROCm setup
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [CTranslate2 ROCm Blog](https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html)

## License

- Whisper: MIT License
- CTranslate2: MIT License
- Wyoming: MIT License
- faster-whisper: MIT License
- Qwen3-TTS: Apache 2.0 License
- Chatterbox Turbo: Apache 2.0 License
- Pocket TTS: Apache 2.0 License
- Kokoro-82M: Apache 2.0 License
