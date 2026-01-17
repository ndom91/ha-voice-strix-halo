# Wyoming Voice Services (ROCm 7.1.1)

Docker setup for Wyoming + `faster-whisper` (STT) and `piper` (TTS) with [ctranslate2-rocm](https://github.com/paralin/ctranslate2-rocm/blob/rocm/ROCM.md) designed for the AMD Ryzen AI Max 395+ (Radeon 8060S)

## Features

- **Whisper Medium Model** for high-quality speech recognition (STT)
- **Piper TTS** for natural text-to-speech synthesis
- **ROCm 7.1.1** GPU acceleration for AMD GPUs
- **Wyoming Protocol** for easy Home Assistant integration
- **CTranslate2-rocm** (paralin fork) for native AMD GPU support with HIP

## Services

- **wyoming-whisper** - Speech-to-Text on port `10300`
- **wyoming-piper** - Text-to-Speech on port `10200`

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

### Piper (TTS) Configuration

Available environment variables:
- `PIPER_VOICE` - Voice model (default: en_US-lessac-medium)
- `PIPER_LENGTH_SCALE` - Speech speed, 1.0 = normal, <1.0 = faster, >1.0 = slower
- `PIPER_NOISE_SCALE` - Voice variation, 0.0-1.0
- `PIPER_NOISE_W` - Phoneme duration variation, 0.0-1.0
- `PIPER_DEBUG` - true/false

Available voices: https://github.com/rhasspy/piper/blob/master/VOICES.md

## Resources

- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Piper TTS](https://github.com/rhasspy/piper)
- [Piper Voice Samples](https://rhasspy.github.io/piper-samples/)
- [paralin/ctranslate2-rocm](https://github.com/paralin/ctranslate2-rocm) - ROCm fork used
- [paralin/whisperX-rocm](https://github.com/paralin/whisperX-rocm) - Reference for ROCm setup
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [CTranslate2 ROCm Blog](https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html)

## License

- Whisper: MIT License
- CTranslate2: MIT License
- Wyoming: MIT License
- faster-whisper: MIT License
