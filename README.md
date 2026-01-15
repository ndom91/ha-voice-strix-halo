# Wyoming Faster Whisper with ROCm 7.1.1

Docker setup for running OpenAI's Whisper speech recognition with Wyoming protocol on AMD GPUs using ROCm 7.1.1. Optimized for Home Assistant integration.

## Features

- **Whisper Medium Model** for high-quality speech recognition
- **ROCm 7.1.1** GPU acceleration for AMD GPUs
- **Wyoming Protocol** for easy Home Assistant integration
- **CTranslate2-rocm** (paralin fork) for native AMD GPU support with HIP

## Prerequisites

- AMD GPU with gfx900 or newer architecture (RDNA/RDNA2/RDNA3 recommended)
- ROCm drivers installed on host (version 6.0+)
- Docker and Docker Compose
- ~5GB VRAM for medium model
- ~15GB disk space for Docker image (larger due to multi-arch build)

## Installation

### 1. Find Your GPU Architecture

```bash
rocminfo | grep "gfx"
```

You'll see output like `gfx1100`, `gfx1030`, `gfx906`, etc.

### 2. Configure GPU Architecture

Edit the `.env` file and set your GPU architecture override:

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

See `.env.example` for more GPU architecture mappings.

### 3. Build and Run

```bash
docker compose up -d --build
```

**Note:** First build will take 20-40 minutes as it:
- Compiles CTranslate2-rocm from source with HIP support
- Builds for multiple GPU architectures (gfx1030-1102)
- Downloads the Whisper medium model (~1.5GB)

### 4. Verify It's Running

Check logs:
```bash
docker compose logs -f
```

You should see output indicating GPU detection and Wyoming server startup.

Test connection:
```bash
curl http://localhost:10300
```

## Home Assistant Integration

### Add Wyoming Integration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for **"Wyoming Protocol"**
3. Enter connection details:
   - **Host**: `<docker-host-ip>` (e.g., `192.168.1.100`)
   - **Port**: `10300`
4. Click **Submit**

### Configure Voice Assistant

1. Go to **Settings** → **Voice Assistants**
2. Select or create an assistant
3. Set **Speech-to-Text** to your Wyoming Whisper instance
4. Save and test with "Hey, what's the weather?"

## Configuration

### Change Whisper Model

Edit `Dockerfile` entrypoint to use different model sizes:

```dockerfile
ENTRYPOINT ["python3", "-m", "wyoming_faster_whisper", \
    "--model", "tiny",  # Options: tiny, base, small, medium, large
    ...
```

Model sizes and VRAM requirements:
- **tiny**: ~1GB VRAM, fastest, good for simple commands
- **base**: ~1.5GB VRAM, balanced
- **small**: ~2GB VRAM, better accuracy
- **medium**: ~5GB VRAM, high accuracy (default)
- **large**: ~10GB VRAM, best accuracy, slower

### Build for Specific GPU Only

To reduce build time, edit `docker-compose.yml` to build only for your GPU:

```yaml
args:
  AMDGPU_TARGETS: gfx1100  # Your specific architecture
  CMAKE_HIP_ARCHITECTURES: gfx1100
```

This cuts build time significantly (10-15 minutes vs 30-40 minutes).

### Adjust Performance

Edit `Dockerfile` entrypoint:

```dockerfile
# Faster, lower quality
--compute-type int8
--beam-size 1

# Slower, higher quality
--compute-type float16
--beam-size 5
```

## Troubleshooting

### Build Fails During CTranslate2 Compilation

**Issue**: CMake or compilation errors

**Solution**:
- Ensure at least 15GB free disk space
- Give Docker sufficient resources (6+ CPU cores, 12GB+ RAM)
- Try building for single GPU arch instead of multi-arch
- Check ROCm base image is accessible
- Try: `docker compose build --no-cache`

### GPU Not Detected

**Issue**: Container shows CPU inference or can't find GPU

**Solution**:
1. Verify ROCm installation on host: `rocminfo`
2. Check devices exist: `ls -la /dev/kfd /dev/dri`
3. Verify user groups: `groups` (should include `video` and `render`)
4. Check `HSA_OVERRIDE_GFX_VERSION` matches your GPU
5. Test GPU in container:
   ```bash
   docker run --rm --device=/dev/kfd --device=/dev/dri \
     rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
     rocminfo
   ```

### Python Import Error for ctranslate2

**Issue**: `ImportError: cannot import name 'Translator' from 'ctranslate2'`

**Solution**:
- Verify CTranslate2 built successfully (check build logs)
- Ensure `LD_LIBRARY_PATH` includes `/usr/local/lib`
- Try rebuilding with `--no-cache`

### Slow Inference / Low Performance

**Issue**: Speech recognition slower than expected

**Solution**:
1. Verify GPU is being used:
   ```bash
   docker compose exec wyoming-whisper python3 -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
   ```
   Should show: `['int8_float16', 'int8', 'float16', ...]`
2. Lower beam-size to 1 for faster inference
3. Use smaller model (small or base)
4. Check GPU load: `rocm-smi` on host
5. Ensure compute-type is `float16` or `int8` (not `float32`)

### Model Download Fails

**Issue**: Cannot download Whisper model during build

**Solution**:
- Check internet connection and firewall
- Retry build: `docker compose build --no-cache`
- Pre-download model and mount:
  ```bash
  mkdir -p ./data/models
  python3 -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', download_root='./data/models')"
  ```

### Container Exits Immediately

**Issue**: Container starts then stops

**Solution**:
1. Check logs: `docker compose logs wyoming-whisper`
2. Verify port 10300 isn't in use: `ss -tuln | grep 10300`
3. Check for error messages about missing libraries
4. Verify GPU access works (see "GPU Not Detected")

## Technical Details

### Architecture

This setup uses:
- **Base Image**: `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`
- **CTranslate2 Fork**: [paralin/ctranslate2-rocm](https://github.com/paralin/ctranslate2-rocm) (rocm branch)
- **Whisper Implementation**: faster-whisper (CTranslate2-based)
- **Protocol**: Wyoming for Home Assistant

### Why CTranslate2-rocm (paralin fork)?

Standard CTranslate2 doesn't support ROCm/HIP. The paralin fork:
- Properly compiles with HIP support (`-DWITH_HIP=ON`)
- Uses ROCm's clang/clang++ compilers
- Includes OpenBLAS for optimized CPU operations
- Well-tested with ROCm 7.0+

Even though we use `device="cuda"` in Python code, ROCm's HIP layer automatically translates CUDA API calls to AMD GPU instructions.

### Multi-Architecture Build

The default `docker-compose.yml` builds for multiple GPU architectures:
- **gfx1100, gfx1101, gfx1102**: RDNA 3 (RX 7000 series)
- **gfx1030, gfx1031, gfx1032**: RDNA 2 (RX 6000 series)

This makes the image portable but increases build time. For production, build only for your specific GPU.

### Device Mounts

```yaml
devices:
  - /dev/kfd:/dev/kfd    # Kernel Fusion Driver (ROCm compute)
  - /dev/dri:/dev/dri    # Direct Rendering Infrastructure (GPU access)
```

Both are required for ROCm GPU access.

### Environment Variables

- **HSA_OVERRIDE_GFX_VERSION**: Overrides GPU architecture detection (required for some GPUs)
- **ROCM_PATH**: ROCm installation location
- **LD_LIBRARY_PATH**: Ensures CTranslate2 shared libraries are found
- **CTRANSLATE2_ROOT**: Installation prefix for CTranslate2

## Performance Expectations

On an AMD RX 7900 XTX with medium model:
- **Real-time factor**: ~0.1-0.2x (10-20% of audio duration)
- **Latency**: ~500ms-1s for typical voice commands
- **VRAM usage**: ~4-5GB

Performance varies by GPU generation and model size.

## Resources

- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- [paralin/ctranslate2-rocm](https://github.com/paralin/ctranslate2-rocm) - ROCm fork used
- [paralin/whisperX-rocm](https://github.com/paralin/whisperX-rocm) - Reference for ROCm setup
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [CTranslate2 ROCm Blog](https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html)

## License

This setup combines multiple open-source projects:
- Whisper: MIT License
- CTranslate2: MIT License
- Wyoming: MIT License
- faster-whisper: MIT License
