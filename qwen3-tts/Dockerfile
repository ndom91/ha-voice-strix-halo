FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ARG HSA_OVERRIDE_GFX_VERSION=11.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} \
    PYTHONUNBUFFERED=1

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    git \
    ninja-build \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Pin transformers version to match model requirements (from preprocessor_config.json)
RUN pip3 install --no-cache-dir \
    "transformers>=4.57.0" \
    accelerate \
    numpy \
    scipy \
    packaging \
    ninja

# Install qwen-tts from GitHub to get latest fixes (PyPI may be outdated)
# The model was just released Jan 22, 2026 and fixes are being actively pushed
RUN pip3 install --no-cache-dir git+https://github.com/QwenLM/Qwen3-TTS.git

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    wyoming \
    onnxruntime-rocm

# Attempt to install flash-attention for ROCm (non-fatal if fails)
# Note: ROCm flash-attention supports MI200x, MI250x, MI300x, MI355x GPUs (CDNA architecture)
# RDNA GPUs (like Radeon 8060S) may not be fully supported - standard attention will be used as fallback
# Compilation can take 5-10 minutes and requires significant build resources
RUN echo "========================================" && \
    echo "Attempting to install flash-attn for ROCm..." && \
    echo "Supported GPUs: MI200x, MI250x, MI300x, MI355x" && \
    echo "RDNA GPUs will fall back to standard attention" && \
    echo "========================================" && \
    MAX_JOBS=4 pip3 install --no-cache-dir flash-attn --no-build-isolation 2>&1 | tee /tmp/flash-attn-install.log || \
    (echo "========================================" && \
     echo "Flash attention installation failed" && \
     echo "This is expected for RDNA GPUs - standard attention will be used" && \
     echo "Build log saved to /tmp/flash-attn-install.log" && \
     echo "========================================") && \
    (pip3 show flash-attn >/dev/null 2>&1 && echo "✓ flash-attn successfully installed" || echo "✗ flash-attn not available - using standard PyTorch attention")

# Create app directory
WORKDIR /app

# Copy application files
COPY qwen_wrapper.py /app/
COPY qwen_handler.py /app/
COPY entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose Wyoming protocol port
EXPOSE 10200

ENTRYPOINT ["/app/entrypoint.sh"]
