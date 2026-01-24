FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ARG HSA_OVERRIDE_GFX_VERSION=11.0.0

ENV DEBIAN_FRONTEND=noninteractive \
    HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    wyoming \
    qwen-tts \
    transformers \
    accelerate \
    numpy \
    scipy

# Attempt to install flash-attention for ROCm (non-fatal if fails)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash attention installation failed - will use standard attention"

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
