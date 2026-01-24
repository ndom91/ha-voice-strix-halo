# FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.11_pytorch_release_2.9.1
FROM rocm/pytorch:rocm7.1_ubuntu22.04_py3.11_pytorch_release_2.9.1

ARG HSA_OVERRIDE_GFX_VERSION=11.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

# Try experimental ROCm 7.1 PyTorch builds from download.pytorch.org
# Falls back to base image torch if not available
RUN pip3 install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/rocm7.1 \
    torch torchaudio triton || echo "Using base image torch"

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    wyoming

# Install Chatterbox TTS with all dependencies, then remove incompatible torchvision
# Torchvision gets installed as a transitive dependency but is incompatible with ROCm
# and causes "operator torchvision::nms does not exist" errors
RUN pip3 install --no-cache-dir \
    git+https://github.com/resemble-ai/chatterbox.git && \
    pip3 uninstall -y torchvision || true

# Create app directory
WORKDIR /app

# Copy application files
COPY chatterbox_wrapper.py /app/
COPY chatterbox_handler.py /app/
COPY entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose Wyoming protocol port
EXPOSE 10200

ENTRYPOINT ["/app/entrypoint.sh"]
