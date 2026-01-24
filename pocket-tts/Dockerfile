FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

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

# Install Python dependencies
# Note: Pocket TTS is CPU-only and doesn't benefit from GPU acceleration
# According to the model card, it runs ~6x real-time on 2 CPU cores
RUN pip3 install --no-cache-dir \
    pocket-tts \
    wyoming \
    scipy \
    numpy

# Create app directory
WORKDIR /app

# Copy application files
COPY pocket_wrapper.py /app/
COPY pocket_handler.py /app/
COPY entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose Wyoming protocol port
EXPOSE 10202

ENTRYPOINT ["/app/entrypoint.sh"]
