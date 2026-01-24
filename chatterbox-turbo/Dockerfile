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

# Upgrade build tools for Python 3.12 compatibility
RUN pip3 install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

# Install numpy first with compatible version for Python 3.12
RUN pip3 install --no-cache-dir "numpy>=1.26.0"

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    scipy \
    soundfile \
    librosa \
    wyoming

# Install Chatterbox TTS dependencies manually to avoid numpy version conflict
RUN pip3 install --no-cache-dir \
    torch \
    torchaudio \
    einops \
    vocos \
    perth \
    packaging

# Install Chatterbox TTS (will use already installed numpy and dependencies)
RUN pip3 install --no-cache-dir --no-deps \
    git+https://github.com/resemble-ai/chatterbox.git

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
