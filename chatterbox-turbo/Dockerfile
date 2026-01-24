FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.11_pytorch_release_2.9.1

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

# Install version-matched PyTorch and torchvision from AMD ROCm 7.1.1 repository
# This fixes the "operator torchvision::nms does not exist" error
RUN wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torch-2.9.1+rocm7.1.1.lw.git351ff442-cp310-cp310-linux_x86_64.whl && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torchvision-0.24.0+rocm7.1.1.gitb919bd0c-cp310-cp310-linux_x86_64.whl && \
    pip3 install --no-cache-dir --force-reinstall \
        torch-2.9.1+rocm7.1.1.lw.git351ff442-cp310-cp310-linux_x86_64.whl \
        torchvision-0.24.0+rocm7.1.1.gitb919bd0c-cp310-cp310-linux_x86_64.whl && \
    rm *.whl

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    wyoming

# Install Chatterbox TTS with all dependencies
RUN pip3 install --no-cache-dir \
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
