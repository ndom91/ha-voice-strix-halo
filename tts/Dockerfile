FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

# Set working directory
WORKDIR /app

# Set environment variables for ROCm
ENV ROCM_VERSION=7.1.1
ENV HIP_VISIBLE_DEVICES=0
ENV HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.5.1}

# Install build dependencies and system packages
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    wget \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set ROCm paths for ONNX Runtime
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/usr/local/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${LD_LIBRARY_PATH}

# Set GPU architecture (build arg)
ARG AMDGPU_TARGETS=gfx1151

# Install ONNX Runtime with ROCm support
# Using the rocm execution provider
RUN pip install --no-cache-dir onnxruntime-rocm

# Install Wyoming protocol and Piper
RUN pip install --no-cache-dir \
    wyoming \
    "wyoming-piper @ git+https://github.com/rhasspy/wyoming-piper.git"

# Create directory for models and voices
RUN mkdir -p /data/models /data/voices

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose Wyoming protocol port
EXPOSE 10200

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
