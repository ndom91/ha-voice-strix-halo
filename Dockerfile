FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

# Set working directory
WORKDIR /app

# Set environment variables for ROCm
ENV ROCM_VERSION=7.1.1
ENV HIP_VISIBLE_DEVICES=0
# GPU architecture override - will be set via docker-compose
ENV HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.0.0}

# Install build dependencies and system packages
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    wget \
    portaudio19-dev \
    libsndfile1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel pybind11

# Set ROCm paths
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/usr/local/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${LD_LIBRARY_PATH}
ENV CTRANSLATE2_ROOT=/usr/local

# Set GPU architecture (build arg, will be overridden per GPU)
ARG AMDGPU_TARGETS=gfx1100,gfx1101,gfx1102,gfx1030,gfx1031,gfx1032
ARG CMAKE_HIP_ARCHITECTURES=gfx1100;gfx1101;gfx1102;gfx1030;gfx1031;gfx1032

# Clone and build CTranslate2-rocm (paralin fork)
RUN git clone --recurse-submodules https://github.com/paralin/ctranslate2-rocm.git /tmp/ctranslate2 && \
    cd /tmp/ctranslate2 && \
    git checkout rocm && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DWITH_HIP=ON \
          -DWITH_MKL=OFF \
          -DWITH_OPENBLAS=ON \
          -DOPENMP_RUNTIME=COMP \
          -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}" \
          -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
          -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_CLI=OFF \
          .. && \
    make -j$(nproc) && \
    make install && \
    cd /tmp/ctranslate2/python && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/ctranslate2

# Install faster-whisper
RUN pip install --no-cache-dir faster-whisper

# Install Wyoming protocol and faster-whisper integration
RUN pip install --no-cache-dir \
    wyoming \
    "wyoming-faster-whisper @ git+https://github.com/rhasspy/wyoming-faster-whisper.git"

# Create directory for models
RUN mkdir -p /data/models

# Download Whisper medium model at build time
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', download_root='/data/models')"

# Expose Wyoming protocol port
EXPOSE 10300

# Set entrypoint - note: device="cuda" is correct for ROCm via HIP
ENTRYPOINT ["python3", "-m", "wyoming_faster_whisper", \
    "--model", "medium", \
    "--device", "cuda", \
    "--compute-type", "float16", \
    "--beam-size", "5", \
    "--uri", "tcp://0.0.0.0:10300", \
    "--data-dir", "/data"]
