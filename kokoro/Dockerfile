FROM python:3.12-slim

# Install system dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    wyoming \
    httpx \
    numpy \
    scipy

# Copy application files
COPY kokoro_handler.py .
COPY kokoro_wrapper.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose Wyoming protocol port
EXPOSE 10203

# Run entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
