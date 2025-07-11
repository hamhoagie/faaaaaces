# FAAAAACES Docker Image
# Face Recognition & Mask Detection Platform

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    # Video processing
    ffmpeg \
    # Build tools
    build-essential \
    pkg-config \
    # GPU support (optional)
    # nvidia-ml-py3 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Try to install optional dependencies (continue if they fail)
RUN pip install --no-cache-dir -r requirements-gpu-detection.txt || true && \
    pip install --no-cache-dir -r requirements-mask-reconstruction.txt || true

# Copy application code
COPY app/ ./app/
COPY run_simple.py ./
COPY deploy/faaaaaces ./deploy/faaaaaces

# Create necessary directories
RUN mkdir -p logs uploads faces faces/reconstructed temp

# Download YOLO model
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# Initialize database
RUN python3 -c "from app.models.database import init_db; init_db()" || true

# Create non-root user
RUN useradd -m -u 1000 faaaaaces && \
    chown -R faaaaaces:faaaaaces /app
USER faaaaaces

# Expose port
EXPOSE 5005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5005/status || exit 1

# Run the application
CMD ["python3", "run_simple.py"]