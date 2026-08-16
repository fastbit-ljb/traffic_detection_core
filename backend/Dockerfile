# Multi-stage Docker build for AI Traffic Management Backend
# Keep the Debian release stable. The unqualified slim tag can move to a new
# Debian version and unexpectedly remove system packages needed by OpenCV.
FROM python:3.11-slim-bookworm AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
# The PyPI PyTorch wheel brings CUDA libraries. The CPU image must use the
# dedicated CPU wheel or a small server will download several unnecessary GB.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1+cpu torchvision==0.20.1+cpu \
    && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim-bookworm AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN groupadd -r traffic && useradd -r -g traffic traffic

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/models /app/uploads /app/output_images /app/output_videos /app/logs /app/static /app/storage \
    && chown -R traffic:traffic /app

# Copy application code
COPY --chown=traffic:traffic ./app ./app

# Keep official weights outside the runtime volume. The deployment script copies
# them into /app/models, including when an existing empty volume is reused.
RUN mkdir -p /opt/bootstrap-models
RUN python - <<'PY'
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

bootstrap_dir = Path("/opt/bootstrap-models")
download_dir = Path("/tmp/yolo-downloads")
download_dir.mkdir(parents=True, exist_ok=True)
for model_name in ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt"):
    os.chdir(download_dir)
    YOLO(model_name)
    downloaded_model = download_dir / model_name
    if not downloaded_model.is_file() or downloaded_model.stat().st_size == 0:
        raise RuntimeError(f"Failed to download {model_name}")
    shutil.copy2(downloaded_model, bootstrap_dir / model_name)
PY

# One compact self-trained traffic weight is kept for the project demonstration.
COPY --chown=traffic:traffic ./models/trained_28068449.pt /opt/bootstrap-models/trained_28068449.pt

# Switch to non-root user
USER traffic

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health | grep -q '"status":"healthy"' || exit 1

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
